# Metis Post-Training Pipeline

## Session Maintenance
- Always update CLAUDE.md when files are added, removed, or significantly changed
- Update README.md as necessary when user-facing behavior or CLI flags change
- Create and update markdown files as needed to preserve context for future sessions (e.g. architecture notes, debugging logs, decisions)
- Keep file descriptions accurate — if a refactor changes responsibilities, update descriptions here

## Project Goal
Run SFT and RL (GRPO) on ACE tasks from Mercor to train models that produce grounded, criteria-based consumer recommendations across Shopping, Food, Gaming, and DIY domains.

## Workflow
1. Read relevant code before making changes
2. Plan approach before implementing
3. Make minimal, focused changes
4. Run tests and lint after changes
5. Don't modify unrelated code

Test as you go to ensure high quality, clean code. When in doubt, ask before assuming.

## Coding Rules (prevent recurring bugs)

### Forward all fields through the call chain
When a function wraps another (e.g. `ace_reward.compute_reward` → `grade_task`), it MUST forward every parameter that the inner function accepts. Check the inner function's signature and pass through all relevant fields from `extra_info`, config, etc. Forgetting a field (like `shop_vs_product`) silently produces wrong results.

### Python falsiness: `[]` is falsy, `None` is falsy — distinguish them
Never write `x or default` when `x` could be a legitimate empty list `[]`. Use `x if x is not None else default` instead. `[] or None` evaluates to `None`, which silently changes behavior (e.g. triggers re-extraction).

### Subprocess cleanup: always use try/finally
Any `subprocess.Popen()` call that manages an expensive resource (GPU server, background service) MUST be wrapped in `try/finally` with `proc.terminate(); proc.wait()` in the finally block. An unguarded subprocess leaks resources on any exception between start and cleanup.

### RL reward functions must not crash the training loop
`compute_reward()` and similar functions called per-rollout during training must catch exceptions from external services (LLM judge, web APIs) and return a safe default (e.g. 0.0) with a warning. A single API failure should not crash an entire training step with hundreds of rollouts.

### Review non-trivial changes before committing
After making multi-file changes or touching core logic (scoring, reward, orchestration), spawn an opus subagent to review the full diff. The reviewer should verify: all parameters forwarded through wrapper functions, no Python falsiness bugs (`[]` vs `None`), subprocess/resource cleanup on all exit paths, nothing deleted that has a future use case, and no silent behavior changes from defaults.

### Don't leave truly dead code, but keep intentional future-use functions
If a function is no longer called and has no foreseeable use, delete it. But if it serves a clear future use case (e.g., `sources_from_response_urls` for scoring tool-less models), keep it with a docstring noting why. Grep for callers before deciding.

## Commands
- Lint: `ruff check . --fix`
- Format: `ruff format .`
- Typecheck: `mypy src/ --strict`
- Test single file: `pytest tests/path/to/test.py -v`
- Test all: `pytest tests/ -v`

Always run lint and typecheck before considering a task complete.

## Subagent Configuration
- Always use opus model for all subagents (Task tool calls should use model: "opus")

## Environment
- Package manager: uv (venv at .venv/)
- **Always activate venv before running commands**: `source /Users/ellenma/mercor-rl/metis-post-training-pipeline/.venv/bin/activate`
- Python: 3.11 (via .python-version)
- Key CLI tools in venv: `modal`, `python`, `uv`
- Git: lazygit available

## Project Structure

### How the pieces fit together

```
                  ┌─────────────────────────────────────┐
                  │         ace_eval_modal.py            │
                  │  Modal orchestrator: dispatches      │
                  │  models to GPU containers, collects  │
                  │  results. Calls ace_eval functions.  │
                  └──────────────┬──────────────────────┘
                                 │ imports eval_single_task,
                                 │ build_eval_summary
                                 ▼
                  ┌─────────────────────────────────────┐
                  │           ace_eval.py                │
                  │  Agent loop: sends prompts to LLM,  │
                  │  executes web_search + browse_page   │
                  │  tool calls, then scores via         │
                  │  ace_scoring. Writes trace + results.│
                  └──────────────┬──────────────────────┘
                                 │ imports grade_task,
                                 │ sources_from_tool_history,
                                 │ format_criteria_for_autograder
                                 ▼
┌──────────────────────────────────────────────────────────────┐
│                     ace_scoring/ package                      │
│  Single source of truth for all ACE grading logic.           │
│                                                              │
│  scorer.py ─── grade_task(), grade_criterion()               │
│  types.py ──── TaskResult, CriterionResult, Stage1/2Result   │
│  prompts.py ── All LLM judge prompt templates                │
│  product.py ── Product extraction + product-source mapping   │
│  sources.py ── Source preparation from tool history + Jina   │
│  llm.py ────── OpenAI client wrapper (call_judge, parse_json)│
│  jina.py ───── Jina Reader: fetch_page(), extract_title()    │
└──────────────────────────────────────────────────────────────┘
                                 ▲
                                 │ imports grade_task,
                                 │ sources_from_tool_history
                                 │
                  ┌──────────────┴──────────────────────┐
                  │           ace_reward.py              │
                  │  Thin verl wrapper: compute_reward() │
                  │  → float for GRPO training loop.     │
                  └─────────────────────────────────────┘

                  ┌─────────────────────────────────────┐
                  │        ace_search_tool.py            │
                  │  verl BaseTool subclasses for        │
                  │  multi-turn RL training:             │
                  │  AceSearchTool + AceBrowseTool       │
                  └─────────────────────────────────────┘
```

### Core Pipeline Files

- **`ace_eval_modal.py`** — Modal orchestrator for batch baseline evaluation. Spins up vLLM on H100 GPUs, dispatches models in parallel (each gets its own container), collects results. Uses `eval_single_task()` and `build_eval_summary()` from ace_eval.py. CLI: `modal run ace_eval_modal.py --model Qwen/Qwen3-8B`
- **`ace_eval.py`** — Core eval logic. Runs an agent loop: sends task prompt to LLM via OpenAI-compatible API, executes `web_search` and `browse_page` tool calls (SearchAPI.io + Jina Reader), then scores the final response via `ace_scoring.grade_task()`. Writes trace.json, 2_scraped_sources.json, and 3_autograder_results.json per task. Also provides `build_eval_summary()` for domain-level aggregation. Can run standalone: `python ace_eval.py --model gpt-4o-mini --workers 4`
- **`ace_reward.py`** — Thin verl GRPO wrapper. `compute_reward(data_source, solution_str, ground_truth, extra_info) -> float`. Calls `grade_task()` from ace_scoring and returns normalized score in [-1.0, 1.0]. Used during RL training.
- **`ace_grpo_modal.py`** — Modal script for GRPO training (verl 0.7.0, Qwen2-0.5B base)
- **`ace_parquet.py`** — Convert ACE dataset (HuggingFace or CSV) → verl parquet format with train/test split
- **`ace_eval_prompts.py`** — System prompt presets for the agent (fewshot, zeroshot). Selected via `--prompt` flag. Add new presets here.
- **`run_baselines.sh`** — Shell wrapper for eval: parquet gen → Modal run → volume download

### Scoring Package (`ace_scoring/`)

Single source of truth for all ACE scoring logic. Used by both eval (ace_eval.py) and RL training (ace_reward.py).

- **`ace_scoring/scorer.py`** — Core grading engine. `grade_task()` → `TaskResult`, `grade_criterion()` → `CriterionResult`. Implements the 3-route scoring logic (link verification, non-grounded, grounded). Link verification scrapes URLs via Jina and classifies with LLM (matches official ACE autograder approach).
- **`ace_scoring/types.py`** — Dataclasses: `TaskResult`, `CriterionResult`, `Stage1Result`, `Stage2Result`. Also `format_criteria_for_autograder()` for parquet→autograder field mapping.
- **`ace_scoring/prompts.py`** — All LLM judge prompt templates (Stage 1, Stage 2, product extraction, link verification). Ported from official autograder.
- **`ace_scoring/product.py`** — Product extraction from response text + product-source mapping via LLM.
- **`ace_scoring/sources.py`** — Source preparation: `sources_from_tool_history()` (for RL/eval with tools), `sources_from_response_urls()` (for models without tool use that embed URLs), `enrich_snippet_sources()` (fetches full page content for snippet-only search results via Jina), `build_source_text_for_grounding()` (concatenates sources for Stage 2 prompts).
- **`ace_scoring/llm.py`** — OpenAI client wrapper. `call_judge()` with selective retry (only retries transient errors: rate limit, connection, timeout, server errors). `parse_json()` with markdown fence stripping. Judge model configurable via `ACE_JUDGE_MODEL` env var.
- **`ace_scoring/jina.py`** — Centralized Jina Reader utilities. `fetch_page(url) -> (markdown, error)` and `extract_title(markdown) -> str`. Used by sources.py, scorer.py, ace_eval.py, and ace_search_tool.py. Single source of truth for `JINA_READER_BASE`.

### Tools (for verl multi-turn RL training)

- **`ace_search_tool.py`** — verl `BaseTool` subclasses: `AceSearchTool` (web search via SearchAPI.io) and `AceBrowseTool` (page browsing via Jina Reader). Manages per-trajectory state (search/browse counts, history). Used during GRPO training for multi-turn rollouts.
- **`ace_tool_config.yaml`** — Tool schema definitions for verl.

### Reference (read-only)
- `ace.pdf` — ACE paper (methodology in Section 4, Figure 4)
- `apex-evals/ace/harness/autograder.py` — Official ACE autograder (Gemini judge, ~1167 lines). Uses Firecrawl + Gemini 2.5 Pro for link verification.
- `apex-evals/ace/pipeline/runner.py` — Official pipeline orchestrator
- `apex-evals/ace/pipeline/grounding-pipeline.py` — Official source scraping (Firecrawl)

## Running Baselines

### Prerequisites
- Parquet data already exists: `ace_verl_data/test.parquet` (20 tasks), `ace_verl_data/train.parquet` (60 tasks)
- To regenerate: `python ace_parquet.py --from_hf`
- Always activate venv first: `source .venv/bin/activate`

### Open-source models (Modal + vLLM on H100s)

Use `ace_eval_modal.py` or `run_baselines.sh`. Each model gets its own GPU container.

```bash
# Quick test: single model, single task
modal run ace_eval_modal.py --model Qwen/Qwen3-8B --task-ids 147 --workers 1

# Full test set (20 tasks)
modal run ace_eval_modal.py --model Qwen/Qwen3-8B

# Multiple models in parallel
modal run ace_eval_modal.py --models "Qwen/Qwen3-8B,Qwen/Qwen2.5-7B-Instruct"

# Or use the convenience script (generates parquet if missing, runs, downloads)
./run_baselines.sh "Qwen/Qwen3-8B,Qwen/Qwen2.5-7B-Instruct"

# Download results after
mkdir -p eval_results && modal volume get ace-eval-results / ./eval_results/
```

CLI flags for `ace_eval_modal.py`:
- `--model` / `--models` — HuggingFace model ID(s), comma-separated for multiple
- `--parquet` — path to parquet file (default: `ace_verl_data/test.parquet`)
- `--domains` — filter domains, comma-separated (e.g. `"shopping,food"`)
- `--task-ids` — filter task IDs, comma-separated
- `--workers` — parallel tasks per container (default: auto — 16 for ≤16B models, 12 for >16B). vLLM batches concurrent requests efficiently.
- `--max-turns` — max agent loop turns (default: 10)
- `--max-searches` / `--max-browses` — per-task limits (default: 5 each)
- `--judge-model` — override judge (default: gpt-4o via `ACE_JUDGE_MODEL` env)
- `--prompt` — system prompt preset: `fewshot` or `zeroshot` (default)
- `--runs` — number of runs per task (default: 1). Use >1 for variance estimation. Results stored in `task_{id}/run_{n}/` subdirectories.
- `--shards` — number of GPU containers per model (default: 1). Each shard loads its own model copy. Only use >1 for large models that saturate a single GPU.

GPU auto-sizing: <=16B params → 1xH100, >16B → 2xH100

**Important: don't shard small models.** An 8B model uses ~16GB of an 80GB H100, leaving plenty of VRAM for KV cache. Use `--shards 1 --workers 16` (1 GPU, 16 parallel tasks) instead of `--shards 4 --workers 4` (4 GPUs, same throughput, 4x cost). Sharding only makes sense for 32B+ models that fill the GPU.

### GPT / API models (local, no GPU needed)

Use `ace_eval.py` directly. Calls the OpenAI API (or any OpenAI-compatible endpoint).

```bash
# GPT-4o on full test set
python ace_eval.py --model gpt-4o --workers 4

# GPT-4o-mini
python ace_eval.py --model gpt-4o-mini --workers 8

# Specific domains
python ace_eval.py --model gpt-4o --domains shopping food --workers 4

# Zero-shot prompt (no example)
python ace_eval.py --model gpt-4o --prompt zeroshot --workers 4

# Custom API endpoint (e.g. local vLLM, together.ai)
python ace_eval.py --model Qwen/Qwen3-8B --api_base http://localhost:8000/v1
```

Requires `OPENAI_API_KEY` and `SEARCHAPI_IO_KEY` env vars set locally.

### Inspecting results

Results are saved per-task, namespaced by model + prompt preset:
```
eval_results/{model_name}_{prompt_preset}/{domain}/task_{task_id}/
├── trace.json                 # prompt, response, tool calls, scoring breakdown
├── 2_scraped_sources.json     # sources for grounding verification
└── 3_autograder_results.json  # per-criterion scores with reasoning
```

Example: `Qwen_Qwen3-8B_fewshot/shopping/task_814/trace.json`

Per-model summary: `eval_results/{model_name}_{prompt_preset}/summary.json`

Modal volume commands:
```bash
modal volume ls ace-eval-results                           # list models
modal volume ls ace-eval-results Qwen_Qwen3-8B_fewshot/    # list domains
modal volume get ace-eval-results / ./eval_results/         # download everything
```

### Current state (as of 2026-02-19 evening)
- **Parquet data**: test.parquet (20 tasks), train.parquet (60 tasks), all.parquet (80 tasks combined)
- **Default prompt**: `zeroshot` (changed from fewshot). Prompt presets in `ace_eval_prompts.py`.
- **GPT-5.2 baselines DONE** (local, `eval_results/`):
  - `gpt-5.2_fewshot/`: 240/240 complete (80 tasks × 3 runs)
  - `gpt-5.2_zeroshot/`: 238/240 complete (2 failed)
  - These need rescoring with `rescore.py` — enrichment had Jina 429s (runs started before JINA_API_KEY was set)
- **Qwen baselines IN PROGRESS** (Modal, `bananas` workspace, sharded):
  - Qwen3-8B fewshot: 4 shards × 1xH100, `--runs 3 --workers 2`
  - Qwen3-8B zeroshot: 4 shards × 1xH100, `--runs 3 --workers 2`
  - Qwen3-32B fewshot: 2 shards × 2xH100, `--runs 3 --workers 2`
  - Qwen3-32B zeroshot: 2 shards × 2xH100, `--runs 3 --workers 2`
  - These have JINA_API_KEY and bug fixes — should NOT need rescoring
- **Modal workspace**: switched from `ellen-64441` to `bananas` (colleague's team plan). Secrets recreated there.
- **Old results on ellen-64441 volume**: stale, can ignore. Old local dirs `eval_results_fewshot/` and `eval_results_verify/` should be archived.
- **TODO next session**:
  1. Check if Qwen Modal runs finished (`modal app list` on bananas workspace)
  2. Download Modal results: `modal volume get ace-eval-results / ./eval_results/`
  3. Rescore GPT-5.2 results: `python rescore.py --results-dir eval_results/gpt-5.2_fewshot --workers 4` (and zeroshot)
  4. Reorganize eval_results into local `baselines/` structure (see below)
  5. Uncommitted changes need committing (shards feature, coordinator fix, bug fixes, etc.)

### Local baselines directory structure

After downloading from Modal volume, reorganize into:
```
baselines/
  Qwen_Qwen3-8B/
    fewshot/
      {domain}/task_{id}/run_{n}/
        trace.json
        2_scraped_sources.json
        3_autograder_results.json
      summary.json
    zeroshot/
      ...
  Qwen_Qwen3-32B/
    fewshot/...
    zeroshot/...
  gpt-5.2/
    fewshot/...
    zeroshot/...
```

Volume format is flat: `{model}_{prompt}/{domain}/task_{id}/...`
Local reorganization splits model and prompt into nested dirs.

### Planned baseline runs (all 80 tasks)

Use `ace_verl_data/all.parquet` for all 80 tasks.

**Open-source models (Modal, --detach for background):**
```bash
source .venv/bin/activate

# Qwen3-8B (1xH100) — fewshot + zeroshot
modal run --detach ace_eval_modal.py --model Qwen/Qwen3-8B --parquet ace_verl_data/all.parquet --prompt fewshot
modal run --detach ace_eval_modal.py --model Qwen/Qwen3-8B --parquet ace_verl_data/all.parquet --prompt zeroshot

# Qwen3-32B (2xH100) — fewshot + zeroshot
modal run --detach ace_eval_modal.py --model Qwen/Qwen3-32B --parquet ace_verl_data/all.parquet --prompt fewshot
modal run --detach ace_eval_modal.py --model Qwen/Qwen3-32B --parquet ace_verl_data/all.parquet --prompt zeroshot
```

**GPT-4o (local, OpenAI API):**
```bash
# Run in background with nohup or similar
python ace_eval.py --model gpt-4o --parquet ace_verl_data/all.parquet --prompt fewshot --workers 4
python ace_eval.py --model gpt-4o --parquet ace_verl_data/all.parquet --prompt zeroshot --workers 4
```

**Results will be namespaced:**
- `Qwen_Qwen3-8B_fewshot/`, `Qwen_Qwen3-8B_zeroshot/`
- `Qwen_Qwen3-32B_fewshot/`, `Qwen_Qwen3-32B_zeroshot/`
- `gpt-4o_fewshot/`, `gpt-4o_zeroshot/`

### Previously planned (lower priority)
- `Qwen/Qwen2.5-7B-Instruct` — standard Qwen baseline
- `meta-llama/Llama-3.1-8B-Instruct` — Llama baseline
- `gpt-4o-mini` — API model baseline (via ace_eval.py, not Modal)
- `gpt-4o` — API model baseline (via ace_eval.py)

## ACE Scoring Methodology (Paper Figure 4)

### Per-Criterion Scoring: Three values {-1, 0, +1}

Three routing paths (matching `autograder.grade_criterion`):

1. **"Provides link(s)" type** → Scrape URLs via Jina, classify with LLM. Score -1 (invalid) or +1 (valid).
2. **Non-grounded** (`grounding_check ≠ "Grounded"`) → Stage 1 only (response text check). Score 0 or +1.
3. **Grounded** (`grounding_check == "Grounded"`) → Stage 1 + Stage 2:
   - Fail Stage 1 (response text) → score **0**
   - Pass Stage 1, pass Stage 2 (grounding verified) → score **+1**
   - Pass Stage 1, fail Stage 2 (hallucination) → score **-1**

### Stage 1: Response Text Check
- Uses canonical product list (extracted by LLM)
- `evaluation_type`: holistic | per_product_all | per_product_any
- `required_pass_count`: -1 means ALL must pass
- Zero background knowledge rule — only explicit statements count

### Stage 2: Grounding Verification
- Checks Stage 1 claims against source material (browsed pages, search results)
- Per-product: verifies each product's claims in mapped sources
- Holistic: verifies overall criterion against all sources
- Catches hallucination — model may claim things not in any source

### Task-Level Aggregation
- `total_score = sum(criterion_scores)` — raw sum (used by leaderboard)
- `total_hurdle_score`: if ANY hurdle criterion ≤ 0 → 0, else → total_score
- RL reward: `total_score / num_criteria` (normalized to [-1.0, 1.0])

### Hurdle Criteria
- Tagged `hurdle_tag == "Hurdle"` in the dataset
- Gate mechanism: if any hurdle fails (score ≤ 0), entire task scores 0
- Critical for Shopping/Gaming domains

## Criterion Field Mapping (parquet <> autograder)

| Parquet field | Autograder field | Values |
|---|---|---|
| `criterion_id` | `criterion_id` | string |
| `description` | `description` | string |
| `criteria_type` | `type` | "standard", "Provides link(s)", etc. |
| `hurdle_tag` | `hurdle_tag` | "Hurdle" or "Not" |
| `grounding_check` | `grounded_status` | "Grounded" or "Not Grounded" |

## Eval Output Structure

```
eval_results/{model_name}/{domain}/task_{task_id}/
├── 2_scraped_sources.json    (autograder input format)
├── 3_autograder_results.json (scored results with per-criterion detail)
└── trace.json                (raw response + tool history)
```

## Key Decisions
- **Judge model**: OpenAI gpt-4o (env: `ACE_JUDGE_MODEL`). Official autograder uses Gemini 2.5 Pro.
- **Link verification**: Jina Reader scrape + GPT classification (replaces Firecrawl + Gemini in official autograder)
- **Source scraping**: Jina Reader (`r.jina.ai`) replaces Firecrawl. Uses `JINA_API_KEY` env var for paid tier (higher rate limits). Free tier gets 429s under parallel load.
- **Web search**: SearchAPI.io (env: `SEARCHAPI_IO_KEY`)
- **GPU sizing**: 16B param threshold for 1xH100 vs 2xH100
- **Tool parsers**: hermes (Qwen), llama3_json (Llama), mistral (Mistral)

## ACE Domains
- **Shopping**: Per-product eval, grounding heavy, has `shop_vs_product` field ("Product"/"Shop")
- **Food**: Recipe/meal plan recommendations, less grounding
- **Gaming**: Per-product eval, grounding for features/prices
- **DIY**: Project guides, holistic evaluation style

## Data Format
- Dataset: `mercor/ACE` on HuggingFace (4 splits: DIY, Food, Gaming, Shopping)
- Parquet: verl format with `prompt`, `reward_model.ground_truth` (JSON criteria), `extra_info`
- Train/test split: 75/25 per domain (ace_parquet.py)
