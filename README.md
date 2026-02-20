# Metis Post-Training Pipeline

GRPO reinforcement learning on [Mercor ACE](https://github.com/Mercor-Intelligence/apex-evals) tasks using [verl](https://github.com/volcengine/verl) on [Modal](https://modal.com).

## Overview

- **ACE** (AI Consumer Index) — evaluation benchmark for grounded AI responses across Shopping, Food, Gaming, and DIY domains
- **verl** — RL training framework (GRPO algorithm)
- **Modal** — cloud GPU infrastructure (H100s)
- **Scoring** — OpenAI GPT-4o as LLM judge, evaluating responses against ACE criteria with 3-stage grading (response text check, grounding verification, link verification)

## Setup

### 1. Install dependencies

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create venv and install deps
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e .
```

### 2. API keys and authentication

You need the following API keys / auth tokens:

| Service | What it's for | How to get it |
|---------|--------------|---------------|
| **OpenAI** | LLM judge for scoring (GPT-4o) | [platform.openai.com/api-keys](https://platform.openai.com/api-keys) |
| **SearchAPI.io** | Web search tool (Google search) | [searchapi.io](https://www.searchapi.io/) — sign up, get API key from dashboard |
| **Jina AI** | Page scraping for source enrichment & browsing (paid tier recommended for higher rate limits) | [jina.ai](https://jina.ai/) — sign up, get API key from dashboard |
| **HuggingFace** | Download models + ACE dataset | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) — create a token with `read` access |
| **Weights & Biases** | Training logs (optional) | [wandb.ai/authorize](https://wandb.ai/authorize) |

Set them as environment variables for local use:

```bash
export OPENAI_API_KEY="sk-..."
export SEARCHAPI_IO_KEY="..."
export JINA_API_KEY="jina_..."   # optional but recommended — free tier has low rate limits
export HF_TOKEN="hf_..."        # or run: huggingface-cli login
export WANDB_API_KEY="..."       # optional
```

### 3. Set up Modal

```bash
# Authenticate
modal setup

# Create Modal secrets (used by GPU containers)
modal secret create openai-secret OPENAI_API_KEY=$OPENAI_API_KEY
modal secret create searchapi-secret SEARCHAPI_IO_KEY=$SEARCHAPI_IO_KEY
modal secret create jina-secret JINA_API_KEY=$JINA_API_KEY        # optional but recommended
modal secret create huggingface-secret HF_TOKEN=$HF_TOKEN
modal secret create wandb-secret WANDB_API_KEY=$WANDB_API_KEY  # optional, for training
```

### 4. Generate parquet data

Converts ACE dataset from HuggingFace into verl-compatible parquet format:

```bash
python ace_parquet.py --from_hf
```

Creates `ace_verl_data/train.parquet` (~60 tasks) and `ace_verl_data/test.parquet` (~20 tasks) across all 4 domains.

Options:
- `--domains diy food` — only specific domains
- `--split_ratio 0.8` — change train/test split (default: 0.75)
- `--ace_dir path/to/csvs` — use local CSVs instead of HuggingFace

## Usage

### Run baseline eval (recommended: Modal)

Runs models on ACE tasks with web search + page browsing tools, scores with the autograder, and saves results to a Modal volume.

```bash
# Quick: single model, one task (for testing)
modal run ace_eval_modal.py --model Qwen/Qwen3-8B --task-ids 147 --workers 1

# Single model, all test tasks
modal run ace_eval_modal.py --model Qwen/Qwen3-8B

# Multiple models in parallel (each gets its own GPU container)
modal run ace_eval_modal.py --models "Qwen/Qwen2.5-7B-Instruct,meta-llama/Llama-3.1-8B-Instruct"

# Specific domains only
modal run ace_eval_modal.py --model Qwen/Qwen3-8B --domains "shopping,food"

# Use a different judge model
modal run ace_eval_modal.py --model Qwen/Qwen3-8B --judge-model gpt-4o-mini

# Run each task 3 times for variance estimation
modal run ace_eval_modal.py --model Qwen/Qwen3-8B --runs 3

# Use 4 GPU containers for 4x throughput (data parallelism)
modal run ace_eval_modal.py --model Qwen/Qwen3-8B --shards 4

# Download results from Modal volume
modal volume get ace-eval-results / ./eval_results/
```

Or use the convenience script:

```bash
# Runs parquet gen + Modal eval + download
./run_baselines.sh

# Custom models
./run_baselines.sh "Qwen/Qwen3-8B,meta-llama/Llama-3.1-8B-Instruct" 4
```

### Run baseline eval (local)

For use with OpenAI models or a local vLLM server:

```bash
# OpenAI models (uses OPENAI_API_KEY)
python ace_eval.py --model gpt-4o-mini --workers 4

# Local vLLM server
python ace_eval.py --model Qwen/Qwen3-8B --api_base http://localhost:8000/v1

# Filter by domain or task
python ace_eval.py --model gpt-4o-mini --domains shopping food
python ace_eval.py --model gpt-4o-mini --task_ids 147 814
```

### Train with GRPO

```bash
# Upload data to Modal volume
modal run ace_grpo_modal.py::prep_dataset

# Quick test (1 training step)
modal run ace_grpo_modal.py::train -- trainer.total_training_steps=1

# Full training (detached, runs in background)
modal run --detach ace_grpo_modal.py::train

# Override config
modal run --detach ace_grpo_modal.py::train -- \
  trainer.total_training_steps=50 \
  trainer.total_epochs=3 \
  actor_rollout_ref.model.path=Qwen/Qwen2-0.5B
```

### Serve a trained model

```bash
modal deploy ace_grpo_modal.py
```

## Eval output structure

Each evaluated task produces three files:

```
eval_results/{model_name}/{domain}/task_{task_id}/
├── trace.json                 # Full agent trace: prompt, response, tool calls, scoring breakdown
├── 2_scraped_sources.json     # Sources used for grounding (autograder input format)
└── 3_autograder_results.json  # Per-criterion scores with detailed reasoning
```

With `--runs N` (N > 1), results are stored per-run:

```
eval_results/{model_name}/{domain}/task_{task_id}/
├── run_1/{trace.json, 2_scraped_sources.json, 3_autograder_results.json}
├── run_2/{...}
└── run_3/{...}
```

A per-model `summary.json` aggregates results across all tasks with domain-level breakdowns. With multiple runs, it includes per-task mean/std scores in `task_aggregation`.

## Project structure

### Core pipeline

| File | Description |
|------|-------------|
| `ace_eval_modal.py` | Modal orchestrator for batch eval. Spins up vLLM on H100 GPUs, dispatches models in parallel, collects results. |
| `ace_eval.py` | Core eval logic. Runs agent loop (prompt → LLM → tool calls → score). Works with any OpenAI-compatible API. |
| `ace_reward.py` | Thin verl wrapper: `compute_reward() -> float` for GRPO training. Calls ace_scoring internally. |
| `ace_grpo_modal.py` | Modal script for GRPO training with verl. |
| `ace_parquet.py` | Converts ACE dataset (HuggingFace/CSV) to verl parquet format. |
| `run_baselines.sh` | Convenience script: parquet gen → Modal eval → download results. |

### Scoring package (`ace_scoring/`)

Single source of truth for all ACE grading logic. Used by both eval and RL training.

| File | Description |
|------|-------------|
| `scorer.py` | Core engine: `grade_task()` → `TaskResult`, `grade_criterion()` → `CriterionResult`. Implements 3-route scoring. |
| `types.py` | Dataclasses: `TaskResult`, `CriterionResult`, `Stage1Result`, `Stage2Result`. |
| `prompts.py` | All LLM judge prompt templates (Stage 1, Stage 2, product extraction, link verification). |
| `product.py` | Product extraction from response text + product-source mapping via LLM. |
| `sources.py` | Source preparation from tool history, response URLs, and Jina Reader enrichment. |
| `llm.py` | OpenAI client wrapper with selective retry. `call_judge()`, `parse_json()`. |
| `jina.py` | Jina Reader utilities: `fetch_page()`, `extract_title()`. Centralized for all page scraping. |

### Tools (for verl multi-turn RL training)

| File | Description |
|------|-------------|
| `ace_search_tool.py` | verl `BaseTool` subclasses: `AceSearchTool` (SearchAPI.io) + `AceBrowseTool` (Jina Reader). |
| `ace_tool_config.yaml` | Tool schema definitions (web_search, browse_page) for verl. |

### Reference (read-only)

| File | Description |
|------|-------------|
| `ace.pdf` | ACE paper (methodology in Section 4, Figure 4). |
| `apex-evals/` | Official ACE benchmark: autograder, datasets, pipeline. |

## ACE scoring methodology

Three scoring routes per criterion (values: -1, 0, +1):

1. **Link criteria** ("Provides link(s)") — Scrape URLs via Jina Reader, classify with LLM. Valid link = +1, invalid = -1.
2. **Non-grounded criteria** — Stage 1 only (check response text against criterion). Pass = +1, fail = 0.
3. **Grounded criteria** — Stage 1 + Stage 2:
   - Fail Stage 1 → **0**
   - Pass Stage 1, pass Stage 2 (grounding verified in sources) → **+1**
   - Pass Stage 1, fail Stage 2 (hallucination) → **-1**

**Hurdle criteria**: If any hurdle criterion scores ≤ 0, entire task score becomes 0.

**RL reward**: `total_score / num_criteria`, normalized to [-1.0, 1.0].

## Key configuration

| Setting | Default | Env var | Notes |
|---------|---------|---------|-------|
| Judge model | `gpt-4o` | `ACE_JUDGE_MODEL` | LLM for scoring. Official autograder uses Gemini 2.5 Pro. |
| Search API | SearchAPI.io | `SEARCHAPI_IO_KEY` | Google search during eval |
| Page scraping | Jina Reader | `JINA_API_KEY` | `r.jina.ai/{url}`. Free tier works but rate-limited (2 concurrent). Paid tier recommended for parallel eval. |
| GPU sizing | auto | — | ≤16B params → 1xH100, >16B → 2xH100 |
| Tool parser | auto | — | hermes (Qwen), llama3_json (Llama), mistral (Mistral) |

## ACE domains

| Domain | Eval style | Notes |
|--------|-----------|-------|
| Shopping | Per-product, grounding heavy | Has `shop_vs_product` field ("Product"/"Shop") |
| Food | Recipe/meal plan recommendations | Less grounding-dependent |
| Gaming | Per-product, grounding for features/prices | |
| DIY | Project guides, holistic evaluation | |

## Training config

Key defaults in `ace_grpo_modal.py`:

| Parameter | Value | Notes |
|-----------|-------|-------|
| Model | Qwen/Qwen2-0.5B | Small model for initial testing |
| Batch size | 32 | ~60 train samples total |
| Max prompt length | 512 tokens | ACE prompts are ~100-350 tokens |
| Max response length | 2048 tokens | ACE expects detailed responses |
| Rollouts per prompt | 5 | GRPO samples per training example |
| GPUs | H100 x 2 | Tensor parallel |
| Judge model | gpt-4o | OpenAI LLM judge, temp=0.0 |
