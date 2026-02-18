# Metis Post-Training Pipeline

GRPO reinforcement learning on [Mercor ACE](https://github.com/Mercor-Intelligence/apex-evals) tasks using [verl](https://github.com/volcengine/verl) on [Modal](https://modal.com).

## Overview

- **ACE** (AI Consumer Index) — evaluation benchmark for grounded AI responses across Shopping, Food, Gaming, and DIY domains
- **verl** — RL training framework (GRPO algorithm)
- **Modal** — cloud GPU infrastructure (H100s)
- **Reward function** — OpenAI gpt-4o-mini as LLM judge, evaluating responses against ACE criteria

## Setup

### 1. Install dependencies

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create venv and install deps
uv venv
source .venv/bin/activate
uv pip install datasets pyarrow modal openai transformers
```

### 2. Configure Modal secrets

You need two Modal secrets for training:

```bash
# OpenAI API key (used by reward function during training)
modal secret create openai-secret OPENAI_API_KEY=<your-key>

# Weights & Biases (for logging — optional, can switch to console-only)
modal secret create wandb-secret WANDB_API_KEY=<your-key>
```

### 3. Authenticate Modal

```bash
modal setup
```

## Usage

### Step 1: Generate parquet data

Converts ACE dataset (from HuggingFace) into verl-compatible parquet format:

```bash
python ace_parquet.py --from_hf
```

This creates `ace_verl_data/train.parquet` (60 tasks) and `ace_verl_data/test.parquet` (20 tasks) across all 4 domains.

Options:
- `--domains diy food` — only convert specific domains
- `--split_ratio 0.8` — change train/test split (default: 0.75)
- `--ace_dir path/to/csvs` — use local CSVs instead of HuggingFace

### Step 2: Upload data to Modal

```bash
modal run ace_grpo_modal.py::prep_dataset
```

### Step 3: Train

```bash
# Quick test (1 training step)
modal run ace_grpo_modal.py::train -- trainer.total_training_steps=1

# Full training (detached so it runs in background)
modal run --detach ace_grpo_modal.py::train
```

Override any training config from the CLI:
```bash
modal run --detach ace_grpo_modal.py::train -- \
  trainer.total_training_steps=50 \
  trainer.total_epochs=3 \
  actor_rollout_ref.model.path=Qwen/Qwen2-0.5B
```

### Step 4: Serve the trained model

After training, deploy an OpenAI-compatible inference endpoint:

```bash
modal deploy ace_grpo_modal.py
```

Query it:
```bash
curl -X POST <modal-url>/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [{"role": "user", "content": "Recommend me some DIY tools for a beginner"}],
    "temperature": 0.7
  }'
```

## Project Structure

```
ace_parquet.py        # Convert ACE CSVs/HF to verl parquet format
ace_reward.py         # Reward function (OpenAI LLM judge)
ace_grpo_modal.py     # Modal training script (GRPO + verl)
ace_verl_data/        # Generated parquet files (train/test)
apex-evals/           # ACE benchmark source (datasets, harness, autograder)
modal-examples/       # Modal reference examples
```

## Training Config

Key defaults in `ace_grpo_modal.py`:

| Parameter | Value | Notes |
|-----------|-------|-------|
| Model | Qwen/Qwen2-0.5B | Small model for initial testing |
| Batch size | 32 | 60 train samples total |
| Max prompt length | 512 tokens | ACE prompts are ~100-350 tokens |
| Max response length | 2048 tokens | ACE expects detailed responses |
| Rollouts per prompt | 5 | GRPO samples per training example |
| GPUs | H100 x 2 | Tensor parallel across 2 GPUs |
| Reward model | gpt-4o-mini | OpenAI LLM judge, temp=0.0 |

## Reward Function

`ace_reward.py` evaluates model responses against ACE criteria:

1. Parses criteria from `ground_truth` (JSON string)
2. Sends all criteria + response in one batched OpenAI call
3. Scores each criterion as pass/fail
4. Applies hurdle logic: if any "Hurdle" criterion fails, reward = 0.0
5. Otherwise returns `passed_count / total_count`
