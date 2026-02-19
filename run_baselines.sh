#!/usr/bin/env bash
set -euo pipefail

# --- Config ---
MODELS="${1:-Qwen/Qwen2.5-7B-Instruct,meta-llama/Llama-3.1-8B-Instruct}"
WORKERS="${2:-4}"
OUTPUT_DIR="./eval_results"

echo "=== ACE Baseline Eval ==="
echo "Models: $MODELS"
echo "Workers: $WORKERS"

# 1. Generate parquet if missing
if [ ! -f ace_verl_data/test.parquet ]; then
    echo "Generating parquet from HuggingFace..."
    python ace_parquet.py --from_hf
fi

# 2. Run eval on Modal
modal run ace_eval_modal.py \
    --models "$MODELS" \
    --workers "$WORKERS"

# 3. Download results
echo "Downloading results..."
mkdir -p "$OUTPUT_DIR"
modal volume get ace-eval-results / "$OUTPUT_DIR"

echo ""
echo "Done. Results in $OUTPUT_DIR/"
ls -la "$OUTPUT_DIR"/
