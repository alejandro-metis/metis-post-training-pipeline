#!/usr/bin/env bash
set -euo pipefail

# Download results from Modal volume and reorganize into baselines/ structure.
#
# Usage:
#   ./run_baselines.sh                  # download + reorganize
#   ./run_baselines.sh --run MODEL ...  # run eval then download
#
# Examples:
#   # Just download and reorganize
#   ./run_baselines.sh
#
#   # Run eval (foreground, blocks until done) then download
#   ./run_baselines.sh --run --model Qwen/Qwen3-8B --prompt fewshot --shards 4 --runs 3 --workers 3
#
#   # Run eval detached (returns immediately, download later)
#   ./run_baselines.sh --run --detach --model Qwen/Qwen3-8B --prompt fewshot --shards 4 --runs 3

BASELINES_DIR="./baselines"
VOLUME="ace-eval-results"

# --- Optional: run eval first ---
if [[ "${1:-}" == "--run" ]]; then
    shift

    # Generate parquet if missing
    if [ ! -f ace_verl_data/all.parquet ]; then
        echo "Generating parquet from HuggingFace..."
        python ace_parquet.py --from_hf
    fi

    echo "=== Running eval ==="
    # Check for --detach flag
    DETACH=""
    ARGS=()
    for arg in "$@"; do
        if [[ "$arg" == "--detach" ]]; then
            DETACH="--detach"
        else
            ARGS+=("$arg")
        fi
    done

    modal run $DETACH ace_eval_modal.py --parquet ace_verl_data/all.parquet "${ARGS[@]}"

    if [[ -n "$DETACH" ]]; then
        echo ""
        echo "Eval running in detached mode. Run this script again (without --run) to download results later."
        exit 0
    fi
fi

# --- Download from volume ---
echo "=== Downloading from Modal volume ==="
TMP_DIR=$(mktemp -d)
modal volume get "$VOLUME" / "$TMP_DIR"

# --- Reorganize into baselines/{model}/{prompt}/ ---
echo "=== Reorganizing into $BASELINES_DIR ==="
mkdir -p "$BASELINES_DIR"

for model_prompt_dir in "$TMP_DIR"/*/; do
    dirname=$(basename "$model_prompt_dir")

    # Split on last underscore to separate model from prompt preset
    # e.g. "Qwen_Qwen3-8B_fewshot" -> model="Qwen_Qwen3-8B", prompt="fewshot"
    if [[ "$dirname" =~ ^(.+)_(fewshot|zeroshot)$ ]]; then
        model="${BASH_REMATCH[1]}"
        prompt="${BASH_REMATCH[2]}"
    else
        # No known prompt suffix — put everything under the dirname
        model="$dirname"
        prompt=""
    fi

    if [[ -n "$prompt" ]]; then
        dest="$BASELINES_DIR/$model/$prompt"
    else
        dest="$BASELINES_DIR/$model"
    fi

    echo "  $dirname -> $dest"
    mkdir -p "$dest"
    cp -r "$model_prompt_dir"* "$dest/"
done

rm -rf "$TMP_DIR"

echo ""
echo "Done. Results in $BASELINES_DIR/"
ls -la "$BASELINES_DIR"/
