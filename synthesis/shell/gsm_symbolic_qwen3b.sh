#!/bin/bash
# Make CSD for GSM-Symbolic using Qwen 3B (lighter, faster).
# Usage: bash synthesis/shell/gsm_symbolic_qwen3b.sh [extra generate_csd args]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

DEVICE="${DEVICE:-auto}"
MAX_ITERATIONS="${MAX_ITERATIONS:-10}"
TEMPERATURE="${TEMPERATURE:-0.7}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/tmp/hf-datasets-local}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
mkdir -p "$HF_DATASETS_CACHE" "$MPLCONFIGDIR"

echo "Making GSM-Symbolic CSD (Qwen 3B) from the preset wrapper..."
echo "The preset allows a final raw arithmetic expression inside << >>; evaluation computes its numeric value."
echo "Offline defaults: HF_HUB_OFFLINE=$HF_HUB_OFFLINE, TRANSFORMERS_OFFLINE=$TRANSFORMERS_OFFLINE, HF_DATASETS_OFFLINE=$HF_DATASETS_OFFLINE."
echo "Tip: set DEVICE=cuda:3 or CUDA_VISIBLE_DEVICES=3 to pin a GPU."
echo ""

python -m synthesis.cli.generate_csd gsm_symbolic \
  --model-preset qwen3b \
  --max-iterations "$MAX_ITERATIONS" \
  --temperature "$TEMPERATURE" \
  --device "$DEVICE" \
  "$@"

echo "GSM-Symbolic CSD (Qwen 3B) done. Run dir: outputs/ (see latest_run.txt)"
