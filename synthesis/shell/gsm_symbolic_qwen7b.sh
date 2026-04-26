#!/bin/bash
# Make CSD for GSM-Symbolic (math reasoning) using Qwen 7B.
# Usage: bash synthesis/shell/gsm_symbolic_qwen7b.sh [extra generate_csd args]
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
export CSD_HELPER_REFERENCE_MODE="${CSD_HELPER_REFERENCE_MODE:-curated}"
mkdir -p "$HF_DATASETS_CACHE" "$MPLCONFIGDIR"

echo "Making GSM-Symbolic CSD (Qwen 7B) from the preset wrapper..."
echo "The preset allows a final raw arithmetic expression inside << >>; evaluation computes its numeric value."
echo "Offline defaults: HF_HUB_OFFLINE=$HF_HUB_OFFLINE, TRANSFORMERS_OFFLINE=$TRANSFORMERS_OFFLINE, HF_DATASETS_OFFLINE=$HF_DATASETS_OFFLINE."
echo "Helper reference mode: $CSD_HELPER_REFERENCE_MODE."
echo "Tip: set DEVICE=cuda:3 or CUDA_VISIBLE_DEVICES=3 to pin a GPU."
echo ""

python -m synthesis.cli.generate_csd gsm_symbolic \
  --model-preset qwen7b \
  --max-iterations "$MAX_ITERATIONS" \
  --temperature "$TEMPERATURE" \
  --device "$DEVICE" \
  "$@"

echo "GSM-Symbolic CSD (Qwen 7B) done. Run dir: outputs/ (see latest_run.txt)"
