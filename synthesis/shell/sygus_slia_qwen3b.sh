#!/bin/bash
# Make CSD for SyGuS SLIA using Qwen 3B.
# Usage: bash synthesis/shell/sygus_slia_qwen3b.sh [extra generate_csd args]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

DEVICE="${DEVICE:-auto}"
MAX_ITERATIONS="${MAX_ITERATIONS:-10}"
TEMPERATURE="${TEMPERATURE:-0.7}"

echo "Making SyGuS SLIA CSD (Qwen 3B) from the preset wrapper..."
echo "Tip: set DEVICE=cuda:3 or CUDA_VISIBLE_DEVICES=3 to pin a GPU."
echo ""

python -m synthesis.cli.generate_csd sygus_slia \
  --model-preset qwen3b \
  --max-iterations "$MAX_ITERATIONS" \
  --temperature "$TEMPERATURE" \
  --device "$DEVICE" \
  "$@"

echo "SyGuS SLIA CSD (Qwen 3B) done. Run dir: outputs/ (see latest_run.txt)"
