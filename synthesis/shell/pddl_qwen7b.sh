#!/bin/bash
# Make CSD for PDDL Blocks World using Qwen 7B.
# Usage: bash synthesis/shell/pddl_qwen7b.sh [extra generate_csd args]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

DEVICE="${DEVICE:-auto}"
MAX_ITERATIONS="${MAX_ITERATIONS:-10}"
TEMPERATURE="${TEMPERATURE:-0.7}"

echo "Making PDDL CSD (Qwen 7B) from the preset wrapper..."
echo "Tip: set DEVICE=cuda:3 or CUDA_VISIBLE_DEVICES=3 to pin a GPU."
echo ""

python -m synthesis.cli.generate_csd pddl \
  --model-preset qwen7b \
  --max-iterations "$MAX_ITERATIONS" \
  --temperature "$TEMPERATURE" \
  --device "$DEVICE" \
  "$@"

echo "PDDL CSD (Qwen 7B) done. Run dir: outputs/ (see latest_run.txt)"
