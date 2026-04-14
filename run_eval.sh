#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")" && pwd)
cd "$REPO_ROOT"

RUN_DIR=${RUN_DIR:-outputs/latest}
MODEL=${MODEL:-Qwen/Qwen2.5-Coder-7B-Instruct}
DEVICE=${DEVICE:-cuda}
LIMIT=${LIMIT:-3}
MAX_STEPS=${MAX_STEPS:-1024}
VOCAB_SIZE=${VOCAB_SIZE:-3000}

exec python -m evaluation.gsm_symbolic.cli \
  --run-dir "$RUN_DIR" \
  --model "$MODEL" \
  --device "$DEVICE" \
  --limit "$LIMIT" \
  --max-steps "$MAX_STEPS" \
  --vocab-size "$VOCAB_SIZE" \
  "$@"
