#!/usr/bin/env bash
# Cycle-1 COLD synthesis — 1.5B, token-0 aligned surface (IterGen-style, NO visible <<>>).
# Spec: spider-win-goal-spec.md. Cycle-0 fork RESOLVED to "drop the <<>> scaffold entirely"
# (token-0 default ON), so:
#   - bare task string (no `<<YOUR QUERY>>` framing).
# COLD: NO --initial-strategy-file (warm starts permanently banned, spec §2.1).
# Author = Sonnet-4.6 via Bedrock ONLY (spec §2.5), thinking enabled / effort high.
# Adaptive helper mask ON (spec §2.2). Synthesis on the TRAIN side of seed334 300x300;
# the win is a LATER held-out re-eval on the TEST side (test-300, official grader).
# SPENDS Bedrock (user approved "both cells" 2026-06-22). GPU 2 (only free card).
set -u
cd ~/csd-generation
# Belt-and-suspenders: run_synthesis load_dotenv reads synthesis/.env (has the token);
# also source root .env for AWS_REGION etc.
set -a; source ~/csd-generation/.env 2>/dev/null; set +a
export SPIDER_DB_DIR=~/csd-generation/synthesis/evaluate/syncode/syncode/utils/sql_spider_eval/databases
export CUDA_VISIBLE_DEVICES=2

SPLIT=environment/benchmark_splits/spider_dev_proportional_300x300_seed334.json
TASK='Generate a single valid SQL query as exactly SQL: YOUR QUERY, using only the provided schema context.'
STAMP=$(date +%Y%m%d_%H%M%S)
NAME=spider1p5b_cycle1_token0_cold_${STAMP}
LOG=logs/${NAME}.log

echo "[launch] $NAME -> $LOG  (author=bedrock sonnet-4.6, COLD, token-0, train split)" | tee -a logs/cycle1_cold_driver_${STAMP}.log
python -m synthesis.run_synthesis \
  --task "$TASK" --dataset spider \
  --generation-model us.anthropic.claude-sonnet-4-6 --generation-backend bedrock \
  --anthropic-thinking enabled --anthropic-effort high \
  --eval-model Qwen/Qwen2.5-1.5B-Instruct --eval-backend vllm \
  --max-iterations 20 \
  --min-accuracy 0.55 --min-syntax-rate 0.85 --no-require-delimiters \
  --eval-sample-size 300 --eval-min-examples-before-threshold-stop 300 \
  --eval-max-steps 1200 --eval-step-token-budget 1 --eval-max-seconds-per-example 300 \
  --restart-after-stuck-iters 0 \
  --vllm-max-model-len 16384 \
  --vllm-gpu-memory-utilization 0.30 --device auto --vllm-tensor-parallel-size 1 \
  --adaptive-helper-mask --helper-selection-policy bandit \
  --spider-split-file "$SPLIT" --spider-split-name train \
  --output-name "$NAME" \
  > "$LOG" 2>&1
echo "[done] $NAME exit=$? at $(date -u)" | tee -a logs/cycle1_cold_driver_${STAMP}.log
echo "[ALL DONE] 1.5B cycle-1 cold complete at $(date -u)" | tee -a logs/cycle1_cold_driver_${STAMP}.log
