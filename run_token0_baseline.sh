#!/usr/bin/env bash
# Cycle-0 aligned baseline: GCD floor (grammar-constrained from token 0) in the
# token-0 (no-delimiter) DEFAULT surface. NOTE: this body is plain
# ConstrainedGeneration-from-token-0 = GCD / grammar_strict, NOT CRANE (CRANE needs
# CoT + <<>> and is a separate run). Pure re-eval (max-iterations 1, bars 0,
# --initial-strategy-file) on the exact seed334 test-300 split, official grader,
# full N=300, for BOTH non-Coder eval models. Author backend is local vLLM and is
# NEVER called (feedback_loop.py:1647 uses the provided body directly; generator
# loads lazily), so ZERO Bedrock spend. Sequential on the only free card (GPU 2).
set -u
cd ~/csd-generation
export SPIDER_DB_DIR=~/csd-generation/synthesis/evaluate/syncode/syncode/utils/sql_spider_eval/databases
export CUDA_VISIBLE_DEVICES=2

STRAT=outputs/strategies/spider_token0_crane_baseline.dfybody
SPLIT=environment/benchmark_splits/spider_dev_proportional_300x300_seed334.json
TASK='Generate a single valid SQL query as exactly SQL: YOUR QUERY, using only the provided schema context.'
STAMP=$(date +%Y%m%d_%H%M%S)

run_one () {
  local EVAL_MODEL="$1"; local UTIL="$2"; local NAME="$3"
  local LOG=logs/${NAME}_${STAMP}.log
  echo "[launch] $NAME  eval=$EVAL_MODEL util=$UTIL  -> $LOG" | tee -a logs/token0_baseline_driver_${STAMP}.log
  python -m synthesis.run_synthesis \
    --task "$TASK" --dataset spider \
    --generation-model Qwen/Qwen2.5-1.5B-Instruct --generation-backend vllm --allow-small-author-model \
    --eval-model "$EVAL_MODEL" --eval-backend vllm \
    --max-iterations 1 --initial-strategy-file "$STRAT" \
    --min-accuracy 0.0 --min-syntax-rate 0.0 \
    --eval-sample-size 300 --eval-min-examples-before-threshold-stop 300 \
    --eval-max-steps 1200 --eval-step-token-budget 1 --eval-max-seconds-per-example 180 \
    --restart-after-stuck-iters 0 \
    --vllm-gpu-memory-utilization "$UTIL" --device auto --vllm-tensor-parallel-size 1 \
    --adaptive-helper-mask --helper-selection-policy bandit \
    --spider-split-file "$SPLIT" --spider-split-name test \
    --output-name "$NAME" \
    > "$LOG" 2>&1
  echo "[done] $NAME exit=$? at $(date -u)" | tee -a logs/token0_baseline_driver_${STAMP}.log
}

run_one "Qwen/Qwen2.5-1.5B-Instruct" 0.20 "spider1p5b_token0_gcd_floor_n300_seed334"
run_one "Qwen/Qwen2.5-7B-Instruct"   0.50 "spider7b_token0_gcd_floor_n300_seed334"
echo "[ALL DONE] token-0 GCD-floor baselines complete at $(date -u)" | tee -a logs/token0_baseline_driver_${STAMP}.log
