#!/usr/bin/env bash
# GSM held-out (test-49) baselines: 3 models x {unconstrained,gcd,crane,itergen}.
# Same runner/args as the 2026-08-03 train collection (run_focal_collection_pool.py
# fixed_strategy_args), only --gsm-split-name test. Sequential on one GPU.
set -uo pipefail

REPO=/home/aadivyar/csd-generation-worktrees/full-baseline-campaign-20260803
PY=/apps/conda/aadivyar/envs/csd/bin/python
GPU="${BASELINE_GPU:-2}"
SPLIT=environment/benchmark_splits/gsm_symbolic_crane_proportional_49x49_seed123.json
OUTROOT="$REPO/outputs/baselines/gsm_heldout_test49_20260812"

cd "$REPO"
export CUDA_VISIBLE_DEVICES="$GPU" CSD_EVAL_GPU_SLOTS="$GPU"
export CONDA_PREFIX=/apps/conda/aadivyar/envs/csd
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH="$REPO"

# Wait for the gsm heldout rerun driver to release GPU 2.
while pgrep -f run_gsm_heldout_reruns.py >/dev/null; do
  echo "[gsm-test-baselines] waiting for heldout driver to finish..."
  sleep 120
done

run_one() {
  local model="$1" mdir="$2" util="$3" strat="$4"
  local out="$OUTROOT/$mdir/$strat.json"
  if [[ -s "$out" ]]; then
    echo "[gsm-test-baselines] SKIP $mdir/$strat (exists)"
    return 0
  fi
  mkdir -p "$OUTROOT/$mdir"
  echo "[gsm-test-baselines] START $mdir/$strat gpu=$GPU util=$util $(date -u +%H:%M:%S)"
  "$PY" -m synthesis.evaluate.run_legacy_fixed_strategy \
    --strategy "$strat" --dataset gsm_symbolic \
    --eval-model "$model" --eval-backend vllm --device cuda \
    --eval-sample-size 49 --eval-max-steps 900 --eval-step-token-budget 1 \
    --gsm-split-file "$SPLIT" --gsm-split-name test \
    --vllm-gpu-memory-utilization "$util" --vllm-tensor-parallel-size 1 \
    --output-json "$out"
  echo "[gsm-test-baselines] FINISH $mdir/$strat exit=$? $(date -u +%H:%M:%S)"
}

for strat in unconstrained gcd crane itergen; do
  run_one "Qwen/Qwen2.5-1.5B-Instruct" qwen25-1p5b 0.35 "$strat"
done
for strat in unconstrained gcd crane itergen; do
  run_one "Qwen/Qwen2.5-7B-Instruct" qwen25-7b 0.55 "$strat"
done
for strat in unconstrained gcd crane itergen; do
  run_one "Qwen/Qwen3.5-4B" qwen35-4b 0.45 "$strat"
done
for strat in unconstrained gcd crane itergen; do
  run_one "Qwen/Qwen3.5-2B" qwen35-2b 0.40 "$strat"
done
echo "[gsm-test-baselines] ALL DONE $(date -u +%H:%M:%S)"
