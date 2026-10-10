#!/usr/bin/env bash
# PURE RE-EVAL of an already-found strategy on a held-out split. NOT synthesis, NOT a warm start:
#   --max-iterations 1, bars 0, --initial-strategy-file <given>  => loads the strategy, evals ONCE,
#   no author call (so $0, no Bedrock billing). This is the ONLY legitimate use of --initial-strategy-file
#   per the warm-start ban. Use it to get the comparable held-out number for a banked best-so-far.
# REQUIRES the csd conda env (transformers 5.5.4 + vllm 0.19.1) for Qwen3.5.
set -uo pipefail

DATASET="${DATASET:?set DATASET=gsm_symbolic|spider|smiles}"
EVAL_MODEL="${EVAL_MODEL:?set EVAL_MODEL=Qwen/Qwen3.5-2B|4B|9B}"
GPU="${GPU:?set GPU=index}"
INITIAL_STRATEGY="${INITIAL_STRATEGY:?set INITIAL_STRATEGY=path to .dfy strategy body}"
SAMPLE_SIZE="${SAMPLE_SIZE:-49}"
EVAL_MAX_STEPS="${EVAL_MAX_STEPS:-900}"
EVAL_MAX_SECONDS="${EVAL_MAX_SECONDS:-90}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.40}"
OUTPUT_NAME="${OUTPUT_NAME:?set OUTPUT_NAME}"
SMILES_CLASS="${SMILES_CLASS:-acrylates}"
SPLIT_NAME="${SPLIT_NAME:-eval}"        # held-out split by default
SPLIT_FILE="${SPLIT_FILE:-}"

REPO=/home/aadivyar/csd-generation
PY=/apps/conda/aadivyar/envs/csd/bin/python
export LD_LIBRARY_PATH=/apps/conda/aadivyar/envs/csd/lib:${LD_LIBRARY_PATH:-}
export CUDA_VISIBLE_DEVICES="$GPU"
export HF_HOME=/home/aadivyar/.cache/huggingface
export TRANSFORMERS_CACHE=/home/aadivyar/.cache/huggingface
set -a; source "$REPO/.env"; set +a

cd "$REPO"
OUT="outputs/generated/$OUTPUT_NAME"
mkdir -p "$OUT"

# bars 0 (re-eval records whatever it gets); max-iterations 1; provide the strategy so NO author call fires.
COMMON=(
  --generation-model us.anthropic.claude-sonnet-4-6 --generation-backend bedrock
  --eval-model "$EVAL_MODEL" --eval-backend vllm
  --initial-strategy-file "$INITIAL_STRATEGY"
  --max-iterations 1
  --output-name "$OUTPUT_NAME" --output-dir "$OUT"
  --min-accuracy 0.0 --min-syntax-rate 0.0
  --eval-sample-size "$SAMPLE_SIZE" --eval-max-steps "$EVAL_MAX_STEPS" --eval-step-token-budget 1
  --eval-max-seconds-per-example "$EVAL_MAX_SECONDS" --eval-min-examples-before-threshold-stop "$SAMPLE_SIZE"
  --max-tokens 32768 --restart-after-stuck-iters 0
  --vllm-gpu-memory-utilization "$GPU_MEM_UTIL" --device auto
  --adaptive-helper-mask --helper-selection-policy bandit --refinement-beam-size 2
  --vllm-tensor-parallel-size 1
)

case "$DATASET" in
  gsm_symbolic)
    SPLIT="${SPLIT_FILE:-/home/aadivyar/csd-generation/environment/benchmark_splits/gsm_symbolic_crane_proportional_49x49_seed123.json}"
    TASK='Solve math word problems step by step, wrapping intermediate symbolic expressions and the final answer inside << >> delimiters.'
    EXTRA=(--dataset gsm_symbolic --gsm-split-file "$SPLIT" --gsm-split-name "$SPLIT_NAME")
    ;;
  spider)
    export CSD_API_MAX_RETRIES=10
    export CSD_RECURRENCE_PENALTY=0.3
    SPLIT="${SPLIT_FILE:-/home/aadivyar/csd-generation/environment/benchmark_splits/spider_dev_proportional_300x300_seed334.json}"
    TASK='Generate a single valid SQL query as exactly `SQL: <<YOUR QUERY>>`, using only the provided schema context.'
    EXTRA=(--dataset spider --spider-split-file "$SPLIT" --spider-split-name "$SPLIT_NAME")
    ;;
  smiles)
    TASK="${SMILES_TASK:?set SMILES_TASK for smiles runs}"
    EXTRA=(--dataset smiles --smiles-classes "$SMILES_CLASS" --smiles-samples-per-class "$SAMPLE_SIZE")
    ;;
  *)
    echo "unknown DATASET=$DATASET"; exit 2;;
esac

echo "REEVAL_START dataset=$DATASET model=$EVAL_MODEL gpu=$GPU sample=$SAMPLE_SIZE split=$SPLIT_NAME strat=$INITIAL_STRATEGY out=$OUT $(date)"
$PY -m synthesis.run_synthesis --task "$TASK" "${COMMON[@]}" "${EXTRA[@]}" > "$OUT/run.log" 2>&1
ec=$?
echo "REEVAL_DONE exit=$ec out=$OUT $(date)"
echo "REEVAL_${OUTPUT_NAME}_SENTINEL_DONE exit=$ec"
