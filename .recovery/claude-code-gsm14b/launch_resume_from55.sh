#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/aadivyar/csd-generation}"
PY="${PY:-/apps/conda/aadivyar/envs/csd/bin/python}"
CLAUDE_EXECUTABLE="${CLAUDE_EXECUTABLE:-/home/aadivyar/.local/bin/claude}"
CLAUDE_CONFIG_DIR="${CLAUDE_CONFIG_DIR:-/home/aadivyar/.claude-csd-synthesis}"
EXPECTED_ACCOUNT="aadivya@fermi.ai"
OUTPUT_NAME="warmfix_gsm14b_0714_r2"
SEED="$REPO/.context/claude_code_resume_0715/gsm14b_attempt55.dfy"
HISTORY="$REPO/.context/claude_code_resume_0715/gsm14b_before55.json"
LOG_FILE="$REPO/logs/paid_synth_warmfix_gsm14b_0714_r2.log"
LOCK_FILE="$REPO/.context/http429_resume_locks/gsm14b.lock"
CLAIM_DIR="$REPO/.context/claude_code_resume_0715/attempt55.claim"
TASK='Solve math word problems step by step, wrapping intermediate symbolic expressions and the final answer inside << >> delimiters.'

log() {
  printf '%s [gsm14b-claude-resume] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

[[ -s "$SEED" ]] || { log "missing_seed path=$SEED" >&2; exit 2; }
[[ -s "$HISTORY" ]] || { log "missing_history path=$HISTORY" >&2; exit 2; }

command=(
  "$PY" -m synthesis.run_synthesis
  --task "$TASK"
  --generation-model claude-opus-5-5 --generation-backend claude
  --claude-executable "$CLAUDE_EXECUTABLE"
  --claude-config-dir "$CLAUDE_CONFIG_DIR"
  --claude-expected-account "$EXPECTED_ACCOUNT"
  --claude-timeout-seconds 900
  --claude-idle-timeout-seconds 900
  --claude-emergency-timeout-seconds 7200
  --claude-max-retries 2
  --claude-retry-delay-seconds 30
  --claude-telemetry-dir "$REPO/.context/claude_stream_telemetry"
  --claude-author-lock-file "$REPO/.context/claude_author.lock"
  --eval-model Qwen/Qwen2.5-14B-Instruct --eval-backend vllm
  --max-iterations 26 --initial-attempt-offset 54
  --initial-strategy-file "$SEED"
  --initial-attempt-history-file "$HISTORY"
  --output-name "$OUTPUT_NAME" --output-dir "outputs/generated/$OUTPUT_NAME"
  --min-accuracy 0.5918 --min-syntax-rate 0.85
  --eval-sample-size 49 --eval-max-steps 900 --eval-step-token-budget 1
  --eval-max-seconds-per-example 600 --eval-min-examples-before-threshold-stop 49
  --max-tokens 32768 --restart-after-stuck-iters 0
  --vllm-gpu-memory-utilization 0.81 --vllm-max-model-len 16384 --device auto
  --adaptive-helper-mask --helper-selection-policy bandit --refinement-beam-size 2
  --vllm-tensor-parallel-size 1
  --dataset gsm_symbolic
  --gsm-split-file "$REPO/environment/benchmark_splits/gsm_symbolic_crane_proportional_49x49_seed123.json"
  --gsm-split-name train
)

if [[ "${DRY_RUN:-0}" == 1 ]]; then
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi

[[ -x "$PY" ]] || { log "python_not_executable path=$PY" >&2; exit 2; }
[[ -x "$CLAUDE_EXECUTABLE" ]] || { log "claude_not_executable path=$CLAUDE_EXECUTABLE" >&2; exit 2; }
[[ -d "$CLAUDE_CONFIG_DIR" ]] || { log "claude_config_missing path=$CLAUDE_CONFIG_DIR" >&2; exit 2; }

mkdir -p "$(dirname "$LOCK_FILE")" "$(dirname "$LOG_FILE")"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  log "duplicate_blocked lock=$LOCK_FILE"
  exit 75
fi
if ! mkdir "$CLAIM_DIR" 2>/dev/null; then
  log "one_time_checkpoint_already_claimed path=$CLAIM_DIR"
  exit 75
fi
if [[ "${CLAIM_ONLY:-0}" == 1 ]]; then
  log "checkpoint_claimed path=$CLAIM_DIR"
  exit 0
fi

export LD_LIBRARY_PATH="/apps/conda/aadivyar/envs/csd/lib:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES=2
export HF_HOME=/home/aadivyar/.cache/huggingface
export TRANSFORMERS_CACHE=/home/aadivyar/.cache/huggingface
cd "$REPO"

nohup "$PY" "$REPO/scripts/runtime/vllm_orphan_reaper.py" 9>&- \
  >>"$REPO/logs/vllm_orphan_reaper.log" 2>&1 </dev/null &

log "START replay_attempt=55 next_new_attempt=56 total_cap=80 provider=claude account=$EXPECTED_ACCOUNT gpu=2" | tee -a "$LOG_FILE"
set +e
"${command[@]}" 2>&1 | tee -a "$LOG_FILE"
status=${PIPESTATUS[0]}
set -e
if [[ "$status" -eq 76 ]]; then
  rmdir "$CLAIM_DIR" 2>/dev/null || true
  log "temporary_provider_exit claim_released=$CLAIM_DIR" | tee -a "$LOG_FILE"
fi
log "DONE status=$status" | tee -a "$LOG_FILE"
exit "$status"
