#!/bin/bash
set -euo pipefail

# Evaluate GSM symbolic using a fixed Python-first baseline strategy,
# without invoking the synthesis/refinement loop.

REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
RUNS_DIR="$REPO_ROOT/outputs/generated-csd/runs"
mkdir -p "$RUNS_DIR"
RUN_DIR="$RUNS_DIR/vanilla_baseline_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"

export REPO_ROOT
export RUN_DIR

echo "Building fixed vanilla baseline strategy in $RUN_DIR ..."

python - <<'PY'
import os
from pathlib import Path

from synthesis.compiler import DafnyCompiler
from synthesis.generator import StrategyGenerator
from synthesis.verifier import DafnyVerifier
from transpiler.transpiler import transpile_contract_library

repo_root = Path(os.environ["REPO_ROOT"])
run_dir = Path(os.environ["RUN_DIR"])
dafny_bin = repo_root / "dafny" / "dafny"

strategy_body = """
# CSD_RATIONALE_BEGIN
# Fixed answer-channel baseline: emit a small expressive preamble, then build the
# final grammar-constrained answer segment token by token.
# CSD_RATIONALE_END
phase = 0
preamble_tokens = 0
exploration_budget = 1
answer_tokens = 0
# invariant lm.ValidTokensIdsLogits()
# invariant 0 <= stepsLeft <= maxSteps - 2
# invariant 0 <= preamble_tokens <= 1
# invariant 0 <= exploration_budget <= 1
# invariant 0 <= answer_tokens
# invariant helpers.ConstrainedWindowValid(generated)
# invariant parser.IsValidPrefix(answer)
# invariant |generated| + |answer| + stepsLeft <= maxSteps - 2
# invariant |answer| == 0 ==> exploration_budget < stepsLeft
# decreases stepsLeft
while stepsLeft > 0 and not parser.IsCompletePrefix(answer):
    next_token = eosToken
    new_steps = stepsLeft
    spend_freeform = phase < 2 and exploration_budget > 0 and preamble_tokens < 1 and stepsLeft > 1
    if spend_freeform:
        next_token, new_steps = helpers.ExpressiveStep(prompt, generated, stepsLeft)
        generated = generated + [next_token]
        stepsLeft = new_steps
        preamble_tokens = preamble_tokens + 1
        exploration_budget = exploration_budget - 1
        if preamble_tokens >= 1:
            phase = 1
        if preamble_tokens >= 1 or stepsLeft <= 1:
            phase = 2
    else:
        next_token, new_steps = helpers.ConstrainedAnswerStep(prompt, generated, answer, stepsLeft)
        answer = answer + [next_token]
        stepsLeft = new_steps
        answer_tokens = answer_tokens + 1
        if answer_tokens >= 1:
            phase = 3
"""

full_code = StrategyGenerator().inject_strategy(strategy_body)
verifier = DafnyVerifier(dafny_path=str(dafny_bin))
verify_result = verifier.verify(full_code)
if not verify_result.success:
    raise SystemExit("Verification failed:\n" + verify_result.get_error_summary())

compiler = DafnyCompiler(dafny_path=str(dafny_bin), output_dir=run_dir)
compile_result = compiler.compile(full_code, output_name="generated_csd")
if not compile_result.success:
    raise SystemExit("Compilation failed:\n" + compile_result.get_error_summary())

python_path = run_dir / "generated_csd.py"
python_path.write_text(full_code, encoding="utf-8")
transpiled = transpile_contract_library(full_code, module_name_hint="generated_csd", axiomatize=False)
if transpiled.is_ok():
    (run_dir / "generated_csd.dfy").write_text(transpiled.value, encoding="utf-8")

(repo_root / "outputs" / "generated-csd" / "latest_run.txt").write_text(str(run_dir) + "\n", encoding="utf-8")
print(f"Compiled run written to: {run_dir}")
PY

echo "Using compiled run: $RUN_DIR"
echo "Starting evaluation..."
echo "Arguments: $*"

python -m evaluations.gsm_symbolic.cli \
  --run-dir "$RUN_DIR" \
  "$@"
