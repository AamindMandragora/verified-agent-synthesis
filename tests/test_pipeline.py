"""
Quick pipeline smoke test for the Python-first CSD synthesis path.

This verifies that we can:
- generate a Python strategy body
- inject it into `generation/csd/GeneratedAgentTemplate.py`
- verify it through the transpiler
- compile the transpiled Dafny to Python
- execute the compiled strategy
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, os.getcwd())

from generation.generator import StrategyGenerator
from verification.verifier import DafnyVerifier
from verification.compiler import DafnyCompiler
from synthesis.runner import StrategyRunner


def test_synthesis():
    repo_root = Path(os.getcwd())
    dafny_bin = os.environ.get("DAFNY", str(repo_root / "dafny" / "dafny"))
    verifier = DafnyVerifier(dafny_path=dafny_bin)
    compiler = DafnyCompiler(dafny_path=dafny_bin)
    runner = StrategyRunner()

    strategy_code = """
# CSD_RATIONALE_BEGIN
# Simple smoke test for the Python-to-Dafny pipeline with explicit expressive free-form steps and a separate constrained answer channel.
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
    generator = StrategyGenerator()
    full_code = generator.inject_strategy(strategy_code)

    print("Testing strategy verification...")
    v_result = verifier.verify(full_code)
    if not v_result.success:
        print("Verification failed:", v_result.get_error_summary())
        return False
    print("Verification successful!")

    print("Testing strategy compilation...")
    c_result = compiler.compile(full_code, output_name="test_csd")
    if not c_result.success:
        print("Compilation failed:", c_result.get_error_summary())
        return False
    print("Compilation successful!")

    print("Testing strategy execution...")
    if c_result.main_module_path is None:
        print("No main module path")
        return False
    r_result = runner.run(c_result.main_module_path)
    if not r_result.success:
        print("Execution failed:", r_result.get_error_summary())
        return False
    print(f"Execution successful! Output length: {len(r_result.output or [])} tokens, steps used: {r_result.cost}")
    return True


if __name__ == "__main__":
    success = test_synthesis()
    if success:
        print("\nPipeline verification PASSED")
        sys.exit(0)
    else:
        print("\nPipeline verification FAILED")
        sys.exit(1)
