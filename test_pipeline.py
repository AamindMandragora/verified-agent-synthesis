"""
Quick pipeline smoke test: verify + compile + run a strategy that uses only
VerifiedAgentSynthesis.dfy primitives (UnconstrainedStep, ConstrainedStep).
Does not call the LLM. Requires Dafny on PATH or set DAFNY env.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, os.getcwd())

from synthesis.generator import StrategyGenerator
from synthesis.verifier import DafnyVerifier
from synthesis.compiler import DafnyCompiler
from synthesis.runner import StrategyRunner


def test_synthesis():
    dafny_bin = os.environ.get("DAFNY", "dafny")
    verifier = DafnyVerifier(dafny_path=dafny_bin)
    compiler = DafnyCompiler(dafny_path=dafny_bin)
    runner = StrategyRunner()

    # Strategy using only CSDHelpers primitives (stepsLeft consumed per step)
    strategy_code = """
    // CSD_RATIONALE_BEGIN
    // Constrained-only loop; each ConstrainedStep consumes one step (stepsLeft).
    // CSD_RATIONALE_END
    generated := [];
    while stepsLeft > 0 && !parser.IsCompletePrefix(generated)
      invariant parser.IsValidPrefix(generated)
      invariant 0 <= stepsLeft <= maxSteps
      invariant |generated| <= maxSteps
      decreases stepsLeft
    {
      CSDHelpers.RollbackPreservesTokenInvariant(lm, parser, generated);
      var next, newSteps := helpers.ConstrainedStep(lm, parser, prompt, generated, stepsLeft);
      generated := generated + [next];
      stepsLeft := newSteps;
    }
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
