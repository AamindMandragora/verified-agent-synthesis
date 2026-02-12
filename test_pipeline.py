
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(os.getcwd())

from synthesis.feedback_loop import SynthesisExhaustionError
from synthesis.generator import StrategyGenerator
from synthesis.verifier import DafnyVerifier
from synthesis.compiler import DafnyCompiler
from synthesis.runner import StrategyRunner
from synthesis.feedback_loop import SynthesisAttempt

def test_synthesis():
    # We won't actually call the LLM in this test to avoid costs/delays
    # Instead, we'll mock the generator to return a specific strategy
    
    # 1. Setup components
    # Use real paths for tools
    dafny_bin = "/home/advayth2/projects/verified-agent-synthesis/dafny-lang/dafny/dafny"
    verifier = DafnyVerifier(dafny_path=dafny_bin)
    compiler = DafnyCompiler(dafny_path=dafny_bin)
    runner = StrategyRunner()
    
    # 2. Mock strategy
    # This strategy uses HybridGeneration and should satisfy cost <= 2 * maxSteps
    strategy_code = """
    // CSD_RATIONALE_BEGIN
    // We use HybridGeneration to interleave reasoning and constrained steps.
    // The cost will be at most 2 * maxSteps.
    // CSD_RATIONALE_END
    generated := helpers.HybridGeneration(lm, parser, prompt, maxSteps);
    """
    cost_contract = "ensures cost <= 2 * maxSteps"
    
    # 3. Inject and verify
    generator = StrategyGenerator()
    full_code = generator.inject_strategy(strategy_code, cost_contract)
    
    # Save to a temporary file
    temp_dfy = Path("temp_test_strategy.dfy")
    temp_dfy.write_text(full_code)
    
    print(f"Testing strategy verification...")
    v_result = verifier.verify(full_code)
    if not v_result.success:
        print(f"Verification failed: {v_result.errors}")
        return False
    print("Verification successful!")
    
    print(f"Testing strategy compilation...")
    c_result = compiler.compile(full_code)
    if not c_result.success:
        print(f"Compilation failed: {c_result.error_message}")
        return False
    print("Compilation successful!")
    
    print(f"Testing strategy execution...")
    r_result = runner.run(c_result.main_module_path)
    if not r_result.success:
        print(f"Execution failed: {r_result.error_message}")
        if r_result.error_traceback:
            print(f"Traceback:\n{r_result.error_traceback}")
        return False
    print(f"Execution successful! Generated {len(r_result.output)} tokens with cost {r_result.cost}")
    
    # Cleanup
    # if temp_dfy.exists():
    #    temp_dfy.unlink()
    
    return True

if __name__ == "__main__":
    success = test_synthesis()
    if success:
        print("\nPipeline verification PASSED")
        sys.exit(0)
    else:
        print("\nPipeline verification FAILED")
        sys.exit(1)
