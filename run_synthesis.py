#!/usr/bin/env python3
"""
CLI entry point for CSD synthesis pipeline.

Usage:
    python run_synthesis.py --task "Generate a strategy that..."
    python run_synthesis.py --task "..." --max-iterations 10
    python run_synthesis.py --task "..." --model Qwen/Qwen2.5-Coder-3B-Instruct
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import secrets


def main():
    parser = argparse.ArgumentParser(
        description="Synthesize constrained decoding strategies using Qwen",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python run_synthesis.py --task "Generate a strategy for JSON output"
  
  # With custom iterations
  python run_synthesis.py --task "Generate a CRANE-style strategy" --max-iterations 10
  
  # Use a smaller model for faster testing
  python run_synthesis.py --task "Generate a simple retry strategy" \\
      --model Qwen/Qwen2.5-Coder-3B-Instruct
  
  # Specify output name
  python run_synthesis.py --task "..." --output-name my_strategy
"""
    )
    
    parser.add_argument(
        "--task", "-t",
        type=str,
        required=True,
        help="Task description for strategy generation"
    )
    
    parser.add_argument(
        "--max-iterations", "-n",
        type=int,
        default=5,
        help="Maximum refinement iterations (default: 5)"
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="Qwen/Qwen2.5-Coder-7B-Instruct",
        help="HuggingFace model name (default: Qwen/Qwen2.5-Coder-7B-Instruct)"
    )
    
    parser.add_argument(
        "--output-name", "-o",
        type=str,
        default="generated_csd",
        help="Name for the output module (default: generated_csd)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Base output directory (default: outputs/generated-csd/). Each run writes into a unique subfolder."
    )
    
    parser.add_argument(
        "--dafny-path",
        type=str,
        default="dafny",
        help="Path to Dafny executable (default: dafny)"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for Qwen (default: 0.7)"
    )
    
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum tokens to generate per attempt (default: 256)"
    )
    
    parser.add_argument(
        "--no-save-reports",
        action="store_true",
        help="Don't save failure/success reports to disk"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "mps", "cpu", "auto"],
        default="auto",
        help="Device for model inference (default: auto)"
    )
    
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing GeneratedCSD.dfy without generating"
    )
    
    parser.add_argument(
        "--compile-only",
        action="store_true",
        help="Verify and compile existing GeneratedCSD.dfy without generating"
    )

    args = parser.parse_args()
    
    # Handle verify-only mode
    if args.verify_only or args.compile_only:
        run_verification_only(args)
        return
    
    # Import here to avoid loading heavy dependencies if just showing help
    from synthesis.generator import StrategyGenerator
    from synthesis.verifier import DafnyVerifier
    from synthesis.compiler import DafnyCompiler
    from synthesis.feedback_loop import SynthesisPipeline, SynthesisExhaustionError
    
    # Create components
    print("Initializing synthesis pipeline...")
    
    device = None if args.device == "auto" else args.device
    
    generator = StrategyGenerator(
        model_name=args.model,
        device=device,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature
    )
    
    verifier = DafnyVerifier(dafny_path=args.dafny_path)
    # Compiler output dir is set per-run inside the pipeline (so runs don't overwrite each other).
    compiler = DafnyCompiler(dafny_path=args.dafny_path, output_dir=args.output_dir)
    # Runner is created by the pipeline with task-appropriate parser mode
    
    pipeline = SynthesisPipeline(
        generator=generator,
        verifier=verifier,
        compiler=compiler,
        runner=None,  # Let pipeline create task-appropriate runner
        max_iterations=args.max_iterations,
        output_dir=args.output_dir,
        save_reports=not args.no_save_reports
    )
    
    # Run synthesis
    try:
        result = pipeline.synthesize(
            task_description=args.task,
            output_name=args.output_name
        )
        
        print("\n" + "=" * 60)
        print("SYNTHESIS COMPLETE")
        print("=" * 60)
        print(f"Strategy: {result.strategy_code}")
        print(f"Compiled module: {result.compiled_module_path}")
        print(f"Output directory: {result.output_dir}")
        if getattr(result, "run_dir", None):
            print(f"Run directory: {result.run_dir}")
        print(f"Total attempts: {len(result.attempts)}")
        print(f"Total time: {result.total_time_ms:.1f}ms")

        sys.exit(0)
        
    except SynthesisExhaustionError as e:
        print("\n" + "=" * 60)
        print("SYNTHESIS FAILED")
        print("=" * 60)
        print(e.get_failure_summary())
        
        sys.exit(1)
        
    except KeyboardInterrupt:
        print("\n\nSynthesis interrupted by user")
        sys.exit(130)
        
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def run_verification_only(args):
    """Run verification/compilation on existing file without generation."""
    from synthesis.verifier import DafnyVerifier
    from synthesis.compiler import DafnyCompiler
    from synthesis.runner import StrategyRunner
    
    dafny_file = Path(__file__).parent / "dafny" / "GeneratedCSD.dfy"
    
    if not dafny_file.exists():
        print(f"Error: {dafny_file} not found")
        sys.exit(1)
    
    print(f"Processing: {dafny_file}")
    
    # Verification
    print("\n[1/3] Verifying...")
    verifier = DafnyVerifier(dafny_path=args.dafny_path)
    result = verifier.verify_file(dafny_file)
    
    if not result.success:
        print("✗ Verification failed:")
        print(result.get_error_summary())
        sys.exit(1)
    
    print("✓ Verification passed")
    
    if args.verify_only:
        sys.exit(0)
    
    # Compilation
    print("\n[2/3] Compiling to Python...")
    base_output_dir = args.output_dir or (Path(__file__).parent / "outputs" / "generated-csd")
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + secrets.token_hex(3)
    run_dir = base_output_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    try:
        (base_output_dir / "latest_run.txt").write_text(str(run_dir) + "\n")
    except Exception:
        pass

    compiler = DafnyCompiler(dafny_path=args.dafny_path, output_dir=run_dir)
    compile_result = compiler.compile_file(dafny_file, args.output_name)
    
    if not compile_result.success:
        print("✗ Compilation failed:")
        print(compile_result.get_error_summary())
        sys.exit(1)
    
    print(f"✓ Compiled to {compile_result.output_dir}")
    print(f"Run directory: {run_dir}")
    
    # Runtime test
    print("\n[3/3] Testing runtime...")
    if compile_result.main_module_path:
        runner = StrategyRunner()
        runtime_result = runner.run(compile_result.main_module_path)
        
        if not runtime_result.success:
            print("✗ Runtime error:")
            print(runtime_result.get_error_summary())
            sys.exit(1)
        
        print(f"✓ Execution successful ({runtime_result.execution_time_ms:.1f}ms)")
    else:
        print("⚠ No main module found for runtime testing")
    
    print("\n✓ All checks passed")
    sys.exit(0)


if __name__ == "__main__":
    main()

