#!/usr/bin/env python3
"""
CLI entry point for CSD synthesis pipeline with evaluation feedback loop.

The pipeline runs: generate → verify → compile → runtime → evaluate → refine
until evaluation thresholds are met or max iterations exhausted.

This is the main entry point for making CSDs. For a quick pipeline smoke test
without the LLM, use test_pipeline.py (optional).

Usage:
    python run_synthesis.py --task "..." --dataset gsm_symbolic \\
        --min-accuracy 0.3 --min-format-rate 0.5 --min-syntax-rate 0.5

    python run_synthesis.py --task "..." --dataset folio \\
        --min-accuracy 0.5 --min-format-rate 0.8 --min-syntax-rate 0.7
"""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Synthesize constrained decoding strategies using Qwen",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # GSM-Symbolic
  python run_synthesis.py --task "Generate math reasoning strategy" \\
      --dataset gsm_symbolic --min-accuracy 0.3 --min-format-rate 0.5 --min-syntax-rate 0.5

  # FOLIO
  python run_synthesis.py --task "Generate FOL reasoning strategy" \\
      --dataset folio --min-accuracy 0.5 --min-format-rate 0.8 --min-syntax-rate 0.7

  # SyGuS SLIA (string manipulation synthesis)
  python run_synthesis.py --task "Generate string manipulation strategy" \\
      --dataset sygus_slia --min-accuracy 0.3 --min-format-rate 0.5 --min-syntax-rate 0.5

  # PDDL Blocks World (sequential planning)
  python run_synthesis.py --task "Generate a PDDL planning strategy" \\
      --dataset pddl --min-accuracy 0.3 --min-format-rate 0.5 --min-syntax-rate 0.5

  # With more iterations and custom eval sample size
  python run_synthesis.py --task "..." --dataset gsm_symbolic \\
      --min-accuracy 0.3 --min-format-rate 0.5 --min-syntax-rate 0.5 \\
      --output-name my_strategy --max-iterations 10 --eval-sample-size 20
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
        default=None,
        help="Path to Dafny executable (default: repo dafny-lang/dafny/dafny if present, else 'dafny')"
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
        default=512,
        help="Max tokens per LLM generation (default: 512). Lower (e.g. 384) speeds up refinement at risk of truncation."
    )
    parser.add_argument(
        "--generation-timeout",
        type=int,
        default=600,
        metavar="SECS",
        help="Max seconds per LLM generation (default: 600). Use 0 to disable. Helps avoid unbounded hangs on slow/CPU runs."
    )
    parser.add_argument(
        "--no-save-reports",
        action="store_true",
        help="Don't save failure/success reports to disk"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device for model inference: auto, cuda, cuda:0, cuda:2, mps, cpu (default: auto)"
    )
    
    # Evaluation arguments (required - evaluation is part of the synthesis loop)
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        choices=["gsm_symbolic", "folio", "sygus_slia", "pddl"],
        required=True,
        help="Dataset to use for evaluation feedback (required)"
    )

    parser.add_argument(
        "--min-accuracy",
        type=float,
        required=True,
        help="Minimum accuracy threshold for evaluation (e.g. 0.3)"
    )

    parser.add_argument(
        "--min-format-rate",
        type=float,
        required=True,
        help="Minimum format validity rate threshold (e.g. 0.5)"
    )

    parser.add_argument(
        "--min-syntax-rate",
        type=float,
        required=True,
        help="Minimum syntax validity rate threshold (e.g. 0.5)"
    )

    parser.add_argument(
        "--eval-sample-size",
        type=int,
        default=1,
        help="Number of examples to evaluate on per iteration (default: 1)"
    )

    parser.add_argument(
        "--eval-max-steps",
        type=int,
        default=150,
        help="Maximum generation steps during evaluation (default: 150)"
    )

    parser.add_argument(
        "--eval-vocab-size",
        type=int,
        default=3000,
        help="Vocabulary size for evaluation (default: 3000)"
    )

    parser.add_argument(
        "--no-eval-4bit",
        action="store_true",
        help="Disable 4-bit evaluation (default: use 4-bit for speed and lower memory)"
    )

    args = parser.parse_args()
    args.eval_4bit = not args.no_eval_4bit

    # Resolve Dafny path: prefer repo-local dafny-lang/dafny/dafny if present
    if args.dafny_path is None:
        repo_root = Path(__file__).resolve().parent
        local_dafny = repo_root / "dafny-lang" / "dafny" / "dafny"
        args.dafny_path = str(local_dafny) if local_dafny.is_file() else "dafny"

    # Normalize output_dir if provided (handle potential backslashes from user input)
    if args.output_dir:
        args.output_dir = Path(str(args.output_dir).replace("\\", "/"))

    # Import here to avoid loading heavy dependencies if just showing help
    from synthesis.generator import StrategyGenerator
    from synthesis.verifier import DafnyVerifier
    from synthesis.compiler import DafnyCompiler
    from synthesis.evaluator import Evaluator
    from synthesis.feedback_loop import SynthesisPipeline, SynthesisExhaustionError

    # Create components
    print("Initializing synthesis pipeline...")

    # Normalize device: "auto" -> None (generator picks); cuda:2 etc. passed through
    device = None if args.device.strip().lower() == "auto" else args.device.strip()

    generator = StrategyGenerator(
        model_name=args.model,
        device=device,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        generation_timeout=args.generation_timeout if args.generation_timeout > 0 else None,
    )

    verifier = DafnyVerifier(dafny_path=args.dafny_path)
    # Compiler output dir is set per-run inside the pipeline (so runs don't overwrite each other).
    compiler = DafnyCompiler(dafny_path=args.dafny_path, output_dir=args.output_dir)
    # Runner is created by the pipeline with task-appropriate parser mode

    # Create evaluator for the feedback loop
    print(f"Setting up evaluator for dataset: {args.dataset}")
    evaluator = Evaluator(
        dataset_name=args.dataset,
        model_name=args.model,
        device=device or "cuda",
        vocab_size=args.eval_vocab_size,
        sample_size=args.eval_sample_size,
        max_steps=args.eval_max_steps,
        load_in_4bit=args.eval_4bit,
    )

    pipeline = SynthesisPipeline(
        evaluator=evaluator,
        generator=generator,
        verifier=verifier,
        compiler=compiler,
        runner=None,  # Let pipeline create task-appropriate runner
        max_iterations=args.max_iterations,
        output_dir=args.output_dir,
        save_reports=not args.no_save_reports,
        # Evaluation thresholds
        min_accuracy=args.min_accuracy,
        min_format_rate=args.min_format_rate,
        min_syntax_rate=args.min_syntax_rate,
        eval_sample_size=args.eval_sample_size,
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


if __name__ == "__main__":
    main()

