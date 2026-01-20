#!/usr/bin/env python3
"""
FOLIO First-Order Logic Reasoning Evaluation with CRANE-Style CSD.

Dataset: yale-nlp/FOLIO (HuggingFace)
Metrics:
  - Answer accuracy (True/False/Uncertain classification)
  - FOL structure validity (has Predicates, Premises, Conclusion, Answer sections)
  - Syntax validity (FOL expressions inside << >> pass grammar validation)

Architecture: Evaluation-level orchestration of CRANE-style windowing with CSD
  - Unconstrained reasoning until << detected
  - Run CSD strategy for constrained FOL expression
  - Validate after >> delimiter
  - Resume unconstrained until Answer: or EOS

Example:
  python -m evaluations.folio.cli \\
    --run-dir outputs/generated-csd/runs/20260109_XXXXXX \\
    --model Qwen/Qwen2.5-Coder-7B-Instruct --device cuda --limit 50
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

# Import from the modular package
from evaluations.folio.dataset import (
    load_folio, 
    load_folio_from_json,
    create_synthetic_folio_examples,
)
from evaluations.folio.prompts import make_folio_prompt
from evaluations.folio.answer_extraction import (
    extract_answer,
    is_valid_fol_structure,
    check_answer_correctness,
)
from evaluations.folio.grammar import (
    load_base_grammar,
    build_dynamic_grammar,
    extract_predicates_from_generation,
    extract_constants_from_generation,
)
from evaluations.folio.prompts import extract_constants_from_problem
from evaluations.common.parser_utils import create_lark_dafny_parser
from evaluations.folio.metrics import FOLIOMetrics
from evaluations.folio.generation import run_crane_csd, run_unconstrained
from evaluations.folio.environment import setup_dafny_environment, verify_critical_tokens

# Project root for default paths
PROJECT_ROOT = Path(__file__).parent.parent.parent


def main():
    ap = argparse.ArgumentParser(
        description="Evaluate FOLIO with CRANE-CSD",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # CRANE-style CSD evaluation
  python -m evaluations.folio.cli \\
    --run-dir outputs/generated-csd/runs/20260110_180926_52ce55 \\
    --model Qwen/Qwen2.5-Coder-7B-Instruct --device cuda --limit 10

  # Use synthetic examples for testing
  python -m evaluations.folio.cli \\
    --run-dir outputs/generated-csd/runs/20260110_180926_52ce55 \\
    --synthetic

  # Load from local JSON file
  python -m evaluations.folio.cli \\
    --run-dir outputs/generated-csd/runs/20260110_180926_52ce55 \\
    --json-path data/folio/folio_validation.json
"""
    )
    ap.add_argument("--run-dir", type=Path, required=True,
                    help="Path to compiled CSD run directory")
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct",
                    help="HuggingFace model ID")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--split", default="validation",
                    help="Dataset split to use")
    ap.add_argument("--limit", type=int, default=100,
                    help="Max examples to evaluate")
    ap.add_argument("--max-steps", type=int, default=1500,
                    help="Max steps for generation")
    ap.add_argument("--vocab-size", type=int, default=2000,
                    help="Token vocabulary size limit")
    ap.add_argument("--grammar", type=Path, default=PROJECT_ROOT / "grammars" / "folio.lark",
                    help="Grammar file for FOL validation")
    ap.add_argument("--num-examples", type=int, default=2,
                    help="Number of few-shot examples in prompt")
    ap.add_argument("--verbose", action="store_true",
                    help="Show per-example details")
    ap.add_argument("--debug-delimiters", action="store_true",
                    help="Debug delimiter detection")
    ap.add_argument("--synthetic", action="store_true",
                    help="Use synthetic examples for testing")
    ap.add_argument("--json-path", type=Path,
                    help="Load from local JSON file instead of HuggingFace")
    ap.add_argument("--output", type=Path,
                    help="Path to save metrics JSON")
    ap.add_argument("--unconstrained", action="store_true",
                    help="Run unconstrained baseline instead of CSD")
    args = ap.parse_args()

    # Load dataset
    print("Loading dataset...")
    if args.synthetic:
        examples = create_synthetic_folio_examples()
        if args.limit and args.limit < len(examples):
            examples = examples[:args.limit]
    elif args.json_path:
        examples = load_folio_from_json(
            str(args.json_path), 
            num_samples=args.limit,
        )
    else:
        examples = load_folio(
            split=args.split,
            num_samples=args.limit,
        )
    
    n = len(examples)
    print(f"Loaded {n} examples")
    
    metrics = FOLIOMetrics()

    # Setup Dafny environment
    print(f"Setting up Dafny environment...")
    dafny_env = setup_dafny_environment(
        run_dir=args.run_dir,
        model_name=args.model,
        device=args.device,
        vocab_size=args.vocab_size,
        grammar_file=args.grammar,
    )
    verify_critical_tokens(dafny_env["tokenizer"])
    print("Model loaded. Starting evaluation...\n")

    # Track start time for ETA
    eval_start_time = time.time()

    # Create a grammar parser for format validation
    from parsers.lark_parser import LarkGrammarParser
    grammar_validator = LarkGrammarParser.from_grammar_file(str(args.grammar))

    for i, example in enumerate(examples):
        problem = example.problem
        question = example.question
        gold_label = example.label

        print(f"[{i+1}/{n}] Processing example {example.id}...", flush=True)
        
        prompt = make_folio_prompt(
            problem=problem,
            question=question,
            num_examples=args.num_examples,
        )

        if args.verbose:
            print(f"  Problem: {problem[:100]}...")
            print(f"  Question: {question[:100]}...")
            print(f"  Gold: {gold_label}")

        # Build dynamic grammar based on problem context
        # Extract potential constants from the problem text (proper nouns, names)
        constants = set(extract_constants_from_problem(problem, question))

        # Build dynamic grammar with extracted constants
        dynamic_grammar_text = None
        dynamic_parser = None
        base_grammar_text = args.grammar.read_text()

        if constants:
            dynamic_grammar_text = build_dynamic_grammar(
                allowed_predicates=None,  # Allow any predicates (model defines them)
                allowed_constants=constants,
                allowed_variables={'x', 'y', 'z'},  # Standard variables
            )
            if args.verbose:
                print(f"  [DEBUG] Dynamic grammar constants: {constants}")

        # Create dynamic parser if we have a dynamic grammar
        if dynamic_grammar_text and dafny_env:
            _dafny = dafny_env["_dafny"]
            VerifiedDecoderAgent = dafny_env["VerifiedDecoderAgent"]
            lm = dafny_env["lm"]

            DynParserClass = create_lark_dafny_parser(
                dynamic_grammar_text,
                VerifiedDecoderAgent,
                _dafny,
                start="start"
            )
            dynamic_parser = DynParserClass(lm._Tokens)
            if args.verbose:
                print(f"  [DEBUG] Created dynamic parser for constants: {constants}")

        # Generate
        if args.unconstrained:
            out_text, tok_count, dt = run_unconstrained(
                dafny_env, prompt, args.max_steps,
                debug=args.verbose,
            )
            constrained_segments = []
        else:
            out_text, tok_count, dt, constrained_segments = run_crane_csd(
                dafny_env, prompt, args.max_steps, args.grammar,
                debug_delimiters=args.debug_delimiters,
                dynamic_parser=dynamic_parser,
            )

        # Extract answer
        pred_answer = extract_answer(out_text)
        
        # Check structure validity
        valid_structure = is_valid_fol_structure(out_text)
        
        # Check correctness
        is_correct = check_answer_correctness(pred_answer, gold_label)

        # Update metrics
        metrics.update(
            predicted=pred_answer,
            gold=gold_label,
            is_correct=is_correct,
            valid_structure=valid_structure,
            fol_segments=constrained_segments,
            time_seconds=dt,
            tokens=tok_count,
            example_id=example.id,
        )

        # Calculate ETA
        avg_time = metrics.total_time / metrics.total
        remaining = n - metrics.total
        eta_seconds = avg_time * remaining
        eta_str = f"{eta_seconds:.0f}s" if eta_seconds < 60 else f"{eta_seconds/60:.1f}m"

        # Progress
        print(f"  -> Tokens: {tok_count} | Time: {dt:.2f}s | "
              f"Pred: {pred_answer} | Gold: {gold_label} | "
              f"Correct: {is_correct} | Valid: {valid_structure} | "
              f"Acc: {metrics.accuracy * 100:.1f}% | ETA: {eta_str}", flush=True)

        if args.verbose or (not is_correct and i < 5):
            print(f"  Problem: {problem}")
            print(f"  Question: {question}")
            print(f"  Output: {out_text[:500]}...")
            if constrained_segments:
                print(f"  FOL segments: {constrained_segments[:3]}")
            print()

    # Final results
    print("\n")
    metrics.print_summary()
    
    # Print additional info
    print(f"\nConfiguration:")
    print(f"  Method: {'Unconstrained' if args.unconstrained else 'CRANE-CSD'}")
    print(f"  Model: {args.model}")
    print(f"  Split: {args.split}")
    print(f"  CSD Run: {args.run_dir}")
    print(f"  Grammar: {args.grammar}")
    
    # Save metrics if output path specified
    if args.output:
        metrics.to_json(str(args.output))
        print(f"\nMetrics saved to: {args.output}")


if __name__ == "__main__":
    main()
