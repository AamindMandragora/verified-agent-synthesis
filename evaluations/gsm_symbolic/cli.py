#!/usr/bin/env python3
"""
GSM-Symbolic Math Reasoning Evaluation with CRANE-Style CSD.

Dataset: apple/GSM-Symbolic (HuggingFace)
Metrics:
  - Answer accuracy (exact numeric match)
  - Syntax validity (math expressions inside << >> pass grammar validation)
  - Valid format rate (outputs contain #### <number>)

Architecture: Evaluation-level orchestration of CRANE-style windowing with CSD
  - Unconstrained reasoning until << detected
  - Run CSD strategy for constrained math expression
  - Validate after >> delimiter
  - Resume unconstrained until #### or EOS

Example:
  python -m evaluations.gsm_symbolic.cli \\
    --run-dir outputs/generated-csd/runs/20260109_XXXXXX \\
    --model Qwen/Qwen2.5-Coder-7B-Instruct --device cuda --limit 50
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

# Import from the modular package
from evaluations.gsm_symbolic.dataset import load_gsm_symbolic
from evaluations.gsm_symbolic.prompts import make_gsm_prompt, symbolize_question
from evaluations.gsm_symbolic.answer_extraction import extract_answer, extract_gold_answer
from evaluations.gsm_symbolic.grammar import build_dynamic_grammar
from evaluations.gsm_symbolic.metrics import GSMMetrics
from evaluations.gsm_symbolic.generation import run_crane_csd, run_unconstrained
from evaluations.gsm_symbolic.environment import (
    setup_dafny_environment,
    verify_critical_tokens,
)
from evaluations.common.parser_utils import create_lark_dafny_parser

# Project root for default paths
PROJECT_ROOT = Path(__file__).parent.parent.parent


def main():
    ap = argparse.ArgumentParser(
        description="Evaluate GSM-Symbolic with CRANE-CSD",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # CRANE-style CSD evaluation
  python -m evaluations.gsm_symbolic.cli \\
    --run-dir outputs/generated-csd/runs/20260110_180926_52ce55 \\
    --model Qwen/Qwen2.5-Coder-7B-Instruct --device cuda --limit 10
"""
    )
    ap.add_argument("--run-dir", type=Path, required=True,
                    help="Path to compiled CSD run directory")
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct",
                    help="HuggingFace model ID (7B recommended for better instruction following)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--config", choices=["main", "p1", "p2"], default="main",
                    help="GSM-Symbolic difficulty level")
    ap.add_argument("--limit", type=int, default=100,
                    help="Max examples to evaluate")
    ap.add_argument("--max-steps", type=int, default=1024,
                    help="Max steps for generation")
    ap.add_argument("--vocab-size", type=int, default=2000,
                    help="Token vocabulary size limit")
    ap.add_argument("--grammar", type=Path, default=PROJECT_ROOT / "grammars" / "gsm.lark",
                    help="Grammar file for math validation")
    ap.add_argument("--verbose", action="store_true",
                    help="Show per-example details")
    ap.add_argument("--debug-delimiters", action="store_true",
                    help="Debug delimiter detection")
    ap.add_argument("--random-sample", action="store_true",
                    help="Randomly sample examples instead of taking first N")
    ap.add_argument("--unconstrained", action="store_true",
                    help="Run unconstrained baseline instead of CSD")
    ap.add_argument("--load-in-4bit", action="store_true",
                    help="Load model in 4-bit quantization")
    ap.add_argument("--load-in-8bit", action="store_true",
                    help="Load model in 8-bit quantization")
    args = ap.parse_args()

    # Load dataset
    ds = load_gsm_symbolic(args.config, limit=args.limit, random_sample=args.random_sample)
    n = len(ds)
    metrics = GSMMetrics()

    # Setup Dafny environment
    print(f"Setting up Dafny environment...")
    dafny_env = setup_dafny_environment(
        run_dir=args.run_dir,
        model_name=args.model,
        device=args.device,
        vocab_size=args.vocab_size,
        grammar_file=args.grammar,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
    )
    verify_critical_tokens(dafny_env["tokenizer"])
    print("Model loaded. Starting evaluation...\n")

    # Track start time for ETA
    eval_start_time = time.time()

    # Create a grammar parser for format validation
    # Use 'any_expr' to validate expressions (allows both symbolic and numeric)
    from parsers.lark_parser import LarkGrammarParser
    grammar_validator = LarkGrammarParser.from_grammar_file(str(args.grammar), start='any_expr')

    for i in range(n):
        example = ds[i]
        question = example.get("question", "")
        gold_answer_text = example.get("answer", "")
        gold_answer = extract_gold_answer(gold_answer_text)

        if not question:
            continue

        print(f"[{i+1}/{n}] Processing example...", flush=True)
        
        # Convert question to symbolic form
        symbolic_question, variable_mapping = symbolize_question(question)
        
        # Build dynamic grammar for this example
        dynamic_grammar_text = None
        base_grammar_text = args.grammar.read_text()
        variables = list(variable_mapping.keys())
        if variables:
            dynamic_grammar_text = build_dynamic_grammar(base_grammar_text, variables)
            if args.verbose:
                print(f"  [DEBUG] Dynamic grammar variables: {variables}")
        
        if args.verbose:
            print(f"  Original: {question[:100]}...")
            print(f"  Symbolic: {symbolic_question[:100]}...")
            print(f"  Variables: {variable_mapping}")
        
        prompt = make_gsm_prompt(question, symbolic_question=symbolic_question)

        # Generate using CSD
        # Create dynamic parser if available
        dynamic_parser = None
        if dynamic_grammar_text and dafny_env:
            _dafny = dafny_env["_dafny"]
            VerifiedDecoderAgent = dafny_env["VerifiedDecoderAgent"]
            lm = dafny_env["lm"]
            
            DynParserClass = create_lark_dafny_parser(dynamic_grammar_text, VerifiedDecoderAgent, _dafny, start="csd_start")
            dynamic_parser = DynParserClass(lm._Tokens)
            if args.verbose:
                print(f"  [DEBUG] Created dynamic parser for variables: {variables}")
            
        # Generate using CSD or unconstrained baseline
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
        pred_answer, valid_format, symbolic_expr = extract_answer(
            out_text,
            variable_mapping=variable_mapping,
            grammar_parser=grammar_validator,
            debug=args.verbose
        )
        
        if args.verbose:
            if symbolic_expr:
                print(f"  Symbolic expression: << {symbolic_expr} >>")
                print(f"  Variable mapping: {variable_mapping}")
                print(f"  Evaluated value: {pred_answer}")
            else:
                print(f"  No symbolic expression found in output")
        
        # Answer accuracy
        is_correct = False
        if pred_answer is not None and gold_answer is not None:
            if abs(pred_answer - gold_answer) < 1e-6:
                is_correct = True
            elif gold_answer != 0 and abs((pred_answer - gold_answer) / gold_answer) < 0.01:
                is_correct = True

        # Update metrics
        metrics.update(
            is_correct=is_correct,
            is_valid_format=valid_format,
            token_count=tok_count,
            time_seconds=dt,
            constrained_segments=constrained_segments
        )

        # Calculate ETA
        avg_time = metrics.total_time / metrics.n
        remaining = n - metrics.n
        eta_seconds = avg_time * remaining
        eta_str = f"{eta_seconds:.0f}s" if eta_seconds < 60 else f"{eta_seconds/60:.1f}m"

        # Progress
        parse_pct = metrics.format_rate()
        print(f"  -> Tokens: {tok_count} | Time: {dt:.2f}s | "
              f"Correct: {is_correct} | Format: {valid_format} | "
              f"Acc: {metrics.accuracy():.1f}% | Parse: {parse_pct:.1f}% | ETA: {eta_str}", flush=True)

        if args.verbose or (not valid_format and i < 5) or (pred_answer is None and valid_format) or (not is_correct and i < 3):
            print(f"  Question: {question}")
            print(f"  Symbolic Q: {symbolic_question}")
            print(f"  Pred: {pred_answer} | Gold: {gold_answer}")
            if symbolic_expr:
                print(f"  Symbolic expression: << {symbolic_expr} >>")
            print(f"  Output: {out_text}")
            if constrained_segments:
                print(f"  Math segments: {constrained_segments[:3]}")
            print()

    # Final results
    print("\n" + "=" * 60)
    print("GSM-SYMBOLIC RESULTS")
    print("=" * 60)
    print(f"Method: {'Unconstrained Baseline' if args.unconstrained else 'CRANE-CSD'}")
    print(f"Model: {args.model}")
    print(f"Config: {args.config}")
    if not args.unconstrained:
        print(f"CSD Run: {args.run_dir}")
        print(f"Grammar: {args.grammar}")
    print()
    print(metrics.summary())


if __name__ == "__main__":
    main()
