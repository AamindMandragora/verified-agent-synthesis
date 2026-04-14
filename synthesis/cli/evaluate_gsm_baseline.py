#!/usr/bin/env python3
"""
Build and evaluate the fixed GSM baseline strategy.

This replaces the old shell script that embedded a large Python heredoc and
hard-coded machine-specific environment variables.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from verification.compiler import DafnyCompiler
from generation.generator import StrategyGenerator
from verification.verifier import DafnyVerifier
from verification.transpiler.transpiler import transpile_contract_library


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build and evaluate the fixed GSM baseline strategy.",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-Coder-7B-Instruct",
        help="Evaluation model name.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device passed to the GSM evaluator.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="Number of examples to evaluate.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1024,
        help="Maximum generation steps during evaluation.",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=3000,
        help="Vocabulary size for evaluation.",
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Load the evaluation model in 4-bit mode.",
    )
    parser.add_argument(
        "--debug-delimiters",
        action="store_true",
        help="Enable delimiter debugging in the evaluator.",
    )
    return parser


def _build_baseline_run() -> Path:
    outputs_dir = PROJECT_ROOT / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    run_dir = outputs_dir / f"vanilla_baseline_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    dafny_bin = PROJECT_ROOT / "dafny" / "dafny"
    full_code = StrategyGenerator().inject_strategy(StrategyGenerator.STARTER_STRATEGY)

    verifier = DafnyVerifier(dafny_path=str(dafny_bin))
    verify_result = verifier.verify(full_code)
    if not verify_result.success:
        raise RuntimeError("Verification failed:\n" + verify_result.get_error_summary())

    compiler = DafnyCompiler(dafny_path=str(dafny_bin), output_dir=run_dir)
    compile_result = compiler.compile(full_code, output_name="generated_csd")
    if not compile_result.success:
        raise RuntimeError("Compilation failed:\n" + compile_result.get_error_summary())

    (run_dir / "generated_csd.py").write_text(full_code, encoding="utf-8")
    transpiled = transpile_contract_library(
        full_code,
        module_name_hint="generated_csd",
        axiomatize=False,
    )
    if transpiled.is_ok():
        (run_dir / "generated_csd.dfy").write_text(transpiled.value, encoding="utf-8")

    (PROJECT_ROOT / "outputs" / "latest_run.txt").write_text(
        str(run_dir) + "\n",
        encoding="utf-8",
    )
    return run_dir


def main() -> int:
    args = _build_arg_parser().parse_args()
    run_dir = _build_baseline_run()

    command = [
        sys.executable,
        "-m",
        "evaluation.gsm_symbolic.cli",
        "--run-dir",
        str(run_dir),
        "--model",
        args.model,
        "--device",
        args.device,
        "--limit",
        str(args.limit),
        "--max-steps",
        str(args.max_steps),
        "--vocab-size",
        str(args.vocab_size),
    ]
    if args.load_in_4bit:
        command.append("--load-in-4bit")
    if args.debug_delimiters:
        command.append("--debug-delimiters")

    print(f"Built baseline run in {run_dir}")
    result = subprocess.run(command, cwd=PROJECT_ROOT, env=os.environ.copy())
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
