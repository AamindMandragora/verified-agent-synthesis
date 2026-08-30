#!/usr/bin/env python3
"""
CLI entry point for CSD synthesis pipeline with evaluation feedback loop.

The pipeline runs: generate → verify → compile → evaluate → refine
until evaluation thresholds are met or max iterations exhausted.

Simplified 2026-07-17: everything with exactly one correct value moved to
synthesis/run_constants.py. The remaining flags are run identity + the
science (dataset, models, bars, iterations, sample size). vLLM sizing
(GPU memory share, context length) and the bar's split side are now
hard-coded settled constants (synthesis always scores itself on train).

Usage:
    python -m synthesis.run_synthesis --task "..." --dataset gsm_symbolic \\
        --min-accuracy 0.3 --min-syntax-rate 0.5
"""

import argparse
import json
import os
import sys
from datetime import date
from pathlib import Path
from synthesis.generate.provider_names import (
    GENERATION_BACKENDS,
    normalize_generation_backend,
)
from synthesis.run_constants import (
    EVAL_BACKEND,
    EVAL_EARLY_STOP_ON_ANSWER,
    GSM_SOURCE_DIR,
    MIN_EXAMPLES_BEFORE_THRESHOLD_STOP,
    OUTPUT_DIR,
    SPLIT_FILE_BY_DATASET,
    SYNTHESIS_SPLIT_NAME,
    SYNTHESIZER_REASONING_BUDGET_DEFAULT,
    TEMPERATURE,
    VLLM_GPU_MEMORY_UTILIZATION,
    VLLM_GPU_MEMORY_UTILIZATION_BY_MODEL,
    VLLM_MAX_MODEL_LEN,
)
from synthesis.safe_logging import display_text
try:
    from synthesis.project_defaults import default_dafny_path
except ImportError:
    from project_defaults import default_dafny_path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass


def _seconds_or_no_cap(raw: str):
    """Parse a seconds value where zero or less means "no cap at all".

    Returned as None, which is what SynthesisPipeline already treats as
    unlimited, so the "off" case needs no special handling downstream.
    """
    value = float(raw)
    return value if value > 0 else None


def _load_initial_attempt_history(path: Path):
    """Restore evaluated attempts needed by refinement and helper selection."""
    from synthesis.evaluate.evaluator import EvaluationResult
    from synthesis.evaluate.feedback_loop import SynthesisAttempt

    records = json.loads(path.read_text(encoding="utf-8"))
    attempts = []
    for record in records:
        eval_result = EvaluationResult(
            success=True,
            accuracy=float(record["accuracy"]),
            contains_delimiters=bool(record["contains_delimiters"]),
            syntax_rate=float(record["syntax_rate"]),
            num_examples=int(record["num_examples"]),
            num_correct=int(record["num_correct"]),
            total_time_seconds=float(record.get("total_time_seconds", 0.0)),
        )
        attempts.append(
            SynthesisAttempt(
                attempt_number=int(record["attempt_number"]),
                strategy_code=str(record["strategy_code"]),
                full_dafny_code="",
                timestamp=str(record.get("timestamp", "restored")),
                eval_result=eval_result,
            )
        )
    return attempts


def _resolve_vllm_gpu_memory_utilization(eval_model: str | None = None) -> float:
    """Per-cell override from the cold-queue scheduler, else the per-model default, else the global constant."""
    raw = os.environ.get("CSD_VLLM_GPU_MEMORY_UTILIZATION", "").strip()
    if raw:
        return float(raw)
    if eval_model in VLLM_GPU_MEMORY_UTILIZATION_BY_MODEL:
        return float(VLLM_GPU_MEMORY_UTILIZATION_BY_MODEL[eval_model])
    return float(VLLM_GPU_MEMORY_UTILIZATION)


def _derive_output_name(dataset: str, eval_model: str) -> str:
    """Auto-derive the run label as <dataset>_<model>_<date>."""
    model_short = eval_model.split("/")[-1].replace(".", "p").replace("-", "_").lower()
    return f"{dataset}_{model_short}_{date.today().isoformat()}"


def evaluator_smiles_classes(dataset: str, raw: str | None) -> list[str] | None:
    """Classes to record on the evaluator: a list for SMILES, None elsewhere.

    Only SMILES runs are given --smiles-classes, so on gsm/spider the argparse
    default would otherwise be recorded as the classes the run evaluated, which
    is both untrue and rejected downstream.
    """
    if dataset != "smiles" or not raw:
        return None
    return [part.strip() for part in raw.split(",") if part.strip()]


def main():
    parser = argparse.ArgumentParser(
        description="Synthesize constrained decoding strategies (author: hosted large reasoning model; eval: vLLM)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # GSM-Symbolic (train side of the canonical split, bar measured on train)
  python -m synthesis.run_synthesis --task "Generate math reasoning strategy" \\
      --dataset gsm_symbolic \\
      --min-accuracy 0.3 --min-syntax-rate 0.5
"""
    )

    # --- run identity + the science -------------------------------------
    parser.add_argument(
        "--task", "-t",
        type=str,
        required=True,
        help="Task description for strategy generation"
    )

    parser.add_argument(
        "--dataset", "-d",
        type=str,
        choices=["gsm_symbolic", "spider", "smiles"],
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
        "--min-syntax-rate",
        type=float,
        required=True,
        help="Minimum syntax validity rate threshold (e.g. 0.5)"
    )

    parser.add_argument(
        "--max-iterations", "-n",
        type=int,
        default=40,
        help="Maximum refinement iterations (default: 40)"
    )

    parser.add_argument(
        "--eval-model",
        type=str,
        default="Qwen/Qwen2.5-Coder-7B-Instruct",
        help="Model for evaluation data generation/runtime (default: Qwen/Qwen2.5-Coder-7B-Instruct)"
    )

    parser.add_argument(
        "--eval-sample-size",
        type=int,
        default=10,
        help="Number of examples to evaluate on per iteration (default: 10)"
    )

    parser.add_argument(
        "--eval-max-steps",
        type=int,
        default=600,
        help="Maximum generation steps during evaluation (default: 600)"
    )

    parser.add_argument(
        "--eval-step-token-budget",
        type=int,
        default=1,
        help="Max tokens per outer generation step (1=token-level default, >1=symbol-level for structured outputs like SQL)"
    )

    parser.add_argument(
        "--refinement-beam-size",
        type=int,
        default=2,
        help="Number of refinement candidates sampled per feedback step.",
    )
    parser.add_argument(
        "--helper-selection-policy",
        choices=["bandit"],
        default="bandit",
        help="Helper selection policy used by the feedback loop.",
    )
    mask_group = parser.add_mutually_exclusive_group()
    mask_group.add_argument(
        "--adaptive-helper-mask",
        dest="adaptive_helper_mask",
        action="store_true",
        default=True,
        help="Enable empirical helper masking.",
    )
    mask_group.add_argument(
        "--no-adaptive-helper-mask",
        dest="adaptive_helper_mask",
        action="store_false",
        help="Disable empirical helper masking.",
    )

    parser.add_argument(
        "--eval-max-seconds-per-example",
        type=float,
        default=90.0,
        help="Runtime budget per evaluated example in seconds (default: 90)"
    )

    parser.add_argument(
        "--max-attempt-seconds",
        type=_seconds_or_no_cap,
        default=3600.0,
        help=(
            "Wall-clock cap on a single synthesis attempt, in seconds "
            "(default: 3600). An attempt that exceeds it is marked timed out "
            "and the loop moves on instead of hanging. The clock covers the "
            "whole attempt, but only the evaluation stage can be interrupted "
            "part-way -- Dafny verification runs to completion. Pass 0 or a "
            "negative number to remove the cap entirely."
        ),
    )

    parser.add_argument(
        "--eval-min-examples-before-threshold-stop",
        type=int,
        default=15,
        help="Minimum number of evaluated examples before threshold-impossible "
        "early stops (target accuracy / target syntax) can fire. Suppresses the "
        "early stop until the synthesis feedback loop has at least this much "
        "evaluation signal. The runtime-budget early stop is unaffected. "
        "Default: 15."
    )

    parser.add_argument(
        "--eval-seed",
        type=int,
        default=None,
        help="Optional RNG seed for reproducible dataset sampling during evaluation "
             "(default: None, i.e. unseeded/fresh random draw each run)"
    )

    # --- author model ----------------------------------------------------
    parser.add_argument(
        "--generation-model",
        type=str,
        default=None,
        help="Model identifier for CSD generation (OpenAI model id when using --generation-backend openai; "
        "Codex uses the fixed gpt-5.6-sol model; Claude Code uses the fixed claude-opus-5 model; Claude Bedrock uses an AWS model id. "
        "OpenAI defaults from OPENAI_GENERATION_MODEL or gpt-5.4.",
    )

    parser.add_argument(
        "--generation-backend",
        type=str,
        choices=GENERATION_BACKENDS,
        default="openai",
        help=(
            "Backend for strategy generation (default: openai). 'claude' uses an "
            "isolated Claude Code Max login (config/account from CSD_CLAUDE_* env); "
            "'codex' uses the pinned Pi provider-only ChatGPT OAuth route and fixed gpt-5.6-sol model; "
            "'claude-bedrock' uses AWS Bedrock; 'anthropic' uses the direct "
            "Anthropic API. API keys always come from the environment/.env (BYOD)."
        ),
    )

    parser.add_argument(
        "--synthesizer-reasoning-budget",
        type=int,
        default=SYNTHESIZER_REASONING_BUDGET_DEFAULT,
        help=(
            "Provider-agnostic extended-thinking budget in tokens for the "
            "author model (budget_tokens on Anthropic/Bedrock). Thinking is "
            f"always on. Default: {SYNTHESIZER_REASONING_BUDGET_DEFAULT}."
        ),
    )

    parser.add_argument(
        "--synthesis-max-tokens",
        "--max-tokens",
        dest="synthesis_max_tokens",
        type=int,
        default=8192,
        help="Maximum tokens for CSD synthesis generation per attempt (default: 8192)"
    )

    # --- pure re-eval / approved recovery inputs -------------------------
    parser.add_argument(
        "--initial-strategy-file",
        type=Path,
        default=None,
        help="Strategy body to use as the first attempt. With "
             "--max-iterations 1 this is a pure re-evaluation; with more "
             "iterations the run keeps improving on the seeded strategy.",
    )

    parser.add_argument(
        "--initial-attempt-offset",
        type=int,
        default=0,
        help="Attempt number offset for recovery runs seeded from an earlier synthesis attempt.",
    )

    parser.add_argument(
        "--initial-attempt-history-file",
        type=Path,
        default=None,
        help="JSON evaluated-attempt history to restore for an approved recovery run.",
    )

    # --- environment-shaped knobs ----------------------------------------
    parser.add_argument(
        "--dafny-path",
        type=str,
        default=default_dafny_path(),
        help="Path to Dafny executable"
    )

    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "mps", "cpu", "auto"],
        default="auto",
        help="Device for model inference (default: auto)"
    )

    parser.add_argument(
        "--vllm-gpu-memory-utilization",
        type=float,
        default=None,
        help=(
            "GPU memory fraction reserved by vLLM for this run. Must match the "
            "scheduler's per-cell reservation when GPUs are shared "
            "(default: CSD_VLLM_GPU_MEMORY_UTILIZATION env, else "
            f"{VLLM_GPU_MEMORY_UTILIZATION})."
        ),
    )

    # --- SMILES-only -----------------------------------------------------
    parser.add_argument(
        "--smiles-samples-per-class",
        type=int,
        default=10,
        help="Number of SMILES attempts per molecular class during synthesis feedback (default: 10)"
    )

    parser.add_argument(
        "--smiles-final-samples-per-class",
        type=int,
        default=100,
        help="Target valid unique SMILES samples per class for final benchmark scripts (default: 100)"
    )

    parser.add_argument(
        "--smiles-classes",
        type=str,
        default="acrylates,chain_extenders,isocyanates",
        help="Comma-separated SMILES classes to evaluate (default: all three CARS molecule classes)"
    )

    args = parser.parse_args()
    args.generation_backend = normalize_generation_backend(args.generation_backend)
    if args.vllm_gpu_memory_utilization is None:
        args.vllm_gpu_memory_utilization = _resolve_vllm_gpu_memory_utilization(
            args.eval_model
        )

    # One canonical split per dataset; synthesis always evaluates the train side.
    split_file = SPLIT_FILE_BY_DATASET[args.dataset]
    split_name = SYNTHESIS_SPLIT_NAME if split_file is not None else None

    # Refuse to launch when the accuracy bar's split side doesn't match the
    # split being evaluated (or is undeclared). Fail here, before any model
    # loads, so a misconfigured run costs seconds, not GPU-hours.
    from synthesis.split_provenance import BarSplitProvenanceError, check_bar_split_provenance

    try:
        check_bar_split_provenance(
            dataset=args.dataset,
            split_file=str(split_file) if split_file is not None else None,
            split_name=split_name,
            min_accuracy=args.min_accuracy,
            bar_split_name="train",
        )
    except BarSplitProvenanceError as exc:
        parser.error(str(exc))

    if split_file is not None and not Path(split_file).is_file():
        raise RuntimeError(
            f"Canonical split file for {args.dataset} not found: {split_file}. "
            "See SPLIT_FILE_BY_DATASET in synthesis/run_constants.py."
        )

    if args.generation_model is None:
        if args.generation_backend == "claude":
            args.generation_model = "claude-opus-5"
        elif args.generation_backend == "codex":
            args.generation_model = "gpt-5.6-sol"
        elif args.generation_backend == "claude-bedrock":
            resolved = os.environ.get("BEDROCK_GENERATION_MODEL") or os.environ.get(
                "AWS_BEDROCK_GENERATION_MODEL"
            )
            if not resolved:
                parser.error(
                    "Bedrock synthesis requires --generation-model or "
                    "BEDROCK_GENERATION_MODEL / AWS_BEDROCK_GENERATION_MODEL in the environment."
                )
            args.generation_model = resolved
        elif args.generation_backend == "openai":
            args.generation_model = os.environ.get("OPENAI_GENERATION_MODEL") or "gpt-5.4"
        else:
            from synthesis.generate.generator import StrategyGenerator as _StrategyGenerator

            args.generation_model = _StrategyGenerator.DEFAULT_MODEL

    # Defense against the "small author model" foot-gun.
    # The author model (--generation-model) writes the Dafny strategy code and
    # must be a large reasoning model. A small open model (1.5B/7B/14B Qwen)
    # cannot hold enough context to author workable strategies, and synthesis
    # silently produces 0% accuracy runs. Raise here so the misconfiguration is
    # caught BEFORE any GPU work. Pure re-evaluation (--max-iterations 1 with
    # --initial-strategy-file) never calls the author, so the guard skips it.
    is_pure_reeval = args.max_iterations == 1 and args.initial_strategy_file is not None
    import re as _re_guard
    _SMALL_AUTHOR_RE = _re_guard.compile(r"\b\d+(?:\.\d+)?\s*[Bb]\b")
    _LOCAL_BACKENDS = {"vllm", "huggingface"}
    if (
        args.generation_backend in _LOCAL_BACKENDS
        and args.generation_model
        and _SMALL_AUTHOR_RE.search(args.generation_model)
        and not is_pure_reeval
    ):
        raise SystemExit(
            "\n[FATAL] Refusing to run: --generation-model="
            f"{args.generation_model!r} looks like a small open model and "
            f"--generation-backend={args.generation_backend!r} is local. The "
            "author model must be a large reasoning model (e.g. gpt-5.4 via "
            "--generation-backend openai). See CLAUDE.md "
            "'Model Configuration Verification'.\n"
        )
    if is_pure_reeval:
        print("[guard] pure re-eval mode (--max-iterations 1 + --initial-strategy-file): author never called")

    # Auto-derived; CSD_OUTPUT_NAME overrides for recovery resumes that must
    # keep writing under their original run name.
    output_name = os.environ.get("CSD_OUTPUT_NAME") or _derive_output_name(
        args.dataset, args.eval_model
    )

    # Prominent startup banner: identify author + eval models so any future
    # "wrong author model" misconfig is obvious in stdout/logs.
    _banner_width = 72
    print("=" * _banner_width)
    print(
        f"  AUTHOR MODEL : {args.generation_model!r} "
        f"via backend={args.generation_backend!r}"
    )
    print(
        f"  EVAL   MODEL : {args.eval_model!r} "
        f"via backend={EVAL_BACKEND!r}"
    )
    print(f"  OUTPUT NAME  : {output_name}")
    if split_file is not None:
        print(f"  SPLIT        : {split_name} side of {Path(split_file).name}")
    print("=" * _banner_width)

    # GSM-Symbolic examples come from the vendored CRANE JSON folder only
    # (HF rows only have numeric prose in question fields).
    if args.dataset == "gsm_symbolic" and not Path(GSM_SOURCE_DIR).is_dir():
        raise RuntimeError(
            f"GSM-Symbolic requires the vendored CRANE JSON folder at "
            f"{GSM_SOURCE_DIR} (see GSM_SOURCE_DIR in synthesis/run_constants.py)."
        )

    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    output_dir = Path(os.environ.get("CSD_OUTPUT_DIR", str(OUTPUT_DIR)))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Import here to avoid loading heavy dependencies if just showing help
    from synthesis.generate.generator import ClaudeTransientError, StrategyGenerator
    from synthesis.verify.tooling import build_default_compiler, build_default_verifier
    from synthesis.evaluate.evaluator import Evaluator
    from synthesis.evaluate.feedback_loop import SynthesisPipeline, SynthesisExhaustionError
    from synthesis.evaluate.benchmarks.registry import resolve_require_delimiters

    # Create components
    print("Initializing synthesis pipeline...")

    device = None if args.device == "auto" else args.device
    vllm_gpu_memory_utilization = args.vllm_gpu_memory_utilization

    generator = StrategyGenerator(
        model_name=args.generation_model,
        backend=args.generation_backend,
        device=device,
        max_new_tokens=args.synthesis_max_tokens,
        temperature=TEMPERATURE,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_max_model_len=VLLM_MAX_MODEL_LEN,
        reasoning_budget_tokens=args.synthesizer_reasoning_budget,
    )

    verifier = build_default_verifier(dafny_path=args.dafny_path)
    # Compiler output dir is set per-run inside the pipeline (so runs don't overwrite each other).
    compiler = build_default_compiler(dafny_path=args.dafny_path, output_dir=output_dir)
    # Runner is created by the pipeline with task-appropriate parser mode

    feedback_sample_size = (
        args.smiles_samples_per_class if args.dataset == "smiles" else args.eval_sample_size
    )

    # Create evaluator for the feedback loop
    print(f"Setting up evaluator for dataset: {args.dataset}")
    print(f"  Generation model: {args.generation_model}")
    print(f"  Evaluation model: {args.eval_model}")
    print(f"  vLLM gpu_memory_utilization: {vllm_gpu_memory_utilization}")
    evaluator = Evaluator(
        dataset_name=args.dataset,
        model_name=args.eval_model,
        backend=EVAL_BACKEND,
        device=device or "cuda",
        sample_size=feedback_sample_size,
        max_steps=args.eval_max_steps,
        step_token_budget=args.eval_step_token_budget,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_max_model_len=VLLM_MAX_MODEL_LEN,
        sample_seed=args.eval_seed,
        max_seconds_per_example=args.eval_max_seconds_per_example,
        gsm_source_dir=str(GSM_SOURCE_DIR) if args.dataset == "gsm_symbolic" else None,
        gsm_split_file=str(split_file) if args.dataset == "gsm_symbolic" else None,
        gsm_split_name=split_name if args.dataset == "gsm_symbolic" else "train",
        spider_split_file=str(split_file) if args.dataset == "spider" else None,
        spider_split_name=split_name if args.dataset == "spider" else "train",
        smiles_classes=evaluator_smiles_classes(args.dataset, args.smiles_classes),
        early_stop_on_answer=EVAL_EARLY_STOP_ON_ANSWER,
    )

    pipeline = SynthesisPipeline(
        evaluator=evaluator,
        generator=generator,
        verifier=verifier,
        compiler=compiler,
        max_iterations=args.max_iterations,
        output_dir=output_dir,
        min_accuracy=args.min_accuracy,
        min_syntax_rate=args.min_syntax_rate,
        bar_split_name="train",
        require_delimiters=resolve_require_delimiters(args.dataset, cli_value=True),
        eval_sample_size=feedback_sample_size,
        eval_max_seconds_per_example=args.eval_max_seconds_per_example,
        max_attempt_seconds=args.max_attempt_seconds,
        min_examples_before_threshold_stop=args.eval_min_examples_before_threshold_stop,
        adaptive_helper_mask=args.adaptive_helper_mask,
        helper_selection_policy=args.helper_selection_policy,
        refinement_beam_size=args.refinement_beam_size,
    )

    initial_strategy_code = None
    if args.initial_strategy_file:
        initial_strategy_code = args.initial_strategy_file.read_text()
        print(f"Loaded initial strategy seed from: {args.initial_strategy_file}")
    initial_attempts = []
    if args.initial_attempt_history_file:
        initial_attempts = _load_initial_attempt_history(args.initial_attempt_history_file)
        print(
            f"Loaded {len(initial_attempts)} prior evaluated attempt(s) from: "
            f"{args.initial_attempt_history_file}"
        )

    # Run synthesis
    try:
        result = pipeline.synthesize(
            task_description=args.task,
            output_name=output_name,
            initial_strategy_code=initial_strategy_code,
            initial_attempt_offset=args.initial_attempt_offset,
            initial_attempts=initial_attempts,
        )

        print("\n" + "=" * 60)
        print("SYNTHESIS COMPLETE")
        print("=" * 60)
        print(display_text("Strategy", result.strategy_code))
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

    except ClaudeTransientError as e:
        print(f"\nTemporary Claude provider failure: {e}")
        sys.exit(76)

    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
