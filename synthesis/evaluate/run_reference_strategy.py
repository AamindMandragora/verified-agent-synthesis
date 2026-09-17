"""
Compile a verified reference Dafny strategy and evaluate it on any benchmark.

This replaces legacy external codebases with our own verified Dafny
reference implementations, ensuring consistent evaluation infrastructure
across all strategy-benchmark combinations.

Available strategies:
  unconstrained  Pure unconstrained decoding (no grammar enforcement)
  gcd            Pure hard-mask constrained decoding (SynCode-style)
  crane          Adaptive constrained-unconstrained switching (CRANE-style)
  itergen        Chunked constrained symbol generation (IterGen-style)
  cars           Adaptive group-boosted constrained steps (CARS-style)

Usage:
  python -m synthesis.evaluate.run_reference_strategy \\
    --strategy crane --dataset smiles \\
    --eval-model Qwen/Qwen2.5-Coder-7B-Instruct \\
    --eval-sample-size 50 --eval-max-steps 900 \\
    --output-json outputs/baselines/crane/smiles.json
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

from synthesis.evaluate.baseline_store import build_metrics_from_eval_samples
from synthesis.generate.generator import inject_strategy_into_template
from synthesis.evaluate.evaluator import Evaluator


STRATEGY_DFY: dict[str, str] = {
    "unconstrained": "unconstrained.dfy",
    "gcd": "gcd.dfy",
    "crane": "crane.dfy",
    "crane_faithful": "crane_faithful.dfy",
    "itergen": "itergen.dfy",
    "cars": "cars.dfy",
}

REFERENCE_DIR = Path(__file__).resolve().parents[1] / "verify" / "reference"


def _compile_reference(strategy: str, output_dir: Path) -> Path:
    """Compile a reference .dfy to Python under output_dir, return GeneratedCSD.py path."""
    from synthesis.verify.tooling import build_default_compiler

    dfy_name = STRATEGY_DFY[strategy]
    source_path = REFERENCE_DIR / dfy_name
    if not source_path.is_file():
        raise FileNotFoundError(f"Reference strategy not found: {source_path}")

    # Each reference is a BODY only. The single copy of the decoder contract lives
    # in synthesis/verify/library/GeneratedCSD.dfy, and the same function the
    # synthesis pipeline uses builds the full file from it.
    source_text = inject_strategy_into_template(source_path.read_text())

    compiler = build_default_compiler(output_dir=output_dir)
    result = compiler.compile(source_text, output_name=f"ref_{strategy}")
    if not result.success:
        errors = "; ".join(e.message for e in result.errors[:5])
        raise RuntimeError(f"Failed to compile {dfy_name}: {errors}")

    if result.main_module_path is None:
        raise RuntimeError(f"Compilation produced no main module for {dfy_name}")

    return result.main_module_path


def _resolve_device(device: str, backend: str) -> str:
    """Turn the "auto" default into a device the vLLM backend accepts."""
    if device == "auto" and backend == "vllm":
        return "cuda"
    return device


def _evaluate(
    compiled_module: Path,
    dataset: str,
    eval_model: str,
    eval_backend: str,
    device: str,
    sample_size: int,
    max_steps: int,
    step_token_budget: int,
    vllm_gpu_memory_utilization: float,
    vllm_max_model_len: int | None,
    smiles_classes: str | None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = dict(
        dataset_name=dataset,
        model_name=eval_model,
        backend=eval_backend,
        device=device,
        sample_size=sample_size,
        max_steps=max_steps,
        step_token_budget=step_token_budget,
        vllm_gpu_memory_utilization=vllm_gpu_memory_utilization,
    )
    if vllm_max_model_len is not None:
        kwargs["vllm_max_model_len"] = vllm_max_model_len
    if smiles_classes is not None:
        kwargs["smiles_classes"] = smiles_classes

    evaluator = Evaluator(**kwargs)
    try:
        result = evaluator.evaluate_sample(compiled_module, sample_size=sample_size)
    finally:
        evaluator.unload_runtime()

    return _payload_from_result(result)


def _payload_from_result(result: Any) -> dict[str, Any]:
    """Turn an evaluation result into the payload written to --output-json.

    Stops instead of reporting when the evaluation did not actually measure
    anything. Both checks matter and they are not the same check: `success` is
    False when the evaluator raised (a missing module, an ungradable row), and
    `num_examples == 0` catches an evaluator that reported success while
    running nothing, which is the shape Spider's fake 0% took.

    Without this, a broken harness came out of here as `accuracy: 0.0` in a
    saved results file and a printed "accuracy=0.000", with the underlying
    error dropped on the floor -- indistinguishable from a model that answered
    every question wrong. Same check as scripts/reevaluate_compiled_csd.py:94-97.
    A genuine 0% over real examples is a measurement and still reports normally.
    """
    if not getattr(result, "success", False):
        raise SystemExit(f"Evaluation failed: {getattr(result, 'error', None)}")
    if not result.num_examples:
        raise SystemExit(
            "Evaluation failed: zero examples were evaluated, so the accuracy "
            "below would not be a measurement. Underlying error: "
            f"{getattr(result, 'error', None)}"
        )

    answers: list[dict[str, Any]] = []
    for sample in result.sample_outputs or []:
        question = str(sample.get("question", ""))
        generated = sample.get("actual")
        if generated is None:
            generated = sample.get("full_output", "")
        row: dict[str, Any] = {"question": question, "generated_answer": str(generated)}
        if sample.get("token_count") is not None:
            row["num_tokens"] = int(sample["token_count"])
        if sample.get("time_seconds") is not None:
            row["generation_seconds"] = round(float(sample["time_seconds"]), 6)
        answers.append(row)

    metrics = build_metrics_from_eval_samples(
        result.sample_outputs or [],
        evaluator_total_time_seconds=result.total_time_seconds,
        evaluator_max_sample_time_seconds=result.max_sample_time_seconds,
    )

    return {
        "accuracy": float(result.accuracy),
        "syntax_rate": float(result.syntax_rate),
        "metrics": metrics,
        "num_correct": result.num_correct,
        "num_examples": result.num_examples,
        "contains_delimiters": result.contains_delimiters,
        "answers": answers,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strategy",
        required=True,
        choices=sorted(STRATEGY_DFY),
        help="Reference strategy to evaluate",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["gsm_symbolic", "spider", "smiles"],
    )
    parser.add_argument("--eval-model", required=True)
    parser.add_argument("--eval-backend", default="vllm")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--eval-sample-size", type=int, default=50)
    parser.add_argument("--eval-max-steps", type=int, default=900)
    parser.add_argument("--eval-step-token-budget", type=int, default=1)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--vllm-max-model-len", type=int, default=None)
    parser.add_argument("--smiles-classes", default=None)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--compiled-cache-dir",
        type=Path,
        default=None,
        help="Directory to cache compiled reference modules (avoids recompilation)",
    )
    args = parser.parse_args()

    cache_dir = args.compiled_cache_dir
    if cache_dir is None:
        repo_root = Path(__file__).resolve().parents[2]
        cache_dir = repo_root / "outputs" / "compiled_references"
    cache_dir.mkdir(parents=True, exist_ok=True)

    strategy_cache = cache_dir / args.strategy
    generated_csd = strategy_cache / f"ref_{args.strategy}" / "GeneratedCSD.py"
    if not generated_csd.is_file():
        print(f"Compiling reference strategy: {args.strategy}", flush=True)
        compiled_path = _compile_reference(args.strategy, strategy_cache)
        print(f"  Compiled to: {compiled_path}", flush=True)
    else:
        compiled_path = generated_csd
        print(f"Using cached compilation: {compiled_path}", flush=True)

    print(
        f"Evaluating {args.strategy} on {args.dataset} with {args.eval_model} "
        f"(n={args.eval_sample_size}, steps={args.eval_max_steps})",
        flush=True,
    )
    payload = _evaluate(
        compiled_module=compiled_path,
        dataset=args.dataset,
        eval_model=args.eval_model,
        eval_backend=args.eval_backend,
        device=_resolve_device(args.device, args.eval_backend),
        sample_size=args.eval_sample_size,
        max_steps=args.eval_max_steps,
        step_token_budget=args.eval_step_token_budget,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_max_model_len=args.vllm_max_model_len,
        smiles_classes=args.smiles_classes,
    )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"Wrote {args.output_json}: accuracy={payload['accuracy']:.3f} "
        f"syntax_rate={payload['syntax_rate']:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
