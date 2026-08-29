#!/usr/bin/env python3
"""Evaluate an already compiled GeneratedCSD.py path and optionally export JSON."""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
from typing import Any

from synthesis.evaluate.baseline_store import save_minimal_baseline_json
from synthesis.evaluate.evaluator import Evaluator
from synthesis.run_constants import EVAL_EARLY_STOP_ON_ANSWER, SPLIT_FILE_BY_DATASET


def _evaluated_source_indices(
    dataset: str, evaluation_result: Any
) -> list[int]:
    """Read and validate every source-index alias from returned evaluator rows."""
    dataset_alias = {
        "spider": "spider_source_index",
        "gsm_symbolic": "crane_source_index",
    }.get(dataset)
    aliases = ("source_index", dataset_alias) if dataset_alias else ("source_index",)
    samples = list(getattr(evaluation_result, "sample_outputs", ()) or ())
    indices: list[int] = []
    for evaluated_index, sample in enumerate(samples):
        present: list[tuple[str, int]] = []
        for key in aliases:
            if key is None or key not in sample:
                continue
            value = sample[key]
            if value is None:
                continue
            if type(value) is not int:
                raise ValueError(
                    f"reevaluation result row {evaluated_index} source alias "
                    f"{key} must be a strict integer"
                )
            present.append((key, value))
        if not present:
            raise ValueError(
                "reevaluation result row "
                f"{evaluated_index} has no resolved source index"
            )
        resolved = present[0][1]
        if any(value != resolved for _, value in present[1:]):
            raise ValueError(
                f"reevaluation result row {evaluated_index} source index aliases disagree"
            )
        indices.append(resolved)
    return indices


def build_reevaluation_provenance(
    args: argparse.Namespace,
    compiled: Path,
    *,
    evaluation_result: Any = None,
) -> dict[str, Any]:
    provenance = {
        "cell_id": args.provenance_cell_id,
        "manifest_commit": args.provenance_manifest_commit,
        "dataset": args.dataset,
        "eval_model": args.eval_model,
        "compiled_csd_path": str(compiled.resolve()),
        "compiled_csd_sha256": hashlib.sha256(compiled.read_bytes()).hexdigest(),
        "sample_size": args.sample_size,
        "max_steps": args.max_steps,
        "step_token_budget": args.step_token_budget,
        "smiles_class": args.smiles_classes,
    }
    revision = getattr(args, "provenance_eval_model_revision", None)
    snapshot_path = getattr(args, "provenance_eval_model_snapshot_path", None)
    snapshot_sha256 = getattr(
        args, "provenance_eval_model_snapshot_sha256", None
    )
    snapshot_file_count = getattr(
        args, "provenance_eval_model_snapshot_file_count", None
    )
    if revision or snapshot_path:
        provenance.update(
            {
                "eval_model_revision": revision,
                "eval_model_snapshot_path": snapshot_path,
                "eval_model_snapshot_sha256": snapshot_sha256,
                "eval_model_snapshot_file_count": snapshot_file_count,
            }
        )
    spider_data_sha256 = getattr(args, "provenance_spider_data_sha256", None)
    if spider_data_sha256:
        provenance.update(
            {
                "spider_data_path": getattr(
                    args, "provenance_spider_data_path", None
                ),
                "spider_data_sha256": spider_data_sha256,
                "spider_data_file_count": getattr(
                    args, "provenance_spider_data_file_count", None
                ),
            }
        )
    if evaluation_result is not None:
        provenance.update(
            {
                "sample_offset": int(getattr(args, "sample_offset", 0)),
                "evaluated_source_indices": _evaluated_source_indices(
                    args.dataset, evaluation_result
                ),
            }
        )
        if args.dataset == "spider":
            resolved_split_file = getattr(args, "spider_split_file", None)
            if resolved_split_file is None:
                resolved_split_file = SPLIT_FILE_BY_DATASET["spider"]
            provenance.update(
                {
                    "spider_split_file": str(resolved_split_file),
                    "spider_split_name": getattr(args, "spider_split_name", None),
                }
            )
        elif args.dataset == "gsm_symbolic":
            resolved_split_file = getattr(args, "gsm_split_file", None)
            if resolved_split_file is None:
                resolved_split_file = SPLIT_FILE_BY_DATASET["gsm_symbolic"]
            provenance.update(
                {
                    "gsm_split_file": str(resolved_split_file),
                    "gsm_split_name": getattr(args, "gsm_split_name", None),
                }
            )
    return provenance


def babysitter_smoke_split_fallback(output_json: Path | None) -> str | None:
    """Return 'train' only for zero-acc babysitter smoke invocations.

    Bootstrap for incident spider-qwen25-1p5b:2:telemetry:1784857356: the live
    babysitter watcher builds this command without a split-name flag, and its
    smoke.py fix cannot take effect until a smoke passes and merges. The smoke
    is a train-side sanity gate by contract (test is reserved for final
    numbers — scripts/runtime/zero_acc_babysitter/AGENTS.md), and it is
    identified by its babysitter-owned report path. Every other invocation
    still hard-fails without an explicit split side.
    """
    if (
        output_json is not None
        and output_json.name == "smoke_report.json"
        and "zero_acc_babysitter" in output_json.as_posix()
    ):
        return "train"
    return None


def _resolve_device(device: str, backend: str) -> str:
    if device == "auto" and backend == "vllm":
        return "cuda"
    return device


def main() -> None:
    _seed = os.environ.get("CSD_PARITY_SEED", "").strip()
    if _seed:
        import random as _random
        _random.seed(int(_seed))
        try:
            import numpy as _np
            _np.random.seed(int(_seed))
        except Exception:
            pass
        try:
            import torch as _torch
            _torch.manual_seed(int(_seed))
            if _torch.cuda.is_available():
                _torch.cuda.manual_seed_all(int(_seed))
        except Exception:
            pass
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "compiled_generated_csd",
        type=Path,
        help="Path to GeneratedCSD.py inside the compiled output folder",
    )
    p.add_argument("--dataset", default="gsm_symbolic")
    p.add_argument("--eval-model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    p.add_argument("--eval-backend", default="vllm")
    p.add_argument("--device", default="cuda")
    p.add_argument("--sample-size", type=int, default=15)
    p.add_argument("--sample-offset", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=900)
    p.add_argument("--step-token-budget", type=int, default=1)
    p.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.8)
    p.add_argument(
        "--vllm-tensor-parallel-size",
        type=int,
        default=None,
        help="vLLM tensor parallel size (default: 1; capped by VAS_MAX_CUDA_DEVICES)",
    )
    p.add_argument("--vllm-max-model-len", type=int, default=16384)
    # Split FILE defaults to the one canonical split per dataset (see
    # SPLIT_FILE_BY_DATASET in synthesis/run_constants.py), but stays an
    # optional override: synthesis/scripts/sharded_eval_core.py invokes this
    # script once per parallel worker, each pointed at its own temporary
    # shard file (a slice of the canonical split) rather than the whole
    # thing, so the flag can't be removed without breaking sharded re-eval.
    # Split NAME (which side of that file) stays a required flag — this
    # script is used to score both the train side (sanity re-checks) and the
    # test/held-out side (final numbers), so silently defaulting to one side
    # is how the 2026-07-17 bar/eval split mixup happened. Spider's held-out
    # side is 'test'; the 'eval' alias was removed.
    p.add_argument("--gsm-split-file", type=Path, default=None)
    p.add_argument("--spider-split-file", type=Path, default=None)
    p.add_argument("--gsm-split-name", choices=["train", "test"], default=None)
    p.add_argument("--spider-split-name", choices=["train", "test"], default=None)
    p.add_argument("--smiles-classes", type=str, default=None)
    p.add_argument("--max-seconds-per-example", type=float, default=None)
    p.add_argument(
        "--completion-mode",
        action="store_true",
        help="Feed the prompt as a raw continuation with NO chat template "
        "(required for base / non-instruction-tuned completion models).",
    )
    p.add_argument("--output-json", type=Path, default=None)
    p.add_argument("--provenance-cell-id")
    p.add_argument("--provenance-manifest-commit")
    p.add_argument("--provenance-eval-model-revision")
    p.add_argument("--provenance-eval-model-snapshot-path")
    p.add_argument("--provenance-eval-model-snapshot-sha256")
    p.add_argument("--provenance-eval-model-snapshot-file-count", type=int)
    p.add_argument("--provenance-spider-data-path")
    p.add_argument("--provenance-spider-data-sha256")
    p.add_argument("--provenance-spider-data-file-count", type=int)
    args = p.parse_args()
    if bool(args.provenance_cell_id) != bool(args.provenance_manifest_commit):
        p.error(
            "--provenance-cell-id and --provenance-manifest-commit must be given together"
        )
    if args.provenance_cell_id and (
        not args.provenance_eval_model_revision
        or not args.provenance_eval_model_snapshot_path
        or not args.provenance_eval_model_snapshot_sha256
        or args.provenance_eval_model_snapshot_file_count is None
    ):
        p.error(
            "provenanced runs require the exact eval model revision and snapshot bytes"
        )
    if args.dataset == "spider" and args.provenance_cell_id and (
        not args.provenance_spider_data_path
        or not args.provenance_spider_data_sha256
        or args.provenance_spider_data_file_count is None
    ):
        p.error("provenanced Spider runs require the exact Spider data binding")

    gsm_split_file = args.gsm_split_file or SPLIT_FILE_BY_DATASET["gsm_symbolic"]
    spider_split_file = args.spider_split_file or SPLIT_FILE_BY_DATASET["spider"]
    # Bind the resolved canonical path onto args so provenance and evaluator
    # output cannot diverge when the CLI omitted an override.
    args.gsm_split_file = gsm_split_file
    args.spider_split_file = spider_split_file

    if args.dataset == "gsm_symbolic" and args.gsm_split_name is None:
        args.gsm_split_name = babysitter_smoke_split_fallback(args.output_json)
        if args.gsm_split_name is None:
            p.error("--gsm-split-name is required for --dataset gsm_symbolic "
                    "(no default side — see synthesis/split_provenance.py)")
        print("[reevaluate] --gsm-split-name defaulted to 'train' (babysitter smoke)")
    if args.dataset == "spider" and args.spider_split_name is None:
        args.spider_split_name = babysitter_smoke_split_fallback(args.output_json)
        if args.spider_split_name is None:
            p.error("--spider-split-name is required for --dataset spider "
                    "(no default side — see synthesis/split_provenance.py)")
        print("[reevaluate] --spider-split-name defaulted to 'train' (babysitter smoke)")

    if os.environ.get("VLLM_WORKER_MULTIPROC_METHOD") is None:
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    compiled = args.compiled_generated_csd.resolve()
    if not compiled.is_file():
        raise SystemExit(f"Not a file: {compiled}")

    from synthesis.evaluate.benchmarks.common.model_utils import resolve_vllm_tensor_parallel_size

    evaluator_kwargs: dict[str, Any] = dict(
        dataset_name=args.dataset,
        model_name=args.eval_model,
        backend=args.eval_backend,
        device=_resolve_device(args.device, args.eval_backend),
        sample_size=args.sample_size,
        max_steps=args.max_steps,
        step_token_budget=args.step_token_budget,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_tensor_parallel_size=resolve_vllm_tensor_parallel_size(args.vllm_tensor_parallel_size),
        vllm_max_model_len=args.vllm_max_model_len,
        vllm_enforce_eager=True,
        max_seconds_per_example=args.max_seconds_per_example,
        # Parity with the in-synthesis evaluator (run_synthesis.py): held-out
        # reeval must use the same answer-stopping semantics as training eval.
        early_stop_on_answer=EVAL_EARLY_STOP_ON_ANSWER,
    )
    if args.dataset == "gsm_symbolic":
        evaluator_kwargs["gsm_split_file"] = str(gsm_split_file)
        evaluator_kwargs["gsm_split_name"] = args.gsm_split_name
    if args.dataset == "spider":
        evaluator_kwargs["spider_split_file"] = str(spider_split_file)
        evaluator_kwargs["spider_split_name"] = args.spider_split_name
    if args.smiles_classes:
        evaluator_kwargs["smiles_classes"] = args.smiles_classes

    ev = Evaluator(**evaluator_kwargs)
    try:
        res = ev.evaluate_sample(
            compiled, sample_size=args.sample_size, sample_offset=args.sample_offset
        )
    finally:
        ev.unload_runtime()
    if not res.success:
        raise SystemExit(f"Evaluation failed: {res.error}")
    if res.num_examples == 0:
        raise SystemExit("Evaluation failed: zero examples were evaluated")

    print(f"accuracy: {res.accuracy:.4f}")
    print(f"syntax_rate: {res.syntax_rate:.4f}")
    print(f"num_correct: {res.num_correct} / {res.num_examples}")
    print(f"contains_delimiters: {res.contains_delimiters}")
    syn = sum(1 for s in res.sample_outputs if s.get("is_syntax_valid"))
    print(f"per_example_syntax_pass: {syn} / {len(res.sample_outputs)}")
    if args.output_json is not None:
        # Embed which split file/side this re-eval ran on, so the JSON is
        # self-describing (split-mismatch incident 2026-07-17).
        save_minimal_baseline_json(
            res,
            args.output_json,
            eval_split=ev.split_provenance(),
            metadata=(
                {
                    "reevaluation_provenance": build_reevaluation_provenance(
                        args, compiled, evaluation_result=res
                    )
                }
                if args.provenance_cell_id
                else None
            ),
        )
        print(f"wrote_json: {args.output_json}")


if __name__ == "__main__":
    main()
