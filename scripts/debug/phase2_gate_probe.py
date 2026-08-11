#!/usr/bin/env python3
"""Cheap Phase-2 gate probe for gsm-qwen25-1p5b (debug session de704a)."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault(
    "CSD_DEBUG_LOG",
    "/home/aadivyar/csd-generation-worktrees/full-baseline-campaign-20260803/.cursor/debug-de704a.log",
)

from synthesis.evaluate.evaluator import Evaluator

CSD = Path(
    "outputs/generated/coldq_fullbaseline_20260803_gsm-qwen25-1p5b/"
    "coldq_fullbaseline_20260803_gsm-qwen25-1p5b_20260804_060028_e0a771/python/"
    "coldq_fullbaseline_20260803_gsm-qwen25-1p5b_20260804_164344_97680c/GeneratedCSD.py"
).resolve()
SPLIT = "environment/benchmark_splits/gsm_symbolic_crane_proportional_49x49_seed123.json"
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def _log(message: str, data: dict, hypothesis_id: str = "meta") -> None:
    payload = {
        "sessionId": "de704a",
        "runId": os.environ.get("CSD_DEBUG_RUN_ID", "probe"),
        "hypothesisId": hypothesis_id,
        "location": "phase2_gate_probe.py",
        "message": message,
        "timestamp": int(time.time() * 1000),
        "data": data,
    }
    with open(os.environ["CSD_DEBUG_LOG"], "a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")
    print(json.dumps(payload), flush=True)


def run_one(split_name: str, sample_size: int, sample_offset: int, label: str) -> None:
    os.environ["CSD_DEBUG_RUN_ID"] = label
    _log(
        "starting eval slice",
        {
            "label": label,
            "split": split_name,
            "sample_size": sample_size,
            "sample_offset": sample_offset,
            "pid": os.getpid(),
            "csd": str(CSD),
        },
        "D",
    )
    ev = Evaluator(
        dataset_name="gsm_symbolic",
        model_name=MODEL,
        backend="vllm",
        device="cuda",
        sample_size=sample_size,
        max_steps=900,
        step_token_budget=1,
        vllm_gpu_memory_utilization=0.3,
        vllm_tensor_parallel_size=1,
        vllm_max_model_len=16384,
        vllm_enforce_eager=True,
        gsm_split_file=SPLIT,
        gsm_split_name=split_name,
        max_seconds_per_example=120.0,
    )
    try:
        res = ev.evaluate_sample(
            CSD,
            sample_size=sample_size,
            sample_offset=sample_offset,
        )
    finally:
        ev.unload_runtime()

    opens = []
    for i, s in enumerate(res.sample_outputs or []):
        ht = s.get("helper_trace") or []
        names = [e.get("helper") for e in ht if isinstance(e, dict)]
        opened = any(n and "OpenConstrained" in n for n in names)
        opens.append(opened)
        _log(
            "example result",
            {
                "label": label,
                "i": i,
                "opened": opened,
                "token_count": s.get("token_count"),
                "is_correct": s.get("is_correct"),
                "is_syntax_valid": s.get("is_syntax_valid"),
                "tail": (s.get("full_output") or s.get("scored_output") or "")[-80:],
                "helper_tail": names[-8:],
            },
            "A,D,E",
        )
    _log(
        "slice done",
        {
            "label": label,
            "success": res.success,
            "accuracy": res.accuracy,
            "syntax_rate": res.syntax_rate,
            "n": res.num_examples,
            "opened_flags": opens,
        },
        "meta",
    )


def main() -> None:
    # Fresh process slices: train skips, train known-open row, held-out
    run_one("train", 2, 0, "fresh-train-offset0")
    run_one("train", 1, 21, "fresh-train-offset21")
    run_one("test", 2, 0, "fresh-test-offset0")


if __name__ == "__main__":
    main()
