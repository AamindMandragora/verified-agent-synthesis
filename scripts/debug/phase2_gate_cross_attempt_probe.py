#!/usr/bin/env python3
"""Cross-attempt CSD reload then winner gate probe (debug session de704a). Hypothesis H."""
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

BASE = Path(
    "outputs/generated/coldq_fullbaseline_20260803_gsm-qwen25-1p5b/"
    "coldq_fullbaseline_20260803_gsm-qwen25-1p5b_20260804_060028_e0a771/python"
)
WINNER = (BASE / "coldq_fullbaseline_20260803_gsm-qwen25-1p5b_20260804_164344_97680c/GeneratedCSD.py").resolve()
SPLIT = "environment/benchmark_splits/gsm_symbolic_crane_proportional_49x49_seed123.json"
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
N_PRIOR = int(os.environ.get("CSD_CROSS_PRIOR", "8"))
N_WINNER = int(os.environ.get("CSD_CROSS_WINNER", "8"))


def _log(message: str, data: dict, hypothesis_id: str = "H") -> None:
    payload = {
        "sessionId": "de704a",
        "runId": os.environ.get("CSD_DEBUG_RUN_ID", "cross-attempt"),
        "hypothesisId": hypothesis_id,
        "location": "phase2_gate_cross_attempt_probe.py",
        "message": message,
        "timestamp": int(time.time() * 1000),
        "data": data,
    }
    with open(os.environ["CSD_DEBUG_LOG"], "a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")
    print(json.dumps(payload), flush=True)


def _prior_csds(n: int) -> list[Path]:
    dirs = sorted(
        [d for d in BASE.iterdir() if d.is_dir() and (d / "GeneratedCSD.py").exists()],
        key=lambda p: p.stat().st_mtime,
    )
    # exclude winner; take earliest n (most like synthesis history before attempt 38)
    dirs = [d for d in dirs if d.name != WINNER.parent.name]
    chosen = dirs[:n]
    return [(d / "GeneratedCSD.py").resolve() for d in chosen]


def main() -> None:
    os.environ["CSD_DEBUG_RUN_ID"] = f"cross-prior{N_PRIOR}-win{N_WINNER}"
    priors = _prior_csds(N_PRIOR)
    _log(
        "starting cross-attempt probe",
        {
            "n_prior": len(priors),
            "prior_names": [p.parent.name[-20:] for p in priors],
            "n_winner": N_WINNER,
            "winner": str(WINNER),
            "pid": os.getpid(),
        },
    )
    ev = Evaluator(
        dataset_name="gsm_symbolic",
        model_name=MODEL,
        backend="vllm",
        device="cuda",
        sample_size=1,
        max_steps=900,
        step_token_budget=1,
        vllm_gpu_memory_utilization=0.3,
        vllm_tensor_parallel_size=1,
        vllm_max_model_len=16384,
        vllm_enforce_eager=True,
        gsm_split_file=SPLIT,
        gsm_split_name="train",
        max_seconds_per_example=120.0,
    )
    try:
        for i, csd in enumerate(priors):
            _log("prior eval begin", {"i": i, "csd_tail": csd.parent.name[-24:]})
            res = ev.evaluate_sample(csd, sample_size=1, sample_offset=i % 20)
            _log(
                "prior eval done",
                {
                    "i": i,
                    "success": res.success,
                    "error": res.error,
                    "n": res.num_examples,
                    "tok": (res.sample_outputs or [{}])[0].get("token_count") if res.sample_outputs else None,
                },
            )
        _log("winner eval begin", {"n": N_WINNER})
        res = ev.evaluate_sample(WINNER, sample_size=N_WINNER, sample_offset=0)
        opens = []
        for j, s in enumerate(res.sample_outputs or []):
            ht = s.get("helper_trace") or []
            names = [e.get("helper") for e in ht if isinstance(e, dict)]
            opened = any(n and "OpenConstrained" in n for n in names)
            opens.append(opened)
            _log(
                "winner example",
                {
                    "j": j,
                    "opened": opened,
                    "token_count": s.get("token_count"),
                    "is_syntax_valid": s.get("is_syntax_valid"),
                },
            )
        _log(
            "cross-attempt done",
            {
                "success": res.success,
                "error": res.error,
                "n": res.num_examples,
                "opened_count": sum(opens),
                "opened_flags": opens,
                "accuracy": res.accuracy,
                "syntax_rate": res.syntax_rate,
            },
        )
    finally:
        ev.unload_runtime()


if __name__ == "__main__":
    main()
