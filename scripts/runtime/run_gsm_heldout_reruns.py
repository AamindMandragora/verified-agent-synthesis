#!/usr/bin/env python3
"""Run the 3 GSM held-out reevals directly (bypassing the controller's phase
barrier), using the controller's own job plumbing so provenance matches.

Cells: gsm-qwen25-1p5b, gsm-qwen25-7b, gsm-qwen35-4b. Sequential on one GPU.
The reevaluator now passes early_stop_on_answer=True (commit 56302a1d).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path("/home/aadivyar/csd-generation-worktrees/full-baseline-campaign-20260803")
PY = Path("/apps/conda/aadivyar/envs/csd/bin/python")
CELLS = ("gsm-qwen25-1p5b", "gsm-qwen25-7b", "gsm-qwen35-4b")
GPU = os.environ.get("HELDOUT_GPU", "2")

sys.path.insert(0, str(REPO))
from scripts.runtime.run_cold_synthesis_queue import (  # noqa: E402
    compiled_csd,
    heldout_command,
    pinned_heldout_csd,
    stamp_job_commit_from_report,
)


def main() -> int:
    manifest = json.loads(
        (REPO / "saved-results/2026-08-05-corrected-full-baseline-cold-manifest.json").read_text()
    )
    jobs = {j["cell_id"]: j for j in manifest["jobs"]}
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = GPU
    env["CSD_EVAL_GPU_SLOTS"] = GPU
    env["PYTHONPATH"] = str(REPO)
    failures = 0
    for cell in CELLS:
        job = stamp_job_commit_from_report(dict(jobs[cell]), REPO, str(jobs[cell]["output_name"]))
        if job.get("run_mode") == "heldout_only":
            csd = pinned_heldout_csd(job, REPO)
        else:
            csd = compiled_csd(
                REPO,
                str(job["output_name"]),
                min_accuracy=float(job["min_accuracy"]),
                min_syntax_rate=float(job["min_syntax_rate"]),
                job=job,
            )
        if csd is None:
            print(f"[heldout-rerun] SKIP {cell}: no acceptable compiled CSD found", flush=True)
            failures += 1
            continue
        out = Path(str(job["heldout_output_json"]))
        if not out.is_absolute():
            out = REPO / out
        if out.is_file():
            print(f"[heldout-rerun] SKIP {cell}: output already exists {out}", flush=True)
            continue
        cmd = heldout_command(job, PY, csd)
        print(f"[heldout-rerun] START {cell} gpu={GPU}", flush=True)
        result = subprocess.run(cmd, cwd=REPO, env=env)
        print(f"[heldout-rerun] FINISH {cell} exit={result.returncode}", flush=True)
        if result.returncode != 0:
            failures += 1
    print(f"[heldout-rerun] ALL DONE failures={failures}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
