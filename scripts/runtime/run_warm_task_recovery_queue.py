#!/usr/bin/env python3
"""Greedily run an explicitly approved warm-recovery manifest on focal GPUs."""

from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import json
import logging
import math
import os
import subprocess
import time
from pathlib import Path
from typing import Any


GPU_SAFETY_MIB = 2_000
TRANSIENT_PROVIDER_EXIT = 76
REQUIRED_FIELDS = {
    "cell_id",
    "last_clean_attempt",
    "total_cap",
    "memory_reservation_mib",
    "source_log",
    "history_file",
    "output_name",
    "log_file",
    "dataset",
    "eval_model",
    "gpu_mem_util",
    "heldout_sample_size",
    "eval_max_steps",
    "eval_max_seconds",
    "heldout_split_name",
    "heldout_output_json",
}

logger = logging.getLogger("warm-task-recovery")


class ConfigError(ValueError):
    pass


def load_manifest(path: Path) -> list[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"invalid manifest {path}: {exc}") from exc
    if not isinstance(payload, list) or not payload:
        raise ConfigError("manifest must be a non-empty JSON list")

    jobs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in payload:
        if not isinstance(raw, dict):
            raise ConfigError("every manifest entry must be an object")
        missing = sorted(REQUIRED_FIELDS - raw.keys())
        if missing:
            raise ConfigError(f"manifest entry is missing {missing}")
        cell_id = str(raw["cell_id"]).strip()
        if not cell_id:
            raise ConfigError("cell_id must not be empty")
        if cell_id in seen:
            raise ConfigError(f"duplicate cell_id: {cell_id}")
        seen.add(cell_id)
        try:
            last_clean = int(raw["last_clean_attempt"])
            total_cap = int(raw["total_cap"])
            reservation = int(raw["memory_reservation_mib"])
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"invalid numeric field for {cell_id}") from exc
        if last_clean < 1 or last_clean >= total_cap:
            raise ConfigError(
                f"last_clean_attempt must be between 1 and total_cap-1 for {cell_id}"
            )
        if total_cap > 80:
            raise ConfigError(f"total_cap exceeds approved recovery ceiling for {cell_id}")
        if reservation <= 0:
            raise ConfigError(f"memory_reservation_mib must be positive for {cell_id}")
        source_log = Path(str(raw["source_log"]))
        history_file = Path(str(raw["history_file"]))
        if not source_log.is_file():
            raise ConfigError(f"source_log missing for {cell_id}: {source_log}")
        if not history_file.is_file():
            raise ConfigError(f"history_file missing for {cell_id}: {history_file}")
        try:
            history = json.loads(history_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ConfigError(f"invalid history_file for {cell_id}: {exc}") from exc
        if not isinstance(history, list):
            raise ConfigError(f"history_file must contain a list for {cell_id}")
        history_attempts = [int(item["attempt_number"]) for item in history]
        if history_attempts and max(history_attempts) >= last_clean:
            raise ConfigError(
                f"history_file for {cell_id} includes replay attempt or later"
            )
        jobs.append({**raw, "cell_id": cell_id})
    return jobs


def gpu_memory_snapshot(nvidia_smi: str) -> dict[int, dict[str, int]]:
    """Return per-GPU used/total/free MiB from nvidia-smi.

    Callers: choose_gpu / choose_gpu_bundle in this file and
    scripts/runtime/run_cold_synthesis_queue.py (imports this helper).
    """
    # Gate facts: existing file pulled from focal; not a new caller surface.
    # Data: nvidia-smi csv ints index,used,total,free (MiB). No dates.
    # User: "what's causing this error on focal for data collection rn? pls fix"
    result = subprocess.run(
        [
            nvidia_smi,
            "--query-gpu=index,memory.used,memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise ConfigError(
            f"GPU status failed: {(result.stderr or result.stdout).strip()}"
        )
    snapshots: dict[int, dict[str, int]] = {}
    try:
        for line in result.stdout.strip().splitlines():
            parts = [int(part.strip()) for part in line.split(",")]
            if len(parts) == 4:
                gpu, used, total, free = parts
            elif len(parts) == 3:
                gpu, used, total = parts
                free = max(0, total - used)
            else:
                raise ValueError(f"unexpected nvidia-smi row: {line!r}")
            snapshots[gpu] = {
                "used_mib": used,
                "total_mib": total,
                "free_mib": free,
            }
    except ValueError as exc:
        raise ConfigError(f"invalid GPU snapshot: {result.stdout!r}") from exc
    if not snapshots:
        raise ConfigError("GPU status returned no devices")
    return snapshots


def choose_gpu(
    job: dict[str, Any],
    snapshots: dict[int, dict[str, int]],
    reservations: dict[int, dict[str, int]],
    baseline_snapshots: dict[int, dict[str, int]],
) -> int | None:
    candidates: list[tuple[int, int]] = []
    for gpu, snapshot in snapshots.items():
        required = required_memory_mib(job, snapshot["total_mib"])
        reserved = sum(reservations.get(gpu, {}).values())
        baseline_used = baseline_snapshots.get(gpu, snapshot)["used_mib"]
        projected_used = max(snapshot["used_mib"], baseline_used + reserved)
        # vLLM requires free >= gpu_memory_utilization * total_memory.
        free_mib = int(
            snapshot.get("free_mib", snapshot["total_mib"] - snapshot["used_mib"])
        )
        projected_free = snapshot["total_mib"] - projected_used
        if free_mib < required + GPU_SAFETY_MIB:
            continue
        if projected_free < required + GPU_SAFETY_MIB:
            continue
        if projected_used + required <= snapshot["total_mib"] - GPU_SAFETY_MIB:
            candidates.append((projected_used, gpu))
    return min(candidates)[1] if candidates else None


def required_memory_mib(job: dict[str, Any], gpu_total_mib: int) -> int:
    """Reserve at least the fraction of physical memory requested from vLLM."""
    configured = int(job["memory_reservation_mib"])
    vllm_required = math.ceil(float(job["gpu_mem_util"]) * gpu_total_mib)
    return max(configured, vllm_required)


def worker_environment(
    job: dict[str, Any], assigned_gpu: int, inherited: dict[str, str]
) -> dict[str, str]:
    env = dict(inherited)
    env.update(
        {
            "RESUME_LAST_ATTEMPT": str(job["last_clean_attempt"]),
            "RESUME_TOTAL_CAP": str(job["total_cap"]),
            "RESUME_GPU": str(assigned_gpu),
            "RESUME_OUTPUT_NAME": str(job["output_name"]),
            "RESUME_SOURCE_LOG": str(job["source_log"]),
            "RESUME_HISTORY_FILE": str(job["history_file"]),
            "RESUME_LOG_FILE": str(job["log_file"]),
        }
    )
    if job.get("seed_file"):
        env["RESUME_SEED_FILE"] = str(job["seed_file"])
    if job.get("launcher_manifest"):
        env["CLAUDE_RECOVERY_MANIFEST"] = str(job["launcher_manifest"])
    if job.get("checkpoint_root"):
        env["CLAUDE_RECOVERY_CHECKPOINT_ROOT"] = str(job["checkpoint_root"])
    return env


def compiled_csd(repo: Path, output_name: str) -> Path | None:
    latest = repo / "outputs" / "generated" / output_name / "latest_run.txt"
    if not latest.is_file():
        return None
    run_dir = Path(latest.read_text(encoding="utf-8").strip())
    if not run_dir.is_absolute():
        run_dir = repo / run_dir
    report = run_dir / "results" / "success_report.json"
    if not report.is_file():
        return None
    payload = json.loads(report.read_text(encoding="utf-8"))
    compiled_dir = Path(str(payload.get("compiled_dir", "")))
    if not compiled_dir.is_absolute():
        compiled_dir = repo / compiled_dir
    candidate = compiled_dir / "GeneratedCSD.py"
    return candidate if candidate.is_file() else None


def heldout_command(job: dict[str, Any], python: Path, csd: Path) -> list[str]:
    command = [
        str(python),
        "-m",
        "synthesis.scripts.reevaluate_compiled_csd",
        str(csd),
        "--dataset",
        str(job["dataset"]),
        "--eval-model",
        str(job["eval_model"]),
        "--eval-backend",
        "vllm",
        "--device",
        "auto",
        "--sample-size",
        str(job["heldout_sample_size"]),
        "--max-steps",
        str(job["eval_max_steps"]),
        "--step-token-budget",
        "1",
        "--max-seconds-per-example",
        str(job["eval_max_seconds"]),
        "--vllm-gpu-memory-utilization",
        str(job["gpu_mem_util"]),
        "--vllm-tensor-parallel-size",
        "1",
        "--output-json",
        str(job["heldout_output_json"]),
    ]
    split_file = str(job.get("heldout_split_file", "")).strip()
    split_name = str(job["heldout_split_name"])
    if job["dataset"] == "gsm_symbolic":
        if split_file:
            command.extend(["--gsm-split-file", split_file])
        command.extend(["--gsm-split-name", split_name])
    elif job["dataset"] == "spider":
        if split_file:
            command.extend(["--spider-split-file", split_file])
        command.extend(["--spider-split-name", split_name])
    else:
        command.extend(["--smiles-classes", str(job["smiles_class"])])
        command.extend(["--smiles-split-name", split_name, "--smiles-trial", "1"])
    return command


def author_free_environment(inherited: dict[str, str], gpu: int) -> dict[str, str]:
    clean = {
        key: value
        for key, value in inherited.items()
        if not key.startswith(("AWS_", "BEDROCK_"))
        and not key.endswith("_API_KEY")
    }
    clean["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return clean


def run_job(
    job: dict[str, Any],
    gpu: int,
    *,
    repo: Path,
    resume_script: Path,
    python: Path,
    dry_run: bool,
) -> int:
    env = worker_environment(job, gpu, os.environ)
    if dry_run:
        env["DRY_RUN"] = "1"
    csd = compiled_csd(repo, str(job["output_name"])) if not dry_run else None
    logger.warning(
        "[warm-recovery] start cell=%s gpu=%d replay=%s next=%d cap=%s reservation_mib=%s",
        job["cell_id"],
        gpu,
        job["last_clean_attempt"],
        int(job["last_clean_attempt"]) + 1,
        job["total_cap"],
        job["memory_reservation_mib"],
    )
    if csd is None:
        status = subprocess.run(
            ["bash", str(resume_script), "worker", str(job["cell_id"])],
            cwd=repo,
            env=env,
            check=False,
        ).returncode
        if dry_run or status != 0:
            logger.warning(
                "[warm-recovery] synthesis-finish cell=%s status=%d",
                job["cell_id"],
                status,
            )
            return status
        csd = compiled_csd(repo, str(job["output_name"]))
    else:
        logger.warning(
            "[warm-recovery] synthesis already succeeded; resuming heldout cell=%s csd=%s",
            job["cell_id"],
            csd,
        )

    if csd is None:
        logger.warning(
            "[warm-recovery] synthesis returned success without persisted CSD cell=%s",
            job["cell_id"],
        )
        return 3
    output_json = Path(str(job["heldout_output_json"]))
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with Path(str(job["log_file"])).open("a", encoding="utf-8") as log_handle:
        logger.warning(
            "[warm-recovery] heldout-start cell=%s gpu=%d csd=%s",
            job["cell_id"],
            gpu,
            csd,
        )
        heldout_status = subprocess.run(
            heldout_command(job, python, csd),
            cwd=repo,
            env=author_free_environment(os.environ, gpu),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            check=False,
        ).returncode
    logger.warning(
        "[warm-recovery] heldout-finish cell=%s status=%d output=%s",
        job["cell_id"],
        heldout_status,
        output_json,
    )
    return heldout_status


def dispatch(
    jobs: list[dict[str, Any]],
    *,
    snapshot,
    worker,
    poll_seconds: float,
    transient_retry_delay_seconds: float = 60.0,
    transient_max_requeues: int = 2,
) -> None:
    pending = [dict(job) for job in jobs]
    reservations: dict[int, dict[str, int]] = {}
    running: dict[concurrent.futures.Future[int], tuple[int, str]] = {}
    failures: list[tuple[str, int]] = []
    baseline = snapshot()
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(jobs)) as executor:
        while pending or running:
            snapshots = snapshot()
            for gpu in snapshots:
                reservations.setdefault(gpu, {})
                if not reservations[gpu]:
                    baseline[gpu] = dict(snapshots[gpu])
            launched = True
            while pending and launched:
                launched = False
                for index, job in enumerate(pending):
                    if time.monotonic() < float(job.get("_transient_not_before", 0.0)):
                        continue
                    gpu = choose_gpu(job, snapshots, reservations, baseline)
                    if gpu is None:
                        continue
                    cell_id = str(job["cell_id"])
                    reservation = required_memory_mib(
                        job, snapshots[gpu]["total_mib"]
                    )
                    reservations[gpu][cell_id] = reservation
                    logger.warning(
                        "[warm-recovery] dispatch cell=%s gpu=%d "
                        "job_reservation_mib=%d reserved_total_mib=%d",
                        cell_id,
                        gpu,
                        reservation,
                        sum(reservations[gpu].values()),
                    )
                    running[executor.submit(worker, job, gpu)] = (gpu, cell_id)
                    pending.pop(index)
                    launched = True
                    break
            finished = [future for future in running if future.done()]
            for future in finished:
                gpu, cell_id = running.pop(future)
                status = future.result()
                reservations[gpu].pop(cell_id, None)
                logger.warning(
                    "[warm-recovery] release cell=%s gpu=%d status=%d",
                    cell_id,
                    gpu,
                    status,
                )
                if status == TRANSIENT_PROVIDER_EXIT:
                    original = next(job for job in jobs if str(job["cell_id"]) == cell_id)
                    retries = int(original.get("_transient_requeues", 0))
                    if retries < transient_max_requeues:
                        replacement = dict(original)
                        replacement["_transient_requeues"] = retries + 1
                        replacement["_transient_not_before"] = (
                            time.monotonic() + transient_retry_delay_seconds
                        )
                        logger.warning(
                            "[warm-recovery] temporary-provider requeue cell=%s "
                            "retry=%d/%d delay_seconds=%.1f",
                            cell_id,
                            retries + 1,
                            transient_max_requeues,
                            transient_retry_delay_seconds,
                        )
                        pending.append(replacement)
                        for index, source_job in enumerate(jobs):
                            if str(source_job["cell_id"]) == cell_id:
                                jobs[index] = replacement
                                break
                    else:
                        failures.append((cell_id, status))
                elif status not in (0, 75):
                    failures.append((cell_id, status))
            if finished:
                continue
            if pending and not running:
                if all(
                    time.monotonic() < float(job.get("_transient_not_before", 0.0))
                    for job in pending
                ):
                    time.sleep(max(0.0, min(poll_seconds, transient_retry_delay_seconds)))
                    continue
                raise ConfigError("no queued job fits available GPU memory")
            if pending or running:
                time.sleep(max(0.0, poll_seconds))
    if failures:
        rendered = ", ".join(f"{cell}:{status}" for cell, status in failures)
        raise ConfigError(f"worker failures require supervisor retry: {rendered}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--resume-script", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--lock-file", type=Path, required=True)
    parser.add_argument("--nvidia-smi", default="nvidia-smi")
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--transient-retry-delay-seconds", type=float, default=60.0)
    parser.add_argument("--transient-max-requeues", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s")
    try:
        jobs = load_manifest(args.manifest)
        args.lock_file.parent.mkdir(parents=True, exist_ok=True)
        with args.lock_file.open("w", encoding="utf-8") as lock:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ConfigError("another warm-recovery controller is active") from exc
            worker = lambda job, gpu: run_job(
                job,
                gpu,
                repo=args.repo,
                resume_script=args.resume_script,
                python=args.python,
                dry_run=args.dry_run,
            )
            if args.dry_run:
                for index, job in enumerate(jobs):
                    status = worker(job, index % 4)
                    if status != 0:
                        raise ConfigError(
                            f"dry-run failed for {job['cell_id']} with status {status}"
                        )
            else:
                dispatch(
                    jobs,
                    snapshot=lambda: gpu_memory_snapshot(args.nvidia_smi),
                    worker=worker,
                    poll_seconds=args.poll_seconds,
                    transient_retry_delay_seconds=args.transient_retry_delay_seconds,
                    transient_max_requeues=args.transient_max_requeues,
                )
        return 0
    except ConfigError as exc:
        logger.error("[warm-recovery] configuration error: %s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
