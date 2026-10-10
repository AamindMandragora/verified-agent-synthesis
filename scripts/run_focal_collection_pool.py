#!/usr/bin/env python3
"""Greedily run Dynamic CSD collection jobs across focal GPUs."""

from __future__ import annotations

import argparse
import csv
import logging
import os
import signal
import subprocess
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple


LOGGER = logging.getLogger("focal-collection-pool")
SPIDER_SPLIT = "environment/benchmark_splits/spider_dev_proportional_300x300_seed334.json"
GSM_SPLIT = "environment/benchmark_splits/gsm_symbolic_crane_proportional_49x49_seed123.json"
FULL_BASELINE_CAMPAIGN = "full-baseline-20260803"
FULL_BASELINE_OUTPUT_NAME = "full_baseline_20260803"
FULL_BASELINE_STRATEGIES = ("unconstrained", "gcd", "crane", "itergen", "cars")
FULL_BASELINE_MODELS = (
    ("qwen25-1p5b", "Qwen/Qwen2.5-1.5B-Instruct", 16_000, 0.30),
    ("qwen25-7b", "Qwen/Qwen2.5-7B-Instruct", 22_000, 0.45),
    ("qwen35-2b", "Qwen/Qwen3.5-2B", 16_384, 0.35),
    ("qwen35-4b", "Qwen/Qwen3.5-4B", 19_000, 0.40),
)


class Job(NamedTuple):
    label: str
    output_json: Path
    log_path: Path
    args: tuple[str, ...]
    attempt: int = 0
    estimated_memory_mib: int = 10_000
    exclusive_gpu: bool = False


class RunningJob(NamedTuple):
    job: Job
    process: subprocess.Popen[bytes]
    log_file: object
    started_at: str


class ExternalJob(NamedTuple):
    process_id: int
    gpu_id: int
    estimated_memory_mib: int
    output_json: Path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def fixed_strategy_args(
    *,
    strategy: str,
    dataset: str,
    model: str,
    backend: str,
    device: str,
    sample_size: int,
    max_steps: int,
    output_json: Path,
    extra: tuple[str, ...] = (),
    vllm_gpu_memory_utilization: float = 0.90,
) -> tuple[str, ...]:
    args = (
        "-m",
        "synthesis.evaluate.run_legacy_fixed_strategy",
        "--strategy",
        strategy,
        "--dataset",
        dataset,
        "--eval-model",
        model,
        "--eval-backend",
        backend,
        "--device",
        device,
        "--eval-sample-size",
        str(sample_size),
        "--eval-max-steps",
        str(max_steps),
        "--eval-step-token-budget",
        "1",
        *extra,
    )
    if backend == "vllm":
        args += (
            "--vllm-gpu-memory-utilization",
            f"{vllm_gpu_memory_utilization:.2f}",
            "--vllm-tensor-parallel-size",
            "1",
        )
    return args + ("--output-json", str(output_json))


def build_manifest(repo: Path) -> list[Job]:
    jobs: list[Job] = []
    cars_root = repo / "outputs/baselines/cars_thinking_off"

    for start in range(125, 300, 25):
        end = start + 25
        slice_name = f"{start:03d}_{end:03d}"
        output_json = cars_root / "Qwen_Qwen3-5-4B/chunks" / f"spider_seed334_test300__tb1__ms600__{slice_name}.json"
        log_path = repo / "logs" / f"focal_collection_cars_4b_{slice_name}.log"
        jobs.append(
            Job(
                f"cars-4b-{start:03d}-{end:03d}",
                output_json,
                log_path,
                fixed_strategy_args(
                    strategy="cars",
                    dataset="spider",
                    model="Qwen/Qwen3.5-4B",
                    backend="huggingface",
                    device="cuda",
                    sample_size=300,
                    max_steps=600,
                    output_json=output_json,
                    extra=(
                        "--eval-start-index",
                        str(start),
                        "--eval-end-index",
                        str(end),
                        "--spider-split-file",
                        SPIDER_SPLIT,
                        "--spider-split-name",
                        "test",
                    ),
                ),
                estimated_memory_mib=10_000,
            )
        )

    for start in range(0, 300, 10):
        end = start + 10
        slice_name = f"{start:03d}_{end:03d}"
        output_json = cars_root / "Qwen_Qwen3-5-9B/chunks" / f"spider_seed334_test300__tb1__ms600__{slice_name}.json"
        log_path = repo / "logs" / f"focal_collection_cars_9b_{slice_name}.log"
        jobs.append(
            Job(
                f"cars-9b-{start:03d}-{end:03d}",
                output_json,
                log_path,
                fixed_strategy_args(
                    strategy="cars",
                    dataset="spider",
                    model="Qwen/Qwen3.5-9B",
                    backend="huggingface",
                    device="cuda",
                    sample_size=300,
                    max_steps=600,
                    output_json=output_json,
                    extra=(
                        "--eval-start-index",
                        str(start),
                        "--eval-end-index",
                        str(end),
                        "--spider-split-file",
                        SPIDER_SPLIT,
                        "--spider-split-name",
                        "test",
                    ),
                ),
                estimated_memory_mib=19_000,
            )
        )

    for strategy in ("unconstrained", "gcd", "crane"):
        output_json = repo / f"outputs/baselines/{strategy}/Qwen_Qwen3-5-9B/spider_seed334_test300__tb1__ms600.json"
        log_path = repo / "logs" / f"focal_collection_spider_9b_{strategy}.log"
        jobs.append(
            Job(
                f"spider-9b-{strategy}",
                output_json,
                log_path,
                fixed_strategy_args(
                    strategy=strategy,
                    dataset="spider",
                    model="Qwen/Qwen3.5-9B",
                    backend="vllm",
                    device="cuda",
                    sample_size=300,
                    max_steps=600,
                    output_json=output_json,
                    extra=(
                        "--spider-split-file",
                        SPIDER_SPLIT,
                        "--spider-split-name",
                        "test",
                    ),
                ),
                # CRANE routes through AdaptiveSynCode/Hugging Face in the fixed-strategy
                # adapter, so it has the same model-sized footprint as GCD rather than
                # reserving 90% of the GPU like the true vLLM paths.
                estimated_memory_mib=19_000 if strategy == "crane" else 39_000,
                exclusive_gpu=strategy != "crane",
            )
        )

    smiles_jobs = (
        ("unconstrained", "acrylates"),
        ("unconstrained", "chain_extenders"),
        ("itergen", "chain_extenders"),
        ("unconstrained", "isocyanates"),
    )
    for strategy, class_name in smiles_jobs:
        output_json = repo / f"outputs/controlled_comparison/smiles_qwen35_9b/{class_name}/{strategy}.json"
        log_path = repo / "logs" / f"focal_collection_smiles_9b_{class_name}_{strategy}.log"
        jobs.append(
            Job(
                f"smiles-9b-{class_name}-{strategy}",
                output_json,
                log_path,
                fixed_strategy_args(
                    strategy=strategy,
                    dataset="smiles",
                    model="Qwen/Qwen3.5-9B",
                    backend="vllm",
                    device="cuda",
                    sample_size=100,
                    max_steps=400,
                    output_json=output_json,
                    extra=(
                        "--smiles-classes",
                        class_name,
                        "--smiles-split-name",
                        "test",
                        "--smiles-trial",
                        "1",
                        "--cars-search-steps",
                        "200",
                    ),
                ),
                estimated_memory_mib=39_000,
                exclusive_gpu=True,
            )
        )

    return jobs


def build_full_baseline_campaign(
    repo: Path,
    *,
    campaign_name: str = FULL_BASELINE_OUTPUT_NAME,
    include_labels: set[str] | None = None,
) -> list[Job]:
    """Build the approved five-strategy, four-model, five-cohort baseline matrix."""

    if Path(campaign_name).name != campaign_name or campaign_name in {"", ".", ".."}:
        raise ValueError("campaign_name must be one safe path component")
    jobs: list[Job] = []
    campaign_root = repo / "outputs/baselines" / campaign_name
    log_root = repo / "logs" / campaign_name
    cohorts = (
        (
            "gsm",
            "gsm_symbolic",
            49,
            900,
            (
                "--gsm-split-file",
                GSM_SPLIT,
                "--gsm-split-name",
                "train",
            ),
        ),
        (
            "spider",
            "spider",
            300,
            176,
            (
                "--spider-split-file",
                SPIDER_SPLIT,
                "--spider-split-name",
                "train",
            ),
        ),
        (
            "smiles-acrylates",
            "smiles",
            50,
            400,
            (
                "--smiles-classes",
                "acrylates",
                "--smiles-samples-per-class",
                "50",
                "--smiles-split-name",
                "train",
                "--smiles-trial",
                "1",
            ),
        ),
        (
            "smiles-chain_extenders",
            "smiles",
            50,
            400,
            (
                "--smiles-classes",
                "chain_extenders",
                "--smiles-samples-per-class",
                "50",
                "--smiles-split-name",
                "train",
                "--smiles-trial",
                "1",
            ),
        ),
        (
            "smiles-isocyanates",
            "smiles",
            50,
            400,
            (
                "--smiles-classes",
                "isocyanates",
                "--smiles-samples-per-class",
                "50",
                "--smiles-split-name",
                "train",
                "--smiles-trial",
                "1",
            ),
        ),
    )

    for cohort, dataset, sample_size, max_steps, extra in cohorts:
        for model_slug, model, reservation_mib, gpu_utilization in FULL_BASELINE_MODELS:
            for strategy in FULL_BASELINE_STRATEGIES:
                output_json = campaign_root / cohort / model_slug / f"{strategy}.json"
                log_path = log_root / f"{cohort}-{model_slug}-{strategy}.log"
                jobs.append(
                    Job(
                        f"{cohort}-{model_slug}-{strategy}",
                        output_json,
                        log_path,
                        fixed_strategy_args(
                            strategy=strategy,
                            dataset=dataset,
                            model=model,
                            backend="vllm",
                            device="cuda",
                            sample_size=sample_size,
                            max_steps=max_steps,
                            output_json=output_json,
                            extra=extra,
                            vllm_gpu_memory_utilization=gpu_utilization,
                        ),
                        estimated_memory_mib=reservation_mib,
                    )
                )

    if include_labels is None:
        return jobs
    known_labels = {job.label for job in jobs}
    unknown_labels = include_labels - known_labels
    if unknown_labels:
        raise ValueError(f"unknown full-baseline labels: {sorted(unknown_labels)}")
    return [job for job in jobs if job.label in include_labels]


def ready_gpu_ids(
    memory_used_mib: dict[int, int],
    busy_gpu_ids: set[int],
    max_idle_memory_mib: int,
) -> list[int]:
    return sorted(
        gpu_id
        for gpu_id, used_mib in memory_used_mib.items()
        if gpu_id not in busy_gpu_ids and used_mib <= max_idle_memory_mib
    )


def claim_fitting_job(
    queue: deque[Job],
    *,
    used_memory_mib: int,
    total_memory_mib: int,
    safety_memory_mib: int,
    idle_memory_mib: int,
) -> tuple[Job | None, list[str]]:
    skipped: list[str] = []
    candidates = len(queue)
    for _ in range(candidates):
        job = queue.popleft()
        if job.output_json.is_file() and job.output_json.stat().st_size > 0:
            skipped.append(job.label)
            continue
        claim_path = job_claim_path(job)
        if claim_path.exists():
            queue.append(job)
            continue
        if job.exclusive_gpu:
            fits = used_memory_mib <= idle_memory_mib
        else:
            fits = used_memory_mib + job.estimated_memory_mib <= total_memory_mib - safety_memory_mib
        if fits:
            claim_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                claim_path.mkdir()
            except FileExistsError:
                queue.append(job)
                continue
            return job, skipped
        queue.append(job)
    return None, skipped


def job_claim_path(job: Job) -> Path:
    return job.output_json.with_name(f"{job.output_json.name}.running")


def process_group_is_alive(process_group_id: int) -> bool:
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def stop_process_group(process_group_id: int, timeout_seconds: float = 30.0) -> bool:
    if not process_group_is_alive(process_group_id):
        return True
    try:
        os.killpg(process_group_id, signal.SIGTERM)
    except ProcessLookupError:
        return True
    deadline = time.monotonic() + timeout_seconds
    while process_group_is_alive(process_group_id) and time.monotonic() < deadline:
        time.sleep(0.1)
    if process_group_is_alive(process_group_id):
        try:
            os.killpg(process_group_id, signal.SIGKILL)
        except ProcessLookupError:
            return True
        deadline = time.monotonic() + 5.0
        while process_group_is_alive(process_group_id) and time.monotonic() < deadline:
            time.sleep(0.1)
    return not process_group_is_alive(process_group_id)


def release_job_claim(job: Job, *, process_group_id: int | None = None) -> bool:
    if process_group_id is not None and process_group_is_alive(process_group_id):
        return False
    try:
        job_claim_path(job).rmdir()
    except FileNotFoundError:
        pass
    return True


def requeue_failed_job(job: Job, queue: deque[Job], max_retries: int) -> bool:
    if job.attempt >= max_retries:
        return False
    queue.append(job._replace(attempt=job.attempt + 1))
    return True


def pid_is_alive(process_id: int) -> bool:
    try:
        os.kill(process_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def external_reserved_memory(external_jobs: dict[int, ExternalJob], gpu_ids: list[int]) -> dict[int, int]:
    reserved = {gpu_id: 0 for gpu_id in gpu_ids}
    for external in external_jobs.values():
        if external.gpu_id in reserved:
            reserved[external.gpu_id] += external.estimated_memory_mib
    return reserved


def projected_gpu_memory(
    *,
    measured_memory_mib: dict[int, int],
    external_reserved_mib: dict[int, int],
    managed_reserved_mib: dict[int, int],
) -> dict[int, int]:
    """Conservatively combine live usage with jobs that have not allocated yet."""

    return {
        gpu_id: max(
            measured_memory_mib[gpu_id] + managed_reserved_mib[gpu_id],
            external_reserved_mib[gpu_id] + managed_reserved_mib[gpu_id],
        )
        for gpu_id in measured_memory_mib
    }


def reconcile_external_jobs(
    external_jobs: dict[int, ExternalJob],
    queue: deque[Job],
    jobs_by_output: dict[Path, Job],
    *,
    is_alive=pid_is_alive,
) -> None:
    queued_outputs = {job.output_json.resolve() for job in queue}
    for process_id, external in list(external_jobs.items()):
        if is_alive(process_id):
            continue
        output_path = external.output_json.resolve()
        output_ok = output_path.is_file() and output_path.stat().st_size > 0
        if output_ok:
            LOGGER.info(
                "[focal-collection-pool] adopted job completed pid=%s output=%s",
                process_id,
                output_path,
            )
        else:
            original = jobs_by_output.get(output_path)
            if original is not None and output_path not in queued_outputs:
                queue.append(original._replace(attempt=max(1, original.attempt)))
                queued_outputs.add(output_path)
                LOGGER.warning(
                    "[focal-collection-pool] adopted job failed pid=%s; requeued output=%s",
                    process_id,
                    output_path,
                )
        del external_jobs[process_id]


def query_gpu_memory(gpu_ids: list[int]) -> dict[int, int]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,memory.used",
        "--format=csv,noheader,nounits",
    ]
    output = subprocess.check_output(command, text=True)
    selected = set(gpu_ids)
    memory: dict[int, int] = {}
    for line in output.splitlines():
        index_text, used_text = (part.strip() for part in line.split(",", maxsplit=1))
        gpu_id = int(index_text)
        if gpu_id in selected:
            memory[gpu_id] = int(used_text)
    missing = selected - set(memory)
    if missing:
        raise RuntimeError(f"nvidia-smi omitted GPU ids: {sorted(missing)}")
    return memory


def write_status(
    status_path: Path,
    *,
    started_at: str,
    label: str,
    status: str,
    exit_code: int | str,
    gpu_id: int | str,
    attempt: int,
    output_json: Path,
    log_path: Path,
) -> None:
    status_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not status_path.exists()
    with status_path.open("a", encoding="utf-8", newline="") as status_file:
        writer = csv.writer(status_file, delimiter="\t")
        if new_file:
            writer.writerow(
                ("started_at", "finished_at", "label", "status", "exit_code", "gpu", "attempt", "output_json", "log")
            )
        writer.writerow(
            (
                started_at,
                utc_now(),
                label,
                status,
                exit_code,
                gpu_id,
                attempt,
                output_json,
                log_path,
            )
        )


def launch_job(job: Job, gpu_id: int, python: Path, repo: Path) -> RunningJob:
    job.output_json.parent.mkdir(parents=True, exist_ok=True)
    job.log_path.parent.mkdir(parents=True, exist_ok=True)
    log_mode = "a" if job.attempt else "w"
    log_file = job.log_path.open(log_mode, encoding="utf-8")
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu_id),
            "CSD_HF_KV_CACHE": "0",
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": "synthesis/evaluate:.",
        }
    )
    command = [str(python), *job.args]
    LOGGER.info(
        "[focal-collection-pool] launch label=%s gpu=%s attempt=%s estimate_mib=%s exclusive=%s output=%s",
        job.label,
        gpu_id,
        job.attempt,
        job.estimated_memory_mib,
        job.exclusive_gpu,
        job.output_json,
    )
    try:
        process = subprocess.Popen(
            command,
            cwd=repo,
            env=environment,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    except BaseException:
        log_file.close()
        release_job_claim(job)
        raise
    return RunningJob(job, process, log_file, utc_now())


def parse_gpu_ids(value: str) -> list[int]:
    gpu_ids = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not gpu_ids or len(gpu_ids) != len(set(gpu_ids)):
        raise argparse.ArgumentTypeError("GPU ids must be a non-empty unique comma-separated list")
    return gpu_ids


def parse_external_job(value: str) -> ExternalJob:
    try:
        process_text, gpu_text, memory_text, output_text = value.split(":", maxsplit=3)
        return ExternalJob(
            int(process_text),
            int(gpu_text),
            int(memory_text),
            Path(output_text),
        )
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(
            "external job must be PID:GPU_ID:ESTIMATED_MEMORY_MIB:OUTPUT_JSON"
        ) from error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("/home/aadivyar/csd-generation"))
    parser.add_argument("--python", type=Path, default=Path("/apps/conda/aadivyar/envs/csd/bin/python"))
    parser.add_argument("--gpu-ids", type=parse_gpu_ids, default=parse_gpu_ids("0,1,2,3"))
    parser.add_argument("--max-idle-memory-mib", type=int, default=1000)
    parser.add_argument("--gpu-total-memory-mib", type=int, default=40_960)
    parser.add_argument("--safety-memory-mib", type=int, default=2_000)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument(
        "--campaign",
        choices=("remaining", FULL_BASELINE_CAMPAIGN),
        default="remaining",
    )
    parser.add_argument(
        "--campaign-output-name",
        default=FULL_BASELINE_OUTPUT_NAME,
        help="Versioned output/log directory name for the full-baseline campaign.",
    )
    parser.add_argument(
        "--include-label",
        action="append",
        default=[],
        help="Run only this exact full-baseline label; repeat for multiple labels.",
    )
    parser.add_argument("--exclude-output-json", action="append", type=Path, default=[])
    parser.add_argument("--external-job", action="append", type=parse_external_job, default=[])
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = args.repo.resolve()
    python = args.python.resolve()
    if args.campaign == FULL_BASELINE_CAMPAIGN:
        status_path = repo / "logs" / args.campaign_output_name / "status.tsv"
        manifest = build_full_baseline_campaign(
            repo,
            campaign_name=args.campaign_output_name,
            include_labels=set(args.include_label) if args.include_label else None,
        )
    else:
        status_path = repo / "logs/focal_collection_pool_status.tsv"
        manifest = build_manifest(repo)
    jobs_by_output = {job.output_json.resolve(): job for job in manifest}
    external_jobs: dict[int, ExternalJob] = {}
    for external in args.external_job:
        output_json = external.output_json if external.output_json.is_absolute() else repo / external.output_json
        external_jobs[external.process_id] = external._replace(output_json=output_json.resolve())

    excluded_outputs = {
        (path if path.is_absolute() else repo / path).resolve() for path in args.exclude_output_json
    }
    excluded_outputs.update(external.output_json for external in external_jobs.values())
    queue = deque(job for job in manifest if job.output_json.resolve() not in excluded_outputs)
    running: dict[int, tuple[int, RunningJob]] = {}

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    LOGGER.info(
        "[focal-collection-pool] start jobs=%s gpu_ids=%s max_workers=%s max_retries=%s excluded=%s adopted=%s",
        len(queue),
        args.gpu_ids,
        args.max_workers,
        args.max_retries,
        len(excluded_outputs),
        len(external_jobs),
    )

    if args.dry_run:
        for job in queue:
            print(f"{job.label}\t{job.output_json}\t{job.log_path}\t{' '.join(job.args)}")
        return 0

    if not repo.is_dir():
        raise SystemExit(f"repo does not exist: {repo}")
    if not python.is_file():
        raise SystemExit(f"python does not exist: {python}")

    interrupted = False

    def stop_running(_signum: int, _frame: object) -> None:
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGTERM, stop_running)
    signal.signal(signal.SIGINT, stop_running)

    while (queue or running or external_jobs) and not interrupted:
        reconcile_external_jobs(external_jobs, queue, jobs_by_output)
        for process_id, (gpu_id, active) in list(running.items()):
            exit_code = active.process.poll()
            if exit_code is None:
                continue
            active.log_file.close()
            group_stopped = stop_process_group(process_id)
            released = release_job_claim(active.job, process_group_id=process_id)
            if not group_stopped or not released:
                LOGGER.error(
                    "[focal-collection-pool] retaining claim for live process group label=%s pgid=%s",
                    active.job.label,
                    process_id,
                )
            output_ok = active.job.output_json.is_file() and active.job.output_json.stat().st_size > 0
            succeeded = exit_code == 0 and output_ok
            status = "done" if succeeded else ("no_output" if exit_code == 0 else "failed")
            requeued = False
            if not succeeded:
                requeued = requeue_failed_job(active.job, queue, args.max_retries)
                if requeued:
                    status = f"{status}_requeued"
            write_status(
                status_path,
                started_at=active.started_at,
                label=active.job.label,
                status=status,
                exit_code=exit_code,
                gpu_id=gpu_id,
                attempt=active.job.attempt,
                output_json=active.job.output_json,
                log_path=active.job.log_path,
            )
            LOGGER.info(
                "[focal-collection-pool] finish label=%s gpu=%s exit=%s status=%s pending=%s",
                active.job.label,
                gpu_id,
                exit_code,
                status,
                len(queue),
            )
            del running[process_id]

        if queue:
            try:
                memory = query_gpu_memory(args.gpu_ids)
            except (OSError, subprocess.SubprocessError, ValueError, RuntimeError) as error:
                LOGGER.warning("[focal-collection-pool] GPU query failed: %s", error)
                time.sleep(args.poll_seconds)
                continue
            external_reserved = external_reserved_memory(external_jobs, args.gpu_ids)
            managed_reserved = {gpu_id: 0 for gpu_id in args.gpu_ids}
            for gpu_id, active in running.values():
                managed_reserved[gpu_id] += active.job.estimated_memory_mib
            projected_memory = projected_gpu_memory(
                measured_memory_mib=memory,
                external_reserved_mib=external_reserved,
                managed_reserved_mib=managed_reserved,
            )
            launched = True
            while queue and len(running) < args.max_workers and launched:
                launched = False
                for gpu_id in sorted(args.gpu_ids, key=projected_memory.__getitem__):
                    if len(running) >= args.max_workers:
                        break
                    job, skipped = claim_fitting_job(
                        queue,
                        used_memory_mib=projected_memory[gpu_id],
                        total_memory_mib=args.gpu_total_memory_mib,
                        safety_memory_mib=args.safety_memory_mib,
                        idle_memory_mib=args.max_idle_memory_mib,
                    )
                    for label in skipped:
                        LOGGER.info("[focal-collection-pool] skip existing label=%s", label)
                    if job is None:
                        continue
                    active = launch_job(job, gpu_id, python, repo)
                    running[active.process.pid] = (gpu_id, active)
                    projected_memory[gpu_id] += job.estimated_memory_mib
                    launched = True

        if queue or running or external_jobs:
            time.sleep(args.poll_seconds)

    if interrupted:
        LOGGER.warning("[focal-collection-pool] stopping %s active jobs", len(running))
        for _gpu_id, active in running.values():
            group_stopped = stop_process_group(active.process.pid)
            active.log_file.close()
            released = release_job_claim(active.job, process_group_id=active.process.pid)
            if not group_stopped or not released:
                LOGGER.error(
                    "[focal-collection-pool] retaining claim for live process group label=%s pgid=%s",
                    active.job.label,
                    active.process.pid,
                )
        return 130

    LOGGER.info("[focal-collection-pool] complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
