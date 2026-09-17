"""Hardened pre-PR smoke gate with domain/path meaningful metrics.

Callers: orchestrator via BabysitterHooks.run_smoke; production_hooks.
API: SmokeGate.evaluate; evaluate_smoke_metrics; parse_smoke_report;
     smoke_criteria_for(path_kind).
User instruction: no merge on SMOKE_FAIL; SMILES UV>0; no soft uniq_tokens proxy.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from scripts.runtime.zero_acc_babysitter.cloud import _safe_branch_slug
from scripts.runtime.zero_acc_babysitter.constants import PathKind, SMOKE_MODEL
from scripts.runtime.zero_acc_babysitter.domains import domain_key
from scripts.runtime.zero_acc_babysitter.persistence import IncidentRecord


@dataclass(frozen=True)
class SmokeCriteria:
    name: str
    description: str
    model: str = SMOKE_MODEL
    max_iters: int = 3
    tiny_n: bool = True


HARNESS_CRITERIA = SmokeCriteria(
    name="harness",
    description=(
        "SMILES: unique_valid_count>0 (primary); "
        "GSM/Spider: Acc>0% on tiny N; no soft uniq_tokens proxy"
    ),
)
HELPER_CRITERIA = SmokeCriteria(
    name="helper",
    description="helper micro-check on PR tip; failing helper list empty",
)
TELEMETRY_CRITERIA = SmokeCriteria(
    name="telemetry",
    description="finished attempt emits Accuracy line under smoke model",
)
MEMORY_CRITERIA = SmokeCriteria(
    name="memory",
    description="short run completes without OOM signature",
)


def smoke_criteria_for(path_kind: str) -> SmokeCriteria:
    if path_kind == PathKind.HARNESS.value:
        return HARNESS_CRITERIA
    if path_kind == PathKind.TELEMETRY.value:
        return TELEMETRY_CRITERIA
    if path_kind in {PathKind.HELPER.value, PathKind.TIER_B_HELPER.value}:
        return HELPER_CRITERIA
    if path_kind == PathKind.MEMORY.value:
        return MEMORY_CRITERIA
    return SmokeCriteria(name="default", description=f"generic smoke for {path_kind}")


@dataclass
class SmokeMetrics:
    """Measured smoke outcomes. uniq_tokens is informational only — never a pass key."""

    accuracy_pct: float | None = None
    unique_valid_count: int | None = None
    unique_valid_rate: float | None = None
    twin_accuracy_pct: float | None = None
    helper_failures_remaining: list[str] = field(default_factory=list)
    accuracy_line_present: bool | None = None
    oom_present: bool | None = None
    uniq_tokens: int | None = None  # soft proxy — never sufficient alone
    process_rc: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SmokeDecision:
    passed: bool
    reason: str
    metrics: SmokeMetrics


@dataclass
class SmokeReport:
    passed: bool
    command: str
    metrics: dict[str, float]
    reason: str = ""


def _smiles_primary_ok(metrics: SmokeMetrics) -> tuple[bool, str]:
    """SMILES harness primary bar: unique-valid > 0 (count or rate)."""
    uv_count = metrics.unique_valid_count
    uv_rate = metrics.unique_valid_rate
    if uv_count is not None and int(uv_count) > 0:
        return True, f"unique_valid_count={uv_count}"
    if uv_rate is not None and float(uv_rate) > 0.0:
        return True, f"unique_valid_rate={uv_rate}"
    # Acc alone at 0% with UV missing/0 is FAIL. Soft uniq_tokens never saves it.
    return False, (
        f"SMILES unique_valid must be >0 "
        f"(count={uv_count}, rate={uv_rate}, "
        f"acc={metrics.accuracy_pct}, uniq_tokens={metrics.uniq_tokens} ignored)"
    )


def _gsm_spider_harness_ok(metrics: SmokeMetrics) -> tuple[bool, str]:
    acc = metrics.accuracy_pct
    twin = metrics.twin_accuracy_pct
    if acc is not None and float(acc) > 0.0:
        return True, f"accuracy_pct={acc}"
    if twin is not None and float(twin) > 0.0:
        return True, f"twin_accuracy_pct={twin}"
    return False, f"GSM/Spider harness needs Acc>0 or twin>0 (acc={acc}, twin={twin})"


def evaluate_smoke_metrics(
    *,
    cell_id: str,
    path_kind: str,
    metrics: SmokeMetrics,
) -> SmokeDecision:
    """Domain/path meaningful smoke gate. Soft uniq_tokens alone never PASSes."""
    if metrics.process_rc is not None and int(metrics.process_rc) != 0:
        return SmokeDecision(
            False,
            f"smoke process_rc={metrics.process_rc}",
            metrics,
        )

    if path_kind == PathKind.MEMORY.value:
        if metrics.oom_present is True:
            return SmokeDecision(False, "OOM signature still present", metrics)
        if metrics.oom_present is False:
            return SmokeDecision(True, "OOM signature absent", metrics)
        return SmokeDecision(False, "memory smoke missing oom_present flag", metrics)

    if path_kind == PathKind.TELEMETRY.value:
        if metrics.accuracy_line_present is True:
            return SmokeDecision(True, "Accuracy line present", metrics)
        return SmokeDecision(
            False,
            "telemetry smoke requires per-attempt Accuracy line",
            metrics,
        )

    if path_kind in {PathKind.HELPER.value, PathKind.TIER_B_HELPER.value}:
        remaining = list(metrics.helper_failures_remaining or [])
        if remaining:
            return SmokeDecision(
                False,
                f"helper micros still failing: {','.join(remaining)}",
                metrics,
            )
        return SmokeDecision(True, "helper micros green", metrics)

    # Harness (and telemetry-family harness recovery uses HARNESS criteria for SMILES UV).
    if path_kind in {PathKind.HARNESS.value, PathKind.TELEMETRY.value}:
        # TELEMETRY already handled above; harness path:
        pass

    if path_kind == PathKind.HARNESS.value:
        key = domain_key(cell_id)
        if key.startswith("smiles-"):
            ok, reason = _smiles_primary_ok(metrics)
            return SmokeDecision(ok, reason, metrics)
        if key in {"gsm", "spider"}:
            ok, reason = _gsm_spider_harness_ok(metrics)
            return SmokeDecision(ok, reason, metrics)
        return SmokeDecision(False, f"unknown harness domain {key}", metrics)

    return SmokeDecision(False, f"no smoke rule for path_kind={path_kind}", metrics)


def parse_smoke_report(
    report_path: Path,
    *,
    process_rc: int | None = None,
    uniq_tokens: int | None = None,
    accuracy_line_present: bool | None = None,
    oom_present: bool | None = None,
    helper_failures_remaining: list[str] | None = None,
    twin_accuracy_pct: float | None = None,
) -> SmokeMetrics:
    """Parse reevaluate / smoke JSON into SmokeMetrics (UV from smiles_paper_trial)."""
    payload: dict[str, Any] = {}
    if report_path.is_file():
        payload = json.loads(report_path.read_text(encoding="utf-8"))

    acc_raw = payload.get("accuracy")
    if acc_raw is None and isinstance(payload.get("metrics"), dict):
        acc_raw = payload["metrics"].get("accuracy")
    accuracy_pct: float | None = None
    if acc_raw is not None:
        acc_f = float(acc_raw)
        accuracy_pct = acc_f * 100.0 if acc_f <= 1.0 else acc_f

    trial = payload.get("smiles_paper_trial") or {}
    if not isinstance(trial, dict):
        trial = {}
    uv_count = trial.get("unique_valid_count")
    uv_rate = trial.get("unique_valid_rate")
    if uv_rate is None and uv_count is not None:
        sample_count = trial.get("sample_count") or payload.get("metrics", {}).get(
            "num_examples"
        )
        try:
            n = int(sample_count) if sample_count else 0
            uv_rate = (float(uv_count) / n) if n > 0 else 0.0
        except (TypeError, ValueError):
            uv_rate = None

    # Acc override for SMILES unique-valid rate when present as fraction in accuracy.
    if (
        accuracy_pct is not None
        and accuracy_pct == 0.0
        and uv_count is None
        and isinstance(payload.get("metrics"), dict)
    ):
        # leave as-is; UV fields dominate for SMILES
        pass

    return SmokeMetrics(
        accuracy_pct=accuracy_pct,
        unique_valid_count=int(uv_count) if uv_count is not None else None,
        unique_valid_rate=float(uv_rate) if uv_rate is not None else None,
        twin_accuracy_pct=twin_accuracy_pct,
        helper_failures_remaining=list(helper_failures_remaining or []),
        accuracy_line_present=accuracy_line_present,
        oom_present=oom_present,
        uniq_tokens=uniq_tokens,
        process_rc=process_rc,
        extra={"report_path": str(report_path)},
    )


class SmokeGate:
    def __init__(self, decide) -> None:
        self._decide = decide

    def evaluate(self, incident: IncidentRecord) -> SmokeReport:
        criteria = smoke_criteria_for(incident.path_kind)
        result = self._decide(incident)
        reason = ""
        metrics_dict: dict[str, float] = {}
        if isinstance(result, SmokeDecision):
            passed = bool(result.passed)
            reason = result.reason
            if result.metrics.accuracy_pct is not None:
                metrics_dict["accuracy_pct"] = float(result.metrics.accuracy_pct)
            if result.metrics.unique_valid_count is not None:
                metrics_dict["unique_valid_count"] = float(
                    result.metrics.unique_valid_count
                )
            if result.metrics.unique_valid_rate is not None:
                metrics_dict["unique_valid_rate"] = float(
                    result.metrics.unique_valid_rate
                )
        else:
            passed = bool(result)
            metrics_dict["acc_proxy"] = 1.0 if passed else 0.0
        command = (
            f"smoke model={criteria.model} kind={incident.path_kind} "
            f"criteria={criteria.name} iters<={criteria.max_iters} tiny-N "
            f"cell={incident.cell_id}"
        )
        return SmokeReport(
            passed=passed,
            command=command,
            metrics=metrics_dict,
            reason=reason,
        )


logger = logging.getLogger("zero-acc-babysitter")

SmokeJobRunner = Callable[..., int]
SMOKE_SAMPLE_SIZE = 5
SMOKE_MAX_STEPS_SMILES = 200
SMOKE_MAX_STEPS_DEFAULT = 256
SMOKE_GPU_MEM_UTIL = 0.3
# GPUs 0 and 3 are routinely occupied by other users (see CLAUDE.md); smokes
# must stay on 1/2 or vLLM engine init fails on a full cuda:0.
SMOKE_ALLOWED_GPUS = ("1", "2")
# vLLM refuses to start unless free >= util * total; headroom covers the gap
# between the nvidia-smi snapshot and actual engine allocation.
SMOKE_GPU_HEADROOM_MB = 1024
SMOKE_GPU_WAIT_SECONDS = 900.0
SMOKE_GPU_POLL_SECONDS = 30.0
SMOKE_JOB_MAX_TRIES = 3
# Engine-init failure signatures: nothing was measured, so the run is safe to
# retry regardless of path kind (unlike a mid-run OOM, which the memory path
# treats as a meaningful signal).
GPU_CONTENTION_SIGNATURES = (
    "free memory on device",
    "engine core initialization failed",
)


def _allowed_gpu_inventory() -> list[tuple[str, int, int]] | None:
    """(index, free_mb, total_mb) rows for SMOKE_ALLOWED_GPUS, or None when
    nvidia-smi is unavailable (non-GPU host, tests)."""
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    rows: list[tuple[str, int, int]] = []
    for line in out.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 3 or parts[0] not in SMOKE_ALLOWED_GPUS:
            continue
        try:
            rows.append((parts[0], int(parts[1]), int(parts[2])))
        except ValueError:
            continue
    return rows or None


def smoke_gpu_required_free_mb(total_mb: int) -> int:
    return int(total_mb * SMOKE_GPU_MEM_UTIL) + SMOKE_GPU_HEADROOM_MB


def pick_smoke_gpu() -> str:
    rows = _allowed_gpu_inventory()
    if not rows:
        return SMOKE_ALLOWED_GPUS[-1]
    return max(rows, key=lambda r: r[1])[0]


def wait_for_smoke_gpu(
    *,
    max_wait_seconds: float = SMOKE_GPU_WAIT_SECONDS,
    poll_seconds: float = SMOKE_GPU_POLL_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
    now: Callable[[], float] = time.monotonic,
) -> str:
    """Block until an allowed GPU has enough free memory for the smoke vLLM
    engine (util * total + headroom), then return its physical index.

    Incident spider-qwen25-1p5b:3:harness:1784874093: the smoke picked GPU 2
    while it showed 19 GiB free, but another queue job loaded its engine in the
    scan-to-init window and the smoke died with rc=1 ("Free memory on device
    cuda:0 (4.5/39.49 GiB) ... desired 11.85 GiB"), burning a cloud repair
    attempt on pure GPU contention. Picking only a GPU that currently fits the
    engine — and waiting for one — closes that race window.

    Falls back to the freest allowed GPU at the deadline (no worse than the
    old behavior), or to the last allowed GPU when nvidia-smi is unavailable.
    """
    deadline = now() + max_wait_seconds
    while True:
        rows = _allowed_gpu_inventory()
        if rows is None:
            return SMOKE_ALLOWED_GPUS[-1]
        ready = [r for r in rows if r[1] >= smoke_gpu_required_free_mb(r[2])]
        if ready:
            return max(ready, key=lambda r: r[1])[0]
        if now() >= deadline:
            best = max(rows, key=lambda r: r[1])
            logger.warning(
                "no smoke GPU freed within %.0fs; best-effort GPU %s (free %s MiB)",
                max_wait_seconds,
                best[0],
                best[1],
            )
            return best[0]
        sleep(poll_seconds)


def gpu_contention_in_log(out_dir: Path) -> bool:
    log_path = out_dir / "smoke.log"
    if not log_path.is_file():
        return False
    text = log_path.read_text(encoding="utf-8", errors="replace").lower()
    return any(sig in text for sig in GPU_CONTENTION_SIGNATURES)


def cell_dataset_and_smiles_class(cell_id: str) -> tuple[str, str | None]:
    key = domain_key(cell_id)
    if key.startswith("smiles-"):
        return "smiles", key.split("-", 1)[1]
    if key == "gsm":
        return "gsm_symbolic", None
    if key == "spider":
        return "spider", None
    raise ValueError(f"unsupported smoke cell domain for {cell_id!r}")


def find_latest_generated_csd(live_repo: Path, cell_id: str) -> Path | None:
    root = live_repo.resolve() / "outputs" / "generated"
    if not root.is_dir():
        return None
    matches = sorted(
        root.glob(f"coldq_{cell_id}_*/**/GeneratedCSD.py"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return matches[0] if matches else None


def repair_branch_for_incident(incident: IncidentRecord) -> str:
    pr = (incident.pr_url or "").strip()
    if pr.startswith("branch:"):
        return pr.split(":", 1)[1]
    return f"babysitter-fix/{_safe_branch_slug(incident.incident_id)}"


def _default_subprocess_runner(
    cmd: list[str], *, cwd: Path, env: dict[str, str], out_dir: Path
) -> int:
    log_path = out_dir / "smoke.log"
    with log_path.open("w", encoding="utf-8") as logf:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            env=env,
            stdout=logf,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return int(proc.returncode)


def run_hardened_smoke_job(
    incident: IncidentRecord,
    *,
    live_repo: Path,
    repair_worktree: Path,
    python_bin: str | None = None,
    runner: SmokeJobRunner | None = None,
    uniq_tokens: int | None = None,
    sample_size: int = SMOKE_SAMPLE_SIZE,
) -> SmokeDecision:
    """Run tiny re-eval on PR tip; attach metrics to incident; return UV/Acc gate.

    Soft uniq_tokens is recorded only for diagnostics and never makes PASS.
    """
    live = live_repo.resolve()
    repair = repair_worktree.resolve()
    out_dir = (
        live
        / "logs"
        / "zero_acc_babysitter"
        / f"smoke_{incident.cell_id}_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "smoke_report.json"

    csd = find_latest_generated_csd(live, incident.cell_id)
    if csd is None:
        decision = SmokeDecision(
            False,
            f"no GeneratedCSD for cell={incident.cell_id}",
            SmokeMetrics(process_rc=2),
        )
        incident.extra = dict(incident.extra or {})
        incident.extra["smoke_fail_reason"] = decision.reason
        incident.extra["smoke_process_rc"] = 2
        incident.extra["smoke_attempt"] = incident.cloud_attempt_count
        incident.extra["smoke_metrics"] = {
            "process_rc": 2,
            "uniq_tokens": uniq_tokens,
        }
        return decision

    dataset, smiles_class = cell_dataset_and_smiles_class(incident.cell_id)
    max_steps = (
        SMOKE_MAX_STEPS_SMILES if dataset == "smiles" else SMOKE_MAX_STEPS_DEFAULT
    )
    py = python_bin or sys.executable
    branch = repair_branch_for_incident(incident)

    # Prefer PR tip code under repair worktree for PYTHONPATH.
    if repair.is_dir() and (repair / ".git").exists():
        fetch = subprocess.run(
            ["git", "fetch", "origin", branch],
            cwd=repair,
            capture_output=True,
            text=True,
            check=False,
        )
        if fetch.returncode == 0:
            subprocess.run(
                ["git", "checkout", "-B", branch, f"origin/{branch}"],
                cwd=repair,
                capture_output=True,
                text=True,
                check=False,
            )
        else:
            subprocess.run(
                ["git", "checkout", "-B", branch],
                cwd=repair,
                capture_output=True,
                text=True,
                check=False,
            )

    code_root = repair if (repair / "synthesis").is_dir() else live
    cmd = [
        py,
        "-m",
        "synthesis.scripts.reevaluate_compiled_csd",
        str(csd),
        "--dataset",
        dataset,
        "--eval-model",
        SMOKE_MODEL,
        "--sample-size",
        str(sample_size),
        "--max-steps",
        str(max_steps),
        "--vllm-gpu-memory-utilization",
        str(SMOKE_GPU_MEM_UTIL),
        "--output-json",
        str(report_path),
    ]
    if smiles_class:
        cmd.extend(["--smiles-classes", smiles_class])
    # Split name is required for gsm/spider (no default side); smoke is a
    # sanity gate, so score the train side — test is reserved for final numbers.
    if dataset == "gsm_symbolic":
        cmd.extend(["--gsm-split-name", "train"])
    elif dataset == "spider":
        cmd.extend(["--spider-split-name", "train"])

    env = os.environ.copy()
    env["PYTHONPATH"] = str(code_root) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    env.setdefault("CSD_VLLM_GPU_MEMORY_UTILIZATION", str(SMOKE_GPU_MEM_UTIL))

    job = runner or _default_subprocess_runner
    rc = 1
    for try_idx in range(1, SMOKE_JOB_MAX_TRIES + 1):
        # Pin (not setdefault) to the awaited GPU: an inherited multi-GPU
        # value would let the eval pool land on a device without room for
        # the smoke engine.
        env["CUDA_VISIBLE_DEVICES"] = wait_for_smoke_gpu()
        logger.info(
            "smoke job start cell=%s kind=%s try=%s gpu=%s csd=%s out=%s cmd=%s",
            incident.cell_id,
            incident.path_kind,
            try_idx,
            env["CUDA_VISIBLE_DEVICES"],
            csd,
            out_dir,
            " ".join(cmd),
        )
        rc = int(job(cmd, cwd=code_root, env=env, out_dir=out_dir))
        if rc == 0 or not gpu_contention_in_log(out_dir):
            break
        logger.warning(
            "smoke job GPU contention cell=%s try=%s/%s rc=%s; retrying",
            incident.cell_id,
            try_idx,
            SMOKE_JOB_MAX_TRIES,
            rc,
        )

    incident.extra = dict(incident.extra or {})
    incident.extra["smoke_report_path"] = str(report_path)
    incident.extra["smoke_process_rc"] = rc
    incident.extra["smoke_attempt"] = incident.cloud_attempt_count
    incident.extra["smoke_out_dir"] = str(out_dir)
    if uniq_tokens is not None:
        incident.extra["smoke_uniq_tokens"] = int(uniq_tokens)

    # Path-kind-specific flags for non-harness smokes (telemetry/memory/helper).
    if incident.path_kind == PathKind.TELEMETRY.value:
        log_text = ""
        log_path = out_dir / "smoke.log"
        if log_path.is_file():
            log_text = log_path.read_text(encoding="utf-8", errors="replace")
        incident.extra["smoke_accuracy_line_present"] = "Accuracy:" in log_text or "accuracy:" in log_text
    if incident.path_kind == PathKind.MEMORY.value:
        log_text = ""
        log_path = out_dir / "smoke.log"
        if log_path.is_file():
            log_text = log_path.read_text(encoding="utf-8", errors="replace").lower()
        incident.extra["smoke_oom_present"] = any(
            sig in log_text
            for sig in (
                "cuda out of memory",
                "torch.outofmemoryerror",
                "out of memory",
                "oom-kill",
            )
        )

    metrics = parse_smoke_report(
        report_path,
        process_rc=rc,
        uniq_tokens=uniq_tokens,
        accuracy_line_present=incident.extra.get("smoke_accuracy_line_present"),
        oom_present=incident.extra.get("smoke_oom_present"),
        helper_failures_remaining=incident.extra.get("smoke_helper_failures_remaining"),
        twin_accuracy_pct=incident.extra.get("smoke_twin_accuracy_pct"),
    )
    decision = evaluate_smoke_metrics(
        cell_id=incident.cell_id,
        path_kind=incident.path_kind,
        metrics=metrics,
    )
    incident.extra["smoke_fail_reason"] = "" if decision.passed else decision.reason
    verdict = {
        "passed": decision.passed,
        "reason": decision.reason,
        "accuracy_pct": metrics.accuracy_pct,
        "unique_valid_count": metrics.unique_valid_count,
        "unique_valid_rate": metrics.unique_valid_rate,
        "uniq_tokens": metrics.uniq_tokens,
        "rc": rc,
    }
    (out_dir / "smoke_verdict.json").write_text(
        json.dumps(verdict, indent=2) + "\n", encoding="utf-8"
    )
    logger.info(
        "smoke job done cell=%s passed=%s reason=%s uv=%s acc=%s rc=%s",
        incident.cell_id,
        decision.passed,
        decision.reason,
        metrics.unique_valid_count,
        metrics.accuracy_pct,
        rc,
    )
    return decision


def run_twin_accuracy_probe(
    cell_id: str,
    *,
    live_repo: Path,
    runner: SmokeJobRunner | None = None,
    python_bin: str | None = None,
    sample_size: int = SMOKE_SAMPLE_SIZE,
) -> float | None:
    """Measure Tier-A twin accuracy: tiny re-eval of the newest GeneratedCSD
    through the live harness under the smoke model.

    Unlike run_hardened_smoke_job this never touches the repair worktree or
    any repair branch — it runs before an incident exists, purely to answer
    "can the live harness still produce a nonzero score?". Returns measured
    accuracy percent, or None when nothing could be measured (missing CSD,
    nonzero rc, unreadable report); callers map None to twin=0.
    """
    live = live_repo.resolve()
    out_dir = (
        live
        / "logs"
        / "zero_acc_babysitter"
        / f"twin_{cell_id}_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "twin_report.json"

    csd = find_latest_generated_csd(live, cell_id)
    if csd is None:
        logger.warning("twin probe: no GeneratedCSD for cell=%s", cell_id)
        return None

    dataset, smiles_class = cell_dataset_and_smiles_class(cell_id)
    max_steps = (
        SMOKE_MAX_STEPS_SMILES if dataset == "smiles" else SMOKE_MAX_STEPS_DEFAULT
    )
    py = python_bin or sys.executable
    cmd = [
        py,
        "-m",
        "synthesis.scripts.reevaluate_compiled_csd",
        str(csd),
        "--dataset",
        dataset,
        "--eval-model",
        SMOKE_MODEL,
        "--sample-size",
        str(sample_size),
        "--max-steps",
        str(max_steps),
        "--vllm-gpu-memory-utilization",
        str(SMOKE_GPU_MEM_UTIL),
        "--output-json",
        str(report_path),
    ]
    if smiles_class:
        cmd.extend(["--smiles-classes", smiles_class])
    if dataset == "gsm_symbolic":
        cmd.extend(["--gsm-split-name", "train"])
    elif dataset == "spider":
        cmd.extend(["--spider-split-name", "train"])

    env = os.environ.copy()
    env["PYTHONPATH"] = str(live) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    env.setdefault("CSD_VLLM_GPU_MEMORY_UTILIZATION", str(SMOKE_GPU_MEM_UTIL))

    job = runner or _default_subprocess_runner
    rc = 1
    for try_idx in range(1, SMOKE_JOB_MAX_TRIES + 1):
        env["CUDA_VISIBLE_DEVICES"] = wait_for_smoke_gpu()
        logger.info(
            "twin probe start cell=%s try=%s gpu=%s csd=%s out=%s cmd=%s",
            cell_id,
            try_idx,
            env["CUDA_VISIBLE_DEVICES"],
            csd,
            out_dir,
            " ".join(cmd),
        )
        rc = int(job(cmd, cwd=live, env=env, out_dir=out_dir))
        if rc == 0 or not gpu_contention_in_log(out_dir):
            break
        logger.warning(
            "twin probe GPU contention cell=%s try=%s/%s rc=%s; retrying",
            cell_id,
            try_idx,
            SMOKE_JOB_MAX_TRIES,
            rc,
        )
    metrics = parse_smoke_report(report_path, process_rc=rc)
    logger.info(
        "twin probe done cell=%s rc=%s acc=%s", cell_id, rc, metrics.accuracy_pct
    )
    if rc != 0:
        return None
    return metrics.accuracy_pct


def path_requires_twin(path_kind: str) -> bool:
    return path_kind in {PathKind.HARNESS.value, PathKind.TELEMETRY.value}
