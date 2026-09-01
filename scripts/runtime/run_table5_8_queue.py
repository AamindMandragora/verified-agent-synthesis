#!/usr/bin/env python3
"""Restart-safe manifest and runner for the missing paper Tables 5--8.

This module only constructs and validates the campaign.  It never contacts a
provider unless ``run`` is explicitly requested, and dry-run is the default.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import logging
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from synthesis.evaluate.benchmarks.gsm_symbolic.prompts import GSM_CRANE_COT_TASK
from synthesis.generate.pi_oauth import (
    PiBridgeFailure,
    PiBridgeTimeout,
    pi_oauth_probe as probe_pi_oauth,
    stored_pi_oauth_route,
)
from synthesis.run_constants import VLLM_GPU_MEMORY_UTILIZATION_BY_MODEL
from synthesis.source_snapshot import (
    execution_source_hashes,
    execution_source_paths,
    execution_source_sha256,
)

LOGGER = logging.getLogger("table5-8-queue")
cold_compiled_csd = None


def sha256_text(value: str) -> str:
    """Hash a short non-secret value for binding evidence without storing it."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def provider_pilots_sha256(pilots: dict[str, Any]) -> str:
    """Hash the embedded pilot object in one stable JSON representation."""
    return sha256_text(json.dumps(pilots, sort_keys=True, separators=(",", ":")))


def hash_file(path: Path) -> str:
    """Return the SHA-256 digest of a file without retaining its contents."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def utc_timestamp(epoch: float | None = None) -> str:
    """Return one stable UTC timestamp for persisted controller evidence."""
    moment = datetime.fromtimestamp(
        time.time() if epoch is None else float(epoch), tz=timezone.utc
    )
    return moment.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _nonnegative_seconds(value: Any, *, field: str) -> float:
    try:
        seconds = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"runtime field {field} is missing or invalid") from exc
    if not math.isfinite(seconds) or seconds < 0:
        raise ConfigError(f"runtime field {field} is missing or invalid")
    return round(seconds, 4)


def _persisted_epoch(state: dict[str, Any], field: str, default: float) -> float:
    """Reuse a phase start across controller restarts or initialize it once."""
    raw = state.get(field)
    if raw is None:
        return float(default)
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"runtime field {field} is invalid") from exc
    if not math.isfinite(value) or value <= 0:
        raise ConfigError(f"runtime field {field} is invalid")
    return value


def _runtime_evidence(
    state: dict[str, Any], report: dict[str, Any], heldout: dict[str, Any]
) -> dict[str, Any]:
    """Build a validated, artifact-storable timing summary for one row."""
    timestamp_fields = (
        "row_started_at",
        "synthesis_started_at",
        "synthesis_finished_at",
        "heldout_started_at",
        "heldout_finished_at",
        "row_finished_at",
    )
    runtime: dict[str, Any] = {}
    for field in timestamp_fields:
        value = state.get(field)
        if not isinstance(value, str):
            raise ConfigError(f"runtime field {field} is missing or invalid")
        try:
            datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ConfigError(f"runtime field {field} is missing or invalid") from exc
        runtime[field] = value
    for field in (
        "synthesis_wall_time_seconds",
        "heldout_wall_time_seconds",
        "total_wall_time_seconds",
    ):
        runtime[field] = _nonnegative_seconds(state.get(field), field=field)
    if runtime["total_wall_time_seconds"] + 0.01 < (
        runtime["synthesis_wall_time_seconds"]
            + runtime["heldout_wall_time_seconds"]
    ):
        raise ConfigError("total row runtime is shorter than its two phases")
    phase_coverage = state.get("phase_timing_coverage")
    if phase_coverage not in {"all_phases", "recovery_anchor"}:
        raise ConfigError("runtime field phase_timing_coverage is missing or invalid")
    runtime["phase_timing_coverage"] = phase_coverage

    total_attempts = report.get("total_attempts")
    attempts = report.get("attempts")
    attempt_times: list[float | None] = []
    coverage = "not_recorded"
    if isinstance(attempts, list) and type(total_attempts) is int:
        for attempt in attempts:
            evaluation = attempt.get("evaluation") if isinstance(attempt, dict) else None
            raw = evaluation.get("total_time_seconds") if isinstance(evaluation, dict) else None
            if isinstance(raw, (int, float)) and math.isfinite(float(raw)) and raw >= 0:
                attempt_times.append(round(float(raw), 4))
            else:
                attempt_times.append(None)
        coverage = (
            "all_attempts"
            if len(attempt_times) == total_attempts and all(value is not None for value in attempt_times)
            else "partial_attempts"
        )
    else:
        evaluation = report.get("evaluation_result")
        raw = evaluation.get("total_time_seconds") if isinstance(evaluation, dict) else None
        if isinstance(raw, (int, float)) and math.isfinite(float(raw)) and raw >= 0:
            attempt_times = [round(float(raw), 4)]
            coverage = "winning_attempt_only"
    runtime["attempt_evaluation_times_seconds"] = attempt_times
    runtime["attempt_timing_coverage"] = coverage

    metrics = heldout.get("metrics")
    metrics = metrics if isinstance(metrics, dict) else {}
    for source_key, output_key in (
        ("evaluator_total_time_seconds", "heldout_evaluator_total_time_seconds"),
        ("run_wall_time_seconds", "heldout_recorded_run_wall_time_seconds"),
    ):
        raw = metrics.get(source_key)
        runtime[output_key] = (
            None
            if raw is None
            else _nonnegative_seconds(raw, field=output_key)
        )
    return runtime


def start_logged_child(
    argv: list[str], *, cwd: Path, env: dict[str, str], log_path: Path
) -> subprocess.Popen:
    """Start one restart-safe child whose output remains in its row log."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("a", encoding="utf-8")
    try:
        process = subprocess.Popen(
            argv,
            cwd=cwd,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
    except Exception:
        handle.close()
        raise
    setattr(process, "_tableq_log_handle", handle)
    return process


def wait_logged_child(process: Any) -> tuple[Any, Any]:
    """Wait for a child and always release the controller's log handle."""
    try:
        return process.communicate()
    finally:
        handle = getattr(process, "_tableq_log_handle", None)
        if handle is not None:
            handle.close()


class ConfigError(ValueError):
    """The manifest or runtime configuration cannot be safely launched."""

EVAL_MODEL = "Qwen/Qwen3.5-2B"
CANONICAL_PYTHON = Path("/apps/conda/aadivyar/envs/csd/bin/python")
CANONICAL_GEMINI_ENV_FILE = Path("/home/aadivyar/csd-generation/synthesis/.env")
CANONICAL_PI_NODE_EXECUTABLE = Path(
    "/home/aadivyar/.local/share/cursor-agent/versions/2026.07.23-e383d2b/node"
)
CANONICAL_PI_BRIDGE_PATH = (
    Path(__file__).resolve().parents[2]
    / "synthesis/generate/pi_oauth/provider/bridge.mjs"
)
CANONICAL_PI_AUTH_PATH = Path("/home/aadivyar/.pi/csd-table5-8/auth.json")
CANONICAL_CLAUDE_CONFIG_DIR = Path("/home/aadivyar/.claude-csd-synthesis")
CANONICAL_CLAUDE_EXPECTED_ACCOUNT = "ssdear@gmail.com"
DISK_FIXED_SAFETY_BYTES = 2 * 1024**3
DISK_BYTES_PER_UNRESOLVED_ROW = 128 * 1024**2
GPU_SAFETY_MIB = 2_000
CANONICAL_CRANE_COMMIT = "616379ce33ac6245933c16e6264b41f7d5800183"
AUTHOR_TOKEN_BUDGET = 32768
AUTHOR_REASONING_BUDGET = 4096
BAR_BINDINGS = {
    "gsm_symbolic": {
        "min_accuracy": 13 / 49,
        "min_syntax_rate": 0.9,
        "source_path": "/home/aadivyar/csd-generation-worktrees/full-baseline-campaign-20260803/.context/claude_recovery_queue_0715/pending_manifest.json",
        "source_sha256": "b6e2c6e4cc22120ef59b6f40b19456a30beb5f52197ea8c690909653747c7e99",
    },
    "spider": {
        "min_accuracy": 59 / 300,
        "min_syntax_rate": 0.9,
        "source_path": "/home/aadivyar/csd-generation-worktrees/spider-u10a-reservation-20260825-luna/saved-results/2026-08-26-spider-evaluator-contract-u10c-manifest.json",
        "source_sha256": "eb669be8ce13c0412f61bbcfa6b5c630167ce30c89288f0e9e8ff7b9b3a41175",
    },
    "smiles": {
        "acrylates": {"min_accuracy": 0.14, "min_syntax_rate": 0.9},
        "chain_extenders": {"min_accuracy": 0.20, "min_syntax_rate": 0.9},
        "isocyanates": {"min_accuracy": 0.30, "min_syntax_rate": 0.9},
        "source_path": "/home/aadivyar/csd-generation-worktrees/full-baseline-campaign-20260803/saved-results/2026-08-05-corrected-full-baseline-cold-manifest.json",
        "source_sha256": "06c285b2c948c16d9d09b3473ed34ed08ff12ac7efd81bbaaf767d53a0a4d05c",
    },
}
SMILES_CLASSES = ("acrylates", "chain_extenders", "isocyanates")
TABLE5_PROFILES = {
    "gpt5.6-sol": {"generation_backend": "codex", "generation_model": "gpt-5.6-sol"},
    "gemini3.7-flash": {"generation_backend": "gemini", "generation_model": "gemini-3.7-flash"},
    "opus5": {"generation_backend": "claude", "generation_model": "claude-opus-5"},
}
CANONICAL_SPLITS = {
    "gsm_symbolic": "environment/benchmark_splits/gsm_symbolic_crane_proportional_49x49_seed123.json",
    "spider": "environment/benchmark_splits/spider_dev_proportional_300x300_seed334.json",
}
TASKS = {
    "gsm_symbolic": GSM_CRANE_COT_TASK,
    "spider": "Generate a single valid SQL query using only the provided schema context. Only output the SQL query.",
    "smiles": "Generate valid SMILES strings that match the requested molecular class while maintaining parser-valid output.",
}
DATASET_SETTINGS = {
    "gsm_symbolic": {"feedback": 49, "heldout": 49, "steps": 900},
    "spider": {"feedback": 300, "heldout": 300, "steps": 176},
    "smiles": {"feedback": 50, "heldout": 100, "steps": 400},
}
SOURCE_PATHS = (
    "scripts/runtime/run_table5_8_queue.py",
    "scripts/runtime/run_cold_synthesis_queue.py",
    "run_all_tests.py",
    "synthesis/run_synthesis.py",
    "synthesis/run_constants.py",
    "synthesis/source_snapshot.py",
    "synthesis/split_provenance.py",
    "synthesis/generate/generator.py",
    "synthesis/generate/provider_names.py",
    "synthesis/generate/pi_oauth/__init__.py",
    "synthesis/generate/pi_oauth/contract.py",
    "synthesis/generate/pi_oauth/provider/bridge.mjs",
    "synthesis/generate/pi_oauth/provider/package.json",
    "synthesis/generate/pi_oauth/provider/package-lock.json",
    "synthesis/evaluate/feedback_loop.py",
    "synthesis/evaluate/evaluator.py",
    "synthesis/evaluate/benchmarks/registry.py",
    "synthesis/scripts/reevaluate_compiled_csd.py",
    "synthesis/evaluate/benchmarks/gsm_symbolic/dataset.py",
    "synthesis/evaluate/benchmarks/gsm_symbolic/eval_logic.py",
    "synthesis/evaluate/benchmarks/sql_spider/dataset.py",
    "synthesis/evaluate/benchmarks/sql_spider/eval_logic.py",
    "synthesis/evaluate/benchmarks/smiles/dataset.py",
    "synthesis/evaluate/benchmarks/smiles/metrics.py",
    "synthesis/evaluate/benchmarks/smiles/eval_logic.py",
    "synthesis/evaluate/benchmarks/gsm_symbolic/prompts.py",
    "synthesis/evaluate/benchmarks/gsm_symbolic/generation.py",
    "synthesis/evaluate/benchmarks/smiles/generation.py",
    "synthesis/evaluate/benchmarks/sql_spider/generation.py",
    "synthesis/evaluate/benchmarks/sql_spider/output_contract.py",
    "synthesis/evaluate/benchmarks/smiles/environment.py",
    "synthesis/evaluate/benchmarks/gsm_symbolic/environment.py",
    "synthesis/evaluate/benchmarks/sql_spider/environment.py",
    "synthesis/evaluate/baseline_store.py",
    "synthesis/generate/prompts.py",
    "synthesis/verify/library/GeneratedCSD.dfy",
    "synthesis/verify/library/VerifiedAgentSynthesis.dfy",
    "synthesis/evaluate/benchmarks/common/model_utils.py",
    "environment/benchmark_splits/gsm_symbolic_crane_proportional_49x49_seed123.json",
    "environment/benchmark_splits/spider_dev_proportional_300x300_seed334.json",
    "synthesis/evaluate/benchmarks/common/parser_utils.py",
    "synthesis/verify/tooling.py",
    ".context/run_post14b_rebar_queue.py",
    "synthesis/evaluate/baselines/crane_repo_runner.py",
)

MANIFEST_KEYS = frozenset({"version", "git_commit", "crane_commit", "crane_source_sha256", "source_sha256", "execution_source_sha256", "external_runtime", "python_runtime", "jobs", "provider_pilots", "provider_pilot_sha256"})
JOB_KEYS = frozenset({
    "cell_id", "table", "table_cell_id", "paper_cells", "benchmark", "dataset", "task", "profile",
    "generation_backend", "generation_model", "eval_model", "synthesis_max_tokens",
    "synthesis_reasoning_budget",
    "effective_output_tokens", "effective_thinking_tokens",
    "smiles_class", "token_budget", "beam_size", "adaptive_helper_mask",
    "helper_selection_policy", "max_iterations", "min_accuracy", "min_syntax_rate",
    "bar_source_path", "bar_source_sha256", "eval_sample_size", "heldout_sample_size",
    "eval_max_steps", "eval_max_seconds", "gpu_mem_util", "memory_reservation_mib",
    "gpu_scope", "gpu_count", "heldout_split_name", "heldout_split_file", "sample_count",
    "output_name", "heldout_output_json", "log_file", "cold_start", "git_commit",
    "launch_commit", "expected_author_route",
})


def _row(cell_id: str, table: int, benchmark: str, profile: str, *, smiles_class: str | None = None, **controls: Any) -> dict[str, Any]:
    settings = DATASET_SETTINGS[benchmark]
    author = TABLE5_PROFILES[profile]
    sample = settings["feedback"]
    table_cell_id = controls.pop("table_cell_id", cell_id)
    paper_cells = controls.pop(
        "paper_cells", [{"table": table, "table_cell_id": table_cell_id}]
    )
    return {
        "cell_id": cell_id,
        "table": table,
        "table_cell_id": table_cell_id,
        "paper_cells": paper_cells,
        "benchmark": benchmark,
        "dataset": benchmark,
        "task": TASKS[benchmark],
        "profile": profile,
        "generation_backend": author["generation_backend"],
        "generation_model": author["generation_model"],
        "eval_model": EVAL_MODEL,
        "synthesis_max_tokens": AUTHOR_TOKEN_BUDGET,
        "synthesis_reasoning_budget": AUTHOR_REASONING_BUDGET,
        "effective_output_tokens": {"opus5": 64000, "gpt5.6-sol": None, "gemini3.7-flash": 32768}[profile],
        "effective_thinking_tokens": {"opus5": 48000, "gpt5.6-sol": None, "gemini3.7-flash": None}[profile],
        "smiles_class": smiles_class,
        "token_budget": controls.pop("token_budget", 1),
        "beam_size": controls.pop("beam_size", 2),
        "adaptive_helper_mask": controls.pop("adaptive_helper_mask", True),
        "helper_selection_policy": controls.pop("helper_selection_policy", "bandit"),
        "max_iterations": 40,
        "min_accuracy": (BAR_BINDINGS[benchmark][smiles_class]["min_accuracy"] if benchmark == "smiles" else BAR_BINDINGS[benchmark]["min_accuracy"]),
        "min_syntax_rate": (BAR_BINDINGS[benchmark][smiles_class]["min_syntax_rate"] if benchmark == "smiles" else BAR_BINDINGS[benchmark]["min_syntax_rate"]),
        "bar_source_path": BAR_BINDINGS[benchmark]["source_path"],
        "bar_source_sha256": BAR_BINDINGS[benchmark]["source_sha256"],
        "eval_sample_size": sample,
        "heldout_sample_size": settings["heldout"],
        "eval_max_steps": settings["steps"],
        "eval_max_seconds": 600.0,
        "gpu_mem_util": float(VLLM_GPU_MEMORY_UTILIZATION_BY_MODEL[EVAL_MODEL]),
        "memory_reservation_mib": 20_480,
        "gpu_scope": [0, 1, 2, 3],
        "gpu_count": 1,
        "heldout_split_name": "test",
        "heldout_split_file": CANONICAL_SPLITS.get(benchmark),
        "sample_count": settings["heldout"],
        "output_name": f"table5_8_{cell_id}",
        "heldout_output_json": f"outputs/reeval/table5_8/{cell_id}.json",
        "log_file": f"outputs/generated/table5_8_{cell_id}/run.log",
        "cold_start": True,
    }


def build_scope(repo: Path) -> list[dict[str, Any]]:
    """Return eight cold runs that populate eleven paper cells."""
    rows: list[dict[str, Any]] = []
    for profile in TABLE5_PROFILES:
        cell = f"t5-{profile}-gsm_symbolic"
        paper_cells = None
        if profile == "opus5":
            paper_cells = [
                {"table": 5, "table_cell_id": "table5-opus5-gsm_symbolic"},
                {"table": 6, "table_cell_id": "t6-opus5-gsm_symbolic-b1-B2-m1"},
                {"table": 7, "table_cell_id": "t7-opus5-gsm_symbolic-b1-B2-m1"},
                {"table": 8, "table_cell_id": "t8-opus5-gsm_symbolic-b1-B2-m1"},
            ]
        rows.append(
            _row(
                cell,
                5,
                "gsm_symbolic",
                profile,
                table_cell_id=f"table5-{profile}-gsm_symbolic",
                **({"paper_cells": paper_cells} if paper_cells is not None else {}),
            )
        )
    for table, settings in (
        (6, [(2, 2, True), (4, 2, True)]),
        (7, [(1, 1, True), (1, 4, True)]),
        (8, [(1, 2, False)]),
    ):
        for token_budget, beam_size, mask in settings:
            cell = f"t{table}-opus5-gsm_symbolic-b{token_budget}-B{beam_size}-m{int(mask)}"
            rows.append(
                _row(
                    cell,
                    table,
                    "gsm_symbolic",
                    "opus5",
                    table_cell_id=cell,
                    token_budget=token_budget,
                    beam_size=beam_size,
                    adaptive_helper_mask=mask,
                )
            )
    return rows


def synthesis_command(row: dict[str, Any], python: Path) -> list[str]:
    cmd = [str(python), "-m", "synthesis.run_synthesis", "--task", row["task"], "--dataset", row["dataset"], "--min-accuracy", str(row["min_accuracy"]), "--min-syntax-rate", str(row["min_syntax_rate"]), "--max-iterations", "40", "--eval-model", EVAL_MODEL, "--eval-sample-size", str(row["eval_sample_size"]), "--eval-max-steps", str(row["eval_max_steps"]), "--eval-step-token-budget", str(row["token_budget"]), "--eval-max-seconds-per-example", "600", "--eval-min-examples-before-threshold-stop", str(row["eval_sample_size"]), "--generation-model", row["generation_model"], "--generation-backend", row["generation_backend"], "--synthesis-max-tokens", str(row["synthesis_max_tokens"]), "--synthesizer-reasoning-budget", str(row["synthesis_reasoning_budget"]), "--device", "auto", "--vllm-gpu-memory-utilization", str(row["gpu_mem_util"]), "--refinement-beam-size", str(row["beam_size"]), "--helper-selection-policy", row["helper_selection_policy"]]
    cmd.append("--adaptive-helper-mask" if row["adaptive_helper_mask"] else "--no-adaptive-helper-mask")
    if row["dataset"] == "smiles":
        cmd += ["--smiles-classes", row["smiles_class"], "--smiles-samples-per-class", str(row["eval_sample_size"]), "--smiles-final-samples-per-class", str(row["heldout_sample_size"])]
    return cmd


def weighted_smiles_rate(values: Iterable[dict[str, Any]]) -> float:
    values = list(values)
    total = sum(int(v["sample_count"]) for v in values)
    if total <= 0:
        raise ValueError("SMILES aggregate needs positive sample counts")
    return sum(float(v["unique_valid_rate"]) * int(v["sample_count"]) for v in values) / total


def constrained_window_rate(value: dict[str, Any]) -> float:
    """Return validated mean constrained work for a held-out result."""
    metrics = value.get("metrics")
    work = metrics.get("mean_constrained_work") if isinstance(metrics, dict) else None
    if not isinstance(work, (int, float)) or work < 0:
        raise ConfigError("result is missing validated mean_constrained_work for CW")
    return float(work)


RUNTIME_EXPORT_KEYS = frozenset(
    {
        "row_started_at",
        "synthesis_started_at",
        "synthesis_finished_at",
        "synthesis_wall_time_seconds",
        "heldout_started_at",
        "heldout_finished_at",
        "heldout_wall_time_seconds",
        "row_finished_at",
        "total_wall_time_seconds",
        "phase_timing_coverage",
        "attempt_evaluation_times_seconds",
        "attempt_timing_coverage",
        "heldout_evaluator_total_time_seconds",
        "heldout_recorded_run_wall_time_seconds",
    }
)


def validated_runtime(value: dict[str, Any], *, cell_id: str) -> dict[str, Any]:
    """Validate the timing summary embedded in a held-out artifact."""
    runtime = value.get("runtime")
    if not isinstance(runtime, dict) or set(runtime) != RUNTIME_EXPORT_KEYS:
        raise ConfigError(f"missing runtime evidence for {cell_id}")
    for field in (
        "row_started_at",
        "synthesis_started_at",
        "synthesis_finished_at",
        "heldout_started_at",
        "heldout_finished_at",
        "row_finished_at",
    ):
        raw = runtime.get(field)
        if not isinstance(raw, str):
            raise ConfigError(f"missing runtime evidence for {cell_id}")
        try:
            datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ConfigError(f"missing runtime evidence for {cell_id}") from exc
    for field in (
        "synthesis_wall_time_seconds",
        "heldout_wall_time_seconds",
        "total_wall_time_seconds",
    ):
        _nonnegative_seconds(runtime.get(field), field=field)
    if runtime.get("phase_timing_coverage") not in {
        "all_phases",
        "recovery_anchor",
    }:
        raise ConfigError(f"missing runtime evidence for {cell_id}")
    attempt_times = runtime.get("attempt_evaluation_times_seconds")
    if not isinstance(attempt_times, list) or any(
        item is not None
        and (
            not isinstance(item, (int, float))
            or not math.isfinite(float(item))
            or item < 0
        )
        for item in attempt_times
    ):
        raise ConfigError(f"missing runtime evidence for {cell_id}")
    if runtime.get("attempt_timing_coverage") not in {
        "all_attempts",
        "partial_attempts",
        "winning_attempt_only",
        "not_recorded",
    }:
        raise ConfigError(f"missing runtime evidence for {cell_id}")
    for field in (
        "heldout_evaluator_total_time_seconds",
        "heldout_recorded_run_wall_time_seconds",
    ):
        if runtime.get(field) is not None:
            _nonnegative_seconds(runtime[field], field=field)
    return dict(runtime)


def provider_pilot_from_report(
    path: Path,
    *,
    profile: str,
    git_commit: str,
    environment: dict[str, str],
) -> dict[str, Any]:
    """Build manifest evidence only from a real, fully evaluated pilot report."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError("provider pilot report is missing or invalid") from exc
    expected_routes = {
        "gpt5.6-sol": ("codex", "gpt-5.6-sol", None, None),
        "gemini3.7-flash": ("gemini", "gemini-3.7-flash", 32768, None),
        "opus5": ("claude", "claude-opus-5", 64000, 48000),
    }
    if profile not in expected_routes:
        raise ConfigError(f"unknown provider pilot profile: {profile}")
    route = payload.get("run_configuration")
    attempts = payload.get("attempts")
    production_success_report = False
    if isinstance(attempts, list) and len(attempts) == 1:
        attempt = attempts[0]
    elif (
        path.name == "success_report.json"
        and isinstance(payload.get("evaluation_result"), dict)
        and payload.get("sample_outputs")
        == payload["evaluation_result"].get("sample_outputs")
    ):
        production_success_report = True
        attempt = {
            "attempt_number": 1,
            "strategy_code": payload.get("strategy_code"),
            "verification": {"success": True},
            "compilation": {
                "success": True,
                "output_dir": payload.get("compiled_dir"),
            },
            "evaluation": payload.get("evaluation_result"),
        }
    else:
        raise ConfigError("provider pilot must be a one-attempt synthesis report")
    if not isinstance(route, dict) or not isinstance(attempt, dict):
        raise ConfigError("provider pilot must be a one-attempt synthesis report")
    if payload.get("total_attempts") != 1 or route.get("max_iterations") != 1:
        raise ConfigError("provider pilot must use exactly one attempt")
    if route.get("git_commit") != git_commit:
        raise ConfigError("provider pilot report is bound to a different code commit")
    source_digest = route.get("execution_source_sha256")
    if not isinstance(source_digest, str) or not re.fullmatch(r"[0-9a-f]{64}", source_digest):
        raise ConfigError("provider pilot report has no source snapshot")
    pilot_python_runtime = route.get("python_runtime")
    if (
        not isinstance(pilot_python_runtime, dict)
        or set(pilot_python_runtime)
        != {
            "executable",
            "python_version",
            "implementation",
            "package_count",
            "packages_sha256",
        }
        or re.fullmatch(
            r"[0-9a-f]{64}",
            str(pilot_python_runtime.get("packages_sha256")),
        )
        is None
    ):
        raise ConfigError("provider pilot report has no Python runtime binding")

    author = route.get("author_model") or {}
    reported_author_route = author.get("route")
    required_author_route = expected_author_route(profile, environment)
    if reported_author_route != required_author_route:
        raise ConfigError("provider pilot report has the wrong author route identity")
    evaluation_config = route.get("evaluation") or {}
    controls = route.get("synthesis_controls") or {}
    backend, model, effective_output, effective_thinking = expected_routes[profile]
    try:
        expected_config = (
            author.get("backend") == backend
            and author.get("model") == model
            and author.get("max_new_tokens") == AUTHOR_TOKEN_BUDGET
            and author.get("reasoning_budget_tokens") == AUTHOR_REASONING_BUDGET
            and route.get("task_description") == TASKS["gsm_symbolic"]
            and evaluation_config.get("dataset") == "gsm_symbolic"
            and evaluation_config.get("eval_model") == EVAL_MODEL
            and evaluation_config.get("eval_sample_size") == 1
            and evaluation_config.get("eval_max_steps") == DATASET_SETTINGS["gsm_symbolic"]["steps"]
            and evaluation_config.get("eval_step_token_budget") == 1
            and float(evaluation_config.get("eval_max_seconds_per_example", -1)) == 600.0
            and evaluation_config.get("min_examples_before_threshold_stop") == 1
            and evaluation_config.get("smiles_classes") in (None, [])
            and controls.get("adaptive_helper_mask") is True
            and controls.get("helper_selection_policy") == "bandit"
            and controls.get("refinement_beam_size") == 2
        )
    except (TypeError, ValueError):
        expected_config = False
    if not expected_config:
        raise ConfigError("provider pilot report has the wrong route or controls")

    verification = attempt.get("verification") or {}
    compilation = attempt.get("compilation") or {}
    evaluation = attempt.get("evaluation") or {}
    strategy_code = attempt.get("strategy_code")
    sample_outputs = evaluation.get("sample_outputs")
    try:
        accuracy = float(evaluation.get("accuracy"))
        syntax_rate = float(evaluation.get("syntax_rate"))
    except (TypeError, ValueError) as exc:
        raise ConfigError("provider pilot evaluation metrics are missing") from exc
    if (
        attempt.get("attempt_number") != 1
        or not isinstance(strategy_code, str)
        or not strategy_code.strip()
        or verification.get("success") is not True
        or compilation.get("success") is not True
        or evaluation.get("success") is not True
        or evaluation.get("early_stopped") is not False
        or evaluation.get("num_examples") != 1
        or not isinstance(sample_outputs, list)
        or len(sample_outputs) != 1
        or not isinstance(sample_outputs[0], dict)
        or not isinstance(sample_outputs[0].get("actual"), str)
        or not sample_outputs[0]["actual"].strip()
        or type(sample_outputs[0].get("is_correct")) is not bool
        or type(sample_outputs[0].get("is_syntax_valid")) is not bool
        or not math.isfinite(accuracy)
        or not math.isfinite(syntax_rate)
        or not 0.0 <= accuracy <= 1.0
        or not 0.0 <= syntax_rate <= 1.0
    ):
        raise ConfigError("provider pilot must reach successful verification and evaluation")

    run_root = path.parent.parent
    if production_success_report:
        dafny_file = Path(str(payload.get("dafny_file") or ""))
        canonical_dafny = Path(str(payload.get("dafny_file_canonical") or ""))
        try:
            dafny_file.resolve().relative_to((run_root / "dafny").resolve())
            canonical_dafny.resolve().relative_to((run_root / "dafny").resolve())
        except (OSError, ValueError) as exc:
            raise ConfigError("provider pilot verified Dafny evidence is outside its run") from exc
        if (
            path.resolve() != (run_root / "results" / "success_report.json").resolve()
            or not dafny_file.is_file()
            or not canonical_dafny.is_file()
        ):
            raise ConfigError("provider pilot verified Dafny evidence is missing or unbound")
    compiled_dir = Path(str(compilation.get("output_dir") or ""))
    compiled_csd = compiled_dir / "GeneratedCSD.py"
    output_name = route.get("output_name")
    try:
        compiled_dir.resolve().relative_to((run_root / "python").resolve())
    except (OSError, ValueError) as exc:
        raise ConfigError("provider pilot compiled artifact is outside its run") from exc
    if (
        not isinstance(output_name, str)
        or not output_name
        or (
            run_root.name != output_name
            and not run_root.name.startswith(f"{output_name}_")
        )
        or compiled_dir.name != output_name
        or not compiled_csd.is_file()
    ):
        raise ConfigError("provider pilot compiled artifact is missing or unbound")

    created_at = payload.get("timestamp")
    if not isinstance(created_at, str):
        raise ConfigError("provider pilot report has no timestamp")
    pilot = {
        "status": "ready",
        "git_commit": git_commit,
        "execution_source_sha256": source_digest,
        "python_runtime": pilot_python_runtime,
        "profile": profile,
        "backend": backend,
        "model": model,
        "attempt_count": 1,
        "synthesis_status": "success",
        "verification_status": "success",
        "evaluation_status": "success",
        "response_sha256": sha256_text(strategy_code),
        "evidence_path": str(path.resolve()),
        "evidence_sha256": hash_file(path),
        "compiled_csd_sha256": hash_file(compiled_csd),
        "created_at": created_at,
        "effective_output_tokens": effective_output,
        "effective_thinking_tokens": effective_thinking,
    }
    if profile == "opus5":
        pilot.update(
            {
                "config_dir": reported_author_route["config_dir"],
                "expected_account": reported_author_route["expected_account"],
            }
        )
    if profile == "gemini3.7-flash":
        key_sha = reported_author_route.get("api_key_sha256")
        if not isinstance(key_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", key_sha):
            raise ConfigError("Gemini provider pilot has no bound API key fingerprint")
        pilot["api_key_sha256"] = key_sha
    return pilot


def codex_auth_probe(environment: dict[str, str] | None = None) -> dict[str, Any]:
    """Verify Pi's exact ChatGPT/Codex OAuth route without an author prompt."""
    inherited = dict(os.environ if environment is None else environment)
    try:
        return probe_pi_oauth(
            node_executable=inherited.get("CSD_PI_NODE_EXECUTABLE"),
            bridge_path=inherited.get("CSD_PI_BRIDGE_PATH"),
            auth_path=inherited.get("CSD_PI_AUTH_PATH"),
            timeout_seconds=90,
        )
    except (PiBridgeFailure, PiBridgeTimeout, OSError, ValueError) as exc:
        return {
            "status": "blocked",
            "reason": type(exc).__name__,
        }


def claude_auth_probe(environment: dict[str, str]) -> dict[str, Any]:
    """Check the exact first-party Max account without sending a prompt."""
    executable = environment.get("CSD_CLAUDE_EXECUTABLE", "claude")
    checked = dict(environment)
    checked["CLAUDE_CONFIG_DIR"] = str(CANONICAL_CLAUDE_CONFIG_DIR)
    try:
        result = subprocess.run(
            [executable, "auth", "status", "--json"],
            env=checked,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"status": "blocked", "reason": type(exc).__name__}
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return {"status": "blocked", "reason": "invalid auth status"}
    exact = (
        result.returncode == 0
        and payload.get("loggedIn") is True
        and payload.get("email") == CANONICAL_CLAUDE_EXPECTED_ACCOUNT
        and payload.get("authMethod") == "claude.ai"
        and payload.get("apiProvider") == "firstParty"
        and str(payload.get("subscriptionType", "")).lower() == "max"
    )
    if not exact:
        return {"status": "blocked", "reason": "wrong Claude account or route"}
    return {
        "status": "ready",
        "account": CANONICAL_CLAUDE_EXPECTED_ACCOUNT,
        "config_dir": str(CANONICAL_CLAUDE_CONFIG_DIR),
    }


def gemini_api_key_probe(environment: dict[str, str]) -> dict[str, Any]:
    """Authenticate the exact AI Studio key and require Gemini 3.7 Flash."""
    api_key = environment.get("GEMINI_API_KEY", "")
    if not api_key:
        return {"status": "blocked", "reason": "missing Gemini API key"}
    request = urllib.request.Request(
        "https://generativelanguage.googleapis.com/v1beta/models?pageSize=1000",
        headers={"x-goog-api-key": api_key},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return {"status": "blocked", "reason": f"HTTP {exc.code}"}
    except (OSError, TimeoutError, urllib.error.URLError, json.JSONDecodeError) as exc:
        return {"status": "blocked", "reason": type(exc).__name__}
    names = {
        model.get("name")
        for model in payload.get("models", [])
        if isinstance(model, dict)
    }
    if "models/gemini-3.7-flash" not in names:
        return {"status": "blocked", "reason": "gemini-3.7-flash unavailable"}
    return {
        "status": "ready",
        "model": "gemini-3.7-flash",
        "api_key_sha256": sha256_text(api_key),
    }


def validate_provider_pilot(
    profile: str,
    pilot: Any,
    git_commit: str | None,
    *,
    repo: Path,
    environment: dict[str, str],
    require_freshness: bool = True,
) -> str | None:
    """Validate a one-attempt provider pilot bound to the exact code bytes."""
    if not isinstance(pilot, dict) or pilot.get("status") != "ready":
        return f"{profile} provider pilot is missing or not ready"
    if not git_commit or pilot.get("git_commit") != git_commit:
        return f"{profile} provider pilot is bound to a different code commit"
    expected = {
        "gpt5.6-sol": ("codex", "gpt-5.6-sol"),
        "gemini3.7-flash": ("gemini", "gemini-3.7-flash"),
        "opus5": ("claude", "claude-opus-5"),
    }
    backend, model = expected[profile]
    if pilot.get("backend") != backend or pilot.get("model") != model:
        return f"{profile} provider pilot has the wrong route"
    if profile == "opus5" and (
        pilot.get("config_dir") != str(CANONICAL_CLAUDE_CONFIG_DIR)
        or pilot.get("expected_account") != CANONICAL_CLAUDE_EXPECTED_ACCOUNT
    ):
        return "opus5 provider pilot has the wrong account or config"
    if pilot.get("attempt_count") != 1:
        return f"{profile} provider pilot must use exactly one attempt"
    if any(pilot.get(key) != "success" for key in (
        "synthesis_status", "verification_status", "evaluation_status"
    )):
        return f"{profile} provider pilot did not complete verification and evaluation"
    response_sha = pilot.get("response_sha256")
    if not isinstance(response_sha, str) or len(response_sha) != 64:
        return f"{profile} provider pilot response is not hash-bound"
    evidence_path = Path(str(pilot.get("evidence_path") or ""))
    evidence_sha = pilot.get("evidence_sha256")
    if (
        not evidence_path.is_file()
        or not isinstance(evidence_sha, str)
        or len(evidence_sha) != 64
        or hash_file(evidence_path) != evidence_sha
    ):
        return f"{profile} provider pilot evidence is missing or changed"
    try:
        report_root = (repo / "outputs" / "generated").resolve()
        resolved_evidence = evidence_path.resolve()
        resolved_evidence.relative_to(report_root)
    except (OSError, ValueError):
        return f"{profile} provider pilot evidence is outside this checkout"
    if evidence_path.name not in {"success_report.json", "failure_report.json"}:
        return f"{profile} provider pilot evidence is not a synthesis report"
    try:
        rebuilt = provider_pilot_from_report(
            evidence_path,
            profile=profile,
            git_commit=str(git_commit),
            environment=environment,
        )
    except ConfigError as exc:
        return f"{profile} provider pilot report is invalid: {exc}"
    if rebuilt != pilot:
        return f"{profile} provider pilot does not match its report"
    created_at = pilot.get("created_at")
    if not isinstance(created_at, str):
        return f"{profile} provider pilot has no timestamp"
    try:
        created = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        age = time.time() - created.astimezone(timezone.utc).timestamp()
    except (TypeError, ValueError, OverflowError):
        return f"{profile} provider pilot timestamp is invalid"
    if require_freshness and (age > 24 * 60 * 60 or age < -5 * 60):
        return f"{profile} provider pilot is stale"
    return None


def validate_startup_provider_pilots(
    rows: list[dict[str, Any]],
    provider_pilots: dict[str, Any],
    *,
    repo: Path,
    environment: dict[str, str],
    require_freshness: bool = True,
) -> None:
    """Require one fresh, exact pilot for every profile before polling starts."""
    checked: set[str] = set()
    for row in rows:
        profile = str(row["profile"])
        if profile in checked:
            continue
        reason = validate_provider_pilot(
            profile,
            provider_pilots.get(profile),
            row.get("git_commit"),
            repo=repo,
            environment=environment,
            require_freshness=require_freshness,
        )
        if reason is not None:
            raise ConfigError(f"{profile} startup provider pilot is invalid: {reason}")
        checked.add(profile)


def profile_block_reason(
    row: dict[str, Any],
    environment: dict[str, str],
    *,
    repo: Path,
    provider_pilots: dict[str, Any] | None = None,
    cached_probes: dict[str, dict[str, Any]] | None = None,
    cached_auth: dict[str, dict[str, Any]] | None = None,
    require_fresh_pilot: bool = True,
) -> str | None:
    """Return a durable pending reason, or None when this row may be admitted."""
    if row["profile"] == "gpt5.6-sol":
        probe = (cached_probes or {}).get("gpt5.6-sol") or codex_auth_probe(environment)
        if probe.get("status") != "ready":
            LOGGER.error("[tableq] auth-block profile=gpt5.6-sol reason=pi-oauth")
            return "Pi ChatGPT/Codex OAuth is unavailable or invalid"
    try:
        validate_profile_gates([row], environment)
    except ConfigError as exc:
        return str(exc)
    pilot_reason = validate_provider_pilot(
        row["profile"],
        (provider_pilots or {}).get(row["profile"]),
        row.get("git_commit"),
        repo=repo,
        environment=environment,
        require_freshness=require_fresh_pilot,
    )
    if pilot_reason:
        return pilot_reason
    if row["profile"] == "gemini3.7-flash":
        pilot = (provider_pilots or {}).get(row["profile"]) or {}
        if pilot.get("api_key_sha256") != sha256_text(environment.get("GEMINI_API_KEY", "")):
            return "gemini3.7-flash provider pilot is not bound to the active API key"
    if row.get("git_commit") and row["profile"] in {"opus5", "gemini3.7-flash"}:
        auth = (cached_auth or {}).get(row["profile"])
        if not auth or auth.get("status") != "ready":
            return f"{row['profile']} live authentication is unavailable"
        pilot = (provider_pilots or {}).get(row["profile"]) or {}
        if row["profile"] == "opus5" and (
            auth.get("account") != pilot.get("expected_account")
            or auth.get("config_dir") != pilot.get("config_dir")
        ):
            return "opus5 live authentication does not match the pilot route"
        if row["profile"] == "gemini3.7-flash" and (
            auth.get("model") != "gemini-3.7-flash"
            or auth.get("api_key_sha256") != pilot.get("api_key_sha256")
        ):
            return "gemini3.7-flash live authentication does not match the pilot route"
    return None


def partition_profile_readiness(
    rows: list[dict[str, Any]],
    environment: dict[str, str],
    *,
    repo: Path,
    provider_pilots: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Keep auth-blocked rows pending while allowing ready profiles to run."""
    ready: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []
    cached_probes: dict[str, dict[str, Any]] = {}
    cached_auth: dict[str, dict[str, Any]] = {}
    for row in rows:
        profile = row["profile"]
        if profile == "gpt5.6-sol" and profile not in cached_probes:
            cached_probes[profile] = codex_auth_probe(environment)
        if row.get("git_commit") and row["profile"] not in cached_auth:
            if row["profile"] == "opus5":
                cached_auth[row["profile"]] = claude_auth_probe(environment)
            elif row["profile"] == "gemini3.7-flash":
                cached_auth[row["profile"]] = gemini_api_key_probe(environment)
        reason = profile_block_reason(
            row,
            environment,
            repo=repo,
            provider_pilots=provider_pilots,
            cached_probes=cached_probes,
            cached_auth=cached_auth,
        )
        if reason is None:
            ready.append(row)
        else:
            blocked.append(dict(row, status="pending", reason=reason))
    return ready, blocked


def make_admission_guard(
    *,
    repo: Path,
    manifest_path: Path,
    expected_manifest_sha256: str,
    environment: dict[str, str],
    auth_ttl_seconds: float = 300.0,
    clock: Any = time.time,
) -> Any:
    """Revalidate frozen bytes, immutable pilot binding, and live auth."""
    cached_probes: dict[str, dict[str, Any]] = {}
    cached_auth: dict[str, dict[str, Any]] = {}
    checked_at: dict[str, float] = {}

    def guard(
        row: dict[str, Any], *, require_provider: bool = True
    ) -> None:
        disk_space_preflight(repo, unresolved_rows=1)
        try:
            manifest_bytes = manifest_path.read_bytes()
            payload = json.loads(manifest_bytes)
        except (OSError, json.JSONDecodeError) as exc:
            raise ConfigError("launch manifest is missing or invalid") from exc
        current_sha = hashlib.sha256(manifest_bytes).hexdigest()
        if current_sha != expected_manifest_sha256:
            raise ConfigError("launch manifest changed after controller validation")
        validated_rows = validate_manifest(repo, payload)
        candidate = next(
            (
                item
                for item in validated_rows
                if item.get("cell_id") == row.get("cell_id")
            ),
            None,
        )
        if candidate is None:
            raise ConfigError(f"launch row disappeared from manifest: {row.get('cell_id')}")

        if not require_provider:
            LOGGER.info(
                "[tableq] fresh-source-valid cell=%s provider-check=not-needed",
                row["cell_id"],
            )
            return

        profile = str(candidate["profile"])
        now = float(clock())
        if now - checked_at.get(profile, float("-inf")) >= auth_ttl_seconds:
            cached_probes.pop(profile, None)
            cached_auth.pop(profile, None)
            if profile == "gpt5.6-sol":
                cached_probes[profile] = codex_auth_probe(environment)
            elif profile == "opus5":
                cached_auth[profile] = claude_auth_probe(environment)
            elif profile == "gemini3.7-flash":
                cached_auth[profile] = gemini_api_key_probe(environment)
            checked_at[profile] = now
        reason = profile_block_reason(
            candidate,
            environment,
            repo=repo,
            provider_pilots=payload.get("provider_pilots"),
            cached_probes=cached_probes,
            cached_auth=cached_auth,
            require_fresh_pilot=False,
        )
        if reason is not None:
            raise ConfigError(f"fresh admission blocked for {row['cell_id']}: {reason}")
        LOGGER.info("[tableq] fresh-admission-valid cell=%s profile=%s", row["cell_id"], profile)

    return guard


def provider_preflight() -> list[dict[str, str]]:
    """Check only local configuration; never call a paid provider."""
    claude_dir = Path(
        os.environ.get("CSD_CLAUDE_CONFIG_DIR", str(CANONICAL_CLAUDE_CONFIG_DIR))
    )
    return [
        {"profile": "gpt5.6-sol", "backend": "codex", "status": "not_checked_without_provider_call"},
        {
            "profile": "gemini3.7-flash",
            "backend": "gemini",
            "status": "api_key_present" if os.environ.get("GEMINI_API_KEY") else "api_key_missing",
            "api_key_sha256": sha256_text(os.environ["GEMINI_API_KEY"]) if os.environ.get("GEMINI_API_KEY") else "",
        },
        {"profile": "opus5", "backend": "claude", "status": "config_present" if claude_dir.is_dir() else "config_missing"},
    ]


def _demand(row: dict[str, Any], total_mib: int) -> int:
    return max(int(row["memory_reservation_mib"]), math.ceil(float(row["gpu_mem_util"]) * total_mib))


def choose_gpu(row: dict[str, Any], snapshot: dict[int, dict[str, int]], reservations: dict[int, int], baseline: dict[int, dict[str, int]], allowed: tuple[int, ...]) -> int | None:
    scope = set(int(g) for g in row.get("gpu_scope", [])) & set(int(g) for g in allowed)
    for gpu in sorted(scope):
        info = snapshot.get(gpu) or baseline.get(gpu) or {}
        total = int(info.get("total_mib", 0))
        free = int(info.get("free_mib", 0))
        if total > 0 and free >= _demand(row, total) + GPU_SAFETY_MIB + int(reservations.get(gpu, 0)):
            return gpu
    return None


def choose_gpus(row: dict[str, Any], snapshot: dict[int, dict[str, int]], reservations: dict[int, int], baseline: dict[int, dict[str, int]], allowed: tuple[int, ...]) -> tuple[int, ...] | None:
    scope = set(int(g) for g in row.get("gpu_scope", [])) & set(int(g) for g in allowed)
    chosen: list[int] = []
    for gpu in sorted(scope):
        info = snapshot.get(gpu) or baseline.get(gpu) or {}
        total = int(info.get("total_mib", 0))
        free = int(info.get("free_mib", 0))
        required = _demand(row, total) + GPU_SAFETY_MIB + int(reservations.get(gpu, 0))
        if total > 0 and free >= required:
            chosen.append(gpu)
            if len(chosen) == int(row.get("gpu_count", 1)):
                return tuple(chosen)
    return None


def manifest_payload(repo: Path, rows: list[dict[str, Any]], provider_pilots: dict[str, Any] | None = None) -> dict[str, Any]:
    source_paths = execution_source_paths(repo)
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--", *source_paths],
        cwd=repo, check=True, capture_output=True, text=True,
    ).stdout.strip()
    if dirty:
        raise ConfigError("execution dependencies have uncommitted changes")
    validate_crane_checkout(repo)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True).stdout.strip()
    sources = execution_source_hashes(repo)
    source_digest = execution_source_sha256(repo)
    crane_sources = crane_source_hashes(repo)
    materialized = materialize_frozen_bar_sources(repo)
    bound_rows = [
        dict(
            row,
            git_commit=commit,
            launch_commit=commit,
            bar_source_path=materialized[row["benchmark"]],
            expected_author_route=expected_author_route(
                row["profile"], dict(os.environ)
            ),
        )
        for row in rows
    ]
    pilots = provider_pilots or {}
    for profile, pilot in pilots.items():
        if pilot.get("execution_source_sha256") != source_digest:
            raise ConfigError(
                f"{profile} provider pilot was not run from the current source bytes"
            )
    pilot_hash = provider_pilots_sha256(pilots)
    external_runtime = external_runtime_binding(dict(os.environ))
    python_runtime = python_runtime_fingerprint(CANONICAL_PYTHON, repo)
    for profile, pilot in pilots.items():
        if pilot.get("python_runtime") != python_runtime:
            raise ConfigError(
                f"{profile} provider pilot used a different Python runtime"
            )
    return {
        "version": 1,
        "git_commit": commit,
        "crane_commit": CANONICAL_CRANE_COMMIT,
        "crane_source_sha256": crane_sources,
        "source_sha256": sources,
        "execution_source_sha256": source_digest,
        "external_runtime": external_runtime,
        "python_runtime": python_runtime,
        "jobs": bound_rows,
        "provider_pilots": provider_pilots or {},
        "provider_pilot_sha256": pilot_hash,
    }


def validate_frozen_bar_sources() -> None:
    for benchmark, binding in BAR_BINDINGS.items():
        path = Path(binding["source_path"])
        expected_sha = binding["source_sha256"]
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha:
            raise ConfigError(f"frozen {benchmark} bar source is missing or changed")


def materialize_frozen_bar_sources(repo: Path) -> dict[str, str]:
    target_dir = repo / ".context" / "table5_8" / "bars"
    target_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    for benchmark, binding in BAR_BINDINGS.items():
        source = Path(binding["source_path"])
        if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest() != binding["source_sha256"]:
            raise ConfigError(f"frozen {benchmark} bar source is missing or changed")
        target = target_dir / f"{benchmark}.json"
        temp = target.with_suffix(".tmp")
        temp.write_bytes(source.read_bytes())
        temp.replace(target)
        paths[benchmark] = str(target.relative_to(repo))
    return paths


def validate_crane_checkout(repo: Path) -> None:
    crane = repo / "legacy" / "CRANE"
    if not (crane / ".git").exists() and not (crane / "HEAD").exists():
        raise ConfigError(f"isolated CRANE checkout is missing: {crane}")
    try:
        head = subprocess.run(["git", "-C", str(crane), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    except subprocess.CalledProcessError as exc:
        raise ConfigError("unable to read isolated CRANE checkout") from exc
    if head != CANONICAL_CRANE_COMMIT:
        raise ConfigError(f"CRANE checkout must be {CANONICAL_CRANE_COMMIT}, got {head}")
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=crane,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        raise ConfigError("isolated CRANE checkout has uncommitted or untracked files")


def crane_source_hashes(repo: Path) -> dict[str, str]:
    """Hash every tracked CRANE file so a clean but altered checkout is rejected."""
    crane = repo / "legacy" / "CRANE"
    names = subprocess.run(
        ["git", "-C", str(crane), "ls-files", "-z"],
        check=True,
        capture_output=True,
    ).stdout.decode("utf-8").split("\0")
    hashes: dict[str, str] = {}
    for name in names:
        if not name:
            continue
        path = crane / name
        if path.is_symlink():
            digest = sha256_text(f"symlink\0{os.readlink(path)}")
        elif path.is_file():
            digest = hash_file(path)
        else:
            raise ConfigError(f"tracked CRANE source is missing or unsupported: {name}")
        hashes[f"legacy/CRANE/{name}"] = digest
    return hashes


def expected_author_route(
    profile: str, environment: dict[str, str]
) -> dict[str, Any]:
    """Return the exact non-secret author identity a report must record."""
    if profile == "gpt5.6-sol":
        try:
            return stored_pi_oauth_route(
                node_executable=environment.get("CSD_PI_NODE_EXECUTABLE"),
                bridge_path=environment.get("CSD_PI_BRIDGE_PATH"),
                auth_path=environment.get("CSD_PI_AUTH_PATH"),
            )
        except ValueError as exc:
            raise ConfigError(
                "gpt5.6-sol requires a bound Pi ChatGPT/Codex OAuth route"
            ) from exc
    if profile == "opus5":
        return {
            "auth_mode": "claude_code_max",
            "config_dir": str(CANONICAL_CLAUDE_CONFIG_DIR),
            "expected_account": CANONICAL_CLAUDE_EXPECTED_ACCOUNT,
            "account_verified": True,
        }
    if profile == "gemini3.7-flash":
        api_key = environment.get("GEMINI_API_KEY", "")
        return {
            "auth_mode": "gemini_api_key",
            "api_key_sha256": sha256_text(api_key) if api_key else None,
        }
    raise ConfigError(f"unknown provider profile: {profile}")


def validate_manifest(repo: Path, payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Validate a manifest before any child process or provider is started."""
    if set(payload) != MANIFEST_KEYS:
        raise ConfigError("manifest contains unknown or missing top-level keys")
    if payload.get("version") != 1:
        raise ConfigError("manifest version must be exactly 1")
    validate_external_runtime_binding(
        payload.get("external_runtime"), dict(os.environ)
    )
    if payload.get("python_runtime") != python_runtime_fingerprint(
        CANONICAL_PYTHON, repo
    ):
        raise ConfigError("Python runtime differs from the manifest")
    if payload.get("crane_commit") != CANONICAL_CRANE_COMMIT:
        raise ConfigError("manifest is not bound to the approved CRANE checkout")
    validate_crane_checkout(repo)
    if payload.get("crane_source_sha256") != crane_source_hashes(repo):
        raise ConfigError("CRANE source bytes differ from the manifest")
    pilots = payload.get("provider_pilots")
    if not isinstance(pilots, dict):
        raise ConfigError("provider_pilots must be a JSON object")
    if payload.get("provider_pilot_sha256") != provider_pilots_sha256(pilots):
        raise ConfigError("embedded provider pilot evidence hash does not match")
    source_digest = payload.get("execution_source_sha256")
    if not isinstance(source_digest, str) or not re.fullmatch(r"[0-9a-f]{64}", source_digest):
        raise ConfigError("manifest has no execution source digest")
    for profile, pilot in pilots.items():
        if pilot.get("execution_source_sha256") != source_digest:
            raise ConfigError(
                f"{profile} provider pilot source does not match the manifest"
            )
        if pilot.get("python_runtime") != payload.get("python_runtime"):
            raise ConfigError(
                f"{profile} provider pilot Python runtime does not match the manifest"
            )
    rows = payload.get("jobs")
    if not isinstance(rows, list) or len(rows) != 8:
        raise ConfigError("manifest must contain exactly 8 Table 5--8 GSM jobs")
    expected = build_scope(repo)
    immutable_fields = {
        "cell_id", "table", "table_cell_id", "paper_cells", "benchmark", "dataset", "task",
        "profile", "generation_backend", "generation_model", "eval_model",
        "smiles_class", "token_budget", "beam_size", "adaptive_helper_mask",
        "helper_selection_policy", "max_iterations", "min_accuracy",
        "min_syntax_rate", "synthesis_max_tokens", "synthesis_reasoning_budget",
        "effective_output_tokens", "effective_thinking_tokens",
        "eval_sample_size",
        "heldout_sample_size", "eval_max_steps", "eval_max_seconds", "gpu_mem_util",
        "memory_reservation_mib", "gpu_scope", "gpu_count", "heldout_split_name",
        "heldout_split_file", "sample_count", "output_name", "heldout_output_json",
        "log_file", "cold_start", "bar_source_sha256",
    }
    for actual, frozen in zip(rows, expected):
        if (
            not JOB_KEYS.issubset(actual)
            or set(actual) - JOB_KEYS
        ):
            raise ConfigError(f"job has unknown or missing fields: {actual.get('cell_id', '<unknown>')}")
        for field in immutable_fields:
            if actual.get(field) != frozen.get(field):
                raise ConfigError(f"manifest field {field} differs for {frozen['cell_id']}")
        if actual.get("git_commit") != payload.get("git_commit"):
            raise ConfigError(f"row commit is not bound to manifest commit: {actual['cell_id']}")
        if actual.get("launch_commit") != payload.get("git_commit"):
            raise ConfigError(f"row launch commit is not bound to manifest commit: {actual['cell_id']}")
        if actual.get("expected_author_route") != expected_author_route(
            str(actual.get("profile")), dict(os.environ)
        ):
            raise ConfigError(
                f"row author route differs from current credentials: {actual['cell_id']}"
            )
        copied_bar = Path(str(actual.get("bar_source_path", "")))
        if copied_bar.is_absolute() or not (repo / copied_bar).is_file() or hashlib.sha256((repo / copied_bar).read_bytes()).hexdigest() != actual.get("bar_source_sha256"):
            raise ConfigError(f"bar source is not a copied immutable artifact: {actual['cell_id']}")
    source_paths = execution_source_paths(repo)
    recorded = payload.get("source_sha256")
    if not isinstance(recorded, dict) or set(recorded) != set(source_paths):
        raise ConfigError("manifest must hash every direct execution dependency")
    if sha256_text(json.dumps(recorded, sort_keys=True, separators=(",", ":"))) != source_digest:
        raise ConfigError("execution source digest does not match its file hashes")
    for relative in source_paths:
        path = repo / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != recorded[relative]:
            raise ConfigError(f"execution dependency changed: {relative}")
    dirty = subprocess.run(["git", "status", "--porcelain", "--", *source_paths], cwd=repo, check=True, capture_output=True, text=True).stdout.strip()
    if dirty:
        raise ConfigError("execution dependencies have uncommitted changes")
    if subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True).stdout.strip() != str(payload.get("git_commit")):
        raise ConfigError("manifest commit is not current HEAD")
    return rows


SAFE_RUNTIME_ENV_KEYS = frozenset(
    {
        "PATH",
        "HOME",
        "USER",
        "LOGNAME",
        "SHELL",
        "LANG",
        "LANGUAGE",
        "TZ",
        "LD_LIBRARY_PATH",
        "LIBRARY_PATH",
        "CPATH",
        "CUDA_HOME",
        "CUDA_PATH",
        "XDG_CACHE_HOME",
        "XDG_DATA_DIRS",
        "TMPDIR",
        "TMP",
        "TEMP",
        "SSL_CERT_FILE",
        "REQUESTS_CA_BUNDLE",
        "CURL_CA_BUNDLE",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "NO_PROXY",
        "http_proxy",
        "https_proxy",
        "no_proxy",
        "HF_HOME",
        "HF_CACHE",
        "TRANSFORMERS_CACHE",
        "SYNCODE_CACHE",
        "ITER_SYNCODE_CACHE",
        "TOKENIZERS_PARALLELISM",
        "CSD_CACHE_ROOT",
        "_CE_CONDA",
        "_CE_M",
    }
)
SAFE_RUNTIME_ENV_PREFIXES = ("CONDA_", "LC_")


def safe_runtime_environment(environment: dict[str, str]) -> dict[str, str]:
    """Copy only operating-system, Python, model-cache, and network settings."""
    return {
        key: value
        for key, value in environment.items()
        if key in SAFE_RUNTIME_ENV_KEYS
        or key.startswith(SAFE_RUNTIME_ENV_PREFIXES)
    }


def campaign_environment(
    environment: dict[str, str] | None = None,
    *,
    credential_file: Path = CANONICAL_GEMINI_ENV_FILE,
) -> dict[str, str]:
    """Install exact non-secret routes and load only the private Gemini key."""
    result = dict(os.environ if environment is None else environment)
    result.update(
        {
            "CSD_PI_NODE_EXECUTABLE": str(CANONICAL_PI_NODE_EXECUTABLE),
            "CSD_PI_BRIDGE_PATH": str(CANONICAL_PI_BRIDGE_PATH),
            "CSD_PI_AUTH_PATH": str(CANONICAL_PI_AUTH_PATH),
            "CSD_CLAUDE_CONFIG_DIR": str(CANONICAL_CLAUDE_CONFIG_DIR),
            "CSD_CLAUDE_EXPECTED_ACCOUNT": CANONICAL_CLAUDE_EXPECTED_ACCOUNT,
        }
    )
    if result.get("GEMINI_API_KEY"):
        return result
    try:
        lines = credential_file.read_text(encoding="utf-8").splitlines()
    except OSError:
        return result
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        name, separator, raw_value = line.partition("=")
        if separator != "=" or name.strip() != "GEMINI_API_KEY":
            continue
        try:
            parsed = shlex.split(raw_value, posix=True)
        except ValueError as exc:
            raise ConfigError("private GEMINI_API_KEY entry is malformed") from exc
        if len(parsed) != 1 or not parsed[0]:
            raise ConfigError("private GEMINI_API_KEY entry is empty or malformed")
        result["GEMINI_API_KEY"] = parsed[0]
        break
    return result


def runtime_data_paths(environment: dict[str, str]) -> dict[str, str]:
    """Resolve model and Spider data from the real login home before isolation."""
    login_home = Path(environment.get("HOME") or Path.home()).expanduser().resolve()
    hf_home = Path(
        environment.get("HF_HOME") or login_home / ".cache" / "huggingface"
    ).expanduser().resolve()
    return {
        "HF_HOME": str(hf_home),
        "HF_CACHE": str(hf_home),
        "TRANSFORMERS_CACHE": str(hf_home),
        "XDG_CACHE_HOME": str(hf_home.parent),
        "SPIDER_DATA_DIR": str(
            (login_home / "spider_data" / "spider_data").resolve()
        ),
    }


def python_runtime_fingerprint(python: Path, repo: Path) -> dict[str, Any]:
    """Read the exact interpreter and installed-package identity in JSON."""
    try:
        resolved = python.expanduser().resolve(strict=True)
    except OSError as exc:
        raise ConfigError(f"bound Python executable is missing: {python}") from exc
    if not os.access(resolved, os.X_OK):
        raise ConfigError(f"bound Python is not executable: {resolved}")
    try:
        completed = subprocess.run(
            [str(resolved), "-m", "synthesis.runtime_fingerprint"],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
            env=safe_runtime_environment(dict(os.environ)),
        )
        payload = json.loads(completed.stdout)
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        raise ConfigError("unable to fingerprint the bound Python runtime") from exc
    required = {
        "executable",
        "python_version",
        "implementation",
        "package_count",
        "packages_sha256",
    }
    if (
        not isinstance(payload, dict)
        or set(payload) != required
        or payload.get("executable") != str(resolved)
        or not isinstance(payload.get("python_version"), str)
        or not isinstance(payload.get("implementation"), str)
        or type(payload.get("package_count")) is not int
        or payload["package_count"] <= 0
        or re.fullmatch(r"[0-9a-f]{64}", str(payload.get("packages_sha256")))
        is None
    ):
        raise ConfigError("bound Python returned an invalid runtime fingerprint")
    return payload


def _directory_tree_binding(root: Path) -> dict[str, Any]:
    """Hash every regular file in a data tree in stable relative-path order."""
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise ConfigError(f"required runtime data directory is missing: {root}")
    digest = hashlib.sha256()
    file_count = 0
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        if path.is_symlink():
            raise ConfigError(f"runtime data tree contains a symbolic link: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hash_file(path).encode("ascii"))
        digest.update(b"\n")
        file_count += 1
    if file_count == 0:
        raise ConfigError(f"runtime data tree is empty: {root}")
    return {
        "path": str(root),
        "file_count": file_count,
        "sha256": digest.hexdigest(),
    }


def _model_snapshot_tree_binding(snapshot: Path, model_root: Path) -> dict[str, Any]:
    """Hash the bytes reached by every file in a Hugging Face snapshot."""
    snapshot = snapshot.resolve()
    blobs = (model_root / "blobs").resolve()
    digest = hashlib.sha256()
    file_count = 0
    for path in sorted(
        snapshot.rglob("*"), key=lambda item: item.relative_to(snapshot).as_posix()
    ):
        if path.is_dir():
            continue
        relative = path.relative_to(snapshot).as_posix()
        try:
            resolved = path.resolve(strict=True)
        except OSError as exc:
            raise ConfigError(f"Qwen snapshot contains a broken file: {path}") from exc
        if path.is_symlink():
            try:
                resolved.relative_to(blobs)
            except ValueError as exc:
                raise ConfigError(
                    f"Qwen snapshot link escapes the model cache: {path}"
                ) from exc
        elif snapshot != resolved.parent and snapshot not in resolved.parents:
            raise ConfigError(f"Qwen snapshot file escapes its root: {path}")
        if not resolved.is_file():
            raise ConfigError(f"Qwen snapshot entry is not a file: {path}")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hash_file(resolved).encode("ascii"))
        digest.update(b"\n")
        file_count += 1
    if file_count == 0:
        raise ConfigError(f"Qwen snapshot is empty: {snapshot}")
    return {"file_count": file_count, "sha256": digest.hexdigest()}


def external_runtime_binding(environment: dict[str, str]) -> dict[str, Any]:
    """Resolve the exact local Qwen snapshot and Spider bytes used by evaluation."""
    paths = runtime_data_paths(environment)
    model_root = (
        Path(paths["HF_HOME"])
        / "hub"
        / "models--Qwen--Qwen3.5-2B"
    )
    ref = model_root / "refs" / "main"
    try:
        revision = ref.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise ConfigError(f"Qwen cache ref is missing: {ref}") from exc
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ConfigError("Qwen cache ref is not an exact 40-hex revision")
    snapshot = (model_root / "snapshots" / revision).resolve()
    if not snapshot.is_dir() or snapshot.parent != (model_root / "snapshots").resolve():
        raise ConfigError(f"Qwen snapshot is missing: {snapshot}")
    snapshot_binding = _model_snapshot_tree_binding(snapshot, model_root)
    return {
        "eval_model": {
            "model": EVAL_MODEL,
            "revision": revision,
            "snapshot_path": str(snapshot),
            "snapshot_file_count": snapshot_binding["file_count"],
            "snapshot_sha256": snapshot_binding["sha256"],
        },
        "spider_data": _directory_tree_binding(Path(paths["SPIDER_DATA_DIR"])),
    }


def validate_external_runtime_binding(
    expected: Any, environment: dict[str, str]
) -> None:
    """Fail closed when local model or Spider inputs differ from the manifest."""
    if not isinstance(expected, dict) or set(expected) != {
        "eval_model",
        "spider_data",
    }:
        raise ConfigError("manifest has no exact external runtime binding")
    actual = external_runtime_binding(environment)
    if expected.get("eval_model") != actual["eval_model"]:
        raise ConfigError("Qwen model bytes or revision differ from the manifest")
    if expected.get("spider_data") != actual["spider_data"]:
        raise ConfigError("Spider data bytes differ from the manifest")


def validate_runtime_data_paths(environment: dict[str, str]) -> None:
    """Fail before provider or GPU work if pinned local data is unavailable."""
    paths = runtime_data_paths(environment)
    for key in ("HF_HOME", "SPIDER_DATA_DIR"):
        if not Path(paths[key]).is_dir():
            raise ConfigError(f"required runtime data directory is missing: {key}")


def disk_space_preflight(repo: Path, *, unresolved_rows: int) -> None:
    """Reserve two GiB plus 128 MiB for every unresolved campaign row."""
    if type(unresolved_rows) is not int or unresolved_rows < 0:
        raise ConfigError("unresolved row count must be a nonnegative integer")
    required = (
        DISK_FIXED_SAFETY_BYTES
        + DISK_BYTES_PER_UNRESOLVED_ROW * unresolved_rows
    )
    free = shutil.disk_usage(repo).free
    if free < required:
        raise ConfigError(
            "insufficient disk space for Table 5--8: "
            f"free={free} required={required} unresolved_rows={unresolved_rows}"
        )


def synthesis_environment(row: dict[str, Any], gpus: tuple[int, ...], inherited: dict[str, str], repo: Path) -> dict[str, str]:
    env = safe_runtime_environment(inherited)
    env.update(runtime_data_paths(inherited))
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["CSD_REDACT_SENSITIVE_LOGS"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu) for gpu in gpus)
    env["CSD_VLLM_GPU_MEMORY_UTILIZATION"] = str(row["gpu_mem_util"])
    env["CSD_VLLM_GPU_MEMORY_UTILIZATION_MAX"] = str(row["gpu_mem_util"])
    env["CSD_OUTPUT_DIR"] = str(
        repo / "outputs" / "generated" / str(row["output_name"])
    )
    env["CSD_OUTPUT_NAME"] = str(row["output_name"])
    if row["profile"] == "gpt5.6-sol":
        env["CSD_PI_NODE_EXECUTABLE"] = inherited.get(
            "CSD_PI_NODE_EXECUTABLE", str(CANONICAL_PI_NODE_EXECUTABLE)
        )
        env["CSD_PI_BRIDGE_PATH"] = inherited.get(
            "CSD_PI_BRIDGE_PATH", str(CANONICAL_PI_BRIDGE_PATH)
        )
        env["CSD_PI_AUTH_PATH"] = inherited.get(
            "CSD_PI_AUTH_PATH", str(CANONICAL_PI_AUTH_PATH)
        )
    if row["profile"] == "opus5":
        env["CSD_CLAUDE_CONFIG_DIR"] = str(CANONICAL_CLAUDE_CONFIG_DIR)
        env["CSD_CLAUDE_EXPECTED_ACCOUNT"] = CANONICAL_CLAUDE_EXPECTED_ACCOUNT
    if row["profile"] != "gpt5.6-sol":
        isolated_home = repo / ".context" / "table5_8" / f"{row['profile']}-home"
        isolated_home.mkdir(parents=True, exist_ok=True)
        env["HOME"] = str(isolated_home)
    if row["profile"] == "gemini3.7-flash":
        api_key = inherited.get("GEMINI_API_KEY")
        if api_key:
            env["GEMINI_API_KEY"] = api_key
        env["CSD_GEMINI_BACKEND"] = "gemini"
        env["CSD_GEMINI_MODEL"] = "gemini-3.7-flash"
    if row["dataset"] == "smiles":
        env["CSD_CONSTRAINED_TEMPERATURE"] = "0.7"
    return env


def heldout_environment(
    row: dict[str, Any],
    gpus: tuple[int, ...],
    inherited: dict[str, str],
    repo: Path,
) -> dict[str, str]:
    """Build the local evaluator environment without any author credential."""
    env = safe_runtime_environment(inherited)
    env.update(runtime_data_paths(inherited))
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["CSD_REDACT_SENSITIVE_LOGS"] = "1"
    isolated_home = repo / ".context" / "table5_8" / "heldout-home"
    isolated_home.mkdir(parents=True, exist_ok=True)
    env["HOME"] = str(isolated_home)
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu) for gpu in gpus)
    env["CSD_VLLM_GPU_MEMORY_UTILIZATION"] = str(row["gpu_mem_util"])
    env["CSD_VLLM_GPU_MEMORY_UTILIZATION_MAX"] = str(row["gpu_mem_util"])
    if row["dataset"] == "smiles":
        env["CSD_CONSTRAINED_TEMPERATURE"] = "0.7"
    return env


def heldout_command(row: dict[str, Any], python: Path, compiled_csd: Path) -> list[str]:
    cmd = [str(python), "-m", "synthesis.scripts.reevaluate_compiled_csd", str(compiled_csd), "--dataset", row["dataset"], "--eval-model", EVAL_MODEL, "--eval-backend", "vllm", "--device", "auto", "--sample-size", str(row["heldout_sample_size"]), "--max-steps", str(row["eval_max_steps"]), "--step-token-budget", str(row["token_budget"]), "--max-seconds-per-example", "600", "--vllm-gpu-memory-utilization", str(row["gpu_mem_util"]), "--vllm-tensor-parallel-size", "1", "--output-json", str(row["heldout_output_json"]), "--provenance-cell-id", str(row["cell_id"]), "--provenance-manifest-commit", str(row.get("manifest_commit") or row.get("git_commit") or "")]
    if row.get("eval_model_revision") and row.get("eval_model_snapshot_path"):
        cmd += [
            "--provenance-eval-model-revision",
            str(row["eval_model_revision"]),
            "--provenance-eval-model-snapshot-path",
            str(row["eval_model_snapshot_path"]),
            "--provenance-eval-model-snapshot-sha256",
            str(row["eval_model_snapshot_sha256"]),
            "--provenance-eval-model-snapshot-file-count",
            str(row["eval_model_snapshot_file_count"]),
        ]
    if row["dataset"] == "gsm_symbolic":
        cmd += ["--gsm-split-file", str(row["heldout_split_file"]), "--gsm-split-name", "test"]
    elif row["dataset"] == "spider":
        cmd += ["--spider-split-file", str(row["heldout_split_file"]), "--spider-split-name", "test", "--provenance-spider-data-path", str(row["spider_data_path"]), "--provenance-spider-data-sha256", str(row["spider_data_sha256"]), "--provenance-spider-data-file-count", str(row["spider_data_file_count"])]
    else:
        cmd += ["--smiles-classes", str(row["smiles_class"])]
    return cmd


def artifact_fingerprint(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    stat = path.stat()
    return {"inode": stat.st_ino, "mtime_ns": stat.st_mtime_ns, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def artifact_is_new_or_replaced(path: Path, before: dict[str, Any] | None) -> bool:
    after = artifact_fingerprint(path)
    return after is not None and after != before


def expected_heldout_indices(row: dict[str, Any]) -> list[int] | None:
    """Read the exact held-out index list from the manifest-bound split."""
    if row["dataset"] == "smiles":
        return None
    split_path = Path(str(row.get("heldout_split_file", "")))
    if not split_path.is_absolute():
        split_path = Path.cwd() / split_path
    try:
        split = json.loads(split_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    indices = split.get("test_indices")
    if not isinstance(indices, list) or len(indices) != int(row["heldout_sample_size"]):
        return None
    if any(type(index) is not int or index < 0 for index in indices):
        return None
    return indices


def heldout_artifact_is_valid(path: Path, row: dict[str, Any]) -> bool:
    """Validate the cold artifact while honoring this row's token budget."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        expected = int(row["heldout_sample_size"])
        metrics = payload.get("metrics") or {}
        answers = payload.get("answers")
        provenance = payload.get("reevaluation_provenance") or {}
        compiled = Path(str(provenance["compiled_csd_path"]))
        compiled_hash = hash_file(compiled)
        split = payload.get("eval_split") or {}
        prefix = "gsm" if row["dataset"] == "gsm_symbolic" else "spider"
        split_ok = row["dataset"] == "smiles" or (split.get(f"{prefix}_split_name") == "test" and str(split.get(f"{prefix}_split_file")) == str(row["heldout_split_file"]))
        indices = provenance.get("evaluated_source_indices")
        if not isinstance(indices, list) or len(indices) != expected or len(set(indices)) != expected:
            return False
        if not all(type(index) is int and index >= 0 for index in indices):
            return False
        expected_indices = expected_heldout_indices(row)
        if row["dataset"] != "smiles" and (expected_indices is None or indices != expected_indices):
            return False
        if not isinstance(answers, list) or any(
            not isinstance(answer, dict)
            or not isinstance(answer.get("generated_answer"), str)
            or not answer.get("generated_answer", "").strip()
            for answer in answers
        ):
            return False
        answer_indices = [answer.get("source_index") for answer in answers]
        if answer_indices != indices:
            return False
        generated_answers = [answer["generated_answer"].strip() for answer in answers]
        if (
            len(set(generated_answers)) == 1
            and float(payload.get("accuracy", -1)) == 0.0
            and float(payload.get("syntax_rate", -1)) == 0.0
        ):
            return False
        work_values = [answer.get("constrained_work") for answer in answers]
        if any(type(work) is not int or work < 0 for work in work_values):
            return False
        expected_total_work = sum(work_values)
        expected_mean_work = round(expected_total_work / expected, 4)
        if (
            type(metrics.get("total_constrained_work")) is not int
            or metrics.get("total_constrained_work") != expected_total_work
            or not isinstance(metrics.get("mean_constrained_work"), (int, float))
            or not math.isfinite(float(metrics["mean_constrained_work"]))
            or not math.isclose(
                float(metrics["mean_constrained_work"]),
                expected_mean_work,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            return False
        if metrics.get("examples_with_constrained_work") != expected:
            return False
        if row["dataset"] == "smiles":
            trial = payload.get("smiles_paper_trial")
            if not isinstance(trial, dict):
                return False
            unique_count = trial.get("unique_valid_count")
            if trial.get("sample_count") != expected or type(unique_count) is not int or not 0 <= unique_count <= expected:
                return False
            if "unique_valid_rate" in trial and not math.isclose(float(trial["unique_valid_rate"]), unique_count / expected, rel_tol=0.0, abs_tol=1e-12):
                return False
        correct_flags = [answer.get("is_correct") for answer in answers]
        syntax_flags = [answer.get("is_syntax_valid") for answer in answers]
        if any(type(flag) is not bool for flag in (*correct_flags, *syntax_flags)):
            return False
        expected_accuracy = (
            float(payload["smiles_paper_trial"]["unique_valid_count"]) / expected
            if row["dataset"] == "smiles"
            else sum(correct_flags) / expected
        )
        if not math.isclose(float(payload["accuracy"]), expected_accuracy, rel_tol=0.0, abs_tol=1e-12):
            return False
        if not math.isclose(float(payload["syntax_rate"]), sum(syntax_flags) / expected, rel_tol=0.0, abs_tol=1e-12):
            return False
        expected_manifest = row.get("manifest_commit") or row.get("git_commit")
        if expected_manifest and provenance.get("manifest_commit") != expected_manifest:
            return False
        bound_compiled = row.get("compiled_csd_path")
        if bound_compiled and Path(str(bound_compiled)).resolve() != compiled.resolve():
            return False
        expected_compiled_hash = row.get("compiled_sha256")
        if expected_compiled_hash and expected_compiled_hash != compiled_hash:
            return False
        if row.get("eval_model_revision") and provenance.get(
            "eval_model_revision"
        ) != row.get("eval_model_revision"):
            return False
        if row.get("eval_model_snapshot_path") and provenance.get(
            "eval_model_snapshot_path"
        ) != row.get("eval_model_snapshot_path"):
            return False
        if row.get("eval_model_snapshot_sha256") and (
            provenance.get("eval_model_snapshot_sha256")
            != row.get("eval_model_snapshot_sha256")
            or provenance.get("eval_model_snapshot_file_count")
            != row.get("eval_model_snapshot_file_count")
        ):
            return False
        if row["dataset"] == "spider" and row.get("spider_data_sha256"):
            if (
                provenance.get("spider_data_path") != row.get("spider_data_path")
                or provenance.get("spider_data_sha256")
                != row.get("spider_data_sha256")
                or provenance.get("spider_data_file_count")
                != row.get("spider_data_file_count")
            ):
                return False
        return (
            int(metrics.get("num_examples") or 0) == expected
            and isinstance(answers, list) and len(answers) == expected
            and isinstance(payload.get("accuracy"), (int, float))
            and isinstance(payload.get("syntax_rate"), (int, float))
            and provenance.get("cell_id") == row["cell_id"]
            and provenance.get("dataset") == row["dataset"]
            and provenance.get("eval_model") == row["eval_model"]
            and provenance.get("smiles_class") == row.get("smiles_class")
            and int(provenance.get("sample_size") or -1) == expected
            and int(provenance.get("max_steps") or -1) == int(row["eval_max_steps"])
            and int(provenance.get("step_token_budget") or -1) == int(row["token_budget"])
            and provenance.get("compiled_csd_sha256") == compiled_hash
            and split_ok
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def controller_manifest_path(input_path: Path, output_path: Path) -> Path:
    if input_path.resolve() == output_path.resolve():
        raise ConfigError("controller cannot overwrite its input manifest")
    return input_path


def validate_controller_paths(args: argparse.Namespace) -> None:
    """Reject output collisions before any GPU or provider work starts."""
    manifest = args.manifest.resolve()
    log = args.log.resolve()
    if log == manifest:
        raise ConfigError("controller log cannot overwrite its input manifest")
    if args.export is not None and args.export.resolve() in {manifest, log}:
        raise ConfigError("controller export must be separate from manifest and log")
    if not args.dry_run and args.export is None:
        raise ConfigError("--export is required for a real controller run")
    try:
        requested_python = args.python.expanduser().resolve(strict=True)
        canonical_python = CANONICAL_PYTHON.resolve(strict=True)
    except OSError as exc:
        raise ConfigError("the canonical Table 5--8 Python runtime is missing") from exc
    if requested_python != canonical_python:
        raise ConfigError(f"controller Python must be {canonical_python}")
    if (
        not args.gpus
        or len(set(args.gpus)) != len(args.gpus)
        or not set(args.gpus).issubset({0, 1, 2, 3})
    ):
        raise ConfigError("GPU scope must be a nonempty unique subset of 0,1,2,3")


def validate_controller_artifact_paths(
    args: argparse.Namespace, rows: list[dict[str, Any]], repo: Path
) -> None:
    """Reject any two campaign artifacts that resolve to the same file."""
    paths: list[tuple[str, Path]] = [
        ("manifest", args.manifest),
        ("controller log", args.log),
        ("controller state", args.state_dir / "controller.json"),
        ("state lock", lock_path(args.state_dir)),
        ("controller lock", controller_lock_path(repo)),
    ]
    if args.export is not None:
        paths.append(("export", args.export))
    for row in rows:
        cell = str(row["cell_id"])
        paths.extend(
            [
                (f"state for {cell}", _state_path(args.state_dir, row)),
                (f"held-out result for {cell}", repo / str(row["heldout_output_json"])),
                (f"worker log for {cell}", repo / str(row["log_file"])),
            ]
        )
    seen: dict[Path, str] = {}
    for label, path in paths:
        resolved = path.resolve()
        prior = seen.get(resolved)
        if prior is not None:
            raise ConfigError(
                f"artifact path collision: {label} and {prior} both use {resolved}"
            )
        seen[resolved] = label


def controller_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dispatch the validated Table 5--8 manifest")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--gpus", type=lambda raw: tuple(int(x) for x in raw.split(",") if x.strip()), default=(0, 1, 2, 3))
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--python", type=Path, default=CANONICAL_PYTHON)
    parser.add_argument("--nvidia-smi", default="nvidia-smi")
    parser.add_argument("--export", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def validate_profile_gates(rows: list[dict[str, Any]], environment: dict[str, str]) -> None:
    for row in rows:
        profile = row["profile"]
        LOGGER.info("[tableq] profile-gate profile=%s", profile)
        if profile == "opus5":
            if (
                environment.get("CSD_CLAUDE_CONFIG_DIR")
                != str(CANONICAL_CLAUDE_CONFIG_DIR)
                or environment.get("CSD_CLAUDE_EXPECTED_ACCOUNT")
                != CANONICAL_CLAUDE_EXPECTED_ACCOUNT
            ):
                raise ConfigError("opus5 requires the exact Max config directory and account")
        elif profile == "gpt5.6-sol":
            expected = {
                "CSD_PI_NODE_EXECUTABLE": str(CANONICAL_PI_NODE_EXECUTABLE),
                "CSD_PI_BRIDGE_PATH": str(CANONICAL_PI_BRIDGE_PATH),
                "CSD_PI_AUTH_PATH": str(CANONICAL_PI_AUTH_PATH),
            }
            for name, value in expected.items():
                if environment.get(name) != value:
                    raise ConfigError(f"gpt5.6-sol requires canonical {name}")
        elif profile == "gemini3.7-flash":
            if not environment.get("GEMINI_API_KEY"):
                LOGGER.error("[tableq] auth-block profile=gemini3.7-flash reason=missing-api-key")
                raise ConfigError("gemini3.7-flash requires GEMINI_API_KEY")
            conflicting = {
                key
                for key in environment
                if key in {
                    "GOOGLE_API_KEY",
                    "GOOGLE_APPLICATION_CREDENTIALS",
                    "GOOGLE_GENAI_USE_VERTEXAI",
                    "VERTEX_AI_PROJECT",
                    "VERTEX_AI_LOCATION",
                    "VERTEX_AI_BASE_URL",
                    "VERTEX_AI_API_KEY",
                    "VERTEX_AI_ACCESS_TOKEN",
                    "GOOGLE_CLOUD_PROJECT",
                    "GOOGLE_CLOUD_LOCATION",
                    "GOOGLE_VERTEX_LOCATION",
                }
                or key.startswith("GEMINI_API_KEY_BACKUP_")
            }
            if conflicting:
                LOGGER.error(
                    "[tableq] auth-block profile=gemini3.7-flash reason=conflicting-google-route"
                )
                raise ConfigError(
                    "gemini3.7-flash requires exactly one direct Gemini API key"
                )


def load_terminal_results(
    repo: Path, rows: list[dict[str, Any]], state_dir: Path
) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for row in rows:
        state = read_state(_state_path(state_dir, row))
        expected_path = (repo / str(row["heldout_output_json"])).resolve()
        if (
            not state
            or state.get("status") != "complete"
            or state.get("manifest_sha256") != row.get("manifest_sha256")
            or Path(str(state.get("heldout_output_json") or "")).resolve()
            != expected_path
        ):
            raise ConfigError(f"terminal state is incomplete or unbound: {row['cell_id']}")
        bound_row = dict(
            row,
            compiled_csd_path=state.get("compiled_csd_path"),
            compiled_sha256=state.get("compiled_sha256"),
            manifest_commit=state.get("manifest_commit"),
        )
        if (
            not expected_path.is_file()
            or state.get("heldout_sha256") != hash_file(expected_path)
            or not _report_binding_is_valid(state, row, repo)
            or not heldout_artifact_is_valid(expected_path, bound_row)
        ):
            raise ConfigError(
                f"held-out artifact is incomplete or unbound: {expected_path}"
            )
        LOGGER.info("[tableq] artifact-valid cell=%s", row["cell_id"])
        payload = json.loads(expected_path.read_text(encoding="utf-8"))
        payload["cell_id"] = row["cell_id"]
        payload.setdefault("sample_count", row["sample_count"])
        payload["paper_artifact_path"] = str(expected_path)
        payload["paper_artifact_sha256"] = hash_file(expected_path)
        payload["synthesis_report_path"] = state["synthesis_report_path"]
        payload["synthesis_report_sha256"] = state[
            "synthesis_report_sha256"
        ]
        report_path = Path(str(state["synthesis_report_path"]))
        report = json.loads(report_path.read_text(encoding="utf-8"))
        runtime = _runtime_evidence(state, report, payload)
        if payload.get("controller_runtime") != runtime:
            raise ConfigError(
                f"held-out runtime evidence is incomplete or unbound: {row['cell_id']}"
            )
        payload["runtime"] = runtime
        attempts = report.get("total_attempts")
        terminal_status = {
            "success_report.json": "accepted",
            "failure_report.json": "exhausted",
        }.get(report_path.name)
        if (
            type(attempts) is not int
            or not 1 <= attempts <= int(row["max_iterations"])
            or terminal_status is None
        ):
            raise ConfigError(
                f"synthesis attempt evidence is invalid: {row['cell_id']}"
            )
        payload["synthesis_attempts"] = attempts
        payload["synthesis_terminal_status"] = terminal_status
        payload["winning_attempt"] = state["winning_attempt"]
        LOGGER.info(
            "[tableq] synthesis-outcome cell=%s attempts=%s status=%s",
            row["cell_id"],
            attempts,
            terminal_status,
        )
        LOGGER.info(
            "[tableq] runtime cell=%s synthesis_seconds=%.4f heldout_seconds=%.4f total_seconds=%.4f",
            row["cell_id"],
            runtime["synthesis_wall_time_seconds"],
            runtime["heldout_wall_time_seconds"],
            runtime["total_wall_time_seconds"],
        )
        values.append(payload)
    return values


def controller_main(args: argparse.Namespace) -> int:
    validate_controller_paths(args)
    with controller_lock(Path.cwd()):
        return _controller_main_locked(args)


def _controller_main_locked(args: argparse.Namespace) -> int:
    repo = Path.cwd()
    manifest_bytes = args.manifest.read_bytes()
    payload = json.loads(manifest_bytes)
    rows = validate_manifest(repo, payload)
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    if args.dry_run:
        for row in rows:
            LOGGER.info("[tableq] dry-run cell=%s", row["cell_id"])
            print(row["cell_id"], shlex.join(synthesis_command(row, args.python)))
        return 0
    disk_space_preflight(repo, unresolved_rows=len(rows))
    validate_runtime_data_paths(dict(os.environ))
    validate_controller_artifact_paths(args, rows, repo)
    pilot_sha = str(payload.get("provider_pilot_sha256") or "")
    prior_controller = read_state(args.state_dir / "controller.json")
    resuming_validated_campaign = bool(
        prior_controller
        and prior_controller.get("status") in {"validated", "complete"}
        and prior_controller.get("manifest_sha256") == manifest_sha
        and prior_controller.get("provider_pilot_sha256") == pilot_sha
    )
    validate_startup_provider_pilots(
        rows,
        payload.get("provider_pilots") or {},
        repo=repo,
        environment=dict(os.environ),
        require_freshness=not resuming_validated_campaign,
    )
    admission_guard = make_admission_guard(
        repo=repo,
        manifest_path=args.manifest,
        expected_manifest_sha256=manifest_sha,
        environment=dict(os.environ),
    )
    args.state_dir.mkdir(parents=True, exist_ok=True)
    write_state(
        args.state_dir / "controller.json",
        {
            "manifest_sha256": manifest_sha,
            "provider_pilot_sha256": pilot_sha,
            "status": "validated",
            "scope": len(rows),
        },
    )
    logging.basicConfig(filename=args.log, level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    LOGGER.info("[tableq] input manifest sha256=%s scope=%d", manifest_sha, len(rows))
    from scripts.runtime.run_cold_synthesis_queue import gpu_memory_snapshot
    rows = [
        dict(
            row,
            manifest_sha256=manifest_sha,
            manifest_commit=payload["git_commit"],
            execution_source_sha256=payload["execution_source_sha256"],
            eval_model_revision=payload["external_runtime"]["eval_model"]["revision"],
            eval_model_snapshot_path=payload["external_runtime"]["eval_model"]["snapshot_path"],
            eval_model_snapshot_sha256=payload["external_runtime"]["eval_model"]["snapshot_sha256"],
            eval_model_snapshot_file_count=payload["external_runtime"]["eval_model"]["snapshot_file_count"],
            spider_data_path=payload["external_runtime"]["spider_data"]["path"],
            spider_data_sha256=payload["external_runtime"]["spider_data"]["sha256"],
            spider_data_file_count=payload["external_runtime"]["spider_data"]["file_count"],
        )
        for row in rows
    ]
    results = dispatch(
        rows,
        repo=repo,
        python=args.python,
        state_dir=args.state_dir,
        allowed=args.gpus,
        snapshot=lambda: gpu_memory_snapshot(args.nvidia_smi),
        poll_seconds=args.poll_seconds,
        admission_check=admission_guard,
    )
    if any(result.get("status") == "failed" for result in results):
        return 1
    values = load_terminal_results(repo, rows, args.state_dir)
    controller_manifest_path(args.manifest, args.export)
    export_results(rows, values, args.export)
    write_state(args.state_dir / "controller.json", {"manifest_sha256": manifest_sha, "provider_pilot_sha256": pilot_sha, "status": "complete", "scope": len(rows), "export": str(args.export)})
    return 0


def _state_path(state_dir: Path, row: dict[str, Any]) -> Path:
    return state_dir / f"{row['cell_id']}.json"


def _report_matches_row(
    report: dict[str, Any], row: dict[str, Any], *, require_exhausted: bool
) -> bool:
    """Check that a synthesis report was produced by this exact queue row."""
    config = report.get("run_configuration") or {}
    author = config.get("author_model") or {}
    evaluation = config.get("evaluation") or {}
    controls = config.get("synthesis_controls") or {}
    thresholds = config.get("thresholds") or {}
    try:
        attempts = int(report["total_attempts"])
        max_iterations = int(row["max_iterations"])
        exact = (
            1 <= attempts <= max_iterations
            and config.get("task_description") == row["task"]
            and config.get("output_name") == row["output_name"]
            and config.get("git_commit") == row.get("git_commit")
            and isinstance(row.get("execution_source_sha256"), str)
            and re.fullmatch(
                r"[0-9a-f]{64}", str(row["execution_source_sha256"])
            )
            is not None
            and config.get("execution_source_sha256")
            == row["execution_source_sha256"]
            and int(config.get("max_iterations") or -1) == max_iterations
            and author.get("backend") == row["generation_backend"]
            and author.get("model") == row["generation_model"]
            and author.get("route") == row.get("expected_author_route")
            and int(author.get("max_new_tokens") or -1)
            == int(row["synthesis_max_tokens"])
            and int(author.get("reasoning_budget_tokens") or -1)
            == int(row["synthesis_reasoning_budget"])
            and evaluation.get("dataset") == row["dataset"]
            and evaluation.get("eval_model") == row["eval_model"]
            and int(evaluation.get("eval_sample_size") or -1)
            == int(row["eval_sample_size"])
            and int(evaluation.get("eval_max_steps") or -1)
            == int(row["eval_max_steps"])
            and int(evaluation.get("eval_step_token_budget") or -1)
            == int(row["token_budget"])
            and math.isclose(
                float(evaluation.get("eval_max_seconds_per_example") or -1),
                float(row["eval_max_seconds"]),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            and int(evaluation.get("min_examples_before_threshold_stop") or -1)
            == int(row["eval_sample_size"])
            and controls.get("adaptive_helper_mask")
            is bool(row["adaptive_helper_mask"])
            and controls.get("helper_selection_policy")
            == row["helper_selection_policy"]
            and int(controls.get("refinement_beam_size") or -1)
            == int(row["beam_size"])
            and math.isclose(
                float(thresholds.get("min_accuracy") or 0.0),
                float(row["min_accuracy"]),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            and math.isclose(
                float(thresholds.get("min_syntax_rate") or 0.0),
                float(row["min_syntax_rate"]),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        )
    except (KeyError, TypeError, ValueError):
        return False
    if not exact or (require_exhausted and attempts != max_iterations):
        return False
    if row["dataset"] == "smiles":
        return evaluation.get("smiles_classes") in (
            row["smiles_class"],
            [row["smiles_class"]],
        )
    split = evaluation.get("split_provenance") or {}
    prefix = "gsm" if row["dataset"] == "gsm_symbolic" else "spider"
    return (
        split.get("bar_split_name") == "train"
        and split.get(f"{prefix}_split_name") == "train"
        and Path(str(split.get(f"{prefix}_split_file"))).name
        == Path(str(row["heldout_split_file"])).name
    )


def _selected_evaluation_is_valid(
    evaluation: Any, expected_examples: int, dataset: str
) -> bool:
    if not isinstance(evaluation, dict):
        return False
    samples = evaluation.get("sample_outputs")
    if (
        evaluation.get("success") is not True
        or evaluation.get("early_stopped") is not False
        or not isinstance(samples, list)
        or len(samples) != expected_examples
        or not all(
            isinstance(sample, dict)
            and type(sample.get("is_correct")) is bool
            and type(sample.get("is_syntax_valid")) is bool
            for sample in samples
        )
    ):
        return False
    correct = sum(sample["is_correct"] for sample in samples)
    syntax = sum(sample["is_syntax_valid"] for sample in samples)
    try:
        reported_correct = int(evaluation["num_correct"])
        denominator = int(evaluation["accuracy_denominator"])
        reported_accuracy = float(evaluation["accuracy"])
        reported_syntax = float(evaluation["syntax_rate"])
    except (KeyError, TypeError, ValueError):
        return False
    if (
        type(evaluation.get("num_correct")) is not int
        or reported_correct != correct
        or type(evaluation.get("accuracy_denominator")) is not int
        or denominator <= 0
        or denominator > expected_examples
        or not math.isclose(
            reported_syntax,
            syntax / expected_examples,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        return False
    if dataset == "smiles":
        trial = (evaluation.get("aux_metrics") or {}).get("smiles_paper_trial")
        if not isinstance(trial, dict):
            return False
        unique_count = trial.get("unique_valid_count")
        if type(unique_count) is not int or not 0 <= unique_count <= expected_examples:
            return False
        expected_accuracy = unique_count / expected_examples
    else:
        expected_accuracy = reported_correct / denominator
    return math.isclose(
        reported_accuracy, expected_accuracy, rel_tol=0.0, abs_tol=1e-12
    )


def _validated_compiled_selection(
    repo: Path,
    output_name: str,
    *,
    min_accuracy: float,
    min_syntax_rate: float,
    job: dict[str, Any],
    report_path_override: Path | None = None,
) -> dict[str, Any] | None:
    """Select only a compiled strategy proven to belong to this cold row."""
    from scripts.runtime.run_cold_synthesis_queue import current_run_dir

    if report_path_override is None:
        run_dir = current_run_dir(repo, output_name)
        if run_dir is None:
            return None
        report_path = None
    else:
        try:
            report_path = report_path_override.resolve(strict=True)
        except OSError:
            return None
        if report_path.name not in {"success_report.json", "failure_report.json"}:
            return None
        run_dir = report_path.parent.parent
    output_root = (repo / "outputs" / "generated" / output_name).resolve()
    try:
        resolved_run = run_dir.resolve()
    except OSError:
        return None
    if (
        resolved_run.parent != output_root
        or not resolved_run.name.startswith(f"{output_name}_")
    ):
        return None
    run_dir = resolved_run
    success_report = run_dir / "results" / "success_report.json"
    failure_report = run_dir / "results" / "failure_report.json"
    if report_path is None:
        report_path = success_report if success_report.is_file() else failure_report
    if not report_path.is_file():
        return None
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    exhausted = report_path == failure_report
    if not _report_matches_row(report, job, require_exhausted=exhausted):
        return None

    def bound_compiled_dir(raw: Any) -> Path | None:
        compiled = Path(str(raw or ""))
        if not compiled.is_absolute():
            compiled = repo / compiled
        try:
            resolved = compiled.resolve()
            resolved.relative_to((run_dir / "python").resolve())
        except (OSError, ValueError):
            return None
        output_name = str(job["output_name"])
        compiler_suffix = re.fullmatch(
            rf"{re.escape(output_name)}_\d{{8}}_\d{{6}}_[0-9a-f]{{6}}",
            resolved.name,
        )
        if resolved.name != output_name and compiler_suffix is None:
            return None
        return resolved if (resolved / "GeneratedCSD.py").is_file() else None

    if not exhausted:
        try:
            total_attempts = int(report.get("total_attempts"))
            evaluation = report.get("evaluation_result") or {}
            examples = int(evaluation.get("num_examples"))
            accuracy = float(evaluation.get("accuracy"))
            syntax = float(evaluation.get("syntax_rate"))
        except (TypeError, ValueError):
            return None
        if (
            type(report.get("total_attempts")) is not int
            or not 1 <= total_attempts <= int(job["max_iterations"])
            or examples != int(job["eval_sample_size"])
            or not math.isfinite(accuracy)
            or not math.isfinite(syntax)
            or not 0.0 <= accuracy <= 1.0
            or not 0.0 <= syntax <= 1.0
            or accuracy < min_accuracy
            or syntax < min_syntax_rate
            or not _selected_evaluation_is_valid(
                evaluation, int(job["eval_sample_size"]), str(job["dataset"])
            )
        ):
            return None
        compiled_dir = bound_compiled_dir(report.get("compiled_dir"))
        if compiled_dir is None:
            return None
        winning_attempt = total_attempts
    else:
        attempts = report.get("attempts")
        if not isinstance(attempts, list) or len(attempts) != int(job["max_iterations"]):
            return None
        attempt_numbers = [
            attempt.get("attempt_number") if isinstance(attempt, dict) else None
            for attempt in attempts
        ]
        if attempt_numbers != list(range(1, int(job["max_iterations"]) + 1)):
            return None
        candidates: list[tuple[float, float, float, int, Path]] = []
        seen_attempt_numbers: set[int] = set()
        for attempt in attempts:
            compilation = attempt.get("compilation") or {}
            verification = attempt.get("verification") or {}
            evaluation = attempt.get("evaluation") or {}
            try:
                attempt_number = int(attempt.get("attempt_number"))
                examples = int(evaluation.get("num_examples"))
                accuracy = float(evaluation.get("accuracy"))
                syntax = float(evaluation.get("syntax_rate"))
            except (TypeError, ValueError):
                continue
            if attempt_number in seen_attempt_numbers:
                return None
            seen_attempt_numbers.add(attempt_number)
            if (
                compilation.get("success") is not True
                or verification.get("success") is not True
                or not _selected_evaluation_is_valid(
                    evaluation, int(job["eval_sample_size"]), str(job["dataset"])
                )
                or not compilation.get("output_dir")
                or examples != int(job["eval_sample_size"])
                or attempt_number < 1
                or attempt_number > int(job["max_iterations"])
                or not math.isfinite(accuracy)
                or not math.isfinite(syntax)
                or not 0.0 <= accuracy <= 1.0
                or not 0.0 <= syntax <= 1.0
            ):
                continue
            shortfall = max(0.0, min_accuracy - accuracy) + max(
                0.0, min_syntax_rate - syntax
            )
            output_dir = bound_compiled_dir(compilation["output_dir"])
            if output_dir is None:
                continue
            candidates.append(
                (
                    shortfall,
                    -accuracy,
                    -syntax,
                    attempt_number,
                    output_dir,
                )
            )
        if not candidates:
            return None
        selected = min(candidates, key=lambda item: item[:4])
        winning_attempt = selected[3]
        compiled_dir = selected[-1]
    candidate = compiled_dir / "GeneratedCSD.py"
    if not candidate.is_file():
        return None
    return {
        "compiled_csd_path": candidate,
        "report_path": report_path,
        "report_sha256": hash_file(report_path),
        "winning_attempt": winning_attempt,
    }


def _validated_compiled_output(
    repo: Path,
    output_name: str,
    *,
    min_accuracy: float,
    min_syntax_rate: float,
    job: dict[str, Any],
) -> Path | None:
    selection = _validated_compiled_selection(
        repo,
        output_name,
        min_accuracy=min_accuracy,
        min_syntax_rate=min_syntax_rate,
        job=job,
    )
    return None if selection is None else selection["compiled_csd_path"]


def _compiled_selection(repo: Path, row: dict[str, Any]) -> dict[str, Any] | None:
    cold_job = dict(
        row,
        train_sample_size=row["eval_sample_size"],
        train_split_file=row.get("heldout_split_file"),
        train_split_name="train",
    )
    return _validated_compiled_selection(
        repo,
        str(row["output_name"]),
        min_accuracy=float(row["min_accuracy"]),
        min_syntax_rate=float(row["min_syntax_rate"]),
        job=cold_job,
    )


def _compiled_output(repo: Path, row: dict[str, Any]) -> Path | None:
    selection = _compiled_selection(repo, row)
    return None if selection is None else selection["compiled_csd_path"]


def _report_binding_is_valid(
    state: dict[str, Any], row: dict[str, Any], repo: Path
) -> bool:
    try:
        report_path = Path(str(state["synthesis_report_path"])).resolve()
        expected_sha = state["synthesis_report_sha256"]
        if (
            not report_path.is_file()
            or not isinstance(expected_sha, str)
            or re.fullmatch(r"[0-9a-f]{64}", expected_sha) is None
            or hash_file(report_path) != expected_sha
        ):
            return False
        selection = _validated_compiled_selection(
            repo,
            str(row["output_name"]),
            min_accuracy=float(row["min_accuracy"]),
            min_syntax_rate=float(row["min_syntax_rate"]),
            job=row,
            report_path_override=report_path,
        )
        if selection is None:
            return False
        compiled = Path(str(state["compiled_csd_path"])).resolve()
        return (
            selection["report_path"].resolve() == report_path
            and selection["report_sha256"] == expected_sha
            and selection["winning_attempt"] == state["winning_attempt"]
            and selection["compiled_csd_path"].resolve() == compiled
            and state.get("compiled_sha256") == hash_file(compiled)
        )
    except (KeyError, OSError, TypeError, ValueError):
        return False


def _selection_state(selection: dict[str, Any]) -> dict[str, Any]:
    return {
        "compiled_csd_path": str(selection["compiled_csd_path"]),
        "compiled_sha256": hash_file(Path(selection["compiled_csd_path"])),
        "synthesis_report_path": str(selection["report_path"]),
        "synthesis_report_sha256": str(selection["report_sha256"]),
        "winning_attempt": int(selection["winning_attempt"]),
    }


def run_row(
    row: dict[str, Any],
    *,
    repo: Path,
    python: Path,
    state_dir: Path,
    gpus: tuple[int, ...],
    reservation_mib: int | None = None,
    dry_run: bool = False,
    runner: Any = None,
    admission_check: Any = None,
) -> dict[str, Any]:
    """Run one synthesis then its held-out evaluation with restart state."""
    path = _state_path(state_dir, row)
    if dry_run:
        return {"cell_id": row["cell_id"], "status": "dry_run", "command": synthesis_command(row, python)}
    def save(payload: dict[str, Any]) -> None:
        with state_lock(state_dir):
            write_state(path, payload)

    with state_lock(state_dir):
        prior = read_state(path) or {"cell_id": row["cell_id"], "status": "pending", "phase": "synthesis"}
        validate_row_state(row, prior)
        if prior.get("manifest_sha256") not in (None, row.get("manifest_sha256")):
            raise ConfigError(f"state is bound to a different manifest: {row['cell_id']}")
        if prior.get("status") in {"complete", "failed"}:
            if prior.get("status") == "complete":
                output = Path(str(prior.get("heldout_output_json", repo / row["heldout_output_json"])))
                if not output.is_absolute():
                    output = repo / output
                current = artifact_fingerprint(output)
                bound_row = dict(row, compiled_csd_path=prior.get("compiled_csd_path"), compiled_sha256=prior.get("compiled_sha256"), manifest_commit=prior.get("manifest_commit"))
                if current is None or current.get("sha256") != prior.get("heldout_sha256") or not _report_binding_is_valid(prior, row, repo) or not heldout_artifact_is_valid(output, bound_row):
                    raise ConfigError(f"completed state failed artifact revalidation: {row['cell_id']}")
            return prior
        if prior.get("status") == "starting":
            raise ConfigError(
                f"ambiguous prelaunch state requires inspection: {row['cell_id']}"
            )
        if prior.get("status") == "running" and child_is_same_process(prior):
            LOGGER.info("[tableq] surviving child cell=%s phase=%s pid=%s", row["cell_id"], prior.get("phase"), prior.get("pid"))
            return prior
        phase = str(prior.get("phase") or "synthesis")
    synthesis_env = synthesis_environment(row, gpus, os.environ, repo)
    heldout_env = heldout_environment(row, gpus, os.environ, repo)
    command = synthesis_command(row, python)
    log_path = repo / str(row["log_file"])
    reserved_mib = int(
        reservation_mib
        if reservation_mib is not None
        else row["memory_reservation_mib"]
    )
    def start(argv: list[str], child_environment: dict[str, str]):
        if runner is not None:
            return runner(argv, cwd=repo, env=child_environment)
        return start_logged_child(
            argv, cwd=repo, env=child_environment, log_path=log_path
        )

    recovered = None
    recovered_selection = None
    if phase == "synthesis" and prior.get("status") == "running":
        same_manifest = prior.get("manifest_sha256") == row.get("manifest_sha256") and prior.get("cell_id") == row.get("cell_id")
        latest = repo / "outputs" / "generated" / str(row["output_name"]) / "latest_run.txt"
        fresh_latest = artifact_is_new_or_replaced(latest, prior.get("output_before"))
        if same_manifest and fresh_latest:
            recovered_selection = _compiled_selection(repo, row)
            recovered = (
                None
                if recovered_selection is None
                else recovered_selection["compiled_csd_path"]
            )
    if recovered is not None:
        recovered_fingerprint = artifact_fingerprint(recovered)
        if recovered_fingerprint is None:
            recovered = None
        else:
            synthesis_finished_epoch = time.time()
            had_measured_synthesis_timing = all(
                prior.get(field) is not None
                for field in (
                    "row_started_epoch",
                    "row_started_at",
                    "synthesis_started_epoch",
                    "synthesis_started_at",
                )
            )
            synthesis_started_epoch = _persisted_epoch(
                prior, "synthesis_started_epoch", synthesis_finished_epoch
            )
            row_started_epoch = _persisted_epoch(
                prior, "row_started_epoch", synthesis_started_epoch
            )
            prior = dict(
                prior,
                phase="heldout",
                row_started_epoch=row_started_epoch,
                row_started_at=prior.get("row_started_at")
                or utc_timestamp(row_started_epoch),
                synthesis_started_epoch=synthesis_started_epoch,
                synthesis_started_at=prior.get("synthesis_started_at")
                or utc_timestamp(synthesis_started_epoch),
                synthesis_finished_at=utc_timestamp(synthesis_finished_epoch),
                synthesis_wall_time_seconds=round(
                    synthesis_finished_epoch - synthesis_started_epoch, 4
                ),
                phase_timing_coverage=prior.get("phase_timing_coverage")
                or (
                    "all_phases"
                    if had_measured_synthesis_timing
                    else "recovery_anchor"
                ),
                **_selection_state(recovered_selection),
            )
            save(prior)
            LOGGER.info(
                "[tableq] phase-finished cell=%s phase=synthesis wall_seconds=%.4f recovered=true",
                row["cell_id"],
                prior["synthesis_wall_time_seconds"],
            )
            phase = "heldout"
    if recovered is None and phase == "synthesis" and prior.get("status") == "running":
        failed = dict(prior, status="failed", reason="synthesis child ended without a new bound compiled artifact", exit_code=1)
        failed.pop("pid", None); failed.pop("pid_start", None)
        save(failed)
        return failed

    if phase == "synthesis":
        latest = repo / "outputs" / "generated" / str(row["output_name"]) / "latest_run.txt"
        before_output = artifact_fingerprint(latest)
        synthesis_started_epoch = time.time()
        row_started_epoch = _persisted_epoch(
            prior, "row_started_epoch", synthesis_started_epoch
        )
        starting = dict(
            prior,
            manifest_sha256=row.get("manifest_sha256"),
            manifest_commit=row.get("manifest_commit")
            or row.get("manifest_sha256")
            or row.get("git_commit"),
            cell_id=row["cell_id"],
            status="starting",
            phase="synthesis",
            assigned_gpus=list(gpus),
            reservation_mib=reserved_mib,
            log_file=str(log_path),
            output_before=before_output,
            row_started_epoch=row_started_epoch,
            row_started_at=prior.get("row_started_at")
            or utc_timestamp(row_started_epoch),
            synthesis_started_epoch=synthesis_started_epoch,
            synthesis_started_at=utc_timestamp(synthesis_started_epoch),
            phase_timing_coverage="all_phases",
        )
        save(starting)
        try:
            process = start(command, synthesis_env)
        except Exception as exc:
            failed = dict(
                starting,
                status="failed",
                exit_code=1,
                reason=f"synthesis child failed to start: {type(exc).__name__}",
            )
            save(failed)
            return failed
        LOGGER.info("[tableq] launch cell=%s phase=synthesis gpus=%s", row["cell_id"], gpus)
        running = dict(
            starting,
            status="running",
            pid=process.pid,
            pid_start=process_start_identity(process.pid),
        )
        save(running)
        _output, _ = wait_logged_child(process)
        exit_code = process.returncode
        synthesis_finished_epoch = time.time()
        running.update(
            synthesis_finished_at=utc_timestamp(synthesis_finished_epoch),
            synthesis_wall_time_seconds=round(
                synthesis_finished_epoch - synthesis_started_epoch, 4
            ),
        )
        LOGGER.info(
            "[tableq] phase-finished cell=%s phase=synthesis wall_seconds=%.4f exit_code=%s",
            row["cell_id"],
            running["synthesis_wall_time_seconds"],
            exit_code,
        )
        running.pop("pid", None); running.pop("pid_start", None)
        has_new_run = artifact_is_new_or_replaced(latest, before_output)
        if not has_new_run:
            failed = dict(running, status="failed", exit_code=1, reason="synthesis returned success without a new run report")
            save(failed)
            return failed
        selection = _compiled_selection(repo, row)
        if selection is None:
            failed = dict(running, status="failed", exit_code=exit_code or 1, reason="synthesis failed or produced no recoverable compiled artifact")
            save(failed)
            return failed
        compiled = Path(selection["compiled_csd_path"])
        prior = dict(
            running,
            phase="heldout",
            synthesis_exit_code=exit_code,
            **_selection_state(selection),
        )
        save(prior)

    compiled = Path(str(prior["compiled_csd_path"]))
    compiled_fingerprint = artifact_fingerprint(compiled)
    if compiled_fingerprint is None:
        failed = dict(prior, status="failed", reason="state-bound compiled artifact is missing", exit_code=1)
        failed.pop("pid", None)
        failed.pop("pid_start", None)
        save(failed)
        return failed
    expected_compiled_sha = prior.get("compiled_sha256")
    if not isinstance(expected_compiled_sha, str) or len(expected_compiled_sha) != 64:
        failed = dict(prior, status="failed", reason="compiled artifact has no state-bound hash", exit_code=1)
        save(failed)
        return failed
    if compiled_fingerprint["sha256"] != expected_compiled_sha:
        failed = dict(prior, status="failed", reason="compiled artifact changed before held-out evaluation", exit_code=1)
        save(failed)
        return failed
    if not _report_binding_is_valid(prior, row, repo):
        failed = dict(
            prior,
            status="failed",
            reason="selected synthesis report changed before held-out evaluation",
            exit_code=1,
        )
        save(failed)
        return failed
    if admission_check is not None:
        admission_check(row, require_provider=False)
    final_output = repo / str(row["heldout_output_json"])
    final_output.parent.mkdir(parents=True, exist_ok=True)
    before = artifact_fingerprint(final_output)
    temporary = final_output.with_name(f".{final_output.name}.{os.getpid()}.tmp")
    heldout_row = dict(
        row,
        heldout_output_json=str(temporary),
        compiled_csd_path=str(compiled),
        compiled_sha256=expected_compiled_sha,
        manifest_commit=prior.get("manifest_commit"),
    )
    heldout_started_epoch = _persisted_epoch(
        prior, "heldout_started_epoch", time.time()
    )
    synthesis_timing_fields = (
        "row_started_epoch",
        "row_started_at",
        "synthesis_started_epoch",
        "synthesis_started_at",
        "synthesis_finished_at",
        "synthesis_wall_time_seconds",
    )
    phase_timing_coverage = prior.get("phase_timing_coverage")
    if phase_timing_coverage is not None and phase_timing_coverage not in {
        "all_phases",
        "recovery_anchor",
    }:
        raise ConfigError("runtime field phase_timing_coverage is invalid")
    if not all(prior.get(field) is not None for field in synthesis_timing_fields):
        prior = dict(
            prior,
            row_started_epoch=heldout_started_epoch,
            row_started_at=utc_timestamp(heldout_started_epoch),
            synthesis_started_epoch=heldout_started_epoch,
            synthesis_started_at=utc_timestamp(heldout_started_epoch),
            synthesis_finished_at=utc_timestamp(heldout_started_epoch),
            synthesis_wall_time_seconds=0.0,
            phase_timing_coverage="recovery_anchor",
        )
    elif phase_timing_coverage is None:
        prior = dict(prior, phase_timing_coverage="all_phases")
    starting = dict(
        prior,
        status="starting",
        phase="heldout",
        assigned_gpus=list(gpus),
        reservation_mib=reserved_mib,
        log_file=str(log_path),
        heldout_output_before=before,
        heldout_started_epoch=heldout_started_epoch,
        heldout_started_at=prior.get("heldout_started_at")
        or utc_timestamp(heldout_started_epoch),
    )
    save(starting)
    try:
        process = start(
            heldout_command(heldout_row, python, compiled), heldout_env
        )
    except Exception as exc:
        failed = dict(
            starting,
            status="failed",
            exit_code=1,
            reason=f"held-out child failed to start: {type(exc).__name__}",
        )
        save(failed)
        return failed
    LOGGER.info("[tableq] launch cell=%s phase=heldout gpus=%s", row["cell_id"], gpus)
    running = dict(
        starting,
        status="running",
        pid=process.pid,
        pid_start=process_start_identity(process.pid),
    )
    save(running)
    _output, _ = wait_logged_child(process)
    exit_code = process.returncode
    heldout_finished_epoch = time.time()
    row_started_epoch = _persisted_epoch(
        running, "row_started_epoch", heldout_started_epoch
    )
    running.update(
        heldout_finished_at=utc_timestamp(heldout_finished_epoch),
        heldout_wall_time_seconds=round(
            heldout_finished_epoch - heldout_started_epoch, 4
        ),
        row_finished_at=utc_timestamp(heldout_finished_epoch),
        total_wall_time_seconds=round(
            heldout_finished_epoch - row_started_epoch, 4
        ),
    )
    LOGGER.info(
        "[tableq] phase-finished cell=%s phase=heldout wall_seconds=%.4f total_seconds=%.4f exit_code=%s",
        row["cell_id"],
        running["heldout_wall_time_seconds"],
        running["total_wall_time_seconds"],
        exit_code,
    )
    running.pop("pid", None); running.pop("pid_start", None)
    if exit_code != 0 or not artifact_is_new_or_replaced(temporary, None) or not heldout_artifact_is_valid(temporary, heldout_row):
        failed = dict(running, status="failed", exit_code=exit_code or 1, reason="held-out evaluation failed or produced no artifact")
        save(failed)
        return failed
    report_path = Path(str(running["synthesis_report_path"]))
    report_payload = json.loads(report_path.read_text(encoding="utf-8"))
    heldout_payload = json.loads(temporary.read_text(encoding="utf-8"))
    heldout_payload["controller_runtime"] = _runtime_evidence(
        running, report_payload, heldout_payload
    )
    temporary.write_text(
        json.dumps(heldout_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not heldout_artifact_is_valid(temporary, heldout_row):
        failed = dict(
            running,
            status="failed",
            exit_code=1,
            reason="held-out runtime evidence failed artifact revalidation",
        )
        save(failed)
        return failed
    temporary.replace(final_output)
    complete = dict(
        running,
        status="complete",
        exit_code=0,
        heldout_output_json=str(final_output),
        heldout_sha256=artifact_fingerprint(final_output)["sha256"],
        compiled_sha256=artifact_fingerprint(compiled)["sha256"],
    )
    save(complete)
    return complete


def dispatch(
    rows: list[dict[str, Any]],
    *,
    repo: Path,
    python: Path,
    state_dir: Path,
    allowed: tuple[int, ...],
    snapshot: Any,
    poll_seconds: float = 30.0,
    dry_run: bool = False,
    admission_check: Any = None,
) -> list[dict[str, Any]]:
    """Dispatch only when a scoped GPU fits; keep polling while work remains."""
    results: list[dict[str, Any]] = []
    pending = list(rows)
    while pending:
        live = snapshot()
        next_pending: list[dict[str, Any]] = []
        admitted: list[tuple[dict[str, Any], tuple[int, ...], int]] = []
        reservations: dict[int, int] = {}
        terminal_cells: set[str] = set()
        survivor_cells: set[str] = set()
        surviving_child = False
        states = {
            str(row["cell_id"]): read_state(_state_path(state_dir, row))
            for row in pending
        }
        for row in pending:
            validate_row_state(row, states[str(row["cell_id"])])
        for row in pending:
            state = states[str(row["cell_id"])]
            if state and state.get("status") in {"complete", "failed"}:
                result = run_row(
                    row,
                    repo=repo,
                    python=python,
                    state_dir=state_dir,
                    gpus=(),
                    reservation_mib=0,
                    dry_run=dry_run,
                )
                results.append(result)
                terminal_cells.add(str(row["cell_id"]))
        for row in pending:
            if str(row["cell_id"]) in terminal_cells:
                continue
            state = states[str(row["cell_id"])]
            if state and state.get("status") == "running" and child_is_same_process(state):
                assigned = state.get("assigned_gpus")
                demand = state.get("reservation_mib")
                if (
                    not isinstance(assigned, list)
                    or len(assigned) != int(row.get("gpu_count", 1))
                    or any(type(gpu) is not int or gpu not in allowed for gpu in assigned)
                    or type(demand) is not int
                    or demand <= 0
                ):
                    raise ConfigError(
                        f"surviving child has no valid GPU reservation: {row['cell_id']}"
                    )
                for gpu in assigned:
                    reservations[gpu] = reservations.get(gpu, 0) + demand
                survivor_cells.add(str(row["cell_id"]))
                LOGGER.info(
                    "[tableq] poll-surviving-child cell=%s phase=%s pid=%s",
                    row["cell_id"],
                    state.get("phase"),
                    state.get("pid"),
                )
                next_pending.append(row)
                surviving_child = True
                continue
        for row in pending:
            if str(row["cell_id"]) in terminal_cells | survivor_cells:
                continue
            state = states[str(row["cell_id"])]
            gpus = choose_gpus(row, live, reservations, live, allowed)
            if gpus is None:
                next_pending.append(row)
                continue
            if admission_check is not None:
                phase = str((state or {}).get("phase") or "synthesis")
                require_provider = phase == "synthesis"
                try:
                    admission_check(row, require_provider=require_provider)
                except ConfigError as exc:
                    if not str(exc).startswith("fresh admission blocked"):
                        raise
                    LOGGER.info(
                        "[tableq] admission-wait cell=%s reason=%s",
                        row["cell_id"],
                        exc,
                    )
                    next_pending.append(row)
                    continue
                live = snapshot()
                gpus = choose_gpus(row, live, reservations, live, allowed)
                if gpus is None:
                    next_pending.append(row)
                    continue
            LOGGER.info("[tableq] admission cell=%s gpus=%s", row["cell_id"], gpus)
            demand = _demand(row, int(live[gpus[0]]["total_mib"]))
            for gpu in gpus:
                reservations[gpu] = reservations.get(gpu, 0) + demand
            admitted.append((row, gpus, demand))
        if admitted:
            with ThreadPoolExecutor(max_workers=len(admitted), thread_name_prefix="tableq") as pool:
                futures = {
                    pool.submit(
                        run_row,
                        row,
                        repo=repo,
                        python=python,
                        state_dir=state_dir,
                        gpus=gpus,
                        reservation_mib=demand,
                        dry_run=dry_run,
                        admission_check=admission_check,
                    ): (row, gpus, demand)
                    for row, gpus, demand in admitted
                }
                for future, (row, gpus, demand) in futures.items():
                    result = future.result()
                    if result.get("status") == "running":
                        next_pending.append(row)
                    else:
                        results.append(result)
        pending = next_pending
        if pending and surviving_child:
            time.sleep(max(0.1, poll_seconds))
        elif pending and not admitted:
            if dry_run:
                results.extend({"cell_id": row["cell_id"], "status": "waiting", "command": synthesis_command(row, python)} for row in pending)
                break
            time.sleep(max(0.1, poll_seconds))
    return results


def write_state(path: Path, payload: dict[str, Any]) -> None:
    """Write state in one replacement so a restart sees an old or new file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)


def read_state(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"invalid queue state {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ConfigError(f"queue state must be an object: {path}")
    return payload


def validate_row_state(
    row: dict[str, Any], state: dict[str, Any] | None
) -> None:
    """Reject unknown or impossible queue states before scheduling work."""
    if state is None:
        return
    status = state.get("status")
    phase = state.get("phase")
    allowed_phases = {
        "pending": {"synthesis"},
        "starting": {"synthesis", "heldout"},
        "running": {"synthesis", "heldout"},
        "complete": {"heldout"},
        "failed": {"synthesis", "heldout"},
    }
    if (
        state.get("cell_id") != row.get("cell_id")
        or status not in allowed_phases
        or phase not in allowed_phases[status]
    ):
        raise ConfigError(f"invalid queue state for {row['cell_id']}")


def process_start_identity(pid: int) -> str | None:
    try:
        fields = Path(f"/proc/{int(pid)}/stat").read_text(encoding="utf-8").rsplit(")", 1)[1].split()
        return fields[19]
    except (OSError, ValueError, IndexError):
        return None


def child_is_same_process(state: dict[str, Any]) -> bool:
    try:
        pid = int(state["pid"])
    except (KeyError, TypeError, ValueError):
        return False
    expected = str(state.get("pid_start", ""))
    actual = process_start_identity(pid)
    if not expected or actual != expected:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def lock_path(state_dir: Path) -> Path:
    return state_dir / "table5_8.state.lock"


def controller_lock_path(repo: Path) -> Path:
    return repo / ".context" / "table5_8" / "table5_8.controller.lock"


@contextmanager
def controller_lock(repo: Path):
    """Keep exactly one Table 5--8 controller alive in this checkout."""
    path = controller_lock_path(repo)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ConfigError("a Table 5--8 controller is already running") from exc
        yield handle
    finally:
        handle.close()


def state_lock(state_dir: Path):
    state_dir.mkdir(parents=True, exist_ok=True)
    handle = lock_path(state_dir).open("a+", encoding="utf-8")
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
    return handle


def export_results(rows: list[dict[str, Any]], values: list[dict[str, Any]], output: Path) -> None:
    by_id = {str(v["cell_id"]): v for v in values}
    if len(by_id) != len(rows) or set(by_id) != {str(row["cell_id"]) for row in rows}:
        raise ConfigError("export requires one result for every queue row")
    manifest_bindings = {row.get("manifest_sha256") for row in rows}
    commit_bindings = {row.get("git_commit") for row in rows}
    if (
        len(manifest_bindings) != 1
        or len(commit_bindings) != 1
        or not isinstance(next(iter(manifest_bindings)), str)
        or not re.fullmatch(r"[0-9a-f]{64}", next(iter(manifest_bindings)))
        or not isinstance(next(iter(commit_bindings)), str)
        or not re.fullmatch(r"[0-9a-f]{40}", next(iter(commit_bindings)))
    ):
        raise ConfigError("export rows are not bound to one manifest and commit")
    cells: list[dict[str, Any]] = []
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        value = dict(by_id[row["cell_id"]])
        provenance = value.get("reevaluation_provenance") or {}
        source = {
            "cell_id": row["cell_id"],
            "profile": row["profile"],
            "generation_backend": row["generation_backend"],
            "generation_model": row["generation_model"],
            "heldout_artifact_path": value.get("paper_artifact_path"),
            "heldout_artifact_sha256": value.get("paper_artifact_sha256"),
            "compiled_csd_path": provenance.get("compiled_csd_path"),
            "compiled_csd_sha256": provenance.get("compiled_csd_sha256"),
            "synthesis_report_path": value.get("synthesis_report_path"),
            "synthesis_report_sha256": value.get("synthesis_report_sha256"),
            "winning_attempt": value.get("winning_attempt"),
            "eval_model_revision": provenance.get("eval_model_revision"),
            "eval_model_snapshot_path": provenance.get("eval_model_snapshot_path"),
            "eval_model_snapshot_sha256": provenance.get(
                "eval_model_snapshot_sha256"
            ),
            "eval_model_snapshot_file_count": provenance.get(
                "eval_model_snapshot_file_count"
            ),
        }
        if row["benchmark"] == "spider":
            source.update(
                {
                    "spider_data_path": provenance.get("spider_data_path"),
                    "spider_data_sha256": provenance.get("spider_data_sha256"),
                    "spider_data_file_count": provenance.get("spider_data_file_count"),
                }
            )
        if (
            not isinstance(source["heldout_artifact_path"], str)
            or not re.fullmatch(r"[0-9a-f]{64}", str(source["heldout_artifact_sha256"]))
            or not isinstance(source["compiled_csd_path"], str)
            or not re.fullmatch(r"[0-9a-f]{64}", str(source["compiled_csd_sha256"]))
            or not isinstance(source["synthesis_report_path"], str)
            or not re.fullmatch(
                r"[0-9a-f]{64}", str(source["synthesis_report_sha256"])
            )
            or type(source["winning_attempt"]) is not int
            or source["winning_attempt"] < 1
        ):
            raise ConfigError(f"missing sealed export evidence for {row['cell_id']}")
        value.update(
            {
                "cell_id": row["cell_id"],
                "benchmark": row["benchmark"],
            }
        )
        value["paper_source"] = source
        value["runtime"] = validated_runtime(value, cell_id=str(row["cell_id"]))
        if row["benchmark"] != "smiles":
            metric = "accuracy"
            if not isinstance(value.get(metric), (int, float)):
                raise ConfigError(f"missing {metric} for {row['cell_id']}")
            syntax_rate = value.get("syntax_rate")
            attempts = value.get("synthesis_attempts")
            terminal_status = value.get("synthesis_terminal_status")
            if (
                not isinstance(syntax_rate, (int, float))
                or not math.isfinite(float(syntax_rate))
                or not 0.0 <= float(syntax_rate) <= 1.0
                or type(attempts) is not int
                or not 1 <= attempts <= int(row["max_iterations"])
                or terminal_status not in {"accepted", "exhausted"}
            ):
                raise ConfigError(
                    f"missing synthesis outcome metrics for {row['cell_id']}"
                )
            value["cw"] = constrained_window_rate(value)
        else:
            trial = value.get("smiles_paper_trial") or {}
            count = trial.get("sample_count")
            unique = trial.get("unique_valid_count")
            if not isinstance(count, int) or count <= 0 or not isinstance(unique, int) or unique < 0 or unique > count:
                raise ConfigError(f"missing validated smiles_paper_trial for {row['cell_id']}")
            value["sample_count"] = count
            value["unique_valid_rate"] = unique / count
        paper_cells = row.get("paper_cells")
        if not isinstance(paper_cells, list) or not paper_cells:
            raise ConfigError(f"missing paper-cell mapping for {row['cell_id']}")
        for mapping in paper_cells:
            if (
                not isinstance(mapping, dict)
                or set(mapping) != {"table", "table_cell_id"}
                or type(mapping.get("table")) is not int
                or mapping["table"] not in {5, 6, 7, 8}
                or not isinstance(mapping.get("table_cell_id"), str)
                or not mapping["table_cell_id"]
            ):
                raise ConfigError(f"invalid paper-cell mapping for {row['cell_id']}")
            mapped = dict(
                value,
                table=mapping["table"],
                table_cell_id=mapping["table_cell_id"],
            )
            groups.setdefault(mapping["table_cell_id"], []).append(mapped)
    for cell_id, group in groups.items():
        item = {"table_cell_id": cell_id, "table": group[0]["table"], "benchmark": group[0]["benchmark"]}
        item["sources"] = [value["paper_source"] for value in group]
        if group[0]["benchmark"] == "smiles":
            item["unique_valid_rate"] = weighted_smiles_rate(group)
            item["sample_count"] = sum(int(v["sample_count"]) for v in group)
        else:
            metric = "accuracy"
            item[metric] = group[0][metric]
            item["syntax_rate"] = group[0]["syntax_rate"]
            item["synthesis_attempts"] = group[0]["synthesis_attempts"]
            item["synthesis_terminal_status"] = group[0][
                "synthesis_terminal_status"
            ]
            item["cw"] = group[0]["cw"]
            item["runtime"] = group[0]["runtime"]
        cells.append(item)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(output.suffix + ".tmp")
    temp.write_text(
        json.dumps(
            {
                "version": 2,
                "manifest_sha256": next(iter(manifest_bindings)),
                "git_commit": next(iter(commit_bindings)),
                "cells": cells,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    temp.replace(output)


def main() -> int:
    private_environment = campaign_environment(dict(os.environ))
    for name in (
        "GEMINI_API_KEY",
        "CSD_PI_NODE_EXECUTABLE",
        "CSD_PI_BRIDGE_PATH",
        "CSD_PI_AUTH_PATH",
        "CSD_CLAUDE_CONFIG_DIR",
        "CSD_CLAUDE_EXPECTED_ACCOUNT",
    ):
        if private_environment.get(name):
            os.environ[name] = private_environment[name]
    if "--controller" in sys.argv[1:]:
        controller_args = [arg for arg in sys.argv[1:] if arg != "--controller"]
        return controller_main(controller_parser().parse_args(controller_args))
    parser = argparse.ArgumentParser(description="Build or dry-run the Table 5--8 queue")
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument(
        "--provider-pilot-report",
        action="append",
        default=[],
        metavar="PROFILE=PATH",
        help="one real one-attempt synthesis report to parse and bind",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    rows = build_scope(args.repo)
    if len(rows) != 8:
        raise SystemExit(f"scope error: expected 8 rows, got {len(rows)}")
    if args.dry_run:
        for row in rows:
            print(row["cell_id"], shlex.join(synthesis_command(row, Path(sys.executable))))
        return 0
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=args.repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    provider_pilots: dict[str, Any] = {}
    for specification in args.provider_pilot_report:
        try:
            profile, raw_path = specification.split("=", 1)
        except ValueError as exc:
            raise SystemExit("--provider-pilot-report must be PROFILE=PATH") from exc
        if profile in provider_pilots:
            raise SystemExit(f"duplicate provider pilot report: {profile}")
        provider_pilots[profile] = provider_pilot_from_report(
            Path(raw_path),
            profile=profile,
            git_commit=commit,
            environment=dict(os.environ),
        )
    payload = manifest_payload(args.repo, rows, provider_pilots=provider_pilots)
    target = args.manifest or args.repo / "outputs/controlled_comparison/table5_8_manifest.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
