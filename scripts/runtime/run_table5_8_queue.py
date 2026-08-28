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
import shlex
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from synthesis.evaluate.benchmarks.gsm_symbolic.prompts import GSM_CRANE_COT_TASK
from synthesis.run_constants import VLLM_GPU_MEMORY_UTILIZATION_BY_MODEL

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


class ConfigError(ValueError):
    """The manifest or runtime configuration cannot be safely launched."""

EVAL_MODEL = "Qwen/Qwen3.5-2B"
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
    "gemini3.1-pro": {"generation_backend": "vertex", "generation_model": "gemini-3.1-pro-preview"},
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
    "synthesis/split_provenance.py",
    "synthesis/generate/generator.py",
    "synthesis/generate/provider_names.py",
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

MANIFEST_KEYS = frozenset({"version", "git_commit", "crane_commit", "crane_source_sha256", "source_sha256", "jobs", "provider_pilots", "provider_pilot_sha256"})
JOB_KEYS = frozenset({
    "cell_id", "table", "table_cell_id", "benchmark", "dataset", "task", "profile",
    "generation_backend", "generation_model", "eval_model", "synthesis_max_tokens",
    "synthesis_reasoning_budget",
    "effective_output_tokens", "effective_thinking_tokens",
    "smiles_class", "token_budget", "beam_size", "adaptive_helper_mask",
    "helper_selection_policy", "max_iterations", "min_accuracy", "min_syntax_rate",
    "bar_source_path", "bar_source_sha256", "eval_sample_size", "heldout_sample_size",
    "eval_max_steps", "eval_max_seconds", "gpu_mem_util", "memory_reservation_mib",
    "gpu_scope", "gpu_count", "heldout_split_name", "heldout_split_file", "sample_count",
    "output_name", "heldout_output_json", "log_file", "cold_start", "git_commit",
    "launch_commit", "vertex_project",
})


def _row(cell_id: str, table: int, benchmark: str, profile: str, *, smiles_class: str | None = None, **controls: Any) -> dict[str, Any]:
    settings = DATASET_SETTINGS[benchmark]
    author = TABLE5_PROFILES[profile]
    sample = settings["feedback"]
    return {
        "cell_id": cell_id,
        "table": table,
        "table_cell_id": controls.pop("table_cell_id", cell_id),
        "benchmark": benchmark,
        "dataset": benchmark,
        "task": TASKS[benchmark],
        "profile": profile,
        "generation_backend": author["generation_backend"],
        "generation_model": author["generation_model"],
        "eval_model": EVAL_MODEL,
        "synthesis_max_tokens": AUTHOR_TOKEN_BUDGET,
        "synthesis_reasoning_budget": AUTHOR_REASONING_BUDGET,
        "effective_output_tokens": {"opus5": 64000, "gpt5.6-sol": None, "gemini3.1-pro": 32768}[profile],
        "effective_thinking_tokens": {"opus5": 48000, "gpt5.6-sol": None, "gemini3.1-pro": None}[profile],
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
        "memory_reservation_mib": 14_336,
        "gpu_scope": [0, 1, 2, 3],
        "gpu_count": 2 if benchmark in {"gsm_symbolic", "spider"} else 1,
        "heldout_split_name": "test",
        "heldout_split_file": CANONICAL_SPLITS.get(benchmark),
        "sample_count": settings["heldout"],
        "output_name": f"table5_8_{cell_id}",
        "heldout_output_json": f"outputs/reeval/table5_8/{cell_id}.json",
        "log_file": f"outputs/generated/table5_8_{cell_id}/run.log",
        "cold_start": True,
    }


def build_scope(repo: Path) -> list[dict[str, Any]]:
    """Return exactly the 31 requested synthesis runs, in stable order."""
    rows: list[dict[str, Any]] = []
    for profile in TABLE5_PROFILES:
        for benchmark in ("gsm_symbolic", "spider"):
            rows.append(_row(f"t5-{profile}-{benchmark}", 5, benchmark, profile, table_cell_id=f"table5-{profile}-{benchmark}"))
        for smiles_class in SMILES_CLASSES:
            rows.append(_row(f"t5-{profile}-smiles-{smiles_class}", 5, "smiles", profile, smiles_class=smiles_class, table_cell_id=f"table5-{profile}-smiles"))
    for table, settings in (
        (6, [(1, 2, True), (2, 2, True), (4, 2, True)]),
        (7, [(1, 1, True), (1, 2, True), (1, 4, True)]),
        (8, [(1, 2, False), (1, 2, True)]),
    ):
        for token_budget, beam_size, mask in settings:
            for benchmark in ("gsm_symbolic", "spider"):
                cell = f"t{table}-opus5-{benchmark}-b{token_budget}-B{beam_size}-m{int(mask)}"
                rows.append(_row(cell, table, benchmark, "opus5", table_cell_id=cell, token_budget=token_budget, beam_size=beam_size, adaptive_helper_mask=mask))
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
    """Return CW, defined by the evaluator's validated syntax/parse rate."""
    syntax_rate = value.get("syntax_rate")
    if not isinstance(syntax_rate, (int, float)):
        raise ConfigError("result is missing validated syntax_rate for CW")
    return float(syntax_rate)


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
        "gemini3.1-pro": ("vertex", "gemini-3.1-pro-preview", 32768, None),
        "opus5": ("claude", "claude-opus-5", 64000, 48000),
    }
    if profile not in expected_routes:
        raise ConfigError(f"unknown provider pilot profile: {profile}")
    route = payload.get("run_configuration")
    attempts = payload.get("attempts")
    if not isinstance(route, dict) or not isinstance(attempts, list) or len(attempts) != 1:
        raise ConfigError("provider pilot must be a one-attempt synthesis report")
    if payload.get("total_attempts") != 1 or route.get("max_iterations") != 1:
        raise ConfigError("provider pilot must use exactly one attempt")
    if route.get("git_commit") != git_commit:
        raise ConfigError("provider pilot report is bound to a different code commit")

    author = route.get("author_model") or {}
    evaluation_config = route.get("evaluation") or {}
    controls = route.get("synthesis_controls") or {}
    backend, model, effective_output, effective_thinking = expected_routes[profile]
    try:
        expected_config = (
            author.get("backend") == backend
            and author.get("model") == model
            and author.get("max_new_tokens") == AUTHOR_TOKEN_BUDGET
            and author.get("reasoning_budget_tokens") == AUTHOR_REASONING_BUDGET
            and route.get("task_description") == TASKS["smiles"]
            and evaluation_config.get("dataset") == "smiles"
            and evaluation_config.get("eval_model") == EVAL_MODEL
            and evaluation_config.get("eval_sample_size") == 1
            and evaluation_config.get("eval_max_steps") == DATASET_SETTINGS["smiles"]["steps"]
            and evaluation_config.get("eval_step_token_budget") == 1
            and float(evaluation_config.get("eval_max_seconds_per_example", -1)) == 600.0
            and evaluation_config.get("min_examples_before_threshold_stop") == 1
            and evaluation_config.get("smiles_classes") in ("acrylates", ["acrylates"])
            and controls.get("adaptive_helper_mask") is True
            and controls.get("helper_selection_policy") == "bandit"
            and controls.get("refinement_beam_size") == 2
        )
    except (TypeError, ValueError):
        expected_config = False
    if not expected_config:
        raise ConfigError("provider pilot report has the wrong route or controls")

    attempt = attempts[0]
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
    compiled_dir = Path(str(compilation.get("output_dir") or ""))
    compiled_csd = compiled_dir / "GeneratedCSD.py"
    try:
        compiled_dir.resolve().relative_to((run_root / "python").resolve())
    except (OSError, ValueError) as exc:
        raise ConfigError("provider pilot compiled artifact is outside its run") from exc
    if (
        route.get("output_name") != run_root.name
        or compiled_dir.name != run_root.name
        or not compiled_csd.is_file()
    ):
        raise ConfigError("provider pilot compiled artifact is missing or unbound")

    created_at = payload.get("timestamp")
    if not isinstance(created_at, str):
        raise ConfigError("provider pilot report has no timestamp")
    pilot = {
        "status": "ready",
        "git_commit": git_commit,
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
                "config_dir": "/home/aadivyar/.claude-csd-synthesis",
                "expected_account": "ssdear@gmail.com",
            }
        )
    if profile == "gemini3.1-pro":
        adc = Path(environment.get("GOOGLE_APPLICATION_CREDENTIALS", ""))
        project = verified_adc_project(environment)
        if not project or not adc.is_file():
            raise ConfigError("Vertex provider pilot has no bound ADC project")
        pilot.update(
            {
                "vertex_project": project,
                "adc_sha256": hash_file(adc),
                "location": "global",
            }
        )
    return pilot


def codex_auth_probe() -> dict[str, Any]:
    """Verify the exact author route with a tiny isolated sentinel request."""
    executable = os.environ.get("CSD_CODEX_EXECUTABLE", "codex")
    sentinel = "CSD_AUTH_SENTINEL_9f7a"
    cwd = Path(tempfile.mkdtemp(prefix="tableq-codex-probe-cwd-"))
    output = cwd / "probe.txt"
    try:
        result = subprocess.run(
            [
                executable, "exec", "--model", "gpt-5.6-sol", "--sandbox", "read-only",
                "--ephemeral", "--ignore-user-config", "--ignore-rules",
                "--skip-git-repo-check", "--cd", str(cwd), "--output-last-message",
                str(output), "-",
            ],
            input=f"Return exactly {sentinel} and nothing else.\n",
            capture_output=True,
            text=True,
                timeout=90,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"returncode": 1, "stdout": "", "stderr": type(exc).__name__}
    try:
        response = output.read_text(encoding="utf-8").strip()
    except OSError:
        response = ""
    finally:
        import shutil
        shutil.rmtree(cwd, ignore_errors=True)
    return {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr, "status": "ready" if result.returncode == 0 and response == sentinel else "blocked"}


def claude_auth_probe(environment: dict[str, str]) -> dict[str, Any]:
    """Check the exact first-party Max account without sending a prompt."""
    executable = environment.get("CSD_CLAUDE_EXECUTABLE", "claude")
    checked = dict(environment)
    checked["CLAUDE_CONFIG_DIR"] = "/home/aadivyar/.claude-csd-synthesis"
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
        and payload.get("email") == "ssdear@gmail.com"
        and payload.get("authMethod") == "claude.ai"
        and payload.get("apiProvider") == "firstParty"
        and str(payload.get("subscriptionType", "")).lower() == "max"
    )
    if not exact:
        return {"status": "blocked", "reason": "wrong Claude account or route"}
    return {
        "status": "ready",
        "account": "ssdear@gmail.com",
        "config_dir": "/home/aadivyar/.claude-csd-synthesis",
    }


def vertex_adc_probe(environment: dict[str, str]) -> dict[str, Any]:
    """Refresh the exact ADC credential through the same google-auth path as generation."""
    try:
        import google.auth
        from google.auth.transport.requests import Request

        adc_value = environment.get("GOOGLE_APPLICATION_CREDENTIALS", "")
        adc = Path(adc_value)
        expected_project = verified_adc_project(environment)
        if not adc.is_file() or not expected_project:
            return {"status": "blocked", "reason": "missing ADC project"}
        credentials, loaded_project = google.auth.load_credentials_from_file(
            str(adc),
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
        credentials.refresh(Request())
    except Exception as exc:  # google-auth uses several provider-specific errors.
        return {"status": "blocked", "reason": type(exc).__name__}
    project = loaded_project or expected_project
    if project != expected_project or not getattr(credentials, "token", None):
        return {"status": "blocked", "reason": "wrong ADC project or empty token"}
    return {
        "status": "ready",
        "vertex_project": expected_project,
        "adc_sha256": hash_file(adc),
        "location": "global",
    }


def validate_provider_pilot(
    profile: str,
    pilot: Any,
    git_commit: str | None,
    *,
    repo: Path,
    environment: dict[str, str],
) -> str | None:
    """Validate a one-attempt provider pilot bound to the exact code bytes."""
    if not isinstance(pilot, dict) or pilot.get("status") != "ready":
        return f"{profile} provider pilot is missing or not ready"
    if not git_commit or pilot.get("git_commit") != git_commit:
        return f"{profile} provider pilot is bound to a different code commit"
    expected = {
        "gpt5.6-sol": ("codex", "gpt-5.6-sol"),
        "gemini3.1-pro": ("vertex", "gemini-3.1-pro-preview"),
        "opus5": ("claude", "claude-opus-5"),
    }
    backend, model = expected[profile]
    if pilot.get("backend") != backend or pilot.get("model") != model:
        return f"{profile} provider pilot has the wrong route"
    if profile == "opus5" and (
        pilot.get("config_dir") != "/home/aadivyar/.claude-csd-synthesis"
        or pilot.get("expected_account") != "ssdear@gmail.com"
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
    if age > 24 * 60 * 60 or age < -5 * 60:
        return f"{profile} provider pilot is stale"
    return None


def execution_source_paths(repo: Path) -> tuple[str, ...]:
    """Return the tracked source closure instead of a hand-maintained subset."""
    names = subprocess.run(["git", "ls-files", "-z"], cwd=repo, check=True, capture_output=True).stdout.decode("utf-8").split("\0")
    selected = [
        name for name in names if name and (
            name.startswith("synthesis/")
            or name.startswith("scripts/runtime/")
            or name.startswith("environment/benchmark_splits/")
            or name in {"run_all_tests.py", ".context/run_post14b_rebar_queue.py"}
        )
    ]
    return tuple(sorted(selected))


def profile_block_reason(
    row: dict[str, Any],
    environment: dict[str, str],
    *,
    repo: Path,
    provider_pilots: dict[str, Any] | None = None,
    cached_probes: dict[str, dict[str, Any]] | None = None,
    cached_auth: dict[str, dict[str, Any]] | None = None,
) -> str | None:
    """Return a durable pending reason, or None when this row may be admitted."""
    if row["profile"] == "gpt5.6-sol":
        probe = (cached_probes or {}).get("gpt5.6-sol") or codex_auth_probe()
        if probe.get("status") != "ready":
            LOGGER.error("[tableq] auth-block profile=gpt5.6-sol reason=codex-local-auth")
            return "codex local authentication is unavailable or invalid"
    checked_environment = dict(environment)
    if row["profile"] == "gemini3.1-pro" and row.get("vertex_project"):
        if row["vertex_project"] != verified_adc_project(environment):
            return "gemini3.1-pro row is not bound to the active ADC project"
        checked_environment["GOOGLE_CLOUD_PROJECT"] = str(row["vertex_project"])
        checked_environment["VERTEX_AI_PROJECT"] = str(row["vertex_project"])
        checked_environment["GOOGLE_CLOUD_LOCATION"] = "global"
    try:
        validate_profile_gates([row], checked_environment)
    except ConfigError as exc:
        return str(exc)
    pilot_reason = validate_provider_pilot(
        row["profile"],
        (provider_pilots or {}).get(row["profile"]),
        row.get("git_commit"),
        repo=repo,
        environment=environment,
    )
    if pilot_reason:
        return pilot_reason
    if row["profile"] == "gemini3.1-pro":
        pilot = (provider_pilots or {}).get(row["profile"]) or {}
        adc = environment.get("GOOGLE_APPLICATION_CREDENTIALS", "")
        if pilot.get("vertex_project") != row.get("vertex_project") or pilot.get("location") != "global":
            return "gemini3.1-pro provider pilot is not bound to the approved global project"
        if not adc or pilot.get("adc_sha256") != hash_file(Path(adc)):
            return "gemini3.1-pro provider pilot is not bound to the active ADC"
    if row.get("git_commit") and row["profile"] in {"opus5", "gemini3.1-pro"}:
        auth = (cached_auth or {}).get(row["profile"])
        if not auth or auth.get("status") != "ready":
            return f"{row['profile']} live authentication is unavailable"
        pilot = (provider_pilots or {}).get(row["profile"]) or {}
        if row["profile"] == "opus5" and (
            auth.get("account") != pilot.get("expected_account")
            or auth.get("config_dir") != pilot.get("config_dir")
        ):
            return "opus5 live authentication does not match the pilot route"
        if row["profile"] == "gemini3.1-pro" and any(
            auth.get(key) != pilot.get(key)
            for key in ("vertex_project", "adc_sha256", "location")
        ):
            return "gemini3.1-pro live authentication does not match the pilot route"
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
            cached_probes[profile] = codex_auth_probe()
        if row.get("git_commit") and row["profile"] not in cached_auth:
            if row["profile"] == "opus5":
                cached_auth[row["profile"]] = claude_auth_probe(environment)
            elif row["profile"] == "gemini3.1-pro":
                cached_auth[row["profile"]] = vertex_adc_probe(environment)
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


def provider_preflight() -> list[dict[str, str]]:
    """Check only local configuration; never call a paid provider."""
    claude_dir = Path(os.environ.get("CSD_CLAUDE_CONFIG_DIR", "/home/aadivyar/.claude-csd-synthesis"))
    return [
        {"profile": "gpt5.6-sol", "backend": "codex", "status": "not_checked_without_provider_call"},
        {"profile": "gemini3.1-pro", "backend": "vertex", "status": "credential_path_present" if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") else "credential_not_declared"},
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
    sources: dict[str, str] = {}
    for relative in source_paths:
        path = repo / relative
        if not path.is_file():
            raise ValueError(f"missing execution dependency: {relative}")
        sources[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    crane_sources = crane_source_hashes(repo)
    materialized = materialize_frozen_bar_sources(repo)
    vertex_project = verified_adc_project(os.environ)
    bound_rows = [dict(row, git_commit=commit, launch_commit=commit, bar_source_path=materialized[row["benchmark"]], **({"vertex_project": vertex_project} if row["profile"] == "gemini3.1-pro" else {})) for row in rows]
    pilots = provider_pilots or {}
    pilot_hash = provider_pilots_sha256(pilots)
    return {
        "version": 1,
        "git_commit": commit,
        "crane_commit": CANONICAL_CRANE_COMMIT,
        "crane_source_sha256": crane_sources,
        "source_sha256": sources,
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
    return {
        f"legacy/CRANE/{name}": hash_file(crane / name)
        for name in names
        if name
    }


def verified_adc_project(environment: dict[str, str]) -> str:
    """Read the project bound to the configured ADC file, never an endpoint token."""
    adc = environment.get("GOOGLE_APPLICATION_CREDENTIALS", "")
    if not adc:
        return ""
    try:
        payload = json.loads(Path(adc).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    project = payload.get("project_id")
    return str(project) if isinstance(project, str) and project else ""


def validate_manifest(repo: Path, payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Validate a manifest before any child process or provider is started."""
    if set(payload) != MANIFEST_KEYS:
        raise ConfigError("manifest contains unknown or missing top-level keys")
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
    rows = payload.get("jobs")
    if not isinstance(rows, list) or len(rows) != 31:
        raise ConfigError("manifest must contain exactly 31 Table 5--8 jobs")
    expected = build_scope(repo)
    immutable_fields = {
        "cell_id", "table", "table_cell_id", "benchmark", "dataset", "task",
        "profile", "generation_backend", "generation_model", "eval_model",
        "smiles_class", "token_budget", "beam_size", "adaptive_helper_mask",
        "helper_selection_policy", "max_iterations", "min_accuracy",
        "min_syntax_rate", "synthesis_max_tokens", "synthesis_reasoning_budget",
        "eval_sample_size",
        "heldout_sample_size", "eval_max_steps", "eval_max_seconds", "gpu_mem_util",
        "memory_reservation_mib", "gpu_scope", "gpu_count", "heldout_split_name",
        "heldout_split_file", "sample_count", "output_name", "heldout_output_json",
        "log_file", "cold_start", "bar_source_sha256",
    }
    for actual, frozen in zip(rows, expected):
        if set(actual) - JOB_KEYS:
            raise ConfigError(f"job contains unknown fields: {actual.get('cell_id', '<unknown>')}")
        for field in immutable_fields:
            if actual.get(field) != frozen.get(field):
                raise ConfigError(f"manifest field {field} differs for {frozen['cell_id']}")
        if actual.get("git_commit") != payload.get("git_commit"):
            raise ConfigError(f"row commit is not bound to manifest commit: {actual['cell_id']}")
        if actual.get("profile") == "gemini3.1-pro" and not actual.get("vertex_project"):
            raise ConfigError(f"Vertex row is missing its approved ADC project: {actual['cell_id']}")
        copied_bar = Path(str(actual.get("bar_source_path", "")))
        if copied_bar.is_absolute() or not (repo / copied_bar).is_file() or hashlib.sha256((repo / copied_bar).read_bytes()).hexdigest() != actual.get("bar_source_sha256"):
            raise ConfigError(f"bar source is not a copied immutable artifact: {actual['cell_id']}")
    source_paths = execution_source_paths(repo)
    recorded = payload.get("source_sha256")
    if not isinstance(recorded, dict) or set(recorded) != set(source_paths):
        raise ConfigError("manifest must hash every direct execution dependency")
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


def synthesis_environment(row: dict[str, Any], gpus: tuple[int, ...], inherited: dict[str, str], repo: Path) -> dict[str, str]:
    env = dict(inherited)
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu) for gpu in gpus)
    env["CSD_VLLM_GPU_MEMORY_UTILIZATION"] = str(row["gpu_mem_util"])
    env["CSD_VLLM_GPU_MEMORY_UTILIZATION_MAX"] = str(row["gpu_mem_util"])
    env["CSD_OUTPUT_DIR"] = str(repo / "outputs/generated")
    env["CSD_OUTPUT_NAME"] = str(row["output_name"])
    if row["profile"] == "opus5":
        env["CSD_CLAUDE_CONFIG_DIR"] = "/home/aadivyar/.claude-csd-synthesis"
        env["CSD_CLAUDE_EXPECTED_ACCOUNT"] = "ssdear@gmail.com"
    if row["profile"] == "gemini3.1-pro":
        for key in (
            "VERTEX_AI_PROJECT", "VERTEX_AI_LOCATION", "VERTEX_AI_BASE_URL",
            "VERTEX_AI_API_KEY", "VERTEX_AI_ACCESS_TOKEN", "GOOGLE_CLOUD_PROJECT",
            "GOOGLE_CLOUD_LOCATION", "GOOGLE_VERTEX_LOCATION", "GOOGLE_API_KEY",
            "GEMINI_API_KEY", "GOOGLE_GENAI_USE_VERTEXAI",
        ):
            env.pop(key, None)
        project = str(row.get("vertex_project") or "")
        if project:
            env["VERTEX_AI_PROJECT"] = project
            env["GOOGLE_CLOUD_PROJECT"] = project
        env["GOOGLE_CLOUD_LOCATION"] = "global"
        env["VERTEX_AI_LOCATION"] = "global"
        env["GOOGLE_VERTEX_LOCATION"] = "global"
        env["VERTEX_AI_BASE_URL"] = "https://aiplatform.googleapis.com/v1"
        env["CSD_GEMINI_BACKEND"] = "vertex"
        env["CSD_GEMINI_MODEL"] = "gemini-3.1-pro-preview"
    if row["dataset"] == "smiles":
        env["CSD_CONSTRAINED_TEMPERATURE"] = "0.7"
    return env


def heldout_command(row: dict[str, Any], python: Path, compiled_csd: Path) -> list[str]:
    cmd = [str(python), "-m", "synthesis.scripts.reevaluate_compiled_csd", str(compiled_csd), "--dataset", row["dataset"], "--eval-model", EVAL_MODEL, "--eval-backend", "vllm", "--device", "auto", "--sample-size", str(row["heldout_sample_size"]), "--max-steps", str(row["eval_max_steps"]), "--step-token-budget", str(row["token_budget"]), "--max-seconds-per-example", "600", "--vllm-gpu-memory-utilization", str(row["gpu_mem_util"]), "--vllm-tensor-parallel-size", "1", "--output-json", str(row["heldout_output_json"]), "--provenance-cell-id", str(row["cell_id"]), "--provenance-manifest-commit", str(row.get("manifest_commit") or row.get("git_commit") or "")]
    if row["dataset"] == "gsm_symbolic":
        cmd += ["--gsm-split-file", str(row["heldout_split_file"]), "--gsm-split-name", "test"]
    elif row["dataset"] == "spider":
        cmd += ["--spider-split-file", str(row["heldout_split_file"]), "--spider-split-name", "test"]
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
    if (
        not args.gpus
        or len(set(args.gpus)) != len(args.gpus)
        or not set(args.gpus).issubset({0, 1, 2, 3})
    ):
        raise ConfigError("GPU scope must be a nonempty unique subset of 0,1,2,3")


def controller_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dispatch the validated Table 5--8 manifest")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--gpus", type=lambda raw: tuple(int(x) for x in raw.split(",") if x.strip()), default=(0, 1, 2, 3))
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--export", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def validate_profile_gates(rows: list[dict[str, Any]], environment: dict[str, str]) -> None:
    for row in rows:
        profile = row["profile"]
        LOGGER.info("[tableq] profile-gate profile=%s", profile)
        if profile == "opus5":
            if environment.get("CSD_CLAUDE_CONFIG_DIR") != "/home/aadivyar/.claude-csd-synthesis" or environment.get("CSD_CLAUDE_EXPECTED_ACCOUNT") != "ssdear@gmail.com":
                raise ConfigError("opus5 requires the exact Max config directory and account")
        elif profile == "gemini3.1-pro":
            adc = environment.get("GOOGLE_APPLICATION_CREDENTIALS", "")
            if not adc or not Path(adc).is_file() or environment.get("GOOGLE_CLOUD_LOCATION", "global") != "global" or not environment.get("GOOGLE_CLOUD_PROJECT"):
                LOGGER.error("[tableq] auth-block profile=gemini3.1-pro reason=vertex-adc")
                raise ConfigError("gemini3.1-pro requires valid global Vertex ADC configuration")
            if any(key in environment for key in ("GOOGLE_API_KEY", "GEMINI_API_KEY", "GOOGLE_GENAI_USE_VERTEXAI")):
                LOGGER.error("[tableq] auth-block profile=gemini3.1-pro reason=api-key-fallback")
                raise ConfigError("gemini3.1-pro campaign rejects API-key fallback configuration")


def load_terminal_results(repo: Path, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for row in rows:
        path = repo / str(row["heldout_output_json"])
        if not heldout_artifact_is_valid(path, row):
            raise ConfigError(f"held-out artifact is incomplete or unbound: {path}")
        LOGGER.info("[tableq] artifact-valid cell=%s", row["cell_id"])
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["cell_id"] = row["cell_id"]
        payload.setdefault("sample_count", row["sample_count"])
        values.append(payload)
    return values


def controller_main(args: argparse.Namespace) -> int:
    validate_controller_paths(args)
    with controller_lock(args.state_dir):
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
    ready_rows, blocked_rows = partition_profile_readiness(
        rows,
        os.environ,
        repo=repo,
        provider_pilots=payload.get("provider_pilots"),
    )
    args.state_dir.mkdir(parents=True, exist_ok=True)
    write_state(args.state_dir / "controller.json", {"manifest_sha256": manifest_sha, "status": "validated", "scope": len(rows), "ready": len(ready_rows), "auth_blocked": len(blocked_rows)})
    logging.basicConfig(filename=args.log, level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    LOGGER.info("[tableq] input manifest sha256=%s scope=%d", manifest_sha, len(rows))
    from scripts.runtime.run_cold_synthesis_queue import gpu_memory_snapshot
    rows = [dict(row, manifest_sha256=manifest_sha, manifest_commit=manifest_sha) for row in ready_rows]
    results = dispatch(rows, repo=repo, python=args.python, state_dir=args.state_dir, allowed=args.gpus, snapshot=gpu_memory_snapshot, poll_seconds=args.poll_seconds)
    for blocked in blocked_rows:
        blocked_row = dict(blocked, manifest_sha256=manifest_sha, manifest_commit=manifest_sha)
        write_state(_state_path(args.state_dir, blocked_row), blocked_row)
        results.append(blocked_row)
    if any(result.get("status") == "failed" for result in results):
        return 1
    if blocked_rows:
        write_state(args.state_dir / "controller.json", {"manifest_sha256": manifest_sha, "status": "pending", "scope": len(rows) + len(blocked_rows), "ready": len(rows), "auth_blocked": len(blocked_rows)})
        return 0
    values = load_terminal_results(repo, rows)
    controller_manifest_path(args.manifest, args.export)
    export_results(rows, values, args.export)
    write_state(args.state_dir / "controller.json", {"manifest_sha256": manifest_sha, "status": "complete", "scope": len(rows), "export": str(args.export)})
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
            and int(config.get("max_iterations") or -1) == max_iterations
            and author.get("backend") == row["generation_backend"]
            and author.get("model") == row["generation_model"]
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


def _validated_compiled_output(
    repo: Path,
    output_name: str,
    *,
    min_accuracy: float,
    min_syntax_rate: float,
    job: dict[str, Any],
) -> Path | None:
    """Select only a compiled strategy proven to belong to this cold row."""
    from scripts.runtime.run_cold_synthesis_queue import current_run_dir

    run_dir = current_run_dir(repo, output_name)
    if run_dir is None:
        return None
    success_report = run_dir / "results" / "success_report.json"
    failure_report = run_dir / "results" / "failure_report.json"
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
    if not exhausted:
        compiled_dir = Path(str(report.get("compiled_dir") or ""))
    else:
        candidates: list[tuple[float, float, float, int, Path]] = []
        seen_attempt_numbers: set[int] = set()
        for attempt in report.get("attempts") or []:
            compilation = attempt.get("compilation") or {}
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
            candidates.append(
                (
                    shortfall,
                    -accuracy,
                    -syntax,
                    attempt_number,
                    Path(str(compilation["output_dir"])),
                )
            )
        if not candidates:
            return None
        compiled_dir = min(candidates, key=lambda item: item[:4])[-1]
    if not compiled_dir.is_absolute():
        compiled_dir = repo / compiled_dir
    candidate = compiled_dir / "GeneratedCSD.py"
    return candidate if candidate.is_file() else None


def _compiled_output(repo: Path, row: dict[str, Any]) -> Path | None:
    try:
        cold_compiled_csd = globals().get("cold_compiled_csd")
        if cold_compiled_csd is None:
            cold_compiled_csd = _validated_compiled_output
        cold_job = dict(
            row,
            train_sample_size=row["eval_sample_size"],
            train_split_file=row.get("heldout_split_file"),
            train_split_name="train",
        )
        candidate = cold_compiled_csd(
            repo,
            str(row["output_name"]),
            min_accuracy=float(row["min_accuracy"]),
            min_syntax_rate=float(row["min_syntax_rate"]),
            job=cold_job,
        )
        if candidate is not None:
            return candidate
        return None
    except (ImportError, OSError, TypeError):
        run_dir = None
    if run_dir is None:
        return None
    results = run_dir / "results"
    if not (results / "success_report.json").is_file() and not (results / "failure_report.json").is_file():
        return None
    candidate = run_dir / "compiled" / "GeneratedCSD.py"
    if candidate.is_file():
        return candidate
    candidates = sorted(run_dir.rglob("GeneratedCSD.py"))
    return candidates[0] if candidates else None


def run_row(row: dict[str, Any], *, repo: Path, python: Path, state_dir: Path, gpus: tuple[int, ...], dry_run: bool = False, runner: Any = None) -> dict[str, Any]:
    """Run one synthesis then its held-out evaluation with restart state."""
    path = _state_path(state_dir, row)
    if dry_run:
        return {"cell_id": row["cell_id"], "status": "dry_run", "command": synthesis_command(row, python)}
    def save(payload: dict[str, Any]) -> None:
        with state_lock(state_dir):
            write_state(path, payload)

    with state_lock(state_dir):
        prior = read_state(path) or {"cell_id": row["cell_id"], "status": "pending", "phase": "synthesis"}
        if prior.get("manifest_sha256") not in (None, row.get("manifest_sha256")):
            raise ConfigError(f"state is bound to a different manifest: {row['cell_id']}")
        if prior.get("status") in {"complete", "failed"}:
            if prior.get("status") == "complete":
                output = Path(str(prior.get("heldout_output_json", repo / row["heldout_output_json"])))
                if not output.is_absolute():
                    output = repo / output
                current = artifact_fingerprint(output)
                bound_row = dict(row, compiled_csd_path=prior.get("compiled_csd_path"), compiled_sha256=prior.get("compiled_sha256"), manifest_commit=prior.get("manifest_commit"))
                if current is None or current.get("sha256") != prior.get("heldout_sha256") or not heldout_artifact_is_valid(output, bound_row):
                    raise ConfigError(f"completed state failed artifact revalidation: {row['cell_id']}")
            return prior
        if prior.get("status") == "running" and child_is_same_process(prior):
            LOGGER.info("[tableq] surviving child cell=%s phase=%s pid=%s", row["cell_id"], prior.get("phase"), prior.get("pid"))
            return prior
        phase = str(prior.get("phase") or "synthesis")
    env = synthesis_environment(row, gpus, os.environ, repo)
    command = synthesis_command(row, python)
    def start(argv: list[str]):
        if runner is not None:
            return runner(argv, cwd=repo, env=env)
        return subprocess.Popen(argv, cwd=repo, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    recovered = None
    if phase == "synthesis" and prior.get("status") == "running":
        same_manifest = prior.get("manifest_sha256") == row.get("manifest_sha256") and prior.get("cell_id") == row.get("cell_id")
        latest = repo / "outputs" / "generated" / str(row["output_name"]) / "latest_run.txt"
        fresh_latest = artifact_is_new_or_replaced(latest, prior.get("output_before"))
        if same_manifest and fresh_latest:
            recovered = _compiled_output(repo, row)
    if recovered is not None:
        prior = dict(prior, phase="heldout", compiled_csd_path=str(recovered))
        save(prior)
        phase = "heldout"
    elif phase == "synthesis" and prior.get("status") == "running":
        failed = dict(prior, status="failed", reason="synthesis child ended without a new bound compiled artifact", exit_code=1)
        failed.pop("pid", None); failed.pop("pid_start", None)
        save(failed)
        return failed

    if phase == "synthesis":
        latest = repo / "outputs" / "generated" / str(row["output_name"]) / "latest_run.txt"
        before_output = artifact_fingerprint(latest)
        process = start(command)
        LOGGER.info("[tableq] launch cell=%s phase=synthesis gpus=%s", row["cell_id"], gpus)
        running = dict(prior, manifest_sha256=row.get("manifest_sha256"), manifest_commit=row.get("manifest_commit") or row.get("manifest_sha256") or row.get("git_commit"), cell_id=row["cell_id"], status="running", phase="synthesis", pid=process.pid, pid_start=process_start_identity(process.pid), output_before=before_output)
        save(running)
        _output, _ = process.communicate()
        exit_code = process.returncode
        running.pop("pid", None); running.pop("pid_start", None)
        has_new_run = artifact_is_new_or_replaced(latest, before_output)
        if not has_new_run:
            failed = dict(running, status="failed", exit_code=1, reason="synthesis returned success without a new run report")
            save(failed)
            return failed
        compiled = _compiled_output(repo, row)
        if compiled is None:
            failed = dict(running, status="failed", exit_code=exit_code or 1, reason="synthesis failed or produced no recoverable compiled artifact")
            save(failed)
            return failed
        prior = dict(running, phase="heldout", compiled_csd_path=str(compiled), compiled_sha256=artifact_fingerprint(compiled)["sha256"], synthesis_exit_code=exit_code)
        save(prior)

    compiled = Path(str(prior["compiled_csd_path"]))
    compiled_fingerprint = artifact_fingerprint(compiled)
    if compiled_fingerprint is None:
        failed = dict(prior, status="failed", reason="state-bound compiled artifact is missing", exit_code=1)
        failed.pop("pid", None)
        failed.pop("pid_start", None)
        save(failed)
        return failed
    if prior.get("compiled_sha256") and compiled_fingerprint["sha256"] != prior["compiled_sha256"]:
        failed = dict(prior, status="failed", reason="compiled artifact changed before held-out evaluation", exit_code=1)
        save(failed)
        return failed
    final_output = repo / str(row["heldout_output_json"])
    final_output.parent.mkdir(parents=True, exist_ok=True)
    before = artifact_fingerprint(final_output)
    temporary = final_output.with_name(f".{final_output.name}.{os.getpid()}.tmp")
    heldout_row = dict(row, heldout_output_json=str(temporary))
    process = start(heldout_command(heldout_row, python, compiled))
    LOGGER.info("[tableq] launch cell=%s phase=heldout gpus=%s", row["cell_id"], gpus)
    running = dict(prior, status="running", phase="heldout", pid=process.pid, pid_start=process_start_identity(process.pid), heldout_output_before=before)
    save(running)
    _output, _ = process.communicate()
    exit_code = process.returncode
    running.pop("pid", None); running.pop("pid_start", None)
    if exit_code != 0 or not artifact_is_new_or_replaced(temporary, None) or not heldout_artifact_is_valid(temporary, heldout_row):
        failed = dict(running, status="failed", exit_code=exit_code or 1, reason="held-out evaluation failed or produced no artifact")
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


def dispatch(rows: list[dict[str, Any]], *, repo: Path, python: Path, state_dir: Path, allowed: tuple[int, ...], snapshot: Any, poll_seconds: float = 30.0, dry_run: bool = False) -> list[dict[str, Any]]:
    """Dispatch only when a scoped GPU fits; keep polling while work remains."""
    results: list[dict[str, Any]] = []
    pending = list(rows)
    reservations: dict[int, int] = {}
    while pending:
        live = snapshot()
        next_pending: list[dict[str, Any]] = []
        admitted: list[tuple[dict[str, Any], tuple[int, ...], int]] = []
        surviving_child = False
        for row in pending:
            state = read_state(_state_path(state_dir, row))
            if state and state.get("status") == "running" and child_is_same_process(state):
                LOGGER.info(
                    "[tableq] poll-surviving-child cell=%s phase=%s pid=%s",
                    row["cell_id"],
                    state.get("phase"),
                    state.get("pid"),
                )
                next_pending.append(row)
                surviving_child = True
                continue
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
                    pool.submit(run_row, row, repo=repo, python=python, state_dir=state_dir, gpus=gpus, dry_run=dry_run): (row, gpus, demand)
                    for row, gpus, demand in admitted
                }
                for future, (row, gpus, demand) in futures.items():
                    result = future.result()
                    if result.get("status") == "running":
                        next_pending.append(row)
                    else:
                        results.append(result)
                    for gpu in gpus:
                        reservations[gpu] -= demand
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


def controller_lock_path(state_dir: Path) -> Path:
    return state_dir / "table5_8.controller.lock"


@contextmanager
def controller_lock(state_dir: Path):
    """Keep exactly one Table 5--8 controller alive for this state directory."""
    state_dir.mkdir(parents=True, exist_ok=True)
    handle = controller_lock_path(state_dir).open("a+", encoding="utf-8")
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
    cells: list[dict[str, Any]] = []
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        value = dict(by_id[row["cell_id"]])
        value.update({"cell_id": row["cell_id"], "table": row["table"], "table_cell_id": row["table_cell_id"], "benchmark": row["benchmark"]})
        if row["benchmark"] != "smiles":
            metric = "accuracy"
            if not isinstance(value.get(metric), (int, float)):
                raise ConfigError(f"missing {metric} for {row['cell_id']}")
            value["cw"] = constrained_window_rate(value)
        else:
            trial = value.get("smiles_paper_trial") or {}
            count = trial.get("sample_count")
            unique = trial.get("unique_valid_count")
            if not isinstance(count, int) or count <= 0 or not isinstance(unique, int) or unique < 0 or unique > count:
                raise ConfigError(f"missing validated smiles_paper_trial for {row['cell_id']}")
            value["sample_count"] = count
            value["unique_valid_rate"] = unique / count
        groups.setdefault(row["table_cell_id"], []).append(value)
    for cell_id, group in groups.items():
        item = {"table_cell_id": cell_id, "table": group[0]["table"], "benchmark": group[0]["benchmark"]}
        if group[0]["benchmark"] == "smiles":
            item["unique_valid_rate"] = weighted_smiles_rate(group)
            item["sample_count"] = sum(int(v["sample_count"]) for v in group)
        else:
            metric = "accuracy"
            item[metric] = group[0][metric]
            item["cw"] = group[0]["cw"]
        cells.append(item)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(output.suffix + ".tmp")
    temp.write_text(json.dumps({"version": 1, "cells": cells}, indent=2) + "\n", encoding="utf-8")
    temp.replace(output)


def main() -> int:
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
    if len(rows) != 31:
        raise SystemExit(f"scope error: expected 31 rows, got {len(rows)}")
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
