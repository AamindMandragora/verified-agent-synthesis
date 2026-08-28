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
import time
from pathlib import Path
from typing import Any, Iterable

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from synthesis.evaluate.benchmarks.gsm_symbolic.prompts import GSM_CRANE_COT_TASK
from synthesis.run_constants import VLLM_GPU_MEMORY_UTILIZATION_BY_MODEL

LOGGER = logging.getLogger("table5-8-queue")
class ConfigError(ValueError):
    """The manifest or runtime configuration cannot be safely launched."""

EVAL_MODEL = "Qwen/Qwen3.5-2B"
GPU_SAFETY_MIB = 2_000
CANONICAL_CRANE_COMMIT = "616379ce33ac6245933c16e6264b41f7d5800183"
AUTHOR_TOKEN_BUDGET = 32768
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
    "environment/benchmark_splits/gsm_symbolic_crane_proportional_49x49_seed123.json",
    "environment/benchmark_splits/spider_dev_proportional_300x300_seed334.json",
    "synthesis/evaluate/benchmarks/common/parser_utils.py",
    "synthesis/verify/tooling.py",
    ".context/run_post14b_rebar_queue.py",
    "synthesis/evaluate/baselines/crane_repo_runner.py",
)


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
    cmd = [str(python), "-m", "synthesis.run_synthesis", "--task", row["task"], "--dataset", row["dataset"], "--min-accuracy", str(row["min_accuracy"]), "--min-syntax-rate", str(row["min_syntax_rate"]), "--max-iterations", "40", "--eval-model", EVAL_MODEL, "--eval-sample-size", str(row["eval_sample_size"]), "--eval-max-steps", str(row["eval_max_steps"]), "--eval-step-token-budget", str(row["token_budget"]), "--eval-max-seconds-per-example", "600", "--eval-min-examples-before-threshold-stop", str(row["eval_sample_size"]), "--generation-model", row["generation_model"], "--generation-backend", row["generation_backend"], "--synthesis-max-tokens", str(row["synthesis_max_tokens"]), "--device", "auto", "--vllm-gpu-memory-utilization", str(row["gpu_mem_util"]), "--refinement-beam-size", str(row["beam_size"]), "--helper-selection-policy", row["helper_selection_policy"]]
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


def helper_call_weight(value: dict[str, Any]) -> float:
    """Compute CW from row-level helper evidence; no evidence means zero."""
    calls = value.get("helper_calls")
    if not isinstance(calls, list):
        raise ConfigError("result is missing row-level helper_calls evidence")
    if not calls:
        return 0.0
    if any(not isinstance(call, dict) or not isinstance(call.get("used"), bool) for call in calls):
        raise ConfigError("helper_calls evidence must contain boolean used values")
    return sum(1 for call in calls if call["used"]) / len(calls)


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


def manifest_payload(repo: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--", *SOURCE_PATHS],
        cwd=repo, check=True, capture_output=True, text=True,
    ).stdout.strip()
    if dirty:
        raise ConfigError("execution dependencies have uncommitted changes")
    validate_crane_checkout(repo)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True).stdout.strip()
    sources: dict[str, str] = {}
    for relative in SOURCE_PATHS:
        path = repo / relative
        if not path.is_file():
            raise ValueError(f"missing execution dependency: {relative}")
        sources[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    materialized = materialize_frozen_bar_sources(repo)
    bound_rows = [dict(row, git_commit=commit, launch_commit=commit, bar_source_path=materialized[row["benchmark"]]) for row in rows]
    return {"version": 1, "git_commit": commit, "crane_commit": CANONICAL_CRANE_COMMIT, "source_sha256": sources, "jobs": bound_rows}


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


def validate_manifest(repo: Path, payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Validate a manifest before any child process or provider is started."""
    if payload.get("crane_commit") != CANONICAL_CRANE_COMMIT:
        raise ConfigError("manifest is not bound to the approved CRANE checkout")
    validate_crane_checkout(repo)
    rows = payload.get("jobs")
    if not isinstance(rows, list) or len(rows) != 31:
        raise ConfigError("manifest must contain exactly 31 Table 5--8 jobs")
    expected = build_scope(repo)
    immutable_fields = {
        "cell_id", "table", "table_cell_id", "benchmark", "dataset", "task",
        "profile", "generation_backend", "generation_model", "eval_model",
        "smiles_class", "token_budget", "beam_size", "adaptive_helper_mask",
        "helper_selection_policy", "max_iterations", "min_accuracy",
        "min_syntax_rate", "synthesis_max_tokens", "eval_sample_size",
        "heldout_sample_size", "eval_max_steps", "eval_max_seconds", "gpu_mem_util",
        "memory_reservation_mib", "gpu_scope", "gpu_count", "heldout_split_name",
        "heldout_split_file", "sample_count", "output_name", "heldout_output_json",
        "log_file", "cold_start", "bar_source_sha256",
    }
    for actual, frozen in zip(rows, expected):
        for field in immutable_fields:
            if actual.get(field) != frozen.get(field):
                raise ConfigError(f"manifest field {field} differs for {frozen['cell_id']}")
        if actual.get("git_commit") != payload.get("git_commit"):
            raise ConfigError(f"row commit is not bound to manifest commit: {actual['cell_id']}")
        copied_bar = Path(str(actual.get("bar_source_path", "")))
        if copied_bar.is_absolute() or not (repo / copied_bar).is_file() or hashlib.sha256((repo / copied_bar).read_bytes()).hexdigest() != actual.get("bar_source_sha256"):
            raise ConfigError(f"bar source is not a copied immutable artifact: {actual['cell_id']}")
    recorded = payload.get("source_sha256")
    if not isinstance(recorded, dict) or set(recorded) != set(SOURCE_PATHS):
        raise ConfigError("manifest must hash every direct execution dependency")
    for relative in SOURCE_PATHS:
        path = repo / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != recorded[relative]:
            raise ConfigError(f"execution dependency changed: {relative}")
    dirty = subprocess.run(["git", "status", "--porcelain", "--", *SOURCE_PATHS], cwd=repo, check=True, capture_output=True, text=True).stdout.strip()
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
        for key in ("GOOGLE_API_KEY", "GEMINI_API_KEY", "GOOGLE_GENAI_USE_VERTEXAI"):
            env.pop(key, None)
        env["GOOGLE_CLOUD_LOCATION"] = "global"
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


def heldout_artifact_is_valid(path: Path, row: dict[str, Any]) -> bool:
    """Validate the cold artifact while honoring this row's token budget."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        expected = int(row["heldout_sample_size"])
        metrics = payload.get("metrics") or {}
        answers = payload.get("answers")
        provenance = payload.get("reevaluation_provenance") or {}
        compiled = Path(str(provenance["compiled_csd_path"]))
        compiled_hash = hashlib.sha256(compiled.read_bytes()).hexdigest()
        split = payload.get("eval_split") or {}
        prefix = "gsm" if row["dataset"] == "gsm_symbolic" else "spider"
        split_ok = row["dataset"] == "smiles" or (split.get(f"{prefix}_split_name") == "test" and str(split.get(f"{prefix}_split_file")) == str(row["heldout_split_file"]))
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
    repo = Path.cwd()
    manifest_bytes = args.manifest.read_bytes()
    payload = json.loads(manifest_bytes)
    rows = validate_manifest(repo, payload)
    validate_profile_gates(rows, os.environ)
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    args.state_dir.mkdir(parents=True, exist_ok=True)
    write_state(args.state_dir / "controller.json", {"manifest_sha256": manifest_sha, "status": "validated", "scope": len(rows)})
    logging.basicConfig(filename=args.log, level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    LOGGER.info("[tableq] input manifest sha256=%s scope=%d", manifest_sha, len(rows))
    if args.dry_run:
        for row in rows:
            LOGGER.info("[tableq] dry-run cell=%s", row["cell_id"])
            print(row["cell_id"], shlex.join(synthesis_command(row, args.python)))
        return 0
    from scripts.runtime.run_cold_synthesis_queue import gpu_memory_snapshot
    rows = [dict(row, manifest_sha256=manifest_sha) for row in rows]
    results = dispatch(rows, repo=repo, python=args.python, state_dir=args.state_dir, allowed=args.gpus, snapshot=gpu_memory_snapshot, poll_seconds=args.poll_seconds)
    if any(result.get("status") == "failed" for result in results):
        return 1
    values = load_terminal_results(repo, rows)
    if args.export is None:
        raise ConfigError("--export is required for a real controller run")
    controller_manifest_path(args.manifest, args.export)
    export_results(rows, values, args.export)
    write_state(args.state_dir / "controller.json", {"manifest_sha256": manifest_sha, "status": "complete", "scope": len(rows), "export": str(args.export)})
    return 0


def _state_path(state_dir: Path, row: dict[str, Any]) -> Path:
    return state_dir / f"{row['cell_id']}.json"


def _compiled_output(repo: Path, row: dict[str, Any]) -> Path | None:
    explicit = row.get("compiled_csd_path")
    if explicit:
        path = Path(str(explicit))
        if not path.is_absolute():
            path = repo / path
        return path if path.is_file() else None
    try:
        from scripts.runtime.run_cold_synthesis_queue import current_run_dir
        run_dir = current_run_dir(repo, str(row["output_name"]))
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
                if current is None or current.get("sha256") != prior.get("heldout_sha256") or not heldout_artifact_is_valid(output, row):
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
        recovered = _compiled_output(repo, row) if phase == "synthesis" else None
        if recovered is not None:
            prior = dict(prior, phase="heldout", compiled_csd_path=str(recovered))
            write_state(path, prior)
            phase = "heldout"
        if phase == "synthesis":
            before_output = None
            process = start(command)
            LOGGER.info("[tableq] launch cell=%s phase=synthesis gpus=%s", row["cell_id"], gpus)
            running = dict(prior, manifest_sha256=row.get("manifest_sha256"), status="running", phase="synthesis", pid=process.pid, pid_start=process_start_identity(process.pid), output_before=before_output)
            write_state(path, running)
            _output, _ = process.communicate()
            exit_code = process.returncode
            running.pop("pid", None); running.pop("pid_start", None)
            if exit_code != 0:
                failed = dict(running, status="failed", exit_code=exit_code, reason="synthesis failed")
                write_state(path, failed)
                return failed
            compiled = _compiled_output(repo, row)
            if compiled is None:
                failed = dict(running, status="failed", exit_code=1, reason="synthesis returned success without a compiled artifact")
                write_state(path, failed)
                return failed
            prior = dict(running, phase="heldout", compiled_csd_path=str(compiled))
            write_state(path, prior)
        compiled = Path(str(prior["compiled_csd_path"]))
        final_output = repo / str(row["heldout_output_json"])
        final_output.parent.mkdir(parents=True, exist_ok=True)
        before = artifact_fingerprint(final_output)
        temporary = final_output.with_name(f".{final_output.name}.{os.getpid()}.tmp")
        heldout_row = dict(row, heldout_output_json=str(temporary))
        process = start(heldout_command(heldout_row, python, compiled))
        LOGGER.info("[tableq] launch cell=%s phase=heldout gpus=%s", row["cell_id"], gpus)
        running = dict(prior, status="running", phase="heldout", pid=process.pid, pid_start=process_start_identity(process.pid), heldout_output_before=before)
        write_state(path, running)
        _output, _ = process.communicate()
        exit_code = process.returncode
        running.pop("pid", None); running.pop("pid_start", None)
        if exit_code != 0 or not artifact_is_new_or_replaced(temporary, None) or not heldout_artifact_is_valid(temporary, heldout_row):
            failed = dict(running, status="failed", exit_code=exit_code or 1, reason="held-out evaluation failed or produced no artifact")
            write_state(path, failed)
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
        write_state(path, complete)
        return complete


def dispatch(rows: list[dict[str, Any]], *, repo: Path, python: Path, state_dir: Path, allowed: tuple[int, ...], snapshot: Any, poll_seconds: float = 30.0, dry_run: bool = False) -> list[dict[str, Any]]:
    """Dispatch only when a scoped GPU fits; keep polling while work remains."""
    results: list[dict[str, Any]] = []
    pending = list(rows)
    reservations: dict[int, int] = {}
    while pending:
        live = snapshot()
        next_pending: list[dict[str, Any]] = []
        progressed = False
        for row in pending:
            gpus = choose_gpus(row, live, reservations, live, allowed)
            if gpus is None:
                next_pending.append(row)
                continue
            progressed = True
            LOGGER.info("[tableq] admission cell=%s gpus=%s", row["cell_id"], gpus)
            for gpu in gpus:
                reservations[gpu] = reservations.get(gpu, 0) + _demand(row, int(live[gpu]["total_mib"]))
            result = run_row(row, repo=repo, python=python, state_dir=state_dir, gpus=gpus, dry_run=dry_run)
            if result.get("status") == "running":
                next_pending.append(row)
            else:
                results.append(result)
            for gpu in gpus:
                reservations[gpu] -= _demand(row, int(live[gpu]["total_mib"]))
        pending = next_pending
        if pending and not progressed:
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
    return state_dir / "table5_8.lock"


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
            metric = "execution_accuracy" if row["benchmark"] == "spider" else "accuracy"
            if not isinstance(value.get(metric), (int, float)):
                raise ConfigError(f"missing {metric} for {row['cell_id']}")
            value["cw"] = helper_call_weight(value)
        elif not isinstance(value.get("unique_valid_rate"), (int, float)):
            raise ConfigError(f"missing unique_valid_rate for {row['cell_id']}")
        groups.setdefault(row["table_cell_id"], []).append(value)
    for cell_id, group in groups.items():
        item = {"table_cell_id": cell_id, "table": group[0]["table"], "benchmark": group[0]["benchmark"]}
        if group[0]["benchmark"] == "smiles":
            item["unique_valid_rate"] = weighted_smiles_rate(group)
            item["sample_count"] = sum(int(v["sample_count"]) for v in group)
        else:
            metric = "execution_accuracy" if group[0]["benchmark"] == "spider" else "accuracy"
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
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    rows = build_scope(args.repo)
    if len(rows) != 31:
        raise SystemExit(f"scope error: expected 31 rows, got {len(rows)}")
    if args.dry_run:
        for row in rows:
            print(row["cell_id"], shlex.join(synthesis_command(row, Path(sys.executable))))
        return 0
    payload = manifest_payload(args.repo, rows)
    target = args.manifest or args.repo / "outputs/controlled_comparison/table5_8_manifest.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
