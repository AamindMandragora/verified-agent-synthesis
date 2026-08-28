#!/usr/bin/env python3
"""Restart-safe held-out baseline queue for the 2026-08-28 paper gaps.

The queue only runs the fixed-strategy evaluator on the exact requested cells.
It records source fingerprints and output provenance, admits a job only when
the live GPU snapshot plus every queued reservation leaves a 2 GiB margin,
and uses the existing post-14B atomic claim helper for at-most-once reruns.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import logging
import os
import re
import subprocess
import sys
import math
import time
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

if __package__ in {None, ""}:
    # Direct ``python scripts/runtime/run_paper_baseline_queue.py`` is a
    # supported operational entry point from the repository root.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.runtime import run_cold_synthesis_queue as cold_queue


LOG = logging.getLogger("paper-baseline-queue")
GPU_SAFETY_MIB = 2_000
STRATEGIES = ("unconstrained", "gcd", "crane", "itergen")
SMILES_CLASSES = ("acrylates", "chain_extenders", "isocyanates")
MODELS = (
    ("qwen25-1p5b", "Qwen/Qwen2.5-1.5B-Instruct"),
    ("qwen25-7b", "Qwen/Qwen2.5-7B-Instruct"),
    ("qwen35-2b", "Qwen/Qwen3.5-2B"),
    ("qwen35-4b", "Qwen/Qwen3.5-4B"),
)
MODEL_SLUG_BY_NAME = {model: slug for slug, model in MODELS}
GSM_MODELS = MODELS[2:]
BASELINE_GPU_MIB = {
    model: cold_queue.EXPECTED_RUNTIME_BY_MODEL[model]["memory_reservation_mib"]
    for _, model in MODELS
}
SPLITS = {
    "gsm_symbolic": Path("environment/benchmark_splits/gsm_symbolic_crane_proportional_49x49_seed123.json"),
    "spider": Path("environment/benchmark_splits/spider_dev_proportional_300x300_seed334.json"),
}
TASKS = {
    "gsm_symbolic": cold_queue.GSM_TASK,
    "spider": cold_queue.SPIDER_TASK,
    "smiles": cold_queue.SMILES_TASK,
}
SOURCE_PATHS = (
    "scripts/runtime/run_paper_baseline_queue.py",
    "scripts/runtime/run_cold_synthesis_queue.py",
    "synthesis/evaluate/run_legacy_fixed_strategy.py",
    "synthesis/evaluate/benchmarks/gsm_symbolic/eval_logic.py",
    "synthesis/evaluate/benchmarks/gsm_symbolic/dataset.py",
    "synthesis/evaluate/baselines/crane_repo_runner.py",
    "synthesis/evaluate/benchmarks/sql_spider/eval_logic.py",
    "synthesis/evaluate/benchmarks/smiles/eval_logic.py",
    "synthesis/evaluate/benchmarks/smiles/dataset.py",
    "synthesis/evaluate/benchmarks/smiles/metrics.py",
    "synthesis/run_synthesis.py",
    "synthesis/generate/generator.py",
    "synthesis/generate/provider_names.py",
    "synthesis/generate/prompts.py",
    "synthesis/generate/rationale.py",
    "synthesis/evaluate/feedback_loop.py",
    "synthesis/evaluate/evaluator.py",
    "synthesis/scripts/reevaluate_compiled_csd.py",
    "synthesis/evaluate/metrics.py",
    "synthesis/prompt_rendering/models/feedback_loop.py",
    ".context/run_post14b_rebar_queue.py",
)
CRANE_COMMIT = "616379ce33ac6245933c16e6264b41f7d5800183"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _slug(value: str) -> str:
    return value.lower().replace("/", "-").replace(".", "").replace("_", "-")


def _git_commit(repo: Path) -> str:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, text=True, capture_output=True, check=False)
    except OSError:
        return ""
    value = result.stdout.strip()
    return value if re.fullmatch(r"[0-9a-f]{40}", value) else ""


def _source_hashes(repo: Path) -> dict[str, str]:
    return {relative: sha256_file(repo / relative) for relative in SOURCE_PATHS if (repo / relative).is_file()}


def _crane_commit(repo: Path) -> str:
    crane = repo / "legacy" / "CRANE"
    try:
        result = subprocess.run(["git", "-C", str(crane), "rev-parse", "HEAD"], text=True, capture_output=True, check=False)
    except OSError:
        return ""
    value = result.stdout.strip()
    if value != CRANE_COMMIT:
        return value
    clean = subprocess.run(["git", "-C", str(crane), "status", "--porcelain", "--untracked-files=all"], text=True, capture_output=True, check=False)
    return value if not clean.stdout.strip() else ""


def _dirty_source_paths(repo: Path) -> set[str]:
    dirty: set[str] = set()
    # Startup is only safe from a fully committed checkout: a changed
    # dependency outside SOURCE_PATHS can still alter provider or evaluator
    # behavior, so reject every tracked staged/unstaged change.
    for diff_args in (("git", "diff", "--name-only"), ("git", "diff", "--cached", "--name-only")):
        result = subprocess.run(list(diff_args), cwd=repo, text=True, capture_output=True, check=False)
        dirty.update(line for line in result.stdout.splitlines() if line)
    return dirty


def _split_fields(repo: Path, dataset: str) -> tuple[str, str]:
    path = repo / SPLITS[dataset]
    return str(path), sha256_file(path) if path.is_file() else ""


def _row(
    repo: Path,
    *,
    dataset: str,
    model_slug: str,
    model: str,
    strategy: str,
    smiles_class: str | None = None,
) -> dict[str, Any]:
    suffix = f"-{smiles_class}" if smiles_class else ""
    cell_id = f"{dataset.replace('_symbolic', '')}-{model_slug}-{strategy}{suffix}"
    if dataset == "gsm_symbolic":
        sample_size = 49
        eval_steps = 900
        split_file, split_sha = _split_fields(repo, dataset)
    elif dataset == "spider":
        sample_size = 300
        eval_steps = 176
        split_file, split_sha = _split_fields(repo, dataset)
    else:
        sample_size = 100
        eval_steps = 400
        split_path = repo / "synthesis" / "evaluate" / "benchmarks" / "smiles" / "data" / f"{smiles_class}.txt"
        split_file, split_sha = str(split_path), sha256_file(split_path) if split_path.is_file() else ""
    output = repo / "outputs" / "controlled_comparison" / "paper_missing_results_20260828" / f"{cell_id}.json"
    return {
        "cell_id": cell_id,
        "dataset": dataset,
        "strategy": strategy,
        "smiles_class": smiles_class,
        "eval_model": model,
        "split_file": split_file,
        "split_sha256": split_sha,
        "heldout_split_name": "test",
        "sample_size": sample_size,
        "eval_max_steps": eval_steps,
        "eval_max_seconds": 600,
        "gpu_mem_util": cold_queue.EXPECTED_RUNTIME_BY_MODEL[model]["gpu_mem_util"],
        "worker_memory_mib": BASELINE_GPU_MIB[model],
        "worker_count": 1,
        "task": TASKS[dataset],
        "output_json": str(output),
        "source_model": model,
        "source_strategy": strategy,
        "source_sample_size": sample_size,
        "source_output_json": str(output),
        "output_sha256": "",
        "metadecode_json": "",
        "metadecode_sha256": "",
        "gpu_scope": [0, 1, 2, 3],
        "git_commit": _git_commit(repo),
        "source_hashes": _source_hashes(repo),
        "crane_commit": _crane_commit(repo),
    }


def build_scope(repo: Path) -> list[dict[str, Any]]:
    """Return exactly the 6 GSM, 2 Spider, and 30 SMILES requested rows."""
    rows: list[dict[str, Any]] = []
    for model_slug, model in GSM_MODELS:
        for strategy in ("unconstrained", "gcd", "itergen"):
            rows.append(_row(repo, dataset="gsm_symbolic", model_slug=model_slug, model=model, strategy=strategy))
    for model_slug, model in GSM_MODELS:
        rows.append(_row(repo, dataset="spider", model_slug=model_slug, model=model, strategy="cars"))
    for model_slug, model in MODELS[:2]:
        for smiles_class in SMILES_CLASSES:
            for strategy in STRATEGIES:
                rows.append(_row(repo, dataset="smiles", model_slug=model_slug, model=model, strategy=strategy, smiles_class=smiles_class))
    for model_slug, model in MODELS[2:]:
        for smiles_class in SMILES_CLASSES:
            rows.append(_row(repo, dataset="smiles", model_slug=model_slug, model=model, strategy="crane", smiles_class=smiles_class))
    return rows


def _require_scope(rows: list[dict[str, Any]], repo: Path) -> None:
    expected_rows = {row["cell_id"]: row for row in build_scope(repo)}
    expected = set(expected_rows)
    actual = {str(row.get("cell_id")) for row in rows}
    if actual != expected or len(rows) != len(expected):
        raise ValueError(f"manifest must contain exactly {len(expected)} requested cells")
    for row in rows:
        expected_row = expected_rows[row["cell_id"]]
        for key in ("dataset", "strategy", "smiles_class", "eval_model", "split_file", "split_sha256", "heldout_split_name", "sample_size", "eval_max_steps", "eval_max_seconds", "gpu_mem_util", "worker_memory_mib", "worker_count", "task", "output_json", "source_model", "source_strategy", "source_sample_size", "source_output_json", "gpu_scope", "git_commit", "source_hashes", "crane_commit"):
            if row.get(key) != expected_row.get(key):
                raise ValueError(f"{row['cell_id']}: immutable field {key} does not match scope")
        if row.get("heldout_split_name") != "test":
            raise ValueError(f"{row['cell_id']}: heldout split must be test")
        if row.get("dataset") in SPLITS:
            path = Path(str(row.get("split_file", "")))
            if path.resolve() != (repo / SPLITS[row["dataset"]]).resolve():
                raise ValueError(f"{row['cell_id']}: wrong canonical source split")
            if not path.is_file() or sha256_file(path) != row.get("split_sha256"):
                raise ValueError(f"{row['cell_id']}: source split hash mismatch")
        elif row.get("dataset") == "smiles":
            path = Path(str(row.get("split_file", "")))
            if not path.is_file() or sha256_file(path) != row.get("split_sha256"):
                raise ValueError(f"{row['cell_id']}: SMILES source hash mismatch")
        for key in ("eval_model", "strategy", "sample_size", "output_json", "split_sha256"):
            if key not in row:
                raise ValueError(f"{row['cell_id']}: missing {key}")


def bind_metadecode_artifacts(rows: list[dict[str, Any]], binding_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(binding_path.read_text(encoding="utf-8"))
    bindings = payload.get("bindings", payload) if isinstance(payload, dict) else None
    if not isinstance(bindings, dict) or set(bindings) != {row["cell_id"] for row in rows}:
        raise ValueError("metaDecode binding file must contain exactly all 38 cell ids")
    bound: list[dict[str, Any]] = []
    for row in rows:
        entry = bindings[row["cell_id"]]
        if not isinstance(entry, dict):
            raise ValueError(f"{row['cell_id']}: invalid frozen metaDecode binding")
        path = Path(str(entry.get("path", ""))).expanduser()
        expected = str(entry.get("sha256", ""))
        if not path.is_file() or not re.fullmatch(r"[0-9a-f]{64}", expected) or sha256_file(path) != expected:
            raise ValueError(f"{row['cell_id']}: frozen metaDecode path/hash mismatch")
        item = dict(row)
        item["metadecode_json"] = str(path.resolve())
        item["metadecode_sha256"] = expected
        bound.append(item)
    return bound


def write_manifest(repo: Path, path: Path, binding_path: Path | None = None) -> None:
    rows = build_scope(repo)
    if binding_path is None:
        raise ValueError("manifest construction requires --metadecode-bindings")
    rows = bind_metadecode_artifacts(rows, binding_path)
    for row in rows:
        # Do the same-row provenance check at creation time, so a manifest
        # cannot be written with a valid hash pointing at the wrong campaign.
        validate_frozen_metadecode(row)
    commit = _git_commit(repo)
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("manifest requires a cleanly identified git commit")
    if _dirty_source_paths(repo):
        raise ValueError("manifest cannot be built from dirty fixed-queue source")
    if any(set(row["source_hashes"]) != set(SOURCE_PATHS) for row in rows):
        raise ValueError("manifest requires hashes for every direct runtime dependency")
    if any(row["crane_commit"] != CRANE_COMMIT for row in rows):
        raise ValueError(f"manifest requires CRANE commit {CRANE_COMMIT}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": 1, "created_at": utc_now(), "repo": str(repo), "git_commit": commit, "source_hashes": rows[0]["source_hashes"], "crane_commit": CRANE_COMMIT, "jobs": rows}
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_manifest(path: Path, repo: Path) -> tuple[str, list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("jobs") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise ValueError("manifest must contain a jobs list")
    commit = str(payload.get("git_commit", ""))
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("manifest must bind a full git commit")
    live_commit = _git_commit(repo)
    if live_commit and live_commit != commit:
        raise ValueError(f"manifest commit {commit} does not match live commit {live_commit}")
    expected_hashes = payload.get("source_hashes")
    if not isinstance(expected_hashes, dict) or set(expected_hashes) != set(SOURCE_PATHS):
        raise ValueError("manifest must bind every direct runtime dependency")
    for relative, expected in expected_hashes.items():
        source = repo / str(relative)
        if not source.is_file() or sha256_file(source) != expected:
            raise ValueError(f"manifest source hash mismatch for {relative}")
    if _dirty_source_paths(repo):
        raise ValueError("manifest startup rejected dirty fixed-queue source")
    if payload.get("crane_commit") != CRANE_COMMIT or any(row.get("crane_commit") != CRANE_COMMIT for row in rows):
        raise ValueError(f"manifest requires CRANE commit {CRANE_COMMIT}")
    _require_scope(rows, repo)
    return sha256_file(path), rows


def fixed_baseline_command(row: dict[str, Any], python: Path) -> list[str]:
    command = [
        str(python), "-m", "synthesis.evaluate.run_legacy_fixed_strategy",
        "--strategy", str(row["strategy"]), "--dataset", str(row["dataset"]),
        "--eval-model", str(row["eval_model"]), "--eval-sample-size", str(row["sample_size"]),
        "--eval-backend", "vllm", "--device", "auto", "--eval-max-steps", str(row["eval_max_steps"]),
        "--eval-step-token-budget", "1",
        "--vllm-gpu-memory-utilization", str(row["gpu_mem_util"]),
        "--output-json", str(row["output_json"]),
    ]
    if row["dataset"] == "gsm_symbolic":
        command += ["--gsm-split-file", str(row["split_file"]), "--gsm-split-name", "test"]
    elif row["dataset"] == "spider":
        command += ["--spider-split-file", str(row["split_file"]), "--spider-split-name", "test"]
    else:
        command += ["--smiles-classes", str(row["smiles_class"]), "--smiles-samples-per-class", str(row["sample_size"])]
    return command


def rerun_thresholds(metrics: dict[str, float], total: int) -> dict[str, Any]:
    accuracy_count = min(total, max(0, int(round(metrics["accuracy"] * total))))
    if accuracy_count == total:
        minimum_accuracy = 0.95
        policy = "perfect_baseline_95_percent_exception"
    else:
        minimum_accuracy = (accuracy_count + 1) / total
        policy = "strict_plus_one"
    syntax_count = min(total, max(0, int(round(metrics["syntax_rate"] * total))))
    return {
        "min_accuracy": minimum_accuracy,
        # The cold queue's approved campaign contract keeps the syntax bar at
        # 0.90; syntax is a required gate, not a row-specific relaxation.
        "min_syntax_rate": 0.90,
        "threshold_policy": policy,
        "baseline_accuracy_count": accuracy_count,
        "baseline_syntax_count": syntax_count,
    }


def rerun_command(row: dict[str, Any], python: Path, thresholds: dict[str, Any] | None = None) -> list[str]:
    """Build the one allowed cold metaDecode rerun command, with no warm input."""
    output_name = rerun_output_name(row)
    limits = thresholds or {"min_accuracy": 0.95, "min_syntax_rate": 0.90}
    command = [
        str(python), "-m", "synthesis.run_synthesis", "--dataset", str(row["dataset"]),
        "--task", str(row["task"]), "--eval-model", str(row["eval_model"]),
        "--max-iterations", "40", "--min-accuracy", str(limits["min_accuracy"]), "--min-syntax-rate", str(limits["min_syntax_rate"]),
        "--eval-sample-size", str(row["sample_size"]), "--eval-max-steps", str(row["eval_max_steps"]),
        "--eval-max-seconds-per-example", str(row["eval_max_seconds"]), "--eval-step-token-budget", "1",
        "--generation-backend", "claude", "--generation-model", "claude-opus-5",
    ]
    if row["dataset"] == "smiles":
        command += ["--smiles-classes", str(row["smiles_class"]), "--smiles-samples-per-class", str(row["sample_size"])]
    return command


def rerun_output_name(row: dict[str, Any]) -> str:
    return f"paper_rerun_{affected_row_id(row)}"


def rerun_job(row: dict[str, Any], repo: Path) -> dict[str, Any]:
    output = repo / "outputs" / "reeval" / "paper_baseline_reruns" / f"{affected_row_id(row)}.json"
    return {
        "cell_id": affected_row_id(row), "dataset": row["dataset"], "eval_model": row["eval_model"],
        "task": row["task"], "min_accuracy": float((row.get("thresholds") or {}).get("min_accuracy", 0.95)),
        "min_syntax_rate": 0.90, "max_iterations": 40, "interrupted_author_calls": 0,
        "git_commit": row["git_commit"], "launch_commit": row["git_commit"],
        "eval_max_steps": row["eval_max_steps"], "eval_max_seconds": row["eval_max_seconds"],
        "gpu_mem_util": row["gpu_mem_util"], "heldout_sample_size": row["sample_size"],
        "heldout_split_file": row["split_file"], "heldout_split_name": "test",
        "heldout_output_json": str(output), "smiles_class": row.get("smiles_class"),
        "output_name": rerun_output_name(row), "eval_sample_size": row["sample_size"],
        "memory_reservation_mib": row["worker_memory_mib"],
    }


def _metric(payload: dict[str, Any], name: str) -> float | None:
    value = payload.get(name)
    if isinstance(value, (int, float)):
        return float(value)
    metrics = payload.get("metrics")
    if isinstance(metrics, dict) and isinstance(metrics.get(name), (int, float)):
        return float(metrics[name])
    return None


def _smiles_uv(payload: dict[str, Any], row: dict[str, Any]) -> float | None:
    for key in ("unique_valid_rate", "unique_valid", "unique_valid_accuracy"):
        value = _metric(payload, key)
        if value is not None:
            return value
    answers = payload.get("answers")
    if not isinstance(answers, list):
        return None
    try:
        from synthesis.evaluate.benchmarks.smiles.dataset import load_smiles
        from synthesis.evaluate.benchmarks.smiles.metrics import evaluate_smiles_output, smiles_trial_metrics
        example = load_smiles(classes=[row["smiles_class"]], samples_per_class=1)[0]
        samples = [{"smiles_eval": evaluate_smiles_output(row["smiles_class"], str(answer.get("generated_answer", "")), example.get("grammar_text", ""), example.get("prompt_exemplars", []), require_rdkit=True)} for answer in answers]
        trial = smiles_trial_metrics(samples)
        return float(trial["unique_valid_count"]) / max(1, len(samples))
    except Exception as exc:
        LOG.warning("[paperq] SMILES metric recomputation failed cell=%s error=%s", row["cell_id"], type(exc).__name__)
        return None


def validate_terminal_artifact(row: dict[str, Any], path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    metrics = payload.get("metrics") if isinstance(payload, dict) else None
    answers = payload.get("answers") if isinstance(payload, dict) else None
    try:
        metric_count = int(metrics.get("num_examples", -1)) if isinstance(metrics, dict) else -1
    except (TypeError, ValueError):
        return None
    if metric_count != int(row["sample_size"]):
        return None
    if not isinstance(answers, list) or len(answers) != int(row["sample_size"]):
        return None
    provenance = payload.get("paper_baseline_provenance")
    expected = {key: row[key] for key in ("cell_id", "dataset", "strategy", "eval_model", "split_file", "split_sha256", "sample_size", "source_model", "source_strategy", "source_sample_size", "source_output_json")}
    if provenance != expected:
        return None
    accuracy = _metric(payload, "accuracy")
    syntax = _metric(payload, "syntax_rate")
    if accuracy is None or syntax is None:
        return None
    result = {"accuracy": accuracy, "syntax_rate": syntax, "sha256": sha256_file(path)}
    if row.get("output_sha256") and row["output_sha256"] != result["sha256"]:
        return None
    if row["dataset"] == "smiles":
        uv = _smiles_uv(payload, row)
        if uv is None:
            return None
        result["unique_valid_rate"] = uv
    return result


def annotate_artifact(row: dict[str, Any], path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["paper_baseline_provenance"] = {key: row[key] for key in ("cell_id", "dataset", "strategy", "eval_model", "split_file", "split_sha256", "sample_size", "source_model", "source_strategy", "source_sample_size", "source_output_json")}
    if row["dataset"] == "smiles":
        uv = _smiles_uv(payload, row)
        if uv is None:
            raise ValueError(f"{row['cell_id']}: unable to compute unique-valid rate")
        payload["unique_valid_rate"] = uv
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _read_metrics(path: Path, row: dict[str, Any]) -> dict[str, float] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    result: dict[str, float] = {}
    for name in ("accuracy", "syntax_rate"):
        value = _metric(payload, name)
        if value is None:
            return None
        result[name] = value
    if row["dataset"] == "smiles":
        value = _smiles_uv(payload, row)
        if value is None:
            return None
        result["unique_valid_rate"] = value
    return result


def validate_frozen_metadecode(row: dict[str, Any]) -> Path:
    path_text = str(row.get("metadecode_json", "")).strip()
    expected_sha = str(row.get("metadecode_sha256", "")).strip()
    if not path_text or len(expected_sha) != 64:
        raise ValueError(f"{row['cell_id']}: frozen metaDecode path and SHA-256 are required")
    path = Path(path_text)
    if not path.is_file() or sha256_file(path) != expected_sha:
        raise ValueError(f"{row['cell_id']}: frozen metaDecode artifact hash mismatch")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{row['cell_id']}: frozen metaDecode JSON is invalid") from exc
    provenance = payload.get("reevaluation_provenance")
    if not isinstance(provenance, dict) or provenance.get("cell_id") != affected_row_id(row) or any(
        provenance.get(key) != row.get(expected)
        for key, expected in (("dataset", "dataset"), ("eval_model", "eval_model"), ("sample_size", "sample_size"), ("smiles_class", "smiles_class"))
    ) or provenance.get("max_steps") != row.get("eval_max_steps") or provenance.get("step_token_budget") != 1:
        raise ValueError(f"{row['cell_id']}: frozen metaDecode provenance does not match row")
    split = payload.get("eval_split") or {}
    if row["dataset"] == "gsm_symbolic":
        if split.get("gsm_split_name") != "test" or Path(str(split.get("gsm_split_file", ""))).name != Path(row["split_file"]).name:
            raise ValueError(f"{row['cell_id']}: frozen metaDecode split does not match row")
    elif row["dataset"] == "spider":
        if split.get("spider_split_name") != "test" or Path(str(split.get("spider_split_file", ""))).name != Path(row["split_file"]).name:
            raise ValueError(f"{row['cell_id']}: frozen metaDecode split does not match row")
    metrics = payload.get("metrics")
    answers = payload.get("answers")
    if not isinstance(metrics, dict) or int(metrics.get("num_examples", -1)) != int(row["sample_size"]):
        raise ValueError(f"{row['cell_id']}: frozen metaDecode metrics count does not match row")
    if not isinstance(answers, list) or len(answers) != int(row["sample_size"]):
        raise ValueError(f"{row['cell_id']}: frozen metaDecode answer count does not match row")
    if _read_metrics(path, row) is None:
        raise ValueError(f"{row['cell_id']}: frozen metaDecode metrics are incomplete")
    return path


def baseline_beats_metadecode(dataset: str, baseline: dict[str, float], metadecode: dict[str, float]) -> bool:
    if dataset == "smiles":
        return baseline["unique_valid_rate"] > metadecode["unique_valid_rate"]
    return baseline["accuracy"] > metadecode["accuracy"] or baseline["syntax_rate"] > metadecode["syntax_rate"]


def _post14b_claim_helper():
    path = Path(__file__).resolve().parents[2] / ".context" / "run_post14b_rebar_queue.py"
    spec = importlib.util.spec_from_file_location("paper_post14b_claims", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load claim helper {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def claim_rerun(claims_dir: Path, cell_id: str, manifest_sha256: str, rerun_spec: dict[str, Any] | None = None) -> bool:
    """Reuse post-14B's atomic mkdir claim; never remove a failed claim."""
    helper = _post14b_claim_helper()
    claimed = bool(helper.claim_cell(claims_dir, cell_id, manifest_sha256))
    if claimed and rerun_spec is not None:
        claim_dir = helper.claim_directory(claims_dir, cell_id)
        spec_path = claim_dir / "rerun.json"
        temporary = spec_path.with_suffix(".tmp")
        spec = dict(rerun_spec)
        spec.setdefault("status", "pending")
        temporary.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(spec_path)
    return claimed


def affected_row_id(row: dict[str, Any]) -> str:
    """Name the metaDecode row shared by all baseline strategies."""
    # Frozen campaign files use the model name, not its registry namespace
    # (for example ``qwen35-2b`` rather than ``qwen-qwen35-2b``).
    model = MODEL_SLUG_BY_NAME.get(str(row["eval_model"])) or _slug(str(row["eval_model"]).rsplit("/", 1)[-1])
    if row["dataset"] == "smiles":
        return f"{row['dataset']}-{row['smiles_class']}-{model}"
    dataset_id = "gsm" if row["dataset"] == "gsm_symbolic" else row["dataset"]
    return f"{dataset_id}-{model}"


def worker_demand_mib(row: dict[str, Any], total_mib: int) -> int:
    configured = int(row["worker_memory_mib"])
    fraction = math.ceil(float(row["gpu_mem_util"]) * int(total_mib))
    return max(configured, fraction)


def choose_gpu(row: dict[str, Any], snapshots: dict[int, dict[str, int]], reservations: dict[int, int], baseline: dict[int, dict[str, int]], allowed: tuple[int, ...]) -> int | None:
    candidates: list[tuple[int, int]] = []
    scope = tuple(int(gpu) for gpu in row.get("gpu_scope", []))
    for gpu in (candidate for candidate in allowed if candidate in scope):
        snapshot = snapshots.get(gpu)
        if snapshot is None:
            continue
        total = int(snapshot["total_mib"])
        demand = worker_demand_mib(row, total) * int(row.get("worker_count", 1))
        used = int(snapshot["used_mib"])
        free = int(snapshot.get("free_mib", total - used))
        projected_used = max(used, int(baseline.get(gpu, snapshot)["used_mib"]) + int(reservations.get(gpu, 0)))
        if free < demand + GPU_SAFETY_MIB:
            continue
        if total - projected_used < demand + GPU_SAFETY_MIB:
            continue
        candidates.append((projected_used, gpu))
    return min(candidates)[1] if candidates else None


def _state_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)


def _process_start_identity(pid: int) -> str | None:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").rsplit(")", 1)[1].split()
        return fields[19]
    except (OSError, IndexError):
        return None


def _child_matches(payload: dict[str, Any]) -> bool:
    try:
        pid = int(payload["pid"])
    except (KeyError, TypeError, ValueError):
        return False
    expected = str(payload.get("pid_start", ""))
    actual = _process_start_identity(pid)
    if actual is None or (expected and actual != expected):
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _read_state(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _surviving_child_guard(state_path: Path | None, output: Path) -> dict[str, Any] | None:
    if state_path is None:
        return None
    prior = _read_state(state_path)
    if not prior or prior.get("status") != "running" or "pid" not in prior:
        return None
    if _child_matches(prior):
        LOG.info("[paperq] waiting for surviving child cell=%s pid=%s", prior.get("cell_id"), prior.get("pid"))
        while _child_matches(prior):
            time.sleep(1.0)
    if output.is_file():
        return None
    return {"status": "failed", "cell_id": prior.get("cell_id"), "reason": "surviving_child_finished_without_terminal_output"}


def _default_run(command: list[str], *, repo: Path, env: dict[str, str], state_path: Path | None, state_payload: dict[str, Any]) -> Any:
    process = subprocess.Popen(command, cwd=repo, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if state_path is not None:
        state_payload = dict(state_payload)
        state_payload.update({"pid": process.pid, "pid_start": _process_start_identity(process.pid)})
        _state_write(state_path, state_payload)
    process.communicate()
    return type("ProcessResult", (), {"returncode": process.returncode})()


def _write_claim_spec(claims_dir: Path, identity: str, update: dict[str, Any]) -> None:
    helper = _post14b_claim_helper()
    path = helper.claim_directory(claims_dir, identity) / "rerun.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.update(update)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _select_rerun_csd(row: dict[str, Any], repo: Path, exit_code: int) -> Path | None:
    """Reuse cold-queue success and exhausted-best-attempt selection."""
    job = rerun_job(row, repo)
    if exit_code == 0:
        return cold_queue.compiled_csd(
            repo, job["output_name"], min_accuracy=float(job["min_accuracy"]),
            min_syntax_rate=float(job["min_syntax_rate"]), job=job,
        )
    if exit_code == 1 and cold_queue.synthesis_was_exhausted(repo, job["output_name"], job):
        return cold_queue.compiled_csd(
            repo, job["output_name"], min_accuracy=float(job["min_accuracy"]),
            min_syntax_rate=float(job["min_syntax_rate"]), job=job,
        )
    return None


def _run_heldout_rerun(row: dict[str, Any], *, repo: Path, python: Path, env: dict[str, str], runner: Callable[..., Any], csd: Path, state_path: Path | None = None, state_payload: dict[str, Any] | None = None) -> int:
    holdout = rerun_job(row, repo)
    holdout_path = Path(holdout["heldout_output_json"])
    holdout_path.parent.mkdir(parents=True, exist_ok=True)
    command = cold_queue.heldout_command(holdout, python, csd)
    payload = dict(state_payload or {})
    payload.update({"phase": "heldout", "csd_path": str(csd), "csd_sha256": sha256_file(csd)})
    if state_path is not None:
        _state_write(state_path, payload)
    result = _default_run(command, repo=repo, env=env, state_path=state_path, state_payload=payload) if runner is subprocess.run else runner(command, cwd=repo, env=env, check=False)
    code = int(getattr(result, "returncode", result if isinstance(result, int) else 1))
    return code if code != 0 or not cold_queue.heldout_is_complete(holdout_path, holdout) else 0


def _run_rerun_row(row: dict[str, Any], *, repo: Path, python: Path, claims_dir: Path, manifest_sha256: str, state_dir: Path | None, runner: Callable[..., Any]) -> dict[str, Any]:
    identity = str(row["rerun_identity"])
    holdout = rerun_job(row, repo)
    holdout_path = Path(holdout["heldout_output_json"])
    prior_state = _read_state(state_dir / f"rerun-{identity}.json") if state_dir is not None else None
    state_path = state_dir / f"rerun-{identity}.json" if state_dir is not None else None
    if prior_state and prior_state.get("status") == "running":
        if _child_matches(prior_state):
            LOG.info("[paperq] waiting for surviving rerun child identity=%s pid=%s", identity, prior_state.get("pid"))
            while _child_matches(prior_state):
                time.sleep(1.0)
        if prior_state.get("phase") == "heldout" and cold_queue.heldout_is_complete(holdout_path, holdout):
            _write_claim_spec(claims_dir, identity, {"status": "finished", "finished_at": utc_now(), "exit_code": 0})
            if state_dir is not None:
                _state_write(state_dir / f"rerun-{identity}.json", {"cell_id": identity, "status": "finished", "exit_code": 0, "manifest_sha256": manifest_sha256})
            return {"status": "finished", "cell_id": identity, "exit_code": 0, "reattached": True}
        if prior_state.get("phase") == "heldout":
            csd_path = Path(str(prior_state.get("csd_path", "")))
            if csd_path.is_file() and str(prior_state.get("csd_sha256", "")) == sha256_file(csd_path):
                env = dict(os.environ)
                env.update({"CUDA_VISIBLE_DEVICES": str(row["assigned_gpu"]), "CSD_VLLM_GPU_MEMORY_UTILIZATION": str(row["gpu_mem_util"]), "CSD_VLLM_GPU_MEMORY_UTILIZATION_MAX": str(row["gpu_mem_util"]), "CSD_CLAUDE_CONFIG_DIR": "/home/aadivyar/.claude-csd-synthesis", "CSD_CLAUDE_EXPECTED_ACCOUNT": "ssdear@gmail.com", "CSD_OUTPUT_NAME": rerun_output_name(row)})
                code = _run_heldout_rerun(row, repo=repo, python=python, env=env, runner=runner, csd=csd_path, state_path=state_path, state_payload={"cell_id": identity, "status": "running", "manifest_sha256": manifest_sha256})
                status = "finished" if code == 0 else "failed"
                _write_claim_spec(claims_dir, identity, {"status": status, "finished_at": utc_now(), "exit_code": code})
                return {"status": status, "cell_id": identity, "exit_code": code, "reattached": True}
        if prior_state.get("phase") == "synthesis":
            for completed_code in (0, 1):
                csd = _select_rerun_csd(row, repo, completed_code)
                if csd is not None:
                    env = dict(os.environ)
                    env.update({"CUDA_VISIBLE_DEVICES": str(row["assigned_gpu"]), "CSD_VLLM_GPU_MEMORY_UTILIZATION": str(row["gpu_mem_util"]), "CSD_VLLM_GPU_MEMORY_UTILIZATION_MAX": str(row["gpu_mem_util"]), "CSD_CLAUDE_CONFIG_DIR": "/home/aadivyar/.claude-csd-synthesis", "CSD_CLAUDE_EXPECTED_ACCOUNT": "ssdear@gmail.com", "CSD_OUTPUT_NAME": rerun_output_name(row)})
                    code = _run_heldout_rerun(row, repo=repo, python=python, env=env, runner=runner, csd=csd, state_path=state_path, state_payload={"cell_id": identity, "status": "running", "manifest_sha256": manifest_sha256})
                    status = "finished" if code == 0 else "failed"
                    _write_claim_spec(claims_dir, identity, {"status": status, "finished_at": utc_now(), "exit_code": code})
                    return {"status": status, "cell_id": identity, "exit_code": code, "reattached": True}
        reason = "surviving_rerun_child_finished_without_recoverable_report"
        _write_claim_spec(claims_dir, identity, {"status": "failed", "reason": reason, "finished_at": utc_now()})
        if state_path is not None:
            _state_write(state_path, {"cell_id": identity, "status": "failed", "phase": "synthesis", "reason": reason, "manifest_sha256": manifest_sha256})
        return {"status": "failed", "cell_id": identity, "reason": reason}
    if row.get("claim_status") in {"started", "running"}:
        # A restarted controller must consume a completed report/held-out
        # artifact, or fail closed. It must not spend a second author attempt
        # when the original process left no reattachable ledger.
        if cold_queue.heldout_is_complete(holdout_path, holdout):
            _write_claim_spec(claims_dir, identity, {"status": "finished", "finished_at": utc_now(), "exit_code": 0})
            return {"status": "finished", "cell_id": identity, "exit_code": 0, "reattached": True}
        for completed_code in (0, 1):
            csd = _select_rerun_csd(row, repo, completed_code)
            if csd is not None:
                env = dict(os.environ)
                env.update({"CUDA_VISIBLE_DEVICES": str(row["assigned_gpu"]), "CSD_VLLM_GPU_MEMORY_UTILIZATION": str(row["gpu_mem_util"]), "CSD_VLLM_GPU_MEMORY_UTILIZATION_MAX": str(row["gpu_mem_util"]), "CSD_CLAUDE_CONFIG_DIR": "/home/aadivyar/.claude-csd-synthesis", "CSD_CLAUDE_EXPECTED_ACCOUNT": "ssdear@gmail.com", "CSD_OUTPUT_NAME": rerun_output_name(row)})
                code = _run_heldout_rerun(row, repo=repo, python=python, env=env, runner=runner, csd=csd, state_path=state_path, state_payload={"cell_id": identity, "status": "running", "manifest_sha256": manifest_sha256})
                status = "finished" if code == 0 else "failed"
                _write_claim_spec(claims_dir, identity, {"status": status, "finished_at": utc_now(), "exit_code": code})
                return {"status": status, "cell_id": identity, "exit_code": code, "reattached": True}
        _write_claim_spec(claims_dir, identity, {"status": "failed", "reason": "dead_claim_without_recoverable_attempt_ledger"})
        return {"status": "failed", "cell_id": identity, "reason": "dead_claim_without_recoverable_attempt_ledger"}
    _write_claim_spec(claims_dir, identity, {"status": "started", "started_at": utc_now()})
    command = list(row["rerun_command"])
    if not command:
        _write_claim_spec(claims_dir, identity, {"status": "failed", "reason": "missing_rerun_command"})
        return {"status": "failed", "cell_id": identity, "reason": "missing_rerun_command"}
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(row["assigned_gpu"])
    env["CSD_VLLM_GPU_MEMORY_UTILIZATION"] = str(row["gpu_mem_util"])
    env["CSD_VLLM_GPU_MEMORY_UTILIZATION_MAX"] = str(row["gpu_mem_util"])
    env["CSD_CLAUDE_CONFIG_DIR"] = "/home/aadivyar/.claude-csd-synthesis"
    env["CSD_CLAUDE_EXPECTED_ACCOUNT"] = "ssdear@gmail.com"
    # run_synthesis takes its isolated campaign name from this environment
    # contract; it has no --output-name command-line option.
    env["CSD_OUTPUT_NAME"] = rerun_output_name(row)
    if row["dataset"] == "smiles":
        env["CSD_CONSTRAINED_TEMPERATURE"] = "0.7"
    if state_dir is not None:
        _state_write(state_path, {"cell_id": identity, "status": "running", "phase": "synthesis", "manifest_sha256": manifest_sha256, "started_at": utc_now()})
    if runner is subprocess.run:
        result = _default_run(command, repo=repo, env=env, state_path=state_path, state_payload={"cell_id": identity, "status": "running", "phase": "synthesis", "manifest_sha256": manifest_sha256, "started_at": utc_now()})
    else:
        result = runner(command, cwd=repo, env=env, check=False)
    code = int(getattr(result, "returncode", result if isinstance(result, int) else 1))
    status = "finished" if code in (0, 1) else "failed"
    if status == "finished":
        csd = _select_rerun_csd(row, repo, code)
        if csd is None:
            status = "failed"
            code = 4
        else:
            heldout_code = _run_heldout_rerun(row, repo=repo, python=python, env=env, runner=runner, csd=csd, state_path=state_path, state_payload={"cell_id": identity, "status": "running", "manifest_sha256": manifest_sha256})
            if heldout_code != 0:
                status = "failed"
                code = heldout_code or 5
    _write_claim_spec(claims_dir, identity, {"status": status, "finished_at": utc_now(), "exit_code": code})
    if state_dir is not None:
        _state_write(state_dir / f"rerun-{identity}.json", {"cell_id": identity, "status": status, "exit_code": code, "manifest_sha256": manifest_sha256})
    return {"status": status, "cell_id": identity, "exit_code": code}


def run_row(row: dict[str, Any], *, repo: Path, python: Path, claims_dir: Path, manifest_sha256: str, state_dir: Path | None = None, dry_run: bool = False, runner: Callable[..., Any] | None = None) -> dict[str, Any]:
    output = Path(str(row["output_json"]))
    if dry_run:
        print(f"[paperq] DRY_RUN cell={row['cell_id']} command={' '.join(fixed_baseline_command(row, python))}", flush=True)
        print(f"[paperq] DRY_RUN cell={row['cell_id']} rerun={' '.join(rerun_command(row, python))}", flush=True)
        return {"status": "dry_run", "cell_id": row["cell_id"]}
    if row.get("job_kind") == "rerun":
        return _run_rerun_row(row, repo=repo, python=python, claims_dir=claims_dir, manifest_sha256=manifest_sha256, state_dir=state_dir, runner=runner or subprocess.run)
    output.parent.mkdir(parents=True, exist_ok=True)
    validated = validate_terminal_artifact(row, output) if output.is_file() else None
    if validated is not None:
        LOG.info("[paperq] terminal artifact accepted cell=%s sha=%s", row["cell_id"], validated["sha256"][:12])
        base = _read_metrics(output, row)
        comparison = "not_checked"
        try:
            meta_path = validate_frozen_metadecode(row)
            meta = _read_metrics(meta_path, row)
            if meta is not None and base is not None and baseline_beats_metadecode(row["dataset"], base, meta):
                comparison = "baseline_win"
                thresholds = rerun_thresholds(base, int(row["sample_size"]))
                rerun_id = affected_row_id(row)
                claimed = claim_rerun(claims_dir, rerun_id, manifest_sha256, {"affected_cell_id": rerun_id, "baseline_cell_id": row["cell_id"], "cold": True, "max_iterations": 40, "warm_start": False, "thresholds": thresholds, "command": rerun_command(row, python, thresholds), "heldout_output_json": str(rerun_job({**row, "thresholds": thresholds}, repo)["heldout_output_json"]), "output_name": rerun_output_name(row)})
            else:
                comparison = "no_baseline_win"
        except ValueError:
            comparison = "comparison_blocked"
        return {"status": "terminal", "comparison": comparison, "rerun_claimed": bool(locals().get("claimed", False)), **validated}
    guarded = _surviving_child_guard(state_dir / f"{row['cell_id']}.json" if state_dir is not None else None, output)
    if guarded is not None:
        return guarded
    temporary_output = output.with_name(f".{output.name}.{os.getpid()}.{time.time_ns()}.tmp")
    execution_row = dict(row)
    execution_row["output_json"] = str(temporary_output)
    command = fixed_baseline_command(execution_row, python)
    if runner is None:
        runner = subprocess.run
    if state_dir is not None:
        _state_write(state_dir / f"{row['cell_id']}.json", {"cell_id": row["cell_id"], "status": "running", "manifest_sha256": manifest_sha256, "started_at": utc_now()})
    env = dict(os.environ)
    assigned_gpu = row.get("assigned_gpu")
    if assigned_gpu is not None:
        if int(assigned_gpu) not in {int(gpu) for gpu in row.get("gpu_scope", [])}:
            raise ValueError(f"{row['cell_id']}: assigned GPU is outside row scope")
        env["CUDA_VISIBLE_DEVICES"] = str(assigned_gpu)
    env["CSD_VLLM_GPU_MEMORY_UTILIZATION"] = str(row["gpu_mem_util"])
    env["CSD_VLLM_GPU_MEMORY_UTILIZATION_MAX"] = str(row["gpu_mem_util"])
    if row["dataset"] == "smiles":
        env["CSD_CONSTRAINED_TEMPERATURE"] = "0.7"
    state_payload = {"cell_id": row["cell_id"], "status": "running", "manifest_sha256": manifest_sha256, "started_at": utc_now()}
    if runner is subprocess.run:
        result = _default_run(command, repo=repo, env=env, state_path=state_dir / f"{row['cell_id']}.json" if state_dir is not None else None, state_payload=state_payload)
    else:
        result = runner(command, cwd=repo, env=env, check=False)
    returncode = int(getattr(result, "returncode", result if isinstance(result, int) else 1))
    if returncode != 0 or not temporary_output.is_file():
        if state_dir is not None:
            _state_write(state_dir / f"{row['cell_id']}.json", {"cell_id": row["cell_id"], "status": "failed", "exit_code": returncode, "manifest_sha256": manifest_sha256})
        return {"status": "failed", "cell_id": row["cell_id"], "exit_code": returncode}
    try:
        annotate_artifact(execution_row, temporary_output)
        temporary_validated = validate_terminal_artifact(execution_row, temporary_output)
        if temporary_validated is None:
            raise ValueError("output failed terminal validation")
        temporary_output.replace(output)
        validated = validate_terminal_artifact(row, output)
        if validated is None:
            raise ValueError("output failed terminal validation")
        comparison = "not_checked"
        try:
            meta_path = validate_frozen_metadecode(row)
            meta = _read_metrics(meta_path, row)
            base = _read_metrics(output, row)
            if meta is not None and base is not None and baseline_beats_metadecode(row["dataset"], base, meta):
                comparison = "baseline_win"
                thresholds = rerun_thresholds(base, int(row["sample_size"]))
                rerun_id = affected_row_id(row)
                claimed = claim_rerun(claims_dir, rerun_id, manifest_sha256, {"affected_cell_id": rerun_id, "baseline_cell_id": row["cell_id"], "cold": True, "max_iterations": 40, "warm_start": False, "thresholds": thresholds, "command": rerun_command(row, python, thresholds), "heldout_output_json": str(rerun_job({**row, "thresholds": thresholds}, repo)["heldout_output_json"]), "output_name": rerun_output_name(row)})
            else:
                comparison = "no_baseline_win"
        except ValueError:
            comparison = "comparison_blocked"
        if state_dir is not None:
            _state_write(state_dir / f"{row['cell_id']}.json", {"cell_id": row["cell_id"], "status": "terminal", "comparison": comparison, "sha256": validated["sha256"], "manifest_sha256": manifest_sha256})
        return {"status": "terminal", "comparison": comparison, "rerun_claimed": bool(locals().get("claimed", False)), **validated}
    except Exception as exc:
        LOG.warning("[paperq] validation failed cell=%s error=%s", row["cell_id"], type(exc).__name__)
        if state_dir is not None:
            _state_write(state_dir / f"{row['cell_id']}.json", {"cell_id": row["cell_id"], "status": "failed", "reason": type(exc).__name__, "manifest_sha256": manifest_sha256})
        return {"status": "failed", "cell_id": row["cell_id"], "reason": type(exc).__name__}


def dispatch(rows: list[dict[str, Any]], *, repo: Path, python: Path, claims_dir: Path, state_dir: Path, manifest_sha256: str, allowed: tuple[int, ...], snapshot: Callable[[], dict[int, dict[str, int]]], poll_seconds: float = 30.0, dry_run: bool = False, runner: Callable[..., Any] | None = None) -> list[dict[str, Any]]:
    if dry_run:
        return [run_row(row, repo=repo, python=python, claims_dir=claims_dir, manifest_sha256=manifest_sha256, dry_run=True) for row in rows]
    for row in rows:
        validate_frozen_metadecode(row)
        if not set(int(gpu) for gpu in allowed).intersection(int(gpu) for gpu in row.get("gpu_scope", [])):
            raise ValueError(f"{row['cell_id']}: CLI GPU scope does not intersect row GPU scope")
    lock_path = state_dir.parent / "paper_baseline_queue.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        reservations: dict[int, int] = {}
        baseline = snapshot()
        pending = list(rows)
        helper = _post14b_claim_helper()
        seen_reruns: set[str] = set()
        for source_row in rows:
            identity = affected_row_id(source_row)
            if identity in seen_reruns:
                continue
            seen_reruns.add(identity)
            spec_path = helper.claim_directory(claims_dir, identity) / "rerun.json"
            if not spec_path.is_file():
                continue
            try:
                spec = json.loads(spec_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if spec.get("status") not in {"pending", "started", "running"}:
                continue
            rerun = dict(source_row)
            rerun.update({"job_kind": "rerun", "cell_id": f"rerun-{identity}", "rerun_identity": identity, "rerun_command": spec.get("command", []), "thresholds": spec.get("thresholds", {}), "output_name": spec.get("output_name", rerun_output_name(source_row)), "claim_status": spec.get("status")})
            pending.append(rerun)
        running: dict[Future, tuple[int, dict[str, Any]]] = {}
        results: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=max(1, len(rows))) as executor:
            while pending or running:
                current = snapshot()
                launched = True
                while pending and launched:
                    launched = False
                    for index, row in enumerate(pending):
                        gpu = choose_gpu(row, current, reservations, baseline, allowed)
                        if gpu is None:
                            continue
                        demand = worker_demand_mib(row, int(current[gpu]["total_mib"])) * int(row.get("worker_count", 1))
                        reservations[gpu] = reservations.get(gpu, 0) + demand
                        run_row_data = dict(row)
                        run_row_data["assigned_gpu"] = gpu
                        future = executor.submit(run_row, run_row_data, repo=repo, python=python, claims_dir=claims_dir, state_dir=state_dir, manifest_sha256=manifest_sha256, runner=runner)
                        running[future] = (gpu, row)
                        pending.pop(index)
                        launched = True
                        LOG.info("[paperq] dispatch cell=%s gpu=%s demand_mib=%s", row["cell_id"], gpu, demand)
                        break
                for future in [f for f in running if f.done()]:
                    gpu, row = running.pop(future)
                    demand = worker_demand_mib(row, int(baseline[gpu]["total_mib"])) * int(row.get("worker_count", 1))
                    reservations[gpu] -= demand
                    try:
                        result = future.result()
                    except Exception as exc:
                        LOG.warning("[paperq] worker failed cell=%s error=%s", row["cell_id"], type(exc).__name__)
                        result = {"status": "failed", "cell_id": row["cell_id"], "reason": type(exc).__name__}
                    results.append(result)
                    if result.get("rerun_claimed"):
                        rerun = dict(row)
                        thresholds = rerun_thresholds({"accuracy": result["accuracy"], "syntax_rate": result["syntax_rate"]}, int(row["sample_size"]))
                        rerun.update({"job_kind": "rerun", "cell_id": f"rerun-{affected_row_id(row)}", "rerun_identity": affected_row_id(row), "thresholds": thresholds, "output_name": rerun_output_name(row), "rerun_command": rerun_command(row, python, thresholds)})
                        pending.append(rerun)
                if pending or running:
                    if not launched and not running:
                        LOG.info("[paperq] waiting for a GPU to fit pending work")
                    time.sleep(max(0.0, poll_seconds))
        return results


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--claims-dir", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--nvidia-smi", default="nvidia-smi")
    parser.add_argument("--write-manifest", action="store_true")
    parser.add_argument("--metadecode-bindings", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    repo = args.repo.resolve()
    if args.write_manifest:
        write_manifest(repo, args.manifest.resolve(), args.metadecode_bindings.resolve() if args.metadecode_bindings else None)
        return 0
    digest, rows = load_manifest(args.manifest.resolve(), repo)
    allowed = cold_queue.parse_gpu_list(args.gpus)
    snapshot = lambda: cold_queue.gpu_memory_snapshot(args.nvidia_smi)
    results = dispatch(rows, repo=repo, python=args.python.resolve(), claims_dir=args.claims_dir.resolve(), state_dir=args.state_dir.resolve(), manifest_sha256=digest, allowed=allowed, snapshot=snapshot, poll_seconds=args.poll_seconds, dry_run=args.dry_run)
    print(json.dumps({"manifest_sha256": digest, "results": results}, sort_keys=True), flush=True)
    return 1 if any(result.get("status") == "failed" for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
