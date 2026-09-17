#!/usr/bin/env python3
"""Build and launch the 2026-08-03 cold campaign after all baselines finish."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from scripts.runtime.win_bar import accuracy_bar

from synthesis.evaluate.benchmarks.gsm_symbolic.prompts import GSM_CRANE_COT_TASK


CAMPAIGN = "full-baseline-20260803"
STRATEGIES = ("unconstrained", "gcd", "crane", "itergen", "cars")
EVIDENCE_PATH = Path("saved-results/2026-08-03-full-baseline-campaign-evidence.json")
MANIFEST_PATH = Path("saved-results/2026-08-03-full-baseline-cold-manifest.json")
MODELS = (
    ("qwen25-1p5b", "Qwen/Qwen2.5-1.5B-Instruct", 16_000, 0.30),
    ("qwen25-7b", "Qwen/Qwen2.5-7B-Instruct", 22_000, 0.45),
    ("qwen35-2b", "Qwen/Qwen3.5-2B", 16_384, 0.35),
    ("qwen35-4b", "Qwen/Qwen3.5-4B", 19_000, 0.40),
)
SPIDER_TASK = (
    "Generate a single valid SQL query using only the provided schema context. "
    "Only output the SQL query."
)
SMILES_TASK = (
    "Generate valid SMILES strings that match the requested molecular class while "
    "maintaining parser-valid output."
)


@dataclass(frozen=True)
class Cohort:
    slug: str
    dataset: str
    sample_size: int
    heldout_sample_size: int
    max_steps: int
    task: str
    smiles_class: str | None = None


COHORTS = (
    Cohort("gsm", "gsm_symbolic", 49, 49, 900, GSM_CRANE_COT_TASK),
    Cohort("spider", "spider", 300, 300, 176, SPIDER_TASK),
    Cohort("smiles-acrylates", "smiles", 50, 100, 400, SMILES_TASK, "acrylates"),
    Cohort(
        "smiles-chain_extenders",
        "smiles",
        50,
        100,
        400,
        SMILES_TASK,
        "chain_extenders",
    ),
    Cohort(
        "smiles-isocyanates",
        "smiles",
        50,
        100,
        400,
        SMILES_TASK,
        "isocyanates",
    ),
)


class CampaignError(ValueError):
    pass


def require_synthesis_unblocked(repo: Path) -> None:
    from scripts.runtime import run_cold_synthesis_queue as queue

    queue.require_synthesis_unblocked(repo)


def baseline_artifact(
    repo: Path,
    cohort: Cohort,
    model: tuple[str, str, int, float],
    strategy: str,
    *,
    campaign_root: Path | None = None,
) -> Path:
    root = campaign_root or repo / "outputs/baselines/full_baseline_20260803"
    return root / cohort.slug / model[0] / f"{strategy}.json"


def _rate_count(value: Any, total: int, label: str, path: Path) -> int:
    if not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 1.0:
        raise CampaignError(f"{path}: invalid {label}")
    count = round(float(value) * total)
    if abs(float(value) - count / total) > 1e-9:
        raise CampaignError(f"{path}: {label} does not resolve to an exact count")
    return count


def _read_baseline_bytes(
    raw: bytes,
    path: Path,
    total: int,
    strategy: str,
    repo: Path,
    *,
    dataset: str | None = None,
    smiles_class: str | None = None,
) -> dict[str, Any]:
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CampaignError(f"invalid baseline artifact {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise CampaignError(f"{path}: baseline artifact must be an object")
    answers = payload.get("answers")
    if not isinstance(answers, list) or len(answers) != total:
        raise CampaignError(f"{path}: answer count must be {total}")
    if any(
        not isinstance(row, dict) or "generated_answer" not in row for row in answers
    ):
        raise CampaignError(f"{path}: every row must contain generated_answer")
    metrics = payload.get("metrics") or {}
    if int(metrics.get("num_examples") or -1) != total:
        raise CampaignError(f"{path}: metrics.num_examples must be {total}")
    generated_answers = [str(row["generated_answer"]).strip() for row in answers]
    metric_source = "artifact_summary"
    if dataset == "smiles":
        if not smiles_class:
            raise CampaignError(
                f"{path}: smiles_class is required for SMILES rescoring"
            )
        from synthesis.evaluate.benchmarks.smiles.dataset import get_smiles_task
        from synthesis.evaluate.benchmarks.smiles.metrics import (
            evaluate_smiles_output,
            smiles_trial_metrics,
        )

        task = get_smiles_task(smiles_class)
        scored_samples = [
            {
                "smiles_eval": evaluate_smiles_output(
                    class_name=smiles_class,
                    output=answer,
                    grammar_text=str(task["grammar_text"]),
                    prompt_exemplars=list(task.get("prompt_exemplars", [])),
                    require_rdkit=True,
                )
            }
            for answer in generated_answers
        ]
        trial = smiles_trial_metrics(
            scored_samples,
            target_unique_valid=total,
            sample_cap=total,
        )
        num_correct = int(trial["unique_valid_count"])
        syntax_count = sum(
            1
            for sample in scored_samples
            if bool(sample["smiles_eval"].get("syntax_valid"))
        )
        accuracy = num_correct / total
        syntax_rate = syntax_count / total
        metric_source = "recomputed_smiles_unique_valid"
    else:
        num_correct = _rate_count(payload.get("accuracy"), total, "accuracy", path)
        syntax_count = _rate_count(
            payload.get("syntax_rate"), total, "syntax_rate", path
        )
        accuracy = float(payload["accuracy"])
        syntax_rate = float(payload["syntax_rate"])
    nonblank_answers = [answer for answer in generated_answers if answer]
    unique_generated_answers = set(nonblank_answers)
    if num_correct == 0 and syntax_count == 0:
        if not nonblank_answers:
            raise CampaignError(f"{path}: all generated answers are blank")
        if len(unique_generated_answers) == 1:
            raise CampaignError(
                f"{path}: one repeated malformed answer cannot support baseline evidence"
            )
    relative = path.relative_to(repo)
    return {
        "strategy": strategy,
        "num_correct": num_correct,
        "syntax_count": syntax_count,
        "num_examples": total,
        "accuracy": accuracy,
        "syntax_rate": syntax_rate,
        "metric_source": metric_source,
        "nonblank_answer_count": len(nonblank_answers),
        "unique_generated_answer_count": len(unique_generated_answers),
        "source_artifact": str(relative),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
    }


def _read_baseline(
    path: Path,
    total: int,
    strategy: str,
    repo: Path,
    *,
    dataset: str | None = None,
    smiles_class: str | None = None,
) -> dict[str, Any]:
    if not path.is_file():
        raise CampaignError(f"missing baseline artifact: {path}")
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise CampaignError(f"invalid baseline artifact {path}: {exc}") from exc
    return _read_baseline_bytes(
        raw,
        path,
        total,
        strategy,
        repo,
        dataset=dataset,
        smiles_class=smiles_class,
    )


def _cell_id(cohort: Cohort, model_slug: str) -> str:
    return f"{cohort.slug}-{model_slug}"


def build_campaign(
    repo: Path,
    git_commit: str,
    *,
    replacement_root: Path | None = None,
    replacement_labels: set[str] | None = None,
    replacement_paths: dict[str, Path] | None = None,
    evidence_path: Path = EVIDENCE_PATH,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if len(git_commit) != 40:
        raise CampaignError("git_commit must be a full 40-character hash")
    selected_replacements = replacement_labels or set()
    selected_paths = replacement_paths or {}
    if bool(replacement_root) != bool(selected_replacements):
        raise CampaignError(
            "replacement_root and replacement_labels must be provided together"
        )
    if selected_paths and (replacement_root is not None or selected_replacements):
        raise CampaignError(
            "replacement_paths cannot be combined with replacement_root or replacement_labels"
        )
    if replacement_root is not None and not replacement_root.is_absolute():
        replacement_root = repo / replacement_root
    normalized_paths = {
        label: path if path.is_absolute() else repo / path
        for label, path in selected_paths.items()
    }
    known_labels = {
        f"{cohort.slug}-{model[0]}-{strategy}"
        for cohort in COHORTS
        for model in MODELS
        for strategy in STRATEGIES
    }
    unknown_replacements = (
        selected_replacements | set(normalized_paths)
    ) - known_labels
    if unknown_replacements:
        raise CampaignError(
            f"unknown replacement labels: {sorted(unknown_replacements)}"
        )
    evidence_cells: dict[str, Any] = {}
    jobs: list[dict[str, Any]] = []
    for cohort in COHORTS:
        for model in MODELS:
            model_slug, eval_model, reservation_mib, gpu_util = model
            cell = _cell_id(cohort, model_slug)
            rows = []
            for strategy in STRATEGIES:
                label = f"{cell}-{strategy}"
                original_path = baseline_artifact(
                    repo,
                    cohort,
                    model,
                    strategy,
                )
                path = normalized_paths.get(label)
                if path is None:
                    path = baseline_artifact(
                        repo,
                        cohort,
                        model,
                        strategy,
                        campaign_root=(
                            replacement_root if label in selected_replacements else None
                        ),
                    )
                row = _read_baseline(
                    path,
                    cohort.sample_size,
                    strategy,
                    repo,
                    dataset=cohort.dataset,
                    smiles_class=cohort.smiles_class,
                )
                if path.resolve() != original_path.resolve():
                    row["supersedes_source_artifact"] = str(
                        original_path.relative_to(repo)
                    )
                    row["supersedes_source_sha256"] = hashlib.sha256(
                        original_path.read_bytes()
                    ).hexdigest()
                rows.append(row)
            max_correct = max(row["num_correct"] for row in rows)
            max_syntax = max(row["syntax_count"] for row in rows)
            bar = accuracy_bar(max_correct, cohort.sample_size)
            min_accuracy, threshold_policy = bar.min_accuracy, bar.policy
            min_syntax = min(max_syntax / cohort.sample_size, 0.90)
            evidence_cells[cell] = {
                "dataset": cohort.dataset,
                "eval_model": eval_model,
                "split_name": "train",
                "smiles_class": cohort.smiles_class,
                "num_examples": cohort.sample_size,
                "baselines": rows,
                "max_accuracy_count": max_correct,
                "max_accuracy_strategies": [
                    row["strategy"] for row in rows if row["num_correct"] == max_correct
                ],
                "max_syntax_count": max_syntax,
                "max_syntax_strategies": [
                    row["strategy"] for row in rows if row["syntax_count"] == max_syntax
                ],
                "min_accuracy": min_accuracy,
                "min_syntax_rate": min_syntax,
                "threshold_policy": threshold_policy,
            }
            output_name = f"coldq_fullbaseline_20260803_{cell}"
            job: dict[str, Any] = {
                "cell_id": cell,
                "task": cohort.task,
                "dataset": cohort.dataset,
                "eval_model": eval_model,
                "max_iterations": 40,
                "interrupted_author_calls": 0,
                "eval_sample_size": cohort.sample_size,
                "baseline_num_correct": max_correct,
                "baseline_num_examples": cohort.sample_size,
                "baseline_source": str(evidence_path),
                "min_accuracy": min_accuracy,
                "min_syntax_rate": min_syntax,
                "threshold_policy": threshold_policy,
                "eval_max_steps": cohort.max_steps,
                "eval_max_seconds": 90.0,
                "memory_reservation_mib": reservation_mib,
                "gpu_mem_util": gpu_util,
                "output_name": output_name,
                "log_file": f"outputs/generated/{output_name}/run.log",
                "heldout_sample_size": cohort.heldout_sample_size,
                "heldout_split_name": "test",
                "heldout_output_json": str(
                    repo / "outputs/reeval/full_baseline_20260803" / f"{cell}.json"
                ),
                "claude_config_dir": "/home/aadivyar/.claude-csd-synthesis",
                "claude_expected_account": "aadivya@fermi.ai",
            }
            if cohort.dataset == "gsm_symbolic":
                job["heldout_split_file"] = (
                    "environment/benchmark_splits/"
                    "gsm_symbolic_crane_proportional_49x49_seed123.json"
                )
            elif cohort.dataset == "spider":
                job["heldout_split_file"] = (
                    "environment/benchmark_splits/"
                    "spider_dev_proportional_300x300_seed334.json"
                )
            else:
                job["smiles_class"] = cohort.smiles_class
            jobs.append(job)
    evidence = {
        "campaign": CAMPAIGN,
        "git_commit": git_commit,
        "baseline_strategies": list(STRATEGIES),
        "cells": evidence_cells,
    }
    manifest = {
        "campaign": CAMPAIGN,
        "git_commit": git_commit,
        "approved_author_call_cap": 800,
        "jobs": jobs,
    }
    return evidence, manifest


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def validate_campaign(jobs: list[dict[str, Any]], repo: Path) -> None:
    expected_by_cell = {
        _cell_id(cohort, model[0]): (cohort, model)
        for cohort in COHORTS
        for model in MODELS
    }
    expected_cells = set(expected_by_cell)
    if len(jobs) != 20 or {str(job.get("cell_id")) for job in jobs} != expected_cells:
        raise CampaignError("full baseline cold campaign must contain exactly 20 cells")
    if sum(int(job.get("max_iterations") or 0) for job in jobs) != 800:
        raise CampaignError(
            "full baseline cold campaign must contain exactly 800 attempts"
        )
    output_names = [str(job.get("output_name")) for job in jobs]
    if len(set(output_names)) != len(output_names):
        raise CampaignError("cold output names must be unique")
    heldout_outputs = [str(job.get("heldout_output_json")) for job in jobs]
    if len(set(heldout_outputs)) != len(heldout_outputs):
        raise CampaignError("heldout outputs must be unique")
    evidence_sources = {str(job.get("baseline_source", "")) for job in jobs}
    if len(evidence_sources) != 1 or not next(iter(evidence_sources)):
        raise CampaignError("all jobs must bind one campaign evidence file")
    relative_evidence_path = Path(next(iter(evidence_sources)))
    evidence_path = (
        relative_evidence_path
        if relative_evidence_path.is_absolute()
        else repo / relative_evidence_path
    )
    try:
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CampaignError(f"invalid campaign evidence: {exc}") from exc
    if evidence.get("campaign") != CAMPAIGN:
        raise CampaignError("campaign evidence has the wrong campaign id")
    for job in jobs:
        cell = str(job["cell_id"])
        cohort, model = expected_by_cell[cell]
        model_slug, eval_model, reservation_mib, gpu_util = model
        output_name = f"coldq_fullbaseline_20260803_{cell}"
        expected_fields: dict[str, Any] = {
            "task": cohort.task,
            "dataset": cohort.dataset,
            "eval_model": eval_model,
            "max_iterations": 40,
            "interrupted_author_calls": 0,
            "eval_sample_size": cohort.sample_size,
            "baseline_num_examples": cohort.sample_size,
            "baseline_source": str(relative_evidence_path),
            "eval_max_steps": cohort.max_steps,
            "eval_max_seconds": 90.0,
            "memory_reservation_mib": reservation_mib,
            "gpu_mem_util": gpu_util,
            "output_name": output_name,
            "log_file": f"outputs/generated/{output_name}/run.log",
            "heldout_sample_size": cohort.heldout_sample_size,
            "heldout_split_name": "test",
            "heldout_output_json": str(
                repo / "outputs/reeval/full_baseline_20260803" / f"{cell}.json"
            ),
            "claude_config_dir": "/home/aadivyar/.claude-csd-synthesis",
            "claude_expected_account": "aadivya@fermi.ai",
        }
        if cohort.dataset == "gsm_symbolic":
            expected_fields["heldout_split_file"] = (
                "environment/benchmark_splits/"
                "gsm_symbolic_crane_proportional_49x49_seed123.json"
            )
        elif cohort.dataset == "spider":
            expected_fields["heldout_split_file"] = (
                "environment/benchmark_splits/"
                "spider_dev_proportional_300x300_seed334.json"
            )
        else:
            expected_fields["smiles_class"] = cohort.smiles_class
        for field, expected in expected_fields.items():
            if job.get(field) != expected:
                raise CampaignError(f"{cell}: {field} must be {expected!r}")
        if any(str(field).startswith("initial") for field in job):
            raise CampaignError(f"{cell}: warm-start fields are forbidden")
        entry = (evidence.get("cells") or {}).get(cell) or {}
        if (
            entry.get("dataset") != cohort.dataset
            or entry.get("eval_model") != eval_model
            or entry.get("split_name") != "train"
            or entry.get("smiles_class") != cohort.smiles_class
            or int(entry.get("num_examples") or -1) != cohort.sample_size
        ):
            raise CampaignError(f"{cell}: evidence metadata does not match the cell")
        rows = entry.get("baselines") or []
        if [row.get("strategy") for row in rows] != list(STRATEGIES):
            raise CampaignError(f"{cell}: evidence must include all five baselines")
        for strategy, row in zip(STRATEGIES, rows):
            source = repo / str(row.get("source_artifact", ""))
            expected_source = baseline_artifact(repo, cohort, model, strategy)
            if not source.is_file() or hashlib.sha256(
                source.read_bytes()
            ).hexdigest() != row.get("source_sha256"):
                raise CampaignError(f"{cell}: baseline artifact hash mismatch")
            if source.resolve() != expected_source.resolve():
                try:
                    relative_source = source.resolve().relative_to(repo.resolve())
                except ValueError as exc:
                    raise CampaignError(
                        f"{cell}: replacement escaped the repository"
                    ) from exc
                if (
                    len(relative_source.parts) < 3
                    or relative_source.parts[:2] != ("outputs", "baselines")
                    or not relative_source.parts[2].startswith("exact-zero-repair-")
                ):
                    raise CampaignError(
                        f"{cell}: replacement is outside an exact-zero repair root"
                    )
                superseded = repo / str(row.get("supersedes_source_artifact", ""))
                if superseded.resolve() != expected_source.resolve():
                    raise CampaignError(
                        f"{cell}: replacement supersedes the wrong source"
                    )
                if not superseded.is_file() or hashlib.sha256(
                    superseded.read_bytes()
                ).hexdigest() != row.get("supersedes_source_sha256"):
                    raise CampaignError(f"{cell}: superseded baseline hash mismatch")
        total = int(job["eval_sample_size"])
        max_correct = max(int(row["num_correct"]) for row in rows)
        max_syntax = max(int(row["syntax_count"]) for row in rows)
        bar = accuracy_bar(max_correct, total)
        expected_accuracy, expected_policy = bar.min_accuracy, bar.policy
        expected_syntax = min(max_syntax / total, 0.90)
        if (
            int(job["baseline_num_correct"]) != max_correct
            or int(job["baseline_num_examples"]) != total
            or abs(float(job["min_accuracy"]) - expected_accuracy) > 1e-12
            or abs(float(job["min_syntax_rate"]) - expected_syntax) > 1e-12
            or job.get("threshold_policy") != expected_policy
        ):
            raise CampaignError(f"{cell}: thresholds do not match baseline maxima")


def controller_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--baseline-controller-pid", type=int, required=True)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--gpus", default="0,2,3")
    args = parser.parse_args()
    while True:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=args.repo,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        try:
            evidence, manifest = build_campaign(args.repo, commit)
            break
        except CampaignError as exc:
            if not controller_is_alive(args.baseline_controller_pid):
                raise CampaignError(
                    f"baseline controller exited before the matrix was complete: {exc}"
                ) from exc
            print(f"[full-baseline-cold] waiting: {exc}", flush=True)
            time.sleep(args.poll_seconds)
    _atomic_json(args.repo / EVIDENCE_PATH, evidence)
    _atomic_json(args.repo / MANIFEST_PATH, manifest)
    validate_campaign(manifest["jobs"], args.repo)
    command = [
        str(args.python),
        "-m",
        "scripts.runtime.run_cold_synthesis_queue",
        "--repo",
        str(args.repo),
        "--manifest",
        str(args.repo / MANIFEST_PATH),
        "--python",
        str(args.python),
        "--lock-file",
        str(args.repo / ".context/full_baseline_20260803_cold.lock"),
        "--state-dir",
        str(args.repo / ".context/full_baseline_20260803_cold_state"),
        "--campaign-profile",
        CAMPAIGN,
        "--gpus",
        args.gpus,
        "--poll-seconds",
        str(args.poll_seconds),
    ]
    require_synthesis_unblocked(args.repo)
    print(f"[full-baseline-cold] launching {len(manifest['jobs'])} jobs", flush=True)
    return subprocess.run(command, cwd=args.repo).returncode


if __name__ == "__main__":
    raise SystemExit(main())
