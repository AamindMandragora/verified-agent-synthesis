"""Seal state-faithful inputs for a Spider synthesis continuation.

Complete evaluations remain useful search evidence even when the outer attempt
timer was exceeded. This preparer selects the same incumbent that the synthesis
loop will restore, keeps every consumed attempt number, and rebuilds the
persistent failure ledger from the summaries saved in the progress report.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping, Sequence


LOGGER = logging.getLogger(__name__)
LOG_PREFIX = "[spider-continuation]"


@dataclass(frozen=True)
class ContinuationPlan:
    initial_attempt_offset: int
    remaining_attempts: int
    incumbent_attempt_number: int
    incumbent_strategy: str


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_text(payload: str) -> str:
    return _sha256_bytes(payload.encode("utf-8"))


def _persistence_summary(text: str) -> str:
    lines = text.splitlines()
    start = next(
        (
            index
            for index, line in enumerate(lines)
            if line == "Cross-attempt mode persistence:"
        ),
        None,
    )
    if start is None:
        return ""
    summary = [lines[start]]
    for line in lines[start + 1 :]:
        if not line.startswith("  - mode_"):
            break
        summary.append(line)
    return "\n".join(summary)


def _is_complete_evaluation(evaluation: Mapping[str, object]) -> bool:
    planned = evaluation.get("planned_num_examples")
    observed = evaluation.get("num_examples")
    samples = evaluation.get("sample_outputs")
    return bool(
        evaluation.get("success") is True
        and isinstance(planned, int)
        and planned > 0
        and isinstance(observed, int)
        and observed == planned
        and isinstance(samples, list)
        and len(samples) == planned
        and evaluation.get("early_stopped") is not True
    )


def _shortfall(
    evaluation: Mapping[str, object], *, min_accuracy: float, min_syntax_rate: float
) -> float:
    return max(0.0, min_accuracy - float(evaluation["accuracy"])) + max(
        0.0, min_syntax_rate - float(evaluation["syntax_rate"])
    )


def build_continuation_plan(
    report: Mapping[str, object],
    *,
    min_accuracy: float,
    min_syntax_rate: float,
    final_attempt_limit: int,
) -> ContinuationPlan:
    """Validate the finalized history and select its best eligible incumbent."""
    attempts = report.get("attempts")
    total = report.get("total_attempts")
    if not isinstance(attempts, list) or not attempts or not isinstance(total, int):
        raise ValueError("progress report has no finalized attempt history")
    if total != len(attempts):
        raise ValueError("progress report attempt count is not finalized")
    if [record.get("attempt_number") for record in attempts] != list(
        range(1, total + 1)
    ):
        raise ValueError("attempt history is not contiguous from attempt 1")
    if total >= final_attempt_limit:
        raise ValueError("no attempts remain under the requested final limit")

    eligible: list[tuple[Mapping[str, object], Mapping[str, object]]] = []
    for record in attempts:
        if not isinstance(record, Mapping):
            raise ValueError("attempt history contains a non-object record")
        strategy = record.get("strategy_code")
        evaluation = record.get("evaluation")
        if not isinstance(strategy, str) or not strategy.strip():
            raise ValueError("attempt history contains an empty strategy")
        if not isinstance(evaluation, Mapping):
            continue
        if not all(field in evaluation for field in ("accuracy", "syntax_rate")):
            continue
        failed_at = record.get("failed_at")
        score_bearing = failed_at in (None, "evaluation") or (
            failed_at == "timeout" and _is_complete_evaluation(evaluation)
        )
        if score_bearing:
            eligible.append((record, evaluation))
    if not eligible:
        raise ValueError("attempt history has no complete score-bearing evaluation")

    winner, winner_evaluation = min(
        eligible,
        key=lambda pair: (
            _shortfall(
                pair[1],
                min_accuracy=min_accuracy,
                min_syntax_rate=min_syntax_rate,
            ),
            -float(pair[1]["accuracy"]),
            -float(pair[1]["syntax_rate"]),
            int(pair[0]["attempt_number"]),
        ),
    )
    LOGGER.info(
        "%s selected incumbent attempt=%d accuracy=%.6f syntax=%.6f offset=%d",
        LOG_PREFIX,
        winner["attempt_number"],
        float(winner_evaluation["accuracy"]),
        float(winner_evaluation["syntax_rate"]),
        total,
    )
    return ContinuationPlan(
        initial_attempt_offset=total,
        remaining_attempts=final_attempt_limit - total,
        incumbent_attempt_number=int(winner["attempt_number"]),
        incumbent_strategy=str(winner["strategy_code"]),
    )


def reconstruct_failure_ledger(
    seed_payload: Mapping[str, object],
    finalized_attempts: Sequence[Mapping[str, object]],
    *,
    seed_through_attempt: int,
    render_cluster_block: Callable[..., object] | None = None,
) -> tuple[dict, list[int], list[dict[str, object]]]:
    """Replay saved feedback summaries and fail closed when they disagree."""
    if seed_payload.get("version") != 1 or not isinstance(
        seed_payload.get("ledger"), Mapping
    ):
        raise ValueError("seed failure ledger is not a version-1 payload")
    payload = copy.deepcopy(dict(seed_payload))
    included = list(payload.get("included_attempts", []))
    if any(
        not isinstance(number, int) or number > seed_through_attempt
        for number in included
    ):
        raise ValueError("seed failure ledger includes an attempt past its boundary")
    known_attempts = set(included)

    if render_cluster_block is None:
        from synthesis.failure_taxonomy import render_cluster_block

    replayed: list[int] = []
    skipped: list[dict[str, object]] = []
    for record in finalized_attempts:
        number = record.get("attempt_number")
        if not isinstance(number, int) or number <= seed_through_attempt:
            raise ValueError("ledger replay received an invalid post-seed attempt")
        if number in known_attempts:
            continue
        actual_summary = _persistence_summary(str(record.get("error_summary", "")))
        if not actual_summary:
            if record.get("failed_at") != "timeout":
                raise ValueError(
                    f"attempt {number} has no saved persistence summary"
                )
            skipped.append(
                {
                    "attempt_number": number,
                    "reason": "timeout_without_saved_persistence_summary",
                    "error_summary_sha256": _sha256_text(
                        str(record.get("error_summary", ""))
                    ),
                }
            )
            continue
        evaluation = record.get("evaluation")
        if not isinstance(evaluation, Mapping) or not isinstance(
            evaluation.get("sample_outputs"), list
        ):
            raise ValueError(f"attempt {number} has no saved samples for replay")
        rendered = render_cluster_block(
            evaluation["sample_outputs"],
            persistent_ledger=payload["ledger"],
            attempt_index=number,
            max_steps=512,
            slow_threshold_seconds=30.0,
            require_delimiters=False,
        )
        expected_summary = _persistence_summary(str(rendered))
        if expected_summary != actual_summary:
            raise ValueError(
                f"attempt {number} persistence summary does not match saved report"
            )
        included.append(number)
        known_attempts.add(number)
        replayed.append(number)
    payload["included_attempts"] = sorted(included)
    LOGGER.info(
        "%s rebuilt ledger replayed=%s skipped_timeouts=%s",
        LOG_PREFIX,
        replayed,
        [entry["attempt_number"] for entry in skipped],
    )
    return payload, replayed, skipped


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _json_bytes(payload: Mapping[str, object]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def prepare(args: argparse.Namespace) -> dict[str, object]:
    report_bytes = args.progress_report.read_bytes()
    report = json.loads(report_bytes)
    seed_ledger = json.loads(args.seed_failure_ledger.read_text(encoding="utf-8"))
    plan = build_continuation_plan(
        report,
        min_accuracy=args.min_accuracy,
        min_syntax_rate=args.min_syntax_rate,
        final_attempt_limit=args.final_attempt_limit,
    )
    post_seed = [
        record
        for record in report["attempts"]
        if record["attempt_number"] > args.seed_through_attempt
    ]
    rebuilt_ledger, replayed, skipped = reconstruct_failure_ledger(
        seed_ledger,
        post_seed,
        seed_through_attempt=args.seed_through_attempt,
    )

    output = args.output_dir.resolve()
    history_path = output / "history.json"
    ledger_path = output / "failure-ledger.json"
    incumbent_path = output / "incumbent.dfyinc"
    manifest_path = output / "manifest.json"
    ledger_bytes = _json_bytes(rebuilt_ledger)
    incumbent_bytes = plan.incumbent_strategy.encode("utf-8")
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_progress_report": str(args.progress_report.resolve()),
        "source_progress_report_sha256": _sha256_bytes(report_bytes),
        "seed_failure_ledger": str(args.seed_failure_ledger.resolve()),
        "seed_failure_ledger_sha256": _sha256_bytes(
            args.seed_failure_ledger.read_bytes()
        ),
        "history_sha256": _sha256_bytes(report_bytes),
        "failure_ledger_sha256": _sha256_bytes(ledger_bytes),
        "incumbent_sha256": _sha256_bytes(incumbent_bytes),
        "incumbent_attempt_number": plan.incumbent_attempt_number,
        "initial_attempt_offset": plan.initial_attempt_offset,
        "remaining_attempts": plan.remaining_attempts,
        "final_attempt_limit": args.final_attempt_limit,
        "thresholds": {
            "min_accuracy": args.min_accuracy,
            "min_syntax_rate": args.min_syntax_rate,
        },
        "ledger_replayed_attempts": replayed,
        "ledger_skipped_timeouts": skipped,
        "artifacts": {
            "history": str(history_path),
            "failure_ledger": str(ledger_path),
            "incumbent": str(incumbent_path),
        },
    }
    _atomic_write(history_path, report_bytes)
    _atomic_write(ledger_path, ledger_bytes)
    _atomic_write(incumbent_path, incumbent_bytes)
    _atomic_write(manifest_path, _json_bytes(manifest))
    LOGGER.info("%s wrote sealed inputs output=%s", LOG_PREFIX, output)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--progress-report", type=Path, required=True)
    parser.add_argument("--seed-failure-ledger", type=Path, required=True)
    parser.add_argument("--seed-through-attempt", type=int, required=True)
    parser.add_argument("--min-accuracy", type=float, required=True)
    parser.add_argument("--min-syntax-rate", type=float, required=True)
    parser.add_argument("--final-attempt-limit", type=int, default=43)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    prepare(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
