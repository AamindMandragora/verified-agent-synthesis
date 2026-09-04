"""Seal the exact inputs for the two-attempt Opus warm continuation."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Mapping, Sequence

from synthesis.failure_taxonomy import make_persistent_ledger, render_cluster_block


LOGGER = logging.getLogger(__name__)
LOG_PREFIX = "[opus-continuation]"
EXPECTED_PROGRESS_REPORT_SHA256 = (
    "803d69e087e963e5f05efebf2037c45491d716532534ea2b6bbc96b0c1d758bf"
)
INITIAL_ATTEMPT_OFFSET = 38
NEW_ITERATIONS = 2
TOTAL_ATTEMPT_CAP = 40


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_sha256(payload: object) -> str:
    return _sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )


def _persistence_summary(text: str) -> str:
    lines = text.splitlines()
    start = next(
        (index for index, line in enumerate(lines) if line == "Cross-attempt mode persistence:"),
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


def validate_attempt_sequence(report: Mapping[str, object]) -> list[int]:
    """Require the complete, immutable attempt-1 through attempt-38 history."""
    attempts = report.get("attempts")
    numbers = (
        [attempt.get("attempt_number") for attempt in attempts]
        if isinstance(attempts, list) and all(isinstance(attempt, Mapping) for attempt in attempts)
        else []
    )
    expected = list(range(1, INITIAL_ATTEMPT_OFFSET + 1))
    if report.get("total_attempts") != INITIAL_ATTEMPT_OFFSET or numbers != expected:
        raise ValueError("attempt history must be contiguous from 1 through 38")
    return expected


def reconstruct_failure_ledger(
    report: Mapping[str, object], *, source_progress_report_sha256: str
) -> tuple[dict[str, object], dict[str, object]]:
    """Replay only saved evaluation failures and prove the saved mode history."""
    validate_attempt_sequence(report)
    ledger = make_persistent_ledger()
    replayed_attempts: list[int] = []
    attempt_proofs: list[dict[str, object]] = []
    for attempt in report["attempts"]:
        if attempt.get("failed_at") != "evaluation":
            continue
        number = attempt["attempt_number"]
        samples = attempt.get("evaluation", {}).get("sample_outputs")
        if not isinstance(samples, list):
            raise ValueError(f"attempt {number} has no saved samples for ledger replay")
        rendered = render_cluster_block(
            samples,
            persistent_ledger=ledger,
            attempt_index=number,
            max_steps=512,
            slow_threshold_seconds=30.0,
            require_delimiters=True,
        )
        saved_summary = _persistence_summary(str(attempt.get("error_summary", "")))
        rendered_summary = _persistence_summary(rendered)
        if not saved_summary or saved_summary != rendered_summary:
            raise ValueError(f"attempt {number} saved mode summary does not match replay")
        replayed_attempts.append(number)
        attempt_proofs.append(
            {
                "attempt_number": number,
                "saved_error_summary_sha256": _sha256(
                    str(attempt.get("error_summary", "")).encode("utf-8")
                ),
                "rendered_cluster_block_sha256": _sha256(rendered.encode("utf-8")),
            }
        )
    payload: dict[str, object] = {"version": 1, "ledger": ledger}
    proof: dict[str, object] = {
        "source_progress_report_sha256": source_progress_report_sha256,
        "replayed_attempts": replayed_attempts,
        "verified_cluster_block_count": len(attempt_proofs),
        "cluster_block_mismatches": [],
        "attempt_proofs": attempt_proofs,
    }
    LOGGER.info(
        "%s rebuilt modes=%d replayed_attempts=%s",
        LOG_PREFIX,
        len(ledger["modes"]),
        replayed_attempts,
    )
    return payload, proof


def _json_bytes(payload: object) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, path)


def prepare(*, progress_report: Path, output_dir: Path) -> dict[str, object]:
    """Write an immutable history, seed, ledger, proof, and one-row manifest."""
    report_bytes = progress_report.read_bytes()
    source_sha = _sha256(report_bytes)
    if source_sha != EXPECTED_PROGRESS_REPORT_SHA256:
        raise ValueError("progress report SHA-256 does not match the approved Opus history")
    report = json.loads(report_bytes)
    validate_attempt_sequence(report)
    incumbent = report["attempts"][-1]
    strategy = incumbent.get("strategy_code")
    if not isinstance(strategy, str) or not strategy.strip():
        raise ValueError("attempt 38 has no incumbent strategy")
    ledger, proof = reconstruct_failure_ledger(
        report, source_progress_report_sha256=source_sha
    )
    ledger_bytes = _json_bytes(ledger)
    proof_bytes = _json_bytes(proof)
    incumbent_bytes = strategy.encode("utf-8")
    output = output_dir.resolve()
    history_path = output / "history.json"
    incumbent_path = output / "incumbent.dfyinc"
    ledger_path = output / "failure-ledger.json"
    proof_path = output / "failure-ledger-proof.json"
    manifest_path = output / "manifest.json"
    manifest: dict[str, object] = {
        "source_progress_report": str(progress_report.resolve()),
        "source_progress_report_sha256": source_sha,
        "history_sha256": source_sha,
        "incumbent_strategy_sha256": _sha256(incumbent_bytes),
        "initial_attempt_offset": INITIAL_ATTEMPT_OFFSET,
        "new_iterations": NEW_ITERATIONS,
        "total_attempt_cap": TOTAL_ATTEMPT_CAP,
        "fixed_warm_continuation": True,
        "reconstructed_failure_ledger": {
            "artifact_sha256": _sha256(ledger_bytes),
            "mode_count": len(ledger["ledger"]["modes"]),
            "proof_sha256": _sha256(proof_bytes),
            "source_progress_report_sha256": source_sha,
            "state_sha256": _canonical_sha256(ledger["ledger"]),
        },
        "artifacts": {
            "history": str(history_path),
            "incumbent": str(incumbent_path),
            "failure_ledger": str(ledger_path),
            "failure_ledger_proof": str(proof_path),
        },
    }
    _atomic_write(history_path, report_bytes)
    _atomic_write(incumbent_path, incumbent_bytes)
    _atomic_write(ledger_path, ledger_bytes)
    _atomic_write(proof_path, proof_bytes)
    _atomic_write(manifest_path, _json_bytes(manifest))
    LOGGER.info("%s sealed continuation inputs output=%s", LOG_PREFIX, output)
    return manifest
