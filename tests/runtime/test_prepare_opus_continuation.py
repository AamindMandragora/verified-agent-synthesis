import hashlib
import json
import os
from pathlib import Path

import pytest

from scripts.runtime.prepare_opus_continuation import (
    prepare,
    reconstruct_failure_ledger,
    validate_attempt_sequence,
)


EXPECTED_PROGRESS_REPORT_SHA256 = (
    "803d69e087e963e5f05efebf2037c45491d716532534ea2b6bbc96b0c1d758bf"
)
EXPECTED_INCUMBENT_STRATEGY_SHA256 = (
    "f58a69d68c8ff5f78b9318a29cfb0c73e6b3159988320a8ed57b5f78b85ba9c1"
)
EXPECTED_FAILURE_LEDGER_STATE_SHA256 = (
    "418d40584120afc8508b85376c2df9e84a5dbd6f7cb4866fa9533b398f92f02e"
)
EXPECTED_REPLAYED_ATTEMPTS = [
    1,
    2,
    3,
    6,
    7,
    11,
    12,
    13,
    14,
    17,
    18,
    20,
    21,
    23,
    24,
    25,
    26,
    29,
    34,
    35,
    36,
    37,
]
DEFAULT_HISTORICAL_REPORT = Path(
    "/home/aadivyar/csd-generation-worktrees/full-baseline-campaign-20260803/"
    "outputs/generated/coldq_fullbaseline_20260803_gsm-qwen25-1p5b/"
    "coldq_fullbaseline_20260803_gsm-qwen25-1p5b_20260804_060028_e0a771/"
    "results/progress_report.json"
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return _sha256(encoded)


@pytest.fixture(scope="module")
def historical_report_path() -> Path:
    configured = os.environ.get("CSD_OPUS_CONTINUATION_REPORT")
    path = Path(configured) if configured else DEFAULT_HISTORICAL_REPORT
    if not path.is_file():
        pytest.skip(f"historical Opus report is unavailable: {path}")
    return path


@pytest.fixture(scope="module")
def historical_report_bytes(historical_report_path: Path) -> bytes:
    payload = historical_report_path.read_bytes()
    assert _sha256(payload) == EXPECTED_PROGRESS_REPORT_SHA256
    return payload


@pytest.fixture(scope="module")
def historical_report(historical_report_bytes: bytes) -> dict:
    return json.loads(historical_report_bytes)


def test_attempt_sequence_requires_exactly_contiguous_attempts_1_through_38():
    valid = {
        "total_attempts": 38,
        "attempts": [
            {"attempt_number": number}
            for number in range(1, 39)
        ],
    }
    assert validate_attempt_sequence(valid) == list(range(1, 39))

    missing = json.loads(json.dumps(valid))
    del missing["attempts"][16]
    missing["total_attempts"] = 37
    with pytest.raises(ValueError, match="contiguous.*1.*38"):
        validate_attempt_sequence(missing)

    duplicate = json.loads(json.dumps(valid))
    duplicate["attempts"][17]["attempt_number"] = 17
    with pytest.raises(ValueError, match="contiguous.*1.*38"):
        validate_attempt_sequence(duplicate)


def test_reconstructs_the_exact_24_mode_ledger_deterministically(
    historical_report: dict,
):
    first_ledger, first_proof = reconstruct_failure_ledger(
        historical_report,
        source_progress_report_sha256=EXPECTED_PROGRESS_REPORT_SHA256,
    )
    second_ledger, second_proof = reconstruct_failure_ledger(
        historical_report,
        source_progress_report_sha256=EXPECTED_PROGRESS_REPORT_SHA256,
    )

    assert first_ledger == second_ledger
    assert first_proof == second_proof
    assert first_ledger["version"] == 1
    assert first_ledger["ledger"]["next_id"] == 24
    assert len(first_ledger["ledger"]["modes"]) == 24
    assert (
        _canonical_sha256(first_ledger["ledger"])
        == EXPECTED_FAILURE_LEDGER_STATE_SHA256
    )
    assert first_proof["source_progress_report_sha256"] == (
        EXPECTED_PROGRESS_REPORT_SHA256
    )
    assert first_proof["replayed_attempts"] == EXPECTED_REPLAYED_ATTEMPTS
    assert first_proof["verified_cluster_block_count"] == 22
    assert first_proof["cluster_block_mismatches"] == []
    assert len(first_proof["attempt_proofs"]) == 22
    assert all(
        proof["attempt_number"] in EXPECTED_REPLAYED_ATTEMPTS
        and len(proof["saved_error_summary_sha256"]) == 64
        and len(proof["rendered_cluster_block_sha256"]) == 64
        for proof in first_proof["attempt_proofs"]
    )


def test_prepare_seals_exact_history_seed_ledger_and_manifest(
    tmp_path: Path,
    historical_report_path: Path,
    historical_report_bytes: bytes,
    historical_report: dict,
):
    output_dir = tmp_path / "sealed"
    manifest = prepare(
        progress_report=historical_report_path,
        output_dir=output_dir,
    )

    history_path = output_dir / "history.json"
    incumbent_path = output_dir / "incumbent.dfyinc"
    ledger_path = output_dir / "failure-ledger.json"
    proof_path = output_dir / "failure-ledger-proof.json"
    manifest_path = output_dir / "manifest.json"

    assert history_path.read_bytes() == historical_report_bytes
    assert _sha256(history_path.read_bytes()) == EXPECTED_PROGRESS_REPORT_SHA256
    assert incumbent_path.read_text(encoding="utf-8") == (
        historical_report["attempts"][-1]["strategy_code"]
    )
    assert _sha256(incumbent_path.read_bytes()) == (
        EXPECTED_INCUMBENT_STRATEGY_SHA256
    )

    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    assert len(ledger["ledger"]["modes"]) == 24
    assert _canonical_sha256(ledger["ledger"]) == (
        EXPECTED_FAILURE_LEDGER_STATE_SHA256
    )

    assert manifest == json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["source_progress_report_sha256"] == (
        EXPECTED_PROGRESS_REPORT_SHA256
    )
    assert manifest["incumbent_strategy_sha256"] == (
        EXPECTED_INCUMBENT_STRATEGY_SHA256
    )
    assert manifest["initial_attempt_offset"] == 38
    assert manifest["new_iterations"] == 2
    assert manifest["total_attempt_cap"] == 40
    assert manifest["fixed_warm_continuation"] is True
    assert manifest["reconstructed_failure_ledger"] == {
        "artifact_sha256": _sha256(ledger_path.read_bytes()),
        "mode_count": 24,
        "proof_sha256": _sha256(proof_path.read_bytes()),
        "source_progress_report_sha256": EXPECTED_PROGRESS_REPORT_SHA256,
        "state_sha256": EXPECTED_FAILURE_LEDGER_STATE_SHA256,
    }
    assert proof["source_progress_report_sha256"] == (
        EXPECTED_PROGRESS_REPORT_SHA256
    )


def test_prepare_rejects_tampered_source_before_writing_outputs(tmp_path: Path):
    tampered = tmp_path / "tampered-progress-report.json"
    tampered.write_text('{"attempts": [], "total_attempts": 0}\n', encoding="utf-8")
    output_dir = tmp_path / "sealed"

    with pytest.raises(ValueError, match="progress report SHA-256"):
        prepare(progress_report=tampered, output_dir=output_dir)

    assert not output_dir.exists()
