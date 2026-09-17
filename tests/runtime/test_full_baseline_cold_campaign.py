import hashlib
import json
from pathlib import Path

import pytest

from scripts.runtime import build_full_baseline_cold_manifest as builder
from scripts.runtime import run_cold_synthesis_queue as queue


def _write_artifact(
    path: Path,
    *,
    correct: int,
    syntax: int,
    total: int,
    generated_answers: list[str] | None = None,
) -> None:
    generated = generated_answers or [f"a-{index}" for index in range(total)]
    assert len(generated) == total
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "accuracy": correct / total,
                "syntax_rate": syntax / total,
                "metrics": {"num_examples": total},
                "answers": [
                    {"question": f"q-{index}", "generated_answer": answer}
                    for index, answer in enumerate(generated)
                ],
            }
        ),
        encoding="utf-8",
    )


def _write_all_artifacts(repo: Path, *, perfect: bool = False) -> None:
    for cohort in builder.COHORTS:
        for model in builder.MODELS:
            for index, strategy in enumerate(builder.STRATEGIES):
                total = cohort.sample_size
                correct = (
                    total
                    if perfect and strategy == "cars"
                    else min(total - 1, index + 1)
                )
                syntax = min(total, total - index)
                _write_artifact(
                    builder.baseline_artifact(repo, cohort, model, strategy),
                    correct=correct,
                    syntax=syntax,
                    total=total,
                )


def test_build_campaign_uses_all_five_baselines_and_exact_thresholds(
    tmp_path: Path,
) -> None:
    _write_all_artifacts(tmp_path)

    evidence, manifest = builder.build_campaign(tmp_path, "a" * 40)

    assert len(evidence["cells"]) == 20
    assert len(manifest["jobs"]) == 20
    assert sum(job["max_iterations"] for job in manifest["jobs"]) == 800
    cell = evidence["cells"]["gsm-qwen25-1p5b"]
    assert [row["strategy"] for row in cell["baselines"]] == list(builder.STRATEGIES)
    assert all(len(row["source_sha256"]) == 64 for row in cell["baselines"])
    assert cell["max_accuracy_count"] == 5
    assert cell["max_syntax_count"] == 49
    job = next(job for job in manifest["jobs"] if job["cell_id"] == "gsm-qwen25-1p5b")
    assert job["baseline_num_correct"] == 5
    assert job["min_accuracy"] == 10 / 49
    assert job["min_syntax_rate"] == 0.9
    assert job["threshold_policy"] == "baseline_plus_10_points"
    assert job["claude_expected_account"] == "aadivya@fermi.ai"
    assert job["eval_max_steps"] == 900


def test_perfect_baseline_must_be_matched(
    tmp_path: Path,
) -> None:
    _write_all_artifacts(tmp_path, perfect=True)

    evidence, manifest = builder.build_campaign(tmp_path, "b" * 40)

    cell = evidence["cells"]["spider-qwen35-4b"]
    job = next(job for job in manifest["jobs"] if job["cell_id"] == "spider-qwen35-4b")
    assert cell["max_accuracy_count"] == 300
    assert job["min_accuracy"] == 1.0
    assert job["threshold_policy"] == "perfect_baseline_must_match"


def test_incomplete_or_tampered_baseline_blocks_manifest(tmp_path: Path) -> None:
    _write_all_artifacts(tmp_path)
    missing = builder.baseline_artifact(
        tmp_path, builder.COHORTS[0], builder.MODELS[0], "gcd"
    )
    missing.unlink()

    with pytest.raises(builder.CampaignError, match="missing baseline artifact"):
        builder.build_campaign(tmp_path, "c" * 40)

    _write_artifact(missing, correct=1, syntax=49, total=49)
    payload = json.loads(missing.read_text(encoding="utf-8"))
    payload["answers"].pop()
    missing.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(builder.CampaignError, match="answer count"):
        builder.build_campaign(tmp_path, "c" * 40)


def test_evidence_hash_binds_each_raw_artifact(tmp_path: Path) -> None:
    _write_all_artifacts(tmp_path)
    evidence, _ = builder.build_campaign(tmp_path, "d" * 40)
    row = evidence["cells"]["smiles-isocyanates-qwen35-4b"]["baselines"][0]
    source = tmp_path / row["source_artifact"]

    assert row["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()


def test_repaired_campaign_overlays_only_named_versioned_artifacts(
    tmp_path: Path,
) -> None:
    _write_all_artifacts(tmp_path)
    replacement_root = tmp_path / "outputs/baselines/exact-zero-repair-20260804"
    label = "spider-qwen35-2b-itergen"
    replacement = replacement_root / "spider/qwen35-2b/itergen.json"
    _write_artifact(replacement, correct=7, syntax=250, total=300)

    evidence, _manifest = builder.build_campaign(
        tmp_path,
        "d" * 40,
        replacement_root=replacement_root,
        replacement_labels={label},
    )

    repaired = next(
        row
        for row in evidence["cells"]["spider-qwen35-2b"]["baselines"]
        if row["strategy"] == "itergen"
    )
    preserved = next(
        row
        for row in evidence["cells"]["spider-qwen35-2b"]["baselines"]
        if row["strategy"] == "gcd"
    )
    assert repaired["source_artifact"] == str(replacement.relative_to(tmp_path))
    assert "full_baseline_20260803" in preserved["source_artifact"]


def test_corrected_campaign_accepts_per_label_paths_and_versioned_evidence(
    tmp_path: Path,
) -> None:
    _write_all_artifacts(tmp_path)
    label = "spider-qwen35-2b-itergen"
    replacement = (
        tmp_path
        / "outputs/baselines/exact-zero-repair-20260805-cache-v8"
        / "spider/qwen35-2b/itergen.json"
    )
    _write_artifact(replacement, correct=58, syntax=300, total=300)
    evidence_path = Path(
        "saved-results/2026-08-05-corrected-full-baseline-evidence.json"
    )

    evidence, manifest = builder.build_campaign(
        tmp_path,
        "d" * 40,
        replacement_paths={label: replacement},
        evidence_path=evidence_path,
    )
    builder._atomic_json(tmp_path / evidence_path, evidence)

    repaired = next(
        row
        for row in evidence["cells"]["spider-qwen35-2b"]["baselines"]
        if row["strategy"] == "itergen"
    )
    assert repaired["source_artifact"] == str(replacement.relative_to(tmp_path))
    assert repaired["supersedes_source_artifact"].endswith(
        "full_baseline_20260803/spider/qwen35-2b/itergen.json"
    )
    assert len(repaired["supersedes_source_sha256"]) == 64
    assert {job["baseline_source"] for job in manifest["jobs"]} == {str(evidence_path)}
    builder.validate_campaign(manifest["jobs"], tmp_path)


@pytest.mark.parametrize(
    ("generated_answers", "reason"),
    [
        ([" ", "\t", "\n"], "all generated answers are blank"),
        (["not-a-valid-answer"] * 3, "one repeated malformed answer"),
    ],
)
def test_exact_zero_degenerate_generation_blocks_evidence(
    tmp_path: Path,
    generated_answers: list[str],
    reason: str,
) -> None:
    path = tmp_path / "baseline.json"
    _write_artifact(
        path,
        correct=0,
        syntax=0,
        total=3,
        generated_answers=generated_answers,
    )

    with pytest.raises(builder.CampaignError, match=reason):
        builder._read_baseline(path, 3, "itergen", tmp_path)


def test_diverse_exact_zero_generation_is_recorded_as_a_real_result(
    tmp_path: Path,
) -> None:
    path = tmp_path / "baseline.json"
    _write_artifact(
        path,
        correct=0,
        syntax=0,
        total=3,
        generated_answers=["bad-a", "bad-b", "bad-c"],
    )

    row = builder._read_baseline(path, 3, "itergen", tmp_path)

    assert row["nonblank_answer_count"] == 3
    assert row["unique_generated_answer_count"] == 3


def test_smiles_baseline_accuracy_counts_unique_valid_molecules(tmp_path: Path) -> None:
    path = tmp_path / "baseline.json"
    _write_artifact(
        path,
        correct=3,
        syntax=3,
        total=3,
        generated_answers=["C=CC(=O)OC", "C=CC(=O)OC", "C=CC(=O)OC"],
    )

    row = builder._read_baseline(
        path,
        3,
        "cars",
        tmp_path,
        dataset="smiles",
        smiles_class="acrylates",
    )

    assert row["num_correct"] == 1
    assert row["accuracy"] == 1 / 3
    assert row["syntax_count"] == 3
    assert row["metric_source"] == "recomputed_smiles_unique_valid"


def test_cold_environment_uses_the_manifest_bound_claude_account(
    tmp_path: Path,
) -> None:
    _write_all_artifacts(tmp_path)
    _, manifest = builder.build_campaign(tmp_path, "e" * 40)
    job = manifest["jobs"][0]

    environment = queue.synthesis_environment(job, (0, 2), {}, tmp_path)

    assert (
        environment["CSD_CLAUDE_CONFIG_DIR"] == "/home/aadivyar/.claude-csd-synthesis"
    )
    assert environment["CSD_CLAUDE_EXPECTED_ACCOUNT"] == "aadivya@fermi.ai"


def test_validation_rejects_manifest_contract_tampering(tmp_path: Path) -> None:
    _write_all_artifacts(tmp_path)
    evidence, manifest = builder.build_campaign(tmp_path, "f" * 40)
    builder._atomic_json(tmp_path / builder.EVIDENCE_PATH, evidence)

    manifest["jobs"][0]["eval_model"] = "Qwen/Other"
    with pytest.raises(builder.CampaignError, match="eval_model"):
        builder.validate_campaign(manifest["jobs"], tmp_path)

    _, manifest = builder.build_campaign(tmp_path, "f" * 40)
    manifest["jobs"][0]["initial_strategy"] = "forbidden"
    with pytest.raises(builder.CampaignError, match="warm-start"):
        builder.validate_campaign(manifest["jobs"], tmp_path)

    _, manifest = builder.build_campaign(tmp_path, "f" * 40)
    manifest["jobs"][1]["heldout_output_json"] = manifest["jobs"][0][
        "heldout_output_json"
    ]
    with pytest.raises(builder.CampaignError, match="heldout outputs"):
        builder.validate_campaign(manifest["jobs"], tmp_path)
