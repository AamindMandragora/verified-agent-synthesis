"""Every command that evaluates SMILES names the SMILES split it uses.

Since 2026-09-26 SMILES train/test are two seed ranges
(synthesis/evaluate/benchmarks/smiles/seed_split.py) and nothing loads SMILES
draws without a split. So each launcher that builds a SMILES evaluation must
pass --smiles-split-name, and must pass the same side it passes for GSM/Spider:
held-out re-evals and the paper baseline queue use test; the babysitter smoke
and the collection pool use train.
"""

from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def _value(command, flag):
    command = [str(part) for part in command]
    assert flag in command, f"{flag} missing from {command}"
    return command[command.index(flag) + 1]


def _trial(command):
    """Every SMILES command also names its trial (1, 2 or 3)."""
    return _value(command, "--smiles-trial")


def test_cold_queue_heldout_reeval_scores_the_smiles_test_split():
    from scripts.runtime import run_cold_synthesis_queue as queue
    from tests.runtime.test_cold_synthesis_queue import _job

    job = _job("smiles")
    job["smiles_class"] = "acrylates"
    command = queue.heldout_command(job, Path("/env/python"), Path("/x/GeneratedCSD.py"))
    assert _value(command, "--smiles-split-name") == "test"
    assert _trial(command) == "1"


def test_table5_8_heldout_reeval_scores_the_smiles_test_split(tmp_path):
    from scripts.runtime import run_table5_8_queue as queue
    from tests.runtime.test_table5_8_queue import _fixture_row

    row = _fixture_row("smiles")
    row.update(
        eval_model_revision="1" * 40,
        eval_model_snapshot_path="/cache/snapshots/" + "1" * 40,
        eval_model_snapshot_sha256="3" * 64,
        eval_model_snapshot_file_count=10,
    )
    command = queue.heldout_command(row, Path("python"), tmp_path / "compiled.py")
    assert _value(command, "--smiles-split-name") == "test"
    assert _trial(command) == "1"


@pytest.mark.parametrize("split_name", ["train", "test"])
def test_warm_recovery_heldout_reeval_passes_its_split_through(split_name):
    from scripts.runtime import run_warm_task_recovery_queue as queue
    from tests.runtime.test_warm_task_recovery_queue import _job

    job = _job(dataset="smiles", smiles_class="acrylates", heldout_split_name=split_name)
    command = queue.heldout_command(job, Path("/env/python"), Path("/x/GeneratedCSD.py"))
    assert _value(command, "--smiles-split-name") == split_name
    assert _trial(command) == "1"


def test_paper_baseline_queue_scores_the_smiles_test_split():
    from scripts.runtime import run_paper_baseline_queue as queue

    row = {
        "strategy": "cars", "dataset": "smiles", "eval_model": "Qwen/Qwen3.5-2B",
        "sample_size": 100, "eval_max_steps": 1024, "gpu_mem_util": 0.3,
        "output_json": "/x/out.json", "smiles_class": "acrylates",
    }
    command = queue.fixed_baseline_command(row, Path("/env/python"))
    assert _value(command, "--smiles-split-name") == "test"
    assert _trial(command) == "1"


def test_paper_baseline_queue_recomputes_smiles_metrics_from_a_named_split():
    from scripts.runtime import run_paper_baseline_queue as queue

    assert 'split_name="test"' in inspect.getsource(queue._smiles_uv)


def test_babysitter_smoke_scores_the_smiles_train_split():
    from scripts.runtime.zero_acc_babysitter import smoke

    for fn in (smoke.run_hardened_smoke_job, smoke.run_twin_accuracy_probe):
        assert '"--smiles-split-name", "train"' in inspect.getsource(fn), fn.__name__
        assert '"--smiles-trial", "1"' in inspect.getsource(fn), fn.__name__


@pytest.mark.parametrize(
    "builder, split_name",
    # Each pool builder already picks one side for Spider; SMILES follows it.
    [("build_manifest", "test"), ("build_full_baseline_campaign", "train")],
)
def test_collection_pool_smiles_lanes_use_the_same_side_as_spider(tmp_path, builder, split_name):
    spec = importlib.util.spec_from_file_location(
        "run_focal_collection_pool", REPO / "scripts" / "run_focal_collection_pool.py"
    )
    pool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pool)
    jobs = getattr(pool, builder)(tmp_path)
    spider_sides = {_value(j.args, "--spider-split-name") for j in jobs if "spider" in j.args}
    assert spider_sides == {split_name}
    smiles_jobs = [job for job in jobs if "smiles" in job.args]
    assert smiles_jobs
    for job in smiles_jobs:
        assert _value(job.args, "--smiles-split-name") == split_name, job.label
        assert _trial(job.args) == "1", job.label


def test_reference_strategy_runner_requires_and_records_a_smiles_split(monkeypatch, tmp_path):
    from synthesis.evaluate import run_reference_strategy as runner

    base = ["run_reference_strategy", "--strategy", "gcd", "--dataset", "smiles",
            "--eval-model", "m", "--output-json", str(tmp_path / "o.json")]
    monkeypatch.setattr("sys.argv", base)
    with pytest.raises(SystemExit):
        runner.main()

    seen = {}
    monkeypatch.setattr(runner, "_compile_reference", lambda *a, **k: tmp_path / "GeneratedCSD.py")

    def fake_evaluate(**kwargs):
        seen.update(kwargs)
        return {"accuracy": 0.0, "syntax_rate": 0.0}

    monkeypatch.setattr(runner, "_evaluate", fake_evaluate)
    monkeypatch.setattr(
        "sys.argv", base + ["--smiles-split-name", "train", "--smiles-trial", "1", "--compiled-cache-dir", str(tmp_path / "c")]
    )
    runner.main()
    assert seen["smiles_split_name"] == "train"
