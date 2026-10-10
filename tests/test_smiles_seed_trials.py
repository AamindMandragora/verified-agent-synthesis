"""SMILES trials: three independent sets of draws inside each split's seed range.

Why (user ruling 2026-09-26, following the CARS paper's Table 1 and the
2026-09-18 ruling "sampled lanes: 3 trials, report mean and std"): a SMILES
baseline lane runs 3 trials. Under the seed split, rerunning a lane replays the
same seeds, so each trial needs its own block of seeds:

    trial t of a split uses seeds  split_start + (t-1)*K  ..  split_start + t*K - 1

What these tests pin down:
  * the three trial blocks never overlap, and all sit inside their split range;
  * train and test use the same layout (trial t starts at the same offset);
  * K is big enough for the biggest spender (CARS: up to 10 x 100 draws per lane);
  * every loader, runner and result JSON carries the trial, and nothing defaults it.
"""

from __future__ import annotations

import importlib
import inspect
import json
import sys

import pytest


def _split():
    return importlib.import_module("synthesis.evaluate.benchmarks.smiles.seed_split")


# --------------------------------------------------------------------------
# The trial blocks
# --------------------------------------------------------------------------


def test_there_are_exactly_three_trials():
    assert _split().SMILES_TRIALS == (1, 2, 3)


@pytest.mark.parametrize("split_name", ["train", "test"])
def test_trial_blocks_are_disjoint_and_inside_the_split(split_name):
    split = _split()
    whole = split.seed_range(split_name)
    blocks = [split.trial_seed_range(split_name, t) for t in split.SMILES_TRIALS]
    for block in blocks:
        assert len(block) == split.TRIAL_SEED_BLOCK
        assert block.start >= whole.start and block.stop <= whole.stop
    for i, a in enumerate(blocks):
        for b in blocks[i + 1:]:
            assert set(a).isdisjoint(b)


def test_train_and_test_use_the_same_trial_layout():
    split = _split()
    for t in split.SMILES_TRIALS:
        train_offset = split.trial_seed_range("train", t).start - split.seed_range("train").start
        test_offset = split.trial_seed_range("test", t).start - split.seed_range("test").start
        assert train_offset == test_offset == (t - 1) * split.TRIAL_SEED_BLOCK


def test_a_trial_block_covers_the_biggest_spender():
    """CARS keeps drawing until it has 100 molecules, capped at 10 x 100 = 1000 draws."""
    assert _split().TRIAL_SEED_BLOCK >= 10 * 100


def test_draw_k_of_trial_t_is_block_start_plus_k():
    split = _split()
    for t in split.SMILES_TRIALS:
        block = split.trial_seed_range("train", t)
        assert [split.smiles_draw_seed("train", t, k) for k in (0, 1, 999)] == [
            block.start, block.start + 1, block.start + 999
        ]


@pytest.mark.parametrize("bad_trial", [0, 4, -1, None, "1", 1.0])
def test_unknown_trials_are_refused(bad_trial):
    with pytest.raises(ValueError):
        _split().smiles_draw_seed("train", bad_trial, 0)


def test_a_draw_past_the_trial_block_is_refused():
    split = _split()
    with pytest.raises(ValueError):
        split.smiles_draw_seed("train", 1, split.TRIAL_SEED_BLOCK)


def test_the_record_names_split_trial_and_the_trial_block():
    split = _split()
    block = split.trial_seed_range("test", 2)
    assert split.seed_range_record("test", 2) == {
        "smiles_split_name": "test",
        "smiles_trial": 2,
        "smiles_seed_range": [block.start, block.stop],
    }
    assert split.seed_range_record(None, None) == {
        "smiles_split_name": None, "smiles_trial": None, "smiles_seed_range": None,
    }


# --------------------------------------------------------------------------
# Loading and evaluating draws of one trial
# --------------------------------------------------------------------------


def test_loading_smiles_draws_requires_a_trial():
    from synthesis.evaluate.benchmarks.smiles.dataset import load_smiles

    with pytest.raises((TypeError, ValueError)):
        load_smiles(classes=["acrylates"], samples_per_class=2, split_name="train")


def test_rows_of_different_trials_get_disjoint_seeds():
    from synthesis.evaluate.benchmarks.smiles.dataset import load_smiles

    split = _split()
    by_trial = {
        t: load_smiles(classes=["acrylates"], samples_per_class=3, split_name="train", trial=t)
        for t in split.SMILES_TRIALS
    }
    for t, rows in by_trial.items():
        assert all(r["smiles_trial"] == t for r in rows)
        assert [r["draw_index"] for r in rows] == [0, 1, 2]
        assert [r["draw_seed"] for r in rows] == [split.smiles_draw_seed("train", t, k) for k in range(3)]
    seeds = [r["draw_seed"] for rows in by_trial.values() for r in rows]
    assert len(set(seeds)) == len(seeds)


def test_every_scored_sample_records_its_trial():
    from synthesis.evaluate.benchmarks.smiles.dataset import load_smiles
    from synthesis.evaluate.evaluator import EvaluationResult

    row = load_smiles(classes=["acrylates"], samples_per_class=1, split_name="test", trial=3)[0]
    assert EvaluationResult._sample_identity_metadata(row, 0)["smiles_trial"] == 3


def _smiles_evaluator(**kwargs):
    from synthesis.evaluate.evaluator import Evaluator

    return Evaluator(
        dataset_name="smiles", backend="huggingface", device="cpu", sample_size=2,
        smiles_classes=["acrylates"], **kwargs,
    )


def test_the_evaluator_refuses_smiles_without_a_trial():
    ev = _smiles_evaluator(smiles_split_name="train")
    with pytest.raises(ValueError, match="trial"):
        ev._load_dataset_sample()


def test_the_evaluator_refuses_an_unknown_trial_at_construction():
    with pytest.raises(ValueError):
        _smiles_evaluator(smiles_split_name="train", smiles_trial=4)


def test_the_evaluator_loads_the_trial_it_was_given_and_records_it():
    split = _split()
    ev = _smiles_evaluator(smiles_split_name="train", smiles_trial=2)
    rows = ev._load_dataset_sample()
    assert [r["draw_seed"] for r in rows] == [split.smiles_draw_seed("train", 2, k) for k in range(2)]
    prov = ev.split_provenance()
    block = split.trial_seed_range("train", 2)
    assert prov["smiles_trial"] == 2
    assert prov["smiles_seed_range"] == [block.start, block.stop]


def test_trial_one_and_trial_two_draws_differ():
    """Same split, same draw index, different trial: independent draws, not replays."""
    from synthesis.evaluate.benchmarks.smiles.dataset import load_smiles
    from tests.test_smiles_seed_split import _fake_lm_draw

    one = load_smiles(classes=["acrylates"], samples_per_class=3, split_name="train", trial=1)
    two = load_smiles(classes=["acrylates"], samples_per_class=3, split_name="train", trial=2)
    assert [_fake_lm_draw(r) for r in one] != [_fake_lm_draw(r) for r in two]


# --------------------------------------------------------------------------
# The trial on the command line
# --------------------------------------------------------------------------

_LANE_ARGV = ["--strategy", "gcd", "--dataset", "smiles", "--eval-model", "m",
              "--output-json", "o.json", "--smiles-classes", "acrylates",
              "--smiles-split-name", "train"]


def test_baseline_runner_requires_a_smiles_trial(capsys):
    from synthesis.evaluate import run_legacy_fixed_strategy as lane

    with pytest.raises(SystemExit) as exc:
        lane._parse_args(_LANE_ARGV)
    assert exc.value.code == 2
    assert "--smiles-trial" in capsys.readouterr().err


@pytest.mark.parametrize("bad", ["0", "4", "t1"])
def test_baseline_runner_rejects_trials_outside_one_to_three(bad):
    from synthesis.evaluate import run_legacy_fixed_strategy as lane

    with pytest.raises(SystemExit):
        lane._parse_args(_LANE_ARGV + ["--smiles-trial", bad])


def test_baseline_runner_does_not_ask_other_datasets_for_a_trial():
    from synthesis.evaluate import run_legacy_fixed_strategy as lane

    args = lane._parse_args(["--strategy", "gcd", "--dataset", "spider", "--eval-model", "m",
                             "--output-json", "o.json"])
    assert args.smiles_trial is None


def test_baseline_runner_records_the_trial_in_its_json(tmp_path):
    from synthesis.evaluate import run_legacy_fixed_strategy as lane

    args = lane._parse_args(_LANE_ARGV + ["--smiles-trial", "3"])
    out = tmp_path / "o.json"
    lane._build_minimal_json(
        [{"question": "acrylates", "llm_response": "C=CC(=O)OC", "correct": True, "syntax_valid": True}],
        out,
        eval_split=lane._eval_split_record(args),
    )
    record = json.loads(out.read_text())["eval_split"]
    block = _split().trial_seed_range("train", 3)
    assert record["smiles_trial"] == 3
    assert record["smiles_seed_range"] == [block.start, block.stop]


def test_every_evaluator_the_baseline_runner_builds_gets_the_trial():
    from synthesis.evaluate import run_legacy_fixed_strategy as lane

    source = inspect.getsource(lane)
    assert source.count("smiles_split_name=args.smiles_split_name if dataset") == source.count(
        "smiles_trial=args.smiles_trial if dataset"
    ) > 0


def test_lanes_that_seed_by_index_use_the_trial():
    from synthesis.evaluate import run_legacy_fixed_strategy as lane

    assert "smiles_draw_seed(args.smiles_split_name, args.smiles_trial, _i)" in inspect.getsource(
        lane.run_unconstrained_smiles_adapter
    )
    assert 'template["smiles_trial"]' in inspect.getsource(lane.run_cars_legacy_adapter)


def _reeval_argv(tmp_path, *extra):
    compiled = tmp_path / "GeneratedCSD.py"
    compiled.write_text("")
    return ["reevaluate_compiled_csd", str(compiled), "--dataset", "smiles",
            "--smiles-classes", "acrylates", "--smiles-split-name", "test",
            "--output-json", str(tmp_path / "o.json"), *extra]


def test_reevaluation_requires_a_smiles_trial(monkeypatch, tmp_path, capsys):
    from synthesis.scripts import reevaluate_compiled_csd as reeval

    monkeypatch.setattr(sys, "argv", _reeval_argv(tmp_path))
    with pytest.raises(SystemExit) as exc:
        reeval.main()
    assert exc.value.code == 2
    assert "--smiles-trial" in capsys.readouterr().err


class _Stop(SystemExit):
    pass


def test_reevaluation_scores_and_records_the_trial(monkeypatch, tmp_path):
    from synthesis.scripts import reevaluate_compiled_csd as reeval

    seen = {}

    def fake_evaluator(**kwargs):
        seen.update(kwargs)
        raise _Stop()

    monkeypatch.setattr(reeval, "Evaluator", fake_evaluator)
    monkeypatch.setattr(sys, "argv", _reeval_argv(tmp_path, "--smiles-trial", "2"))
    with pytest.raises(_Stop):
        reeval.main()
    assert seen["smiles_trial"] == 2
    assert "\"smiles_trial\"" in inspect.getsource(reeval.build_reevaluation_provenance)


def test_babysitter_smoke_fallback_gives_trial_one(monkeypatch, tmp_path):
    """The live babysitter builds commands without split flags; its smoke scores train, trial 1."""
    from synthesis.scripts import reevaluate_compiled_csd as reeval

    smoke_dir = tmp_path / "logs" / "zero_acc_babysitter" / "smoke_x"
    smoke_dir.mkdir(parents=True)
    compiled = tmp_path / "GeneratedCSD.py"
    compiled.write_text("")
    seen = {}

    def fake_evaluator(**kwargs):
        seen.update(kwargs)
        raise _Stop()

    monkeypatch.setattr(reeval, "Evaluator", fake_evaluator)
    monkeypatch.setattr(sys, "argv", ["reevaluate_compiled_csd", str(compiled), "--dataset", "smiles",
                                      "--smiles-classes", "acrylates",
                                      "--output-json", str(smoke_dir / "smoke_report.json")])
    with pytest.raises(_Stop):
        reeval.main()
    assert (seen["smiles_split_name"], seen["smiles_trial"]) == ("train", 1)


def test_reference_runner_requires_and_records_the_trial(monkeypatch, tmp_path):
    from synthesis.evaluate import run_reference_strategy as runner

    base = ["run_reference_strategy", "--strategy", "gcd", "--dataset", "smiles", "--eval-model", "m",
            "--output-json", str(tmp_path / "o.json"), "--smiles-split-name", "train",
            "--compiled-cache-dir", str(tmp_path / "c")]
    assert "seed_range_record(smiles_split_name, smiles_trial)" in inspect.getsource(runner._evaluate)
    monkeypatch.setattr("sys.argv", base)
    with pytest.raises(SystemExit):
        runner.main()
    seen = {}
    monkeypatch.setattr(runner, "_compile_reference", lambda *a, **k: tmp_path / "GeneratedCSD.py")

    def fake_evaluate(**kwargs):
        seen.update(kwargs)
        return {"accuracy": 0.0, "syntax_rate": 0.0}

    monkeypatch.setattr(runner, "_evaluate", fake_evaluate)
    monkeypatch.setattr("sys.argv", base + ["--smiles-trial", "2"])
    runner.main()
    assert seen["smiles_trial"] == 2


def test_synthesis_scores_train_trial_one():
    from synthesis.run_constants import SYNTHESIS_SMILES_TRIAL

    assert SYNTHESIS_SMILES_TRIAL == 1
