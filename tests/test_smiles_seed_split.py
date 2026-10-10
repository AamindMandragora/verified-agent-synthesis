"""SMILES train/test split = two disjoint ranges of per-draw seeds.

Why (user ruling 2026-09-26): SMILES has no example list to split. Every draw
sends the same class prompt; what makes one draw different from another is the
random seed its sampler starts from. So a SMILES "example" is (class, seed), and
the fair split is two disjoint seed ranges: baselines on TRAIN seeds set the bar,
synthesis scores TRAIN seeds, the winner is re-evaluated on TEST seeds against
baselines on TEST seeds.

What these tests pin down, straight from that requirement:
  * the train and test seed ranges never overlap;
  * a draw's seed depends only on (split, draw index), so each draw can be
    replayed on its own, in any order, in any process;
  * nothing loads SMILES draws without being told which split;
  * every result JSON says which split and which seed range it used;
  * synthesis scores train; held-out re-eval and baselines must name a split.
"""

from __future__ import annotations

import json
import sys

import pytest

SPLIT_MODULE = "synthesis.evaluate.benchmarks.smiles.seed_split"


def _split():
    import importlib

    return importlib.import_module(SPLIT_MODULE)


# --------------------------------------------------------------------------
# The seed ranges themselves
# --------------------------------------------------------------------------


def test_train_and_test_seed_ranges_do_not_overlap():
    split = _split()
    train = split.seed_range("train")
    test = split.seed_range("test")
    assert len(train) > 0 and len(test) > 0
    assert set(train).isdisjoint(set(test))


def test_every_draw_seed_stays_inside_its_own_split_range():
    split = _split()
    for name in ("train", "test"):
        rng = split.seed_range(name)
        for draw_index in (0, 1, 99, 999, split.TRIAL_SEED_BLOCK - 1):
            assert split.smiles_draw_seed(name, 1, draw_index) in rng


def test_a_draw_seed_is_a_pure_function_of_split_and_draw_index():
    split = _split()
    first = [split.smiles_draw_seed("train", 1, i) for i in range(200)]
    again = [split.smiles_draw_seed("train", 1, i) for i in range(200)]
    assert first == again
    assert len(set(first)) == 200


@pytest.mark.parametrize("bad_split", ["eval", "", None, "TRAIN"])
def test_unknown_split_names_are_refused(bad_split):
    split = _split()
    with pytest.raises(ValueError):
        split.smiles_draw_seed(bad_split, 1, 0)


def test_a_draw_index_outside_the_range_is_refused():
    split = _split()
    with pytest.raises(ValueError):
        split.smiles_draw_seed("train", 1, -1)
    with pytest.raises(ValueError):
        split.smiles_draw_seed("train", 1, split.TRIAL_SEED_BLOCK)


# --------------------------------------------------------------------------
# Loading draws
# --------------------------------------------------------------------------


def test_loading_smiles_draws_requires_a_split():
    from synthesis.evaluate.benchmarks.smiles.dataset import load_smiles

    with pytest.raises((TypeError, ValueError)):
        load_smiles(classes=["acrylates"], samples_per_class=2)


def test_loaded_rows_carry_their_split_draw_index_and_seed():
    from synthesis.evaluate.benchmarks.smiles.dataset import load_smiles

    split = _split()
    train_rows = load_smiles(classes=["acrylates"], samples_per_class=3, split_name="train", trial=1)
    test_rows = load_smiles(classes=["acrylates"], samples_per_class=3, split_name="test", trial=1)

    assert [r["draw_index"] for r in train_rows] == [0, 1, 2]
    assert all(r["smiles_split_name"] == "train" for r in train_rows)
    assert all(r["smiles_split_name"] == "test" for r in test_rows)
    assert [r["draw_seed"] for r in train_rows] == [
        split.smiles_draw_seed("train", 1, i) for i in range(3)
    ]
    assert {r["draw_seed"] for r in train_rows}.isdisjoint({r["draw_seed"] for r in test_rows})


def test_a_later_slice_of_the_same_split_gets_the_later_seeds():
    """The confirmation slice / a shard starting at draw k must not replay draw 0."""
    from synthesis.evaluate.benchmarks.smiles.dataset import load_smiles

    split = _split()
    rows = load_smiles(
        classes=["acrylates"], samples_per_class=2, split_name="train", trial=1, first_draw_index=5
    )
    assert [r["draw_index"] for r in rows] == [5, 6]
    assert [r["draw_seed"] for r in rows] == [
        split.smiles_draw_seed("train", 1, 5),
        split.smiles_draw_seed("train", 1, 6),
    ]


def _smiles_evaluator(**kwargs):
    from synthesis.evaluate.evaluator import Evaluator

    return Evaluator(
        dataset_name="smiles",
        backend="huggingface",
        device="cpu",
        sample_size=2,
        smiles_classes=["acrylates"],
        **kwargs,
    )


def test_the_evaluator_refuses_smiles_without_a_split():
    ev = _smiles_evaluator()
    with pytest.raises(ValueError, match="split"):
        ev._load_dataset_sample()


def test_the_evaluator_loads_the_split_it_was_given():
    split = _split()
    ev = _smiles_evaluator(smiles_split_name="test", smiles_trial=1)
    rows = ev._load_dataset_sample()
    assert [r["draw_seed"] for r in rows] == [
        split.smiles_draw_seed("test", 1, i) for i in range(2)
    ]


def test_the_evaluator_confirmation_offset_moves_to_later_draws():
    split = _split()
    ev = _smiles_evaluator(smiles_split_name="train", smiles_trial=1)
    ev.sample_offset = 2
    rows = ev._load_dataset_sample()
    assert [r["draw_seed"] for r in rows] == [
        split.smiles_draw_seed("train", 1, 2),
        split.smiles_draw_seed("train", 1, 3),
    ]


# --------------------------------------------------------------------------
# Determinism: the evaluator seeds each draw from the row, with a fake LM
# --------------------------------------------------------------------------


def _fake_lm_draw(example: dict) -> list[int]:
    """Run the real `_evaluate_one_example` with a fake LM that samples 24 tokens.

    The fake `logic.build_dynamic_parser` is the first call inside the try. It
    samples from a fixed distribution with torch (as the real constrained
    sampler does) and then raises, which the evaluator catches, so no model,
    GPU or Dafny is needed.
    """
    import torch

    from synthesis.evaluate.evaluator import Evaluator

    captured: list[list[int]] = []
    probs = torch.full((50,), 1.0 / 50)

    class FakeLogic:
        def build_dynamic_parser(self, evaluator, env, ex):
            captured.append(torch.multinomial(probs, 24, replacement=True).tolist())
            raise RuntimeError("fake LM finished")

        def accuracy_applicable(self, aux):
            return True

    ev = Evaluator.__new__(Evaluator)
    ev.dataset_name = "smiles"
    ev.max_steps = 1
    ev.max_seconds_per_example = None
    ev.early_stop_on_answer = False
    ev._format_prompt = lambda ex: "prompt"
    ev._get_expected_answer = lambda ex: "acrylates"
    ev._accuracy_applicable_for_example = lambda aux: True

    Evaluator._evaluate_one_example(
        ev, 0, dict(example), 1, {"lm": None}, FakeLogic(), lambda **kwargs: None, {}
    )
    assert captured, "the fake LM was never called"
    return captured[0]


def test_each_draw_replays_identically_in_any_order():
    import torch

    from synthesis.evaluate.benchmarks.smiles.dataset import load_smiles

    rows = load_smiles(classes=["acrylates"], samples_per_class=4, split_name="train", trial=1)
    forward = [_fake_lm_draw(r) for r in rows]

    torch.manual_seed(987654)  # disturb the global RNG between the two passes
    torch.rand(13)
    backward = [_fake_lm_draw(r) for r in reversed(rows)][::-1]

    assert forward == backward
    assert len({tuple(d) for d in forward}) == 4


def test_train_and_test_draws_differ():
    from synthesis.evaluate.benchmarks.smiles.dataset import load_smiles

    train = load_smiles(classes=["acrylates"], samples_per_class=3, split_name="train", trial=1)
    test = load_smiles(classes=["acrylates"], samples_per_class=3, split_name="test", trial=1)
    assert {tuple(_fake_lm_draw(r)) for r in train}.isdisjoint(
        {tuple(_fake_lm_draw(r)) for r in test}
    )


# --------------------------------------------------------------------------
# The split is recorded in every output JSON
# --------------------------------------------------------------------------


def test_split_provenance_records_the_smiles_split_and_seed_range():
    split = _split()
    ev = _smiles_evaluator(smiles_split_name="test", smiles_trial=1)
    prov = ev.split_provenance()
    rng = split.trial_seed_range("test", 1)
    assert prov["smiles_split_name"] == "test"
    assert prov["smiles_trial"] == 1
    assert prov["smiles_seed_range"] == [rng.start, rng.stop]


def test_every_scored_sample_records_its_split_draw_and_seed():
    """The evaluator stamps each sample with the row's identity (sequential and pool paths)."""
    from synthesis.evaluate.benchmarks.smiles.dataset import load_smiles
    from synthesis.evaluate.evaluator import EvaluationResult

    row = load_smiles(classes=["acrylates"], samples_per_class=2, split_name="test", trial=1)[1]
    meta = EvaluationResult._sample_identity_metadata(row, 1)
    assert meta["smiles_split_name"] == "test"
    assert meta["draw_index"] == 1
    assert meta["draw_seed"] == _split().smiles_draw_seed("test", 1, 1)


def test_split_provenance_has_no_smiles_fields_for_other_datasets():
    from synthesis.split_provenance import build_split_provenance

    prov = build_split_provenance(gsm_split_file="x.json", gsm_split_name="train")
    assert prov["smiles_split_name"] is None
    assert prov["smiles_seed_range"] is None
    assert prov["smiles_trial"] is None


def test_the_baseline_runner_writes_the_split_into_its_json(tmp_path):
    from synthesis.evaluate import run_legacy_fixed_strategy as lane

    args = lane._parse_args(
        [
            "--strategy", "gcd", "--dataset", "smiles", "--eval-model", "m",
            "--output-json", str(tmp_path / "o.json"), "--smiles-classes", "acrylates",
            "--smiles-split-name", "train", "--smiles-trial", "1",
        ]
    )
    out = tmp_path / "o.json"
    lane._build_minimal_json(
        [{"question": "acrylates", "llm_response": "C=CC(=O)OC", "correct": True, "syntax_valid": True}],
        out,
        eval_split=lane._eval_split_record(args),
    )
    payload = json.loads(out.read_text())
    rng = _split().trial_seed_range("train", 1)
    assert payload["eval_split"]["smiles_split_name"] == "train"
    assert payload["eval_split"]["smiles_trial"] == 1
    assert payload["eval_split"]["smiles_seed_range"] == [rng.start, rng.stop]


# --------------------------------------------------------------------------
# The split choice on the command line
# --------------------------------------------------------------------------


def test_baseline_runner_requires_a_smiles_split(capsys):
    from synthesis.evaluate import run_legacy_fixed_strategy as lane

    with pytest.raises(SystemExit) as exc:
        lane._parse_args(
            ["--strategy", "gcd", "--dataset", "smiles", "--eval-model", "m",
             "--output-json", "o.json", "--smiles-classes", "acrylates"]
        )
    assert exc.value.code == 2
    assert "--smiles-split-name" in capsys.readouterr().err


def test_baseline_runner_rejects_a_split_name_outside_train_test():
    from synthesis.evaluate import run_legacy_fixed_strategy as lane

    with pytest.raises(SystemExit):
        lane._parse_args(
            ["--strategy", "gcd", "--dataset", "smiles", "--eval-model", "m",
             "--output-json", "o.json", "--smiles-split-name", "eval"]
        )


def test_baseline_runner_passes_the_split_to_the_lane(monkeypatch, tmp_path):
    from synthesis.evaluate import run_legacy_fixed_strategy as lane

    seen = {}

    def fake_gcd(args):
        seen["split"] = args.smiles_split_name
        return 0

    monkeypatch.setattr(lane, "run_gcd_legacy_adapter", fake_gcd)
    monkeypatch.setattr(lane, "_ensure_repo_cache_env", lambda: tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_legacy_fixed_strategy", "--strategy", "gcd", "--dataset", "smiles",
         "--eval-model", "m", "--output-json", str(tmp_path / "o.json"),
         "--smiles-classes", "acrylates", "--smiles-split-name", "test",
         "--smiles-trial", "1"],
    )
    with pytest.raises(SystemExit) as exc:
        lane.main()
    assert exc.value.code == 0
    assert seen["split"] == "test"


def test_reevaluation_requires_a_smiles_split(monkeypatch, tmp_path, capsys):
    from synthesis.scripts import reevaluate_compiled_csd as reeval

    compiled = tmp_path / "GeneratedCSD.py"
    compiled.write_text("")
    monkeypatch.setattr(
        sys,
        "argv",
        ["reevaluate_compiled_csd", str(compiled), "--dataset", "smiles",
         "--smiles-classes", "acrylates", "--output-json", str(tmp_path / "o.json")],
    )
    with pytest.raises(SystemExit) as exc:
        reeval.main()
    assert exc.value.code == 2
    assert "--smiles-split-name" in capsys.readouterr().err


class _StopAtEvaluator(SystemExit):
    """SystemExit so no broad `except Exception` swallows it."""


def test_reevaluation_scores_the_split_it_was_given(monkeypatch, tmp_path):
    from synthesis.scripts import reevaluate_compiled_csd as reeval

    compiled = tmp_path / "GeneratedCSD.py"
    compiled.write_text("")
    seen = {}

    def fake_evaluator(**kwargs):
        seen.update(kwargs)
        raise _StopAtEvaluator()

    monkeypatch.setattr(reeval, "Evaluator", fake_evaluator)
    monkeypatch.setattr(
        sys,
        "argv",
        ["reevaluate_compiled_csd", str(compiled), "--dataset", "smiles",
         "--smiles-classes", "acrylates", "--smiles-split-name", "test",
         "--smiles-trial", "1"],
    )
    with pytest.raises(_StopAtEvaluator):
        reeval.main()
    assert seen["smiles_split_name"] == "test"


def test_synthesis_scores_the_train_seeds(monkeypatch, tmp_path):
    """run_synthesis has no split flag: it always scores train, for every dataset."""
    import synthesis.evaluate.benchmarks.registry as registry
    import synthesis.evaluate.evaluator as evaluator_mod
    import synthesis.evaluate.feedback_loop as feedback_loop
    import synthesis.generate.generator as generator_mod
    import synthesis.verify.tooling as tooling
    from synthesis import run_synthesis

    captured: dict = {}

    class FakePipeline:
        def __init__(self, **kwargs):
            captured["pipeline"] = kwargs

        def synthesize(self, **kwargs):
            raise _StopAtEvaluator()

    class FakeEvaluator:
        def __init__(self, **kwargs):
            captured["evaluator"] = kwargs

    class FakeGenerator:
        DEFAULT_MODEL = "claude-opus-5-5"

        def __init__(self, **kwargs):
            pass

    monkeypatch.setattr(feedback_loop, "SynthesisPipeline", FakePipeline)
    monkeypatch.setattr(evaluator_mod, "Evaluator", FakeEvaluator)
    monkeypatch.setattr(generator_mod, "StrategyGenerator", FakeGenerator)
    monkeypatch.setattr(tooling, "build_default_verifier", lambda **_: object())
    monkeypatch.setattr(tooling, "build_default_compiler", lambda **_: object())
    monkeypatch.setattr(registry, "resolve_require_delimiters", lambda *a, **k: True)
    monkeypatch.setenv("CSD_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("CSD_OUTPUT_NAME", "smiles_seed_split_test")
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_synthesis", "--task", "t", "--dataset", "smiles", "--min-accuracy", "0.3",
         "--min-syntax-rate", "0.9", "--generation-backend", "claude",
         "--generation-model", "claude-opus-5-5", "--smiles-classes", "acrylates",
         "--bar-from-train"],
    )
    with pytest.raises(_StopAtEvaluator):
        run_synthesis.main()
    assert captured["evaluator"]["smiles_split_name"] == "train"
    assert captured["evaluator"]["smiles_trial"] == 1


# --------------------------------------------------------------------------
# The bar-split guard now covers SMILES like GSM/Spider
# --------------------------------------------------------------------------


def test_a_test_side_bar_on_a_train_scoring_smiles_run_is_refused():
    from synthesis.split_provenance import BarSplitProvenanceError, check_bar_split_provenance

    with pytest.raises(BarSplitProvenanceError):
        check_bar_split_provenance(
            dataset="smiles", split_file=None, split_name="train",
            min_accuracy=0.3, bar_split_name="test",
        )


def test_a_train_side_bar_on_a_train_scoring_smiles_run_is_accepted():
    from synthesis.split_provenance import check_bar_split_provenance

    check_bar_split_provenance(
        dataset="smiles", split_file=None, split_name="train",
        min_accuracy=0.3, bar_split_name="train",
    )
