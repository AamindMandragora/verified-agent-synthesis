"""run_synthesis must be told, on every launch, which split the accuracy bar came from.

Why (2026-09-26): run_synthesis hard-coded bar_split_name="train". The launch
guard (synthesis/split_provenance.py::check_bar_split_provenance) compares the
bar's split with the split synthesis scores (always train), so the hard-code
meant the guard could never fire, and every progress report claimed the bar was
from train -- even when the bar had been measured on test. That is the July
2026 incident again, for all GSM and Spider cells.

The fix: two flags, --bar-from-train and --bar-from-test, exactly one of which
is required. The chosen split flows into the guard and into the pipeline
(which writes it into progress_report.json via split_provenance).
"""

from __future__ import annotations

import sys

import pytest

from synthesis import run_synthesis


class _StopAfterPipelineBuilt(SystemExit):
    """SystemExit subclass so main()'s broad `except Exception` does not swallow it."""


def _base_argv(dataset: str = "gsm_symbolic") -> list[str]:
    argv = [
        "run_synthesis",
        "--task", "t",
        "--dataset", dataset,
        "--min-accuracy", "0.3",
        "--min-syntax-rate", "0.9",
        "--generation-backend", "claude",
        "--generation-model", "claude-opus-5",
    ]
    if dataset == "smiles":
        argv += ["--smiles-classes", "acrylates"]
    return argv


@pytest.fixture
def stubbed_heavy_parts(monkeypatch, tmp_path):
    """Replace model/Dafny/eval construction with recorders so main() stays off-GPU."""
    import synthesis.evaluate.benchmarks.registry as registry
    import synthesis.evaluate.evaluator as evaluator_mod
    import synthesis.evaluate.feedback_loop as feedback_loop
    import synthesis.generate.generator as generator_mod
    import synthesis.verify.tooling as tooling

    captured: dict = {}

    class FakePipeline:
        def __init__(self, **kwargs):
            captured["pipeline_kwargs"] = kwargs

        def synthesize(self, **kwargs):
            raise _StopAfterPipelineBuilt()

    class FakeEvaluator:
        def __init__(self, **kwargs):
            captured["evaluator_kwargs"] = kwargs

    class FakeGenerator:
        DEFAULT_MODEL = "claude-opus-5"

        def __init__(self, **kwargs):
            pass

    monkeypatch.setattr(feedback_loop, "SynthesisPipeline", FakePipeline)
    monkeypatch.setattr(evaluator_mod, "Evaluator", FakeEvaluator)
    monkeypatch.setattr(generator_mod, "StrategyGenerator", FakeGenerator)
    monkeypatch.setattr(tooling, "build_default_verifier", lambda **_: object())
    monkeypatch.setattr(tooling, "build_default_compiler", lambda **_: object())
    monkeypatch.setattr(registry, "resolve_require_delimiters", lambda *a, **k: True)
    monkeypatch.setenv("CSD_OUTPUT_DIR", str(tmp_path))
    # The GSM data folder is an ignored checkout; its presence is not under test.
    monkeypatch.setattr(run_synthesis, "GSM_SOURCE_DIR", str(tmp_path))
    monkeypatch.setenv("CSD_OUTPUT_NAME", "bar_split_flag_test")
    return captured


def _run_main(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", argv)
    run_synthesis.main()


def test_launch_without_either_bar_flag_is_a_parse_error(monkeypatch, capsys):
    with pytest.raises(SystemExit) as exc:
        _run_main(monkeypatch, _base_argv())
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "--bar-from-train" in err and "--bar-from-test" in err


def test_launch_with_both_bar_flags_is_a_parse_error(monkeypatch, capsys):
    with pytest.raises(SystemExit) as exc:
        _run_main(monkeypatch, _base_argv() + ["--bar-from-train", "--bar-from-test"])
    assert exc.value.code == 2
    assert "not allowed with argument" in capsys.readouterr().err


@pytest.mark.parametrize("dataset", ["gsm_symbolic", "spider"])
def test_bar_from_train_reaches_the_pipeline_as_train(monkeypatch, stubbed_heavy_parts, dataset):
    with pytest.raises(_StopAfterPipelineBuilt):
        _run_main(monkeypatch, _base_argv(dataset) + ["--bar-from-train"])
    assert stubbed_heavy_parts["pipeline_kwargs"]["bar_split_name"] == "train"


def test_bar_from_test_reaches_the_pipeline_as_test(monkeypatch, stubbed_heavy_parts):
    # With no accuracy bar (--min-accuracy 0) the guard has nothing to compare,
    # so a test-side declaration launches; this proves the recorded value follows
    # the flag rather than a hard-coded "train". (Since 2026-09-26 SMILES has a
    # train/test seed split, so a test-side SMILES bar is refused like GSM/Spider.)
    argv = _base_argv("smiles")
    argv[argv.index("--min-accuracy") + 1] = "0"
    with pytest.raises(_StopAfterPipelineBuilt):
        _run_main(monkeypatch, argv + ["--bar-from-test"])
    assert stubbed_heavy_parts["pipeline_kwargs"]["bar_split_name"] == "test"


@pytest.mark.parametrize("dataset", ["gsm_symbolic", "spider", "smiles"])
def test_bar_from_test_on_a_train_scoring_run_refuses_to_launch(
    monkeypatch, stubbed_heavy_parts, capsys, dataset
):
    with pytest.raises(SystemExit) as exc:
        _run_main(monkeypatch, _base_argv(dataset) + ["--bar-from-test"])
    assert exc.value.code == 2
    assert "Bar split mismatch" in capsys.readouterr().err
    assert "pipeline_kwargs" not in stubbed_heavy_parts


def test_chosen_bar_split_is_logged_at_launch(monkeypatch, stubbed_heavy_parts, caplog):
    caplog.set_level("INFO", logger="synthesis.run_synthesis")
    with pytest.raises(_StopAfterPipelineBuilt):
        _run_main(monkeypatch, _base_argv() + ["--bar-from-train"])
    assert any(
        "bar_split=train" in record.getMessage() for record in caplog.records
    )


def test_bar_split_is_not_hard_coded_anywhere_in_the_entry_point():
    import inspect

    source = inspect.getsource(run_synthesis)
    assert 'bar_split_name="train"' not in source
    assert "bar_split_name='train'" not in source
