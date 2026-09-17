"""The constrained-span sampling temperature belongs to the benchmark.

SMILES is scored on how many DISTINCT valid molecules come out. At
temperature 0.0 the sampler is an argmax, so every example decodes to the
same molecule and the score collapses to about zero. SMILES therefore needs
a positive temperature to be scored at all.

This used to be an environment variable, `CSD_CONSTRAINED_TEMPERATURE`, that
the queue scripts exported for SMILES rows. Anything launched another way --
`python -m synthesis.scripts.reevaluate_compiled_csd ... --dataset smiles`,
say -- silently got argmax and a garbage number.

Now it is a benchmark fact, like `force_open_span()`: a default of 0.0 in
`benchmark_defaults`, an override of 0.7 in the SMILES eval logic, handed to
the language model by the generation path. The environment is not consulted.
"""

from __future__ import annotations

import pytest

from tests.test_forced_span_contract import (
    Dafny,
    Tokenizer,
    _Parser,
    _strategy_returning,
)


# --------------------------------------------------------------------------
# (a) What each benchmark reports
# --------------------------------------------------------------------------


def test_the_shared_default_is_argmax():
    from synthesis.evaluate.benchmarks.common import benchmark_defaults as defaults

    assert defaults.constrained_temperature() == 0.0


@pytest.mark.parametrize("dataset", ["gsm_symbolic", "spider"])
def test_gsm_and_spider_stay_exactly_argmax(dataset):
    from synthesis.evaluate.benchmarks.registry import get_logic

    assert get_logic(dataset).constrained_temperature() == 0.0


def test_smiles_samples_within_the_span():
    from synthesis.evaluate.benchmarks.registry import get_logic

    assert get_logic("smiles").constrained_temperature() == 0.7


# --------------------------------------------------------------------------
# (b) What a real generation run hands the language model
# --------------------------------------------------------------------------


def _lm_after_a_run(dataset, monkeypatch):
    """Run one generation through `dataset`'s runner; return the model it used."""
    from synthesis.evaluate.benchmarks.common.model_utils import _TensorizedLMBase
    from synthesis.evaluate.benchmarks.registry import get_logic

    # Prove the environment is ignored: set it to a value nobody wants.
    monkeypatch.setenv("CSD_CONSTRAINED_TEMPERATURE", "0.123")

    generated_csd, _ = _strategy_returning(["<<", "ok"], False, [])
    lm = _TensorizedLMBase(Dafny(), Tokenizer(), ["ok"], [1])
    env = {
        "_dafny": Dafny,
        "GeneratedCSD": generated_csd,
        "lm": lm,
        "parser": _Parser(),
        "model_name": "Qwen/Qwen2.5-7B-Instruct",
    }
    get_logic(dataset).get_generation_runner()(
        env=env, prompt_text="question", max_steps=8, grammar_file=None
    )
    return lm


def test_a_smiles_run_configures_the_model_to_sample(monkeypatch):
    assert _lm_after_a_run("smiles", monkeypatch)._constrained_temperature == 0.7


@pytest.mark.parametrize("dataset", ["gsm_symbolic", "spider"])
def test_a_gsm_or_spider_run_configures_the_model_for_argmax(dataset, monkeypatch):
    assert _lm_after_a_run(dataset, monkeypatch)._constrained_temperature == 0.0


def test_a_fresh_model_is_argmax_whatever_the_environment_says(monkeypatch):
    """Construction never reads the environment."""
    from synthesis.evaluate.benchmarks.common.model_utils import _TensorizedLMBase

    monkeypatch.setenv("CSD_CONSTRAINED_TEMPERATURE", "0.9")
    lm = _TensorizedLMBase(Dafny(), Tokenizer(), ["ok"], [1])

    assert lm._constrained_temperature == 0.0


# --------------------------------------------------------------------------
# (c) The environment variable is gone
# --------------------------------------------------------------------------


def test_no_module_under_synthesis_reads_the_environment_variable():
    import subprocess
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    hits = subprocess.run(
        [
            "grep",
            "-rn",
            "--include=*.py",
            "CSD_CONSTRAINED_TEMPERATURE",
            str(repo / "synthesis"),
        ],
        capture_output=True,
        text=True,
    ).stdout.strip()

    assert hits == "", f"the env var is still read under synthesis/:\n{hits}"


def test_the_cold_queue_no_longer_exports_it():
    import scripts.runtime.run_cold_synthesis_queue as cold

    job = {
        "dataset": "smiles",
        "cell_id": "smiles-acrylates-qwen25-1p5b",
        "output_name": "coldq_smiles_test",
        "gpu_mem_util": 0.4,
        "smiles_class": "acrylates",
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
    }
    from pathlib import Path

    env = cold.synthesis_environment(job, (3,), {"PATH": "/bin"}, Path("/repo"))
    assert "CSD_CONSTRAINED_TEMPERATURE" not in env

    clean = cold.author_free_environment({"PATH": "/bin"}, 3)
    assert "CSD_CONSTRAINED_TEMPERATURE" not in clean


def test_the_paper_and_table_queues_no_longer_export_it():
    import subprocess
    from pathlib import Path

    runtime = Path(__file__).resolve().parents[1] / "scripts" / "runtime"
    hits = subprocess.run(
        [
            "grep",
            "-rn",
            "CSD_CONSTRAINED_TEMPERATURE",
            str(runtime / "run_paper_baseline_queue.py"),
            str(runtime / "run_table5_8_queue.py"),
            str(runtime / "zero_acc_babysitter" / "smoke.py"),
        ],
        capture_output=True,
        text=True,
    ).stdout.strip()

    assert hits == "", f"queue scripts still set the env var:\n{hits}"
