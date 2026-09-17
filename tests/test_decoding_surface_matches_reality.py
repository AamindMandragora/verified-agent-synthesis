"""What the author is TOLD about the span contract must match what happens.

The bug this guards
-------------------
A strategy reaches the constrained region one of two ways:

  - it opens the span itself with `OpenConstrainedSpan`, which puts a literal
    `<<` into the output; or
  - the runtime opened the span before the strategy ran, so the output already
    starts with `<<` and the strategy is already inside it.

Spider and SMILES are the second kind. Authors kept writing strategies whose
only route into constrained mode was `if next == "<<"`, which never fires when
the `<<` is already there, so nothing was ever constrained.

The fix tells the author which of the two it is on. That only helps if the
claim is TRUE, and two separate pieces of code decide it:

  - `force_open_span()`        -> what the author's prompt is told
  - `get_generation_runner()`  -> what evaluation actually does

Nothing but a comment keeps them in step, and a comment does not fail a build.
If they drift, the author is confidently told the wrong thing and writes a
strategy that cannot work -- the same silent failure as before, wearing a fix.

So this file does not check the prompt wording (covered elsewhere). It checks
the claim against the behaviour.
"""

from __future__ import annotations

import importlib

import pytest


FORCED_SPAN_BENCHMARKS = {
    "spider": "synthesis.evaluate.benchmarks.sql_spider.generation",
    "smiles": "synthesis.evaluate.benchmarks.smiles.generation",
    "gsm_symbolic": "synthesis.evaluate.benchmarks.gsm_symbolic.generation",
}


def _surface_actually_used(monkeypatch, dataset: str) -> bool:
    """Run the real generation runner and observe what it asks for.

    The runner imports `run_crane_csd` when it is called, not at module load,
    so replacing it on the generation module beforehand intercepts the call
    without loading a model.
    """
    from synthesis.evaluate.benchmarks.registry import get_logic

    generation = importlib.import_module(FORCED_SPAN_BENCHMARKS[dataset])
    seen: dict = {}

    def _capture(*args, **kwargs):
        seen.update(kwargs)
        return ("", 0, 0.0, [], [])

    monkeypatch.setattr(generation, "run_crane_csd", _capture, raising=False)
    monkeypatch.setattr(generation, "_run_crane_csd", _capture, raising=False)

    runner = get_logic(dataset).get_generation_runner()
    if runner is generation.run_crane_csd or runner is _capture:
        # GSM hands back the bare function; calling it would run the real thing.
        return False
    runner()
    return bool(seen.get("force_open_span", False))


@pytest.mark.parametrize(
    ("dataset", "expected"),
    [("spider", True), ("smiles", True), ("gsm_symbolic", False)],
)
def test_the_author_is_told_the_surface_that_is_actually_used(
    monkeypatch, dataset, expected
):
    from synthesis.evaluate.benchmarks.registry import get_logic

    claimed = bool(get_logic(dataset).force_open_span())
    actual = _surface_actually_used(monkeypatch, dataset)

    assert claimed == actual, (
        f"{dataset}: the author's prompt is told force_open_span={claimed}, but "
        f"evaluation actually runs with force_open_span={actual}. The author "
        "will write a strategy for the wrong starting point and silently "
        "constrain nothing."
    )
    assert actual is expected, (
        f"{dataset} was expected to run with force_open_span={expected}, but "
        f"produced {actual}. If that changed on purpose, update this test and "
        "the author prompt together -- they must not drift apart."
    )


@pytest.mark.parametrize("dataset", sorted(FORCED_SPAN_BENCHMARKS))
def test_the_benchmark_registry_exposes_the_surface_to_the_feedback_loop(dataset):
    """The feedback loop finds this hook by name through the registry.

    It looks the benchmark up with `get_logic(dataset_name)` and then reads
    `force_open_span` off it with getattr, falling back to False when absent.
    A rename would therefore not raise -- it would quietly report "you must
    open the span yourself" for every benchmark, which is the pre-fix bug.
    """
    from synthesis.evaluate.benchmarks.registry import get_logic

    hook = getattr(get_logic(dataset), "force_open_span", None)

    assert hook is not None, (
        f"{dataset} no longer exposes force_open_span(). The feedback loop's "
        "getattr lookup will silently fall back to False and tell every author "
        "it has to open the span itself."
    )
    assert isinstance(hook(), bool)
