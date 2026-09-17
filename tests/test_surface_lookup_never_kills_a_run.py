"""Looking up the decoding surface must never be able to end a run.

The bug this pins
-----------------
`_force_open_span()` decides one thing: a sentence of wording in the author's
prompt. It says whether the runtime already opened the span, so the author
closes it instead of waiting for a `<<` that has already gone by.

That is a *hint*. Getting it wrong costs one weaker strategy. It must never
cost the run.

But the helper looks the benchmark up with `get_logic(dataset_name)`, and that
function does not return None for a benchmark it doesn't know -- it raises:

    raise ValueError(f"Unknown dataset: {dataset_name}")   # registry.py:16

So any run whose dataset is not in the registry now dies while building a
prompt. Its own docstring already promises otherwise -- "Falls back to False
(visible-delimiter surface, the historical default) if the benchmark doesn't
declare it" -- and it does handle the benchmark that exists but declares
nothing, via `getattr(logic, ..., None)`. It just never handled the benchmark
that isn't there at all.

This was found by running the three fixes together: on its own the surface fix
passed, and on its own the attempt-cap fix passed, but the cap's tests drive the
loop with a stand-in dataset name, and the combination crashed. A decorative
prompt hint had been given the power to abort synthesis.

The fallback direction is False on purpose: "open the span yourself", which is
how every benchmark behaved before this hook existed.
"""

from __future__ import annotations

import pytest

from synthesis.evaluate.feedback_loop import SynthesisPipeline


class _StandInEvaluator:
    """Just enough of an evaluator to be asked its dataset name."""

    def __init__(self, dataset_name: str) -> None:
        self.dataset_name = dataset_name


class _StandInLoop:
    """Carries only the attribute the helper actually reads.

    The helper is called as an unbound function so that no real
    SynthesisPipeline has to be constructed -- building one would load models
    and write directories, none of which this behaviour depends on.
    """

    def __init__(self, dataset_name: str) -> None:
        self.evaluator = _StandInEvaluator(dataset_name)


def _surface_for(dataset_name: str) -> bool:
    return SynthesisPipeline._force_open_span(_StandInLoop(dataset_name))


@pytest.mark.parametrize(
    "dataset_name",
    [
        "fake_dataset",       # what the attempt-cap tests use
        "not_a_benchmark",
        "",                   # never set
    ],
)
def test_an_unknown_benchmark_falls_back_instead_of_raising(dataset_name):
    try:
        surface = _surface_for(dataset_name)
    except Exception as exc:  # noqa: BLE001 -- the point is that nothing escapes
        pytest.fail(
            f"Looking up the decoding surface for an unregistered dataset "
            f"{dataset_name!r} raised {type(exc).__name__}: {exc}. This value "
            "only chooses a sentence in the author's prompt, so a missed guess "
            "should cost one weaker strategy -- not the whole run."
        )

    assert surface is False, (
        f"An unknown benchmark reported force_open_span={surface!r}. It must "
        "fall back to False -- open the span yourself -- which is how every "
        "benchmark behaved before this hook existed."
    )


def test_a_real_benchmark_still_reports_its_true_surface():
    """The fallback must not swallow the answer for benchmarks that do declare one."""
    assert _surface_for("spider") is True, (
        "spider runs with a runtime-opened span and must still say so. If a "
        "try/except now hides a real failure inside spider's own hook, this is "
        "where it shows up -- the fallback would quietly report the wrong "
        "surface for every benchmark, which is the bug the hook was added to fix."
    )
