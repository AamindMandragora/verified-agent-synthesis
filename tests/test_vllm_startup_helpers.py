"""The two helpers the default evaluation path needs in order to start.

Why this module has to exist
----------------------------
`Evaluator._setup_environment` (evaluator.py:1940) imports two names:

    from synthesis.evaluate.benchmarks.common.vllm_startup import (
        is_vllm_startup_memory_error,
        vllm_util_retry_candidates,
    )

That module does not exist and never has, in any commit on any branch. The
import runs whenever backend == "vllm", which is the DEFAULT
(run_synthesis.py:162), so the standard evaluation path cannot start. The
resulting ModuleNotFoundError is swallowed by a broad `except Exception`
upstream and reported as accuracy=0.0, syntax_rate=0.0, num_examples=0 -- a
missing file wearing the costume of a model that got everything wrong.

What they are for
-----------------
The GPU box is shared. vLLM startup competes with sibling jobs for memory, so
`_setup_environment` walks a ladder of gpu_memory_utilization values, retrying
on memory pressure and giving up on anything else.

Both failure directions are real, which is why the ladder is not simply
"back off":
  - utilization too HIGH -> out of memory while allocating
  - utilization too LOW  -> KV cache too small to hold the model

Contracts read off the real call site (evaluator.py:1980-2024):
  - element 0 of the ladder is the requested value; it is tried first
  - the loop compares `util > requested` to print "higher"/"lower", so values
    on both sides of the requested one are expected
  - `is_vllm_startup_memory_error(exc)` decides retry-vs-reraise at line 2009
"""

from __future__ import annotations

import pytest


MODULE = "synthesis.evaluate.benchmarks.common.vllm_startup"


def _helpers():
    import importlib

    module = importlib.import_module(MODULE)
    return module.vllm_util_retry_candidates, module.is_vllm_startup_memory_error


def test_the_module_the_default_path_imports_exists():
    import importlib

    try:
        importlib.import_module(MODULE)
    except ModuleNotFoundError as exc:
        pytest.fail(
            f"{MODULE} is missing, so Evaluator._setup_environment raises "
            f"ModuleNotFoundError on the default --eval-backend=vllm path "
            f"before it ever loads a model. Upstream that becomes "
            f"accuracy=0.0 with num_examples=0. Original error: {exc}"
        )


def test_it_exports_exactly_the_two_names_the_evaluator_imports():
    import importlib

    module = importlib.import_module(MODULE)
    for name in ("is_vllm_startup_memory_error", "vllm_util_retry_candidates"):
        assert hasattr(module, name), (
            f"evaluator.py:1940 imports {name!r} from this module; without it "
            "the import still fails and the fake-zero path is unchanged."
        )


@pytest.mark.parametrize("requested", [0.9, 0.85, 0.5, 0.3])
def test_the_requested_value_is_tried_first(requested):
    """Retrying must not silently change the setting the user asked for."""
    candidates, _ = _helpers()

    ladder = candidates(requested)

    assert ladder, "The ladder must never be empty; the caller iterates it."
    assert ladder[0] == pytest.approx(requested), (
        f"First attempt used {ladder[0]}, not the requested {requested}. "
        "evaluator.py:1991 treats candidates[0] as the requested value when "
        "deciding what to print, so a different first entry misreports what "
        "was actually tried."
    )


@pytest.mark.parametrize("requested", [0.9, 0.85, 0.5])
def test_every_candidate_is_a_usable_fraction(requested):
    candidates, _ = _helpers()

    ladder = candidates(requested)

    assert all(0.0 < value < 1.0 for value in ladder), (
        f"gpu_memory_utilization must be a fraction strictly between 0 and 1; "
        f"vLLM rejects anything else, which would turn a retry into a crash. "
        f"Got: {ladder}"
    )
    assert len(ladder) == len(set(ladder)), (
        f"Duplicate values waste a full model-load attempt each (~24s) on a "
        f"setting already known to fail. Got: {ladder}"
    )
    assert len(ladder) > 1, "A ladder with one rung is not a retry ladder."


def test_retry_ladder_respects_an_explicit_upper_limit():
    candidates, _ = _helpers()

    ladder = candidates(0.35, maximum=0.35)

    assert ladder[0] == pytest.approx(0.35)
    assert all(value <= 0.35 for value in ladder)
    assert any(value < 0.35 for value in ladder)


def test_a_missing_setting_still_produces_a_usable_ladder():
    """`self.vllm_gpu_memory_utilization` can be None; the caller does not
    guard for it, so the ladder must."""
    candidates, _ = _helpers()

    ladder = candidates(None)

    assert ladder and all(0.0 < v < 1.0 for v in ladder), (
        f"None must yield a sensible default ladder, not crash or return "
        f"nothing. Got: {ladder}"
    )


def test_out_of_memory_errors_are_recognised_so_the_retry_can_happen():
    _, is_memory_error = _helpers()

    memory_failures = [
        RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"),
        RuntimeError("No available memory for the cache blocks"),
        ValueError(
            "The model's max seq len is larger than the maximum number of "
            "tokens that can be stored in the KV cache"
        ),
        MemoryError("Out of memory"),
    ]

    for exc in memory_failures:
        assert is_memory_error(exc), (
            f"{type(exc).__name__}({exc}) was not recognised as GPU memory "
            "pressure, so evaluator.py:2010 re-raises instead of retrying at a "
            "lower utilization. On a shared box that turns a lost race for "
            "memory into a failed run."
        )


def test_unrelated_errors_are_not_disguised_as_memory_pressure():
    """The dangerous direction, and the reason this test exists.

    If this returned True for everything, a genuine bug -- a typo, a bad
    argument, a missing file -- would be retried as though it were memory
    pressure, then reported as a startup failure. That is precisely the bug
    class this whole sweep is about: a real fault wearing the costume of an
    expected, benign condition.
    """
    _, is_memory_error = _helpers()

    unrelated_failures = [
        AttributeError("'NoneType' object has no attribute 'generate'"),
        ModuleNotFoundError("No module named 'synthesis.scripts.eval_worker_pool'"),
        TypeError("__init__() got an unexpected keyword argument 'dtype'"),
        FileNotFoundError("No such file or directory: 'spider.sqlite'"),
        KeyboardInterrupt(),
    ]

    for exc in unrelated_failures:
        assert not is_memory_error(exc), (
            f"{type(exc).__name__}({exc}) was classified as GPU memory "
            "pressure. It will be retried on a ladder that cannot fix it, and "
            "the real cause will be buried under retry noise."
        )
