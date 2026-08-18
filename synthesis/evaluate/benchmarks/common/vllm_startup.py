"""The two helpers the default evaluation path needs in order to start.

Why this module has to exist
----------------------------
`Evaluator._setup_environment` imports two names from this module whenever
the eval backend is vLLM, which is the default. Before this file existed,
that import raised `ModuleNotFoundError` on every run, and a broad
`except Exception` further up turned that missing file into
accuracy=0.0, syntax_rate=0.0, num_examples=0 -- a missing file dressed up
as "the model got everything wrong".

What they are for
------------------
The GPU box this runs on is shared with other jobs. Starting vLLM can fail
because of memory pressure in two opposite directions:
  - the requested memory share is too HIGH -> vLLM runs out of memory while
    loading the model
  - the requested memory share is too LOW -> there isn't enough room left
    for the KV cache (the space vLLM needs to hold conversation state while
    generating)

So the retry list below tries the originally requested setting first, then
backs off to lower settings (the more common fix), then tries higher
settings, in case the box actually has more room than expected.

`is_vllm_startup_memory_error` decides whether a given startup failure is
one of these memory problems (worth retrying on the list) or something else
entirely (a bug, a typo, a missing file) that no amount of retrying can fix.
Getting this too loose recreates the exact bug this file exists to prevent:
a real error would get retried pointlessly and then reported as "startup
failed" with the actual cause buried under retry noise. So this only says
yes for GPU-memory problems it can specifically recognize, never as a
default guess.
"""

from __future__ import annotations

# Values are gpu_memory_utilization fractions: the share of GPU memory vLLM
# is allowed to use. Must stay strictly between 0 and 1 -- vLLM rejects 0 or
# 1 outright.
_DEFAULT_REQUESTED_UTILIZATION = 0.85
_MIN_UTILIZATION = 0.05
_MAX_UTILIZATION = 0.97
_LOWER_STEP_DOWN = 0.15
_UPPER_STEP_UP = 0.1


def vllm_util_retry_candidates(
    requested: float | None, *, maximum: float | None = None
) -> list[float]:
    """Build the list of gpu_memory_utilization values to try, in order.

    The first entry is always exactly the requested value (or a sensible
    default if none was given) -- the caller treats candidates[0] as "the
    value that was actually asked for" when deciding what to print, so this
    must never be adjusted or clamped.

    After that, we back off to lower values first (freeing up memory is the
    more common fix for an out-of-memory startup failure), then step up to
    higher values (in case the real problem was too little room left for the
    KV cache).
    """
    first = requested if requested is not None else _DEFAULT_REQUESTED_UTILIZATION

    if maximum is not None:
        if not 0.0 < maximum < 1.0:
            raise ValueError("maximum gpu_memory_utilization must be between 0 and 1")
        if first > maximum:
            raise ValueError("requested gpu_memory_utilization exceeds its maximum")

    candidates: list[float] = [first]

    lower = first
    while True:
        lower = round(lower - _LOWER_STEP_DOWN, 2)
        if lower <= _MIN_UTILIZATION:
            break
        if lower not in candidates:
            candidates.append(lower)

    higher = first
    while True:
        higher = round(higher + _UPPER_STEP_UP, 2)
        if higher >= _MAX_UTILIZATION:
            break
        if maximum is not None and higher > maximum:
            break
        if higher not in candidates:
            candidates.append(higher)

    if len(candidates) < 2:
        # The requested value sat so close to an edge that backing off or
        # stepping up produced nothing usable. Fall back to a fixed pair of
        # safe values so the caller still gets a real ladder to retry on.
        for fallback in (_MIN_UTILIZATION + 0.1, _MAX_UTILIZATION - 0.1):
            fallback = round(fallback, 2)
            if (
                fallback not in candidates
                and 0.0 < fallback < 1.0
                and (maximum is None or fallback <= maximum)
            ):
                candidates.append(fallback)

    print(f"[vllm] gpu_memory_utilization retry ladder: {candidates}")
    return candidates


# Substrings that show up in real GPU-memory-pressure startup failures.
# Matched case-insensitively against str(exc).
_MEMORY_PRESSURE_MESSAGE_MARKERS = (
    "cuda out of memory",
    "no available memory for the cache blocks",
    "no free memory to run vllm",
    "kv cache",
    # vLLM's own wording when the requested gpu_memory_utilization is larger
    # than what is actually free -- the exact case the retry ladder exists to
    # back off from. On a shared GPU this is the message you get when another
    # user already holds the card, so missing it meant the run gave up instead
    # of retrying at a smaller share.
    "desired gpu memory utilization",
    "free memory on device",
)

# Deliberately NOT a marker: "Engine core initialization failed". That is
# vLLM's generic wrapper around *any* engine startup failure -- a missing
# module or a bad argument produces it too. Treating it as memory pressure
# would send real errors around the utilization ladder and then report them
# as a generic startup failure, hiding the actual cause.


def is_vllm_startup_memory_error(exc: BaseException) -> bool:
    """Is this startup failure GPU-memory pressure, and therefore worth
    retrying at a different gpu_memory_utilization?

    Deliberately narrow: this only says yes for the specific exception types
    and message text that real memory-pressure failures produce. Everything
    else -- a typo, a missing module, a bad argument -- says no, so the real
    cause surfaces immediately instead of being retried on a ladder that
    cannot fix it and then reported as generic "startup failed".
    """
    # KeyboardInterrupt is not an Exception subclass (it is a BaseException),
    # so guard against non-Exception input before doing anything else with it.
    if not isinstance(exc, Exception):
        return False

    if isinstance(exc, MemoryError):
        return True

    if isinstance(exc, (RuntimeError, ValueError)):
        message = str(exc).lower()
        if any(marker in message for marker in _MEMORY_PRESSURE_MESSAGE_MARKERS):
            return True
        print(
            f"[vllm] {type(exc).__name__} at startup did not match known "
            f"memory-pressure text, treating as a real failure: {exc}"
        )
        return False

    print(
        f"[vllm] {type(exc).__name__} at startup is not a memory-pressure "
        f"error type, treating as a real failure: {exc}"
    )
    return False
