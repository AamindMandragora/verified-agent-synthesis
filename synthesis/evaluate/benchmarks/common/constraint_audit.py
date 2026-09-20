"""Track decoding steps where the grammar constraint was NOT applied.

Why this exists
----------------
`SyncodeLogitsProcessor` is supposed to force every generated token to match
a grammar. But it has two paths where it gives up and lets the model pick any
token at all: a parse error, or a partial code with no acceptable next token.
When that happens, the step was decoded unconstrained -- the grammar had no
effect on it. That does not necessarily make the output wrong (the model can
still write something valid on its own), which is exactly why it is dangerous:
a run can silently drop the constraint on some steps and still show a good
validity score, and that score gets read as evidence that constrained
decoding works.

This module is the count that used to not exist. Before this, the decoder had
only `self.parse_failed`, a bool that could not tell one skipped step apart
from ten thousand, and a second failure path that did not even set that bool.

Kept pure and dependency-free (no torch/transformers) so it can be imported
and unit-tested in an environment where the heavy decoding stack is not
installed.
"""

from __future__ import annotations

MAX_DETAIL_SAMPLES_PER_REASON = 5


class ConstraintAudit:
    """Counts decoding steps where no grammar constraint was applied.

    Call `record_unconstrained_step` every time the decoder skips masking the
    scores for a step. Read `total_unconstrained_steps`, `counts_by_reason`,
    and `was_fully_constrained` afterward to find out whether the run's
    output can actually be trusted as "grammar-constrained."
    """

    def __init__(self) -> None:
        self._counts_by_reason: dict[str, int] = {}
        self._detail_samples_by_reason: dict[str, list[str]] = {}
        self._total_unconstrained_steps = 0
        self._total_masked_steps = 0

    def record_masked_step(self) -> None:
        """Record one decoding step where the grammar mask WAS applied to the scores.

        Lets a caller check "masked steps == generated tokens" for a decoder that is
        supposed to mask every token, instead of inferring it from the absence of
        unconstrained steps.
        """
        self._total_masked_steps += 1

    def record_unconstrained_step(self, reason: str, detail: str) -> None:
        """Record one decoding step that skipped the grammar constraint.

        `reason` is a short label for why (e.g. "parse_error",
        "no_valid_tokens"). `detail` is extra context for debugging; only the
        first few samples per reason are kept, so a run with tens of
        thousands of unconstrained steps does not fill memory with strings.
        """
        self._total_unconstrained_steps += 1
        self._counts_by_reason[reason] = self._counts_by_reason.get(reason, 0) + 1

        samples = self._detail_samples_by_reason.setdefault(reason, [])
        if len(samples) < MAX_DETAIL_SAMPLES_PER_REASON:
            samples.append(detail)

    @property
    def total_masked_steps(self) -> int:
        return self._total_masked_steps

    @property
    def total_unconstrained_steps(self) -> int:
        return self._total_unconstrained_steps

    @property
    def counts_by_reason(self) -> dict[str, int]:
        # Return a copy so callers can't mutate our internal state.
        return dict(self._counts_by_reason)

    @property
    def was_fully_constrained(self) -> bool:
        """True only if every decoding step had the grammar applied."""
        return self._total_unconstrained_steps == 0

    def summary(self) -> str:
        """A human-readable statement of the damage, or "" if there is none.

        Worded so someone reading run output understands their result is
        compromised -- not that a minor warning fired.
        """
        if self.was_fully_constrained:
            return ""

        reason_breakdown = ", ".join(
            f"{reason}={count}" for reason, count in sorted(self._counts_by_reason.items())
        )
        return (
            f"UNRELIABLE: {self._total_unconstrained_steps} decoding step(s) were NOT "
            f"grammar-constrained ({reason_breakdown}). Any validity rate from this "
            "run is not evidence that constrained decoding worked -- some tokens "
            "were chosen with no constraint applied at all."
        )
