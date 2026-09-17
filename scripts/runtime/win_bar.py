"""The one place that turns a baseline score into the accuracy bar synthesis must clear."""

from __future__ import annotations

import math
from dataclasses import dataclass

MARGIN_POINTS = 10


@dataclass(frozen=True)
class AccuracyBar:
    min_accuracy: float
    policy: str


def accuracy_bar(baseline_correct: int, total: int) -> AccuracyBar:
    """Best baseline plus 10 absolute points, counted in whole examples.

    Near the ceiling 10 points do not fit, so the bar is halfway from the
    baseline to a perfect score instead (rounded up to a whole example).
    """
    if total <= 0 or not 0 <= baseline_correct <= total:
        raise ValueError(f"bad baseline counts: {baseline_correct}/{total}")
    margin = math.ceil(total * MARGIN_POINTS / 100)
    halfway = math.ceil((total - baseline_correct) / 2)
    if baseline_correct == total:
        return AccuracyBar(1.0, "perfect_baseline_must_match")
    if margin <= halfway:
        return AccuracyBar((baseline_correct + margin) / total, "baseline_plus_10_points")
    return AccuracyBar((baseline_correct + halfway) / total, "halfway_to_perfect")
