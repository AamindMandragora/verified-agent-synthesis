"""Default benchmark plugin hooks shared by GSM-Symbolic and Spider eval_logic."""

from __future__ import annotations

from typing import Any


def force_open_span() -> bool:
    """Whether the runtime opens this benchmark's constrained span itself.

    True means generation starts with the output already equal to ``["<<"]``
    and the strategy already inside the span; the strategy closes it with
    ``CloseConstrainedSpan``, and the verified template closes a complete span
    the strategy left open. The Python runtime never writes the closer."""
    return False


def example_syntax_pass_from_segments(
    all_valid_syntax: bool,
    segments: list[tuple[str, bool]],
    aux: dict[str, Any] | None,
) -> bool:
    return bool(segments) and all_valid_syntax


def accuracy_applicable_always(aux: dict[str, Any] | None) -> bool:
    return True


def accuracy_upper_bound_with_remaining(
    num_correct: int,
    remaining: int,
    num_accuracy_examples: int,
    total_planned_examples: int,
) -> float:
    return (num_correct + remaining) / max(1, total_planned_examples)


def final_accuracy_denominator_all_examples(
    num_examples: int,
    num_accuracy_examples: int,
) -> int:
    return num_examples


def invalid_outputs_excluded_none(
    num_examples: int,
    num_accuracy_examples: int,
) -> int:
    return 0


def accuracy_definition_standard() -> str:
    return "correct_examples_over_all_examples"
