"""Typed data for builder #8: EvaluationResult.get_feedback_summary.

Every field here is a value the surrounding evaluator.py method already
computes (accuracy/syntax percentages, the sub-summary lists from the
existing `_summarize_*` helpers, the cluster block from
`render_cluster_block`). This model only carries that data to the
`feedback/feedback_summary.j2` template; it does not compute anything new.

Percentages and other formatted numbers are pre-formatted into strings by
the caller (matching the exact Python format spec the original method used,
e.g. `:.1%` / `:.2f` / `:.3f`) so the template only has to place them — this
keeps the refactor byte-identical without re-deriving float formatting rules
in Jinja.
"""
from typing import List, Optional

from synthesis.prompt_rendering.base import PromptModel


class SmilesTrialFields(PromptModel):
    """SMILES Quality Metrics block (from aux_metrics["smiles_paper_trial"])."""

    validity_pct: str
    membership_pct: str
    diversity_display: str
    retro_score_display: str
    samples_to_target_display: str
    unique_valid_display: str


class AntiDegeneracyFields(PromptModel):
    """Anti-Degeneracy Diagnostics block (from aux_metrics["anti_degeneracy"])."""

    churn_ratio_display: str
    tiny_span_pct: str
    max_steps_hit_pct: str
    penalty_pct: str
    adjusted_membership_pct: str


class EarlyStopMetricsFields(PromptModel):
    """Early Stop block (from aux_metrics["early_stop"]).

    Distinct from the top-level `early_stopped` / `early_stop_reason` fields,
    which describe the evaluator's own early-stop, not this aux_metrics dict.
    """

    reason: str
    max_possible_pct: str
    target_pct: str
    evaluated_display: str


class FailureModeEntry(PromptModel):
    """One row of the Primary Failure Modes list."""

    mode: str
    count: int
    detail: str


class FeedbackSummaryModel(PromptModel):
    """All data needed to render the full feedback summary."""

    eval_count_label: str
    accuracy_pct: str
    num_correct: int
    accuracy_denominator_display: int
    show_contains_delimiters: bool
    contains_delimiters_yesno: str
    syntax_pct: str
    total_time_str: str
    max_sample_time_str: str
    early_stopped: bool
    early_stop_reason: Optional[str]
    accuracy_definition_display: Optional[str]
    invalid_outputs_excluded_from_accuracy: int
    task_guidance: List[str]

    smiles_trial: Optional[SmilesTrialFields]
    anti_degeneracy: Optional[AntiDegeneracyFields]
    early_stop_metrics: Optional[EarlyStopMetricsFields]

    failure_modes: List[FailureModeEntry]
    output_run_summary: List[str]
    aggregate_failure_stats_lines: List[str]
    model_call_cost_lines: List[str]
    diagnostic_metrics: List[str]
    show_structural_metrics: bool
    structural_metrics: List[str]

    cluster_block: str
