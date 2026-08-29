"""
Evaluation module for synthesis feedback loop.

Provides quick evaluation of synthesized CSD strategies on dataset samples
to enable feedback-driven refinement based on actual performance metrics.
"""

from __future__ import annotations

import importlib
import json
import os
import re
from collections import Counter
from synthesis.safe_logging import display_text, safe_logging_enabled

# SQL keyword set for failure-preview anonymization. Identifiers outside this
# set are replaced with <id>; numbers with <num>; string literals with <str>.
# Punctuation and the keywords themselves pass through verbatim. The result
# preserves structural shape (SELECT-FROM-WHERE etc.) without leaking
# synthesis-sample schema/identifier text.
_SQL_KEYWORDS = {
    "SELECT","FROM","WHERE","JOIN","INNER","LEFT","RIGHT","FULL","OUTER","CROSS","ON","AS",
    "GROUP","BY","HAVING","ORDER","ASC","DESC","LIMIT","OFFSET","UNION","INTERSECT","EXCEPT",
    "ALL","DISTINCT","AND","OR","NOT","IN","IS","NULL","LIKE","BETWEEN","EXISTS",
    "COUNT","SUM","AVG","MIN","MAX","CASE","WHEN","THEN","ELSE","END","CAST","INT","REAL","TEXT",
    "TRUE","FALSE",
}
import re as _re

def _anonymize_sql_preview(s: str) -> str:
    """Replace identifiers/numbers/strings with placeholders, keep keywords + punctuation."""
    if not s:
        return s
    out_parts = []
    i = 0
    while i < len(s):
        ch = s[i]
        if ch.isspace():
            out_parts.append(ch)
            i += 1
            continue
        if ch == "'":
            # consume string literal
            j = s.find("'", i + 1)
            j = j + 1 if j >= 0 else len(s)
            out_parts.append("<str>")
            i = j
            continue
        if ch.isdigit() or (ch == "." and i + 1 < len(s) and s[i + 1].isdigit()):
            m = _re.match(r"\d+(?:\.\d+)?", s[i:])
            out_parts.append("<num>")
            i += m.end() if m else 1
            continue
        if ch.isalpha() or ch == "_":
            m = _re.match(r"[A-Za-z_][A-Za-z0-9_]*", s[i:])
            tok = m.group(0) if m else ch
            out_parts.append(tok if tok.upper() in _SQL_KEYWORDS else "<id>")
            i += m.end() if m else 1
            continue
        out_parts.append(ch)
        i += 1
    return "".join(out_parts)

import time
import signal
import inspect
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# region agent log
_AGENT_DEBUG_SESSION = "d7d0bd"
_AGENT_DEBUG_FOCAL_PATH = "/tmp/debug-d7d0bd-focal.ndjson"
_AGENT_DEBUG_SMILES_CAP = 5
_agent_debug_smiles_logged = 0


def _agent_debug_log(
    hypothesis_id: str,
    location: str,
    message: str,
    data: Optional[Dict[str, Any]] = None,
    run_id: str = "pre-fix",
) -> None:
    """Append one NDJSON debug line for session d7d0bd (focal-local file)."""
    payload = {
        "sessionId": _AGENT_DEBUG_SESSION,
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data or {},
        "timestamp": int(time.time() * 1000),
    }
    try:
        with open(_AGENT_DEBUG_FOCAL_PATH, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, default=str) + "\n")
    except Exception:
        pass
# endregion

try:
    from synthesis.failure_taxonomy import render_cluster_block
except ImportError:
    from failure_taxonomy import render_cluster_block

try:
    from synthesis.prompt_rendering import render as _render_prompt
    from synthesis.prompt_rendering.models.feedback import (
        AntiDegeneracyFields,
        EarlyStopMetricsFields,
        FailureModeEntry,
        FeedbackSummaryModel,
        SmilesTrialFields,
    )
    from synthesis.prompt_rendering.models.feedback_loop import HintLinesModel
    from synthesis.prompt_rendering.models.evaluator import (
        ModeExampleEntry,
        ModeExamplesModel,
    )
except ImportError:
    from prompt_rendering import render as _render_prompt
    from prompt_rendering.models.feedback import (
        AntiDegeneracyFields,
        EarlyStopMetricsFields,
        FailureModeEntry,
        FeedbackSummaryModel,
        SmilesTrialFields,
    )
    from prompt_rendering.models.feedback_loop import HintLinesModel
    from prompt_rendering.models.evaluator import (
        ModeExampleEntry,
        ModeExamplesModel,
    )

from synthesis.evaluate.benchmarks.smiles.rolling_prompt import (
    apply_suffix as _apply_smiles_rolling_suffix,
    update_suffix as _update_smiles_rolling_suffix,
)
from synthesis.evaluate.benchmarks.common.ungradable import UngradableExample
from synthesis.evaluate.metrics import choose_denominator_basis
from synthesis.run_constants import VLLM_ENFORCE_EAGER


class PerExampleTimeout(Exception):
    """Raised when a single evaluation example exceeds its runtime budget."""


# Datasets whose per-iteration eval can run on the persistent data-parallel
# worker pool (synthesis/scripts/eval_worker_pool.py). GSM and Spider examples
# carry no state between examples, so splitting them across workers and
# merging in original order reproduces the sequential result exactly. SMILES
# is excluded: its rolling-prompt path (_apply_smiles_rolling_suffix /
# _update_smiles_rolling_suffix below) threads state from each example into
# the next, which a data-parallel split cannot reproduce, and SMILES also has
# its own `should_stop_collected` early stop — SMILES keeps the untouched
# single-process path.
POOLABLE_DATASETS = {"gsm_symbolic", "spider"}
_POOL_LOG = "[sharded-eval]"


def _resolve_eval_pool_loader():
    """Return the parallel eval pool's entry point, or None if it is
    unavailable (meaning: use the sequential path).

    The pool is a speed optimisation, not a correctness requirement -- a
    missing or broken pool module must never be allowed to blow up and get
    laundered into a fake 0% score by the broad `except Exception` around the
    eval method. So this catches everything and reports None instead.
    """
    try:
        pool_module = importlib.import_module("synthesis.scripts.eval_worker_pool")
        return pool_module.get_synthesis_eval_pool
    except Exception as e:
        print(
            f"{_POOL_LOG} eval worker pool unavailable ({type(e).__name__}: {e}); "
            "falling back to the slower sequential eval path.",
            flush=True,
        )
        return None


# Pathological-strategy guard: stop after this many timed-out examples. Also
# used post-hoc by _posthoc_early_stop to replay the same decision over a
# pool-merged, full (non-early-stopped) batch of results.
_MAX_TIMEOUTS_PATHOLOGICAL = 10

# Marker prefix for EvaluationResult.early_stop_reason when the per-attempt
# wall-clock cap (SynthesisPipeline.max_attempt_seconds) fires. feedback_loop.py
# checks for this exact prefix to tell "this attempt ran out of its overall
# time budget" apart from every other early-stop reason (which are ordinary
# threshold misses, not timeouts, and must not be recorded as one).
ATTEMPT_DEADLINE_EARLY_STOP_REASON = "attempt wall-clock budget exceeded"


_MAX_GSM_SCORING_EXPRESSION_CHARS = 512
_MAX_GSM_SCORING_EXPRESSION_TOKENS = 160
_MAX_GSM_SCORING_EXPRESSION_OPERATORS = 80
_MAX_GSM_SCORING_DIGIT_RUN = 64


def _print_realtime_completion(
    example_number: int, total_examples: int, completion: str
) -> None:
    """Write one generated completion to stdout with unambiguous boundaries."""
    prefix = f"  [EVAL]   Sample {example_number}/{total_examples} completion"
    if safe_logging_enabled():
        print(display_text(prefix, completion), flush=True)
        return
    print(f"{prefix} begin", flush=True)
    print(completion, flush=True)
    print(f"{prefix} end", flush=True)


_STRATEGY_SAMPLE_EVIDENCE_FIELDS = (
    "strategy_output_relation",
    "strategy_mutation",
    "strategy_removed_sampled_token_ids",
)


def _strategy_sample_evidence_fields(evidence: Any) -> dict[str, Any]:
    """Copy strategy-origin fields from token evidence into the sample row."""
    if not isinstance(evidence, dict):
        return {}
    return {
        key: evidence[key]
        for key in _STRATEGY_SAMPLE_EVIDENCE_FIELDS
        if key in evidence
    }


def _is_pathological_gsm_scoring_expression(expression: str) -> bool:
    """Return whether a generated GSM expression is too large for safe scoring.

    This guard is intentionally far above the current GSM split's gold answers:
    a 2026-06-29 audit found max gold length 119 chars and max operator count
    22 across the train/eval split. Outputs above this guard are treated as
    wrong before entering native parser/prover code that Python alarms may not
    interrupt reliably.
    """

    text = str(expression or "").strip()
    if len(text) > _MAX_GSM_SCORING_EXPRESSION_CHARS:
        return True
    if len(text.split()) > _MAX_GSM_SCORING_EXPRESSION_TOKENS:
        return True
    if len(re.findall(r"[+\-*/%()]", text)) > _MAX_GSM_SCORING_EXPRESSION_OPERATORS:
        return True
    if re.search(r"\d{" + str(_MAX_GSM_SCORING_DIGIT_RUN) + r",}", text):
        return True
    return False


# Numbers, names, and any other single non-space character. Used when no
# tokenizer is available; unlike a whitespace split it still grows with the
# length of the text.
_SPAN_UNIT_PATTERN = re.compile(r"\d+|[A-Za-z_]\w*|\S")


def span_token_length(tokenizer: Any, text: str) -> int:
    """Return how many tokens a generated span is.

    Measured with the eval model's own tokenizer when one is available. This
    used to be a whitespace split, which is wrong for every benchmark we run:
    GSM answers are arithmetic and SMILES answers are molecule strings, and
    neither contains spaces, so a 1555-character runaway span counted as one
    "token". That number is one of the axes `synthesis/failure_taxonomy.py`
    clusters on, so runaway spans were filed under "spans are too tiny" and the
    author model was told to lengthen spans that were already spiralling.
    """
    text = str(text or "").strip()
    if not text:
        return 0
    if tokenizer is not None:
        try:
            return len(tokenizer.encode(text, add_special_tokens=False))
        except Exception:
            pass  # fall through to the pattern below
    return len(_SPAN_UNIT_PATTERN.findall(text))


class _PerExampleTimer:
    """Unix wall-clock timer for interrupting a single long-running example."""

    def __init__(self, seconds: Optional[float]):
        self.seconds = seconds
        self._old_handler = None
        self._old_timer = None

    def __enter__(self):
        if self.seconds is None:
            return self
        if self.seconds <= 0:
            raise PerExampleTimeout(f"Example exceeded {self.seconds:.2f}s runtime budget")

        def _raise_timeout(signum, frame):
            raise PerExampleTimeout(f"Example exceeded {self.seconds:.2f}s runtime budget")

        self._old_handler = signal.getsignal(signal.SIGALRM)
        self._old_timer = signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, _raise_timeout)
        signal.setitimer(signal.ITIMER_REAL, self.seconds)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.seconds is not None:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
            if self._old_handler is not None:
                signal.signal(signal.SIGALRM, self._old_handler)
            if self._old_timer and self._old_timer[0] > 0:
                signal.setitimer(signal.ITIMER_REAL, self._old_timer[0], self._old_timer[1])
        return False


@dataclass
class EvaluationResult:
    """
    Result of evaluating a CSD strategy on a dataset sample.

    Contains metrics and sample outputs for feedback to the generator.
    """
    success: bool
    accuracy: float  # 0.0 to 1.0
    contains_delimiters: bool
    syntax_rate: float  # 0.0 to 1.0
    num_examples: int
    num_correct: int
    total_time_seconds: float
    accuracy_denominator: Optional[int] = None
    accuracy_definition: str = "correct_examples_over_all_examples"
    invalid_outputs_excluded_from_accuracy: int = 0
    max_sample_time_seconds: float = 0.0
    early_stopped: bool = False
    early_stop_reason: Optional[str] = None
    planned_num_examples: Optional[int] = None
    task_guidance: List[str] = field(default_factory=list)

    # Sample outputs for feedback (question, expected, actual, is_correct)
    sample_outputs: List[Dict[str, Any]] = field(default_factory=list)

    # Error information if evaluation failed
    error: Optional[str] = None
    aux_metrics: Dict[str, Any] = field(default_factory=dict)

    def meets_threshold(
        self,
        min_accuracy: float = 0.0,
        min_syntax_rate: float = 0.0,
        require_delimiters: bool = True,
        max_seconds_per_example: Optional[float] = None,
    ) -> bool:
        """Check if aggregate metrics meet the specified thresholds."""
        if not self.sample_outputs:
            return False
        if self.early_stopped:
            return False
        runtime_ok = True
        if max_seconds_per_example is not None:
            runtime_ok = self.max_sample_time_seconds <= max_seconds_per_example
        return (
            runtime_ok
            and self.accuracy >= min_accuracy
            and self.syntax_rate >= min_syntax_rate
        )

    def get_feedback_summary(self, require_delimiters: bool = True) -> str:
        """Generate a summary for feedback to the generator.

        When ``require_delimiters`` is False, the visible ``<<``/``>>`` span
        diagnostics are omitted: the "Contains << >>" header line, the whole
        "Structural Generation Metrics" block, and the span-centric lines of
        the diagnostic decomposition. Those statistics are meaningless when the
        model is not asked to emit spans and have misled the author into
        chasing a non-issue.

        Callers get that flag from
        ``registry.resolve_require_delimiters(dataset, cli_value)``, which asks
        the benchmark itself via ``emits_visible_delimiters()``. GSM can emit
        spans, so there the CLI flag decides; Spider and SMILES generate whole
        outputs and cannot emit a span at all, so for them it is always False.
        """
        eval_count_label = (
            f"{self.num_examples}/{self.planned_num_examples}"
            if self.early_stopped and self.planned_num_examples
            else str(self.num_examples)
        )

        smiles_trial_raw = self.aux_metrics.get("smiles_paper_trial")
        smiles_trial = None
        if isinstance(smiles_trial_raw, dict):
            smiles_trial = SmilesTrialFields(
                validity_pct=f"{smiles_trial_raw.get('validity_rdkit', 0.0):.1%}",
                membership_pct=f"{smiles_trial_raw.get('membership', 0.0):.1%}",
                diversity_display=(
                    str(smiles_trial_raw.get("diversity_tanimoto"))
                    if smiles_trial_raw.get("diversity_tanimoto") is not None
                    else "n/a"
                ),
                retro_score_display=(
                    str(smiles_trial_raw.get("retro_score"))
                    if smiles_trial_raw.get("retro_score") is not None
                    else "n/a"
                ),
                samples_to_target_display=str(
                    smiles_trial_raw.get("samples_to_target_unique_valid", "n/a")
                ),
                unique_valid_display=(
                    f"{smiles_trial_raw.get('unique_valid_count', 0)}/"
                    f"{smiles_trial_raw.get('sample_count', 0)}"
                ),
            )

        anti_raw = self.aux_metrics.get("anti_degeneracy")
        anti_degeneracy = None
        if isinstance(anti_raw, dict):
            anti_degeneracy = AntiDegeneracyFields(
                churn_ratio_display=f"{anti_raw.get('delimiter_churn_ratio', 0.0):.3f}",
                tiny_span_pct=f"{anti_raw.get('tiny_span_rate', 0.0):.1%}",
                max_steps_hit_pct=f"{anti_raw.get('max_steps_hit_rate', 0.0):.1%}",
                penalty_pct=f"{anti_raw.get('penalty', 0.0):.1%}",
                adjusted_membership_pct=(
                    f"{anti_raw.get('adjusted_membership_score', self.accuracy):.1%}"
                ),
            )

        early_stop_raw = self.aux_metrics.get("early_stop")
        early_stop_metrics = None
        if isinstance(early_stop_raw, dict):
            early_stop_metrics = EarlyStopMetricsFields(
                reason=str(early_stop_raw.get("reason", "unknown")),
                max_possible_pct=(
                    f"{float(early_stop_raw.get('max_possible_accuracy', 0.0)):.1%}"
                ),
                target_pct=f"{float(early_stop_raw.get('target_accuracy', 0.0)):.1%}",
                evaluated_display=(
                    f"{early_stop_raw.get('evaluated_examples', 0)}/"
                    f"{early_stop_raw.get('total_examples', self.num_examples)}"
                ),
            )

        failure_modes = [
            FailureModeEntry(mode=mode, count=count, detail=detail)
            for mode, count, detail in self._summarize_failure_modes()
        ]

        output_run_summary = self._summarize_output_run()

        # Anonymized failure summary (April 25): we used to dump up to 3 specific
        # failed examples (question + expected + actual) into the feedback to
        # gpt-5.4. That content leaked synthesis-sample specifics into the
        # generator's context, biasing strategies to fit those exact examples
        # and producing 12-22 pp held-out drops. We now report only aggregate
        # statistics — failure_mode counts (already populated above), runtime
        # budget exceedances, and any unexpected exception types — without any
        # question text, expected SQL, or actual SQL strings.
        aggregate_failure_stats_lines: List[str] = []
        if self.sample_outputs:
            n_total = len(self.sample_outputs)
            n_runtime_exceeded = sum(
                1 for s in self.sample_outputs if s.get("runtime_budget_exceeded")
            )
            if n_runtime_exceeded:
                aggregate_failure_stats_lines.append(
                    f"  {n_runtime_exceeded}/{n_total} examples timed out "
                    f"(exceeded the per-example time limit) and were scored as failures "
                    f"(accuracy 0, syntax 0 for each timed-out example)."
                )

        diagnostic_metrics = self._summarize_diagnostic_metrics(require_delimiters)

        # The structural block is entirely visible-`<<`-span statistics, which
        # are meaningless when delimiters are not required — omit the whole block.
        structural_metrics: List[str] = []
        if require_delimiters:
            structural_metrics = self._summarize_structural_metrics()

        # Change 1: replace the 5 flat aggregate blocks (Diagnostic Error
        # Decomposition, Output Provenance, Correct-vs-Wrong Contrast,
        # Structural Generation Metrics, Representative Factual Snapshots)
        # with a single cluster-organized failure block. Each wrong sample
        # is reduced to a 17-axis fingerprint; failures are grouped by
        # Hamming distance ≤ 1 into discovered modes. The cluster view shows
        # which axes are constant across all clusters (true of every failure)
        # versus which vary (where the real distinct failure modes diverge).
        #
        # Persistent cluster IDs across attempts (Change 2) are wired in by
        # feedback_loop.py via attach_cluster_ledger().
        cluster_block = render_cluster_block(
            self.sample_outputs,
            max_steps=512,
            slow_threshold_seconds=30.0,
            persistent_ledger=getattr(self, "_failure_ledger", None),
            attempt_index=getattr(self, "_attempt_index", None),
            require_delimiters=require_delimiters,
        )

        model = FeedbackSummaryModel(
            eval_count_label=eval_count_label,
            accuracy_pct=f"{self.accuracy:.1%}",
            num_correct=self.num_correct,
            accuracy_denominator_display=self.accuracy_denominator or self.num_examples,
            show_contains_delimiters=require_delimiters,
            contains_delimiters_yesno="yes" if self.contains_delimiters else "no",
            syntax_pct=f"{self.syntax_rate:.1%}",
            total_time_str=f"{self.total_time_seconds:.2f}",
            max_sample_time_str=f"{self.max_sample_time_seconds:.2f}",
            early_stopped=self.early_stopped,
            early_stop_reason=self.early_stop_reason,
            accuracy_definition_display=(
                self.accuracy_definition
                if self.accuracy_definition != "correct_examples_over_all_examples"
                else None
            ),
            invalid_outputs_excluded_from_accuracy=self.invalid_outputs_excluded_from_accuracy,
            task_guidance=list(self.task_guidance),
            smiles_trial=smiles_trial,
            anti_degeneracy=anti_degeneracy,
            early_stop_metrics=early_stop_metrics,
            failure_modes=failure_modes,
            output_run_summary=list(output_run_summary),
            aggregate_failure_stats_lines=aggregate_failure_stats_lines,
            diagnostic_metrics=list(diagnostic_metrics),
            show_structural_metrics=bool(structural_metrics),
            structural_metrics=list(structural_metrics),
            cluster_block=cluster_block,
        )

        rendered = _render_prompt(model, "feedback/feedback_summary.j2")
        # The original implementation joined its lines with "\n" and never
        # added a trailing newline; keep_trailing_newline on the shared Jinja
        # environment means the template's own final line terminator survives
        # rendering, so strip exactly that one trailing newline back off.
        if rendered.endswith("\n"):
            rendered = rendered[:-1]
        return rendered

    @staticmethod
    def _format_trace_event(event: Dict[str, Any]) -> str:
        helper = event.get("helper", "unknown")
        detail = event.get("detail") or ""
        before = event.get("cost_before")
        after = event.get("cost_after")
        cost_part = ""
        if before is not None or after is not None:
            cost_part = f" [cost {before}->{after}]"
        return f"{helper}: {detail}{cost_part}".strip()

    @staticmethod
    def _redact_artifact_preview(value: Any, max_chars: int = 96) -> str:
        """Return a compact structural preview without preserving dataset-specific text."""
        if value is None:
            return "none"
        text = str(value).replace("\\n", " ").replace("\\r", " ")
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return "empty"
        text = _anonymize_sql_preview(text)
        keep_words = {"true", "false", "yes", "no", "none", "null", "and", "or", "not"}
        text = re.sub(r"[-+]?\d+(?:\.\d+)?", "<num>", text)
        text = re.sub(
            r"(?<!<)\b[A-Za-z_][A-Za-z0-9_]*\b(?!>)",
            lambda m: m.group(0) if m.group(0).lower() in keep_words or m.group(0).upper() in _SQL_KEYWORDS else "<id>",
            text,
        )
        if len(text) > max_chars:
            text = text[: max_chars - 3] + "..."
        return text

    @staticmethod
    def _format_counter(counter: Counter[str], denominator: int, max_items: int = 5) -> str:
        from synthesis.evaluate.benchmarks.common.formatting import format_named_counter

        return format_named_counter(
            counter,
            denominator,
            max_items=max_items,
            min_denominator=1,
        )

    @classmethod
    def _helper_counts_for_sample(cls, sample: Dict[str, Any]) -> Counter[str]:
        return Counter(
            event.get("helper", "unknown")
            for event in sample.get("helper_trace") or []
        )

    @classmethod
    def _control_tags_for_sample(cls, sample: Dict[str, Any]) -> List[str]:
        helpers = set(cls._helper_counts_for_sample(sample))
        tags: List[str] = []
        if helpers & cls._UNCONSTRAINED_HELPERS:
            tags.append("free_lm_generation")
        if "UnconstrainedChunk" in helpers:
            tags.append("free_lm_chunking")
        if "EnterObservedConstrainedSpan" in helpers:
            tags.append("observed_span_entry")
        if "OpenConstrainedSpan" in helpers:
            tags.append("explicit_span_entry")
        if helpers & cls._HARD_TOKEN_HELPERS:
            tags.append("hard_token_constrained")
        if helpers & cls._CONFIDENCE_HELPERS:
            tags.append("confidence_gated")
        if helpers & cls._GROUP_OR_ADAPTIVE_HELPERS:
            tags.append("group_or_adaptive_bias")
        if helpers & cls._SAFE_LOGIT_STEP_HELPERS:
            tags.append("safe_logit_step")
        if helpers & cls._SOFT_CONSTRAINED_HELPERS:
            tags.append("soft_then_hard_fallback")
        if helpers & cls._SYMBOL_HELPERS:
            tags.append("symbol_or_chunk_acceptance")
        if helpers & cls._REPAIR_HELPERS:
            tags.append("parser_repair_or_rollback")
        if not any(event.get("helper", "unknown") in cls._CONSTRAINED_HELPERS for event in sample.get("helper_trace") or []):
            tags.append("no_constrained_activity")
        return tags or ["no_trace"]

    @staticmethod
    def _sample_identity_metadata(example: Dict[str, Any], example_index: int) -> Dict[str, Any]:
        """Return stable sample identity fields for observability and ledgers."""
        metadata: Dict[str, Any] = {"example_index": example_index}
        for key in (
            "crane_source_index",
            "spider_source_index",
            "id_orig",
            "id_shuffled",
            "db_id",
        ):
            if key in example:
                metadata[key] = example[key]
        if "crane_source_index" in metadata:
            metadata["source_index"] = metadata["crane_source_index"]
        elif "spider_source_index" in metadata:
            metadata["source_index"] = metadata["spider_source_index"]
        elif "source_index" in example:
            metadata["source_index"] = example["source_index"]
        return metadata

    @classmethod
    def _derive_answer_provenance(cls, sample: Dict[str, Any]) -> str:
        source = sample.get("answer_source") or "none"
        tags = set(sample.get("provenance_tags") or cls._control_tags_for_sample(sample))
        if source == "last_visible_span":
            if "no_constrained_activity" in tags:
                return "last_visible_span_without_constrained_activity"
            if "explicit_span_entry" in tags:
                return "last_visible_span_after_explicit_entry"
            if "observed_span_entry" in tags:
                return "last_visible_span_after_observed_entry"
            return "last_visible_span_with_constrained_activity"
        if source == "text_fallback":
            return "free_text_fallback"
        if source == "hidden_or_task_extractor":
            if "no_constrained_activity" in tags:
                return "task_extractor_without_constrained_activity"
            return "task_extractor_with_constrained_activity"
        return "no_scored_answer"

    @classmethod
    def _derive_failure_location(cls, sample: Dict[str, Any]) -> str:
        if sample.get("runtime_budget_exceeded"):
            return "time_budget_exceeded"
        if sample.get("error"):
            return "output_record_error"
        if sample.get("is_correct"):
            return "correct"
        if not sample.get("has_extracted_answer", False):
            return "answer_extraction_or_completion"
        if sample.get("hit_max_steps"):
            return "token_budget_exhausted"
        if not sample.get("uses_hidden_chunks"):
            if int(sample.get("num_visible_spans", 0) or 0) == 0:
                return "span_absent"
            if int(sample.get("num_valid_visible_spans", 0) or 0) == 0:
                return "no_valid_visible_span"
            if not sample.get("is_syntax_valid"):
                return "visible_span_syntax"
        if sample.get("is_syntax_valid") and not sample.get("is_correct"):
            return "syntax_valid_semantic_mismatch"
        if cls._sample_has_constrained_activity(sample):
            return "wrong_after_constrained_activity"
        return "wrong_without_constrained_activity"

    @classmethod
    def _annotate_sample_observability(cls, sample: Dict[str, Any]) -> Dict[str, Any]:
        tags = cls._control_tags_for_sample(sample)
        sample["provenance_tags"] = tags
        sample["answer_provenance"] = cls._derive_answer_provenance(sample)
        sample["failure_location"] = cls._derive_failure_location(sample)
        return sample

    def get_provenance_counts(self) -> Dict[str, Counter[str]]:
        """Return neutral provenance/localization buckets for prompt deltas."""
        answer_provenance: Counter[str] = Counter()
        control_tags: Counter[str] = Counter()
        failure_location: Counter[str] = Counter()
        for sample in self.sample_outputs:
            if "provenance_tags" not in sample:
                self._annotate_sample_observability(sample)
            answer_provenance[sample.get("answer_provenance", "unknown")] += 1
            failure_location[sample.get("failure_location", "unknown")] += 1
            for tag in sample.get("provenance_tags") or []:
                control_tags[tag] += 1
        return {
            "answer_provenance": answer_provenance,
            "control_tags": control_tags,
            "failure_location": failure_location,
        }

    def _summarize_provenance_metrics(self) -> List[str]:
        if not self.sample_outputs:
            return []
        n = len(self.sample_outputs)
        counts = self.get_provenance_counts()
        return [
            "Answer provenance: " + self._format_counter(counts["answer_provenance"], n),
            "Control path tags: " + self._format_counter(counts["control_tags"], n),
            "Failure localization: " + self._format_counter(counts["failure_location"], n),
        ]

    def _summarize_correct_wrong_contrast(self) -> List[str]:
        if not self.sample_outputs:
            return []

        def summarize(label: str, samples: List[Dict[str, Any]]) -> str:
            n = len(samples)
            if n == 0:
                return f"{label}: 0 examples"
            provenance = Counter(sample.get("answer_provenance", "unknown") for sample in samples)
            tags: Counter[str] = Counter()
            locations = Counter(sample.get("failure_location", "unknown") for sample in samples)
            for sample in samples:
                for tag in sample.get("provenance_tags") or []:
                    tags[tag] += 1
            avg_tokens = self._mean([float(sample.get("token_count", 0) or 0) for sample in samples]) or 0.0
            avg_valid_spans = self._mean([float(sample.get("num_valid_visible_spans", 0) or 0) for sample in samples]) or 0.0
            syntax_valid = sum(1 for sample in samples if sample.get("is_syntax_valid"))
            return (
                f"{label}: {n} examples; syntax_valid {syntax_valid}/{n}; "
                f"avg_tokens {avg_tokens:.2f}; avg_valid_spans {avg_valid_spans:.2f}; "
                f"provenance {self._format_counter(provenance, n, max_items=3)}; "
                f"control_tags {self._format_counter(tags, n, max_items=4)}; "
                f"locations {self._format_counter(locations, n, max_items=4)}"
            )

        correct = [sample for sample in self.sample_outputs if sample.get("is_correct")]
        wrong = [sample for sample in self.sample_outputs if not sample.get("is_correct")]
        return [summarize("Correct examples", correct), summarize("Wrong examples", wrong)]

    def get_behavioral_context_summary(
        self, max_examples: int = 1, max_trace_events: int = 12, require_delimiters: bool = True
    ) -> str:
        traced_examples = [s for s in self.sample_outputs if s.get("helper_trace")]
        if not traced_examples:
            return ""

        lines = ["Recent evaluated behavior from the most recent compiled/evaluated attempt:"]
        for idx, sample in enumerate(traced_examples[:max_examples]):
            trace = sample.get("helper_trace") or []
            counts = Counter(event.get("helper", "unknown") for event in trace)
            lines.append(f"Example {idx + 1}:")
            if "provenance_tags" not in sample:
                self._annotate_sample_observability(sample)
            # The "Contains << >>" status only matters when delimiters are required;
            # omit it when the dataset does not require them so the author is not
            # fed an irrelevant span signal on tasks (e.g. SQL) that use no << >>
            # delimiters.
            delim_segment = (
                f"Contains << >>: {'yes' if sample.get('contains_delimiters') else 'no'} | "
                if require_delimiters
                else ""
            )
            lines.append(
                f"  Token count: {sample.get('token_count', 'N/A')} | "
                f"{delim_segment}"
                f"Syntax rate: {sample.get('syntax_rate', 0.0):.1%} | "
                f"Provenance: {sample.get('answer_provenance', 'unknown')} | "
                f"Location: {sample.get('failure_location', 'unknown')}"
            )
            if counts:
                counts_summary = ", ".join(
                    f"{name}={count}" for name, count in counts.most_common(8)
                )
                lines.append(f"  Helper call counts: {counts_summary}")
            tail = trace[-max_trace_events:]
            if tail:
                lines.append("  Helper trace tail:")
                for event in tail:
                    lines.append(f"    - {self._format_trace_event(event)}")

        model = HintLinesModel(lines=lines)
        rendered = _render_prompt(model, "feedback_loop/hint_lines.j2")
        # The original implementation joined its lines with "\n" and never added
        # a trailing newline; keep_trailing_newline on the shared Jinja
        # environment means the template's own final line terminator survives
        # rendering, so strip exactly that one trailing newline back off (same
        # idiom as get_feedback_summary / get_failure_summary).
        if rendered.endswith("\n"):
            rendered = rendered[:-1]
        return rendered

    def _classify_sample_failure_modes(self, sample: Dict[str, Any]) -> List[Tuple[str, str]]:
        """Classify a single sample's observable failure modes.

        Returns an ordered list of (mode_key, detail_string) tuples for every
        failure-mode bucket this sample matches, in cascade order (first-seen
        detail order). Shared by `_summarize_failure_modes` (aggregate counts)
        and `_pick_representative_samples_by_mode` (example picking) so the
        two never disagree on how a sample is classified.

        NOTE: the delimiter rules below deliberately use `<<`/`>>` counts and the
        parser's span accounting (num_visible_spans / num_valid_visible_spans),
        NOT the old "is this substring present anywhere" checks. This is an
        intentional feedback-quality fix, not accidental drift from the
        byte-identical prompt refactor: the old rules only fired in the
        all-or-nothing extreme and mislabeled mixed outputs (one span dangling,
        an extra close, or 1-of-N spans invalid) as "other". Covered by
        tests/prompt_rendering/test_feedback_content_fixes.py. These labels feed
        only the author feedback text, never any scored metric.
        """
        error = sample.get("error")
        full_output = sample.get("full_output") or ""
        actual = sample.get("actual") or ""
        contains_delimiters = sample.get("contains_delimiters", False)
        uses_hidden_chunks = sample.get("uses_hidden_chunks", False)
        visible_delimiters = sample.get("visible_delimiters", contains_delimiters)
        used_constrained_chunk = sample.get("used_constrained_chunk", contains_delimiters)
        syntax_rate = float(sample.get("syntax_rate", 0.0))
        matched = False
        modes: List[Tuple[str, str]] = []

        if sample.get("runtime_budget_exceeded"):
            modes.append(("too_slow", "(generation exceeded the per-example runtime budget)"))
            matched = True
            if error:
                return modes

        if error:
            modes.append(("runtime_or_generation_error", f"(first error: {str(error)[:120]})"))
            return modes

        n_open = full_output.count("<<")
        n_close = full_output.count(">>")
        if not uses_hidden_chunks and n_open > n_close:
            if n_close == 0:
                detail = "(opened `<<` but did not close `>>`)"
            else:
                detail = f"(opened {n_open} `<<` but closed only {n_close} `>>`)"
            modes.append(("unterminated_constrained_segment", detail))
            matched = True

        if uses_hidden_chunks:
            if not used_constrained_chunk:
                modes.append(("missing_constrained_chunk", "(no internal parser-governed chunk was used)"))
                matched = True
        elif not contains_delimiters and "<<" not in full_output:
            modes.append(("missing_constrained_segment", "(no `<< >>` segment detected)"))
            matched = True

        if not uses_hidden_chunks and n_close > n_open:
            if n_open == 0:
                detail = "(generated `>>` without a matching opening `<<`)"
            else:
                detail = f"(closed {n_close} `>>` but only {n_open} `<<` were opened)"
            modes.append(("premature_or_unmatched_closure", detail))
            matched = True

        num_visible_spans = int(sample.get("num_visible_spans", 0))
        num_valid_visible_spans = int(sample.get("num_valid_visible_spans", 0))
        if (
            not uses_hidden_chunks
            and num_visible_spans > 0
            and num_valid_visible_spans < num_visible_spans
        ):
            detail = (
                f"({num_visible_spans - num_valid_visible_spans}/{num_visible_spans} "
                "closed spans failed syntax checks)"
            )
            modes.append(("malformed_constrained_content", detail))
            matched = True

        if not uses_hidden_chunks and self._looks_like_early_constrained_entry(full_output):
            modes.append((
                "entered_constrained_mode_too_early",
                "(output entered `<<` almost immediately after the prompt continuation began)",
            ))
            matched = True

        if self._has_repetition_loop(full_output):
            modes.append(("repetition_loop", "(local token pattern repeated in output)"))
            matched = True

        if not actual:
            modes.append(("answer_extraction_failed", "(no extractable final answer)"))
            matched = True

        if not matched and not sample.get("is_correct", False):
            modes.append((
                "other_observed_failure",
                "(uncategorized wrong answer; see representative rollout examples "
                "for observed output)",
            ))

        return modes

    def _summarize_failure_modes(self) -> List[Tuple[str, int, str]]:
        """Classify the most common observable evaluation failure patterns."""
        counters: Dict[str, int] = {}
        details: Dict[str, str] = {}

        for sample in self.sample_outputs:
            for mode, detail in self._classify_sample_failure_modes(sample):
                counters[mode] = counters.get(mode, 0) + 1
                if mode not in details:
                    details[mode] = detail

        ranked = sorted(counters.items(), key=lambda item: (-item[1], item[0]))
        return [(mode, count, details.get(mode, "")) for mode, count in ranked]

    def _pick_representative_samples_by_mode(self) -> List[Tuple[str, Dict[str, Any]]]:
        """For each representative failure mode, pick up to 3 samples by `full_output`
        length: shortest, median, longest (N=2 -> shortest+longest; N=1 -> only).

        Returns a flat list [(mode_key, sample_dict), ...] ordered by mode
        frequency. This intentionally caps representative examples separately
        from aggregate failure-mode counts, which list all detected buckets.
        Within a mode bucket picks appear in shortest->median->longest order.
        """
        mode_to_samples: Dict[str, List[Dict[str, Any]]] = {}
        counters: Dict[str, int] = {}

        def _add(mode: str, sample: Dict[str, Any]) -> None:
            mode_to_samples.setdefault(mode, []).append(sample)
            counters[mode] = counters.get(mode, 0) + 1

        for sample in self.sample_outputs:
            for mode, _detail in self._classify_sample_failure_modes(sample):
                _add(mode, sample)

        ranked = sorted(counters.items(), key=lambda item: (-item[1], item[0]))[:4]
        picks: List[Tuple[str, Dict[str, Any]]] = []
        for mode, _count in ranked:
            candidates = mode_to_samples.get(mode, [])
            if not candidates:
                continue
            by_len = sorted(candidates, key=lambda s: len(s.get("full_output") or ""))
            n = len(by_len)
            if n == 1:
                chosen_idxs = [0]
            elif n == 2:
                chosen_idxs = [0, 1]
            else:
                chosen_idxs = [0, n // 2, n - 1]
            for idx in chosen_idxs:
                picks.append((mode, by_len[idx]))
        return picks

    def _render_mode_examples(self) -> str:
        """Render one verbatim failing-rollout block per top-4 failure mode.

        Each block: GSM prompt, Qwen full output, extracted answer, correct
        answer — no truncation, no telemetry, no narrative framing. Returns
        empty string when there are no failed samples.
        """
        if not self.sample_outputs:
            return ""
        picks = self._pick_representative_samples_by_mode()
        if not picks:
            return ""
        blocks: List[ModeExampleEntry] = []
        for mode, sample in picks:
            prompt = sample.get("question_full") or sample.get("question") or ""
            qwen_output = sample.get("full_output") or ""
            actual_val = sample.get("actual")
            actual_str = "" if actual_val is None else str(actual_val)
            expected_val = sample.get("expected")
            expected_str = "" if expected_val is None else str(expected_val)
            blocks.append(
                ModeExampleEntry(
                    mode=mode,
                    prompt=prompt,
                    qwen_output=qwen_output,
                    actual_str=actual_str,
                    expected_str=expected_str,
                )
            )
        model = ModeExamplesModel(blocks=blocks)
        rendered = _render_prompt(model, "evaluator/mode_examples.j2")
        # The original implementation joined its blocks with "\n\n" and never
        # added a trailing newline; keep_trailing_newline on the shared Jinja
        # environment means the template's own final line terminator survives
        # rendering, so strip exactly that one trailing newline back off (same
        # idiom as get_feedback_summary / get_failure_summary).
        if rendered.endswith("\n"):
            rendered = rendered[:-1]
        return rendered

    @staticmethod
    def _mean(values: List[float]) -> Optional[float]:
        if not values:
            return None
        return sum(values) / len(values)

    @staticmethod
    def _median(values: List[float]) -> Optional[float]:
        if not values:
            return None
        ordered = sorted(values)
        mid = len(ordered) // 2
        if len(ordered) % 2:
            return ordered[mid]
        return (ordered[mid - 1] + ordered[mid]) / 2

    @staticmethod
    def _visible_span_shape(output: str) -> Dict[str, Any]:
        """Return delimiter/span shape statistics without preserving span text."""
        opens = output.count("<<")
        closes = output.count(">>")
        spans = re.findall(r"<<\s*(.*?)\s*>>", output, flags=re.DOTALL)

        active_opens = 0
        unmatched_closes = 0
        i = 0
        while i < len(output):
            if output.startswith("<<", i):
                active_opens += 1
                i += 2
            elif output.startswith(">>", i):
                if active_opens > 0:
                    active_opens -= 1
                else:
                    unmatched_closes += 1
                i += 2
            else:
                i += 1

        first_open_tokens: Optional[int] = None
        first_open = output.find("<<")
        if first_open >= 0:
            first_open_tokens = len(output[:first_open].split())

        complete_spans = len(spans)
        return {
            "opens": opens,
            "closes": closes,
            "complete_spans": complete_spans,
            "first_open_tokens": first_open_tokens,
            "unterminated": active_opens > 0 or opens > complete_spans,
            "unmatched_close": unmatched_closes > 0 or closes > complete_spans,
            "balanced_with_span": complete_spans > 0 and opens == closes == complete_spans,
        }

    _HARD_TOKEN_HELPERS = {"ConstrainedStep"}
    _CONFIDENCE_HELPERS = {"ConfidenceGatedStep"}
    _GROUP_OR_ADAPTIVE_HELPERS = {"AdaptiveConstrainedStep", "GroupBoostedConstrainedStep"}
    _SAFE_LOGIT_STEP_HELPERS = {
        "SafeBoostedConstrainedStep",
        "SafePenalizedConstrainedStep",
        "SafeRepetitionPenaltyStep",
        "SafeTemperatureConstrainedStep",
    }
    _SOFT_CONSTRAINED_HELPERS = {"SoftConstrainedStep", "SafeSoftConstrainedStep"}
    _SYMBOL_HELPERS = {"ConstrainedSymbol", "ConstrainedSymbolInGenerated"}
    # Non-"Safe" logit-step variants (traced constrained-step helpers, previously omitted).
    _LOGIT_STEP_HELPERS = {
        "BoostedConstrainedStep",
        "PenalizedConstrainedStep",
        "RepetitionPenaltyStep",
        "TemperatureConstrainedStep",
        "AdaptiveConstrainedStepWithPenalties",
    }
    # Rollout/generation constrained helpers (traced constrained-step helpers, previously omitted).
    _ROLLOUT_GENERATION_HELPERS = {
        "SpeculativeConstrainedRollout",
        "RolloutConstrainedWithPenalties",
        "ConstrainedGeneration",
        "CraneGeneration",
    }
    _REPAIR_HELPERS = {
        "RollbackConstrainedSpan",
        "RollbackConstrainedSuffix",
        "RollbackToValidPrefix",
        # Traced constrained-step helpers, previously omitted.
        "RollbackAndContinue",
        "RegenerateUnitOnCheckFailure",
    }
    _CONSTRAINED_HELPERS = {
        "OpenConstrainedSpan",
        "EnterObservedConstrainedSpan",
        "CloseConstrainedSpan",
        "AppendConstrainedToken",
        *_HARD_TOKEN_HELPERS,
        *_CONFIDENCE_HELPERS,
        *_GROUP_OR_ADAPTIVE_HELPERS,
        *_SAFE_LOGIT_STEP_HELPERS,
        *_SOFT_CONSTRAINED_HELPERS,
        *_SYMBOL_HELPERS,
        *_REPAIR_HELPERS,
        *_LOGIT_STEP_HELPERS,
        *_ROLLOUT_GENERATION_HELPERS,
    }

    _UNCONSTRAINED_HELPERS = {"UnconstrainedStep", "UnconstrainedChunk"}

    @classmethod
    def _sample_has_valid_span_or_chunk(cls, sample: Dict[str, Any]) -> bool:
        return (
            int(sample.get("num_valid_visible_spans", 0) or 0) > 0
            or bool(sample.get("used_constrained_chunk") and sample.get("uses_hidden_chunks"))
        )

    @classmethod
    def _sample_has_constrained_activity(cls, sample: Dict[str, Any]) -> bool:
        return any(
            event.get("helper", "unknown") in cls._CONSTRAINED_HELPERS
            for event in sample.get("helper_trace") or []
        )

    def get_diagnostic_counts(self) -> Dict[str, int]:
        """Return neutral per-example diagnostic buckets for prompt deltas."""
        samples = self.sample_outputs
        n = len(samples)
        syntax_valid = [s for s in samples if s.get("is_syntax_valid")]
        syntax_invalid = [s for s in samples if not s.get("is_syntax_valid")]
        valid_span = [s for s in samples if self._sample_has_valid_span_or_chunk(s)]
        constrained_activity = [s for s in samples if self._sample_has_constrained_activity(s)]
        visible_span_no_activity = [
            s
            for s in samples
            if int(s.get("num_visible_spans", 0) or 0) > 0
            and not self._sample_has_constrained_activity(s)
        ]

        return {
            "examples": n,
            "syntax_valid_correct": sum(1 for s in syntax_valid if s.get("is_correct")),
            "syntax_valid_wrong": sum(1 for s in syntax_valid if not s.get("is_correct")),
            "syntax_invalid_correct": sum(1 for s in syntax_invalid if s.get("is_correct")),
            "syntax_invalid_wrong": sum(1 for s in syntax_invalid if not s.get("is_correct")),
            "no_complete_span_wrong": sum(
                1
                for s in samples
                if int(s.get("num_visible_spans", 0) or 0) == 0
                and not s.get("is_correct")
            ),
            "answer_from_last_visible_span": sum(
                1 for s in samples if s.get("answer_source") == "last_visible_span"
            ),
            "answer_from_text_fallback": sum(
                1 for s in samples if s.get("answer_source") == "text_fallback"
            ),
            "answer_from_hidden_or_task_extractor": sum(
                1 for s in samples if s.get("answer_source") == "hidden_or_task_extractor"
            ),
            "no_extracted_answer": sum(
                1 for s in samples if not s.get("has_extracted_answer", False)
            ),
            "examples_with_final_answer_span": sum(
                1 for s in samples if s.get("answer_source") == "last_visible_span"
            ),
            "examples_with_valid_nonfinal_spans_only": sum(
                1
                for s in valid_span
                if s.get("answer_source") != "last_visible_span"
            ),
            "examples_with_no_valid_span": n - len(valid_span),
            "examples_with_constrained_activity": len(constrained_activity),
            "examples_without_constrained_activity": n - len(constrained_activity),
            "correct_with_constrained_activity": sum(
                1 for s in constrained_activity if s.get("is_correct")
            ),
            "wrong_with_constrained_activity": sum(
                1 for s in constrained_activity if not s.get("is_correct")
            ),
            "correct_without_constrained_activity": sum(
                1
                for s in samples
                if not self._sample_has_constrained_activity(s) and s.get("is_correct")
            ),
            "wrong_without_constrained_activity": sum(
                1
                for s in samples
                if not self._sample_has_constrained_activity(s) and not s.get("is_correct")
            ),
            "visible_span_without_constrained_activity": len(visible_span_no_activity),
        }

    @staticmethod
    def _is_helper_api_error(error: str) -> bool:
        """Heuristic bucket for actual helper/API exceptions, not behavior metrics."""
        lowered = error.lower()
        helper_terms = (
            "helper",
            "csdhelpers",
            "constrainedstep",
            "adaptiveconstrainedstep",
            "groupboostedconstrainedstep",
            "constrainedsymbol",
            "constrainedsymbolingenerated",
            "appendconstrainedtoken",
            "openconstrainedspan",
            "enterobservedconstrainedspan",
            "closeconstrainedspan",
            "rollbackconstrainedsuffix",
        )
        api_error_terms = (
            "attributeerror",
            "nameerror",
            "typeerror",
            "missing",
            "undefined",
            "not defined",
            "has no attribute",
            "takes",
            "argument",
        )
        return any(term in lowered for term in helper_terms) and any(
            term in lowered for term in api_error_terms
        )

    def _summarize_output_run(self) -> List[str]:
        """Summarize factual output behavior without causal labels."""
        if not self.sample_outputs:
            return []

        n = len(self.sample_outputs)
        errors = [str(sample.get("error") or "") for sample in self.sample_outputs if sample.get("error")]
        timeouts = sum(1 for sample in self.sample_outputs if sample.get("runtime_budget_exceeded"))
        nonempty_outputs = sum(
            1
            for sample in self.sample_outputs
            if int(sample.get("token_count", 0) or 0) > 0
            or bool((sample.get("scored_output") or sample.get("full_output") or "").strip())
        )

        lines = [
            "verification: passed",
            "compilation: passed",
            f"generated-token outputs: {nonempty_outputs}/{n} nonempty",
        ]
        if not errors and timeouts == 0:
            lines.append("All evaluated examples completed and returned generated output records.")
            return lines

        if errors:
            lines.append(f"examples without completed output records: {len(errors)}/{n}")
        if timeouts:
            lines.append(f"per-example time budget exceeded: {timeouts}/{n}")
        return lines

    def _summarize_diagnostic_metrics(self, require_delimiters: bool = True) -> List[str]:
        """Summarize where failures enter without exposing example content.

        When ``require_delimiters`` is False, the span-centric lines
        (no-complete-span counts, answer-extraction source, span usefulness) are
        omitted — they only describe visible ``<<``/``>>`` behavior, which is not
        expected on datasets that do not require delimiters (Spider, SMILES).
        """
        if not self.sample_outputs:
            return []

        counts = self.get_diagnostic_counts()
        n = counts["examples"]
        metrics = [
            (
                "Correctness by syntax bucket: "
                f"syntax_valid_correct {counts['syntax_valid_correct']}/{n}, "
                f"syntax_valid_wrong {counts['syntax_valid_wrong']}/{n}, "
                f"syntax_invalid_correct {counts['syntax_invalid_correct']}/{n}, "
                f"syntax_invalid_wrong {counts['syntax_invalid_wrong']}/{n}"
            ),
        ]
        if require_delimiters:
            metrics.extend([
                f"No-complete-span wrong answers: {counts['no_complete_span_wrong']}/{n}",
                (
                    "Answer extraction source: "
                    f"last_visible_span {counts['answer_from_last_visible_span']}/{n}, "
                    f"text_fallback {counts['answer_from_text_fallback']}/{n}, "
                    f"hidden_or_task_extractor {counts['answer_from_hidden_or_task_extractor']}/{n}, "
                    f"none {counts['no_extracted_answer']}/{n}"
                ),
                (
                    "Span usefulness: "
                    f"final_answer_span {counts['examples_with_final_answer_span']}/{n}, "
                    f"valid_nonfinal_spans_only {counts['examples_with_valid_nonfinal_spans_only']}/{n}, "
                    f"no_valid_span {counts['examples_with_no_valid_span']}/{n}"
                ),
            ])
        metrics.extend([
            (
                "Constrained intervention activity: "
                f"examples_with_activity {counts['examples_with_constrained_activity']}/{n}, "
                f"examples_without_activity {counts['examples_without_constrained_activity']}/{n}, "
                f"visible_span_without_activity {counts['visible_span_without_constrained_activity']}/{n}"
            ),
            (
                "Correctness conditioned on constrained activity: "
                f"correct_with_activity {counts['correct_with_constrained_activity']}/{n}, "
                f"wrong_with_activity {counts['wrong_with_constrained_activity']}/{n}, "
                f"correct_without_activity {counts['correct_without_constrained_activity']}/{n}, "
                f"wrong_without_activity {counts['wrong_without_constrained_activity']}/{n}"
            ),
        ])
        return metrics

    def _summarize_structural_metrics(self) -> List[str]:
        """Summarize neutral span/search behavior for refinement feedback."""
        if not self.sample_outputs:
            return []

        n = len(self.sample_outputs)
        shapes = [
            self._visible_span_shape(s.get("scored_output") or s.get("full_output") or "")
            for s in self.sample_outputs
        ]
        helper_counts: Counter[str] = Counter()

        constrained_calls = 0
        unconstrained_calls = 0
        for sample in self.sample_outputs:
            for event in sample.get("helper_trace") or []:
                helper = event.get("helper", "unknown")
                helper_counts[helper] += 1
                constrained_calls += int(helper in EvaluationResult._CONSTRAINED_HELPERS)
                unconstrained_calls += int(helper in EvaluationResult._UNCONSTRAINED_HELPERS)

        total_opens = sum(int(shape["opens"]) for shape in shapes)
        total_complete_spans = sum(int(shape["complete_spans"]) for shape in shapes)
        span_completion_rate = total_complete_spans / total_opens if total_opens else None

        first_open_positions = [
            float(shape["first_open_tokens"])
            for shape in shapes
            if shape["first_open_tokens"] is not None
        ]
        # Read the per-sample token lengths recorded at generation time rather
        # than re-deriving them from the output text: there is one measurement
        # of span length, taken with the model's tokenizer, not two.
        visible_span_lengths = [
            float(length)
            for sample in self.sample_outputs
            for length in sample.get("visible_span_token_lengths", [])
        ]
        valid_span_lengths = [
            float(length)
            for sample in self.sample_outputs
            for length in sample.get("valid_visible_span_token_lengths", [])
        ]
        output_tokens = [
            float(sample.get("token_count", 0) or 0)
            for sample in self.sample_outputs
        ]
        runtimes = [
            float(sample.get("time_seconds", 0.0) or 0.0)
            for sample in self.sample_outputs
        ]

        examples_with_visible_open = sum(1 for shape in shapes if shape["opens"] > 0)
        examples_with_complete_span = sum(1 for shape in shapes if shape["complete_spans"] > 0)
        examples_with_balanced_span = sum(1 for shape in shapes if shape["balanced_with_span"])
        examples_with_unterminated = sum(1 for shape in shapes if shape["unterminated"])
        examples_with_unmatched_close = sum(1 for shape in shapes if shape["unmatched_close"])
        examples_without_complete_span = n - examples_with_complete_span
        examples_with_valid_span = sum(
            1 for sample in self.sample_outputs
            if self._sample_has_valid_span_or_chunk(sample)
        )
        examples_with_parser_span_failure = sum(
            1 for sample in self.sample_outputs
            if int(sample.get("num_visible_spans", 0) or 0)
            > int(sample.get("num_valid_visible_spans", 0) or 0)
        )
        examples_with_valid_span_wrong = sum(
            1 for sample in self.sample_outputs
            if (
                int(sample.get("num_valid_visible_spans", 0) or 0) > 0
                or bool(sample.get("used_constrained_chunk") and sample.get("uses_hidden_chunks"))
            )
            and not sample.get("is_correct")
        )
        examples_format_valid_wrong = sum(
            1 for sample in self.sample_outputs
            if sample.get("is_syntax_valid") and not sample.get("is_correct")
        )
        examples_without_extracted_answer = sum(
            1 for sample in self.sample_outputs
            if not sample.get("has_extracted_answer", False)
        )
        examples_with_tiny_valid_spans = sum(
            1 for sample in self.sample_outputs
            if sample.get("valid_visible_span_token_lengths")
            and max(sample.get("valid_visible_span_token_lengths")) <= 2
        )
        examples_with_long_visible_span = sum(
            1 for sample in self.sample_outputs
            if any(
                length > 64
                for length in sample.get("visible_span_token_lengths", [])
            )
        )
        examples_hitting_max_steps = sum(
            1 for sample in self.sample_outputs if sample.get("hit_max_steps")
        )

        mean_spans = self._mean([float(shape["complete_spans"]) for shape in shapes])
        mean_first_open = self._mean(first_open_positions)
        median_first_open = self._median(first_open_positions)
        mean_visible_span_len = self._mean(visible_span_lengths)
        median_visible_span_len = self._median(visible_span_lengths)
        mean_valid_span_len = self._mean(valid_span_lengths)
        mean_output_tokens = self._mean(output_tokens)
        median_output_tokens = self._median(output_tokens)
        max_output_tokens = max(output_tokens, default=0.0)
        total_runtime = sum(runtimes)
        total_tokens = sum(output_tokens)
        time_per_token = total_runtime / total_tokens if total_tokens else None
        constrained_fraction = (
            constrained_calls / (constrained_calls + unconstrained_calls)
            if constrained_calls + unconstrained_calls
            else None
        )

        def fmt_optional(value: Optional[float], suffix: str = "") -> str:
            return "n/a" if value is None else f"{value:.2f}{suffix}"

        lines = [
            f"Examples with visible `<<`: {examples_with_visible_open}/{n}",
            f"Examples with complete visible spans: {examples_with_complete_span}/{n}",
            f"Examples without complete visible spans: {examples_without_complete_span}/{n}",
            f"Examples with balanced visible spans: {examples_with_balanced_span}/{n}",
            f"Examples with unterminated visible spans: {examples_with_unterminated}/{n}",
            f"Examples with unmatched visible close: {examples_with_unmatched_close}/{n}",
            f"Visible span completion rate: {fmt_optional(span_completion_rate)}",
            f"Avg complete visible spans/example: {fmt_optional(mean_spans)}",
            (
                "Tokens before first visible open: "
                f"avg {fmt_optional(mean_first_open)}, median {fmt_optional(median_first_open)}"
            ),
            (
                "Visible span token length: "
                f"avg {fmt_optional(mean_visible_span_len)}, median {fmt_optional(median_visible_span_len)}"
            ),
            f"Valid visible span token length avg: {fmt_optional(mean_valid_span_len)}",
            f"Examples with at least one valid span/chunk: {examples_with_valid_span}/{n}",
            f"Examples with visible parser span failure: {examples_with_parser_span_failure}/{n}",
            f"Examples with valid span/chunk but wrong answer: {examples_with_valid_span_wrong}/{n}",
            f"Examples syntax-valid but wrong answer: {examples_format_valid_wrong}/{n}",
            f"Examples without extracted answer: {examples_without_extracted_answer}/{n}",
            f"Examples with only tiny valid visible spans: {examples_with_tiny_valid_spans}/{n}",
            f"Examples with long visible span (>64 tokens): {examples_with_long_visible_span}/{n}",
            f"Examples hitting max steps: {examples_hitting_max_steps}/{n}",
            (
                "Generated tokens/example: "
                f"avg {fmt_optional(mean_output_tokens)}, median {fmt_optional(median_output_tokens)}, "
                f"max {max_output_tokens:.0f}"
            ),
            f"Runtime per generated token: {fmt_optional(time_per_token, 's')}",
        ]
        if helper_counts:
            top_helpers = ", ".join(
                f"{name}={count}" for name, count in helper_counts.most_common(8)
            )
            lines.append(f"Top helper calls: {top_helpers}")
            lines.append(
                "Constrained helper call fraction: "
                f"{fmt_optional(constrained_fraction)}"
            )

        return lines

    def _looks_like_early_constrained_entry(self, output: str) -> bool:
        """Heuristic: flags outputs that open a constrained segment almost immediately."""
        if "<<" not in output:
            return False
        prefix = output.split("<<", 1)[0].strip()
        if not prefix:
            return True
        return len(prefix.split()) <= 4

    def _has_repetition_loop(self, output: str) -> bool:
        """Detect short repeated local patterns that often indicate degenerate decoding."""
        tokens = output.split()
        if len(tokens) < 6:
            return False
        for width in (1, 2, 3):
            for start in range(0, len(tokens) - 3 * width + 1):
                chunk = tokens[start:start + width]
                if (
                    tokens[start + width:start + 2 * width] == chunk
                    and tokens[start + 2 * width:start + 3 * width] == chunk
                ):
                    return True
        return False

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "success": self.success,
            "accuracy": self.accuracy,
            "contains_delimiters": self.contains_delimiters,
            "syntax_rate": self.syntax_rate,
            "num_examples": self.num_examples,
            "num_correct": self.num_correct,
            "accuracy_denominator": self.accuracy_denominator or self.num_examples,
            "accuracy_definition": self.accuracy_definition,
            "invalid_outputs_excluded_from_accuracy": self.invalid_outputs_excluded_from_accuracy,
            "total_time_seconds": self.total_time_seconds,
            "max_sample_time_seconds": self.max_sample_time_seconds,
            "early_stopped": self.early_stopped,
            "early_stop_reason": self.early_stop_reason,
            "planned_num_examples": self.planned_num_examples,
            "error": self.error,
            "sample_outputs": self.sample_outputs,
            "aux_metrics": self.aux_metrics,
        }


# ---------------------------------------------------------------------------
# CRANE-faithful GSM-Symbolic correctness check.
#
# Ported from legacy/CRANE/src/prompting/gsm_symbolic.py so OUR grader scores
# correctness exactly the way the CRANE baseline does. The primary method is a
# z3 "for-all" proof: the model expression is counted correct only if it can be
# proven equal to the gold expression for EVERY valid assignment of the named
# variables (subject to their integer/float type constraints) -- not by plugging
# in one instance's numbers. A 1000-random-sample check is the fallback when the
# solver fails or times out, matching CRANE. Argument order matches CRANE
# parse_answer: expr1 = model completion, expr2 = gold answer.
# ---------------------------------------------------------------------------
def _crane_floor_div_replacer(expression: str) -> str:
    regex_with_groups = r"(?P<left>.+?)\s*//\s*(?P<right>.+)"

    def replace_floor_div(match):
        left = match.group("left").strip()
        right = match.group("right").strip()
        return f"z3_floor_div({left}, {right})"

    return re.sub(regex_with_groups, replace_floor_div, expression)


def _crane_test_expression_equivalence(expr1_gsm, expr2_gsm, var_names, var_types):
    """1000-random-sample fallback (faithful CRANE test_expression_equivalence)."""
    import random as _rng

    for _ in range(1000):
        test_case = {}
        for var in var_names:
            # Plain indexing, not .get(), matching upstream (CRANE
            # gsm_symbolic.py:101). A variable the regex picked up that has no
            # entry at all raises KeyError and stops the run, exactly as it does
            # upstream. Softening this to .get() returns a verdict upstream
            # never produces. A variable that IS present but carries a type
            # upstream does not recognise is the different case handled below.
            vt = var_types[var]
            if vt == "float between 0 and 1":
                test_case[var] = _rng.uniform(0.001, 1)
            elif vt == "float":
                test_case[var] = _rng.uniform(0.001, 100)
            elif vt == "int":
                test_case[var] = _rng.randint(1, 100)
            # An untyped variable is deliberately left unassigned, matching
            # upstream (CRANE gsm_symbolic.py:100-107). It survives into the
            # substituted string, so the eval below raises NameError -- and
            # which answer that produces depends on which expression it sits in,
            # because the two catches are asymmetric. Refusing to sample instead
            # is stricter than CRANE and undercounts against their published
            # number. Pinned by tests/test_crane_parity_is_exact.py.
        expr1_sub = expr1_gsm
        expr2_sub = expr2_gsm
        for var, value in test_case.items():
            expr1_sub = re.sub(rf"\b{re.escape(var)}\b", str(value), expr1_sub)
            expr2_sub = re.sub(rf"\b{re.escape(var)}\b", str(value), expr2_sub)
        try:
            ans1 = eval(expr1_sub)  # noqa: S307 - numeric arithmetic only (CRANE parity)
        except Exception:
            return False
        try:
            ans2 = eval(expr2_sub)  # noqa: S307
        except Exception:
            return True
        if ans1 != ans2:
            return False
    return True


def _crane_validate_expression_equivalence(expr1, expr2, var_types) -> bool:
    """z3 for-all proof of equivalence (faithful CRANE validate_expression_equivalence).

    expr1 = model completion, expr2 = gold answer. Returns True only if expr1 is
    provably equal to expr2 for all valid variable values. Model answers
    containing round( are rejected. Falls back to random sampling on solver/eval
    failure or timeout.
    """
    from z3 import Solver, unsat, unknown, Real, ToInt, ToReal, And, If

    original_expr1 = expr1
    original_expr2 = expr2

    var_names = set(re.findall(r"\b[a-zA-Z_]\w*\b", expr1 + " " + expr2))
    var_names -= {"int"}

    def Floor(x):
        return If(x >= 0, ToInt(x), ToInt(x) - If(ToReal(ToInt(x)) == x, 0, 1))

    def Ceiling(x):
        return If(x >= 0, ToInt(x) + If(ToReal(ToInt(x)) == x, 0, 1), ToInt(x))

    def IntegerCheck(x):
        return And(x == Floor(x), x == Ceiling(x))

    # Two CRANE golds that z3 cannot encode cleanly -> random-sample method.
    if original_expr1 in (
        "int(p * (1 + r1/100) * (1 - r2/100)) * n",
        "(int(length / (plant_width + space)) - owned) * cost",
    ):
        return _crane_test_expression_equivalence(
            original_expr1, original_expr2, var_names, var_types
        )

    vars_dict = {}
    constraints = []
    for name in var_names:
        var = Real(name)
        vars_dict[name] = var
        var_type = var_types.get(name, "str")
        if var_type == "float between 0 and 1":
            constraints.append(var > 0)
            constraints.append(var <= 1)
        elif var_type == "float":
            constraints.append(var > 0)
        elif var_type == "int":
            constraints.append(var > 0)
            constraints.append(IntegerCheck(var))
        else:
            return False

    expr1 = re.sub(r"\bint\(", "ToInt(", expr1)
    expr2 = re.sub(r"\bint\(", "ToInt(", expr2)

    if "round(" in expr1:
        return False
    # Deliberately dead, and it must stay dead. Upstream writes this pattern as
    # r'\round\(' (CRANE gsm_symbolic.py:168), where \r is the carriage-return
    # escape -- so it matches nothing and the gold keeps its round(. safe_eval
    # then throws, because the eval environment has no builtins, and grading
    # falls through to the random sampler, where a plain eval does have round()
    # and computes it correctly.
    #
    # Repairing the typo is not a fix. ToInt truncates and round rounds, so
    # rewriting both int( and round( to ToInt( makes them the same function: a
    # model answering int(x/3) gets proved equal to a gold of round(x/3) and
    # scored correct. Pinned by tests/test_crane_parity_is_exact.py.
    expr2 = re.sub(r"\round\(", "ToInt(", expr2)

    if "//" in expr1:
        expr1 = _crane_floor_div_replacer(expr1)
    if "//" in expr2:
        expr2 = _crane_floor_div_replacer(expr2)

    def z3_floor_div(x, y):
        return If(y != 0, ToInt(x / y), 0)

    def safe_eval(expr):
        return eval(  # noqa: S307 - restricted env: only z3 vars + ToInt/z3_floor_div
            expr,
            {"__builtins__": None},
            {**vars_dict, "ToInt": ToInt, "z3_floor_div": z3_floor_div},
        )

    try:
        expr2_z3 = safe_eval(expr2)
    except Exception:
        return _crane_test_expression_equivalence(
            original_expr1, original_expr2, var_names, var_types
        )
    try:
        expr1_z3 = safe_eval(expr1)
    except Exception:
        return _crane_test_expression_equivalence(
            original_expr1, original_expr2, var_names, var_types
        )

    s = Solver()
    s.set("timeout", 5000)
    s.add(constraints)
    try:
        s.add(expr1_z3 != expr2_z3)
    except Exception:
        return _crane_test_expression_equivalence(
            original_expr1, original_expr2, var_names, var_types
        )

    result = s.check()
    if result == unsat:
        return True
    elif result == unknown:
        return _crane_test_expression_equivalence(
            original_expr1, original_expr2, var_names, var_types
        )
    return False


class Evaluator:
    """
    Evaluates synthesized CSD strategies on dataset samples.
    """

    def __init__(
        self,
        dataset_name: str = "gsm_symbolic",
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        backend: str = "huggingface",
        device: str = "cuda",
        sample_size: int = 10,
        max_steps: int = 600,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        vllm_tensor_parallel_size: Optional[int] = None,
        vllm_pipeline_parallel_size: int = 1,
        vllm_gpu_memory_utilization: float = 0.8,
        vllm_max_model_len: int = 16384,
        vllm_enforce_eager: bool = VLLM_ENFORCE_EAGER,
        sample_seed: Optional[int] = None,
        max_seconds_per_example: Optional[float] = None,
        step_token_budget: int = 1,
        gsm_source_dir: str | Path | None = None,
        gsm_split_file: str | Path | None = None,
        gsm_split_name: str = "train",
        spider_split_file: str | Path | None = None,
        spider_split_name: str = "train",
        smiles_classes: Optional[List[str]] = None,
        grammars_dir: str | Path | None = None,
        early_stop_on_answer: bool = False,
    ):
        """
        Initialize the evaluator.

        Args:
            dataset_name: Dataset to evaluate on ("gsm_symbolic", "spider", or "smiles")
            model_name: HuggingFace model for generation
            backend: Runtime LM backend ("huggingface" or "vllm")
            device: Device to run on ("cuda", "mps", "cpu")
            sample_size: Number of examples to evaluate on
            max_steps: Maximum generation steps per example
            load_in_4bit: Whether to load model in 4-bit quantization
            load_in_8bit: Whether to load model in 8-bit quantization
            vllm_tensor_parallel_size: Explicit tensor parallel size for vLLM
            vllm_pipeline_parallel_size: Explicit pipeline parallel size for vLLM
            vllm_gpu_memory_utilization: GPU memory fraction reserved by vLLM
            vllm_max_model_len: Max context length passed to vLLM
            vllm_enforce_eager: Disable cudagraph/compile in vLLM for stability
            sample_seed: Optional RNG seed for reproducible dataset sampling
            max_seconds_per_example: Optional runtime budget per example in seconds
            gsm_split_file: Optional JSON manifest with train_indices/test_indices for GSM.
            gsm_split_name: Which split from gsm_split_file to use ("train" or "test").
            spider_split_file: Optional JSON manifest with train_indices/test_indices for Spider.
            spider_split_name: Which split from spider_split_file to use ("train" or "test").
        """
        if backend not in {"huggingface", "vllm"}:
            raise NotImplementedError(
                "Evaluation backend must be 'huggingface' or 'vllm'. "
                "Hosted API backends are not supported by the current CSD runtime because "
                "the generated Dafny strategy needs direct token logits, masking, and tokenizer access."
            )

        self.dataset_name = dataset_name
        self.model_name = model_name
        self.backend = backend
        self.device = device
        self.sample_size = sample_size
        self.max_steps = max_steps
        self.load_in_4bit = load_in_4bit
        self.load_in_8bit = load_in_8bit
        from synthesis.evaluate.benchmarks.common.model_utils import resolve_vllm_tensor_parallel_size

        self.vllm_tensor_parallel_size = resolve_vllm_tensor_parallel_size(vllm_tensor_parallel_size)
        self.vllm_pipeline_parallel_size = vllm_pipeline_parallel_size
        self.vllm_gpu_memory_utilization = vllm_gpu_memory_utilization
        self.vllm_max_model_len = vllm_max_model_len
        self.vllm_enforce_eager = vllm_enforce_eager
        self.sample_seed = sample_seed
        # Confirmation-slice offset for the greedy hill-climb search (slice B).
        # 0 (the default) is the normal slice-A sampling behavior; a caller
        # passes a nonzero value into evaluate_sample() to draw the NEXT
        # sample_size examples instead. Reset on every evaluate_sample() call.
        self.sample_offset: int = 0
        self.max_seconds_per_example = max_seconds_per_example
        self.step_token_budget = step_token_budget
        # CRANE-style answer early stop (default OFF): stop generation once the
        # output contains a finished final-answer span instead of running to
        # the max_steps cap. See run_crane_csd(early_stop_on_answer=...).
        self.early_stop_on_answer = early_stop_on_answer
        self.gsm_source_dir = gsm_source_dir
        self.gsm_split_file = Path(gsm_split_file) if gsm_split_file is not None else None
        self.gsm_split_name = gsm_split_name
        self.spider_split_file = Path(spider_split_file) if spider_split_file is not None else None
        self.spider_split_name = spider_split_name
        self.smiles_classes = smiles_classes
        self.grammars_dir = Path(grammars_dir).expanduser() if grammars_dir is not None else None

        # Lazy-loaded components
        self._dataset = None
        self._env = None
        self._env_cache_key: Optional[tuple[Any, ...]] = None
        self._grammar_file = None
        self._base_grammar_text: Optional[str] = None
        self._dynamic_parser_factory_cache: Dict[Tuple[Any, ...], Any] = {}
        self._syntax_parser_cache: Dict[Tuple[str, ...], Any] = {}

    def unload_runtime(self) -> None:
        """Release cached runtime model state so the generator can reclaim GPU memory."""
        self._env = None
        self._env_cache_key = None
        if self.backend == "vllm":
            try:
                from synthesis.evaluate.benchmarks.common.model_utils import clear_vllm_engine_cache

                clear_vllm_engine_cache()
            except Exception:
                pass
        else:
            import gc

            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

    def split_provenance(self, bar_split_name: str | None = None) -> dict:
        """The split-provenance dict every output JSON embeds (one shape, one place)."""
        from synthesis.split_provenance import build_split_provenance

        return build_split_provenance(
            gsm_split_file=self.gsm_split_file,
            gsm_split_name=self.gsm_split_name if self.gsm_split_file is not None else None,
            spider_split_file=self.spider_split_file,
            spider_split_name=self.spider_split_name if self.spider_split_file is not None else None,
            bar_split_name=bar_split_name,
        )

    def _read_split_manifest(self, split_file: str | Path) -> dict:
        """Load a benchmark split manifest from a filesystem path."""
        return json.loads(Path(split_file).read_text())

    def _load_gsm_split_indices(self) -> Optional[List[int]]:
        """Load explicit GSM example indices from a train/test split manifest."""
        if self.gsm_split_file is None:
            return None

        # GSM's two sides are named "train" and "test", same as Spider — the
        # manifests' eval_indices key was renamed test_indices 2026-07-17 so
        # one concept has one name everywhere.
        if self.gsm_split_name not in ("train", "test"):
            raise ValueError(
                f"gsm_split_name must be 'train' or 'test', got "
                f"'{self.gsm_split_name}'. GSM's held-out side is named "
                "'test' (the 'eval' alias was removed 2026-07-17)."
            )
        manifest = self._read_split_manifest(self.gsm_split_file)
        key = f"{self.gsm_split_name}_indices"
        if key not in manifest:
            available = sorted(k for k in manifest.keys() if k.endswith("_indices"))
            raise ValueError(
                f"Split file {self.gsm_split_file} does not contain {key}. "
                f"Available index fields: {available}"
            )

        indices = manifest[key]
        if not isinstance(indices, list) or not all(isinstance(i, int) for i in indices):
            raise ValueError(f"{key} in {self.gsm_split_file} must be a list of integers")
        # Skip the first `sample_offset` entries so the caller's existing
        # limit-to-sample_size slicing takes the NEXT N indices instead of
        # the usual first N -- this is the greedy hill-climb search's
        # slice-B confirmation eval (see evaluate_sample's sample_offset).
        offset = self.sample_offset or 0
        return indices[offset:]

    def _load_spider_split_indices(self) -> Optional[List[int]]:
        """Load explicit Spider example indices from a train/test split manifest."""
        if self.spider_split_file is None:
            return None

        # Spider's two sides are named "train" and "test" — exactly the index
        # keys the manifests store. The legacy "eval" alias (and its
        # eval_indices fallback) was removed 2026-07-17: aliases are how the
        # bar/eval split mixup stayed invisible.
        if self.spider_split_name not in ("train", "test"):
            raise ValueError(
                f"spider_split_name must be 'train' or 'test', got "
                f"'{self.spider_split_name}'. Spider's held-out side is named "
                "'test' (the 'eval' alias was removed 2026-07-17)."
            )
        manifest = self._read_split_manifest(self.spider_split_file)
        split_name = self.spider_split_name
        key = f"{split_name}_indices"
        if key not in manifest:
            available = sorted(k for k in manifest.keys() if k.endswith("_indices"))
            raise ValueError(
                f"Split file {self.spider_split_file} does not contain {key}. "
                f"Available index fields: {available}"
            )

        indices = manifest[key]
        if not isinstance(indices, list) or not all(isinstance(i, int) for i in indices):
            raise ValueError(f"{key} in {self.spider_split_file} must be a list of integers")
        # See _load_gsm_split_indices: skip the first `sample_offset` entries
        # so slice-B confirmation draws the NEXT N indices.
        offset = self.sample_offset or 0
        return indices[offset:]

    def _get_grammar_file(self) -> Path:
        """Get the grammar file path for the dataset."""
        if self._grammar_file is None:
            grammars_dir = self.grammars_dir or Path(
                os.environ.get(
                    "CSD_GRAMMARS_DIR",
                    str(Path(__file__).parent / "grammars"),
                )
            ).expanduser()
            from synthesis.evaluate.benchmarks.registry import get_logic

            logic = get_logic(self.dataset_name)
            self._grammar_file = logic.get_grammar_file(self, grammars_dir)
        return self._grammar_file

    def _normalize_smiles_classes(self) -> List[str]:
        """Return the selected SMILES classes as a normalized list."""
        from synthesis.evaluate.benchmarks.smiles.eval_logic import normalize_classes

        return normalize_classes(self)

    def _get_grammar_text(self) -> str:
        """Load and cache the active grammar text."""
        if self._base_grammar_text is None:
            self._base_grammar_text = self._get_grammar_file().read_text()
        return self._base_grammar_text

    def _get_syntax_parser(self, example: Optional[dict] = None):
        """Create or reuse a syntax parser for one example's allowed variables."""
        logic = self._benchmark_logic()
        return logic.get_syntax_parser(self, example)

    def _load_dataset_sample(self) -> list:
        """Load a sample of the dataset for evaluation."""
        if self._dataset is not None:
            return self._dataset
        from synthesis.evaluate.benchmarks.registry import get_logic

        logic = get_logic(self.dataset_name)
        self._dataset = logic.load_dataset_sample(self)

        return self._dataset

    def _setup_environment(self, compiled_module_path: Path) -> Dict[str, Any]:
        """
        Set up the Dafny environment for evaluation.

        Args:
            compiled_module_path: Path to the compiled CSD module

        Returns:
            Environment dict with loaded modules
        """
        run_dir = compiled_module_path.parent
        if run_dir.name in {"generated_csd", "python"}:
            run_dir = run_dir.parent

        memory_utilization_max_raw = os.environ.get(
            "CSD_VLLM_GPU_MEMORY_UTILIZATION_MAX", ""
        ).strip()
        env_cache_key = (
            str(run_dir.resolve()),
            self.dataset_name,
            self.model_name,
            self.backend,
            self.device,
            self.load_in_4bit,
            self.load_in_8bit,
            self.vllm_tensor_parallel_size,
            self.vllm_pipeline_parallel_size,
            self.vllm_gpu_memory_utilization,
            self.vllm_max_model_len,
            self.vllm_enforce_eager,
            memory_utilization_max_raw,
        )
        if self._env is not None and self._env_cache_key == env_cache_key:
            return self._env

        if self.dataset_name == "gsm_symbolic":
            from synthesis.evaluate.benchmarks.gsm_symbolic.environment import setup_dafny_environment
        elif self.dataset_name == "spider":
            from synthesis.evaluate.benchmarks.sql_spider.environment import setup_dafny_environment
        elif self.dataset_name == "smiles":
            from synthesis.evaluate.benchmarks.smiles.environment import setup_dafny_environment
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")

        def _make_env(
            gpu_memory_utilization: float,
            tensor_parallel_size: int | None,
        ) -> Dict[str, Any]:
            return setup_dafny_environment(
                run_dir=run_dir,
                model_name=self.model_name,
                backend=self.backend,
                device=self.device,
                grammar_file=self._get_grammar_file(),
                load_in_4bit=self.load_in_4bit,
                load_in_8bit=self.load_in_8bit,
                vllm_tensor_parallel_size=tensor_parallel_size,
                vllm_pipeline_parallel_size=self.vllm_pipeline_parallel_size,
                vllm_gpu_memory_utilization=gpu_memory_utilization,
                vllm_max_model_len=self.vllm_max_model_len,
                vllm_enforce_eager=self.vllm_enforce_eager,
            )

        if self.backend != "vllm":
            env = _make_env(self.vllm_gpu_memory_utilization, self.vllm_tensor_parallel_size)
            self._env = env
            self._env_cache_key = env_cache_key
            return env

        from synthesis.evaluate.benchmarks.common.model_utils import (
            clear_vllm_engine_cache,
            narrow_cuda_visible_devices_to_index,
            pick_cuda_device_index_with_most_free_memory,
            visible_cuda_device_ids,
        )
        from synthesis.evaluate.benchmarks.common.vllm_startup import (
            is_vllm_startup_memory_error,
            vllm_util_retry_candidates,
        )

        try:
            import torch
        except Exception:
            torch = None  # type: ignore[assignment]

        requested_tp = self.vllm_tensor_parallel_size or 1

        tp_candidates: List[int] = []
        for candidate in (requested_tp, 1):
            if candidate >= 1 and candidate not in tp_candidates:
                tp_candidates.append(candidate)

        memory_utilization_max = (
            float(memory_utilization_max_raw) if memory_utilization_max_raw else None
        )
        util_candidates = vllm_util_retry_candidates(
            self.vllm_gpu_memory_utilization, maximum=memory_utilization_max
        )

        def _narrow_to_freest_gpu(reason: str) -> None:
            best_idx = pick_cuda_device_index_with_most_free_memory()
            chosen = narrow_cuda_visible_devices_to_index(best_idx)
            print(
                f"Retrying vLLM evaluator startup on {reason} "
                f"(CUDA_VISIBLE_DEVICES={chosen})"
            )
            clear_vllm_engine_cache()
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()

        def _multiple_devices_visible() -> bool:
            visible_ids = visible_cuda_device_ids()
            if visible_ids:
                return len(visible_ids) > 1
            if torch is not None and torch.cuda.is_available():
                return torch.cuda.device_count() > 1
            return False

        narrowed_visible_devices = False

        def _try_startup_ladder() -> Dict[str, Any]:
            nonlocal narrowed_visible_devices
            last_error: Exception | None = None
            for tp in tp_candidates:
                if tp == 1 and tp != requested_tp and not narrowed_visible_devices:
                    narrowed_visible_devices = True
                    _narrow_to_freest_gpu("a single GPU")
                util_idx = 0
                while util_idx < len(util_candidates):
                    util = util_candidates[util_idx]
                    try:
                        if tp != requested_tp and util == util_candidates[0]:
                            print(
                                f"Retrying vLLM evaluator startup with "
                                f"tensor_parallel_size={tp}"
                            )
                        elif util != self.vllm_gpu_memory_utilization:
                            direction = (
                                "higher"
                                if util > self.vllm_gpu_memory_utilization
                                else "lower"
                            )
                            print(
                                f"Retrying vLLM evaluator startup with {direction} "
                                f"gpu_memory_utilization={util:.2f}"
                            )
                        return _make_env(util, tp)
                    except Exception as exc:
                        last_error = exc
                        if not is_vllm_startup_memory_error(exc):
                            raise
                        clear_vllm_engine_cache()
                        if torch is not None and torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        if (
                            tp == 1
                            and not narrowed_visible_devices
                            and _multiple_devices_visible()
                        ):
                            # A sibling process may hold the default visible device;
                            # move to the freest GPU and restart the utilization ladder
                            # instead of shrinking utilization on the occupied device.
                            narrowed_visible_devices = True
                            _narrow_to_freest_gpu("the freest GPU")
                            util_idx = 0
                            continue
                        is_last_attempt = tp == tp_candidates[-1] and util == util_candidates[-1]
                        if is_last_attempt:
                            raise
                        util_idx += 1
            if last_error is not None:
                raise last_error
            raise RuntimeError("Failed to initialize evaluation environment.")

        # When the only visible GPU is transiently held by a sibling process,
        # exhausting the utilization ladder in a few minutes kills the attempt
        # without an Accuracy telemetry line. Wait out the pressure (bounded)
        # and retry the whole ladder before giving up.
        import time as _time

        startup_wait_s = float(os.environ.get("CSD_VLLM_STARTUP_WAIT_S", "900"))
        retry_interval_s = float(os.environ.get("CSD_VLLM_STARTUP_RETRY_INTERVAL_S", "30"))
        deadline = _time.monotonic() + startup_wait_s
        wait_round = 0
        while True:
            try:
                env = _try_startup_ladder()
                self._env = env
                self._env_cache_key = env_cache_key
                return env
            except Exception as exc:
                if not is_vllm_startup_memory_error(exc):
                    raise
                remaining = deadline - _time.monotonic()
                if remaining <= 0:
                    raise
                wait_round += 1
                wait_s = min(retry_interval_s, remaining)
                print(
                    f"[vllm] Evaluator startup blocked by GPU memory pressure "
                    f"(round {wait_round}); waiting {wait_s:.0f}s for a sibling "
                    f"process to release memory ({remaining:.0f}s budget left)...",
                    flush=True,
                )
                clear_vllm_engine_cache()
                if torch is not None and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                _time.sleep(wait_s)

    def _extract_constrained_content(self, output: str) -> List[str]:
        """Extract content within << >> delimiters.

        Non-greedy `.*?` with DOTALL so spans containing `<`, `>`, or
        comparison operators (Spider SQL: `>`, `<`, `>=`, `<=`, `<>`)
        match — SQL has no `>>` operator, so the next `>>` always closes.
        """
        return re.findall(r"<<\s*(.*?)\s*>>", output, flags=re.DOTALL)

    def _truncate_gsm_output(self, output: str) -> str:
        """Trim obvious prompt restarts so scoring focuses on the first answer block."""
        cut_points: List[int] = []
        for marker in [
            "\nAssistant:",
            "\n\nAssistant:",
            "\nQ:",
            "\n\nQ:",
            "\nSolve the question above.",
            "\n\nSolve the question above.",
        ]:
            idx = output.find(marker)
            if idx > 0:
                cut_points.append(idx)

        if not cut_points:
            return output

        return output[:min(cut_points)].rstrip()

    # Seven more methods lived here, all of them only reachable from the
    # deleted numeric grader below: _parse_variable_assignments,
    # _parse_symbolic_assignments, _safe_eval_arithmetic,
    # _evaluate_symbolic_expression, _resolve_symbolic_assignments,
    # _extract_answer_expression_gsm and _evaluate_gsm_expression. Together
    # they pulled variable values out of the model's prose and substituted them
    # into its formula to get one number. Nothing needs a number any more.

    # _extract_answer_gsm used to live here. It pulled a single number out of the
    # model's output so GSM answers could be graded by comparing that number to
    # the gold. CRANE has no such grader -- its parse_answer (gsm_symbolic.py:28-56)
    # either proves the two formulas equivalent or leaves correct = False -- so
    # every example it scored was measured a way CRANE does not measure, then
    # averaged into a figure reported against CRANE's published number. Deleted
    # rather than left dormant behind a branch nobody takes. See
    # tests/test_unreadable_types_does_not_switch_graders.py.

    def _extract_answer_smiles(self, output: str, example: Optional[dict] = None) -> Optional[str]:
        """Extract the generated SMILES string from raw output."""
        from synthesis.evaluate.benchmarks.smiles.metrics import clean_smiles_output

        smiles = clean_smiles_output(output)
        return smiles or None

    # _answers_match used to live here -- the string comparison the deleted
    # numeric grader above ended in. GSM was its only caller anywhere in
    # synthesis/, so it went with it.

    def _gsm_symbolic_equivalence(
        self, model_expr: Optional[str], expected_expr: str, variable_types: dict
    ) -> bool:
        """Check symbolic equivalence the way CRANE does: a z3 for-all proof that the
        model expression equals the gold for EVERY valid variable assignment, with a
        1000-random-sample fallback. This replaces the old single-instance numeric
        substitution, which mis-scored every int()-containing gold (z3 port lives in
        _crane_validate_expression_equivalence above)."""
        if model_expr is None:
            return False
        model_expr = str(model_expr).strip()
        expected_expr = str(expected_expr).strip()
        if _is_pathological_gsm_scoring_expression(model_expr):
            return False
        # CRANE rejects model completions containing ** (parse_answer guard).
        if "**" in model_expr:
            return False
        # Nothing below is caught, deliberately. A variable_types field we cannot
        # read, or a prover that will not run, is a fault in our setup -- not
        # evidence about the model. Swallowing either one grades every example in
        # the split as wrong and reports 0% accuracy, which is indistinguishable
        # from a model that genuinely got everything wrong. That is the exact
        # failure that cost weeks on Spider. Before this was removed, two
        # identical formulas graded False on any machine without z3, because the
        # ModuleNotFoundError landed in the catch below.
        #
        # Upstream CRANE has no catch here either: parse_answer calls plain
        # eval() on this field (gsm_symbolic.py:40) and lets a bad value stop the
        # run, so parity and honest reporting agree. Pinned by
        # tests/test_gsm_grader_does_not_swallow_setup_failures.py.
        if isinstance(variable_types, str):
            import ast as _ast

            variable_types = _ast.literal_eval(variable_types)
        if not isinstance(variable_types, dict):
            raise UngradableExample(
                "GSM variable_types must be a mapping of variable name to type, "
                f"got {type(variable_types).__name__}: {variable_types!r}"
            )
        return bool(
            _crane_validate_expression_equivalence(
                model_expr, expected_expr, variable_types
            )
        )

    def _get_expected_answer(self, example: dict) -> str:
        """Get the expected answer from a dataset example."""
        from synthesis.evaluate.benchmarks.registry import get_logic

        logic = get_logic(self.dataset_name)
        return logic.expected_answer(self, example)

    def _format_prompt(self, example: dict) -> Union[str, List[dict]]:
        """Format a dataset example as a prompt."""
        from synthesis.evaluate.benchmarks.registry import get_logic

        logic = get_logic(self.dataset_name)
        return logic.format_prompt(self, example)

    def _contains_delimiters(self, output: str) -> bool:
        """Check if the output contains at least one non-empty << >> segment."""
        return "<<" in output and ">>" in output

    def _check_syntax_validity(
        self,
        output: str,
        example: Optional[dict] = None,
    ) -> Tuple[bool, List[Tuple[str, bool]]]:
        """
        Check if constrained segments have valid syntax.

        Returns:
            Tuple of (all_valid, list of (segment, is_valid) tuples)
        """
        from lark.exceptions import LarkError

        if self.dataset_name == "smiles":
            from synthesis.evaluate.benchmarks.smiles.metrics import evaluate_smiles_output

            class_name = (example or {}).get("class_name", "smiles")
            prompt_exemplars = (example or {}).get("prompt_exemplars", [])
            grammar_text = (example or {}).get("grammar_text", "")
            eval_row = evaluate_smiles_output(
                class_name,
                output,
                grammar_text,
                prompt_exemplars,
                require_rdkit=True,
            )
            smiles = eval_row["smiles"]
            # region agent log
            global _agent_debug_smiles_logged
            if _agent_debug_smiles_logged < _AGENT_DEBUG_SMILES_CAP:
                _agent_debug_smiles_logged += 1
                _agent_debug_log(
                    "C",
                    "evaluator.py:_check_syntax_validity:smiles",
                    "smiles score snapshot",
                    {
                        "class_name": class_name,
                        "cleaned_smiles": smiles,
                        "cleaned_len": len(smiles or ""),
                        "grammar_ok": bool(eval_row.get("grammar_valid")),
                        "rdkit_ok": eval_row.get("rdkit_valid"),
                        "membership_ok": bool(eval_row.get("class_membership")),
                        "syntax_ok": bool(eval_row.get("syntax_valid")),
                        "raw_output_len": len(output or ""),
                        "log_index": _agent_debug_smiles_logged,
                    },
                )
            # endregion
            if not smiles:
                return False, []
            return bool(eval_row["syntax_valid"]), [(smiles, bool(eval_row["syntax_valid"]))]

        if self.dataset_name == "gsm_symbolic":
            # CRANE parity: grade ONLY the final <<...>> block (check_gsm_parsed),
            # not every visible span. Delegated to the benchmark logic so the
            # same semantics apply in both evaluate_sample and reevaluate_compiled_csd.
            logic = self._benchmark_logic()
            return logic.check_syntax(self, output, example)

        segments: List[Tuple[str, bool]] = []
        matches = self._extract_constrained_content(output)

        if not matches:
            return True, []

        # If the grammar can't even be built (missing file, bad grammar, bad
        # import), there is no honest value to return here -- the result can
        # only say "valid" or "invalid", and neither is true when nothing was
        # actually checked. So we let that error stop the run instead of
        # quietly reporting a fake 100% valid result.
        parser = self._get_syntax_parser(example)
        for match in matches:
            try:
                parser.parse(match.strip())
                segments.append((match, True))
            except LarkError:
                segments.append((match, False))

        all_valid = all(is_valid for _, is_valid in segments) if segments else True
        return all_valid, segments

    def _ensure_smiles_rdkit_available(self) -> None:
        logic = self._benchmark_logic()
        logic.ensure_runtime_prereqs(self)

    def _compute_smiles_aux_metrics(
        self,
        sample_outputs: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        logic = self._benchmark_logic()
        return logic.compute_aux_metrics(self, sample_outputs)

    def _benchmark_logic(self):
        from synthesis.evaluate.benchmarks.registry import get_logic

        return get_logic(self.dataset_name)

    def _extract_actual_for_example(
        self,
        scored_output: str,
        example: dict[str, Any],
    ) -> tuple[Optional[str], str, Optional[dict[str, Any]]]:
        logic = self._benchmark_logic()
        actual, answer_source, aux = logic.extract_actual(self, scored_output, example)
        return actual, answer_source, aux

    def _is_correct_for_example(
        self,
        actual: Optional[str],
        expected: str,
        example: dict[str, Any],
        aux: Optional[dict[str, Any]],
        scored_output: str,
    ) -> bool:
        logic = self._benchmark_logic()
        return bool(logic.is_correct(self, actual, expected, example, aux, scored_output))

    def _uses_hidden_chunks(self) -> bool:
        logic = self._benchmark_logic()
        return bool(logic.uses_hidden_chunks())

    def _example_syntax_pass(
        self,
        all_valid_syntax: bool,
        segments: list[tuple[str, bool]],
        used_hidden_chunk: bool,
        aux: Optional[dict[str, Any]],
    ) -> bool:
        logic = self._benchmark_logic()
        return bool(logic.example_syntax_pass(all_valid_syntax, segments, used_hidden_chunk, aux))

    def _accuracy_applicable_for_example(self, aux: Optional[dict[str, Any]]) -> bool:
        logic = self._benchmark_logic()
        return bool(logic.accuracy_applicable(aux))

    def _evaluate_one_example(
        self,
        i: int,
        example: Any,
        dataset_len: int,
        env: Dict[str, Any],
        logic: Any,
        run_crane_csd: Any,
        smiles_suffix: dict,
    ) -> Dict[str, Any]:
        """Evaluate exactly one example and return its sample dict.

        This is the unit of work shared by the sequential loop (below) and the
        data-parallel worker pool (synthesis/scripts/eval_worker_pool.py): it
        does not read or write any state beyond `smiles_suffix` (SMILES-only
        rolling-prompt carry, unused for GSM/Spider), so it is safe to call
        from a worker process on an arbitrary slice of examples in any order.
        Almost every failure is caught and returned as a sample dict rather
        than raised. The two exceptions, listed explicitly at the catch below,
        are a broken harness (ModuleNotFoundError/ImportError) and a row that
        cannot be graded at all (UngradableExample) — neither of which produced
        a measurement, so recording one would be inventing a result. It never
        decides to stop early; that decision is made by the caller
        (sequentially in-process, or post-hoc over a merged pool result in
        `_posthoc_early_stop`).
        """
        print(f"  [EVAL] Processing example {i+1}/{dataset_len}...", flush=True)
        example_start = time.time()
        if self.dataset_name == "smiles" and os.environ.get("CSD_SMILES_ROLLING_PROMPT", "1") != "0":
            _apply_smiles_rolling_suffix(example, smiles_suffix)
        prompt = self._format_prompt(example)
        expected = self._get_expected_answer(example)
        benchmark_aux: Optional[dict[str, Any]] = None
        tokenizer = env.get("tokenizer")
        generation_token_evidence: Optional[dict[str, Any]] = None
        constrained_work: Optional[int] = None
        prompt_contract: Optional[dict[str, Any]] = None
        from synthesis.evaluate.benchmarks.sql_spider.prompts import SpiderPromptRenderError
        from synthesis.evaluate.benchmarks.sql_spider.output_contract import (
            SpiderEvidenceContractError,
        )

        try:
            print(f"  [EVAL]   Running CSD strategy (max_steps={self.max_steps})...", flush=True)
            dynamic_parser = logic.build_dynamic_parser(self, env, example)
            # region agent log
            try:
                _sig = inspect.signature(run_crane_csd)
                _params = list(_sig.parameters)
                _agent_debug_log(
                    "B",
                    "evaluator.py:_evaluate_one_example:pre_run_crane_csd",
                    "run_crane_csd signature vs early_stop_on_answer kwarg",
                    {
                        "dataset_name": self.dataset_name,
                        "example_index": i,
                        "early_stop_on_answer": bool(self.early_stop_on_answer),
                        "has_early_stop_param": "early_stop_on_answer" in _sig.parameters,
                        "param_names": _params,
                    },
                )
            except Exception as _sig_exc:
                _agent_debug_log(
                    "B",
                    "evaluator.py:_evaluate_one_example:pre_run_crane_csd",
                    "failed to inspect run_crane_csd signature",
                    {"error": repr(_sig_exc)},
                )
            # endregion
            with _PerExampleTimer(self.max_seconds_per_example):
                generation_result = run_crane_csd(
                    env=env,
                    prompt_text=prompt,
                    max_steps=self.max_steps,
                    step_token_budget=self.step_token_budget,
                    grammar_file=self._get_grammar_file(),
                    dynamic_parser=dynamic_parser,
                    early_stop_on_answer=self.early_stop_on_answer,
                )
            if len(generation_result) == 6:
                output_text, token_count, gen_time, constrained_segments, helper_trace, constrained_work = generation_result
            elif len(generation_result) == 5:
                output_text, token_count, gen_time, constrained_segments, helper_trace = generation_result
                constrained_work = None
            else:
                raise ValueError(f"CSD generation returned {len(generation_result)} values; expected 5 or 6")
            generation_token_evidence = getattr(
                env.get("lm"), "_last_generation_evidence", None
            )
            prompt_contract = getattr(env.get("lm"), "_last_prompt_contract", None)
            example_time = time.time() - example_start
            print(f"  [EVAL]   Generated {token_count} tokens in {example_time:.2f}s", flush=True)
            # region agent log
            _agent_debug_log(
                "A",
                "evaluator.py:_evaluate_one_example:success",
                "generation succeeded",
                {
                    "dataset_name": self.dataset_name,
                    "example_index": i,
                    "token_count": int(token_count) if token_count is not None else None,
                    "example_time": float(example_time),
                    "gen_time": float(gen_time) if gen_time is not None else None,
                    "output_empty": not bool((output_text or "").strip()),
                    "output_len": len(output_text or ""),
                },
            )
            # endregion

            if self.dataset_name == "spider":
                # Spider CSD returns generated text only.  Treat an echoed
                # prompt as invalid output so the strict contract can reject it.
                completion = str(output_text or "")
            else:
                from synthesis.evaluate.completion_text import completion_for_scoring

                completion = completion_for_scoring(prompt, output_text)
            _print_realtime_completion(i + 1, dataset_len, completion)
            scored_output = (
                self._truncate_gsm_output(completion)
                if self.dataset_name == "gsm_symbolic"
                else completion
            )

            # Scoring can spin too (e.g. a Lark Earley parse of a
            # degenerate span), not just generation — run it under the
            # same per-example timer so a wedged example times out and
            # is recorded as a failure instead of hanging the run
            # (2+ hour wedge observed 2026-06-10 on GSM-1.5B).
            if self.dataset_name == "spider":
                self._active_generation_token_evidence = generation_token_evidence
            try:
                with _PerExampleTimer(self.max_seconds_per_example):
                    actual, answer_source, benchmark_aux = self._extract_actual_for_example(scored_output, example)
                    is_correct = self._is_correct_for_example(
                        actual,
                        expected,
                        example,
                        benchmark_aux,
                        scored_output,
                    )
            finally:
                if self.dataset_name == "spider":
                    self._active_generation_token_evidence = None

            visible_delimiters = self._contains_delimiters(scored_output)
            used_hidden_chunk = bool(constrained_segments) or any(
                event.get("helper") in EvaluationResult._CONSTRAINED_HELPERS
                for event in (helper_trace or [])
            )
            contains_delimiters = used_hidden_chunk if self._uses_hidden_chunks() else visible_delimiters

            all_valid_syntax, segments = self._check_syntax_validity(scored_output, example=example)
            # Per-example syntax pass:
            # - GSM: visible <<...>> chunks must exist and parse.
            # - SMILES: the full output is the generated molecule string.
            # - Spider: chunks are internal/hidden; visible delimiter tokens are not
            #   part of the answer contract, so count parser-governed chunk usage.
            example_syntax_pass = self._example_syntax_pass(
                all_valid_syntax,
                segments,
                used_hidden_chunk,
                benchmark_aux,
            )
            if self.dataset_name == "smiles" and os.environ.get("CSD_SMILES_ROLLING_PROMPT", "1") != "0":
                _update_smiles_rolling_suffix(
                    example,
                    smiles_suffix,
                    actual,
                    bool(benchmark_aux and benchmark_aux.get("syntax_valid")),
                )
            accuracy_applicable = self._accuracy_applicable_for_example(benchmark_aux)
            example_syntax_rate = 1.0 if example_syntax_pass else 0.0
            if self._uses_hidden_chunks() and self.dataset_name == "smiles":
                visible_span_lengths = (
                    [span_token_length(tokenizer, (benchmark_aux or {}).get("smiles", ""))]
                    if actual
                    else []
                )
                valid_visible_span_lengths = visible_span_lengths if example_syntax_pass else []
                num_valid_visible_spans = 1 if example_syntax_pass and actual else 0
                segments = [(actual or "", example_syntax_pass)] if actual else []
            else:
                visible_span_lengths = [
                    span_token_length(tokenizer, segment) for segment, _ in segments
                ]
                valid_visible_span_lengths = [
                    span_token_length(tokenizer, segment)
                    for segment, is_valid in segments
                    if is_valid
                ]
                num_valid_visible_spans = sum(1 for _, is_valid in segments if is_valid)

            if hasattr(example, "conclusion"):
                q_full = example.premises + " | " + example.conclusion
            else:
                q_full = example.get("question", str(example.get("premises", "")))
            q_str = q_full[:200]
            sample = {
                "question": q_str,
                "question_full": q_full,
                "expected": expected,
                "actual": actual if self.dataset_name == "spider" else actual or completion[:100],
                "full_output": completion,
                "scored_output": scored_output,
                "answer_source": answer_source,
                "has_extracted_answer": actual is not None if self.dataset_name == "spider" else actual is not None or answer_source == "text_fallback",
                "is_correct": is_correct,
                "accuracy_applicable": accuracy_applicable,
                "contains_delimiters": contains_delimiters,
                "visible_delimiters": visible_delimiters,
                "used_constrained_chunk": used_hidden_chunk,
                "uses_hidden_chunks": self._uses_hidden_chunks(),
                "is_syntax_valid": example_syntax_pass,
                "syntax_rate": example_syntax_rate,
                "num_visible_spans": len(segments),
                "num_valid_visible_spans": num_valid_visible_spans,
                "visible_span_token_lengths": visible_span_lengths,
                "valid_visible_span_token_lengths": valid_visible_span_lengths,
                "token_count": token_count,
                "constrained_work": constrained_work,
                "hit_max_steps": token_count >= self.max_steps,
                "time_seconds": gen_time,
                "runtime_budget_exceeded": (
                    self.max_seconds_per_example is not None
                    and gen_time > self.max_seconds_per_example
                ),
                "timed_out": False,
                "helper_trace": helper_trace,
                "generation_token_evidence": (
                    generation_token_evidence if self.dataset_name == "spider" else None
                ),
                **_strategy_sample_evidence_fields(generation_token_evidence),
                "prompt_contract": prompt_contract,
                "removed_terminal_token_count": (
                    benchmark_aux.get("removed_terminal_token_count")
                    if self.dataset_name == "spider" and benchmark_aux
                    else None
                ),
                "task_guidance": getattr(env.get("lm"), "task_guidance", None),
                "smiles_eval": benchmark_aux if self.dataset_name == "smiles" else None,
                "output_contract_valid": (
                    benchmark_aux.get("output_contract_valid")
                    if self.dataset_name == "spider" and benchmark_aux
                    else None
                ),
                "output_rejection_reason": (
                    benchmark_aux.get("output_rejection_reason")
                    if self.dataset_name == "spider" and benchmark_aux
                    else None
                ),
            }
            if self.dataset_name == "smiles":
                sample["smiles_eval"] = benchmark_aux
            return EvaluationResult._annotate_sample_observability(sample)

        except (ModuleNotFoundError, ImportError, UngradableExample):
            # These two are not results, so they must not become one.
            #
            # A missing module means nothing ran: the sample dict built below
            # records is_correct: False with accuracy_applicable: True, and
            # that sample is counted in num_accuracy_examples (the accuracy
            # denominator), so a lost file and a model that got the question
            # wrong come out as the same number. That is exactly how Spider's
            # 0% was mistaken for a model failure for weeks.
            #
            # An UngradableExample means the dataset row has no field to grade
            # against, so there is no verdict to record either.
            #
            # Everything else stays caught on purpose. A model that produced
            # nonsense, a timeout, a solver that fell over on one expression --
            # those are real outcomes of running that example, and aborting the
            # whole run over one of them loses every other example's result.
            #
            # This list is fixed at three names and is meant to stay that way;
            # tests/test_harness_failures_are_not_scored_as_wrong_answers.py
            # fails if it grows or shrinks. In particular it does not name
            # TypeError or ValueError: those are what any ordinary bug in the
            # generation path raises, and re-raising them would turn a stray
            # None into an aborted evaluation run.
            raise

        except Exception as e:
            if self.dataset_name == "spider" and isinstance(
                e, (SpiderPromptRenderError, SpiderEvidenceContractError)
            ):
                # A Spider prompt renderer/setup is part of the harness entry
                # contract. Its failure must abort the run, not become a scored
                # generation_error sample. Other datasets retain their legacy
                # per-example error surface.
                raise
            if hasattr(example, "conclusion"):
                q_full = example.premises + " | " + example.conclusion
            else:
                q_full = example.get("question", str(example.get("premises", "")))
            q_str = q_full[:200]
            elapsed = time.time() - example_start
            timed_out = isinstance(e, PerExampleTimeout)
            if timed_out:
                print(
                    f"  [EVAL]   Timed out after {elapsed:.2f}s (per-example runtime budget)",
                    flush=True,
                )
            # region agent log
            _err_msg = str(e)
            _hyp = "B" if "early_stop_on_answer" in _err_msg else ("A" if self.dataset_name != "smiles" else "E")
            _agent_debug_log(
                _hyp,
                "evaluator.py:_evaluate_one_example:except",
                "exception swallowed into empty sample",
                {
                    "dataset_name": self.dataset_name,
                    "example_index": i,
                    "exc_type": type(e).__name__,
                    "exc_message": _err_msg[:500],
                    "elapsed": float(elapsed),
                    "timed_out": bool(timed_out),
                    "full_output_empty": True,
                    "token_count": 0,
                },
            )
            # endregion
            if self.dataset_name == "spider":
                # Read the published evidence once after a late generation
                # failure; the count below must describe this same object.
                generation_token_evidence = getattr(
                    env.get("lm"), "_last_generation_evidence", None
                )
                prompt_contract = getattr(env.get("lm"), "_last_prompt_contract", None)
            sample = {
                "question": q_str,
                "question_full": q_full,
                "expected": expected,
                "actual": None,
                "full_output": "",
                "scored_output": "",
                "answer_source": "none",
                "has_extracted_answer": False,
                "is_correct": False,
                "accuracy_applicable": self._accuracy_applicable_for_example(None),
                "contains_delimiters": False,
                "visible_delimiters": False,
                "used_constrained_chunk": False,
                "uses_hidden_chunks": self._uses_hidden_chunks(),
                "is_syntax_valid": False,
                "syntax_rate": 0.0,
                "num_visible_spans": 0,
                "num_valid_visible_spans": 0,
                "visible_span_token_lengths": [],
                "valid_visible_span_token_lengths": [],
                "token_count": 0,
                "hit_max_steps": False,
                "time_seconds": elapsed,
                "runtime_budget_exceeded": timed_out or (
                    self.max_seconds_per_example is not None
                    and elapsed > self.max_seconds_per_example
                ),
                "generation_token_evidence": (
                    generation_token_evidence if self.dataset_name == "spider" else None
                ),
                **_strategy_sample_evidence_fields(generation_token_evidence),
                "prompt_contract": prompt_contract,
                "removed_terminal_token_count": (
                    len((generation_token_evidence or {}).get("removed_terminal_token_ids", ()))
                    if self.dataset_name == "spider"
                    else None
                ),
                "output_contract_valid": False if self.dataset_name == "spider" else None,
                "output_rejection_reason": "generation_error" if self.dataset_name == "spider" else None,
                "timed_out": timed_out,
                "error": str(e),
                "helper_trace": [],
                "task_guidance": getattr(env.get("lm"), "task_guidance", None),
            }
            return EvaluationResult._annotate_sample_observability(sample)

    def _evaluate_examples_sequential_with_early_stop(
        self,
        dataset: list,
        env: Dict[str, Any],
        logic: Any,
        target_min_accuracy: Optional[float],
        early_stop_min_syntax_rate: Optional[float],
        early_stop_runtime_failures: Optional[int],
        deadline: Optional[float] = None,
    ) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """Single-process path: evaluate examples one at a time, checking the
        early-stop conditions after each one (unchanged behavior from before
        the worker-pool refactor). Used for SMILES always (its rolling-prompt
        state and `should_stop_collected` early stop are not poolable), and as
        the disaster fallback for GSM/Spider if the worker pool cannot be
        started or every worker dies.

        `deadline`, if given, is an absolute `time.time()` timestamp for the
        whole attempt (not just this evaluation stage). It is checked between
        examples so a strategy that loops to its max steps on every example
        cannot burn unbounded wall-clock time -- the loop stops as soon as the
        deadline has passed and returns whatever examples finished.
        """
        run_crane_csd = logic.get_generation_runner()
        early_stop_enabled = (
            target_min_accuracy is not None
            or early_stop_min_syntax_rate is not None
            or early_stop_runtime_failures is not None
        )
        collected_stop = getattr(logic, "should_stop_collected", None)
        sample_outputs: List[Dict[str, Any]] = []
        smiles_suffix: dict = {}
        n_timeouts = 0

        def early_stop_reason_if_any() -> Optional[str]:
            if callable(collected_stop) and sample_outputs:
                reason = collected_stop(sample_outputs)
                if reason:
                    return reason
            if not early_stop_enabled or not sample_outputs:
                return None
            runtime_failures = sum(
                1 for sample in sample_outputs if sample.get("runtime_budget_exceeded")
            )
            if (
                early_stop_runtime_failures is not None
                and runtime_failures >= early_stop_runtime_failures
            ):
                return (
                    "threshold-impossible early stop: "
                    f"{runtime_failures} example(s) exceeded the per-example runtime budget."
                )
            return None

        for i, example in enumerate(dataset):
            sample = self._evaluate_one_example(
                i, example, len(dataset), env, logic, run_crane_csd, smiles_suffix
            )
            sample.update(EvaluationResult._sample_identity_metadata(example, i))
            sample_outputs.append(sample)
            if sample.get("timed_out"):
                n_timeouts += 1

            if deadline is not None and time.time() >= deadline:
                reason = (
                    f"{ATTEMPT_DEADLINE_EARLY_STOP_REASON}; "
                    f"N={len(sample_outputs)} of {len(dataset)} examples completed"
                )
                print(f"  [EVAL] Early stopping eval: {reason}", flush=True)
                return sample_outputs, reason

            early_reason = early_stop_reason_if_any()
            if early_reason:
                print(f"  [EVAL] Early stopping synthesis eval: {early_reason}", flush=True)
                return sample_outputs, early_reason
            if sample.get("timed_out") and n_timeouts >= _MAX_TIMEOUTS_PATHOLOGICAL:
                reason = (
                    f"eval stopped early after {n_timeouts} timed-out examples "
                    f"(pathological-strategy guard); "
                    f"N={len(sample_outputs)} of {len(dataset)}"
                )
                print(f"  [EVAL] Early stopping eval: {reason}", flush=True)
                return sample_outputs, reason

        return sample_outputs, None

    def _posthoc_early_stop(
        self,
        sample_outputs: List[Dict[str, Any]],
        early_stop_runtime_failures: Optional[int],
    ) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """Replay the sequential early-stop decisions over an already-computed,
        original-order list of per-example results (a full pass, evaluated by
        the worker pool with no per-shard early stop). GSM/Spider carry no
        state between examples, so replaying the same cumulative counts over
        the merged list reproduces the exact point (if any) at which the
        sequential path would have stopped, and the exact reported reason.

        Only the two GSM/Spider-applicable sequential gates are replayed here
        (runtime-failure threshold, pathological-timeout guard) — the
        SMILES-only `should_stop_collected` gate does not apply to pooled
        datasets. If a future poolable dataset needs order-dependent state
        this function cannot reproduce, do not approximate it: raise instead
        (that dataset must stay off POOLABLE_DATASETS).
        """
        runtime_failures = 0
        n_timeouts = 0
        for idx, sample in enumerate(sample_outputs):
            if sample.get("runtime_budget_exceeded"):
                runtime_failures += 1
            if (
                early_stop_runtime_failures is not None
                and runtime_failures >= early_stop_runtime_failures
            ):
                reason = (
                    "threshold-impossible early stop: "
                    f"{runtime_failures} example(s) exceeded the per-example runtime budget."
                )
                print(f"{_POOL_LOG} post-hoc early stop at example {idx + 1}: {reason}", flush=True)
                return sample_outputs[: idx + 1], reason
            if sample.get("timed_out"):
                n_timeouts += 1
            if n_timeouts >= _MAX_TIMEOUTS_PATHOLOGICAL:
                reason = (
                    f"eval stopped early after {n_timeouts} timed-out examples "
                    f"(pathological-strategy guard); "
                    f"N={idx + 1} of {len(sample_outputs)}"
                )
                print(f"{_POOL_LOG} post-hoc early stop at example {idx + 1}: {reason}", flush=True)
                return sample_outputs[: idx + 1], reason
        return sample_outputs, None

    def _build_evaluation_result(
        self,
        sample_outputs: List[Dict[str, Any]],
        early_stop_reason: Optional[str],
        planned_num_examples: int,
        target_min_accuracy: Optional[float],
        early_stop_min_syntax_rate: Optional[float],
        logic: Any,
        start_time: float,
    ) -> EvaluationResult:
        """Aggregate a list of per-example sample dicts into an EvaluationResult.

        Purely a reduction over `sample_outputs` (plus the early-stop reason
        and the planned sample size) — mechanically identical to the
        `build_result` closure this replaces, just parameterized so both the
        sequential path and the pool path call the same aggregation code
        (keep-B-only: one copy of this math).
        """
        total_time = time.time() - start_time
        evaluated_count = len(sample_outputs)
        num_correct = sum(1 for s in sample_outputs if s.get("is_correct"))
        num_examples_syntax_pass = sum(1 for s in sample_outputs if s.get("is_syntax_valid"))
        num_accuracy_examples = sum(1 for s in sample_outputs if s.get("accuracy_applicable"))
        all_examples_contain_delimiters = all(
            s.get("contains_delimiters") for s in sample_outputs
        ) if sample_outputs else True
        max_sample_time = max(
            (float(sample.get("time_seconds", 0.0)) for sample in sample_outputs),
            default=0.0,
        )

        def _accuracy_upper_bound() -> float:
            remaining = max(0, planned_num_examples - evaluated_count)
            return logic.accuracy_upper_bound(
                num_correct,
                remaining,
                num_accuracy_examples,
                planned_num_examples,
            )

        denominator_basis = choose_denominator_basis(
            early_stop_reason, planned_num_examples, evaluated_count
        )
        accuracy_denominator = logic.final_accuracy_denominator(
            denominator_basis,
            num_accuracy_examples,
        )
        accuracy_definition = logic.accuracy_definition()
        invalid_excluded = logic.invalid_outputs_excluded(
            evaluated_count,
            num_accuracy_examples,
        )
        aux_metrics = self._compute_smiles_aux_metrics(sample_outputs)
        if early_stop_reason is not None:
            aux_metrics["early_stop"] = {
                "reason": early_stop_reason,
                "target_accuracy": target_min_accuracy,
                "target_syntax_rate": early_stop_min_syntax_rate,
                "max_possible_accuracy": _accuracy_upper_bound(),
                "evaluated_examples": evaluated_count,
                "total_examples": planned_num_examples,
                "remaining_examples": max(0, planned_num_examples - evaluated_count),
            }
        task_guidance = sorted({
            sample.get("task_guidance")
            for sample in sample_outputs
            if sample.get("task_guidance")
        })
        # Benchmark-specific accuracy override. Only SMILES defines this hook,
        # repurposing `accuracy` to the unique-valid RATE (CARS axis); for all
        # other datasets getattr returns None and the default formula stands.
        final_accuracy = num_correct / max(1, accuracy_denominator)
        _override = getattr(logic, "override_accuracy", None)
        if callable(_override):
            _ov = _override(aux_metrics, evaluated_count)
            if _ov is not None:
                final_accuracy = _ov
        return EvaluationResult(
            success=True,
            accuracy=final_accuracy,
            contains_delimiters=all_examples_contain_delimiters,
            syntax_rate=num_examples_syntax_pass / max(1, denominator_basis),
            num_examples=evaluated_count,
            num_correct=num_correct,
            accuracy_denominator=accuracy_denominator,
            accuracy_definition=accuracy_definition,
            invalid_outputs_excluded_from_accuracy=invalid_excluded,
            total_time_seconds=total_time,
            max_sample_time_seconds=max_sample_time,
            early_stopped=early_stop_reason is not None,
            early_stop_reason=early_stop_reason,
            planned_num_examples=planned_num_examples,
            error=early_stop_reason,
            sample_outputs=sample_outputs,
            task_guidance=task_guidance,
            aux_metrics=aux_metrics,
        )

    def evaluate_sample(
        self,
        compiled_module_path: Path,
        sample_size: Optional[int] = None,
        sample_offset: int = 0,
        min_accuracy: Optional[float] = None,
        early_stop_min_accuracy: Optional[float] = None,
        early_stop_min_syntax_rate: Optional[float] = None,
        early_stop_runtime_failures: Optional[int] = None,
        min_examples_before_threshold_stop: Optional[int] = None,
        deadline: Optional[float] = None,
    ) -> EvaluationResult:
        """
        Evaluate the compiled CSD on a sample of the dataset.

        Args:
            compiled_module_path: Path to the compiled GeneratedCSD.py module
            sample_size: Number of examples to evaluate (overrides init value)
            sample_offset: Number of leading examples to skip before taking
                sample_size examples. 0 (the default) is the normal slice A.
                The greedy hill-climb search's slice-B confirmation eval
                passes a nonzero offset (typically == sample_size) to draw
                the NEXT sample_size examples instead of re-scoring slice A
                (a no-op under deterministic greedy decoding). For GSM/Spider
                this skips the first `offset` entries of the fixed
                split-index list (see _load_gsm_split_indices /
                _load_spider_split_indices), so slice B is drawn from the
                same canonical train split, disjoint from slice A. Datasets
                with no index manifest (e.g. SMILES) have no list to offset
                into, so the confirmation slice is instead drawn by shifting
                the sample seed by `offset` -- a weaker pairing than the
                index-based slices (see planning/monotonic-search-plan.md
                §9). Reset to the passed value (default 0) on every call so
                a later normal call is never contaminated by an earlier
                confirmation-slice call.
            min_accuracy: Backward-compatible target accuracy for early stop.
            early_stop_min_accuracy: Optional target accuracy for early stop.
            early_stop_min_syntax_rate: Optional target syntax rate for early stop.
            early_stop_runtime_failures: Optional runtime-failure count for early stop.
            min_examples_before_threshold_stop: If set, the threshold-impossible
                accuracy and syntax-rate early stops are suppressed until at
                least this many examples have been evaluated. The runtime-budget
                early stop is unaffected (it is a different signal). Lets the
                synthesis feedback loop see usable data even when the strategy
                cannot possibly clear the acceptance threshold.
            deadline: Optional absolute `time.time()` timestamp. Checked between
                examples (sequential path only); if crossed, evaluation stops
                early and returns whatever examples finished, with
                `early_stop_reason` starting with ATTEMPT_DEADLINE_EARLY_STOP_REASON
                so the caller can tell this apart from a normal early stop.

        Returns:
            EvaluationResult with metrics and sample outputs
        """
        if sample_size is not None:
            self.sample_size = sample_size
        self.sample_offset = sample_offset

        # Always re-sample so each iteration gets a fresh random example
        self._dataset = None

        start_time = time.time()
        sample_outputs: List[Dict[str, Any]] = []

        # Datasets with a fixed split-index manifest (GSM/Spider) apply
        # sample_offset by skipping entries in the index list itself (see
        # _load_gsm_split_indices / _load_spider_split_indices). Datasets
        # with no index manifest have no list to offset into, so the
        # confirmation slice for non-indexed datasets is drawn by shifting
        # the sample seed instead -- restored after this call so it never
        # leaks into a later normal (offset=0) evaluation.
        has_split_manifest = (
            (self.dataset_name == "gsm_symbolic" and self.gsm_split_file is not None)
            or (self.dataset_name == "spider" and self.spider_split_file is not None)
        )
        original_sample_seed = self.sample_seed
        if self.sample_offset and not has_split_manifest:
            self.sample_seed = (self.sample_seed or 0) + self.sample_offset

        try:
            try:
                self._ensure_smiles_rdkit_available()
                dataset = self._load_dataset_sample()
                planned_num_examples = len(dataset)
                target_min_accuracy = (
                    early_stop_min_accuracy
                    if early_stop_min_accuracy is not None
                    else min_accuracy
                )
                # Cheap (registry lookup only, no engine load) -- safe to call in the
                # main process regardless of which branch below actually runs eval.
                logic = self._benchmark_logic()

                get_synthesis_eval_pool = (
                    _resolve_eval_pool_loader()
                    if self.dataset_name in POOLABLE_DATASETS
                    else None
                )

                if get_synthesis_eval_pool is not None:
                    # Do NOT call self._setup_environment() here: it loads a vLLM engine
                    # into the CALLING process. For the pooled path, each worker
                    # subprocess calls its own _setup_environment (and thus loads its own
                    # engine) once, on its own pinned GPU. Loading an engine here too
                    # would waste GPU memory in the main process and can starve/crash a
                    # worker pinned to the same GPU -- this was observed directly: the
                    # first identity-test run had the main process eagerly load an
                    # engine on GPU 0, leaving too little free memory for the pool's
                    # worker (also assigned GPU 0), which then failed immediately.
                    pool = get_synthesis_eval_pool(self)
                    sample_outputs = pool.evaluate_examples(self, compiled_module_path, dataset)
                    for evaluated_index, (example, sample) in enumerate(
                        zip(dataset, sample_outputs)
                    ):
                        sample.update(
                            EvaluationResult._sample_identity_metadata(example, evaluated_index)
                        )
                    sample_outputs, early_stop_reason = self._posthoc_early_stop(
                        sample_outputs, early_stop_runtime_failures
                    )
                else:
                    env = self._setup_environment(compiled_module_path)
                    sample_outputs, early_stop_reason = self._evaluate_examples_sequential_with_early_stop(
                        dataset,
                        env,
                        logic,
                        target_min_accuracy,
                        early_stop_min_syntax_rate,
                        early_stop_runtime_failures,
                        deadline,
                    )

                return self._build_evaluation_result(
                    sample_outputs,
                    early_stop_reason,
                    planned_num_examples,
                    target_min_accuracy,
                    early_stop_min_syntax_rate,
                    logic,
                    start_time,
                )

            except Exception as e:
                return EvaluationResult(
                    success=False,
                    accuracy=0.0,
                    contains_delimiters=False,
                    syntax_rate=0.0,
                    num_examples=0,
                    num_correct=0,
                    total_time_seconds=time.time() - start_time,
                    error=str(e),
                    sample_outputs=sample_outputs,
                    task_guidance=sorted({
                        sample.get("task_guidance")
                        for sample in sample_outputs
                        if sample.get("task_guidance")
                    }),
                )
        finally:
            self.sample_seed = original_sample_seed
