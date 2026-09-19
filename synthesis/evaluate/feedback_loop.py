"""
Main synthesis pipeline with feedback-based refinement.

Orchestrates the generate -> verify -> compile -> run loop with
iterative refinement based on errors.
"""

import copy
import json
import logging
import math
import os
import re
import secrets
from difflib import unified_diff
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Callable, Optional

from ..verify.compiler import CompilationResult, DafnyCompiler
from .evaluator import ATTEMPT_DEADLINE_EARLY_STOP_REASON, Evaluator, EvaluationResult
from ..generate.generator import StrategyGenerator
from ..generate import prompts as generation_prompts
from ..generate.rationale import extract_rationale
from ..verify.tooling import build_default_compiler, build_default_verifier
from ..verify.verifier import DafnyVerifier, VerificationResult
from synthesis.safe_logging import display_text
from synthesis.runtime_fingerprint import current_runtime_fingerprint

try:
    from synthesis.prompt_rendering import render as _render_prompt
    from synthesis.prompt_rendering.models.feedback_loop import (
        ConstraintBypassedModel,
        DelimiterMissDefaultModel,
        DelimiterMissOpenNotClosedModel,
        HintLinesModel,
        SpanNotClosedModel,
        TokenCapExhaustionModel,
    )
except ImportError:
    from prompt_rendering import render as _render_prompt
    from prompt_rendering.models.feedback_loop import (
        ConstraintBypassedModel,
        DelimiterMissDefaultModel,
        DelimiterMissOpenNotClosedModel,
        HintLinesModel,
        SpanNotClosedModel,
        TokenCapExhaustionModel,
    )


logger = logging.getLogger(__name__)


def _write_json_atomic(path: Path, payload: object) -> None:
    """Replace a JSON file only after its complete contents reach disk."""
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _delimiter_miss_hint(require_delimiters: bool, contains_delimiters: bool, sample_outputs=None) -> str:
    """Localized diagnostic when the eval required << >> spans but produced none.

    Two distinct root causes need different advice:
    1. Spans never opened: strategy waits for model to emit "<<" but weak model never does.
       Fix: force "<<" via a direct-control helper.
    2. Spans opened but never closed: strategy gets "<<" in the output but the step budget
       runs out before ">>" is produced — the span terminates the span incorrectly.
       Fix: improve how the strategy decides a span is complete and exits it.
    Returns "" when delimiters are present or not required.
    """
    if not require_delimiters or contains_delimiters:
        return ""

    # contains_delimiters is True only when ALL examples had spans (strict AND).
    # If sample data shows spans are being produced for most outputs, the "none
    # had spans" message is wrong — suppress the hint entirely so the author isn't
    # misled into forcing delimiters when they're already mostly working.
    if sample_outputs:
        n = len(sample_outputs)
        n_with = sum(1 for s in sample_outputs if s.get("contains_delimiters", False))
        if n > 0 and n_with / n >= 0.05:
            return ""

    # Determine which root cause applies by inspecting sample outputs.
    # n_open_not_closed: outputs where "<<" IS present but ">>" is absent —
    # the span was opened but the strategy never closed it.
    if sample_outputs:
        n = len(sample_outputs)
        n_open_not_closed = sum(
            1 for s in sample_outputs
            if "<<" in (s.get("full_output") or "")
            and ">>" not in (s.get("full_output") or "")
        )
        if n > 0 and n_open_not_closed >= 0.2 * n:
            # Root cause 2: spans opened but never closed.
            model = DelimiterMissOpenNotClosedModel(n_open_not_closed=n_open_not_closed, n=n)
            return _render_prompt(model, "feedback_loop/delimiter_miss_open_not_closed.j2")

    return _render_prompt(DelimiterMissDefaultModel(), "feedback_loop/delimiter_miss_default.j2")


def _token_cap_exhaustion_hint(sample_outputs, max_steps) -> str:
    """Causal diagnostic when most outputs ran all the way to the step cap.

    Flag-gated (SynthesisPipeline token_cap_feedback, default ON). Without
    this hint the author sees only counts ("Examples hitting max steps: X/N")
    with no explanation of WHY that produces wrong/invalid answers. Fires when
    >=50% of evaluated outputs hit max_steps. Names the mechanism (truncation
    mid-generation; the grader scores the last complete span standing at
    cutoff) and the decoding-level direction (terminate once the answer is
    complete) — mechanism guidance only, no task content."""
    if not sample_outputs:
        return ""
    n = len(sample_outputs)
    n_capped = sum(1 for s in sample_outputs if s.get("hit_max_steps"))
    if n_capped / n < 0.5:
        return ""
    model = TokenCapExhaustionModel(n_capped=n_capped, n=n, max_steps=max_steps)
    return _render_prompt(model, "feedback_loop/token_cap_exhaustion.j2")


def _span_not_closed_hint(require_delimiters: bool, sample_outputs) -> str:
    """Nudge when spans OPEN but never CLOSE before the step budget runs out.

    Distinct from _delimiter_miss_hint (no span opened at all): here the model
    DID emit "<<" but the strategy never brought the span to a ">>", so it burns
    the whole step budget inside one open span and the eval records no usable
    answer (accuracy/syntax collapse toward 0). Common when the strategy advances
    many constrained tokens per step with no way to recognize/terminate a complete
    span. We deliberately name only the SYMPTOM and the AREA to reconsider (how the
    strategy progresses and decides it is done inside a span) and leave the specific
    mechanism to the author — a general decoding-mechanism nudge, not task guidance.
    Returns "" when not required, no samples, or the pattern is not prevalent.
    """
    if not require_delimiters or not sample_outputs:
        return ""
    n = len(sample_outputs)
    n_unterminated = sum(
        1 for s in sample_outputs
        if "<<" in (s.get("full_output") or "")
        and ">>" not in (s.get("full_output") or "")
    )
    n_maxsteps_nodelim = sum(
        1 for s in sample_outputs
        if s.get("hit_max_steps") and not s.get("contains_delimiters", False)
    )
    n_affected = max(n_unterminated, n_maxsteps_nodelim)
    if n_affected == 0 or n_affected < 0.1 * n:
        return ""
    model = SpanNotClosedModel(n_affected=n_affected, n=n)
    return _render_prompt(model, "feedback_loop/span_not_closed.j2")


def _constraint_bypassed_hint(require_delimiters: bool, contains_delimiters: bool, sample_outputs=None) -> str:
    """Nudge when << >> spans APPEAR in the text but the constraint barely engaged.

    Distinct from the other two delimiter hints:
      - _delimiter_miss_hint  : no spans at all (delimiters absent).
      - _span_not_closed_hint : spans open but never reach ">>".
    Here the delimiters ARE present, yet the strategy's constrained branch ran on
    almost none of the examples (`used_constrained_chunk` low) — so the span content
    was produced UNCONSTRAINED and its syntax is at the raw model's mercy. This is the
    failure mode where a reactive span-entry trigger (e.g. `next == "<<"`) never matches
    because the model emits a space-prefixed delimiter token, so the constrained path is
    skipped even though "<<" still shows up in the output text.

    We name only the SYMPTOM (constraint engaged on few examples) and the AREA to
    reconsider (span ENTRY — how the strategy decides to enter the constrained branch)
    and leave the mechanism to the author. Decoding-mechanism nudge, not task guidance.
    Returns "" when not required, delimiters absent, no samples, or the field is unrecorded.
    """
    if not require_delimiters or not contains_delimiters or not sample_outputs:
        return ""
    # Only examples that (a) actually show delimiters and (b) record whether the
    # constrained branch ran can tell us whether the span content was constrained.
    relevant = [
        s for s in sample_outputs
        if s.get("contains_delimiters", False)
        and "used_constrained_chunk" in s
    ]
    n_rel = len(relevant)
    if n_rel == 0:
        return ""
    n_engaged = sum(1 for s in relevant if s.get("used_constrained_chunk"))
    n_bypassed = n_rel - n_engaged
    n = len(sample_outputs)
    if n_bypassed < 0.2 * n or n_engaged >= 0.5 * n_rel:
        return ""
    model = ConstraintBypassedModel(n_engaged=n_engaged, n_rel=n_rel, n_bypassed=n_bypassed)
    return _render_prompt(model, "feedback_loop/constraint_bypassed.j2")


def _final_span_failure_hint(require_delimiters: bool, sample_outputs=None) -> str:
    """Classify delimiter-bearing syntax failures into machine-checkable categories.

    The three categories are:
      - ``final_span_unclosed``: output has ``<<`` after the last ``>>``, so the
        final answer span opened but the generation stopped before ``>>`` was
        emitted.  (rfind('<<') > rfind('>>'))
      - ``no_span_emitted``: output has no ``<<`` at all — the model never
        started a constrained span.
      - ``final_span_invalid``: the final span closed (a complete ``<< >>``
        block exists) but the block fails the CRANE GSM syntax check (e.g.
        contains ``{``, ``}``, or ``**``).

    Returns a non-empty hint string when ``require_delimiters`` is True, there
    are sample outputs, and at least one syntax-failing example exists.
    Matches the style of ``_delimiter_miss_hint`` / ``_span_not_closed_hint``
    (symptom + area, no task-specific guidance).
    """
    if not require_delimiters or not sample_outputs:
        return ""

    # Collect syntax-failing examples (ones where the final span was expected
    # but either absent or invalid).  We look at full_output for each sample
    # that is NOT marked is_syntax_valid.
    unclosed: list[str] = []
    no_span: list[str] = []
    invalid: list[str] = []

    for s in sample_outputs:
        if s.get("is_syntax_valid"):
            continue
        full_output = s.get("full_output") or ""
        last_open = full_output.rfind("<<")
        last_close = full_output.rfind(">>")
        if last_open == -1:
            # No << at all
            no_span.append(full_output)
        elif last_open > last_close:
            # << exists but either >> is absent or << appears after the last >>
            unclosed.append(full_output)
        else:
            # Closed block exists but failed syntax — final_span_invalid
            invalid.append(full_output)

    total_classified = len(unclosed) + len(no_span) + len(invalid)
    if total_classified == 0:
        return ""

    def _tail(output: str, max_chars: int = 120) -> str:
        """Return the last ``max_chars`` characters of ``output``, trimmed."""
        trimmed = output.strip()
        if len(trimmed) <= max_chars:
            return trimmed
        return "..." + trimmed[-max_chars:]

    lines = [
        "",
        f"  Delimiter syntax-failure breakdown ({total_classified} examples):",
    ]

    if unclosed:
        lines.append(
            f"    final_span_unclosed: {len(unclosed)} example(s) — "
            f"the generation emitted `<<` at the end and then stopped "
            f"(EOS or dead-end) before producing any span content or `>>`."
        )
        for ex in unclosed[:2]:
            lines.append(f"      output tail: {_tail(ex)!r}")

    if no_span:
        lines.append(
            f"    no_span_emitted: {len(no_span)} example(s) — "
            f"no `<<` appeared anywhere in the output."
        )
        for ex in no_span[:2]:
            lines.append(f"      output tail: {_tail(ex)!r}")

    if invalid:
        lines.append(
            f"    final_span_invalid: {len(invalid)} example(s) — "
            f"a `<< >>` block was present but failed the syntax check "
            f"(e.g. contained `{{`, `}}`, `**`, or was otherwise unparseable)."
        )
        for ex in invalid[:2]:
            lines.append(f"      output tail: {_tail(ex)!r}")

    lines.append(
        "    What to reconsider: these are decoding-mechanism failures at "
        "span entry/exit — not task guidance issues. Each category points to "
        "a different span-lifecycle step to strengthen."
    )
    model = HintLinesModel(lines=lines)
    return _render_prompt(model, "feedback_loop/hint_lines.j2")


class FailureStage(Enum):
    """Stage where synthesis attempt failed."""

    SEARCH_CONTRACT = "search_contract"
    VERIFICATION = "verification"
    COMPILATION = "compilation"
    RUNTIME = "runtime"
    EVALUATION = "evaluation"
    TIMEOUT = "timeout"
    HARNESS = "harness"


WORKER_HARD_TIMEOUT_PREFIX = "[worker-hard-timeout]"


def classify_eval_failure(eval_result) -> FailureStage:
    """Tell a broken evaluator apart from a strategy that scored badly.

    A failed evaluation can mean two very different things:
      - the strategy actually ran against real examples and scored badly
        (a genuine EVALUATION failure -- useful feedback for the next
        attempt to act on), or
      - the evaluator itself never ran any examples at all, for example
        because a required Python module was missing (a HARNESS failure --
        no amount of rewriting the strategy can fix a broken harness).

    The distinguishing fact is `num_examples`: a real run, even a bad one,
    still evaluates its examples. Zero examples evaluated means nothing was
    measured, so this must be reported as a broken harness rather than a
    bad strategy.
    """
    error = getattr(eval_result, "error", None) or ""
    if error.startswith(WORKER_HARD_TIMEOUT_PREFIX):
        return FailureStage.TIMEOUT
    num_examples = getattr(eval_result, "num_examples", None)
    if not num_examples:
        return FailureStage.HARNESS
    return FailureStage.EVALUATION


def parse_strategy_type(strategy_code: str) -> dict:
    """
    Parse the generated strategy code to extract strategy type and parameters.
    Useful for research analysis comparing dynamic vs static strategies.

    Returns:
        dict with keys: strategy_name, parameters, category
    """
    import re

    # Strip any embedded rationale block so pattern matching reflects the actual Dafny statements.
    extracted = extract_rationale(strategy_code)
    strategy_code_for_match = (
        extracted.body_without_rationale.strip() if extracted.has_markers else strategy_code.strip()
    )

    # Pattern matching for each strategy type
    patterns = {
        "PureConstrainedGeneration": {
            "pattern": r"PureConstrainedGeneration|ConstrainedGeneration",
            "category": "fully_constrained",
            "comparable_to": "SynCode",
        },
        "TryUnconstrainedThenConstrained": {
            "pattern": r"TryUnconstrainedThenConstrained.*?(\d+)",
            "category": "optimistic_with_fallback",
            "comparable_to": "IterGen-like",
        },
        "HybridGeneration": {
            "pattern": r"HybridGeneration.*?(\d+)",
            "category": "interleaved",
            "comparable_to": "Novel",
        },
        "SpeculativeGeneration": {
            "pattern": r"SpeculativeGeneration.*?(\d+)",
            "category": "speculative",
            "comparable_to": "SpecDec-like",
        },
        "CraneGeneration": {
            "pattern": r"CraneGeneration",
            "category": "crane_style",
            "comparable_to": "CRANE",
        },
    }

    for name, info in patterns.items():
        match = re.search(info["pattern"], strategy_code_for_match)
        if match:
            params: dict[str, int] = {}
            if match.groups():
                if name == "TryUnconstrainedThenConstrained":
                    params["unconstrained_steps"] = int(match.group(1))
                elif name == "HybridGeneration":
                    params["interval"] = int(match.group(1))
                elif name == "SpeculativeGeneration":
                    params["window_size"] = int(match.group(1))

            return {
                "strategy_name": name,
                "parameters": params,
                "category": info["category"],
                "comparable_to": info["comparable_to"],
                "raw_code": strategy_code,
            }

    return {
        "strategy_name": "Unknown",
        "parameters": {},
        "category": "unknown",
        "comparable_to": "N/A",
        "raw_code": strategy_code,
    }


@dataclass
class SynthesisAttempt:
    """Record of a single synthesis attempt."""

    attempt_number: int
    strategy_code: str
    full_dafny_code: str
    timestamp: str

    # Results from each stage (None if stage not reached)
    verification_result: Optional[VerificationResult] = None
    compilation_result: Optional[CompilationResult] = None
    eval_result: Optional[EvaluationResult] = None

    # Failure information
    failed_at: Optional[FailureStage] = None
    error_summary: str = ""

    def succeeded(self) -> bool:
        """Check if this attempt passed verify, compile, and evaluation."""
        if self.failed_at is not None:
            return False
        return (
            self.verification_result is not None
            and self.verification_result.success
            and self.compilation_result is not None
            and self.compilation_result.success
            and self.eval_result is not None
            and self.eval_result.success
        )

    def get_strategy_analysis(self) -> dict:
        """Get parsed strategy information for research analysis."""
        return parse_strategy_type(self.strategy_code)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        strategy_analysis = self.get_strategy_analysis()
        return {
            "attempt_number": self.attempt_number,
            "strategy_code": self.strategy_code,
            "strategy_analysis": strategy_analysis,  # For research comparison
            "timestamp": self.timestamp,
            "succeeded": self.succeeded(),
            "failed_at": self.failed_at.value if self.failed_at else None,
            "error_summary": self.error_summary,
            "verification": {
                "success": self.verification_result.success if self.verification_result else None,
                "error_count": len(self.verification_result.errors) if self.verification_result else 0,
            }
            if self.verification_result
            else None,
            "compilation": {
                "success": self.compilation_result.success if self.compilation_result else None,
                "output_dir": str(self.compilation_result.output_dir)
                if self.compilation_result and self.compilation_result.output_dir
                else None,
            }
            if self.compilation_result
            else None,
            "evaluation": self.eval_result.to_dict() if self.eval_result else None,
        }


class SynthesisExhaustionError(Exception):
    """
    Raised when synthesis fails after exhausting all attempts.

    Contains detailed information about all attempts for debugging.
    """

    def __init__(
        self,
        message: str,
        attempts: list[SynthesisAttempt],
        report_path: Optional[Path] = None,
    ):
        super().__init__(message)
        self.attempts = attempts
        self.report_path = report_path

    def get_failure_summary(self) -> str:
        """Get a summary of failure patterns across attempts."""
        if not self.attempts:
            return "No attempts were made"

        lines = [f"Synthesis failed after {len(self.attempts)} attempt(s):", ""]

        # Count failures by stage
        stage_counts = {stage: 0 for stage in FailureStage}
        for attempt in self.attempts:
            if attempt.failed_at:
                stage_counts[attempt.failed_at] += 1

        lines.append("Failure breakdown by stage:")
        for stage, count in stage_counts.items():
            if count > 0:
                lines.append(f"  - {stage.value}: {count}")

        lines.append("")
        lines.append("Individual attempt summaries:")

        for attempt in self.attempts:
            status = (
                "✓ SUCCESS"
                if attempt.succeeded()
                else f"✗ Failed at {attempt.failed_at.value if attempt.failed_at else 'unknown'}"
            )
            lines.append(f"  Attempt {attempt.attempt_number}: {status}")
            if attempt.error_summary:
                # Truncate long error messages
                error_preview = attempt.error_summary[:200]
                if len(attempt.error_summary) > 200:
                    error_preview += "..."
                lines.append(f"    Error: {error_preview}")

        if self.report_path:
            lines.append("")
            lines.append(f"Full report saved to: {self.report_path}")

        model = HintLinesModel(lines=lines)
        rendered = _render_prompt(model, "feedback_loop/hint_lines.j2")
        # The original implementation joined its lines with "\n" and never added
        # a trailing newline; keep_trailing_newline on the shared Jinja
        # environment means the template's own final line terminator survives
        # rendering, so strip exactly that one trailing newline back off (same
        # idiom as CompilationResult.get_error_summary).
        if rendered.endswith("\n"):
            rendered = rendered[:-1]
        return rendered


@dataclass
class SynthesisResult:
    """Result of a successful synthesis."""

    success: bool
    strategy_code: str
    full_dafny_code: str
    compiled_module_path: Optional[Path]
    output_dir: Optional[Path]
    run_dir: Optional[Path]
    attempts: list[SynthesisAttempt]
    total_time_ms: float

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "success": self.success,
            "strategy_code": self.strategy_code,
            "compiled_module_path": str(self.compiled_module_path)
            if self.compiled_module_path
            else None,
            "output_dir": str(self.output_dir) if self.output_dir else None,
            "run_dir": str(self.run_dir) if self.run_dir else None,
            "num_attempts": len(self.attempts),
            "total_time_ms": self.total_time_ms,
        }


@dataclass
class _Incumbent:
    """The best evaluated strategy so far in the greedy hill-climb search.

    Carries the incumbent's own per-slice EvaluationResults (never just a
    scalar) so accept/reject decisions can be recomputed against slice A
    alone (stage 1) and pooled A+B (stage 2) without re-evaluating anything.
    Only slice-A numbers from this object are ever shown to the author
    (planning/monotonic-search-plan.md §6).
    """

    strategy_code: str
    attempt_number: int
    a_result: EvaluationResult
    b_result: EvaluationResult


class SynthesisPipeline:
    """
    Main pipeline for synthesizing CSD strategies.

    Orchestrates:
    1. Initial strategy generation with Qwen
    2. Dafny verification
    3. Compilation to Python
    4. Runtime testing
    5. Evaluation on dataset sample (optional)
    6. Feedback-based refinement on failure
    """

    DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent.parent / "outputs" / "generated"
    NON_PRUNABLE_HELPERS = {
        "AppendTaskGuidance",
        "UnconstrainedStep",
        "ConstrainedStep",
        "AppendConstrainedToken",
        "OpenConstrainedSpan",
        "EnterObservedConstrainedSpan",
        "CloseConstrainedSpan",
        "CloseSpanIfComplete",
        "Contains",
        "RenderPrefix",
        "GenerateLogits",
        "ChooseNextToken",
        "ChooseNextTokenUnconstrained",
        "GenerateUnconstrainedChunk",
        "MaskValidNextAndEos",
        "BoostValidNextAndEos",
        "IdToToken",
        "TokenToId",
        "TokenToIdRecursive",
        "IdToLogit",
        "TokenToLogit",
        "TokensToLogits",
        "IdsToLogits",
        "MaskToken",
        "MaskTokens",
        "MaskTokensExcept",
        "IsMasked",
        "HasUnmaskedToken",
        "IsValidPrefix",
        "IsCompletePrefix",
        "IsDeadPrefix",
        "ValidNextTokenCountUpTo",
        "ValidNextToken",
        "ValidNextTokens",
        "ParseG",
        "IsTokenValidNext",
        "ValidTokenCount",
        "DeadEndDetection",
        "TopValidCandidates",
        "RollbackConstrainedSuffix",
        "LastTokenBefore",
        "RegenerateUnitOnGroundingFailure",
        "CloseSpanWithinBudget",
        # These form one state-management API. Keeping both visible prevents
        # adaptive menu pruning from offering save without restore (or vice versa).
        "SaveLogitsSnapshot",
        "RestoreLogitsSnapshot",
    }
    PRUNABLE_HELPERS = {
        "UnconstrainedGeneration",
        "ConstrainedGeneration",
        "CraneGeneration",
        "UnconstrainedChunk",
        "ManagedStep",
        "GenerateWithManagedSpan",
        "GenerateWithPrefixAndManagedSpan",
        "ConstrainedSymbol",
        "ConstrainedSymbolInGenerated",
        "ConfidenceGatedStep",
        "SafeBoostedConstrainedStep",
        "SafePenalizedConstrainedStep",
        "SafeRepetitionPenaltyStep",
        "SafeTemperatureConstrainedStep",
        "SafeSoftConstrainedStep",
        "GroupBoostedConstrainedStep",
        "GroupHasValidMember",
        "BoostValidGroups",
        "DeadEndAvoidingStep",
        "RollbackAndRegenerate",
        "RegenerateUnitOnCheckFailure",
        "RollbackAndContinue",
        "RollbackConstrainedToComplete",
        "RollbackToCompletePrefix",
        "AdaptiveConstrainedStep",
        "AdaptiveConstrainedStepWithPenalties",
        "PenalizedConstrainedStep",
        "BoostedConstrainedStep",
        "SoftConstrainedStep",
        "BoostTokenLogits",
        "PenalizeTokenLogits",
        "SafeBoostTokenLogits",
        "SafePenalizeTokenLogits",
        "MaskTokensInPrefix",
        "GetHighestLogitToken",
        "GetLogitGap",
        "GetTopKTokens",
        "GetTokenLogit",
        "ScaleAllLogits",
        "SpeculativeConstrainedRollout",
        "RolloutConstrainedWithPenalties",
        "RepetitionPenaltyStep",
        "TemperatureConstrainedStep",
        "RollbackConstrainedSpan",
        "ExtractAfterKeyword",
        "IntersectTokenSets",
        "SubtractTokenSets",
        "RollbackToValidPrefix",
        "FlattenTokenGroups",
        "GroupContaining",
        "PrefixToString",
        "ExtractContentBetweenDelimiters",
        "CountSubstring",
        "CountTokenOccurrences",
        "OccurrencesInRange",
        "TokensSinceLastOccurrence",
        "PrefixAppearsInPrompt",
        "PrefixResemblesPromptExamples",
    }
    def __init__(
        self,
        evaluator: Evaluator,
        generator: Optional[StrategyGenerator] = None,
        verifier: Optional[DafnyVerifier] = None,
        compiler: Optional[DafnyCompiler] = None,
        max_iterations: int = 5,
        output_dir: Optional[Path] = None,
        save_reports: bool = True,
        # Evaluation thresholds
        min_accuracy: float = 0.0,
        min_syntax_rate: float = 0.0,
        # Split the accuracy bar was measured on (see synthesis/split_provenance.py);
        # recorded in run_configuration so cross-split comparisons are auditable.
        bar_split_name: Optional[str] = None,
        require_delimiters: bool = True,
        eval_sample_size: int = 10,
        eval_max_seconds_per_example: Optional[float] = 90.0,
        min_examples_before_threshold_stop: Optional[int] = 15,
        max_attempt_seconds: Optional[float] = 3600.0,
        adaptive_helper_mask: bool = True,
        helper_selection_policy: str = "bandit",
        helper_mask_min_evals: int = 4,
        helper_mask_min_uses: int = 2,
        helper_mask_margin: float = 0.25,
        helper_mask_max_disabled: int = 6,
        helper_bandit_min_evals: int = 3,
        helper_bandit_top_k: int = 12,
        helper_bandit_ucb_c: float = 0.35,
        helper_bandit_explore_untried: int = 1,
        refinement_beam_size: int = 2,
        local_neighborhood_refinement: bool = True,
        max_local_edit_ratio: float = 0.65,
        beam_verify_candidates: bool = True,
        token_cap_feedback: bool = True,
    ):
        """
        Initialize the synthesis pipeline.

        Args:
            evaluator: Evaluator for dataset-based feedback (required)
            generator: Strategy generator (creates default if None)
            verifier: Dafny verifier (creates default if None)
            compiler: Dafny compiler (creates default if None)
            max_iterations: Maximum refinement iterations
            output_dir: Directory for outputs and reports
            save_reports: Whether to write the incremental progress report
                (progress_report.json) after each attempt. The end-of-run
                failure report is written either way -- it is the audit trail
                for the run, so turning it off would mean a finished run left
                no record of what it did.
            min_accuracy: Minimum accuracy threshold for evaluation
            min_syntax_rate: Minimum syntax validity rate threshold
            require_delimiters: Whether evaluated outputs must contain << >> spans
            eval_sample_size: Number of examples to evaluate on
            eval_max_seconds_per_example: Optional runtime budget per example in seconds
            min_examples_before_threshold_stop: Minimum number of examples that
                must be evaluated before threshold-impossible early stops can
                fire. Decouples the synthesis feedback budget from the
                acceptance threshold so the synthesizer always sees a usable
                amount of evaluation data. None means no minimum (legacy).
            max_attempt_seconds: Wall-clock cap, in seconds, on a single
                attempt's evaluation stage (the part that can spin -- a bad
                strategy looping to its max steps on every example). Checked
                between sequential examples and again after evaluation. A
                partial result is recorded as FailureStage.TIMEOUT. A complete
                result remains eligible to guide search but cannot end the run
                or become an accuracy-only fallback. Defaults to 3600s (1
                hour); pass None to disable (no cap -- not recommended).
            adaptive_helper_mask: Enable empirical helper pruning contract
            helper_selection_policy: Helper selection policy (`bandit` only; UCB-style)
            helper_mask_min_evals: Evaluated attempts before pruning can start
            helper_mask_min_uses: Minimum helper usage count before pruning
            helper_mask_margin: Margin below run-wide mean utility to prune a helper
            helper_mask_max_disabled: Maximum helpers disabled in one run
            helper_bandit_min_evals: Evaluated attempts before bandit selection starts
            helper_bandit_top_k: Number of prunable helpers to keep active under bandit
            helper_bandit_ucb_c: UCB exploration coefficient
            helper_bandit_explore_untried: Number of unseen helpers to force-explore
            refinement_beam_size: Number of refinement candidates to sample per step
            local_neighborhood_refinement: Prefer local edits during refinement
            max_local_edit_ratio: Soft bound on changed-line ratio for local edits
            beam_verify_candidates: Verify beam candidates before selecting one
            token_cap_feedback: Append the token-cap exhaustion hint to author
                feedback when most outputs ran to the maxSteps cap.
        """
        self.evaluator = evaluator
        self.generator = generator or StrategyGenerator()
        self.verifier = verifier or build_default_verifier()
        self.compiler = compiler or build_default_compiler()
        self.max_iterations = max_iterations
        self.output_dir = output_dir or self.DEFAULT_OUTPUT_DIR
        self.save_reports = save_reports
        # Greedy hill-climb search: the single best evaluated strategy so far
        # (planning/monotonic-search-plan.md). Every refinement is based on
        # this, never on a rejected candidate. Replaces the old Pareto-anchor
        # bookkeeping outright.
        self._incumbent: "_Incumbent | None" = None

        self.min_accuracy = min_accuracy
        self.min_syntax_rate = min_syntax_rate
        self.bar_split_name = bar_split_name
        self.require_delimiters = require_delimiters
        self.eval_sample_size = eval_sample_size
        self.eval_max_seconds_per_example = eval_max_seconds_per_example
        self.min_examples_before_threshold_stop = min_examples_before_threshold_stop
        self.max_attempt_seconds = max_attempt_seconds
        self.adaptive_helper_mask = adaptive_helper_mask
        normalized_policy = helper_selection_policy.strip().lower()
        if normalized_policy != "bandit":
            raise ValueError(
                "helper_selection_policy must be 'bandit' (UCB/bandit only)"
            )
        self.helper_selection_policy = normalized_policy
        self.helper_mask_min_evals = max(1, helper_mask_min_evals)
        self.helper_mask_min_uses = max(1, helper_mask_min_uses)
        self.helper_mask_margin = helper_mask_margin
        self.helper_mask_max_disabled = max(0, helper_mask_max_disabled)
        self.helper_bandit_min_evals = max(1, helper_bandit_min_evals)
        self.helper_bandit_top_k = max(1, helper_bandit_top_k)
        self.helper_bandit_ucb_c = max(0.0, helper_bandit_ucb_c)
        self.helper_bandit_explore_untried = max(0, helper_bandit_explore_untried)
        self.refinement_beam_size = max(1, refinement_beam_size)
        self.local_neighborhood_refinement = local_neighborhood_refinement
        self.max_local_edit_ratio = max(0.0, max_local_edit_ratio)
        self.beam_verify_candidates = beam_verify_candidates
        # Always on since 2026-07-17: append _token_cap_exhaustion_hint to the
        # author feedback when most outputs ran to the maxSteps cap.
        self.token_cap_feedback = token_cap_feedback
        self._helper_universe = self._extract_helper_universe_from_prompts()
        from synthesis.source_snapshot import execution_source_sha256

        self._execution_source_sha256 = execution_source_sha256(
            Path(__file__).resolve().parents[2]
        )

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _force_open_span(self) -> bool:
        """Whether the runtime opens this benchmark's span itself, so generation
        starts with a `<<` already in the output and the strategy already inside
        the span. Passed to the author's prompt so it states the right starting
        point. Falls back to False if the benchmark doesn't declare it, or isn't
        registered at all.

        This value only chooses a sentence of wording in the author's prompt,
        so it must never be able to end a run. get_logic() raises for a dataset
        it doesn't know, so that case falls back here instead of escaping."""
        from .benchmarks.registry import get_logic

        dataset_name = self.evaluator.dataset_name
        try:
            logic = get_logic(dataset_name)
        except Exception as exc:
            print(
                f"[surface] No benchmark registered for {dataset_name!r} "
                f"({type(exc).__name__}: {exc}); telling the author it is on "
                "the visible-delimiter surface."
            )
            return False

        hook = getattr(logic, "force_open_span", None)
        return bool(hook()) if hook is not None else False

    def _git_commit_hash(self) -> str:
        """Best-effort repo commit hash for run provenance; 'unknown' if unavailable."""
        import subprocess

        try:
            repo_root = Path(__file__).resolve().parent.parent.parent
            return (
                subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=repo_root, stderr=subprocess.DEVNULL
                )
                .decode()
                .strip()
            )
        except Exception:
            return "unknown"

    def _run_configuration_metadata(self, task_description: str, output_name: str) -> dict:
        """Return run-level provenance that is useful for experiment analysis."""
        evaluator = self.evaluator
        generator = self.generator
        return {
            "task_description": task_description,
            "output_name": output_name,
            "git_commit": self._git_commit_hash(),
            "execution_source_sha256": self._execution_source_sha256,
            "python_runtime": current_runtime_fingerprint(),
            "max_iterations": self.max_iterations,
            "thresholds": {
                "min_accuracy": self.min_accuracy,
                "min_syntax_rate": self.min_syntax_rate,
                "require_delimiters": self.require_delimiters,
            },
            "author_model": {
                "backend": getattr(generator, "backend", None),
                "model": getattr(generator, "model_name", None),
                "max_new_tokens": getattr(generator, "max_new_tokens", None),
                "reasoning_budget_tokens": getattr(
                    generator, "reasoning_budget_tokens", None
                ),
                "anthropic_thinking": getattr(generator, "anthropic_thinking", None),
                "anthropic_effort": getattr(generator, "anthropic_effort", None),
                "anthropic_thinking_display": getattr(
                    generator, "anthropic_thinking_display", None
                ),
                "route": getattr(
                    generator, "author_route_identity", lambda: None
                )(),
            },
            "evaluation": {
                "dataset": getattr(evaluator, "dataset_name", None),
                "eval_model": getattr(evaluator, "model_name", None),
                "eval_backend": getattr(evaluator, "backend", None),
                "eval_sample_size": self.eval_sample_size,
                "eval_max_steps": getattr(evaluator, "max_steps", None),
                "eval_step_token_budget": getattr(evaluator, "step_token_budget", None),
                "eval_max_seconds_per_example": self.eval_max_seconds_per_example,
                "eval_seed": getattr(evaluator, "sample_seed", None),
                "smiles_classes": getattr(evaluator, "smiles_classes", None),
                "min_examples_before_threshold_stop": self.min_examples_before_threshold_stop,
                # Which split file/side this run evaluated on, plus the declared
                # split of the accuracy bar — absence of this field is what made
                # the 2026-07-17 train-vs-eval mismatch unrecoverable from reports.
                "split_provenance": evaluator.split_provenance(self.bar_split_name),
            },
            "synthesis_controls": {
                "adaptive_helper_mask": self.adaptive_helper_mask,
                "helper_selection_policy": self.helper_selection_policy,
                "helper_bandit_min_evals": self.helper_bandit_min_evals,
                "helper_bandit_top_k": self.helper_bandit_top_k,
                "helper_bandit_ucb_c": self.helper_bandit_ucb_c,
                "helper_bandit_explore_untried": self.helper_bandit_explore_untried,
                "refinement_beam_size": self.refinement_beam_size,
                "local_neighborhood_refinement": self.local_neighborhood_refinement,
                "max_local_edit_ratio": self.max_local_edit_ratio,
                "beam_verify_candidates": self.beam_verify_candidates,
            },
        }

    def _unload_evaluator_runtime_before_refinement(self) -> None:
        """Release evaluator GPU runtime only when the author model also needs it."""
        generator_backend = getattr(self.generator, "backend", "")
        if generator_backend not in {"huggingface", "vllm"}:
            print(
                "  Evaluator runtime kept warm; generation backend is hosted "
                f"({generator_backend or 'unknown'})"
            )
            return
        self.evaluator.unload_runtime()
        print("  Evaluator runtime unloaded to free GPU memory")

    def _get_recent_behavioral_context(self, attempts: list[SynthesisAttempt]) -> str:
        """Return a compact behavior summary from the most recent evaluated attempt."""
        for attempt in reversed(attempts):
            if attempt.eval_result is not None:
                summary = attempt.eval_result.get_behavioral_context_summary(
                    require_delimiters=self.require_delimiters
                )
                if summary:
                    return summary
        return ""

    def _get_verification_history_summary(self, attempts: list[SynthesisAttempt], max_attempts: int = 3) -> str:
        """Summarize recent verification failures so refinement can avoid oscillation."""
        recent_failures = [
            attempt
            for attempt in attempts
            if attempt.failed_at == FailureStage.VERIFICATION and attempt.verification_result is not None
        ][-max_attempts:]
        if not recent_failures:
            return ""

        lines = []
        for attempt in recent_failures:
            diagnostics = attempt.verification_result.diagnostics or []
            if not diagnostics:
                preview = attempt.error_summary.splitlines()[0] if attempt.error_summary else "Unknown verification failure"
                lines.append(f"Attempt {attempt.attempt_number}: {preview}")
                continue

            primary = diagnostics[0]
            line = f"Attempt {attempt.attempt_number}: {primary.obligation_kind} at line {primary.line}"
            if primary.call_name:
                line += f" around {primary.call_name}(...)"
            if primary.failing_text:
                line += f" | failing code: {primary.failing_text}"
            if primary.related_file and primary.related_line:
                line += f" | related contract: {Path(primary.related_file).name}:{primary.related_line}"
            lines.append(line)
        model = HintLinesModel(lines=lines)
        rendered = _render_prompt(model, "feedback_loop/hint_lines.j2")
        # See get_failure_summary above: strip the one trailing newline the
        # template's keep_trailing_newline behavior adds back, to match the
        # original "\n".join(lines) idiom exactly.
        if rendered.endswith("\n"):
            rendered = rendered[:-1]
        return rendered

    @staticmethod
    def _remove_marked_comment_block(text: str, begin_marker: str, end_marker: str) -> str:
        """Remove a generated comment block while leaving the strategy code intact."""
        lines = text.splitlines()
        output: list[str] = []
        i = 0
        while i < len(lines):
            if lines[i].strip() == begin_marker:
                j = i + 1
                while j < len(lines) and lines[j].strip() != end_marker:
                    j += 1
                if j < len(lines):
                    i = j + 1
                    continue
            output.append(lines[i])
            i += 1
        return "\n".join(output).strip()

    def _get_strategy_body_for_evaluation_history(self, strategy_code: str) -> str:
        """Return strategy code without rationale/proof-sketch prose."""
        body = extract_rationale(strategy_code).body_without_rationale
        body = self._remove_marked_comment_block(
            body,
            "// CSD_PROOF_SKETCH_BEGIN",
            "// CSD_PROOF_SKETCH_END",
        )
        return body.strip()

    def _get_helper_calls_for_evaluation_history(self, strategy_code: str) -> list[str]:
        """Return model-facing helper/CSDHelpers calls used by a strategy body."""
        body = self._get_strategy_body_for_evaluation_history(strategy_code)
        calls = re.findall(r"\b(?:helpers|CSDHelpers)\.([A-Za-z_][A-Za-z0-9_]*)\s*\(", body)
        return sorted(set(calls))

    @staticmethod
    def _extract_helper_universe_from_prompts() -> set[str]:
        """Extract helper method names referenced in the prompt tool API docs."""
        prompt_helper_names = getattr(generation_prompts, "_ALL_HELPER_NAMES", None)
        if prompt_helper_names is not None:
            return set(prompt_helper_names)
        tool_reference = getattr(
            generation_prompts,
            "TOOL_REFERENCE",
            generation_prompts.SYSTEM_PROMPT,
        )
        calls = re.findall(
            r"\b(?:helpers|CSDHelpers)\.([A-Za-z_][A-Za-z0-9_]*)\b",
            tool_reference,
        )
        return set(calls)

    def _evaluation_scalar_score(self, result: EvaluationResult) -> float:
        """Scalar score used for helper utility estimates.

        Accuracy is weighted 3x so it dominates the two secondary terms, which
        are each scaled to 0.25. Runtime is graded as the fraction of examples
        that stayed within the per-example budget (not a binary 0/1), so a
        single slow example cannot sink an otherwise-strong attempt below a
        low-accuracy-but-fast one.
        """
        delimiter_score = 1.0 if (result.contains_delimiters or not self.require_delimiters) else 0.0

        if self.eval_max_seconds_per_example is None:
            runtime_frac = 1.0
        else:
            samples = getattr(result, "sample_outputs", None) or []
            if not samples:
                runtime_frac = 1.0
            else:
                within = sum(
                    1 for s in samples if not s.get("runtime_budget_exceeded", False)
                )
                runtime_frac = within / len(samples)

        return (
            3.0 * result.accuracy
            + result.syntax_rate
            + 0.25 * delimiter_score
            + 0.25 * runtime_frac
        )

    def _compute_prunable_helper_marginals(
        self,
        evaluated_attempts: list[SynthesisAttempt],
    ) -> dict[str, tuple[float, int, bool]]:
        """Counterfactual marginal credit for each prunable helper.

        For each prunable helper that appeared in at least one attempt, credit
        it by `mean(scalar | helper used) - mean(scalar | helper NOT used)`.
        This isolates the helper's own contribution instead of appending the
        full composite scalar of every attempt it co-occurred in (which let a
        slow/bad co-occurring helper drag down a helper that actually drove
        accuracy).

        Returns {helper: (marginal, n_with, ubiquitous)} where:
          - marginal: with_mean - without_mean (0.0 when ubiquitous)
          - n_with: number of attempts that used the helper (UCB pull count)
          - ubiquitous: True when EVERY attempt used the helper (no "without"
            set; marginal is undefined, so the bandit protects rather than
            prunes it)
        Untried prunable helpers (n_with == 0) are omitted; the bandit keeps
        every untried helper on the menu separately.
        """
        per_attempt: list[tuple[float, set[str]]] = []
        for attempt in evaluated_attempts:
            score = self._evaluation_scalar_score(attempt.eval_result)
            used = set(self._get_helper_calls_for_evaluation_history(attempt.strategy_code))
            per_attempt.append((score, used))

        marginals: dict[str, tuple[float, int, bool]] = {}
        for helper in (self.PRUNABLE_HELPERS & self._helper_universe):
            with_scores = [s for s, used in per_attempt if helper in used]
            without_scores = [s for s, used in per_attempt if helper not in used]
            if not with_scores:
                continue
            if not without_scores:
                marginals[helper] = (0.0, len(with_scores), True)
                continue
            with_mean = sum(with_scores) / len(with_scores)
            without_mean = sum(without_scores) / len(without_scores)
            marginals[helper] = (with_mean - without_mean, len(with_scores), False)
        return marginals

    def _pareto_best_prunable_helpers(
        self,
        evaluated_attempts: list[SynthesisAttempt],
        prunable_pool: list[str],
    ) -> set[str]:
        """Prunable helpers used by the incumbent (refinement-anchor) attempt.

        Seeds the bandit's keep-set from the SAME incumbent the author is told
        to beat, so the protected helpers belong to the branch being chased
        rather than to whichever attempt happened to win the composite scalar.
        Falls back to the scalar-best attempt when there is no incumbent yet
        (e.g. nothing evaluated on >=1 example yet).
        """
        incumbent_n = self._incumbent.attempt_number if self._incumbent else None
        best_attempt = next(
            (a for a in evaluated_attempts if a.attempt_number == incumbent_n),
            None,
        )
        if best_attempt is None:
            best_attempt = max(
                evaluated_attempts,
                key=lambda attempt: self._evaluation_scalar_score(attempt.eval_result),
            )
        best_helpers = set(self._get_helper_calls_for_evaluation_history(best_attempt.strategy_code))
        return best_helpers & set(prunable_pool)

    def _compute_allowed_helpers_utility(
        self,
        evaluated_attempts: list[SynthesisAttempt],
    ) -> tuple[list[str], str]:
        """With-vs-without comparative helper pruning.

        For each prunable helper, compare the mean evaluation score of
        attempts that USED the helper to the mean score of attempts that DID
        NOT use it. Prune the helper when `without_mean - with_mean >=
        helper_mask_margin` — i.e., not using the helper is empirically
        better than using it by at least the margin.

        This replaces the old "compare with-mean to run-wide baseline-mean"
        rule, which had a blind spot: if a harmful helper was used in MOST
        attempts, its with-mean and the baseline-mean would be very close
        (because the baseline mean was itself dragged down by the helper),
        so the helper never crossed the pruning threshold even when removing
        it would have improved scores. The with-vs-without rule isolates the
        helper's marginal effect and surfaces ubiquitous-but-harmful helpers
        as well as occasional-but-harmful ones.
        """
        allowed_helpers = set(self._helper_universe)
        if len(evaluated_attempts) < self.helper_mask_min_evals:
            return sorted(allowed_helpers), (
                f"helper mask warm-up ({len(evaluated_attempts)}/{self.helper_mask_min_evals} evaluated attempts)"
            )

        best_attempt = max(
            evaluated_attempts,
            key=lambda attempt: self._evaluation_scalar_score(attempt.eval_result),
        )
        best_helpers = set(self._get_helper_calls_for_evaluation_history(best_attempt.strategy_code))

        # Precompute (score, used_helpers) for every evaluated attempt once.
        per_attempt: list[tuple[float, set[str]]] = []
        for attempt in evaluated_attempts:
            score = self._evaluation_scalar_score(attempt.eval_result)
            used = set(self._get_helper_calls_for_evaluation_history(attempt.strategy_code))
            per_attempt.append((score, used))

        # Universe of prunable helpers: prunable ∩ helper_universe, minus
        # NON_PRUNABLE (locked-on protections) and minus the helpers used by
        # the current best attempt (never prune what's working).
        prunable_candidates = (
            (self.PRUNABLE_HELPERS & self._helper_universe)
            - self.NON_PRUNABLE_HELPERS
            - best_helpers
        )

        harmful_helpers: list[tuple[float, int, int, str, float, float]] = []
        for helper in prunable_candidates:
            with_scores = [s for s, used in per_attempt if helper in used]
            without_scores = [s for s, used in per_attempt if helper not in used]
            if len(with_scores) < self.helper_mask_min_uses:
                # Not enough data on the "with" side to make a call.
                continue
            if not without_scores:
                # Every attempt used this helper; with-vs-without is
                # undefined. Keep helper enabled and wait for opus to drop it.
                continue
            with_mean = sum(with_scores) / len(with_scores)
            without_mean = sum(without_scores) / len(without_scores)
            delta = without_mean - with_mean  # positive = harmful
            if delta >= self.helper_mask_margin:
                # Sort key: most harmful (largest delta) first, then more
                # "with" usages (stronger evidence), then alphabetical.
                harmful_helpers.append(
                    (-delta, -len(with_scores), -len(without_scores), helper, with_mean, without_mean)
                )

        harmful_helpers.sort()
        disabled: list[str] = []
        disabled_with_evidence: list[str] = []
        for neg_delta, _n_with, _n_without, helper, with_mean, without_mean in harmful_helpers:
            if len(disabled) >= self.helper_mask_max_disabled:
                break
            disabled.append(helper)
            delta = -neg_delta
            disabled_with_evidence.append(
                f"{helper} (with={with_mean:.2f}, without={without_mean:.2f}, "
                f"Δ=+{delta:.2f})"
            )

        allowed_helpers -= set(disabled)
        if disabled:
            status = (
                "helper mask active (with-vs-without utility); disabled: "
                + ", ".join(disabled_with_evidence)
            )
        else:
            status = (
                "helper mask active (with-vs-without utility); "
                "no helpers showed a harmful with-vs-without delta yet"
            )
        return sorted(allowed_helpers), status

    def _compute_allowed_helpers_bandit(
        self,
        evaluated_attempts: list[SynthesisAttempt],
    ) -> tuple[list[str], str]:
        """Bandit-style helper selection using UCB over prunable helpers."""
        allowed_helpers = set(self._helper_universe)
        if len(evaluated_attempts) < self.helper_bandit_min_evals:
            return sorted(allowed_helpers), (
                f"helper mask warm-up ({len(evaluated_attempts)}/{self.helper_bandit_min_evals} evaluated attempts for bandit)"
            )

        prunable_pool = sorted(self.PRUNABLE_HELPERS & self._helper_universe)
        if not prunable_pool:
            return sorted(allowed_helpers), "helper bandit active; no prunable helpers in universe"

        marginals = self._compute_prunable_helper_marginals(evaluated_attempts)
        pulls = {helper: (marginals[helper][1] if helper in marginals else 0) for helper in prunable_pool}
        # UCB exploitation term is each helper's counterfactual marginal
        # (with-minus-without), not the absolute mean scalar of attempts using
        # it -- so a helper is not credited or penalised for what its
        # co-occurring helpers did.
        means = {helper: (marginals[helper][0] if helper in marginals else 0.0) for helper in prunable_pool}
        ubiquitous = {helper for helper in prunable_pool if helper in marginals and marginals[helper][2]}

        # Protect the helpers used by the pareto-best (refinement-anchor)
        # attempt -- the same branch the author is told to beat -- rather than
        # whichever attempt won the composite scalar.
        keep_prunable = set(self._pareto_best_prunable_helpers(evaluated_attempts, prunable_pool))
        # Helpers used by EVERY attempt have no counterfactual signal; protect
        # them from pruning rather than ranking them out on exploration alone.
        keep_prunable.update(ubiquitous)

        total_pulls = max(1, sum(pulls.values()))
        # Keep EVERY untried helper on the menu each iteration. An arm with zero
        # pulls has no evidence against it, so pruning it would hide a helper the
        # author never had the chance to try (the old policy kept only the single
        # alphabetically-first untried helper, which froze the rest out for the
        # whole run in low-adoption cases). The mask still prunes helpers that HAVE
        # been tried and did worse -- see the UCB ranking + target_size below --
        # it just never prunes untried ones. (User ruling 2026-06-19.)
        untried = [helper for helper in prunable_pool if pulls[helper] == 0 and helper not in keep_prunable]
        keep_prunable.update(untried)

        ranked_tried = sorted(
            (helper for helper in prunable_pool if pulls[helper] > 0 and helper not in keep_prunable),
            key=lambda helper: (
                means[helper]
                + self.helper_bandit_ucb_c
                * math.sqrt(math.log(total_pulls + 1.0) / pulls[helper])
            ),
            reverse=True,
        )

        target_size = min(max(self.helper_bandit_top_k, len(keep_prunable)), len(prunable_pool))
        for helper in ranked_tried:
            if len(keep_prunable) >= target_size:
                break
            keep_prunable.add(helper)

        disabled = sorted(set(prunable_pool) - keep_prunable)
        allowed_helpers -= set(disabled)
        n_untried_kept = sum(1 for helper in keep_prunable if pulls[helper] == 0)
        status = (
            "helper mask active (bandit/UCB); "
            f"kept {len(keep_prunable)}/{len(prunable_pool)} prunable helpers "
            f"(top_k={self.helper_bandit_top_k}, all {n_untried_kept} untried kept, "
            "only tried-and-worse helpers pruned)"
        )
        return sorted(allowed_helpers), status

    def _shortfall(self, accuracy: float, syntax_rate: float) -> float:
        """Distance to both bars; 0.0 once both are met (no credit above a bar)."""
        return max(0.0, self.min_accuracy - accuracy) + max(0.0, self.min_syntax_rate - syntax_rate)

    def _is_complete_evaluation(self, result: EvaluationResult) -> bool:
        """Return whether ``result`` contains the whole planned sample.

        A pooled evaluator can return every example after the outer attempt
        budget has elapsed. That is still a complete measurement and must stay
        eligible for the search. Partial and early-stopped results remain
        ineligible.
        """
        if not result.success:
            return False
        planned = result.planned_num_examples or self.eval_sample_size
        observed = result.num_examples or 0
        samples = result.sample_outputs or []
        if result.early_stopped:
            return False
        return (
            planned > 0
            and observed == planned
            and len(samples) == planned
            and all(isinstance(sample, dict) for sample in samples)
        )

    def _restore_incumbent_from_attempts(
        self, attempts: list[SynthesisAttempt]
    ) -> None:
        """Rebuild the greedy incumbent from restored slice-A evaluations."""
        self._incumbent = None
        for attempt in attempts:
            result = attempt.eval_result
            complete_evaluation = bool(
                result is not None and self._is_complete_evaluation(result)
            )
            complete_timeout = bool(
                complete_evaluation
                and attempt.failed_at is FailureStage.TIMEOUT
            )
            if (
                not complete_evaluation
                or (
                    attempt.failed_at not in {None, FailureStage.EVALUATION}
                    and not complete_timeout
                )
            ):
                continue
            if complete_timeout:
                logger.warning(
                    "[warm-resume] preserving complete over-budget evaluation "
                    "attempt=%d examples=%d planned=%d",
                    attempt.attempt_number,
                    result.num_examples or 0,
                    result.planned_num_examples or self.eval_sample_size,
                )
            candidate_accuracy = result.accuracy or 0.0
            candidate_syntax = result.syntax_rate or 0.0
            candidate_shortfall = self._shortfall(
                candidate_accuracy, candidate_syntax
            )
            if self._incumbent is not None:
                incumbent_result = self._incumbent.a_result
                incumbent_accuracy = incumbent_result.accuracy or 0.0
                incumbent_syntax = incumbent_result.syntax_rate or 0.0
                incumbent_shortfall = self._shortfall(
                    incumbent_accuracy, incumbent_syntax
                )
                if not self._should_accept(
                    candidate_shortfall,
                    candidate_accuracy,
                    candidate_syntax,
                    incumbent_shortfall,
                    incumbent_accuracy,
                    incumbent_syntax,
                ):
                    continue
            self._incumbent = _Incumbent(
                strategy_code=attempt.strategy_code,
                attempt_number=attempt.attempt_number,
                a_result=result,
                b_result=result,
            )
        if self._incumbent is not None:
            result = self._incumbent.a_result
            logger.warning(
                "[warm-resume] restored incumbent attempt=%d accuracy=%.6f "
                "syntax_rate=%.6f evaluated_history=%d",
                self._incumbent.attempt_number,
                result.accuracy or 0.0,
                result.syntax_rate or 0.0,
                sum(
                    1
                    for attempt in attempts
                    if attempt.eval_result is not None
                    and attempt.eval_result.success
                ),
            )

    def _should_accept(
        self,
        cand_shortfall: float,
        cand_acc: float,
        cand_syn: float,
        inc_shortfall: float,
        inc_acc: float,
        inc_syn: float,
    ) -> bool:
        """Greedy hill-climb acceptance rule (plan §3).

        Accept iff shortfall strictly decreases, or shortfall is equal and
        (accuracy, syntax_rate) is lexicographically higher. Everything else,
        including an exact tie, rejects.
        """
        if cand_shortfall < inc_shortfall:
            return True
        if cand_shortfall > inc_shortfall:
            return False
        return (cand_acc, cand_syn) > (inc_acc, inc_syn)

    def _pooled_metrics(
        self, a: EvaluationResult, b: EvaluationResult
    ) -> tuple[float, float]:
        """Example-count-weighted average of (accuracy, syntax_rate) over two
        eval results (plan §2 stage 2). Used only for the internal accept
        decision -- pooled numbers never reach the author (plan §6)."""
        a_n = a.num_examples or 0
        b_n = b.num_examples or 0
        total_n = a_n + b_n
        if total_n == 0:
            return 0.0, 0.0
        pooled_acc = ((a.accuracy or 0.0) * a_n + (b.accuracy or 0.0) * b_n) / total_n
        pooled_syn = ((a.syntax_rate or 0.0) * a_n + (b.syntax_rate or 0.0) * b_n) / total_n
        return pooled_acc, pooled_syn

    @staticmethod
    def _exceeded_attempt_budget(result: EvaluationResult) -> bool:
        attempt_budget = (result.aux_metrics or {}).get(
            "attempt_runtime_budget", {}
        )
        return bool(
            isinstance(attempt_budget, dict)
            and attempt_budget.get("over_budget")
        )

    def _meets_threshold(self, result: EvaluationResult) -> bool:
        if self._exceeded_attempt_budget(result):
            return False
        return result.meets_threshold(
            min_accuracy=self.min_accuracy,
            min_syntax_rate=self.min_syntax_rate,
            require_delimiters=self.require_delimiters,
        )

    def _validate_fixed_warm_continuation(
        self,
        *,
        initial_strategy_code: str | None,
        initial_attempt_offset: int,
        attempts: list[SynthesisAttempt],
    ) -> None:
        """Fail closed before the fixed two-attempt Opus continuation starts."""
        if not attempts:
            raise ValueError("fixed warm continuation requires restored history")
        if initial_attempt_offset != len(attempts):
            raise ValueError("fixed warm continuation offset does not match history")
        if [attempt.attempt_number for attempt in attempts] != list(
            range(1, initial_attempt_offset + 1)
        ):
            raise ValueError("fixed warm continuation history is not contiguous")
        if initial_attempt_offset != 38 or self.max_iterations != 2:
            raise ValueError(
                "fixed warm continuation requires offset 38 and exactly 2 new attempts"
            )
        if not initial_strategy_code or initial_strategy_code != attempts[-1].strategy_code:
            raise ValueError("fixed warm continuation seed does not match restored S38")
        if (
            self._incumbent is None
            or self._incumbent.attempt_number != initial_attempt_offset
            or self._incumbent.strategy_code != initial_strategy_code
        ):
            raise ValueError("fixed warm continuation seed is not the restored incumbent")
        logger.info(
            "[fixed-warm] validated restored history attempts=%d seed_attempt=%d",
            len(attempts),
            self._incumbent.attempt_number,
        )

    def _render_flip_diff(
        self, incumbent_result: EvaluationResult, candidate_result: EvaluationResult
    ) -> str:
        """Per-example pass/fail diff between two slice-A evals of the same
        examples, paired by list position (early-stopped evals can be
        shorter, so only pair up to the shorter length).

        "pass->fail" = incumbent's sample was correct, candidate's wasn't --
        rendered in full (actual answer + helper trace) for up to 3 examples,
        with a count note for the rest. "fail->pass" is a count only. Returns
        "" when nothing flipped either way.
        """
        inc_samples = incumbent_result.sample_outputs or []
        cand_samples = candidate_result.sample_outputs or []
        n = min(len(inc_samples), len(cand_samples))

        regressions: list[tuple[int, dict, dict]] = []
        improvements = 0
        for i in range(n):
            inc_s = inc_samples[i] or {}
            cand_s = cand_samples[i] or {}
            inc_correct = bool(inc_s.get("is_correct", False))
            cand_correct = bool(cand_s.get("is_correct", False))
            if inc_correct and not cand_correct:
                regressions.append((i, inc_s, cand_s))
            elif not inc_correct and cand_correct:
                improvements += 1

        if not regressions and not improvements:
            return ""

        cap = 3
        lines = []
        if regressions:
            lines.append(
                "pass->fail (incumbent got these right, candidate got them wrong):"
            )
            for i, inc_s, cand_s in regressions[:cap]:
                inc_trace = [e.get("helper", "?") for e in inc_s.get("helper_trace", []) or []]
                cand_trace = [e.get("helper", "?") for e in cand_s.get("helper_trace", []) or []]
                lines.append(f"  example {i}:")
                lines.append(
                    f"    incumbent actual: {inc_s.get('actual', '?')}  "
                    f"(helpers: {' -> '.join(inc_trace) or 'none'})"
                )
                lines.append(
                    f"    candidate actual: {cand_s.get('actual', '?')}  "
                    f"(helpers: {' -> '.join(cand_trace) or 'none'})"
                )
            if len(regressions) > cap:
                lines.append(f"  ... {len(regressions) - cap} more")
        if improvements:
            lines.append(
                f"fail->pass: {improvements} example(s) the candidate fixed "
                "relative to the incumbent"
            )
        return "\n".join(lines)

    def _render_rejected_candidate_block(
        self,
        candidate_code: str,
        candidate_a_result: EvaluationResult,
        candidate_diagnostics: str,
        incumbent: "_Incumbent",
    ) -> str:
        """Author-facing feedback block for a REJECTED candidate.

        Quotes only slice-A numbers -- the candidate's and the incumbent's --
        never slice-B or pooled numbers (plan §6), so the confirmation slice
        can't be indirectly optimized against over a long run. The flip diff
        (also slice-A vs slice-A) is built the same way, and only appears
        when at least one example actually flipped -- e.g. an eval-error
        candidate with no sample_outputs yields no flips and no section.
        """
        cand_acc = candidate_a_result.accuracy or 0.0
        cand_syn = candidate_a_result.syntax_rate or 0.0
        cand_shortfall = self._shortfall(cand_acc, cand_syn)
        inc_acc = incumbent.a_result.accuracy or 0.0
        inc_syn = incumbent.a_result.syntax_rate or 0.0
        inc_shortfall = self._shortfall(inc_acc, inc_syn)
        delta = cand_shortfall - inc_shortfall
        flip_diff = self._render_flip_diff(incumbent.a_result, candidate_a_result)
        parts = [
            "The following candidate strategy was just tried and REJECTED --",
            "keep refining the incumbent strategy above, not this one.",
            "",
            "Rejected candidate code:",
            candidate_code,
            "",
            f"Rejected candidate accuracy: {cand_acc:.1%}, syntax: {cand_syn:.1%}",
            "",
            candidate_diagnostics,
        ]
        if flip_diff:
            parts += ["", flip_diff]
        parts += [
            "",
            f"REJECTED: candidate shortfall {cand_shortfall:.3f} vs incumbent "
            f"shortfall {inc_shortfall:.3f} (delta {delta:+.3f})",
        ]
        return "\n".join(parts)

    def _eval_error_candidate_feedback(self, eval_result: EvaluationResult) -> str:
        """Diagnostic text for a candidate whose evaluation raised an error
        (not a clean threshold miss)."""
        return eval_result.get_feedback_summary(self.require_delimiters) + _final_span_failure_hint(
            self.require_delimiters, eval_result.sample_outputs
        )

    def _threshold_miss_candidate_feedback(self, eval_result: EvaluationResult) -> str:
        """Diagnostic text for a candidate that ran cleanly but missed a bar."""
        return (
            "Required thresholds:\n"
            f"  Accuracy: {self.min_accuracy:.1%}\n"
            f"  Syntax Rate: {self.min_syntax_rate:.1%}\n\n"
            + ("  Contains << >>: required\n" if self.require_delimiters else "")
            + (
                f"  Max Runtime / Example: {self.eval_max_seconds_per_example:.2f}s\n"
                if self.eval_max_seconds_per_example is not None
                else ""
            )
            + _delimiter_miss_hint(
                self.require_delimiters, eval_result.contains_delimiters, eval_result.sample_outputs
            )
            + _span_not_closed_hint(self.require_delimiters, eval_result.sample_outputs)
            + _constraint_bypassed_hint(
                self.require_delimiters, eval_result.contains_delimiters, eval_result.sample_outputs
            )
            + _final_span_failure_hint(self.require_delimiters, eval_result.sample_outputs)
            + (
                _token_cap_exhaustion_hint(eval_result.sample_outputs, self.evaluator.max_steps)
                if self.token_cap_feedback
                else ""
            )
            + "\n"
            + eval_result.get_feedback_summary(self.require_delimiters)
        )

    def _compute_allowed_helpers(self, attempts: list[SynthesisAttempt]) -> tuple[list[str] | None, str]:
        """
        Build a per-attempt helper-call contract from empirical policy.

        Returns:
            (allowed_helpers, status_text). `allowed_helpers=None` disables the
            contract block in prompts.
        """
        if not self.adaptive_helper_mask or not self._helper_universe:
            return None, ""

        # Attempts whose evaluation completed zero examples carry no real signal
        # for the helper mask: their scalar score reflects defaults (0 accuracy,
        # 0 syntax_rate) rather than actual helper behavior. Excluding them keeps
        # the utility/bandit policies from training on phantom rewards.
        with_eval = [attempt for attempt in attempts if attempt.eval_result is not None]
        evaluated = [
            attempt for attempt in with_eval
            if (attempt.eval_result.num_examples or 0) > 0
        ]
        skipped_zero_n = len(with_eval) - len(evaluated)
        allowed, status = self._compute_allowed_helpers_bandit(evaluated)
        if skipped_zero_n:
            suffix = (
                f"; excluded {skipped_zero_n} attempt(s) with zero evaluated "
                "examples from helper-mask scoring"
            )
            status = (status + suffix) if status else suffix.lstrip("; ")
        return allowed, status

    def _get_disallowed_helper_calls(
        self,
        strategy_code: str,
        allowed_helpers: list[str] | None,
    ) -> list[str]:
        """Return helper calls in strategy_code that violate the active contract."""
        if not allowed_helpers:
            return []
        allowed = set(allowed_helpers)
        used = set(self._get_helper_calls_for_evaluation_history(strategy_code))
        return sorted(used - allowed)

    def _build_attempt_outcome_ledger(
        self,
        attempts: list[SynthesisAttempt],
        best_attempt_number: int | None,
    ) -> str:
        """Build compact empirical context from evaluated attempts."""
        evaluated = [
            attempt for attempt in attempts
            if attempt.eval_result is not None
            and (attempt.eval_result.num_examples or 0) > 0
        ]
        if not evaluated:
            return ""

        best_attempt = next(
            (attempt for attempt in evaluated if attempt.attempt_number == best_attempt_number),
            None,
        )
        if best_attempt is None:
            best_attempt = max(
                evaluated,
                key=lambda attempt: self._evaluation_scalar_score(attempt.eval_result),
            )
        best_result = best_attempt.eval_result
        if best_result is None:
            return ""

        recent = [
            attempt for attempt in evaluated
            if attempt.attempt_number != best_attempt.attempt_number
        ][-3:]

        def rationale_summary(attempt: SynthesisAttempt) -> str:
            rationale = extract_rationale(attempt.strategy_code).rationale
            if not rationale:
                return "none captured"
            summarizer = getattr(self.generator, "summarize_rationale_claim", None)
            if callable(summarizer):
                try:
                    return str(summarizer(rationale)).strip() or rationale
                except Exception:
                    return rationale
            return rationale

        def failure_locations(result: EvaluationResult) -> str:
            counts: dict[str, int] = {}
            for sample in result.sample_outputs or []:
                if sample.get("is_correct"):
                    continue
                location = str(sample.get("failure_location") or "unknown")
                counts[location] = counts.get(location, 0) + 1
            if not counts:
                return "none"
            return ", ".join(f"{key}={counts[key]}" for key in sorted(counts))

        def attempt_line(attempt: SynthesisAttempt, *, include_delta: bool) -> list[str]:
            result = attempt.eval_result
            if result is None:
                return []
            lines = [
                (
                    f"- Attempt {attempt.attempt_number}: "
                    f"accuracy {result.accuracy:.1%}, syntax {result.syntax_rate:.1%}; "
                    f"rationale claim: {rationale_summary(attempt)}"
                )
            ]
            if include_delta:
                acc_delta = (result.accuracy - best_result.accuracy) * 100.0
                syn_delta = (result.syntax_rate - best_result.syntax_rate) * 100.0
                lines.append(
                    f"  measured effect vs best: accuracy {acc_delta:+.1f}pp, "
                    f"syntax {syn_delta:+.1f}pp"
                )
            lines.append(f"  failure locations: {failure_locations(result)}")
            return lines

        lines = [
            "Use this as empirical search context, not as a recipe.",
            "Best result:",
            *attempt_line(best_attempt, include_delta=False),
        ]
        if recent:
            lines.append("Recent evaluated branches:")
            for attempt in recent:
                lines.extend(attempt_line(attempt, include_delta=True))
        model = HintLinesModel(lines=lines)
        rendered = _render_prompt(model, "feedback_loop/hint_lines.j2")
        # See get_failure_summary above: strip the one trailing newline the
        # template's keep_trailing_newline behavior adds back, to match the
        # original "\n".join(lines) idiom exactly.
        if rendered.endswith("\n"):
            rendered = rendered[:-1]
        return rendered

    @staticmethod
    def _strategy_change_ratio(before: str, after: str) -> float:
        """Return an approximate changed-line ratio between two strategy bodies."""
        before_lines = before.splitlines()
        after_lines = after.splitlines()
        diff_lines = list(
            unified_diff(
                before_lines,
                after_lines,
                fromfile="before",
                tofile="after",
                lineterm="",
                n=0,
            )
        )
        changed = sum(
            1
            for line in diff_lines
            if line and line[0] in {"+", "-"} and not line.startswith(("+++", "---"))
        )
        denom = max(1, len(before_lines))
        return changed / denom

    @staticmethod
    def _normalized_strategy_key(strategy_code: str) -> str:
        """Normalize strategy text for duplicate suppression in beam search."""
        return re.sub(r"\s+", " ", strategy_code).strip()

    def _helper_overlap_ratio(self, before: str, after: str) -> float:
        """Jaccard overlap of helper call sets between strategies."""
        before_set = set(self._get_helper_calls_for_evaluation_history(before))
        after_set = set(self._get_helper_calls_for_evaluation_history(after))
        union = before_set | after_set
        if not union:
            return 1.0
        return len(before_set & after_set) / len(union)

    def _refine_with_beam(
        self,
        *,
        stage_label: str,
        previous_strategy: str,
        allowed_helpers: list[str] | None,
        refine_once: Callable[[], str],
    ) -> str:
        """
        Sample multiple local refinements and select the best candidate.

        Ranking preference:
        1) helper-call contract satisfaction,
        2) optional pre-verification success (if enabled),
        3) local-neighborhood preference,
        4) helper-set overlap with previous strategy,
        5) smaller edit ratio.
        """
        beam_size = max(1, self.refinement_beam_size)
        if beam_size == 1:
            return refine_once()

        candidates: list[dict] = []
        seen: set[str] = {self._normalized_strategy_key(previous_strategy)}

        for _ in range(beam_size):
            candidate = refine_once()
            key = self._normalized_strategy_key(candidate)
            if not key or key in seen:
                continue
            seen.add(key)

            disallowed = self._get_disallowed_helper_calls(candidate, allowed_helpers)
            contract_ok = len(disallowed) == 0
            change_ratio = self._strategy_change_ratio(previous_strategy, candidate)
            locality_ok = (not self.local_neighborhood_refinement) or (
                change_ratio <= self.max_local_edit_ratio
            )
            overlap = self._helper_overlap_ratio(previous_strategy, candidate)

            verification_ok = False
            if self.beam_verify_candidates:
                try:
                    verification_ok = self.verifier.verify(
                        self.generator.inject_strategy(candidate)
                    ).success
                except Exception:
                    verification_ok = False

            score = (
                1 if contract_ok else 0,
                1 if verification_ok else 0,
                1 if locality_ok else 0,
                overlap,
                -change_ratio,
                -len(disallowed),
            )
            candidates.append(
                {
                    "strategy": candidate,
                    "score": score,
                    "contract_ok": contract_ok,
                    "verification_ok": verification_ok,
                    "locality_ok": locality_ok,
                    "change_ratio": change_ratio,
                    "disallowed": disallowed,
                }
            )

        if not candidates:
            return refine_once()

        best = max(candidates, key=lambda item: item["score"])
        print(
            f"  Beam {stage_label}: {len(candidates)} candidate(s), "
            f"selected contract_ok={best['contract_ok']} "
            f"verify_ok={best['verification_ok']} "
            f"local={best['locality_ok']} "
            f"edit_ratio={best['change_ratio']:.2f}"
        )
        if best["disallowed"]:
            print(
                "  Beam selected candidate still violates helper contract: "
                + ", ".join(best["disallowed"])
            )
        return best["strategy"]

    def synthesize(
        self,
        task_description: str,
        output_name: str = "generated_csd",
        initial_strategy_code: str | None = None,
        initial_attempt_offset: int = 0,
        initial_attempts: list[SynthesisAttempt] | None = None,
        initial_failure_ledger: dict | None = None,
        fixed_warm_continuation: bool = False,
    ) -> SynthesisResult:
        """
        Synthesize a CSD strategy for the given task.

        Args:
            task_description: Description of what the strategy should accomplish
            output_name: Name for the output module

        Returns:
            SynthesisResult on success

        Raises:
            SynthesisExhaustionError: If all attempts fail
        """
        import time

        try:
            from synthesis.failure_taxonomy import make_persistent_ledger
        except ImportError:
            from failure_taxonomy import make_persistent_ledger

        start_time = time.time()
        attempts: list[SynthesisAttempt] = list(initial_attempts or [])
        self._restore_incumbent_from_attempts(attempts)
        history_only_continuation = bool(attempts) and initial_strategy_code is None
        if history_only_continuation and initial_failure_ledger is None:
            raise ValueError(
                "history-only continuation requires a restored failure ledger"
            )
        self._failure_ledger = (
            copy.deepcopy(initial_failure_ledger)
            if initial_failure_ledger is not None
            else make_persistent_ledger()
        )
        if initial_failure_ledger is not None:
            logger.warning(
                "[warm-resume] restored failure ledger modes=%d prior_attempts=%d",
                len(self._failure_ledger["modes"]),
                len(
                    {
                        attempt_number
                        for mode in self._failure_ledger["modes"]
                        for attempt_number in mode["attempts"]
                    }
                ),
            )

        # Create an isolated output directory for this run. The directory layout is:
        #   outputs/generated/<output_name>_<run_id>/
        #     - dafny/
        #     - python/
        #     - results/
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + secrets.token_hex(3)
        run_dir = self.output_dir / f"{output_name}_{run_id}"
        run_dafny_dir = run_dir / "dafny"
        run_python_dir = run_dir / "python"
        run_results_dir = run_dir / "results"
        run_dafny_dir.mkdir(parents=True, exist_ok=True)
        run_python_dir.mkdir(parents=True, exist_ok=True)
        run_results_dir.mkdir(parents=True, exist_ok=True)

        # Persist exact prompt/response records under the repo's single logs tree.
        from synthesis.project_defaults import synthesis_prompt_log_dir

        prompt_log_dir = synthesis_prompt_log_dir(output_name, run_id)
        prompt_log_dir.mkdir(parents=True, exist_ok=True)
        os.environ["CSD_PROMPT_LOG_DIR"] = str(prompt_log_dir)

        # Update a convenience pointer to the most recent run
        try:
            (self.output_dir / "latest_run.txt").write_text(str(run_dir) + "\n")
        except Exception:
            pass

        # Use a per-run compiler output directory.
        compiler = DafnyCompiler(
            dafny_path=self.compiler.dafny_path,
            output_dir=run_python_dir,
            timeout=self.compiler.timeout,
            extra_args=list(self.compiler.extra_args),
        )

        self.generator.set_synthesis_context(
            eval_model=self.evaluator.model_name,
            dataset=self.evaluator.dataset_name,
            max_steps=self.evaluator.max_steps,
            step_token_budget=self.evaluator.step_token_budget,
        )
        self.generator.set_task_description(task_description)

        allowed_helpers, helper_status = self._compute_allowed_helpers(attempts)
        if helper_status:
            print(f"Helper policy: {helper_status}")

        # Initial generation, a caller-provided recovery seed, or a continuation
        # that authors the next candidate from the restored incumbent.
        if history_only_continuation:
            incumbent = self._incumbent
            if incumbent is None:
                raise ValueError(
                    "history-only continuation requires a score-bearing incumbent"
                )
            logger.warning(
                "[warm-resume] authoring first new attempt from restored incumbent=%d",
                incumbent.attempt_number,
            )
            incumbent.a_result._failure_ledger = self._failure_ledger
            incumbent.a_result._attempt_index = incumbent.attempt_number
            strategy_code = self._refine_with_beam(
                stage_label="warm_resume_seed",
                previous_strategy=incumbent.strategy_code,
                allowed_helpers=allowed_helpers,
                refine_once=lambda: self.generator.refine_after_evaluation_failure(
                    previous_strategy=incumbent.strategy_code,
                    previous_accuracy=incumbent.a_result.accuracy or 0.0,
                    previous_syntax_rate=incumbent.a_result.syntax_rate or 0.0,
                    num_examples=incumbent.a_result.num_examples or 0,
                    goal_accuracy=self.min_accuracy,
                    goal_syntax_rate=self.min_syntax_rate,
                    evaluation_feedback=self._threshold_miss_candidate_feedback(
                        incumbent.a_result
                    ),
                    best_strategy=None,
                    best_accuracy=None,
                    best_syntax_rate=None,
                    allowed_helpers=allowed_helpers,
                    eval_max_seconds_per_example=self.eval_max_seconds_per_example,
                    mode_examples=incumbent.a_result._render_mode_examples(),
                    attempt_outcome_ledger=self._build_attempt_outcome_ledger(
                        attempts, incumbent.attempt_number
                    ),
                ),
            )
        elif initial_strategy_code is not None:
            logger.warning(
                "[warm-resume] task context initialized before strategy replay; "
                "task_chars=%d initial_attempt_offset=%d",
                len(task_description),
                initial_attempt_offset,
            )
            print("Using caller-provided initial strategy seed")
            strategy_code = initial_strategy_code
        else:
            print(f"Generating initial strategy for: {task_description}")
            strategy_code = self.generator.generate_initial(
                task_description,
                allowed_helpers=allowed_helpers,
                force_open_span=self._force_open_span(),
            )

        if fixed_warm_continuation:
            self._validate_fixed_warm_continuation(
                initial_strategy_code=initial_strategy_code,
                initial_attempt_offset=initial_attempt_offset,
                attempts=attempts,
            )
            incumbent = self._incumbent
            assert incumbent is not None  # validated immediately above
            logger.info(
                "[fixed-warm] authoring attempt=39 from restored attempt=%d",
                incumbent.attempt_number,
            )
            strategy_code = self._refine_with_beam(
                stage_label="fixed_warm_seed",
                previous_strategy=incumbent.strategy_code,
                allowed_helpers=allowed_helpers,
                refine_once=lambda: self.generator.refine_after_evaluation_failure(
                    previous_strategy=incumbent.strategy_code,
                    previous_accuracy=incumbent.a_result.accuracy or 0.0,
                    previous_syntax_rate=incumbent.a_result.syntax_rate or 0.0,
                    num_examples=incumbent.a_result.num_examples or 0,
                    goal_accuracy=self.min_accuracy,
                    goal_syntax_rate=self.min_syntax_rate,
                    evaluation_feedback=(
                        "Fixed warm continuation: refine the restored incumbent "
                        "for the first of two scheduled continuation attempts."
                    ),
                    best_strategy=None,
                    best_accuracy=None,
                    best_syntax_rate=None,
                    allowed_helpers=allowed_helpers,
                    eval_max_seconds_per_example=self.eval_max_seconds_per_example,
                    mode_examples=incumbent.a_result._render_mode_examples(),
                    attempt_outcome_ledger=self._build_attempt_outcome_ledger(
                        attempts, incumbent.attempt_number
                    ),
                ),
            )

        # Index in `attempts` after which we last performed a fresh restart.
        # Used to bound the "consecutive verification failures since last restart"
        # counter so that a restart resets it.
        last_restart_index = 0

        for iteration in range(self.max_iterations):
            attempt_num = initial_attempt_offset + iteration + 1
            attempt_total = initial_attempt_offset + self.max_iterations
            allowed_helpers, helper_status = self._compute_allowed_helpers(attempts)
            print(f"\n{'='*60}")
            print(f"Attempt {attempt_num}/{attempt_total}")
            print(f"{'='*60}")
            if helper_status:
                print(f"Helper policy: {helper_status}")
            print(display_text("Strategy", strategy_code))

            # Create full Dafny code
            full_code = self.generator.inject_strategy(strategy_code)

            # Wall-clock start of this attempt, for the max_attempt_seconds cap.
            attempt_start_time = time.time()

            # Create attempt record
            attempt = SynthesisAttempt(
                attempt_number=attempt_num,
                strategy_code=strategy_code,
                full_dafny_code=full_code,
                timestamp=datetime.now().isoformat(),
            )

            disallowed_helpers = self._get_disallowed_helper_calls(strategy_code, allowed_helpers)
            if disallowed_helpers:
                print("  ✗ Strategy contract violation")
                error_msg = (
                    "Strategy contract violation.\n"
                    f"Violations: {', '.join(disallowed_helpers)}"
                )
                attempt.failed_at = FailureStage.SEARCH_CONTRACT
                attempt.error_summary = error_msg
                attempts.append(attempt)
                self._save_progress_report(attempts, task_description, output_name, run_results_dir)

                print("  Refining based on strategy contract violation...")
                next_allowed_helpers, next_helper_status = self._compute_allowed_helpers(attempts)
                if next_helper_status:
                    print(f"  Helper policy: {next_helper_status}")
                strategy_code = self._refine_with_beam(
                    stage_label="search_contract",
                    previous_strategy=strategy_code,
                    allowed_helpers=next_allowed_helpers,
                    refine_once=lambda: self.generator.refine_after_verification_error(
                        strategy_code,
                        error_msg,
                        allowed_helpers=next_allowed_helpers,
                    ),
                )
                continue

            # Stage 1: Verification
            print("\n[1/4] Verifying with Dafny...")
            verification_result = self.verifier.verify(full_code)
            attempt.verification_result = verification_result

            if not verification_result.success:
                print("  ✗ Verification failed")
                print(f"  Error: {verification_result.get_error_summary()[:300]}")
                attempt.failed_at = FailureStage.VERIFICATION
                attempt.error_summary = verification_result.get_error_summary()
                attempts.append(attempt)
                self._save_progress_report(attempts, task_description, output_name, run_results_dir)

                # Check if we're stuck on the same error repeatedly
                error_msg = verification_result.get_error_summary()
                consecutive_same = 0
                for prev in reversed(attempts[:-1]):
                    if prev.failed_at == FailureStage.VERIFICATION and prev.error_summary == error_msg:
                        consecutive_same += 1
                    else:
                        break

                if consecutive_same >= 2:
                    # After 3+ identical errors, abandon refinement and start fresh
                    print(f"  Stuck on same error for {consecutive_same + 1} attempts — restarting with fresh generation...")
                    next_allowed_helpers, next_helper_status = self._compute_allowed_helpers(attempts)
                    if next_helper_status:
                        print(f"  Helper policy: {next_helper_status}")
                    strategy_code = self.generator.generate_initial(
                        task_description,
                        allowed_helpers=next_allowed_helpers,
                        force_open_span=self._force_open_span(),
                    )
                    last_restart_index = len(attempts)
                    continue

                # Also restart if the last 3 attempts since the most recent
                # restart all failed verification, even when the specific
                # errors differ. This catches cases where the model is
                # rewriting the strategy every iteration and each broken
                # version surfaces a fresh error.
                post_restart_attempts = attempts[last_restart_index:]
                consecutive_verif_failures = 0
                for prev in reversed(post_restart_attempts):
                    if prev.failed_at == FailureStage.VERIFICATION:
                        consecutive_verif_failures += 1
                    else:
                        break

                if consecutive_verif_failures >= 3:
                    print(
                        f"  {consecutive_verif_failures} consecutive verification failures "
                        f"since last restart — restarting with fresh generation..."
                    )
                    next_allowed_helpers, next_helper_status = self._compute_allowed_helpers(attempts)
                    if next_helper_status:
                        print(f"  Helper policy: {next_helper_status}")
                    strategy_code = self.generator.generate_initial(
                        task_description,
                        allowed_helpers=next_allowed_helpers,
                        force_open_span=self._force_open_span(),
                    )
                    last_restart_index = len(attempts)
                    continue

                # Refine based on verification error
                print("  Refining based on verification error...")
                structured_feedback = verification_result.get_structured_feedback()
                error_history = self._get_verification_history_summary(attempts)
                behavioral_context = self._get_recent_behavioral_context(attempts)
                next_allowed_helpers, next_helper_status = self._compute_allowed_helpers(attempts)
                if next_helper_status:
                    print(f"  Helper policy: {next_helper_status}")
                strategy_code = self._refine_with_beam(
                    stage_label="verification",
                    previous_strategy=strategy_code,
                    allowed_helpers=next_allowed_helpers,
                    refine_once=lambda: self.generator.refine_after_verification_error(
                        strategy_code,
                        error_msg,
                        behavioral_context=behavioral_context,
                        structured_feedback=structured_feedback,
                        error_history=error_history,
                        allowed_helpers=next_allowed_helpers,
                    ),
                )
                continue

            print("  ✓ Verification passed")

            # Stage 2: Compilation
            print("\n[2/4] Compiling to Python...")
            compilation_result = compiler.compile(full_code, output_name)
            attempt.compilation_result = compilation_result

            if not compilation_result.success:
                print("  ✗ Compilation failed")
                attempt.failed_at = FailureStage.COMPILATION
                attempt.error_summary = compilation_result.get_error_summary()
                attempts.append(attempt)
                self._save_progress_report(attempts, task_description, output_name, run_results_dir)

                # Refine based on compilation error
                print("  Refining based on compilation error...")
                next_allowed_helpers, next_helper_status = self._compute_allowed_helpers(attempts)
                if next_helper_status:
                    print(f"  Helper policy: {next_helper_status}")
                strategy_code = self._refine_with_beam(
                    stage_label="compilation",
                    previous_strategy=strategy_code,
                    allowed_helpers=next_allowed_helpers,
                    refine_once=lambda: self.generator.refine_after_compilation_error(
                        strategy_code,
                        compilation_result.get_error_summary(),
                        allowed_helpers=next_allowed_helpers,
                    ),
                )
                continue

            print(f"  ✓ Compiled to {compilation_result.output_dir}")

            if compilation_result.main_module_path is None:
                print("  ✗ No main module found")
                attempt.failed_at = FailureStage.RUNTIME
                attempt.error_summary = "No main module path in compilation result"
                attempts.append(attempt)
                self._save_progress_report(attempts, task_description, output_name, run_results_dir)

                next_allowed_helpers, next_helper_status = self._compute_allowed_helpers(attempts)
                if next_helper_status:
                    print(f"  Helper policy: {next_helper_status}")
                strategy_code = self._refine_with_beam(
                    stage_label="runtime",
                    previous_strategy=strategy_code,
                    allowed_helpers=next_allowed_helpers,
                    refine_once=lambda: self.generator.refine_after_runtime_error(
                        strategy_code,
                        "Compilation succeeded but no Python module was generated",
                        allowed_helpers=next_allowed_helpers,
                    ),
                )
                continue

            print("\n[3/4] Evaluating compiled strategy (runtime smoke test removed).")

            # Stage 4: Evaluation
            print("\n[4/4] Evaluating on dataset sample...")
            # Unload generator model to free GPU memory for eval model
            if self.generator._model is not None:
                del self.generator._model
                self.generator._model = None
                import gc
                gc.collect()
                import torch
                torch.cuda.empty_cache()
                print("  Generator model (HF) unloaded to free GPU memory")
            # Also unload vllm engine if present (vllm backend keeps workers in subprocesses)
            if getattr(self.generator, '_vllm', None) is not None:
                import gc
                import torch
                vllm_obj = self.generator._vllm
                self.generator._vllm = None
                try:
                    vllm_obj._run_engine = None  # sever reference to engine
                except Exception:
                    pass
                del vllm_obj
                try:
                    from vllm.distributed import destroy_model_parallel, destroy_distributed_environment
                    destroy_model_parallel()
                    destroy_distributed_environment()
                except Exception:
                    pass
                gc.collect()
                torch.cuda.empty_cache()
                print("  Generator vllm engine unloaded to free GPU memory")

            # Eval seed is fixed across iterations so per-iter deltas are
            # statistically trustworthy and opus can anchor on best-so-far.
            # Overfitting concern is handled by the final held-out eval.
            print(f"  [synthesis] eval seed for this iteration: {self.evaluator.sample_seed}")
            attempt_deadline = (
                attempt_start_time + self.max_attempt_seconds
                if self.max_attempt_seconds is not None
                else None
            )
            eval_result = self.evaluator.evaluate_sample(
                compiled_module_path=compilation_result.main_module_path,
                sample_size=self.eval_sample_size,
                early_stop_min_accuracy=self.min_accuracy
                if os.environ.get("CSD_SYNTHESIS_EVAL_EARLY_STOP", "1") != "0"
                else None,
                early_stop_min_syntax_rate=self.min_syntax_rate
                if os.environ.get("CSD_SYNTHESIS_EVAL_EARLY_STOP", "1") != "0"
                else None,
                early_stop_runtime_failures=(
                    int(os.environ.get("CSD_SYNTHESIS_EVAL_EARLY_STOP_RUNTIME_FAILURES", "3"))
                    if os.environ.get("CSD_SYNTHESIS_EVAL_EARLY_STOP", "1") != "0"
                    else None
                ),
                min_examples_before_threshold_stop=self.min_examples_before_threshold_stop,
                deadline=attempt_deadline,
            )
            # Change 2: stamp the cross-attempt cluster ledger on the result
            # so EvaluationResult.get_feedback_summary can emit persistent
            # mode IDs (mode_A appeared in attempts 1,3,5…).
            if not hasattr(self, "_failure_ledger"):
                try:
                    from synthesis.failure_taxonomy import make_persistent_ledger
                except ImportError:
                    from failure_taxonomy import make_persistent_ledger
                self._failure_ledger = make_persistent_ledger()
            eval_result._failure_ledger = self._failure_ledger
            eval_result._attempt_index = attempt.attempt_number
            attempt.eval_result = eval_result

            # Attempt-level wall-clock cap: hit either when evaluate_sample
            # itself stopped early because `deadline` was crossed (the reason
            # string carries the marker), or -- as a defense-in-depth backstop
            # -- when the attempt overran its budget for any other reason
            # (e.g. a slow verification/compilation stage). A complete result
            # remains search evidence but cannot end the run as a success; an
            # incomplete result remains a timeout. Classify both before the
            # normal success/threshold branches below see them.
            attempt_elapsed = time.time() - attempt_start_time
            deadline_early_stop = bool(
                eval_result.early_stop_reason
                and eval_result.early_stop_reason.startswith(ATTEMPT_DEADLINE_EARLY_STOP_REASON)
            )
            wall_clock_over_budget = bool(
                self.max_attempt_seconds is not None
                and attempt_elapsed > self.max_attempt_seconds
            )
            attempt_hit_deadline = deadline_early_stop or wall_clock_over_budget
            complete_over_budget = bool(
                attempt_hit_deadline
                and self._is_complete_evaluation(eval_result)
            )
            if complete_over_budget:
                original_early_stop_reason = eval_result.early_stop_reason
                eval_result.aux_metrics["attempt_runtime_budget"] = {
                    "limit_seconds": self.max_attempt_seconds,
                    "elapsed_seconds": attempt_elapsed,
                    "over_budget": True,
                    "complete_evaluation_preserved": True,
                    "original_early_stop_reason": original_early_stop_reason,
                }
                logger.warning(
                    "[attempt-budget] preserving complete evaluation "
                    "attempt=%d examples=%d planned=%d elapsed=%.1fs limit=%.1fs",
                    attempt.attempt_number,
                    eval_result.num_examples or 0,
                    eval_result.planned_num_examples or self.eval_sample_size,
                    attempt_elapsed,
                    self.max_attempt_seconds,
                )
                print(
                    f"  ! Attempt {attempt.attempt_number} exceeded the "
                    f"{self.max_attempt_seconds:.0f}s attempt time cap, but "
                    "the full evaluation completed; preserving its score for search."
                )
            elif attempt_hit_deadline:
                strategy_code = self._handle_attempt_timeout(
                    attempt,
                    attempts,
                    attempt_elapsed,
                    task_description,
                    output_name,
                    run_results_dir,
                )
                continue

            smiles_trial = (eval_result.aux_metrics or {}).get("smiles_paper_trial", {})
            if isinstance(smiles_trial, dict) and smiles_trial:
                membership = smiles_trial.get("membership")
                validity = smiles_trial.get("validity_rdkit")
                samples_to_target = smiles_trial.get("samples_to_target_unique_valid")
                unique_valid = smiles_trial.get("unique_valid_count")
                sample_count = smiles_trial.get("sample_count")
                print("  [smiles] paper-aligned metrics:")
                if membership is not None:
                    print(f"    Membership: {float(membership):.1%}")
                if validity is not None:
                    print(f"    RDKit Validity: {float(validity):.1%}")
                print(
                    "    Samples to 100 unique valid (cap 1000): "
                    f"{samples_to_target}"
                )
                print(
                    "    Unique valid molecules this eval: "
                    f"{unique_valid}/{sample_count}"
                )

            if not eval_result.success:
                print(f"  ✗ Evaluation failed: {eval_result.error}")
                failure_stage = classify_eval_failure(eval_result)
                attempt.failed_at = failure_stage
                attempt.error_summary = eval_result.error or "Evaluation failed"
                attempts.append(attempt)
                # Save what we have so far before doing anything else, so a
                # harness failure below does not lose the attempts already
                # recorded.
                self._save_progress_report(attempts, task_description, output_name, run_results_dir)

                if failure_stage is FailureStage.HARNESS:
                    print(
                        "  ✗✗✗ HARNESS FAILURE: the evaluator ran zero examples, "
                        "so nothing was actually measured. This is not a bad "
                        "strategy -- it is a broken evaluation setup (for "
                        "example, a missing Python file), and no strategy "
                        "rewrite can fix it."
                    )
                    print(f"      Underlying error: {eval_result.error}")
                    raise SynthesisExhaustionError(
                        "Synthesis stopped: the evaluator never ran any "
                        f"examples (harness failure), not a bad strategy. "
                        f"Underlying error: {eval_result.error}",
                        attempts,
                    )

                if failure_stage is FailureStage.TIMEOUT:
                    self._unload_evaluator_runtime_before_refinement()
                    next_allowed_helpers, next_helper_status = self._compute_allowed_helpers(attempts)
                    if next_helper_status:
                        print(f"  Helper policy: {next_helper_status}")
                    print("  Restarting with fresh generation after worker hard timeout...")
                    strategy_code = self.generator.generate_initial(
                        task_description,
                        allowed_helpers=next_allowed_helpers,
                        force_open_span=self._force_open_span(),
                    )
                    continue

                self._unload_evaluator_runtime_before_refinement()
                print("  Refining based on evaluation error...")
                candidate_feedback = self._eval_error_candidate_feedback(eval_result)
                mode_examples = eval_result._render_mode_examples()
                next_allowed_helpers, next_helper_status = self._compute_allowed_helpers(attempts)
                if next_helper_status:
                    print(f"  Helper policy: {next_helper_status}")
                incumbent_attempt_number = self._incumbent.attempt_number if self._incumbent else None
                attempt_outcome_ledger = self._build_attempt_outcome_ledger(attempts, incumbent_attempt_number)

                if self._incumbent is None:
                    # Nothing to reject back to yet (the very first candidate
                    # errored before ever seeding an incumbent) -- fall back
                    # to refining the errored candidate itself.
                    print("  [greedy] no incumbent yet; refining the errored candidate directly")
                    prev_acc = eval_result.accuracy or 0.0
                    prev_syn = eval_result.syntax_rate or 0.0
                    prev_n = eval_result.num_examples or 0
                    strategy_code = self._refine_with_beam(
                        stage_label="evaluation_error",
                        previous_strategy=strategy_code,
                        allowed_helpers=next_allowed_helpers,
                        refine_once=lambda: self.generator.refine_after_evaluation_failure(
                            previous_strategy=strategy_code,
                            previous_accuracy=prev_acc,
                            previous_syntax_rate=prev_syn,
                            num_examples=prev_n,
                            goal_accuracy=self.min_accuracy,
                            goal_syntax_rate=self.min_syntax_rate,
                            evaluation_feedback=candidate_feedback,
                            best_strategy=None,
                            best_accuracy=None,
                            best_syntax_rate=None,
                            allowed_helpers=next_allowed_helpers,
                            eval_max_seconds_per_example=self.eval_max_seconds_per_example,
                            mode_examples=mode_examples,
                            attempt_outcome_ledger=attempt_outcome_ledger,
                        ),
                    )
                else:
                    # REJECT: an errored candidate can never become incumbent.
                    # Rebase onto the incumbent's own code/scores before
                    # refining -- the author must never be handed a broken
                    # candidate to build on top of.
                    inc = self._incumbent
                    print(
                        f"  [greedy] REJECT (evaluation error): rebasing on incumbent "
                        f"attempt {inc.attempt_number} (acc={inc.a_result.accuracy:.1%}, "
                        f"syn={inc.a_result.syntax_rate:.1%})"
                    )
                    rejected_feedback = self._render_rejected_candidate_block(
                        strategy_code, eval_result, candidate_feedback, inc
                    )
                    strategy_code = inc.strategy_code
                    strategy_code = self._refine_with_beam(
                        stage_label="evaluation_error",
                        previous_strategy=strategy_code,
                        allowed_helpers=next_allowed_helpers,
                        refine_once=lambda: self.generator.refine_after_evaluation_failure(
                            previous_strategy=strategy_code,
                            previous_accuracy=inc.a_result.accuracy or 0.0,
                            previous_syntax_rate=inc.a_result.syntax_rate or 0.0,
                            num_examples=inc.a_result.num_examples or 0,
                            goal_accuracy=self.min_accuracy,
                            goal_syntax_rate=self.min_syntax_rate,
                            evaluation_feedback=rejected_feedback,
                            best_strategy=None,
                            best_accuracy=None,
                            best_syntax_rate=None,
                            allowed_helpers=next_allowed_helpers,
                            eval_max_seconds_per_example=self.eval_max_seconds_per_example,
                            mode_examples=mode_examples,
                            attempt_outcome_ledger=attempt_outcome_ledger,
                        ),
                    )
                continue

            # Evaluation ran cleanly on slice A. Greedy hill-climb accept/
            # reject + slice-B confirmation (planning/monotonic-search-plan.md).
            eval_kwargs = dict(
                early_stop_min_accuracy=self.min_accuracy
                if os.environ.get("CSD_SYNTHESIS_EVAL_EARLY_STOP", "1") != "0"
                else None,
                early_stop_min_syntax_rate=self.min_syntax_rate
                if os.environ.get("CSD_SYNTHESIS_EVAL_EARLY_STOP", "1") != "0"
                else None,
                early_stop_runtime_failures=(
                    int(os.environ.get("CSD_SYNTHESIS_EVAL_EARLY_STOP_RUNTIME_FAILURES", "3"))
                    if os.environ.get("CSD_SYNTHESIS_EVAL_EARLY_STOP", "1") != "0"
                    else None
                ),
                min_examples_before_threshold_stop=self.min_examples_before_threshold_stop,
                deadline=attempt_deadline,
            )

            class _ConfirmationDeadlineHit(Exception):
                """Internal signal: the slice-B eval crossed the attempt's
                wall-clock cap. Raised before any incumbent state is mutated
                so a timed-out confirmation can never corrupt the incumbent."""

                def __init__(self, elapsed: float):
                    self.elapsed = elapsed

            def _run_confirmation_eval() -> EvaluationResult:
                """Slice B: the NEXT eval_sample_size train examples, used
                only to confirm a would-be accept (plan §1). Same sample_size
                and early-stop/deadline args as slice A."""
                print(f"  [greedy] confirming on slice B ({self.eval_sample_size} examples)...")
                b_result = self.evaluator.evaluate_sample(
                    compiled_module_path=compilation_result.main_module_path,
                    sample_size=self.eval_sample_size,
                    sample_offset=self.eval_sample_size,
                    **eval_kwargs,
                )
                b_elapsed = time.time() - attempt_start_time
                b_hit_deadline = bool(
                    b_result.early_stop_reason
                    and b_result.early_stop_reason.startswith(ATTEMPT_DEADLINE_EARLY_STOP_REASON)
                ) or (self.max_attempt_seconds is not None and b_elapsed > self.max_attempt_seconds)
                if b_hit_deadline:
                    raise _ConfirmationDeadlineHit(b_elapsed)
                return b_result

            cand_acc_a = eval_result.accuracy or 0.0
            cand_syn_a = eval_result.syntax_rate or 0.0
            cand_shortfall_a = self._shortfall(cand_acc_a, cand_syn_a)
            candidate_b_result: EvaluationResult | None = None

            # --- TEMPORARY A-ONLY (2026-08-10) ---------------------------------
            # GSM/Spider train splits have an empty slice B (sample_offset past
            # the end of train_indices), so dual-slice confirmation cannot win.
            # Active path: hill-climb and SUCCESS on slice A only.
            # Dual-slice body preserved under `if False:` immediately below —
            # flip to `if True:` (and remove the else) to restore B confirmation.
            # -----------------------------------------------------------------
            if False:  # dual-slice B confirmation (DISABLED — restore later)
                candidate_b_result: EvaluationResult | None = None

                try:
                    if self._incumbent is None:
                        print(
                            f"  [greedy] seeding incumbent from attempt {attempt_num} "
                            f"(A: acc={cand_acc_a:.1%}, syn={cand_syn_a:.1%})"
                        )
                        candidate_b_result = _run_confirmation_eval()
                        self._incumbent = _Incumbent(
                            strategy_code=strategy_code,
                            attempt_number=attempt_num,
                            a_result=eval_result,
                            b_result=candidate_b_result,
                        )
                        became_incumbent = True
                    else:
                        inc = self._incumbent
                        inc_acc_a = inc.a_result.accuracy or 0.0
                        inc_syn_a = inc.a_result.syntax_rate or 0.0
                        inc_shortfall_a = self._shortfall(inc_acc_a, inc_syn_a)

                        stage1_pass = self._should_accept(
                            cand_shortfall_a, cand_acc_a, cand_syn_a,
                            inc_shortfall_a, inc_acc_a, inc_syn_a,
                        )
                        if not stage1_pass:
                            print(
                                f"  [greedy] REJECT (stage 1, slice A): attempt {attempt_num} "
                                f"shortfall {cand_shortfall_a:.3f} vs incumbent attempt "
                                f"{inc.attempt_number} shortfall {inc_shortfall_a:.3f}"
                            )
                            became_incumbent = False
                        else:
                            candidate_b_result = _run_confirmation_eval()
                            cand_pooled_acc, cand_pooled_syn = self._pooled_metrics(eval_result, candidate_b_result)
                            inc_pooled_acc, inc_pooled_syn = self._pooled_metrics(inc.a_result, inc.b_result)
                            cand_pooled_shortfall = self._shortfall(cand_pooled_acc, cand_pooled_syn)
                            inc_pooled_shortfall = self._shortfall(inc_pooled_acc, inc_pooled_syn)
                            stage2_pass = self._should_accept(
                                cand_pooled_shortfall, cand_pooled_acc, cand_pooled_syn,
                                inc_pooled_shortfall, inc_pooled_acc, inc_pooled_syn,
                            )
                            if stage2_pass:
                                print(
                                    f"  [greedy] ACCEPT (stage 2, pooled A+B): attempt {attempt_num} "
                                    f"pooled shortfall {cand_pooled_shortfall:.3f} vs incumbent "
                                    f"pooled shortfall {inc_pooled_shortfall:.3f} -- new incumbent"
                                )
                                self._incumbent = _Incumbent(
                                    strategy_code=strategy_code,
                                    attempt_number=attempt_num,
                                    a_result=eval_result,
                                    b_result=candidate_b_result,
                                )
                                became_incumbent = True
                            else:
                                print(
                                    f"  [greedy] REJECT (stage 2, pooled A+B): attempt {attempt_num} "
                                    f"pooled shortfall {cand_pooled_shortfall:.3f} vs incumbent "
                                    f"pooled shortfall {inc_pooled_shortfall:.3f}"
                                )
                                became_incumbent = False
                except _ConfirmationDeadlineHit as deadline_hit:
                    strategy_code = self._handle_attempt_timeout(
                        attempt, attempts, deadline_hit.elapsed, task_description, output_name, run_results_dir,
                    )
                    continue

                success = (
                    became_incumbent
                    and self._meets_threshold(eval_result)
                    and self._meets_threshold(candidate_b_result)
                )
            else:
                # A-only hill-climb / win (slice B not evaluated).
                try:
                    if self._incumbent is None:
                        print(
                            f"  [greedy] seeding incumbent from attempt {attempt_num} "
                            f"(A-only: acc={cand_acc_a:.1%}, syn={cand_syn_a:.1%})"
                        )
                        self._incumbent = _Incumbent(
                            strategy_code=strategy_code,
                            attempt_number=attempt_num,
                            a_result=eval_result,
                            b_result=eval_result,
                        )
                        became_incumbent = True
                    else:
                        inc = self._incumbent
                        inc_acc_a = inc.a_result.accuracy or 0.0
                        inc_syn_a = inc.a_result.syntax_rate or 0.0
                        inc_shortfall_a = self._shortfall(inc_acc_a, inc_syn_a)
                        stage1_pass = self._should_accept(
                            cand_shortfall_a, cand_acc_a, cand_syn_a,
                            inc_shortfall_a, inc_acc_a, inc_syn_a,
                        )
                        if not stage1_pass:
                            print(
                                f"  [greedy] REJECT (stage 1, slice A): attempt {attempt_num} "
                                f"shortfall {cand_shortfall_a:.3f} vs incumbent attempt "
                                f"{inc.attempt_number} shortfall {inc_shortfall_a:.3f}"
                            )
                            became_incumbent = False
                        else:
                            print(
                                f"  [greedy] ACCEPT (A-only): attempt {attempt_num} "
                                f"shortfall {cand_shortfall_a:.3f} vs incumbent "
                                f"shortfall {inc_shortfall_a:.3f} -- new incumbent"
                            )
                            self._incumbent = _Incumbent(
                                strategy_code=strategy_code,
                                attempt_number=attempt_num,
                                a_result=eval_result,
                                b_result=eval_result,
                            )
                            became_incumbent = True
                except _ConfirmationDeadlineHit as deadline_hit:
                    strategy_code = self._handle_attempt_timeout(
                        attempt, attempts, deadline_hit.elapsed, task_description, output_name, run_results_dir,
                    )
                    continue

                success = (
                    became_incumbent
                    and self._meets_threshold(eval_result)
                )


            if success:
                if fixed_warm_continuation and iteration + 1 < self.max_iterations:
                    attempts.append(attempt)
                    self._save_progress_report(
                        attempts, task_description, output_name, run_results_dir
                    )
                    incumbent = self._incumbent
                    assert incumbent is not None
                    logger.info(
                        "[fixed-warm] paper bars met at attempt=%d; authoring "
                        "the required final attempt=40 from incumbent=%d",
                        attempt.attempt_number,
                        incumbent.attempt_number,
                    )
                    next_allowed_helpers, next_helper_status = self._compute_allowed_helpers(
                        attempts
                    )
                    if next_helper_status:
                        print(f"  Helper policy: {next_helper_status}")
                    strategy_code = self._refine_with_beam(
                        stage_label="fixed_warm_final",
                        previous_strategy=incumbent.strategy_code,
                        allowed_helpers=next_allowed_helpers,
                        refine_once=lambda: self.generator.refine_after_evaluation_failure(
                            previous_strategy=incumbent.strategy_code,
                            previous_accuracy=incumbent.a_result.accuracy or 0.0,
                            previous_syntax_rate=incumbent.a_result.syntax_rate or 0.0,
                            num_examples=incumbent.a_result.num_examples or 0,
                            goal_accuracy=self.min_accuracy,
                            goal_syntax_rate=self.min_syntax_rate,
                            evaluation_feedback=(
                                "Fixed warm continuation: the paper bars are met, "
                                "but one scheduled final continuation attempt remains."
                            ),
                            best_strategy=None,
                            best_accuracy=None,
                            best_syntax_rate=None,
                            allowed_helpers=next_allowed_helpers,
                            eval_max_seconds_per_example=self.eval_max_seconds_per_example,
                            mode_examples=incumbent.a_result._render_mode_examples(),
                            attempt_outcome_ledger=self._build_attempt_outcome_ledger(
                                attempts, incumbent.attempt_number
                            ),
                        ),
                    )
                    continue
                print(f"  ✓ Evaluation passed (A-only; slice B confirmation disabled):")
                print(f"    Accuracy: {cand_acc_a:.1%}")
                print(f"    Contains << >>: {'yes' if eval_result.contains_delimiters else 'no'}")
                print(f"    Syntax: {cand_syn_a:.1%}")

                # Success!
                attempts.append(attempt)
                self._save_progress_report(attempts, task_description, output_name, run_results_dir)
                total_time = (time.time() - start_time) * 1000

                print(f"\n{'='*60}")
                print(f"SUCCESS after {attempt_num} attempt(s)")
                print(f"Total time: {total_time:.1f}ms")
                print(f"{'='*60}")

                # Save successful strategy
                self._save_success_report(
                    strategy_code,
                    full_code,
                    compilation_result,
                    attempts,
                    task_description,
                    output_name,
                    run_dir,
                    run_dafny_dir,
                    run_results_dir,
                    eval_result,
                )

                return SynthesisResult(
                    success=True,
                    strategy_code=strategy_code,
                    full_dafny_code=full_code,
                    compiled_module_path=compilation_result.main_module_path,
                    output_dir=compilation_result.output_dir,
                    run_dir=run_dir,
                    attempts=attempts,
                    total_time_ms=total_time,
                )

            # Not success: either rejected outright, or accepted as the new
            # incumbent but still short of the bars on slice A and/or slice B
            # (a single-slice bar crossing never ends the run -- plan §4).
            print(f"  ✗ Evaluation below threshold (slice A):")
            print(f"    Accuracy: {cand_acc_a:.1%} (min: {self.min_accuracy:.1%})")
            if self.require_delimiters:
                print(
                    "    Contains << >>: "
                    f"{'yes' if eval_result.contains_delimiters else 'no'} "
                    f"(required: {'yes' if self.require_delimiters else 'no'})"
                )
            _delim_hint = _delimiter_miss_hint(
                self.require_delimiters, eval_result.contains_delimiters, eval_result.sample_outputs
            )
            if _delim_hint:
                print(_delim_hint.rstrip("\n"))
            _close_hint = _span_not_closed_hint(self.require_delimiters, eval_result.sample_outputs)
            if _close_hint:
                print(_close_hint.rstrip("\n"))
            _bypass_hint = _constraint_bypassed_hint(
                self.require_delimiters, eval_result.contains_delimiters, eval_result.sample_outputs
            )
            if _bypass_hint:
                print(_bypass_hint.rstrip("\n"))
            if self.token_cap_feedback:
                _cap_hint = _token_cap_exhaustion_hint(eval_result.sample_outputs, self.evaluator.max_steps)
                if _cap_hint:
                    print(_cap_hint.rstrip("\n"))
            print(f"    Syntax: {cand_syn_a:.1%} (min: {self.min_syntax_rate:.1%})")
            if self.eval_max_seconds_per_example is not None:
                print(
                    f"    Slowest Example Time: {eval_result.max_sample_time_seconds:.2f}s "
                    f"(max: {self.eval_max_seconds_per_example:.2f}s)"
                )
            # TEMPORARY A-ONLY: confirmation-miss path disabled (B not evaluated).
            # Original:
            # confirmation_missed = (
            #     became_incumbent
            #     and candidate_b_result is not None
            #     and self._meets_threshold(eval_result)
            #     and not self._meets_threshold(candidate_b_result)
            # )
            # if confirmation_missed:
            #     print("  [greedy] confirmation miss: slice A met the bars but slice B did not -- not a SUCCESS")
            confirmation_missed = False

            attempt.failed_at = FailureStage.EVALUATION
            attempt.error_summary = eval_result.get_feedback_summary(self.require_delimiters) + (
                " (confirmation miss: slice A met the bars, slice B did not)" if confirmation_missed else ""
            )
            attempts.append(attempt)
            self._save_progress_report(attempts, task_description, output_name, run_results_dir)

            if (
                (fixed_warm_continuation or history_only_continuation)
                and iteration + 1 == self.max_iterations
            ):
                logger.info(
                    "[warm-resume] final scheduled attempt=%d recorded; skipping "
                    "an unused follow-on refinement",
                    attempt.attempt_number,
                )
                continue

            self._unload_evaluator_runtime_before_refinement()
            candidate_feedback = self._threshold_miss_candidate_feedback(eval_result)
            threshold_mode_examples = eval_result._render_mode_examples()
            next_allowed_helpers, next_helper_status = self._compute_allowed_helpers(attempts)
            if next_helper_status:
                print(f"  Helper policy: {next_helper_status}")
            incumbent_attempt_number = self._incumbent.attempt_number if self._incumbent else None
            attempt_outcome_ledger = self._build_attempt_outcome_ledger(attempts, incumbent_attempt_number)
            inc = self._incumbent  # always set by this point (seeded at worst by this attempt)

            if became_incumbent:
                # This candidate IS the incumbent (seeded, or accepted but
                # still short of a full win) -- refine from its own slice-A
                # scores, no REJECTED framing since nothing was rejected.
                print("  Refining based on evaluation results...")
                strategy_code = self._refine_with_beam(
                    stage_label="evaluation_threshold",
                    previous_strategy=strategy_code,
                    allowed_helpers=next_allowed_helpers,
                    refine_once=lambda: self.generator.refine_after_evaluation_failure(
                        previous_strategy=inc.strategy_code,
                        previous_accuracy=inc.a_result.accuracy or 0.0,
                        previous_syntax_rate=inc.a_result.syntax_rate or 0.0,
                        num_examples=inc.a_result.num_examples or 0,
                        goal_accuracy=self.min_accuracy,
                        goal_syntax_rate=self.min_syntax_rate,
                        evaluation_feedback=candidate_feedback,
                        best_strategy=None,
                        best_accuracy=None,
                        best_syntax_rate=None,
                        allowed_helpers=next_allowed_helpers,
                        eval_max_seconds_per_example=self.eval_max_seconds_per_example,
                        mode_examples=threshold_mode_examples,
                        attempt_outcome_ledger=attempt_outcome_ledger,
                    ),
                )
            else:
                # REJECT: rebase onto the incumbent's own code/scores before
                # refining, and tell the author this candidate was rejected.
                print(
                    f"  [greedy] rebasing refinement on incumbent attempt "
                    f"{inc.attempt_number} (acc={inc.a_result.accuracy:.1%}, "
                    f"syn={inc.a_result.syntax_rate:.1%})"
                )
                rejected_feedback = self._render_rejected_candidate_block(
                    strategy_code, eval_result, candidate_feedback, inc
                )
                strategy_code = inc.strategy_code
                strategy_code = self._refine_with_beam(
                    stage_label="evaluation_threshold",
                    previous_strategy=strategy_code,
                    allowed_helpers=next_allowed_helpers,
                    refine_once=lambda: self.generator.refine_after_evaluation_failure(
                        previous_strategy=strategy_code,
                        previous_accuracy=inc.a_result.accuracy or 0.0,
                        previous_syntax_rate=inc.a_result.syntax_rate or 0.0,
                        num_examples=inc.a_result.num_examples or 0,
                        goal_accuracy=self.min_accuracy,
                        goal_syntax_rate=self.min_syntax_rate,
                        evaluation_feedback=rejected_feedback,
                        best_strategy=None,
                        best_accuracy=None,
                        best_syntax_rate=None,
                        allowed_helpers=next_allowed_helpers,
                        eval_max_seconds_per_example=self.eval_max_seconds_per_example,
                        mode_examples=threshold_mode_examples,
                        attempt_outcome_ledger=attempt_outcome_ledger,
                    ),
                )
            continue

        # All attempts exhausted
        total_time = (time.time() - start_time) * 1000

        if fixed_warm_continuation and self._incumbent is not None and self._meets_threshold(
            self._incumbent.a_result
        ):
            incumbent = self._incumbent
            winner_attempt = next(
                attempt
                for attempt in attempts
                if attempt.attempt_number == incumbent.attempt_number
            )
            winner_full_code = (
                winner_attempt.full_dafny_code
                or self.generator.inject_strategy(incumbent.strategy_code)
            )
            winner_compilation = compiler.compile(winner_full_code, output_name)
            if not winner_compilation.success or winner_compilation.main_module_path is None:
                raise SynthesisExhaustionError(
                    "fixed warm continuation could not compile its selected incumbent",
                    attempts,
                )
            logger.info(
                "[fixed-warm] selected paper winner attempt=%d after attempts=%d",
                incumbent.attempt_number,
                len(attempts),
            )
            self._save_success_report(
                incumbent.strategy_code,
                winner_full_code,
                winner_compilation,
                attempts,
                task_description,
                output_name,
                run_dir,
                run_dafny_dir,
                run_results_dir,
                incumbent.a_result,
            )
            return SynthesisResult(
                success=True,
                strategy_code=incumbent.strategy_code,
                full_dafny_code=winner_full_code,
                compiled_module_path=winner_compilation.main_module_path,
                output_dir=winner_compilation.output_dir,
                run_dir=run_dir,
                attempts=attempts,
                total_time_ms=total_time,
            )

        print(f"\n{'='*60}")
        print(f"FAILED after {self.max_iterations} attempts")
        print(f"Total time: {total_time:.1f}ms")
        print(f"{'='*60}")

        # Save failure report (always — reports are how runs are audited)
        report_path = self._save_failure_report(
            attempts,
            task_description,
            output_name,
            run_dir,
            run_results_dir,
        )

        if fixed_warm_continuation and self._incumbent is not None:
            incumbent = self._incumbent
            winner_attempt = next(
                attempt
                for attempt in attempts
                if attempt.attempt_number == incumbent.attempt_number
            )
            winner_full_code = (
                winner_attempt.full_dafny_code
                or self.generator.inject_strategy(incumbent.strategy_code)
            )
            logger.info(
                "[fixed-warm] selected below-bar incumbent attempt=%d after attempts=%d",
                incumbent.attempt_number,
                len(attempts),
            )
            return SynthesisResult(
                success=False,
                strategy_code=incumbent.strategy_code,
                full_dafny_code=winner_full_code,
                compiled_module_path=None,
                output_dir=None,
                run_dir=run_dir,
                attempts=attempts,
                total_time_ms=total_time,
            )

        # Best-accuracy fallback: if any attempt's accuracy beat min_accuracy
        # (even if syntax fell short of min_syntax_rate), save a side-channel
        # `fallback_winner.json` so this cell can be harvested as an accuracy-only
        # win in post-hoc analysis. Does NOT change exception semantics — the
        # subprocess still exits non-zero so existing flows aren't affected.
        fallback_winner = None
        for att in attempts:
            if (
                att.eval_result is not None
                and not att.eval_result.early_stopped
                and att.failed_at is not FailureStage.TIMEOUT
                and not self._exceeded_attempt_budget(att.eval_result)
                and att.eval_result.accuracy >= self.min_accuracy
                and (
                    fallback_winner is None
                    or att.eval_result.accuracy > fallback_winner.eval_result.accuracy
                )
            ):
                fallback_winner = att

        if fallback_winner is not None:
            fb_path = run_results_dir / "fallback_winner.json"
            try:
                with open(fb_path, "w") as f:
                    json.dump(
                        {
                            "fallback_reason": "accuracy_met_but_syntax_below_threshold",
                            "min_accuracy": self.min_accuracy,
                            "min_syntax_rate": self.min_syntax_rate,
                            "winner_attempt_number": fallback_winner.attempt_number,
                            "winner_accuracy": fallback_winner.eval_result.accuracy,
                            "winner_syntax_rate": fallback_winner.eval_result.syntax_rate,
                            "winner_strategy_code": fallback_winner.strategy_code,
                            "winner_full_dafny_code": fallback_winner.full_dafny_code,
                            "evaluation": fallback_winner.eval_result.to_dict(),
                        },
                        f,
                        indent=2,
                    )
                print(
                    f"[FALLBACK] accuracy-only winner: attempt "
                    f"{fallback_winner.attempt_number}, "
                    f"acc={fallback_winner.eval_result.accuracy:.1%} "
                    f"(syntax {fallback_winner.eval_result.syntax_rate:.1%} below "
                    f"required {self.min_syntax_rate:.1%}). Saved to {fb_path}."
                )
            except Exception as fb_err:
                print(f"[FALLBACK] Failed to write fallback_winner.json: {fb_err}")
        else:
            print(
                "[FALLBACK] No accuracy-only candidate found either "
                "(no attempt met the min_accuracy threshold); no fallback saved."
            )

        error = SynthesisExhaustionError(
            f"Synthesis failed after {self.max_iterations} attempts", attempts, report_path
        )

        print(error.get_failure_summary())
        raise error

    def _build_run_report(
        self,
        attempts: list[SynthesisAttempt],
        task_description: str,
        output_name: str,
    ) -> dict:
        """Build the report dict shared by the incremental progress report and
        the run-ending failure report -- same shape, just written at different
        times."""
        return {
            "run_configuration": self._run_configuration_metadata(task_description, output_name),
            "task_description": task_description,
            "total_attempts": len(attempts),
            "timestamp": datetime.now().isoformat(),
            "attempts": [attempt.to_dict() for attempt in attempts],
            "failure_patterns": self._analyze_failure_patterns(attempts),
        }

    def _save_progress_report(
        self,
        attempts: list[SynthesisAttempt],
        task_description: str,
        output_name: str,
        results_dir: Path,
    ) -> None:
        """Write every attempt finished so far to disk immediately.

        Called right after each attempt is appended to `attempts` (whether it
        succeeded, failed, or timed out) -- not only once at the very end of
        the run. This is what lets a killed or crashed run keep every
        attempt's per-example evaluation records instead of losing everything
        already computed.
        """
        if not self.save_reports:
            return
        progress_path = results_dir / "progress_report.json"
        try:
            with open(progress_path, "w") as f:
                json.dump(self._build_run_report(attempts, task_description, output_name), f, indent=2)
        except Exception as e:
            # Never let a progress-save hiccup take down the synthesis loop --
            # but never swallow it silently either.
            print(f"Warning: could not save incremental progress report: {e}")

    def _handle_attempt_timeout(
        self,
        attempt: "SynthesisAttempt",
        attempts: list["SynthesisAttempt"],
        elapsed_seconds: float,
        task_description: str,
        output_name: str,
        results_dir: Path,
    ) -> str:
        """Record `attempt` as timed out (not a crash, not a legitimate
        score), persist progress immediately, and return a fresh strategy so
        the loop moves on to the next attempt instead of dying."""
        print(
            f"  ✗ Attempt {attempt.attempt_number} exceeded the "
            f"{self.max_attempt_seconds:.0f}s attempt time cap "
            f"(ran {elapsed_seconds:.1f}s) — marking as timed out and moving on."
        )
        attempt.failed_at = FailureStage.TIMEOUT
        attempt.error_summary = (
            f"Attempt timed out: exceeded the {self.max_attempt_seconds:.0f}s wall-clock "
            f"budget for one attempt (ran {elapsed_seconds:.1f}s before being stopped)."
        )
        attempts.append(attempt)
        self._save_progress_report(attempts, task_description, output_name, results_dir)

        next_allowed_helpers, next_helper_status = self._compute_allowed_helpers(attempts)
        if next_helper_status:
            print(f"  Helper policy: {next_helper_status}")
        print("  Restarting with fresh generation after timeout...")
        return self.generator.generate_initial(
            task_description,
            allowed_helpers=next_allowed_helpers,
            force_open_span=self._force_open_span(),
        )

    def _save_failure_report(
        self,
        attempts: list[SynthesisAttempt],
        task_description: str,
        output_name: str,
        run_dir: Path,
        results_dir: Path,
    ) -> Path:
        """Save a detailed failure report to disk."""
        report_path = results_dir / "failure_report.json"

        report = self._build_run_report(attempts, task_description, output_name)

        _write_json_atomic(report_path, report)

        print(f"Failure report saved to: {report_path}")

        # Create 'latest' symlink in the generated directory even on failure
        try:
            latest_link = self.output_dir / "latest"
            if latest_link.exists() or latest_link.is_symlink():
                latest_link.unlink()
            latest_link.symlink_to(run_dir.name, target_is_directory=True)
            print(f"Latest run link (failed) updated: {latest_link}")
        except Exception as e:
            print(f"Warning: Could not create 'latest' symlink: {e}")

        return report_path

    def _save_success_report(
        self,
        strategy_code: str,
        full_code: str,
        compilation_result: CompilationResult,
        attempts: list[SynthesisAttempt],
        task_description: str,
        output_name: str,
        run_dir: Path,
        dafny_dir: Path,
        results_dir: Path,
        evaluation_result: EvaluationResult,
    ) -> None:
        """Save a success report and the final strategy."""
        # Save the Dafny source
        dafny_path = dafny_dir / f"{output_name}.dfy"
        with open(dafny_path, "w") as f:
            f.write(full_code)
        canonical_dafny_path = dafny_dir / "GeneratedCSD.dfy"
        with open(canonical_dafny_path, "w") as f:
            f.write(full_code)

        # NOTE: We do NOT overwrite synthesis/verify/library/GeneratedCSD.dfy
        # here because it contains
        # the template markers (QWEN_INSERT_STRATEGY_HERE) needed for future runs.
        # The final Dafny code is saved in the run directory instead.

        rationale_extracted = extract_rationale(strategy_code)

        # Save a report
        report_path = results_dir / "success_report.json"
        report = {
            "run_configuration": self._run_configuration_metadata(
                task_description=task_description,
                output_name=output_name,
            ),
            "strategy_code": strategy_code,
            "tool_choice_rationale": rationale_extracted.rationale,
            "dafny_file": str(dafny_path),
            "dafny_file_canonical": str(canonical_dafny_path),
            "compiled_dir": str(compilation_result.output_dir),
            "total_attempts": len(attempts),
            "timestamp": datetime.now().isoformat(),
            "evaluation_result": evaluation_result.to_dict(),
            "sample_outputs": evaluation_result.sample_outputs,
        }

        _write_json_atomic(report_path, report)

        print(f"Strategy saved to: {dafny_path}")
        print(f"Success report saved to: {report_path}")

        # Create 'latest' symlink in the generated directory
        try:
            latest_link = self.output_dir / "latest"
            if latest_link.exists() or latest_link.is_symlink():
                latest_link.unlink()
            latest_link.symlink_to(run_dir.name, target_is_directory=True)
            print(f"Latest run link updated: {latest_link}")
        except Exception as e:
            print(f"Warning: Could not create 'latest' symlink: {e}")

    def _analyze_failure_patterns(self, attempts: list[SynthesisAttempt]) -> dict:
        """Analyze common failure patterns across attempts."""
        patterns = {
            "search_contract_failures": 0,
            "verification_failures": 0,
            "compilation_failures": 0,
            "runtime_failures": 0,
            "common_errors": [],
        }

        error_counts: dict[str, int] = {}

        for attempt in attempts:
            if attempt.failed_at == FailureStage.SEARCH_CONTRACT:
                patterns["search_contract_failures"] += 1
            elif attempt.failed_at == FailureStage.VERIFICATION:
                patterns["verification_failures"] += 1
            elif attempt.failed_at == FailureStage.COMPILATION:
                patterns["compilation_failures"] += 1
            elif attempt.failed_at == FailureStage.RUNTIME:
                patterns["runtime_failures"] += 1

            # Extract key error phrases
            if attempt.error_summary:
                if "GuaranteesValidOutput" in attempt.error_summary:
                    error_counts["GuaranteesValidOutput lemma failed"] = error_counts.get(
                        "GuaranteesValidOutput lemma failed", 0
                    ) + 1
                if "Free" in attempt.error_summary:
                    error_counts["Uses Free without fallback"] = error_counts.get(
                        "Uses Free without fallback", 0
                    ) + 1
                if "type" in attempt.error_summary.lower():
                    error_counts["Type error"] = error_counts.get("Type error", 0) + 1

        patterns["common_errors"] = [
            {"error": error, "count": count}
            for error, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True)
        ]

        return patterns
