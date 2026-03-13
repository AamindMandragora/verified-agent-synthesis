"""
Main synthesis pipeline with feedback-based refinement.

Orchestrates the generate -> verify -> compile -> run loop with
iterative refinement based on errors.
"""

import json
import secrets
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

from .compiler import CompilationResult, DafnyCompiler
from .evaluator import Evaluator, EvaluationResult
from .generator import StrategyGenerator
from .rationale import extract_rationale
from .runner import RuntimeResult, StrategyRunner
from .verifier import DafnyVerifier, VerificationResult


class FailureStage(Enum):
    """Stage where synthesis attempt failed."""

    VERIFICATION = "verification"
    COMPILATION = "compilation"
    RUNTIME = "runtime"
    EVALUATION = "evaluation"


def repair_verification_strategy(strategy_code: str, error_summary: str) -> tuple[str, bool]:
    """
    Apply known fixes to strategy code when verification fails with specific errors.
    Returns (repaired_code, True) if any fix was applied, else (strategy_code, False).
    """
    import re
    repaired = strategy_code
    changed = False

    # Fix: "Duplicate local-variable name: stepsLeft" -> template already declares it; remove duplicate
    if "Duplicate local-variable name: stepsLeft" in error_summary:
        # Remove lines that are just "var stepsLeft := maxSteps;" (any spacing)
        line_pattern = re.compile(r"^\s*var\s+stepsLeft\s*:=\s*maxSteps\s*;\s*$", re.IGNORECASE | re.MULTILINE)
        new_repaired = line_pattern.sub("", repaired)
        if new_repaired != repaired:
            repaired = new_repaired
            changed = True

    # Fix: "Duplicate local-variable name: helpers" -> template already declares "var helpers := new CSDHelpers();"; remove duplicate
    if "Duplicate local-variable name: helpers" in error_summary:
        helpers_line = re.compile(r"^\s*var\s+helpers\s*:=\s*new\s+CSDHelpers\s*\(\s*\)\s*;\s*$", re.IGNORECASE | re.MULTILINE)
        new_repaired = helpers_line.sub("", repaired)
        if new_repaired != repaired:
            repaired = new_repaired
            changed = True

    # Fix: "member 'IsValidNextToken' does not exist" -> use ValidNextToken
    if "IsValidNextToken" in error_summary or "isvalidnexttoken" in error_summary.lower():
        if "IsValidNextToken" in repaired:
            repaired = repaired.replace("IsValidNextToken", "ValidNextToken")
            changed = True

    # Fix: "type seq<...> does not have a member Length" -> Dafny uses |seq| for length
    if "Length" in error_summary and ("member" in error_summary.lower() or "does not have" in error_summary):
        length_pattern = re.compile(r"(\w+)\.Length\b")
        new_repaired = length_pattern.sub(r"|\1|", repaired)
        if new_repaired != repaired:
            repaired = new_repaired
            changed = True

    # Fix: "does not have a member Exists" / "type seq ... Exists" -> use Dafny exists quantifier
    if "Exists" in error_summary and ("member" in error_summary.lower() or "type" in error_summary.lower()):
        pattern = re.compile(
            r"(\w+)\.Exists\s*\(\s*(\w+)\s*=>\s*parser\.ValidNextToken\s*\(\s*(\w+)\s*,\s*\2\s*\)\s*\)",
            re.IGNORECASE
        )
        def repl(m):
            seq_var, tok_var, prefix_var = m.group(1), m.group(2), m.group(3)
            return f"(exists {tok_var} :: {tok_var} in {seq_var} && parser.ValidNextToken({prefix_var}, {tok_var}))"
        new_repaired = pattern.sub(repl, repaired)
        if new_repaired != repaired:
            repaired = new_repaired
            changed = True
        if ".Exists(" in repaired and not changed:
            repaired = re.sub(
                r"(\w+)\.Exists\s*\(\s*(\w+)\s*=>\s*([^)]+)\)\s*\)",
                r"(exists \2 :: \2 in \1 && \3)",
                repaired
            )
            changed = ".Exists(" not in repaired or repaired != strategy_code

    # Fix: "decreases expression might not decrease" — usually caused by resetting stepsLeft in the loop (e.g. stepsLeft := maxSteps).
    # Replace the else branch that resets stepsLeft with a ConstrainedStep so the loop always decreases stepsLeft.
    if "decreases expression might not decrease" in error_summary and "stepsLeft := maxSteps" in repaired:
        else_block = re.compile(
            r"\s*else\s*\{\s*(?:\s*//[^\n]*\n)*\s*"
            r"generated\s*:=\s*helpers\.RollbackToValidPrefix\s*\(\s*parser\s*,\s*generated\s*\)\s*;\s*"
            r"\s*stepsLeft\s*:=\s*maxSteps\s*;\s*"
            r"\s*\}",
            re.MULTILINE,
        )
        replacement = (
            " else {\n"
            "    CSDHelpers.RollbackPreservesTokenInvariant(lm, parser, generated);\n"
            "    var next, newSteps := helpers.ConstrainedStep(lm, parser, prompt, generated, stepsLeft);\n"
            "    generated := generated + [next];\n"
            "    stepsLeft := newSteps;\n"
            "  }"
        )
        new_repaired = else_block.sub(replacement, repaired)
        if new_repaired != repaired:
            repaired = new_repaired
            changed = True

    # Fix: "invariant could not be proved" for |generated| + stepsLeft == maxSteps
    # RollbackToValidPrefix in the loop changes |generated| without changing stepsLeft, breaking the invariant.
    # Remove the rollback call from inside the loop so only ConstrainedStep updates generated/stepsLeft.
    if "invariant could not be proved" in error_summary and "stepsLeft == maxSteps" in error_summary:
        if "RollbackToValidPrefix" in repaired:
            # Remove line(s) that assign generated from RollbackToValidPrefix inside the loop body
            repaired = re.sub(
                r"\s*generated\s*:=\s*helpers\.RollbackToValidPrefix\s*\(\s*parser\s*,\s*generated\s*\)\s*;\s*",
                "\n",
                repaired,
            )
            changed = True

    # Fix: "unresolved identifier: stepCounter" or "unresolved identifier: hasValid" — declare before the while loop
    for var_name, default in [("stepCounter", "0"), ("hasValid", "false")]:
        if f"unresolved identifier: {var_name}" in error_summary and var_name in repaired:
            decl = f"var {var_name} := {default};"
            if f"var {var_name}" not in repaired:
                # Insert after "generated := [];" so the variable is in scope for the loop
                new_repaired = re.sub(
                    r"(generated\s*:=\s*\[\]\s*;\s*\n)",
                    r"\1  " + decl + "\n",
                    repaired,
                    count=1,
                )
                if new_repaired != repaired:
                    repaired = new_repaired
                    changed = True
                    break

    # Fix: "unresolved identifier: next" / "unresolved identifier: newSteps" — LLM declares var next, newSteps inside if/else or at loop top so they're out of scope. Declare at start of while body and assign in each branch.
    has_next_scope_error = (
        "unresolved identifier: next" in error_summary
        or "unresolved identifier: newSteps" in error_summary
        or ("unresolved identifier" in error_summary and " next" in error_summary and " newSteps" in error_summary)
    )
    has_next_assignments = "next, newSteps :=" in repaired or "var next, newSteps := helpers." in repaired
    if has_next_scope_error and has_next_assignments:
        # Replace "var next, newSteps :=" with "next, newSteps :=" everywhere so we assign to outer variables
        new_repaired = re.sub(r"\bvar\s+next\s*,\s*newSteps\s*:=", "next, newSteps :=", repaired)
        if new_repaired != repaired:
            repaired = new_repaired
            changed = True
        # Insert declaration at start of while loop body so next/newSteps are in scope for all branches
        if "var next: Token" not in repaired and "var newSteps: nat" not in repaired:
            # Match: while ... { \n then indent of first statement
            loop_body_start = re.compile(
                r"(while\s+stepsLeft\s*>\s*0\s*&&\s*!parser\.IsCompletePrefix\s*\(generated\)[^\n]*\n"
                r"(?:\s*invariant[^\n]*\n)*\s*decreases\s+stepsLeft\s*)\n(\s*\{\s*\n)(\s+)",
                re.MULTILINE,
            )
            def insert_decl_after_brace(m: re.Match) -> str:
                pre, brace_newline, indent = m.group(1), m.group(2), m.group(3)
                return f"{pre}\n{brace_newline}{indent}var next: Token; var newSteps: nat;\n{indent}"
            new_repaired = loop_body_start.sub(insert_decl_after_brace, repaired, count=1)
            if new_repaired != repaired:
                repaired = new_repaired
                changed = True
            else:
                # Fallback: insert before the first "if (" in the strategy
                match = re.search(r"^(\s*)(if\s*\()", repaired, re.MULTILINE)
                if match:
                    indent = match.group(1)
                    repaired = repaired.replace(match.group(0), f"{indent}var next: Token; var newSteps: nat;\n{match.group(0)}", 1)
                    changed = True

    # Fix: "precondition for this call could not be proved" when "stepsLeft >= 1" — guard Step calls so we only call when stepsLeft >= 1.
    if "precondition for this call could not be proved" in error_summary and "stepsLeft >= 1" in error_summary:
        # Wrap the 3-line block (Step call; generated := ...; stepsLeft := newSteps;) in "if stepsLeft >= 1 { ... } else { break; }"
        step_block = re.compile(
            r"^(\s*)((?:var\s+)?next\s*,\s*newSteps\s*:=\s*helpers\.(?:ConstrainedStep|UnconstrainedStep)\s*\([^)]*\)\s*;\s*\n"
            r"\s*generated\s*:=\s*generated\s*\+\s*\[next\]\s*;\s*\n"
            r"\s*stepsLeft\s*:=\s*newSteps\s*;\s*)$",
            re.MULTILINE,
        )
        def wrap_steps_guard(m: re.Match) -> str:
            indent, block = m.group(1), m.group(2)
            inner = "\n".join(f"{indent}  {line.strip()}" for line in block.strip().split("\n"))
            return f"{indent}if stepsLeft >= 1 {{\n{inner}\n{indent}}} else {{ break; }}\n"
        new_repaired = step_block.sub(wrap_steps_guard, repaired)
        if new_repaired != repaired:
            repaired = new_repaired
            changed = True

    # Fix: "precondition for this call could not be proved" for ConstrainedStep (ValidNextTokens in lm.Tokens)
    # Insert lemma call CSDHelpers.RollbackPreservesTokenInvariant(lm, parser, generated) before ConstrainedStep if missing.
    if "precondition for this call could not be proved" in error_summary and "ValidNextTokens" in error_summary:
        lemma_call = "CSDHelpers.RollbackPreservesTokenInvariant(lm, parser, generated);"
        if "RollbackPreservesTokenInvariant" not in repaired and "ConstrainedStep" in repaired:
            def insert_lemma_before_line(match: re.Match) -> str:
                indent, full_line = match.group(1), match.group(2)
                return f"{indent}{lemma_call}\n{indent}{full_line}"
            # Match "var next, newSteps :=" or "next, newSteps :=" (after scope fix) + helpers.ConstrainedStep(...);
            for pattern in [
                re.compile(r"^(\s*)(var\s+next\s*,\s*newSteps\s*:=\s*helpers\.ConstrainedStep\s*\([^)]*\)\s*;\s*)$", re.MULTILINE),
                re.compile(r"^(\s*)(next\s*,\s*newSteps\s*:=\s*helpers\.ConstrainedStep\s*\([^)]*\)\s*;\s*)$", re.MULTILINE),
            ]:
                new_repaired = pattern.sub(insert_lemma_before_line, repaired, count=1)
                if new_repaired != repaired:
                    repaired = new_repaired
                    changed = True
                    break

    # Fix: "the method returns 1 value but is assigned to 0 variable" for RollbackToValidPrefix — LLM calls it without assigning. Must assign: generated := ...RollbackToValidPrefix(...)
    if "returns 1 value but is assigned to 0 variable" in error_summary and "RollbackToValidPrefix" in repaired:
        # Standalone call "helpers.RollbackToValidPrefix(...)" or "CSDHelpers.RollbackToValidPrefix(...)" -> assign to generated
        for prefix in [r"helpers", r"CSDHelpers"]:
            standalone_rollback = re.compile(
                rf"^(\s*){re.escape(prefix)}\.RollbackToValidPrefix\s*\(\s*parser\s*,\s*generated\s*\)\s*;(?:\s*//[^\n]*)?\s*$",
                re.MULTILINE,
            )
            def make_assign(pfx: str):
                def assign_rollback(m: re.Match) -> str:
                    return f"{m.group(1)}generated := {pfx}.RollbackToValidPrefix(parser, generated);"
                return assign_rollback
            new_repaired = standalone_rollback.sub(make_assign(prefix), repaired)
            if new_repaired != repaired:
                repaired = new_repaired
                changed = True
                break

    # Fix: "invariant could not be proved" for stepCounter <= maxSteps — stepCounter is redundant with stepsLeft; remove the invariant so verification can succeed.
    if "invariant could not be proved" in error_summary and "stepCounter" in error_summary and "maxSteps" in error_summary:
        new_repaired = re.sub(
            r"\s*invariant\s+stepCounter\s*>=\s*0\s*&&\s*stepCounter\s*<=\s*maxSteps\s*\n",
            "\n",
            repaired,
        )
        if new_repaired != repaired:
            repaired = new_repaired
            changed = True

    # Fix: "decreases expression might not decrease" when using "decreases maxSteps - |generated|" — use stepsLeft as variant instead.
    if "decreases expression might not decrease" in error_summary and "decreases maxSteps - |generated|" in repaired:
        new_repaired = re.sub(
            r"decreases\s+maxSteps\s*-\s*\|generated\|\s*",
            "decreases stepsLeft ",
            repaired,
        )
        if new_repaired != repaired:
            repaired = new_repaired
            changed = True

    # Fix: "invariant could not be proved" for parser.IsValidPrefix(generated) when loop uses UnconstrainedStep — use IsPermissive + lemma so else branch only runs when parser allows any token.
    if "invariant could not be proved" in error_summary and "IsValidPrefix(generated)" in error_summary and "UnconstrainedStep" in repaired and "ConstrainedStep" in repaired:
        # (1) Insert RollbackPreservesTokenInvariant at loop start if missing
        if "RollbackPreservesTokenInvariant" not in repaired:
            loop_start = re.compile(r"(while\s+stepsLeft\s*>\s*0\s*&&\s*!parser\.IsCompletePrefix\s*\(generated\)\s*\n(?:\s*invariant[^\n]*\n)*\s*decreases\s+stepsLeft\s*)\n(\s*\{)\n(\s*)", re.MULTILINE)
            def add_lemma_at_loop_start(m: re.Match) -> str:
                pre, brace, indent = m.group(1), m.group(2), m.group(3)
                return f"{pre}\n{brace}\n{indent}CSDHelpers.RollbackPreservesTokenInvariant(lm, parser, generated);\n{indent}"
            new_repaired = loop_start.sub(add_lemma_at_loop_start, repaired, count=1)
            if new_repaired != repaired:
                repaired = new_repaired
                changed = True
        # (2) Insert UnconstrainedPreservesValidWhenPermissive(parser, generated, next) after UnconstrainedStep in else branch (so Dafny can prove invariant after generated := generated + [next])
        if "UnconstrainedPreservesValidWhenPermissive" not in repaired:
            unconstrained_line = re.compile(
                r"^(\s*)(var\s+next\s*,\s*newSteps\s*:=\s*helpers\.UnconstrainedStep\s*\([^)]*\)\s*;\s*)\n",
                re.MULTILINE,
            )
            def add_lemma_after_unconstrained(m: re.Match) -> str:
                indent, line = m.group(1), m.group(2)
                return f"{indent}{line}\n{indent}CSDHelpers.UnconstrainedPreservesValidWhenPermissive(parser, generated, next);\n"
            new_repaired = unconstrained_line.sub(add_lemma_after_unconstrained, repaired, count=1)
            if new_repaired != repaired:
                repaired = new_repaired
                changed = True
        # (3) Guard ConstrainedStep branch with !parser.IsPermissive(generated) so else (UnconstrainedStep) is only taken when parser is permissive
        if "IsPermissive(generated)" not in repaired:
            for old_cond, new_cond in [
                (r"if\s*\((hasValid\s*\|\|\s*stepCounter\s*%\s*2\s*==\s*0)\)\s*\{", r"if (!parser.IsPermissive(generated) || (hasValid || stepCounter % 2 == 0)) {"),
                (r"if\s*\((hasValid)\)\s*\{", r"if (!parser.IsPermissive(generated) || (hasValid)) {"),
                (r"if\s+hasValid\s*\|\|\s*stepCounter\s*%\s*2\s*==\s*0\s*\{", r"if !parser.IsPermissive(generated) || hasValid || stepCounter % 2 == 0 {"),
            ]:
                new_repaired = re.sub(old_cond, new_cond, repaired, count=1)
                if new_repaired != repaired:
                    repaired = new_repaired
                    changed = True
                    break

    return repaired, changed


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

    # Detect which primitives are used (VerifiedAgentSynthesis.dfy only has these three)
    uses_constrained = "ConstrainedStep" in strategy_code_for_match
    uses_unconstrained = "UnconstrainedStep" in strategy_code_for_match
    uses_rollback = "RollbackToValidPrefix" in strategy_code_for_match

    if uses_constrained and uses_unconstrained:
        category = "interleaved"
    elif uses_constrained:
        category = "constrained_only"
    elif uses_unconstrained:
        category = "unconstrained_only"
    else:
        category = "unknown"

    return {
        "strategy_name": "CustomLoop",
        "parameters": {"uses_rollback": uses_rollback},
        "category": category,
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
    runtime_result: Optional[RuntimeResult] = None
    eval_result: Optional[EvaluationResult] = None

    # Failure information
    failed_at: Optional[FailureStage] = None
    error_summary: str = ""

    def succeeded(self) -> bool:
        """Check if this attempt succeeded completely."""
        return (
            self.verification_result is not None
            and self.verification_result.success
            and self.compilation_result is not None
            and self.compilation_result.success
            and self.runtime_result is not None
            and self.runtime_result.success
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
            "runtime": {
                "success": self.runtime_result.success if self.runtime_result else None,
                "output_length": len(self.runtime_result.output)
                if self.runtime_result and self.runtime_result.output
                else 0,
                "cost": self.runtime_result.cost if self.runtime_result else 0,
                "execution_time_ms": self.runtime_result.execution_time_ms if self.runtime_result else 0,
            }
            if self.runtime_result
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

        return "\n".join(lines)


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

    DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent / "outputs" / "generated-csd"

    def __init__(
        self,
        evaluator: Evaluator,
        generator: Optional[StrategyGenerator] = None,
        verifier: Optional[DafnyVerifier] = None,
        compiler: Optional[DafnyCompiler] = None,
        runner: Optional[StrategyRunner] = None,
        max_iterations: int = 5,
        output_dir: Optional[Path] = None,
        save_reports: bool = True,
        # Evaluation thresholds
        min_accuracy: float = 0.0,
        min_format_rate: float = 0.0,
        min_syntax_rate: float = 0.0,
        eval_sample_size: int = 1,
    ):
        """
        Initialize the synthesis pipeline.

        Args:
            evaluator: Evaluator for dataset-based feedback (required)
            generator: Strategy generator (creates default if None)
            verifier: Dafny verifier (creates default if None)
            compiler: Dafny compiler (creates default if None)
            runner: Strategy runner (creates default if None)
            max_iterations: Maximum refinement iterations
            output_dir: Directory for outputs and reports
            save_reports: Whether to save failure reports to disk
            min_accuracy: Minimum accuracy threshold for evaluation
            min_format_rate: Minimum format validity rate threshold
            min_syntax_rate: Minimum syntax validity rate threshold
            eval_sample_size: Number of examples to evaluate on
        """
        self.evaluator = evaluator
        self.generator = generator or StrategyGenerator()
        self.verifier = verifier or DafnyVerifier()
        self.compiler = compiler or DafnyCompiler()
        self.runner = runner  # Will be created per-task in synthesize()
        self.max_iterations = max_iterations
        self.output_dir = output_dir or self.DEFAULT_OUTPUT_DIR
        self.save_reports = save_reports

        # Evaluation thresholds
        self.min_accuracy = min_accuracy
        self.min_format_rate = min_format_rate
        self.min_syntax_rate = min_syntax_rate
        self.eval_sample_size = eval_sample_size

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def synthesize(
        self,
        task_description: str,
        output_name: str = "generated_csd",
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

        start_time = time.time()
        attempts: list[SynthesisAttempt] = []

        # Create runner if not already provided
        if self.runner is None:
            runner = StrategyRunner(parser_mode="permissive")
        else:
            runner = self.runner

        # Create an isolated output directory for this run
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + secrets.token_hex(3)
        run_dir = self.output_dir / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        # Update a convenience pointer to the most recent run
        try:
            (self.output_dir / "latest_run.txt").write_text(str(run_dir) + "\n")
        except Exception:
            pass

        # Use a per-run compiler output directory.
        compiler = DafnyCompiler(
            dafny_path=self.compiler.dafny_path,
            output_dir=run_dir,
            timeout=self.compiler.timeout,
            extra_args=list(self.compiler.extra_args),
        )

        # Initial generation
        print(f"Generating initial strategy for: {task_description}")
        strategy_code = self.generator.generate_initial(task_description)

        for iteration in range(self.max_iterations):
            attempt_num = iteration + 1
            print(f"\n{'='*60}")
            print(f"Attempt {attempt_num}/{self.max_iterations}")
            print(f"{'='*60}")
            print(f"Strategy: {strategy_code}")

            # Create full Dafny code
            full_code = self.generator.inject_strategy(strategy_code)

            # Create attempt record
            attempt = SynthesisAttempt(
                attempt_number=attempt_num,
                strategy_code=strategy_code,
                full_dafny_code=full_code,
                timestamp=datetime.now().isoformat(),
            )

            # Stage 1: Verification
            print("\n[1/4] Verifying with Dafny...")
            verification_result = self.verifier.verify(full_code)
            attempt.verification_result = verification_result

            if not verification_result.success:
                print("  ✗ Verification failed")
                error_summary = verification_result.get_error_summary()
                print(error_summary)
                attempt.failed_at = FailureStage.VERIFICATION
                attempt.error_summary = error_summary
                attempts.append(attempt)

                # Check if we're stuck on the same error repeatedly
                error_msg = verification_result.get_error_summary()
                consecutive_same = 0
                for prev in reversed(attempts[:-1]):
                    if prev.failed_at == FailureStage.VERIFICATION and prev.error_summary == error_msg:
                        consecutive_same += 1
                    else:
                        break

                if consecutive_same >= 2:
                    # After 3+ identical errors, prepend strong guidance
                    error_msg = (
                        f"WARNING: This is the SAME error for {consecutive_same + 1} consecutive attempts. "
                        f"Your previous fixes did NOT work. You MUST use a COMPLETELY DIFFERENT approach. "
                        f"The ONLY methods on helpers are: UnconstrainedStep, ConstrainedStep, and "
                        f"(static) RollbackToValidPrefix. Implement a loop that calls these; do NOT call any other method.\n\n"
                        f"Original error:\n{error_msg}"
                    )

                # Try automatic repair for known errors (apply until no change so multiple fixes apply in one go)
                pre_repair_code = strategy_code
                while True:
                    strategy_code, repair_changed = repair_verification_strategy(strategy_code, error_msg)
                    if not repair_changed:
                        break
                if strategy_code != pre_repair_code:
                    print("  Applied automatic fix (e.g. duplicate stepsLeft / Rollback assignment / precondition); re-verifying...")
                    continue
                # Refine based on verification error via model
                print("  Refining based on verification error...")
                strategy_code = self.generator.refine_after_verification_error(
                    strategy_code, error_msg
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

                # Refine based on compilation error
                print("  Refining based on compilation error...")
                strategy_code = self.generator.refine_after_compilation_error(
                    strategy_code, compilation_result.get_error_summary()
                )
                continue

            print(f"  ✓ Compiled to {compilation_result.output_dir}")

            # Save Dafny source into the run directory (overwritten each successful compile)
            dafny_path = run_dir / f"{output_name}.dfy"
            dafny_path.write_text(full_code, encoding="utf-8")
            print(f"  Dafny CSD saved to: {dafny_path}")

            # Stage 3: Runtime test
            print("\n[3/4] Testing runtime execution...")

            if compilation_result.main_module_path is None:
                print("  ✗ No main module found")
                attempt.failed_at = FailureStage.RUNTIME
                attempt.error_summary = "No main module path in compilation result"
                attempts.append(attempt)

                strategy_code = self.generator.refine_after_runtime_error(
                    strategy_code,
                    "Compilation succeeded but no Python module was generated",
                )
                continue

            runtime_result = runner.run(compilation_result.main_module_path)
            attempt.runtime_result = runtime_result

            if not runtime_result.success:
                print(f"  ✗ Runtime error: {runtime_result.error_type}")
                attempt.failed_at = FailureStage.RUNTIME
                attempt.error_summary = runtime_result.get_error_summary()
                attempts.append(attempt)

                # Refine based on runtime error
                print("  Refining based on runtime error...")
                strategy_code = self.generator.refine_after_runtime_error(
                    strategy_code, runtime_result.get_error_summary()
                )
                continue

            print(f"  ✓ Execution successful ({runtime_result.execution_time_ms:.1f}ms)")
            print(f"  Output length: {len(runtime_result.output or [])} tokens")

            # Stage 4: Evaluation — use same device as generator to avoid loading on a full GPU
            if getattr(self.generator, "device", None):
                self.evaluator.device = self.generator.device
            print("\n[4/4] Evaluating on dataset sample...")
            eval_result = self.evaluator.evaluate_sample(
                compiled_module_path=compilation_result.main_module_path,
                sample_size=self.eval_sample_size,
            )
            attempt.eval_result = eval_result

            if not eval_result.success:
                print(f"  ✗ Evaluation failed: {eval_result.error}")
                if eval_result.sample_outputs:
                    print(eval_result.get_detailed_samples(max_samples=2))
                attempt.failed_at = FailureStage.EVALUATION
                attempt.error_summary = eval_result.error or "Evaluation failed"
                attempts.append(attempt)

                print("  Refining based on evaluation error...")
                strategy_code = self.generator.refine_after_evaluation_failure(
                    strategy_code, eval_result.get_feedback_summary()
                )
                continue

            # Check if evaluation meets thresholds
            if not eval_result.meets_threshold(
                min_accuracy=self.min_accuracy,
                min_format_rate=self.min_format_rate,
                min_syntax_rate=self.min_syntax_rate,
            ):
                print(f"  ✗ Evaluation below threshold:")
                print(f"    Accuracy: {eval_result.accuracy:.1%} (min: {self.min_accuracy:.1%})")
                print(f"    Format: {eval_result.format_rate:.1%} (min: {self.min_format_rate:.1%})")
                print(f"    Syntax: {eval_result.syntax_rate:.1%} (min: {self.min_syntax_rate:.1%})")
                print(eval_result.get_detailed_samples(max_samples=3))
                attempt.failed_at = FailureStage.EVALUATION
                attempt.error_summary = eval_result.get_feedback_summary()
                attempts.append(attempt)

                print("  Refining based on evaluation results...")
                strategy_code = self.generator.refine_after_evaluation_failure(
                    strategy_code, eval_result.get_feedback_summary()
                )
                continue

            print(f"  ✓ Evaluation passed:")
            print(f"    Accuracy: {eval_result.accuracy:.1%}")
            print(f"    Format: {eval_result.format_rate:.1%}")
            print(f"    Syntax: {eval_result.syntax_rate:.1%}")

            # Success!
            attempts.append(attempt)
            total_time = (time.time() - start_time) * 1000

            print(f"\n{'='*60}")
            print(f"SUCCESS after {attempt_num} attempt(s)")
            print(f"Total time: {total_time:.1f}ms")
            print(f"{'='*60}")

            # Save successful strategy
            self._save_success_report(
                strategy_code, full_code, compilation_result, attempts, output_name, run_dir
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

        # All attempts exhausted
        total_time = (time.time() - start_time) * 1000

        print(f"\n{'='*60}")
        print(f"FAILED after {self.max_iterations} attempts")
        print(f"Total time: {total_time:.1f}ms")
        print(f"{'='*60}")

        # Save failure report
        report_path = None
        if self.save_reports:
            report_path = self._save_failure_report(attempts, task_description, run_dir)

        error = SynthesisExhaustionError(
            f"Synthesis failed after {self.max_iterations} attempts", attempts, report_path
        )

        print(error.get_failure_summary())
        raise error

    def _save_failure_report(self, attempts: list[SynthesisAttempt], task_description: str, run_dir: Path) -> Path:
        """Save a detailed failure report to disk."""
        report_path = run_dir / "failure_report.json"

        report = {
            "task_description": task_description,
            "total_attempts": len(attempts),
            "timestamp": datetime.now().isoformat(),
            "attempts": [attempt.to_dict() for attempt in attempts],
            "failure_patterns": self._analyze_failure_patterns(attempts),
        }

        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        print(f"Failure report saved to: {report_path}")

        # Create 'latest' symlink in the runs directory even on failure
        try:
            latest_link = run_dir.parent / "latest"
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
        output_name: str,
        run_dir: Path,
    ) -> None:
        """Save a success report and the final strategy."""
        # Save the Dafny source
        dafny_path = run_dir / f"{output_name}.dfy"
        with open(dafny_path, "w") as f:
            f.write(full_code)

        # NOTE: We do NOT overwrite dafny/GeneratedCSD.dfy here because it contains
        # the template markers (QWEN_INSERT_STRATEGY_HERE) needed for future runs.
        # The final Dafny code is saved in the run directory instead.

        rationale_extracted = extract_rationale(strategy_code)

        # Save a report
        report_path = run_dir / "success_report.json"
        report = {
            "strategy_code": strategy_code,
            "tool_choice_rationale": rationale_extracted.rationale,
            "dafny_file": str(dafny_path),
            "compiled_dir": str(compilation_result.output_dir),
            "total_attempts": len(attempts),
            "timestamp": datetime.now().isoformat(),
        }

        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        print(f"Strategy saved to: {dafny_path}")
        print(f"Success report saved to: {report_path}")

        # Create 'latest' symlink in the runs directory
        try:
            latest_link = run_dir.parent / "latest"
            if latest_link.exists() or latest_link.is_symlink():
                latest_link.unlink()
            latest_link.symlink_to(run_dir.name, target_is_directory=True)
            print(f"Latest run link updated: {latest_link}")
        except Exception as e:
            print(f"Warning: Could not create 'latest' symlink: {e}")

    def _analyze_failure_patterns(self, attempts: list[SynthesisAttempt]) -> dict:
        """Analyze common failure patterns across attempts."""
        patterns = {
            "verification_failures": 0,
            "compilation_failures": 0,
            "runtime_failures": 0,
            "common_errors": [],
        }

        error_counts: dict[str, int] = {}

        for attempt in attempts:
            if attempt.failed_at == FailureStage.VERIFICATION:
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


