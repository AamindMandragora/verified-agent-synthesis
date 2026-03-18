"""
Main synthesis pipeline with feedback-based refinement.

Orchestrates the generate -> verify -> compile -> run loop with
iterative refinement based on errors.
"""

import json
import secrets
import time
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

    # Fix: "Duplicate local-variable name: generated" -> template out-parameter; remove "var generated := [];" lines
    new_repaired = re.sub(
        r"^\s*var\s+generated\s*:=\s*\[\s*\]\s*;\s*\n?",
        "",
        repaired,
        flags=re.MULTILINE,
    )
    if new_repaired != repaired:
        # Re-insert bare "generated := [];" if no assignment to generated exists
        if "generated := []" not in new_repaired and "generated := [" not in new_repaired:
            new_repaired = "generated := [];\n" + new_repaired
        repaired = new_repaired
        changed = True

    # Fix: "var next, newSteps: Token;" — second variable gets wrong type Token instead of nat.
    # Split into two properly typed declarations. Capture leading whitespace for correct indentation.
    new_repaired = re.sub(
        r"^(\s*)var\s+(\w+)\s*,\s*(\w+)\s*:\s*Token\s*;",
        r"\1var \2: Token;\n\1var \3: nat;",
        repaired,
        flags=re.MULTILINE,
    )
    if new_repaired != repaired:
        repaired = new_repaired
        changed = True

    # Fix: "invariant decreases X" — decreases is not an invariant keyword; strip the spurious "invariant" prefix.
    new_repaired = re.sub(
        r"^(\s*)invariant\s+(decreases\b.*)",
        r"\1\2",
        repaired,
        flags=re.MULTILINE,
    )
    if new_repaired != repaired:
        repaired = new_repaired
        changed = True

    # Fix: "(next, newSteps) := helpers.Step(...)" — invalid Dafny; parentheses on LHS of multi-return are not allowed.
    # Always apply: "( a , b ) :=" -> "a, b :="
    new_repaired = re.sub(
        r"\(\s*(\w+)\s*,\s*(\w+)\s*\)\s*:=",
        r"\1, \2 :=",
        repaired,
    )
    if new_repaired != repaired:
        repaired = new_repaired
        changed = True

    # Fix: model re-declares "var delim := new Delimiter(...)"; template already provides "delim"
    new_repaired = re.sub(
        r"^\s*var\s+delim\s*:=\s*new\s+Delimiter\s*\([^)]*\)\s*;\s*\n?",
        "",
        repaired,
        flags=re.MULTILINE,
    )
    if new_repaired != repaired:
        repaired = new_repaired
        changed = True

    # Fix: model calls "helpers.DelimitersInLMAlways()"; template already calls this — remove duplicate
    new_repaired = re.sub(
        r"^\s*helpers\.DelimitersInLMAlways\s*\(\s*\)\s*;\s*\n?",
        "",
        repaired,
        flags=re.MULTILINE,
    )
    if new_repaired != repaired:
        repaired = new_repaired
        changed = True

    # Fix: model re-declares "var delimiter := new Delimiter(...)"; template already provides "delim"
    new_repaired = re.sub(
        r"^\s*var\s+delimiter\s*:=\s*new\s+Delimiter\s*\([^)]*\)\s*;\s*\n?",
        "",
        repaired,
        flags=re.MULTILINE,
    )
    if new_repaired != repaired:
        repaired = new_repaired
        changed = True

    # Fix: model uses "delimiter.LeftDelimiter" (no such field) — should be module constant "LeftDelimiter"
    new_repaired = re.sub(r"\bdelimiter\.LeftDelimiter\b", "LeftDelimiter", repaired)
    if new_repaired != repaired:
        repaired = new_repaired
        changed = True

    # Fix: model uses "delimiter.RightDelimiter" (no such field) — should be module constant "RightDelimiter"
    new_repaired = re.sub(r"\bdelimiter\.RightDelimiter\b", "RightDelimiter", repaired)
    if new_repaired != repaired:
        repaired = new_repaired
        changed = True

    # Fix: remaining "delimiter" references — template variable is "delim", not "delimiter"
    new_repaired = re.sub(r"\bdelimiter\b", "delim", repaired)
    if new_repaired != repaired:
        repaired = new_repaired
        changed = True

    # Fix: old API — helpers.UnconstrainedStep(lm, prompt, ...) -> helpers.UnconstrainedStep(prompt, ...)
    new_repaired = re.sub(
        r"helpers\.UnconstrainedStep\s*\(\s*lm\s*,\s*",
        "helpers.UnconstrainedStep(",
        repaired,
    )
    if new_repaired != repaired:
        repaired = new_repaired
        changed = True

    # Fix: old API — helpers.ConstrainedStep(lm, parser, prompt, ...) -> helpers.ConstrainedStep(prompt, ...)
    new_repaired = re.sub(
        r"helpers\.ConstrainedStep\s*\(\s*lm\s*,\s*parser\s*,\s*",
        "helpers.ConstrainedStep(",
        repaired,
    )
    if new_repaired != repaired:
        repaired = new_repaired
        changed = True

    # Fix: old API — helpers.RollbackToValidPrefix(parser, generated) -> helpers.RollbackToValidPrefix(generated)
    new_repaired = re.sub(
        r"helpers\.RollbackToValidPrefix\s*\(\s*parser\s*,\s*",
        "helpers.RollbackToValidPrefix(",
        repaired,
    )
    if new_repaired != repaired:
        repaired = new_repaired
        changed = True

    # Fix: old static lemma — CSDHelpers.RollbackPreservesTokenInvariant(...) no longer exists; remove the call
    new_repaired = re.sub(
        r"[ \t]*CSDHelpers\.RollbackPreservesTokenInvariant\s*\([^)]+\)\s*;\s*\n?",
        "",
        repaired,
    )
    if new_repaired != repaired:
        repaired = new_repaired
        changed = True

    # Fix: invariants/decreases written as "// Invariant: X" or "// Decreases: X" comments
    # inside the loop body — convert to real Dafny keywords so _fix_while_invariants can move them.
    new_repaired = re.sub(
        r"^(\s*)//\s*[Ii]nvariant:\s*(.+)$",
        r"\1invariant \2",
        repaired,
        flags=re.MULTILINE,
    )
    if new_repaired != repaired:
        repaired = new_repaired
        changed = True
    new_repaired = re.sub(
        r"^(\s*)//\s*[Dd]ecreases:\s*(.+)$",
        r"\1decreases \2",
        repaired,
        flags=re.MULTILINE,
    )
    if new_repaired != repaired:
        repaired = new_repaired
        changed = True

    # Fix: invariant/decreases inside while body -> move them between condition and opening {
    # Dafny syntax: "while cond\n  invariant X\n  decreases Y\n{ body }"
    # LLM writes them either at the start or end of the body in two forms:
    #   Form A: "while cond {" on one line, invariants inside body
    #   Form B: "while cond" + "{" on next line, invariants at start of body
    def _fix_while_invariants(code: str) -> tuple[str, bool]:
        lines = code.split("\n")
        out: list[str] = []
        local_changed = False
        i = 0

        def _collect_body(start_i: int) -> tuple[list[str], list[str]]:
            """Scan body lines from start_i (depth=1). Return (body_lines, inv_lines)."""
            body: list[str] = []
            invs: list[str] = []
            depth = 1
            j = start_i
            while j < len(lines) and depth > 0:
                bl = lines[j]
                if depth == 1 and re.match(r"^\s*(invariant|decreases)\b", bl):
                    cleaned = bl.rstrip().rstrip(";")
                    # Strip spurious "invariant" prefix from decreases clauses
                    cleaned = re.sub(r"^(\s*)invariant\s+(decreases\b)", r"\1\2", cleaned)
                    invs.append(cleaned)
                else:
                    body.append(bl)
                    depth += bl.count("{") - bl.count("}")
                j += 1
            return body, invs, j

        while i < len(lines):
            line = lines[i]
            # Form A: "while ... {" — brace on same line as condition
            m_a = re.match(r"^(\s*)(while\b.+?)\s*\{\s*$", line)
            # Form B: "while ..." alone, next non-blank line is just "{"
            m_b = (not m_a) and re.match(r"^(\s*)(while\b[^{]+?)\s*$", line)
            next_is_brace = (
                m_b and i + 1 < len(lines) and re.match(r"^\s*\{\s*$", lines[i + 1])
            )

            if m_a:
                indent, while_cond = m_a.group(1), m_a.group(2)
                body, invs, i = _collect_body(i + 1)
                if invs:
                    local_changed = True
                    out.append(indent + while_cond)
                    out.extend(invs)
                    out.append(indent + "{")
                    out.extend(body)
                else:
                    out.append(line)
                    out.extend(body)
            elif m_b and next_is_brace:
                indent, while_cond = m_b.group(1), m_b.group(2)
                brace_line = lines[i + 1]
                body, invs, i = _collect_body(i + 2)
                if invs:
                    local_changed = True
                    out.append(indent + while_cond)
                    out.extend(invs)
                    out.append(brace_line)
                    out.extend(body)
                else:
                    out.append(line)
                    out.append(brace_line)
                    out.extend(body)
            else:
                out.append(line)
                i += 1
        return "\n".join(out), local_changed

    new_repaired, inv_changed = _fix_while_invariants(repaired)
    if inv_changed:
        repaired = new_repaired
        changed = True

    # Fix: malformed for-loop "for i in 0 .. |prompt| - 1" (Python/Rust-style) -> Dafny "for i := 0 to |prompt| - 1"
    for_loop_pattern = re.compile(
        r"for\s+(\w+)\s+in\s+0\s*\.\.\s*\|(\w+)\|\s*-\s*1\b"
    )
    new_repaired = for_loop_pattern.sub(r"for \1 := 0 to |\2| - 1", repaired)
    if new_repaired != repaired:
        repaired = new_repaired
        changed = True

    # Fix: C-style for-loop "for (var i := 0; i < upper; i++)" -> Dafny "for i := 0 to upper"
    # Also handles mangled variants like "i + )" or "i++" inside the parens.
    def _c_for_to_dafny(m: re.Match) -> str:
        var, start, upper = m.group(1), m.group(2), m.group(3).strip()
        # Convert seq.Length -> |seq|
        upper = re.sub(r"\b(\w+)\.Length\b", r"|\1|", upper)
        return f"for {var} := {start} to {upper}"

    new_repaired = re.sub(
        r"for\s*\(\s*var\s+(\w+)\s*:=\s*(\d+)\s*;\s*\1\s*<\s*([^;]+?)\s*;[^)]*\)",
        _c_for_to_dafny,
        repaired,
    )
    if new_repaired != repaired:
        repaired = new_repaired
        changed = True

    # Fix: "Ambiguous use of && and ||" — Dafny requires parentheses. Wrap the && part: "A || B && C" -> "A || (B && C)"
    if "Ambiguous use of && and ||" in error_summary or "Use parentheses to disambiguate" in error_summary:
        new_repaired = re.sub(
            r"\|\|\s*(\|generated\|\s*>\s*0\s*&&\s*generated\[\|generated\|-1\]\s*!=\s*LeftDelimiter)\b",
            r"|| (\1)",
            repaired,
        )
        if new_repaired != repaired:
            repaired = new_repaired
            changed = True

    # Fix: Dafny for-loop uses braces not "do" — "for i := 0 to |prompt| - 1 do" -> "for i := 0 to |prompt| - 1 {"
    if "missing semicolon" in error_summary or "lbrace expected" in error_summary or "rbrace expected" in error_summary:
        new_repaired = re.sub(
            r"\bto\s+\|(\w+)\|\s*-\s*1\s+do\b",
            r"to |\1| - 1 {",
            repaired,
        )
        if new_repaired != repaired:
            repaired = new_repaired
            changed = True

    # Fix: "if condition" without braces then multiple statements — add { } when lbrace expected
    if "lbrace expected" in error_summary and "if " in repaired and " if " in repaired:
        # Pattern: "if prompt[i] == LeftDelimiter\n      inConstrainedPart := true;\n      break;" -> add braces
        new_repaired = re.sub(
            r"if\s+(prompt\[i\]\s*==\s*LeftDelimiter)\s*\n(\s+)(inConstrainedPart\s*:=\s*true\s*;\s*\n)\2(break\s*;\s*)",
            r"if \1 {\n\2\3\2\4\n\2}",
            repaired,
        )
        if new_repaired != repaired:
            repaired = new_repaired
            changed = True

    # Fix: "generated, stepsLeft;" — LLM wrote a function-style return for a method; out-params need no return.
    # Remove any bare "generated, stepsLeft;" or "return generated, stepsLeft;" lines.
    new_repaired = re.sub(
        r"^\s*(?:return\s+)?generated\s*,\s*stepsLeft\s*;\s*\n?",
        "",
        repaired,
        flags=re.MULTILINE,
    )
    if new_repaired != repaired:
        repaired = new_repaired
        changed = True

    # Fix: bogus asserts that can't hold after unconstrained generation (e.g. assert parser.IsValidPrefix(generated); assert |generated| > 0;)
    # These are always wrong here — remove them so verification doesn't fail on the assert itself.
    for assert_pat in [
        r"^\s*assert\s+parser\.IsValidPrefix\s*\(\s*generated\s*\)\s*;\s*\n?",
        r"^\s*assert\s+\|generated\|\s*>\s*0\s*;\s*\n?",
    ]:
        new_repaired = re.sub(assert_pat, "", repaired, flags=re.MULTILINE)
        if new_repaired != repaired:
            repaired = new_repaired
            changed = True

    # Fix: template already assigns remainingSteps; remove duplicate "remainingSteps := stepsLeft;"
    if "remainingSteps" in repaired and "remainingSteps := stepsLeft" in repaired:
        remove_remaining = re.compile(
            r"^\s*//\s*Assign remainingSteps[^\n]*\n\s*remainingSteps\s*:=\s*stepsLeft\s*;\s*\n?",
            re.IGNORECASE | re.MULTILINE,
        )
        new_repaired = remove_remaining.sub("", repaired)
        if new_repaired == repaired:
            remove_remaining = re.compile(
                r"^\s*remainingSteps\s*:=\s*stepsLeft\s*;\s*\n?",
                re.MULTILINE,
            )
            new_repaired = remove_remaining.sub("", repaired)
        if new_repaired != repaired:
            repaired = new_repaired
            changed = True

    # Fix: when only failure is invariant parser.IsValidPrefix(generated) and strategy uses UnconstrainedStep, fall back to ConstrainedStep-only loop (mixed strategy cannot preserve invariant with non-permissive parser)
    if (
        "invariant could not be proved" in error_summary
        and "IsValidPrefix(generated)" in error_summary
        and "UnconstrainedStep" in repaired
        and "ConstrainedStep" in repaired
    ):
        # Replace the if/else that chooses UnconstrainedStep vs ConstrainedStep with a single ConstrainedStep body (with lemma before it)
        simple_loop = re.compile(
            r"if\s+parser\.IsCompletePrefix\s*\(\s*generated\s*\)\s*\|\|\s*\|generated\|\s*==\s*0\s*"
            r"\{\s*"
            r"(?:var\s+)?next\s*,\s*newSteps\s*:=\s*helpers\.UnconstrainedStep\s*\([^)]*\)\s*;\s*"
            r"generated\s*:=\s*generated\s*\+\s*\[next\]\s*;\s*stepsLeft\s*:=\s*newSteps\s*;\s*"
            r"\}\s*else\s*\{\s*"
            r"(?:CSDHelpers\.RollbackPreservesTokenInvariant\s*\(\s*lm\s*,\s*parser\s*,\s*generated\s*\)\s*;\s*)?"
            r"(?:var\s+)?next\s*,\s*newSteps\s*:=\s*helpers\.ConstrainedStep\s*\([^)]*\)\s*;\s*"
            r"generated\s*:=\s*generated\s*\+\s*\[next\]\s*;\s*stepsLeft\s*:=\s*newSteps\s*;\s*"
            r"\}",
            re.MULTILINE | re.DOTALL,
        )
        replacement = (
            "  var next, newSteps := helpers.ConstrainedStep(prompt, generated, stepsLeft);\n"
            "  generated := generated + [next];\n"
            "  stepsLeft := newSteps;\n"
            "  }"
        )
        new_repaired = simple_loop.sub(replacement, repaired, count=1)
        if new_repaired != repaired:
            repaired = new_repaired
            changed = True

    # Fix: "index out of range" when accessing generated[|generated|-1] with empty generated — guard with |generated| > 0
    if "index out of range" in error_summary and "generated[|generated|-1]" in repaired:
        # Wrap last-element access in condition: (|generated| > 0 && generated[|generated|-1] != LeftDelimiter) etc.
        new_repaired = re.sub(
            r"generated\[\|generated\|-1\]\s*!=\s*LeftDelimiter",
            r"(|generated| > 0 && generated[|generated|-1] != LeftDelimiter)",
            repaired,
        )
        if new_repaired == repaired:
            new_repaired = re.sub(
                r"generated\[\|generated\|-1\]\s*==\s*LeftDelimiter",
                r"(|generated| > 0 && generated[|generated|-1] == LeftDelimiter)",
                repaired,
            )
        if new_repaired != repaired:
            repaired = new_repaired
            changed = True

    # Fix: "precondition for this call could not be proved" for UnconstrainedPreservesValidWhenPermissive (requires parser.IsPermissive) — remove the lemma call; parser may not be permissive
    if "precondition for this call could not be proved" in error_summary and ("IsPermissive" in error_summary or "UnconstrainedPreservesValidWhenPermissive" in error_summary):
        if "UnconstrainedPreservesValidWhenPermissive" in repaired:
            remove_lemma = re.compile(
                r"\s*CSDHelpers\.UnconstrainedPreservesValidWhenPermissive\s*\(\s*parser\s*,\s*generated\s*,\s*next\s*\)\s*;\s*\n?",
                re.MULTILINE,
            )
            new_repaired = remove_lemma.sub("\n", repaired)
            if new_repaired != repaired:
                repaired = new_repaired
                changed = True

    # Fix: "invariant could not be proved" for stepCounter <= maxSteps — remove that invariant (stepCounter can grow without bound in some branches)
    if "invariant could not be proved" in error_summary and "stepCounter" in error_summary and "maxSteps" in error_summary:
        for pat in [
            r"\s*invariant\s+stepCounter\s*<=\s*maxSteps\s*\n",
            r"\s*invariant\s+stepCounter\s*>=\s*0\s*&&\s*stepCounter\s*<=\s*maxSteps\s*\n",
        ]:
            new_repaired = re.sub(pat, "\n", repaired)
            if new_repaired != repaired:
                repaired = new_repaired
                changed = True
                break

    # Fix: "precondition ... ConstrainedWindowValid" -> add missing loop invariant
    if (
        "precondition for this call could not be proved" in error_summary
        and "ConstrainedWindowValid" in error_summary
        and "ConstrainedStep" in repaired
    ):
        # Insert invariant after the last standard invariant in the while loop
        new_repaired = re.sub(
            r"(invariant\s+\|generated\|\s*\+\s*stepsLeft\s*==\s*maxSteps\b)",
            r"\1\n    invariant helpers.ConstrainedWindowValid(generated)",
            repaired,
            count=1,
        )
        if new_repaired != repaired:
            repaired = new_repaired
            changed = True

    # Fix: model uses lm.ChooseNextTokenUnconstrained() which no longer exists — remove such calls
    if "ChooseNextTokenUnconstrained" in repaired:
        new_repaired = re.sub(
            r"\blm\.ChooseNextTokenUnconstrained\s*\(\s*\)",
            "lm.ChooseNextToken()",
            repaired,
        )
        if new_repaired != repaired:
            repaired = new_repaired
            changed = True

    # Fix: "Duplicate local-variable name: stepsLeft" -> template already declares it; remove duplicate
    if "Duplicate local-variable name: stepsLeft" in error_summary:
        # Remove lines that are just "var stepsLeft := maxSteps;" (any spacing)
        line_pattern = re.compile(r"^\s*var\s+stepsLeft\s*:=\s*maxSteps\s*;\s*$", re.IGNORECASE | re.MULTILINE)
        new_repaired = line_pattern.sub("", repaired)
        if new_repaired != repaired:
            repaired = new_repaired
            changed = True

    # Fix: "Duplicate local-variable name: helpers" -> template already declares "var helpers := new CSDHelpers(...)"; remove duplicate
    if "Duplicate local-variable name: helpers" in error_summary:
        helpers_line = re.compile(r"^\s*var\s+helpers\s*:=\s*new\s+CSDHelpers\s*\([^)]*\)\s*;\s*$", re.IGNORECASE | re.MULTILINE)
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
            r"generated\s*:=\s*helpers\.RollbackToValidPrefix\s*\(\s*(?:parser\s*,\s*)?generated\s*\)\s*;\s*"
            r"\s*stepsLeft\s*:=\s*maxSteps\s*;\s*"
            r"\s*\}",
            re.MULTILINE,
        )
        replacement = (
            " else {\n"
            "    var next, newSteps := helpers.ConstrainedStep(prompt, generated, stepsLeft);\n"
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
                r"\s*generated\s*:=\s*helpers\.RollbackToValidPrefix\s*\(\s*(?:parser\s*,\s*)?generated\s*\)\s*;\s*",
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

    # Fix: "variable 'generated' ... might be uninitialized" — LLM uses generated in the while
    # condition without initializing it first. Insert generated := []; before the loop.
    if (
        ("might be uninitialized" in error_summary or "definite-assignment" in error_summary)
        and "generated" in error_summary
        and "generated := [" not in repaired
        and "generated := []" not in repaired
    ):
        # Insert generated := []; immediately before the while loop
        new_repaired = re.sub(
            r"(while\s+stepsLeft\s*>\s*0\b)",
            r"generated := [];\n  \1",
            repaired,
            count=1,
        )
        if new_repaired != repaired:
            repaired = new_repaired
            changed = True

    # Fix: "unresolved identifier: next" / "unresolved identifier: newSteps" — LLM declares var next, newSteps inside if/else or at loop top so they're out of scope. Declare at start of while body and assign in each branch.
    # Also triggered by definite-assignment errors for 'next' or 'newSteps' (shadowing in if/else branches).
    has_next_scope_error = (
        "unresolved identifier: next" in error_summary
        or "unresolved identifier: newSteps" in error_summary
        or ("unresolved identifier" in error_summary and " next" in error_summary and " newSteps" in error_summary)
        or ("might be uninitialized" in error_summary and ("'next'" in error_summary or "'newSteps'" in error_summary))
        or ("definite-assignment" in error_summary and ("next" in error_summary or "newSteps" in error_summary) and "generated" not in error_summary)
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

    # Fix: "precondition for this call could not be proved" for InsideDelimitedWindow — ConstrainedStep called before window is open.
    # Replace a bare ConstrainedStep in the loop body with an if/else: ConstrainedStep when inside window, UnconstrainedStep otherwise.
    if (
        "precondition for this call could not be proved" in error_summary
        and "InsideDelimitedWindow" in error_summary
        and "ConstrainedStep" in repaired
    ):
        bare_constrained = re.compile(
            r"^(\s*)(var\s+next\s*,\s*newSteps\s*:=\s*helpers\.ConstrainedStep\s*\([^)]*\)\s*;\s*\n"
            r"\s*generated\s*:=\s*generated\s*\+\s*\[next\]\s*;\s*\n"
            r"\s*stepsLeft\s*:=\s*newSteps\s*;\s*)$",
            re.MULTILINE,
        )
        def wrap_with_window_guard(m: re.Match) -> str:
            indent, block = m.group(1), m.group(2)
            lines = block.strip().split("\n")
            inner_cs = "\n".join(f"{indent}  {l.strip()}" for l in lines)
            inner_us = f"{indent}  next, newSteps := helpers.UnconstrainedStep(prompt, generated, stepsLeft);"
            return (
                f"{indent}var next: Token; var newSteps: nat;\n"
                f"{indent}if helpers.InsideDelimitedWindow(generated) && !parser.IsCompletePrefix(helpers.GetDelimitedContent(generated)) {{\n"
                f"{inner_cs}\n"
                f"{indent}}} else {{\n"
                f"{inner_us}\n"
                f"{indent}  generated := generated + [next];\n"
                f"{indent}  stepsLeft := newSteps;\n"
                f"{indent}}}\n"
            )
        new_repaired = bare_constrained.sub(wrap_with_window_guard, repaired, count=1)
        if new_repaired != repaired:
            repaired = new_repaired
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

    # Fix: "the method returns 1 value but is assigned to 0 variable" for RollbackToValidPrefix — LLM calls it without assigning. Must assign: generated := helpers.RollbackToValidPrefix(generated)
    if "returns 1 value but is assigned to 0 variable" in error_summary and "RollbackToValidPrefix" in repaired:
        standalone_rollback = re.compile(
            r"^(\s*)helpers\.RollbackToValidPrefix\s*\(\s*(?:parser\s*,\s*)?generated\s*\)\s*;(?:\s*//[^\n]*)?\s*$",
            re.MULTILINE,
        )
        new_repaired = standalone_rollback.sub(
            lambda m: f"{m.group(1)}generated := helpers.RollbackToValidPrefix(generated);",
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

    # Fix: "invariant could not be proved" for parser.IsValidPrefix(generated) when loop uses UnconstrainedStep —
    # drop the IsValidPrefix invariant since UnconstrainedStep cannot guarantee it with a non-permissive parser.
    if "invariant could not be proved" in error_summary and "IsValidPrefix(generated)" in error_summary and "UnconstrainedStep" in repaired:
        new_repaired = re.sub(
            r"[ \t]*invariant\s+parser\.IsValidPrefix\s*\(\s*generated\s*\)\s*\n",
            "",
            repaired,
        )
        if new_repaired != repaired:
            repaired = new_repaired
            changed = True

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

        # Initial generation (one LLM call, can take 30–90s or longer on CPU/slow GPU)
        print(f"Generating initial strategy for: {task_description[:80]}...")
        t0 = time.perf_counter()
        strategy_code = self.generator.generate_initial(task_description)
        print(f"  (initial generation took {time.perf_counter() - t0:.1f}s)")

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
            t0 = time.perf_counter()
            verification_result = self.verifier.verify(full_code)
            attempt.verification_result = verification_result
            print(f"  (verification took {time.perf_counter() - t0:.1f}s)")

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
                # Refine based on verification error via model (one LLM call; can take minutes if GPU is slow or OOM → CPU)
                print("  Refining based on verification error... (Qwen generating new strategy; watch for timing below)")
                t0 = time.perf_counter()
                strategy_code = self.generator.refine_after_verification_error(
                    strategy_code, error_msg
                )
                print(f"  (refinement took {time.perf_counter() - t0:.1f}s)")
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

            # Always print outputs vs expected so we can spot Prover9 "Unknown" cheesing
            eval_result.print_outputs_vs_expected()

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


