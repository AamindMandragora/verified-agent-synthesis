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

from transpiler.transpiler import transpile_contract_library

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

    # Python-side fix: if branch-local tuple assignments feed later uses of next_token/new_steps,
    # predeclare them before the branch so transpilation can emit outer-scope variables.
    if (
        "unresolved identifier: next_token" in error_summary
        or "unresolved identifier: new_steps" in error_summary
    ):
        pattern = re.compile(
            r"(?m)^(?P<indent>[ \t]*)if (?P<cond>[^\n]+):\n"
            r"(?P=indent)[ \t]+(?P<name1>[A-Za-z_]\w*)\s*,\s*(?P<name2>[A-Za-z_]\w*)\s*=\s*helpers\.(?:ConstrainedAnswerStep|ExpressiveStep|UnconstrainedStep)\([^\n]+\)\n"
            r"(?P=indent)else:\n"
            r"(?P=indent)[ \t]+(?P=name1)\s*,\s*(?P=name2)\s*=\s*helpers\.(?:ConstrainedAnswerStep|ExpressiveStep|UnconstrainedStep)\([^\n]+\)"
        )

        def _predeclare_branch_outputs(m: re.Match) -> str:
            indent = m.group("indent")
            name1 = m.group("name1")
            name2 = m.group("name2")
            prefix = f"{indent}{name1} = eosToken\n{indent}{name2} = stepsLeft\n"
            return prefix + m.group(0)

        new_repaired = pattern.sub(_predeclare_branch_outputs, repaired)
        if new_repaired != repaired:
            repaired = new_repaired
            changed = True

    # Python-first normalization for legacy Dafny-ish outputs.
    new_repaired = re.sub(r"(?m)^(\s*)//", r"\1#", repaired)
    new_repaired = new_repaired.replace(":=", "=")
    new_repaired = new_repaired.replace("&&", " and ")
    new_repaired = new_repaired.replace("||", " or ")
    new_repaired = re.sub(r"(?<![=!])!(?!=)", " not ", new_repaired)
    new_repaired = re.sub(r"(?m)^(\s*)(invariant|decreases)\b", r"\1# \2", new_repaired)
    if new_repaired != repaired:
        repaired = new_repaired
        changed = True

    # Fix: Python-only constructs that the strategy transpiler cannot lower to Dafny.
    if (
        "Unsupported comparison operator: IsNot" in error_summary
        or "unresolved identifier: isinstance" in error_summary
        or "unresolved identifier: str" in error_summary
        or "type seq<char> does not have a member isalpha" in error_summary
        or "type seq<char> does not have a member isdigit" in error_summary
        or "type of 'null' is a reference type" in error_summary
    ):
        python_only_patterns = [
            (r"(?m)^(\s*next_token\s*=\s*)None(\s*)$", r"\1eosToken\2"),
            (r"(?m)^(\s*new_steps\s*=\s*)None(\s*)$", r"\1stepsLeft\2"),
            (r"\bisinstance\s*\([^)]*\)", "True"),
            (r"([A-Za-z_][A-Za-z0-9_\[\]\-]*)\.isalpha\(\)", "True"),
            (r"([A-Za-z_][A-Za-z0-9_\[\]\-]*)\.isdigit\(\)", "True"),
            (r"\b([A-Za-z_]\w*)\s+is\s+not\s+None\b", "True"),
            (r"\b([A-Za-z_]\w*)\s+is\s+None\b", "False"),
        ]
        for pattern, replacement in python_only_patterns:
            new_repaired = re.sub(pattern, replacement, repaired)
            if new_repaired != repaired:
                repaired = new_repaired
                changed = True

    # Template already initializes generated/stepsLeft and finalizes remainingSteps.
    for pattern in [
        r"^\s*generated\s*=\s*\[\s*\]\s*\n?",
        r"^\s*stepsLeft\s*=\s*maxSteps\s*\n?",
        r"^\s*remainingSteps\s*=\s*stepsLeft\s*\n?",
        r"^\s*delim\s*=\s*Delimiter\s*\([^)]*\)\s*\n?",
        r"^\s*helpers\s*=\s*CSDHelpers\s*\([^)]*\)\s*\n?",
        r"^\s*helpers\.DelimitersInLMAlways\s*\(\s*\)\s*\n?",
        r"^\s*lm\.ValidTokensIdsLogitsAlways\s*\(\s*\)\s*\n?",
    ]:
        new_repaired = re.sub(pattern, "", repaired, flags=re.MULTILINE)
        if new_repaired != repaired:
            repaired = new_repaired
            changed = True

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

    # Fix: invariant/decreases clauses in the wrong place relative to the while loop.
    # Handles four patterns:
    #   Form A: "while cond {" — invariants inside the body (at top or bottom)
    #   Form B: "while cond" + "{" on next line — invariants inside the body
    #   Form C: invariants/decreases lines BEFORE the while keyword (outside the loop)
    #   Form D: invariants/decreases lines AFTER the while loop's closing "}" (outside the loop)
    # In all cases, move them to between the while condition and the opening "{".
    def _fix_while_invariants(code: str) -> tuple[str, bool]:
        lines = code.split("\n")
        out: list[str] = []
        local_changed = False
        i = 0

        def _clean_inv(line: str) -> str:
            cleaned = line.rstrip().rstrip(";")
            return re.sub(r"^(\s*)invariant\s+(decreases\b)", r"\1\2", cleaned)

        def _collect_body(start_i: int) -> tuple[list[str], list[str], int]:
            """Scan body lines from start_i (depth=1). Return (body_lines, inv_lines, next_i)."""
            body: list[str] = []
            invs: list[str] = []
            depth = 1
            j = start_i
            while j < len(lines) and depth > 0:
                bl = lines[j]
                if depth == 1 and re.match(r"^\s*(invariant|decreases)\b", bl):
                    invs.append(_clean_inv(bl))
                else:
                    body.append(bl)
                    depth += bl.count("{") - bl.count("}")
                j += 1
            return body, invs, j

        def _collect_trailing_invs(start_i: int) -> tuple[list[str], int]:
            """Collect invariant/decreases lines after a while loop's closing '}'.
            Skips blank lines and comments that appear before the first invariant.
            Returns (inv_lines, next_i)."""
            invs: list[str] = []
            j = start_i
            while j < len(lines):
                l = lines[j]
                if re.match(r"^\s*(invariant|decreases)\b", l):
                    invs.append(_clean_inv(l))
                    j += 1
                elif not l.strip():
                    j += 1
                elif re.match(r"^\s*//", l) and not invs:
                    # Skip comments only before the first invariant
                    j += 1
                else:
                    break
            return invs, j

        while i < len(lines):
            line = lines[i]

            # Form C: invariant/decreases lines appearing BEFORE the while keyword.
            # Collect them, then check if a while loop immediately follows.
            if re.match(r"^\s*(invariant|decreases)\b", line):
                pre_invs: list[str] = []
                j = i
                while j < len(lines):
                    l = lines[j]
                    if re.match(r"^\s*(invariant|decreases)\b", l):
                        pre_invs.append(_clean_inv(l))
                        j += 1
                    elif not l.strip():
                        j += 1
                    else:
                        break
                # Check if followed by a while loop
                if j < len(lines):
                    wl = lines[j]
                    m_a = re.match(r"^(\s*)(while\b.+?)\s*\{\s*$", wl)
                    m_b = (not m_a) and re.match(r"^(\s*)(while\b[^{]+?)\s*$", wl)
                    next_is_brace = m_b and j + 1 < len(lines) and re.match(r"^\s*\{\s*$", lines[j + 1])
                    if m_a:
                        indent, while_cond = m_a.group(1), m_a.group(2)
                        body, inner_invs, new_i = _collect_body(j + 1)
                        trail_invs, trail_i = _collect_trailing_invs(new_i)
                        all_invs = pre_invs + inner_invs + trail_invs
                        out.append(indent + while_cond)
                        out.extend(all_invs)
                        out.append(indent + "{")
                        out.extend(body)
                        i = trail_i
                        local_changed = True
                        continue
                    elif m_b and next_is_brace:
                        indent, while_cond = m_b.group(1), m_b.group(2)
                        brace_line = lines[j + 1]
                        body, inner_invs, new_i = _collect_body(j + 2)
                        trail_invs, trail_i = _collect_trailing_invs(new_i)
                        all_invs = pre_invs + inner_invs + trail_invs
                        out.append(indent + while_cond)
                        out.extend(all_invs)
                        out.append(brace_line)
                        out.extend(body)
                        i = trail_i
                        local_changed = True
                        continue
                # No while follows — output as-is
                out.append(line)
                i += 1
                continue

            # Form A: "while ... {" — brace on same line as condition
            m_a = re.match(r"^(\s*)(while\b.+?)\s*\{\s*$", line)
            # Form B: "while ..." alone, next non-blank line is just "{"
            m_b = (not m_a) and re.match(r"^(\s*)(while\b[^{]+?)\s*$", line)
            next_is_brace = (
                m_b and i + 1 < len(lines) and re.match(r"^\s*\{\s*$", lines[i + 1])
            )

            if m_a:
                indent, while_cond = m_a.group(1), m_a.group(2)
                body, invs, new_i = _collect_body(i + 1)
                trail_invs, trail_i = _collect_trailing_invs(new_i)
                all_invs = invs + trail_invs
                if all_invs:
                    local_changed = True
                    out.append(indent + while_cond)
                    out.extend(all_invs)
                    out.append(indent + "{")
                    out.extend(body)
                    i = trail_i
                else:
                    out.append(line)
                    out.extend(body)
                    i = new_i
            elif m_b and next_is_brace:
                indent, while_cond = m_b.group(1), m_b.group(2)
                brace_line = lines[i + 1]
                body, invs, new_i = _collect_body(i + 2)
                trail_invs, trail_i = _collect_trailing_invs(new_i)
                all_invs = invs + trail_invs
                if all_invs:
                    local_changed = True
                    out.append(indent + while_cond)
                    out.extend(all_invs)
                    out.append(brace_line)
                    out.extend(body)
                    i = trail_i
                else:
                    out.append(line)
                    out.append(brace_line)
                    out.extend(body)
                    i = new_i
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

    # Fix: "invariants" (plural) block -> each line must be "invariant <expr>" (singular). Dafny has no "invariants" keyword.
    if "missing semicolon" in error_summary and "invariants" in repaired:
        lines_out = []
        line_list = repaired.split("\n")
        i = 0
        while i < len(line_list):
            line = line_list[i]
            m_inv = re.match(r"^(\s*)invariants\s*\s*$", line)
            if m_inv:
                base_indent = m_inv.group(1)
                inv_indent_len = len(base_indent) + 1
                i += 1
                while i < len(line_list):
                    raw = line_list[i]
                    if not raw.strip():
                        lines_out.append(raw)
                        i += 1
                        continue
                    if len(raw) - len(raw.lstrip()) < inv_indent_len:
                        break
                    content = raw.strip()
                    if re.match(r"^decreases\b", content):
                        lines_out.append(base_indent + content)
                    elif re.match(r"^(if\s|var\s|\}\s*$|else\s)", content):
                        break
                    else:
                        lines_out.append(base_indent + "invariant " + content)
                    i += 1
                if i < len(line_list):
                    lines_out.append(line_list[i])
                    i += 1
                continue
            lines_out.append(line)
            i += 1
        new_repaired = "\n".join(lines_out)
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

    # Fix: "decreases expression might not decrease" in Python state-machine loops where a branch
    # sets a string mode like "done" without consuming budget. Replace that terminal assignment
    # with `break` so the loop exits instead of taking a non-decreasing back-edge.
    if "decreases expression might not decrease" in error_summary:
        new_repaired = re.sub(
            r"(?m)^(\s*)([A-Za-z_]\w*)\s*=\s*(['\"])done\3\s*$",
            r"\1break",
            repaired,
        )
        if new_repaired != repaired:
            repaired = new_repaired
            changed = True

    # Fix: if a mode-membership invariant is failing because the loop introduces a terminal "done"
    # state, drop the overly specific invariant instead of repeating the same invalid state machine.
    if "invariant could not be proved" in error_summary and "current_step" in error_summary:
        new_repaired = re.sub(
            r"(?m)^[ \t]*#\s*invariant\s+current_step\s+in\s+\[[^\n]+\]\n?",
            "",
            repaired,
        )
        if new_repaired != repaired:
            repaired = new_repaired
            changed = True

    # Fix: finalization requires a nonempty valid answer. If the strategy can exit with an empty
    # answer, inject one last constrained answer step when budget remains.
    if (
        "precondition for this call could not be proved" in error_summary
        and ("parser.IsValidPrefix(answer)" in error_summary or "|answer| > 0" in error_summary)
        and "helpers.FinalizeDelimitedAnswer" in error_summary
        and "if len(answer) == 0 and stepsLeft > 0 and not parser.IsCompletePrefix(answer):" not in repaired
    ):
        guard = (
            "\nif len(answer) == 0 and stepsLeft > 0 and not parser.IsCompletePrefix(answer):\n"
            "    next_token, new_steps = helpers.ConstrainedAnswerStep(prompt, generated, answer, stepsLeft)\n"
            "    answer = answer + [next_token]\n"
            "    stepsLeft = new_steps\n"
        )
        repaired = repaired.rstrip() + guard
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

    # Fix: "precondition for this call could not be proved" for lm.ValidTokensIdsLogits() —
    # helper methods (UnconstrainedStep, ConstrainedStep, GetDelimitedContent) require it.
    # Add invariant lm.ValidTokensIdsLogits() to the while loop (provable: template establishes
    # it before the loop; step postconditions maintain it).
    if (
        "precondition for this call could not be proved" in error_summary
        and "ValidTokensIdsLogits" in error_summary
        and "invariant lm.ValidTokensIdsLogits()" not in repaired
    ):
        # Insert as first invariant in the while loop, before other invariants or decreases
        new_repaired = re.sub(
            r"(while\s+[^\n]+\n)((\s+invariant\b|\s+decreases\b))",
            r"\1    invariant lm.ValidTokensIdsLogits()\n\2",
            repaired,
            count=1,
        )
        if new_repaired == repaired:
            # Fallback: insert before the opening { of the while loop
            new_repaired = re.sub(
                r"(while\s+[^\n]+\n)(\s*\{)",
                r"\1    invariant lm.ValidTokensIdsLogits()\n\2",
                repaired,
                count=1,
            )
        if new_repaired != repaired:
            repaired = new_repaired
            changed = True

    # Fix: "invariant could not be proved" for |generated| <= maxSteps — the correct provable invariant
    # is |generated| + stepsLeft <= maxSteps (sum unchanged per step, decreases on rollback).
    # Replace the broken |generated| <= maxSteps invariant with the correct form.
    if (
        "invariant could not be proved" in error_summary
        and "|generated| <= maxSteps" in error_summary
        and "UnconstrainedStep" in repaired
    ):
        new_repaired = re.sub(
            r"\binvariant\s+\|generated\|\s*<=\s*maxSteps\b",
            "invariant |generated| + stepsLeft <= maxSteps",
            repaired,
        )
        if new_repaired != repaired:
            repaired = new_repaired
            changed = True

    # Fix: always normalize old |generated| <= maxSteps invariant to |generated| + stepsLeft <= maxSteps
    # (unconditional: the old form is never correct since Dafny cannot prove |generated| < maxSteps
    # from it alone; the new form is always provable and rollback-compatible)
    new_repaired = re.sub(
        r"\binvariant\s+\|generated\|\s*<=\s*maxSteps\b",
        "invariant |generated| + stepsLeft <= maxSteps",
        repaired,
    )
    if new_repaired != repaired:
        repaired = new_repaired
        changed = True

    # Fix: "seq<...> does not have a member Contains" or "generated.Contains(X)" ->
    # Dafny seq has no Contains method; use membership: X in generated.
    if "Contains" in repaired and ("does not have a member Contains" in error_summary or ".Contains(" in repaired):
        new_repaired = re.sub(
            r"generated\.Contains\s*\(\s*([^)]+)\s*\)",
            r"(\1) in generated",
            repaired,
        )
        if new_repaired != repaired:
            repaired = new_repaired
            changed = True

    # Fix: "member 'LeftDelimiter' does not exist in class 'Delimiter'" for delim.LeftDelimiter —
    # Delimiter class has no LeftDelimiter field (it's Left); module constant is LeftDelimiter.
    new_repaired = re.sub(r"\bdelim\.LeftDelimiter\b", "LeftDelimiter", repaired)
    if new_repaired != repaired:
        repaired = new_repaired
        changed = True

    # Fix: "member 'RightDelimiter' does not exist in class 'Delimiter'" for delim.RightDelimiter
    new_repaired = re.sub(r"\bdelim\.RightDelimiter\b", "RightDelimiter", repaired)
    if new_repaired != repaired:
        repaired = new_repaired
        changed = True

    # Fix: "invariant could not be proved" for hasValid ==> parser.IsValidPrefix(generated) —
    # this invariant cannot be maintained after UnconstrainedStep (not guaranteed valid prefix).
    # Remove it; it's an LLM-invented invariant that has no path to verification.
    if (
        "invariant could not be proved" in error_summary
        and "IsValidPrefix" in error_summary
        and "hasValid" in error_summary
    ):
        new_repaired = re.sub(
            r"[ \t]*invariant\s+hasValid\s*==>\s*parser\.IsValidPrefix\s*\(\s*generated\s*\)\s*\n",
            "",
            repaired,
        )
        if new_repaired != repaired:
            repaired = new_repaired
            changed = True

    # Fix: "precondition ... ConstrainedWindowValid" -> add missing loop invariant
    # Applies when ConstrainedStep or UnconstrainedStep precondition fails due to missing invariant
    if (
        "precondition for this call could not be proved" in error_summary
        and "ConstrainedWindowValid" in error_summary
        and ("ConstrainedStep" in repaired or "UnconstrainedStep" in repaired)
    ):
        # Insert invariant after |generated| + stepsLeft <= maxSteps or |generated| <= maxSteps invariant
        new_repaired = re.sub(
            r"(invariant\s+\|generated\|\s*(?:\+\s*stepsLeft\s*)?<=\s*maxSteps\b)",
            r"\1\n    invariant helpers.ConstrainedWindowValid(generated)",
            repaired,
            count=1,
        )
        if new_repaired == repaired:
            # Fallback: insert after lm.ValidTokensIdsLogits() invariant
            new_repaired = re.sub(
                r"(invariant\s+lm\.ValidTokensIdsLogits\s*\(\s*\))",
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

    # Fix: "decreases expression might not decrease" — caused by resetting or increasing stepsLeft in the loop.
    # Replace the else branch that does RollbackToValidPrefix + stepsLeft := ... with a step so the loop always decreases.
    if "decreases expression might not decrease" in error_summary and "RollbackToValidPrefix" in repaired:
        step_replacement = (
            " else {\n"
            "    var next, newSteps := helpers.ConstrainedStep(prompt, generated, stepsLeft);\n"
            "    generated := generated + [next];\n"
            "    stepsLeft := newSteps;\n"
            "  }"
        )
        # Match else { ... RollbackToValidPrefix(...); ... stepsLeft := maxSteps... ; ... } (stepsLeft must only decrease)
        else_block = re.compile(
            r"\s*else\s*\{\s*(?:\s*//[^\n]*\n)*(?:\s*if\s*\([^)]+\)\s*\{\s*(?:\s*[^\n]*\n)*?\s*\})?\s*"
            r"generated\s*:=\s*helpers\.RollbackToValidPrefix\s*\(\s*(?:parser\s*,\s*)?generated\s*\)\s*;\s*\n"
            r"\s*stepsLeft\s*:=\s*maxSteps(?:\s*-\s*\|generated\|)?\s*;\s*\n"
            r"(?:\s*[^\n]*\n)*?\s*\}",
            re.MULTILINE,
        )
        new_repaired = else_block.sub(step_replacement, repaired, count=1)
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

    # Fix: "function precondition could not be proved" for parser.ValidNextToken inside an exists
    # quantifier — requires parser.IsValidPrefix(generated) which is not guaranteed after
    # unconstrained generation. Remove the entire line (e.g. "hasValid := exists token :: ...").
    if (
        "function precondition could not be proved" in error_summary
        and "IsValidPrefix" in error_summary
        and "ValidNextToken" in error_summary
    ):
        new_repaired = re.sub(
            r"^\s*\w+\s*:=\s*exists\s+\w+\s*::\s*\w+\s+in\s+\w+\s*&&\s*parser\.ValidNextToken\s*\([^)]+\)\s*;\s*\n?",
            "",
            repaired,
            flags=re.MULTILINE,
        )
        if new_repaired != repaired:
            repaired = new_repaired
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
                if new_repaired == repaired:
                    # Fallback: insert before the while loop
                    new_repaired = re.sub(
                        r"(while\s+stepsLeft\s*>\s*0\b)",
                        decl + r"\n  \1",
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

    # Detect which primitives are used.
    uses_constrained = "ConstrainedStep" in strategy_code_for_match or "ConstrainedAnswerStep" in strategy_code_for_match
    uses_unconstrained = (
        "UnconstrainedStep" in strategy_code_for_match or "ExpressiveStep" in strategy_code_for_match
    )
    uses_rollback = "RollbackToValidPrefix" in strategy_code_for_match
    uses_answer_channel = "ConstrainedAnswerStep" in strategy_code_for_match

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
        "parameters": {
            "uses_rollback": uses_rollback,
            "uses_answer_channel": uses_answer_channel,
        },
        "category": category,
        "comparable_to": "N/A",
        "raw_code": strategy_code,
    }


@dataclass
class SynthesisAttempt:
    """Record of a single synthesis attempt."""

    attempt_number: int
    strategy_code: str
    timestamp: str
    full_python_code: str = ""
    full_dafny_code: str = ""

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

    def __post_init__(self) -> None:
        if not self.full_python_code and self.full_dafny_code:
            self.full_python_code = self.full_dafny_code
        elif self.full_python_code and not self.full_dafny_code:
            self.full_dafny_code = self.full_python_code

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
    compiled_module_path: Optional[Path]
    output_dir: Optional[Path]
    run_dir: Optional[Path]
    attempts: list[SynthesisAttempt]
    total_time_ms: float
    full_python_code: str = ""
    full_dafny_code: str = ""

    def __post_init__(self) -> None:
        if not self.full_python_code and self.full_dafny_code:
            self.full_python_code = self.full_dafny_code
        elif self.full_python_code and not self.full_dafny_code:
            self.full_dafny_code = self.full_python_code

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
        print("  (LLM steps can take 1–2 min each; evaluation can take several minutes. Interrupt with Ctrl+C if needed.)")
        t0 = time.perf_counter()
        strategy_code = self.generator.generate_initial(task_description)
        print(f"  (initial generation took {time.perf_counter() - t0:.1f}s)")

        long_run_message_shown = False
        for iteration in range(self.max_iterations):
            attempt_num = iteration + 1
            # If we've been running a while, remind user once that this is expected
            elapsed = time.time() - start_time
            if elapsed >= 90 and not long_run_message_shown:
                print(f"\n  [Synthesis still running (~{int(elapsed / 60)} min elapsed). LLM and evaluation steps are slow; you can interrupt with Ctrl+C.]\n")
                long_run_message_shown = True
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
                full_python_code=full_code,
                timestamp=datetime.now().isoformat(),
            )

            # Stage 1: Verification
            print("\n[1/4] Verifying transpiled Python strategy...")
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
                        f"Keep the explicit answer-channel architecture: build expressive free-form text with "
                        f"helpers.ExpressiveStep(...), build the grammar-constrained answer in `answer` with "
                        f"helpers.ConstrainedAnswerStep(...), and leave delimiter insertion to the template.\n\n"
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
                print("  Refining based on verification error... (LLM call may take 1–2 min)")
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

            python_path = run_dir / f"{output_name}.py"
            python_path.write_text(full_code, encoding="utf-8")
            transpiled_result = transpile_contract_library(full_code, module_name_hint=python_path.stem, axiomatize=False)
            if transpiled_result.is_ok():
                transpiled_dafny_path = run_dir / f"{output_name}.dfy"
                transpiled_dafny_path.write_text(transpiled_result.value, encoding="utf-8")
                print(f"  Python CSD saved to: {python_path}")
                print(f"  Transpiled Dafny saved to: {transpiled_dafny_path}")
            else:
                print(f"  Python CSD saved to: {python_path}")

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

            if len(runtime_result.output or []) == 0:
                print("  ✗ Runtime smoke test produced 0 tokens")
                attempt.failed_at = FailureStage.RUNTIME
                attempt.error_summary = (
                    "Runtime smoke test produced 0 tokens. "
                    "The strategy must execute at least one decoding step in the smoke environment."
                )
                attempts.append(attempt)

                print("  Refining based on empty runtime output...")
                strategy_code = self.generator.refine_after_runtime_error(
                    strategy_code,
                    attempt.error_summary,
                )
                continue

            # Stage 4: Evaluation — use same device as generator to avoid loading on a full GPU
            if getattr(self.generator, "device", None):
                self.evaluator.device = self.generator.device
            print("\n[4/4] Evaluating on dataset sample... (may take several minutes)")
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
                full_python_code=full_code,
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
        python_path = run_dir / f"{output_name}.py"
        with open(python_path, "w") as f:
            f.write(full_code)

        transpiled_result = transpile_contract_library(full_code, module_name_hint=output_name, axiomatize=False)
        dafny_path = None
        if transpiled_result.is_ok():
            dafny_path = run_dir / f"{output_name}.dfy"
            with open(dafny_path, "w") as f:
                f.write(transpiled_result.value)

        rationale_extracted = extract_rationale(strategy_code)

        # Save a report
        report_path = run_dir / "success_report.json"
        report = {
            "strategy_code": strategy_code,
            "tool_choice_rationale": rationale_extracted.rationale,
            "python_file": str(python_path),
            "transpiled_dafny_file": str(dafny_path) if dafny_path else None,
            "compiled_dir": str(compilation_result.output_dir),
            "total_attempts": len(attempts),
            "timestamp": datetime.now().isoformat(),
        }

        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        print(f"Strategy saved to: {python_path}")
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
