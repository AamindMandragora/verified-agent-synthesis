"""TDD tests for CRANE-faithful GSM syntax metric (Deliverable 2).

These tests must FAIL (red) against the OLD all-spans per-example parser and
pass (green) once the CRANE-faithful final-block-only check is in place.

CRANE semantics (from get_avgs.py / check_gsm_parsed):
  - Take ONLY the FINAL <<...>> block.
  - If no block exists → False.
  - If block contains '{' or '}' → False.
  - If block contains 'round(' → False.
  - Parse the full delimited block against the STATIC grammar (any CNAME ok).
  - True iff the block parses completely.

Run on focal:
    cd /home/aadivyar/csd-generation
    /apps/conda/advayth2/envs/advayth2/bin/python -m pytest tests/test_crane_gsm_syntax.py -v
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

# ── project root on focal / local ───────────────────────────────────────────
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

# ---------------------------------------------------------------------------
# Helpers shared by all tests.
# ---------------------------------------------------------------------------

def _make_evaluator():
    """Return a GSM evaluator without loading any model or dataset."""
    from synthesis.evaluate.evaluator import Evaluator

    return Evaluator(dataset_name="gsm_symbolic", backend="huggingface")


def _syntax_pass(ev, output: str, example: dict | None = None) -> bool:
    """Run the full syntax-pass pipeline as the eval loop does."""
    all_valid, segments = ev._check_syntax_validity(output, example=example)
    return ev._example_syntax_pass(all_valid, segments, None)


# ---------------------------------------------------------------------------
# Test 1 — Phantom variable should PASS under CRANE metric (any CNAME is ok)
# ---------------------------------------------------------------------------

def test_phantom_variable_passes_crane_metric():
    """CRANE uses a static grammar where VARIABLE = any CNAME.

    A span like <<n1 * cur>> should pass even when 'cur' is not in the
    example's variable_types, because CRANE's metric does not restrict
    variable names.

    RED (old all-spans per-example parser): fails because 'cur' is not in
      allowed_vars=['n','x'] → per-example grammar rejects it.
    GREEN (CRANE-faithful final-block): passes because any CNAME is allowed.
    """
    ev = _make_evaluator()
    example = {"variable_types": {"n": "int", "x": "float"}}
    # Phantom variable 'cur' — not in variable_types.
    output = "Let me compute step by step.\n<<n1 * cur>>"
    assert _syntax_pass(ev, output, example), (
        "Phantom variable 'cur' must pass CRANE-faithful syntax (any CNAME allowed). "
        "If this fails you are still running the old per-example strict parser."
    )


# ---------------------------------------------------------------------------
# Test 2 — Brace token should FAIL (explicit rejection in CRANE metric)
# ---------------------------------------------------------------------------

def test_brace_in_final_span_fails():
    """A span containing '{' or '}' must fail CRANE-faithful syntax.

    RED against CRANE-faithful: if the check_syntax function is not wired
    in, the old parser would see '{daily}' and fail for a different reason
    (brace is not a grammar token), giving True in the fallback path.
    """
    ev = _make_evaluator()
    example = {}
    output = "The answer is <<p1 + {daily}>>"
    assert not _syntax_pass(ev, output, example), (
        "Span containing '{' must fail CRANE-faithful syntax (explicit rejection)."
    )


def test_closing_brace_in_final_span_fails():
    """Closing '}' also triggers the explicit brace-rejection rule."""
    ev = _make_evaluator()
    output = "<<cost} + fee>>"
    assert not _syntax_pass(ev, output, {}), (
        "Span containing '}' must fail CRANE-faithful syntax."
    )


# ---------------------------------------------------------------------------
# Test 3 — '**' (exponentiation) should FAIL (not in GSM grammar)
# ---------------------------------------------------------------------------

def test_double_star_in_final_span_fails():
    """The '**' operator is absent from gsm.lark.

    CRANE's grammar only has *, /, //, %, +, -.
    A span like <<n0 * (1 + r) ** d>> must fail because '**' produces a
    parse error against the static grammar.

    Note: this was previously also failing our old per-example parser for the
    same reason (grammar doesn't have **), but we want to ensure it still
    fails under the new path.
    """
    ev = _make_evaluator()
    output = "<<n0 * (1 + r) ** d>>"
    assert not _syntax_pass(ev, output, {}), (
        "'**' operator must fail GSM syntax (not in grammar)."
    )


# ---------------------------------------------------------------------------
# Test 4 — No span at all should FAIL
# ---------------------------------------------------------------------------

def test_no_span_fails():
    """Output with no << >> span must fail syntax.

    GREEN in both old and new code (empty segments → False from
    example_syntax_pass_from_segments).
    """
    ev = _make_evaluator()
    output = "The answer is 42."
    assert not _syntax_pass(ev, output, {}), (
        "Output with no << >> span must fail syntax."
    )


# ---------------------------------------------------------------------------
# Test 5 — Multiple spans: only the FINAL one is checked
# ---------------------------------------------------------------------------

def test_multiple_spans_only_final_checked():
    """CRANE checks only the FINAL <<...>> block.

    If intermediate spans have phantom (undeclared) variable names but the
    final span is a valid arithmetic expression with any CNAME, the example
    should PASS under CRANE semantics.

    RED (old all-spans per-example parser): fails because ghost_var is not
      in variable_types ({'n': 'int', 'x': 'int'}), so the restricted parser
      rejects all intermediate spans → all_valid=False → False.
    GREEN (CRANE-faithful final-block, static grammar): only the last span
      <<5 + 3>> is checked and it parses under any-CNAME grammar → True.
    """
    ev = _make_evaluator()
    # Provide variable_types so the old code would use a restricted grammar.
    example = {"variable_types": {"n": "int", "x": "int"}}
    # Intermediate spans have phantom vars that old restricted grammar rejects.
    # Final span is a clean valid expression.
    output = (
        "Let me track intermediate steps: <<ghost_var * 3>>  "
        "and: <<another_phantom + 1>>"
        "Final: <<5 + 3>>"
    )
    assert _syntax_pass(ev, output, example), (
        "With multiple spans, only the final span should be checked (CRANE semantics). "
        "If this fails you're still checking all spans (old behavior)."
    )


def test_multiple_spans_final_has_brace_fails():
    """When the FINAL span has a brace, the example fails even if earlier spans are ok."""
    ev = _make_evaluator()
    output = "<<5 + 3>>  then: <<p + {frac}>>"
    assert not _syntax_pass(ev, output, {}), (
        "Final span with brace must fail even if earlier spans are valid."
    )


# ---------------------------------------------------------------------------
# Test 6 — round() should FAIL (explicit CRANE rejection)
# ---------------------------------------------------------------------------

def test_round_function_in_final_span_fails():
    """'round(' triggers explicit rejection in CRANE's check_gsm_parsed."""
    ev = _make_evaluator()
    output = "<<round(n * r)>>"
    assert not _syntax_pass(ev, output, {}), (
        "'round(' must fail CRANE-faithful syntax (explicit rejection)."
    )


# ---------------------------------------------------------------------------
# Test 7 — A valid arithmetic expression with any CNAME passes
# ---------------------------------------------------------------------------

def test_valid_static_grammar_expression_passes():
    """A syntactically valid GSM expression with arbitrary variable names passes."""
    ev = _make_evaluator()
    for expr in ["n * 3 + k", "total / count", "a + b * c - d // e % f"]:
        output = f"<<{expr}>>"
        assert _syntax_pass(ev, output, {}), (
            f"Valid expression <<{expr}>> must pass CRANE-faithful syntax."
        )


# ---------------------------------------------------------------------------
# Test 8 — SOURCE-LEVEL check: evaluator must delegate to benchmark logic
#           for GSM, not use the old all-spans per-example path.
# ---------------------------------------------------------------------------

def test_evaluator_source_delegates_gsm_to_benchmark_logic():
    """AST check: _check_syntax_validity must have a gsm_symbolic branch that
    calls logic.check_syntax (not _get_syntax_parser over all spans).

    This is a structural/source-level guard so the delegation can't silently
    regress without this test failing.
    """
    evaluator_path = _REPO / "synthesis" / "evaluate" / "evaluator.py"
    tree = ast.parse(evaluator_path.read_text())

    # Find _check_syntax_validity method body.
    method_body = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == "_check_syntax_validity":
                method_body = node
                break

    assert method_body is not None, "_check_syntax_validity not found in evaluator.py"

    # Collect all string constants in the method to check for "gsm_symbolic" branch.
    string_values = {
        node.value
        for node in ast.walk(method_body)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert "gsm_symbolic" in string_values, (
        "evaluator._check_syntax_validity must have a 'gsm_symbolic' branch. "
        "The old code falls through to the shared all-spans path, which is too strict. "
        "Add: if self.dataset_name == 'gsm_symbolic': return logic.check_syntax(...)"
    )

    # Check that there is a call to check_syntax somewhere in the method.
    call_names = set()
    for node in ast.walk(method_body):
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Attribute):
                call_names.add(f.attr)
            elif isinstance(f, ast.Name):
                call_names.add(f.id)
    assert "check_syntax" in call_names, (
        "evaluator._check_syntax_validity's gsm_symbolic branch must call "
        "logic.check_syntax(self, output, example) — the CRANE-faithful final-block check."
    )
