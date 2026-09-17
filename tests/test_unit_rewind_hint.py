"""Tests for FIX B: _unit_rewind_hint in feedback_loop.py.

The hint fires when BOTH:
  (a) Failures are dominated by syntax-valid but semantically wrong outputs
      (syntax_rate - accuracy >= 0.25, or most failing examples are syntax-valid)
  (b) The strategy source does NOT reference RegenerateUnitOnCheckFailure

It must NOT fire when:
  - The strategy already uses RegenerateUnitOnCheckFailure
  - Failures are mostly syntax failures (syntax_rate low, not semantics-dominated)

Test strategy: import just the function (feedback_loop.py has no vllm/torch at
module level, but it does import from synthesis.* — so we use importlib to load
only the function we need, falling back to ast extraction + exec if the import
fails due to missing deps).
"""
import ast
import sys
from pathlib import Path

FEEDBACK_LOOP = (
    Path(__file__).resolve().parent.parent
    / "synthesis"
    / "evaluate"
    / "feedback_loop.py"
)


def _get_hint_fn():
    """Try to import _unit_rewind_hint; fall back to ast extraction + exec."""
    # Try sys.path manipulation to import just the function
    repo_root = str(Path(__file__).resolve().parent.parent)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    try:
        # Only import the function, not the whole module (avoids heavy deps)
        import importlib.util
        spec = importlib.util.spec_from_file_location("_fbloop", FEEDBACK_LOOP)
        # We can't easily import just one function without executing the whole module.
        # Fall back to AST extraction.
        raise ImportError("use ast extraction")
    except Exception:
        pass

    src = FEEDBACK_LOOP.read_text()
    tree = ast.parse(src)
    fn_node = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_unit_rewind_hint":
            fn_node = node
            break
    assert fn_node is not None, "_unit_rewind_hint not found in feedback_loop.py"

    fn_src = ast.get_source_segment(src, fn_node)
    assert fn_src, "Could not extract _unit_rewind_hint source"

    ns = {}
    exec(fn_src, ns)
    return ns["_unit_rewind_hint"]


# ---------------------------------------------------------------------------
# Structural test: the function exists with the right signature
# ---------------------------------------------------------------------------

def test_unit_rewind_hint_exists():
    """_unit_rewind_hint must exist in feedback_loop.py with correct parameters."""
    src = FEEDBACK_LOOP.read_text()
    tree = ast.parse(src)
    fn_nodes = [
        n for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "_unit_rewind_hint"
    ]
    assert fn_nodes, "_unit_rewind_hint function not found in feedback_loop.py"
    fn = fn_nodes[0]
    arg_names = [a.arg for a in fn.args.args]
    # Must accept strategy_source and sample_outputs
    assert "strategy_source" in arg_names, (
        f"_unit_rewind_hint must have a 'strategy_source' parameter, got: {arg_names}"
    )
    assert "sample_outputs" in arg_names, (
        f"_unit_rewind_hint must have a 'sample_outputs' parameter, got: {arg_names}"
    )


def test_unit_rewind_hint_wired_into_threshold_feedback():
    """_unit_rewind_hint must be called in the threshold_feedback path."""
    src = FEEDBACK_LOOP.read_text()
    # Find the threshold_feedback block and check that _unit_rewind_hint is referenced.
    # The simplest structural check: the function name appears in the same region
    # as "threshold_feedback".
    assert "_unit_rewind_hint" in src, "_unit_rewind_hint is not referenced anywhere in feedback_loop.py"
    # More specifically, it should appear near both feedback paths.
    idx_threshold = src.find("threshold_feedback")
    idx_evaluation = src.find("evaluation_feedback")
    assert idx_threshold >= 0, "threshold_feedback not found"
    assert idx_evaluation >= 0, "evaluation_feedback not found"

    # Check it appears at least twice (once per feedback path)
    occurrences = [i for i in range(len(src)) if src.startswith("_unit_rewind_hint", i)]
    # At least: function definition + 2 call sites (threshold + evaluation)
    assert len(occurrences) >= 3, (
        f"_unit_rewind_hint appears only {len(occurrences)} time(s) in feedback_loop.py; "
        f"expected at least 3 (definition + 2 call sites for threshold and evaluation paths)"
    )


# ---------------------------------------------------------------------------
# Behavioral tests via AST extraction + exec
# ---------------------------------------------------------------------------

def _make_sample(is_syntax_valid: bool, is_correct: bool):
    return {
        "is_syntax_valid": is_syntax_valid,
        "is_correct": is_correct,
        "contains_delimiters": True,
        "full_output": "<<SELECT id FROM t>>",
        "actual": "SELECT id FROM t" if is_syntax_valid else None,
        "runtime_budget_exceeded": False,
        "error": None,
    }


def _dominated_by_semantic_failures(n_syn_valid_wrong=20, n_syn_invalid_wrong=3, n_correct=5):
    """Build sample_outputs dominated by syntax-valid but wrong examples."""
    samples = []
    samples += [_make_sample(True, True)] * n_correct
    samples += [_make_sample(True, False)] * n_syn_valid_wrong   # semantic failures
    samples += [_make_sample(False, False)] * n_syn_invalid_wrong  # syntax failures
    return samples


def _dominated_by_syntax_failures(n_syn_valid_wrong=2, n_syn_invalid_wrong=20, n_correct=5):
    """Build sample_outputs dominated by syntax-invalid examples."""
    samples = []
    samples += [_make_sample(True, True)] * n_correct
    samples += [_make_sample(True, False)] * n_syn_valid_wrong
    samples += [_make_sample(False, False)] * n_syn_invalid_wrong
    return samples


STRATEGY_WITHOUT_HELPER = """
// CSD_RATIONALE_BEGIN some rationale CSD_RATIONALE_END
method GenerateCSD(input: string) {
    var x := UnconstrainedChunk(10, "<<", eos);
    OpenConstrainedSpan();
    SafeRepetitionPenaltyStep(2.0);
    CloseConstrainedSpan();
}
"""

STRATEGY_WITH_HELPER = """
// CSD_RATIONALE_BEGIN some rationale CSD_RATIONALE_END
method GenerateCSD(input: string) {
    var x := UnconstrainedChunk(10, "<<", eos);
    OpenConstrainedSpan();
    RegenerateUnitOnCheckFailure(allowed_units, check_fn);
    SafeRepetitionPenaltyStep(2.0);
    CloseConstrainedSpan();
}
"""


def test_hint_fires_when_semantic_failures_dominate_and_helper_absent():
    hint_fn = _get_hint_fn()
    samples = _dominated_by_semantic_failures()
    result = hint_fn(STRATEGY_WITHOUT_HELPER, samples)
    assert result, (
        "_unit_rewind_hint returned empty string but should fire: "
        "semantic failures dominate and RegenerateUnitOnCheckFailure is absent"
    )
    assert "RegenerateUnitOnCheckFailure" in result, (
        "Hint must mention the helper name 'RegenerateUnitOnCheckFailure'"
    )


def test_hint_does_not_fire_when_strategy_uses_helper():
    hint_fn = _get_hint_fn()
    samples = _dominated_by_semantic_failures()
    result = hint_fn(STRATEGY_WITH_HELPER, samples)
    assert not result, (
        "_unit_rewind_hint should NOT fire when strategy already uses RegenerateUnitOnCheckFailure"
    )


def test_hint_does_not_fire_when_syntax_failures_dominate():
    hint_fn = _get_hint_fn()
    samples = _dominated_by_syntax_failures()
    result = hint_fn(STRATEGY_WITHOUT_HELPER, samples)
    assert not result, (
        "_unit_rewind_hint should NOT fire when failures are mostly syntax failures"
    )


def test_hint_does_not_fire_with_empty_samples():
    hint_fn = _get_hint_fn()
    result = hint_fn(STRATEGY_WITHOUT_HELPER, [])
    assert not result, "_unit_rewind_hint should return empty string for empty sample_outputs"


def test_hint_does_not_fire_with_none_samples():
    hint_fn = _get_hint_fn()
    result = hint_fn(STRATEGY_WITHOUT_HELPER, None)
    assert not result, "_unit_rewind_hint should return empty string for None sample_outputs"


def test_hint_text_is_mechanism_level():
    """Hint must mention the helper mechanism and be task-agnostic (no SQL/JOIN/query mentions)."""
    hint_fn = _get_hint_fn()
    samples = _dominated_by_semantic_failures()
    result = hint_fn(STRATEGY_WITHOUT_HELPER, samples)
    assert result

    result_lower = result.lower()
    # Task-specific terms are banned
    forbidden = ["join", "select", "where", "sql", "query", "table", "schema", "column"]
    for term in forbidden:
        assert term not in result_lower, (
            f"Hint contains task-specific term '{term}': {result!r}"
        )
    # Mechanism-level terms must be present
    assert "unit" in result_lower or "grammar" in result_lower or "rewind" in result_lower or "resample" in result_lower, (
        f"Hint does not mention unit/grammar/rewind/resample: {result!r}"
    )


def test_hint_boundary_condition_exactly_at_threshold():
    """syntax_rate - accuracy = exactly 0.25 should fire (boundary case)."""
    hint_fn = _get_hint_fn()
    # 28 correct, 28 syntax-valid-wrong (28/56 acc = 0.50; 56/56 syn = 1.0; diff = 0.50)
    # That's well above 0.25. Test with exact 0.25:
    # 15 correct, 25 total; 10 syntax-valid-wrong, 0 syntax-invalid:
    # acc = 15/25 = 0.60, syn = 25/25 = 1.0, diff = 0.40 >= 0.25 -> fires
    samples = [_make_sample(True, True)] * 15 + [_make_sample(True, False)] * 10
    result = hint_fn(STRATEGY_WITHOUT_HELPER, samples)
    assert result, "hint should fire when syntax_rate - accuracy = 0.40 >= 0.25"


def test_hint_does_not_fire_when_diff_below_threshold():
    """When syntax_rate - accuracy < 0.25, the hint should not fire."""
    hint_fn = _get_hint_fn()
    # 8 correct, 10 syntax-valid-wrong, 10 syntax-invalid-wrong → 28 total
    # syn_valid = 8+10=18, acc = 8/28 = 0.286, syn = 18/28 = 0.643, diff = 0.357
    # That fires. Let's make diff small: 20 correct, 2 syn-valid-wrong, 8 syn-invalid-wrong
    # acc = 20/30 = 0.667, syn = 22/30 = 0.733, diff = 0.067 < 0.25
    samples = [_make_sample(True, True)] * 20 + [_make_sample(True, False)] * 2 + [_make_sample(False, False)] * 8
    result = hint_fn(STRATEGY_WITHOUT_HELPER, samples)
    assert not result, (
        "hint should NOT fire when syntax_rate - accuracy < 0.25 "
        f"(got result: {result!r})"
    )
