"""Spider's per-example syntax verdict is the bare-SQL output contract (the IterGen-aligned rule),
applied to the span content. It is not the generic "every closed span parses" rule."""
from synthesis.evaluate.benchmarks.sql_spider import eval_logic


def test_syntax_verdict_is_the_contract_verdict_not_the_segment_verdict():
    segments = [("SELECT 1", True)]
    assert eval_logic.example_syntax_pass(True, segments, {"syntax_valid": False}) is False
    assert eval_logic.example_syntax_pass(False, [], {"syntax_valid": True}) is True
    assert eval_logic.example_syntax_pass(True, segments, None) is False


def test_contract_sees_span_content_without_the_delimiters():
    assert eval_logic._span_content_for_scoring("<<SELECT 1>>") == "SELECT 1"
    assert eval_logic._span_content_for_scoring("SELECT 1") == "SELECT 1"


def test_only_a_closed_span_is_an_answer():
    assert eval_logic._span_content_for_scoring("<<SELECT 1>> trailing free text") == "SELECT 1"
    assert eval_logic._span_content_for_scoring("<<SELECT 1>> <<SELECT 2>>") == "SELECT 2"
    assert eval_logic._span_content_for_scoring("<<SELECT 1") == ""
    assert eval_logic._span_content_for_scoring("<<SELECT 1>> <<SELECT 2") == "SELECT 1"
    assert eval_logic._span_content_for_scoring("") == ""
