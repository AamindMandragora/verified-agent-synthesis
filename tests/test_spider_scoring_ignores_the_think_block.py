from synthesis.evaluate.benchmarks.sql_spider.eval_logic import _span_content_for_scoring


def test_empty_think_block_is_dropped_before_a_bare_answer():
    assert _span_content_for_scoring("<think>\n\n</think>\n\nSELECT 1") == "SELECT 1"


def test_a_span_inside_the_think_block_is_not_the_answer():
    assert _span_content_for_scoring("<think>maybe <<SELECT 0>></think><<SELECT 1>>") == "SELECT 1"


def test_unfinished_think_block_means_no_answer():
    assert _span_content_for_scoring("<think>\nThe user wants <<SELECT 1>>") == ""


def test_outputs_without_a_think_block_are_unchanged():
    assert _span_content_for_scoring("<<SELECT 1>>") == "SELECT 1"
    assert _span_content_for_scoring("SELECT 1") == "SELECT 1"


def test_unfinished_think_block_after_an_answer_is_cut_off():
    out = "<<SELECT a FROM t>>\n<think>wait, maybe <<SELECT b FROM t>>"
    assert _span_content_for_scoring(out) == "SELECT a FROM t"
