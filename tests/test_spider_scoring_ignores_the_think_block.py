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


def test_a_leading_sql_label_is_not_part_of_the_answer():
    # Unconstrained models continue the few-shot format and write "SQL: SELECT ...".
    assert _span_content_for_scoring("<think>\n\n</think>\n\nSQL: SELECT 1") == "SELECT 1"
    assert _span_content_for_scoring("sql:SELECT 1") == "SELECT 1"


def test_a_span_free_answer_ends_at_the_first_blank_line():
    # Qwen3.5-4B answers, leaves a blank line, then starts the answer over again.
    out = " SELECT a FROM t\n\n<think>\n\n</think>\n\nSQL: SELECT a FROM t"
    assert _span_content_for_scoring(out) == "SELECT a FROM t"
