"""Qwen2.5 wraps Spider answers in ```sql fences (290/300 rows for 7B unconstrained on
2026-09-30). The fence is formatting, not SQL: scoring takes the fenced query, for every
method (span-free baselines and span content alike). User ruling 2026-10-01."""
from synthesis.evaluate.benchmarks.sql_spider.eval_logic import _span_content_for_scoring


def test_span_free_fenced_answer_scores_the_query():
    raw = " \n```sql\nSELECT name, country, age \nFROM singer \nORDER BY age DESC;\n```"
    assert _span_content_for_scoring(raw) == "SELECT name, country, age \nFROM singer \nORDER BY age DESC"


def test_fence_after_prose_scores_the_fenced_query():
    raw = "First we join the tables.\n```sql\nSELECT a FROM t\n```\nThis returns a."
    assert _span_content_for_scoring(raw) == "SELECT a FROM t"


def test_fence_inside_a_span_is_stripped():
    assert _span_content_for_scoring("<<```sql\nSELECT 1 FROM t\n```>>") == "SELECT 1 FROM t"


def test_unclosed_fence_scores_what_follows_it():
    assert _span_content_for_scoring("```sql\nSELECT a FROM t") == "SELECT a FROM t"


def test_unfenced_answers_are_unchanged():
    assert _span_content_for_scoring("SQL: SELECT a FROM t;\n\nSQL: SELECT b") == "SELECT a FROM t"
    assert _span_content_for_scoring("<<SELECT a FROM t>>") == "SELECT a FROM t"
