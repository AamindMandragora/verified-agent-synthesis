"""SQL-aware cleanup for span-free Spider answers belongs in the scorer.

Why this file exists
---------------------
The unconstrained Spider baseline used to stop GENERATING as soon as its output
contained a semicolon, a repeated ``SELECT``, or a repeated few-shot label
(``SQL:``, ``question:``, ...). That gave it grammar-aware help it isn't supposed
to have. This cleanup is still correct, though -- it's just a scoring-time question
("where does the actual query end"), and every span-free Spider answer should get
it, not only the unconstrained baseline's. ``_span_content_for_scoring``'s no-spans
branch (see test_spider_scoring_ignores_the_think_block.py for the existing
sql-label/blank-line behavior it must keep) is where it now lives.
"""

from __future__ import annotations

from synthesis.evaluate.benchmarks.sql_spider.eval_logic import _span_content_for_scoring


def test_a_span_free_answer_is_cut_at_a_semicolon():
    assert _span_content_for_scoring("SELECT * FROM singer;") == "SELECT * FROM singer"


def test_a_semicolon_at_the_very_start_does_not_cut_an_empty_answer():
    # Matches the existing "> 0" guard: a leading ';' isn't a real query boundary.
    assert _span_content_for_scoring(";SELECT * FROM singer") == ";SELECT * FROM singer"


def test_a_span_free_answer_is_cut_at_a_repeated_select():
    out = "SELECT * FROM singer SelEct name FROM singer"
    assert _span_content_for_scoring(out) == "SELECT * FROM singer"


def test_a_span_free_answer_is_cut_at_a_repeated_sql_label():
    out = "SELECT * FROM singer\nSQL: SELECT name FROM singer"
    assert _span_content_for_scoring(out) == "SELECT * FROM singer"


def test_a_span_free_answer_is_cut_at_a_chat_role_label():
    out = "SELECT * FROM singer\nHuman: what about this one"
    assert _span_content_for_scoring(out) == "SELECT * FROM singer"


def test_a_span_free_answer_is_cut_at_a_repeated_schema_label():
    out = "SELECT * FROM singer\ndb_info: # other_table"
    assert _span_content_for_scoring(out) == "SELECT * FROM singer"
