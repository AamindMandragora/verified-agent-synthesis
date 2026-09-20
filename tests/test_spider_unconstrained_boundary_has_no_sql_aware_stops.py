"""The unconstrained Spider baseline must not stop generation on SQL-aware markers.

Why this file exists
---------------------
``_spider_completion_has_clean_boundary`` decided when ``run_unconstrained_spider_adapter``
could stop generating. Before this fix it stopped not only on a blank line, but also on a
semicolon, a repeated ``SELECT``, and few-shot labels (``SQL:``, ``question:``, ``db_info:``,
...). Those are SQL-aware stops: they assume the answer is a SQL query and know what one
looks like. An "unconstrained" baseline is supposed to have NO grammar-driven help at
generation time -- those checks belong in the scorer (applied uniformly to every span-free
answer, not just this one baseline), not in the stopping criteria. This file pins the
generation-time function down to its one legitimate stop: a blank line.
"""

from __future__ import annotations

from synthesis.evaluate.run_legacy_fixed_strategy import (
    _spider_completion_has_clean_boundary,
)


def test_blank_line_is_still_a_clean_boundary():
    assert _spider_completion_has_clean_boundary("SELECT * FROM singer\n\nHuman: next") is True


def test_no_text_yet_is_not_a_clean_boundary():
    assert _spider_completion_has_clean_boundary("") is False


def test_a_semicolon_alone_no_longer_stops_generation():
    """Cutting at ';' assumes the answer is SQL -- that belongs in the scorer, not here."""
    assert _spider_completion_has_clean_boundary("SELECT * FROM singer;") is False


def test_a_repeated_select_no_longer_stops_generation():
    assert _spider_completion_has_clean_boundary("SELECT * FROM singer SelEct name") is False


def test_a_repeated_few_shot_label_no_longer_stops_generation():
    for marker_text in (
        "SELECT * FROM singer\nSQL: SELECT name FROM singer",
        "SELECT * FROM singer\nquestion: another one?",
        "SELECT * FROM singer\ndb_info: # other_table",
        "SELECT * FROM singer\nHuman: what about now",
    ):
        assert _spider_completion_has_clean_boundary(marker_text) is False, marker_text
