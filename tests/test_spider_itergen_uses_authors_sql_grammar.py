"""The Spider IterGen lane must load a grammar that names the units it backs up on.

Why this file exists
---------------------
``_itergen_generate_spider`` asks IterGen for the units ``column_name`` and
``table_name`` (``iter_gen.forward(units=["column_name", "table_name"])``,
then ``iter_gen.view("column_name")`` / ``.view("table_name")``). IterGen can
only ever report or back up on a symbol its own grammar defines. Our repo's
``synthesis/evaluate/grammars/sql.lark`` names the same concepts
``column_ref`` / ``table_ref`` -- neither symbol name IterGen was asked
for -- so ``view()`` always returned an empty list and the back-up step
(``iter_gen.backward(...)``) could never fire. The "Spider IterGen" lane was
really a single unconstrained decode wearing IterGen's name.

The fix pins IterGen's own SQL grammar in-repo (see
``synthesis/evaluate/baselines/itergen_authors/README.md``) and loads it for
the Spider IterGen lane specifically. This test checks the grammar text
itself defines the two symbols the loop depends on, so a future edit to
either file cannot silently reintroduce the mismatch.
"""

from __future__ import annotations

import re

from synthesis.evaluate.run_legacy_fixed_strategy import (
    _itergen_spider_authors_grammar_text,
)


def test_authors_grammar_defines_column_name_and_table_name():
    grammar_text = _itergen_spider_authors_grammar_text()

    assert re.search(r"(?m)^column_name\s*:", grammar_text), (
        "The grammar the Spider IterGen lane loads must define `column_name` "
        "as a rule -- that is the exact unit name the loop asks view()/backward() "
        "for. Without it the back-up step can never fire."
    )
    assert re.search(r"(?m)^table_name\s*:", grammar_text), (
        "The grammar the Spider IterGen lane loads must define `table_name` "
        "as a rule, for the same reason as column_name above."
    )


def test_authors_grammar_is_not_our_own_sql_grammar():
    """Our own grammar names these symbols differently; loading it would silently
    reintroduce the vacuous loop this fix closes."""
    from pathlib import Path

    grammar_text = _itergen_spider_authors_grammar_text()
    our_grammar_text = (
        Path(__file__).resolve().parents[1]
        / "synthesis"
        / "evaluate"
        / "grammars"
        / "sql.lark"
    ).read_text()

    assert grammar_text != our_grammar_text
    assert "column_ref" in our_grammar_text and "column_ref" not in grammar_text
