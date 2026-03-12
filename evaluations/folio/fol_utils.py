"""
FOL (first-order logic) utilities shared by FOLIO CLI and synthesis evaluator.

Maps grammar {keyword} placeholders to Unicode symbols for Prover9.
"""

from __future__ import annotations

FOL_KEYWORD_TO_UNICODE = {
    "{forall}": "∀",
    "{exists}": "∃",
    "{and}": "∧",
    "{or}": "∨",
    "{xor}": "⊕",
    "{not}": "¬",
    "{implies}": "→",
    "{iff}": "↔",
}


def fol_keyword_to_unicode(text: str) -> str:
    """Convert {keyword} FOL syntax from grammar to Unicode symbols for Prover9."""
    for keyword, symbol in FOL_KEYWORD_TO_UNICODE.items():
        text = text.replace(keyword, symbol)
    return text
