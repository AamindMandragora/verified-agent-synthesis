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

# Reverse mapping: show gold FOL in prompt using grammar keywords so model can copy into << >>
FOL_UNICODE_TO_KEYWORD = {v: k for k, v in FOL_KEYWORD_TO_UNICODE.items()}


def fol_keyword_to_unicode(text: str) -> str:
    """Convert {keyword} FOL syntax from grammar to Unicode symbols for Prover9."""
    for keyword, symbol in FOL_KEYWORD_TO_UNICODE.items():
        text = text.replace(keyword, symbol)
    return text


def fol_unicode_to_keyword(text: str) -> str:
    """Convert Unicode FOL (e.g. from dataset) to grammar {keyword} form for prompts."""
    if not text:
        return text
    out = text
    for symbol, keyword in FOL_UNICODE_TO_KEYWORD.items():
        out = out.replace(symbol, keyword)
    return out
