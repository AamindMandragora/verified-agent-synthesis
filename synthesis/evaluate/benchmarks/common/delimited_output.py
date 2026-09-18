"""Shared << >> delimited-span extraction for benchmark scoring."""

from __future__ import annotations

import re

# ``.*?`` (not ``[^<>]``) so a span body containing single ``<``/``>`` chars --
# e.g. SQL comparison operators ``age < 30`` / ``COUNT(*) > 1`` -- is captured whole.
# Lazy match still stops at the first ``>>`` delimiter; DOTALL keeps the old
# behaviour of spanning newlines inside a delimited region.
DELIMITED_SPAN_PATTERN = re.compile(r"<<\s*(.*?)\s*>>", re.DOTALL)


def find_delimited_spans(text: str) -> list[str]:
    """Return inner spans from all ``<< ... >>`` regions in document order."""
    if not text:
        return []
    return DELIMITED_SPAN_PATTERN.findall(text)


def normalize_inline_text(text: str, *, strip_semicolon: bool = False) -> str:
    """Collapse whitespace and optional trailing semicolons for SQL-like spans."""
    cleaned = text.replace("\n", " ").replace("\r", " ").strip()
    cleaned = " ".join(cleaned.split())
    if strip_semicolon:
        cleaned = cleaned.rstrip(";").strip()
    return cleaned


def extract_last_delimited_span(
    text: str,
    *,
    normalize_whitespace: bool = False,
    strip_semicolon: bool = False,
) -> tuple[str | None, bool]:
    """
    Return the last ``<< >>`` inner span and whether any delimiter was found.

    When ``normalize_whitespace`` is true, apply :func:`normalize_inline_text`.
    """
    matches = find_delimited_spans(text)
    if not matches:
        return None, False
    span = matches[-1]
    if normalize_whitespace:
        span = normalize_inline_text(span, strip_semicolon=strip_semicolon)
    else:
        span = span.strip()
    return (span or None), True
