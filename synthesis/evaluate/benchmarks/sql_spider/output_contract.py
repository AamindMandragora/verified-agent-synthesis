"""Strict output contract for Spider SQL completions."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any


class SpiderEvidenceContractError(RuntimeError):
    """Raised when committed Spider token evidence cannot match scored text."""


@dataclass(frozen=True)
class SpiderOutputContractResult:
    accepted: bool
    sql: str | None
    rejection_reason: str | None
    raw_output: str


def strip_terminal_special_token_ids(
    token_ids: list[int],
    tokenizer: Any,
    *,
    terminal_stop_token_ids: Any,
) -> list[int]:
    """Remove only IDs in the generation adapter's exact terminal-stop set."""
    del tokenizer
    declared = {int(value) for value in (terminal_stop_token_ids or ())}
    result = [int(token_id) for token_id in token_ids]
    while result and result[-1] in declared:
        result.pop()
    return result



def _flat_token_ids(token_ids: Any) -> list[int]:
    if hasattr(token_ids, "detach"):
        token_ids = token_ids.detach().cpu()
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    while isinstance(token_ids, list) and len(token_ids) == 1 and isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    return [int(token_id) for token_id in token_ids]


def generation_token_evidence(
    token_ids: Any,
    tokenizer: Any,
    *,
    terminal_stop_token_ids: Any,
) -> dict[str, Any]:
    """Decode generated IDs before/after exact terminal-stop removal."""
    raw_ids = _flat_token_ids(token_ids)
    decoded_ids = strip_terminal_special_token_ids(
        raw_ids,
        tokenizer,
        terminal_stop_token_ids=terminal_stop_token_ids,
    )
    removed_ids = raw_ids[len(decoded_ids):]

    def _decode(ids: list[int]) -> str:
        try:
            return str(tokenizer.decode(ids, skip_special_tokens=False))
        except TypeError:
            return str(tokenizer.decode(ids))

    return {
        "raw_token_ids": raw_ids,
        "raw_decoded_text": _decode(raw_ids),
        "removed_terminal_token_ids": removed_ids,
        "decoded_text": _decode(decoded_ids),
    }


_OUTSIDE_MARKERS = (
    "<<",
    ">>",
    "```",
    "<think>",
    "</think>",
    "<|assistant|>",
    "<|user|>",
    "<|system|>",
)
_LABEL_RE = re.compile(
    r"(?i)\b(?:sql|db_id|db_info|question|assistant|user|system)\s*:"
)
_ROLE_MARKER_RE = re.compile(r"(?i)<\|(?:assistant|user|system)\|>")
_PROSE_PREFIX_RE = re.compile(
    r"(?is)^\s*(?:here\s+is|the\s+(?:sql\s+)?query|answer|explanation)\b"
)
_PROSE_SUFFIX_RE = re.compile(
    r"(?is)(?:^|\r?\n)\s*(?:here\s+is|the\s+(?:sql\s+)?query|answer|explanation|this\s+is)\b"
)


def _outside_spans(text: str):
    in_string = False
    in_line_comment = False
    index = 0
    while index < len(text):
        if in_line_comment:
            if text[index] == "\n":
                in_line_comment = False
            index += 1
            continue
        if in_string:
            if text[index] == "'":
                if index + 1 < len(text) and text[index + 1] == "'":
                    index += 2
                    continue
                in_string = False
            index += 1
            continue
        if text[index] == "'":
            in_string = True
            index += 1
            continue
        if text.startswith("--", index):
            in_line_comment = True
            index += 2
            continue
        yield index
        index += 1


def _outside_marker(text: str) -> str | None:
    outside = set(_outside_spans(text))
    for marker in _OUTSIDE_MARKERS:
        for index in outside:
            if text.startswith(marker, index):
                return "prompt_or_wrapper"
    for match in _ROLE_MARKER_RE.finditer(text):
        if match.start() in outside:
            return "prompt_or_wrapper"
    for match in _LABEL_RE.finditer(text):
        if match.start() in outside:
            return "prompt_or_wrapper"
    if _PROSE_PREFIX_RE.match(text):
        return "prompt_or_wrapper"
    for match in _PROSE_SUFFIX_RE.finditer(text):
        if match.start() in outside:
            return "prompt_or_wrapper"
    return None


def _trailing_whitespace_only(text: str) -> bool:
    return not text.strip()


def _parser_lexical_view(text: str) -> str:
    """Normalize parser-hostile layout without changing the scored candidate."""
    pieces: list[str] = []
    in_string = False
    in_line_comment = False
    index = 0
    while index < len(text):
        char = text[index]
        if in_line_comment:
            if char in "\r\n":
                in_line_comment = False
                pieces.append(" ")
                if char == "\r" and index + 1 < len(text) and text[index + 1] == "\n":
                    index += 2
                else:
                    index += 1
            else:
                index += 1
            continue
        if in_string:
            pieces.append(char)
            if char == "'":
                if index + 1 < len(text) and text[index + 1] == "'":
                    pieces.append("'")
                    index += 2
                    continue
                in_string = False
            index += 1
            continue
        if char == "'":
            in_string = True
            pieces.append(char)
            index += 1
            continue
        if text.startswith("--", index):
            in_line_comment = True
            pieces.append(" ")
            index += 2
            continue
        if char in "\r\n":
            pieces.append(" ")
            if char == "\r" and index + 1 < len(text) and text[index + 1] == "\n":
                index += 2
            else:
                index += 1
            continue
        pieces.append(char)
        index += 1
    return "".join(pieces)


def _single_statement(text: str) -> tuple[str | None, str | None]:
    semicolons = [index for index in _outside_spans(text) if text[index] == ";"]
    if not semicolons:
        return text.strip(), None
    if len(semicolons) != 1:
        return None, "multiple_statements"
    index = semicolons[0]
    if not _trailing_whitespace_only(text[index + 1:]):
        return None, "multiple_statements"
    return text[:index].strip(), None

def validate_bare_sql(output: str, *, parser: Any = None) -> SpiderOutputContractResult:
    """Accept one parser-valid SQL statement with no outer prompt/prose wrapper."""
    raw_output = output if isinstance(output, str) else str(output)
    candidate = raw_output.strip()
    if not candidate:
        return SpiderOutputContractResult(False, None, "empty", raw_output)
    marker_reason = _outside_marker(candidate)
    if marker_reason is not None:
        return SpiderOutputContractResult(False, None, marker_reason, raw_output)
    sql, statement_reason = _single_statement(candidate)
    if statement_reason is not None or not sql:
        return SpiderOutputContractResult(
            False,
            None,
            statement_reason or "empty",
            raw_output,
        )
    if parser is not None:
        try:
            parser.parse(_parser_lexical_view(sql))
        except Exception:
            return SpiderOutputContractResult(
                False,
                None,
                "invalid_or_non_bare_sql",
                raw_output,
            )
    return SpiderOutputContractResult(True, sql, None, raw_output)
