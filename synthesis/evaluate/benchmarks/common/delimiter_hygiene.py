"""Keep span delimiters out of free text, so the output's spans are exactly the ones the decoder opened.

The decoder contract speaks about the spans of the final output: every `<< ... >>` region must be
parser-valid. That is only meaningful if free (unconstrained) text can never create or close a span on
its own. Tokenizers make that easy to get wrong: ` <<` is one token, `<<<` is one token, and `<` + `<`
forms a delimiter across two tokens. This module is the single text-level rule both unconstrained
sampling paths use:

  * `walk_spans` reads rendered output and says where the spans are and whether the text ends inside one.
    It mirrors the Dafny ghost function `SpansOf`, and the evaluator's `<<(.*?)>>` extraction.
  * `build_delimiter_token_sets` classifies a vocabulary once per tokenizer.
  * `banned_ids_outside_span` gives the token ids that may not be sampled as free text, given how the
    free text currently ends.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

OPEN = "<<"
CLOSE = ">>"


@dataclass(frozen=True)
class Span:
    content: str
    closed: bool


@dataclass(frozen=True)
class SpanWalk:
    spans: tuple[Span, ...]
    ends_inside: bool
    # Free text outside every span, concatenated. Must never contain a delimiter.
    free_text_tail: str


def walk_spans(text: str) -> SpanWalk:
    """Split rendered output into spans. Outside a span `<<` opens one; inside, the first `>>` closes it.

    Every task's output carries its spans in the text: when the runtime opens the span itself, it does so
    by putting a literal `<<` in the output, so there is one rule here and no hidden-span variant.
    """
    spans: list[Span] = []
    inside = False
    span_start = 0
    free_tail_start = 0
    i = 0
    n = len(text)
    while i < n:
        if not inside and text.startswith(OPEN, i):
            inside = True
            i += len(OPEN)
            span_start = i
        elif inside and text.startswith(CLOSE, i):
            spans.append(Span(text[span_start:i], closed=True))
            inside = False
            i += len(CLOSE)
            free_tail_start = i
        else:
            i += 1
    if inside:
        spans.append(Span(text[span_start:], closed=False))
        return SpanWalk(tuple(spans), True, "")
    return SpanWalk(tuple(spans), False, text[free_tail_start:])


def free_text_is_clean(text: str, start_inside: bool = False) -> bool:
    """True when no `>>` sits outside a span. (`<<` outside a span always opens one, so it cannot be stray.)"""
    inside = start_inside
    i = 0
    while i < len(text):
        if not inside and text.startswith(OPEN, i):
            inside = True
            i += 2
        elif inside and text.startswith(CLOSE, i):
            inside = False
            i += 2
        elif not inside and text.startswith(CLOSE, i):
            return False
        else:
            i += 1
    return True


@dataclass(frozen=True)
class DelimiterTokenSets:
    # Token text is optional whitespace then exactly `<<` (e.g. "<<", " <<"): a span opener in rendered text.
    opener_ids: frozenset[int]
    # The subset whose text is exactly `<<`. The one-token-at-a-time path allows only these, because a
    # decoder recognises an opener by comparing the token to "<<"; " <<" would slip through as free text.
    exact_opener_ids: frozenset[int]
    # Contains `>>`, or contains `<<` in any other shape (`<<<`, `<<(`, `)<<`, ...). Never legal as free text.
    always_banned_ids: frozenset[int]
    starts_with_lt_ids: frozenset[int]
    starts_with_gt_ids: frozenset[int]


def is_opener_text(token_text: str) -> bool:
    return token_text.lstrip() == OPEN and token_text.endswith(OPEN)


def build_delimiter_token_sets(id_to_text: Mapping[int, str]) -> DelimiterTokenSets:
    openers, exact, banned, lt, gt = set(), set(), set(), set(), set()
    for tid, text in id_to_text.items():
        if is_opener_text(text):
            openers.add(tid)
            if text == OPEN:
                exact.add(tid)
        elif OPEN in text or CLOSE in text:
            banned.add(tid)
        if text.startswith("<"):
            lt.add(tid)
        if text.startswith(">"):
            gt.add(tid)
    return DelimiterTokenSets(frozenset(openers), frozenset(exact), frozenset(banned), frozenset(lt), frozenset(gt))


def banned_ids_outside_span(sets: DelimiterTokenSets, free_text_tail: str) -> frozenset[int]:
    """Token ids that may not be sampled as free text when the free text so far ends with `free_text_tail`."""
    banned = sets.always_banned_ids | (sets.opener_ids - sets.exact_opener_ids)
    if free_text_tail.endswith("<"):
        # `<` + `<...` would form `<<` across the token boundary (and `<` + `<<` would form `<<<`).
        banned = banned | sets.starts_with_lt_ids
    if free_text_tail.endswith(">"):
        banned = banned | sets.starts_with_gt_ids
    return banned


def scrub_free_text(free_text_tail: str, text: str) -> str:
    """Free text produced several tokens at once cannot be masked token by token, so clean it after
    the fact: drop any `>` that would follow a `>`, and a leading `<` that would join a `<` already
    in the output. An in-text `<<` is kept: the caller treats it as the span opener."""
    kept: list[str] = []
    prev = free_text_tail[-1:]
    for i, ch in enumerate(text):
        if ch == ">" and prev == ">":
            continue
        if i == 0 and ch == "<" and prev == "<":
            continue
        kept.append(ch)
        prev = ch
    return "".join(kept)


def closer_allowed(content_text: str, token_text: str) -> bool:
    """Inside a span: may `token_text` follow `content_text`?

    `>>` is how the output text ends a span, and it is always written by the span helper
    (CloseConstrainedSpan), never by the grammar: no grammar here accepts `>>` in span content. The
    grammar mask over-approximates, though (whitespace-led tokens such as ` >>` slip through), so this
    bans every token that would put `>>` into the content, whole or completed across the boundary.
    """
    return ">>" not in content_text + token_text


def apply_closer_rule(accept_mask, tokens_with_gt, content_text: str):
    """Copy of `accept_mask` with every token that breaks `closer_allowed` switched off.
    `tokens_with_gt` is the precomputed (index, text) list of vocabulary tokens containing `>`."""
    banned = [
        index for index, text in tokens_with_gt
        if index < len(accept_mask) and bool(accept_mask[index])
        and not closer_allowed(content_text, text)
    ]
    if not banned:
        return accept_mask
    out = accept_mask.clone()
    out[banned] = False
    return out
