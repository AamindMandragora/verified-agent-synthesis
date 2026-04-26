from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeAlias


Token: TypeAlias = str
Prefix: TypeAlias = list[Token]
Id: TypeAlias = int
Logit: TypeAlias = float

MODULE_NAME = "VerifiedDecoderAgent"
LeftDelimiter: Token = "<<"
RightDelimiter: Token = ">>"


@dataclass(frozen=True)
class DafnySpec:
    kind: str
    reads: tuple[str, ...] = ()
    modifies: tuple[str, ...] = ()
    requires: tuple[str, ...] = ()
    ensures: tuple[str, ...] = ()
    decreases: tuple[str, ...] = ()
    axiom: bool = False
    extern: bool = False


def dafny_spec(
    *,
    kind: str,
    reads: tuple[str, ...] = (),
    modifies: tuple[str, ...] = (),
    requires: tuple[str, ...] = (),
    ensures: tuple[str, ...] = (),
    decreases: tuple[str, ...] = (),
    axiom: bool = False,
    extern: bool = False,
):
    def decorate(obj: Any) -> Any:
        obj.__dafny_spec__ = DafnySpec(
            kind=kind,
            reads=reads,
            modifies=modifies,
            requires=requires,
            ensures=ensures,
            decreases=decreases,
            axiom=axiom,
            extern=extern,
        )
        return obj

    return decorate


@dafny_spec(kind="predicate")
def Contains(s: str, sub: str) -> bool:
    return sub in s


@dafny_spec(kind="predicate")
def PrefixContains(p: Prefix, t: Token) -> bool:
    return any(tok == t for tok in p)


@dafny_spec(kind="predicate")
def DelimitedAnswerValidForParser(parser: "Parser", prefix: Prefix) -> bool:
    # Kept for backward compatibility with evaluator answer extraction.
    delim = Delimiter(LeftDelimiter, RightDelimiter)
    content = delim.GetDelimitedContent(prefix)
    return (
        PrefixContains(prefix, LeftDelimiter)
        and PrefixContains(prefix, RightDelimiter)
        and (not delim.InsideDelimitedWindow(prefix))
        and parser.IsValidPrefix(content)
        and len(content) > 0
    )


class LM:
    Tokens: Prefix
    Ids: list[Id]
    Logits: list[Logit]

    @dafny_spec(
        kind="constructor",
        ensures=("ValidTokensIdsLogits()",),
        axiom=True,
        extern=True,
    )
    def __init__(self) -> None:
        self.Tokens = [LeftDelimiter, RightDelimiter]
        self.Ids = [0, 1]
        self.Logits = [0.0, 0.0]

    @dafny_spec(
        kind="predicate",
        reads=("this", "this.Logits"),
    )
    def ValidTokensIdsLogits(self) -> bool:
        return (
            len(self.Tokens) == len(self.Ids)
            and len(self.Ids) == len(self.Logits)
            and len(self.Ids) > 0
            and self.Ids[0] == 0
            and all(i == self.Ids[i] and i in self.Ids for i in range(len(self.Ids)))
            and all(
                self.Tokens[i] != self.Tokens[j]
                for i in range(len(self.Tokens))
                for j in range(len(self.Tokens))
                if i != j
            )
            and all(token in self.Tokens for token in self.Tokens)
            and all(any(self.Tokens[i] == token for i in range(len(self.Ids))) for token in self.Tokens)
            and all(-1e9 <= logit <= 1e9 for logit in self.Logits)
        )

    @dafny_spec(
        kind="lemma",
        ensures=("ValidTokensIdsLogits()",),
        axiom=True,
    )
    def ValidTokensIdsLogitsAlways(self) -> None:
        assert self.ValidTokensIdsLogits()

    @dafny_spec(
        kind="function",
        reads=("this", "this.Logits"),
        requires=("ValidTokensIdsLogits()", "id in Ids"),
        ensures=(
            "token in Tokens",
            "Tokens[id] == token",
            "id == TokenToId(token)",
            "ValidTokensIdsLogits()",
        ),
    )
    def IdToToken(self, id: Id) -> Token:
        return self.Tokens[id]

    @dafny_spec(
        kind="function",
        reads=("this", "this.Logits"),
        requires=("ValidTokensIdsLogits()", "token in Tokens"),
        ensures=(
            "id in Ids",
            "Tokens[id] == token",
            "TokenToId(Tokens[id]) == id",
            "ValidTokensIdsLogits()",
        ),
    )
    def TokenToId(self, token: Token) -> Id:
        return self.TokenToIdRecursive(token, 0)

    @dafny_spec(
        kind="function",
        reads=("this", "this.Logits"),
        requires=(
            "ValidTokensIdsLogits()",
            "token in Tokens",
            "0 <= offset < |Tokens|",
            "(Tokens[offset] == token) || (token in Tokens[offset + 1..])",
        ),
        ensures=(
            "id in Ids",
            "0 <= TokenToIdRecursive(token, offset) < |Ids|",
            "Tokens[id] == token",
            "ValidTokensIdsLogits()",
        ),
        decreases=("|Tokens| - offset",),
    )
    def TokenToIdRecursive(self, token: Token, offset: int) -> Id:
        return offset if self.Tokens[offset] == token else self.TokenToIdRecursive(token, offset + 1)

    @dafny_spec(
        kind="function",
        reads=("this", "this.Logits"),
        requires=("ValidTokensIdsLogits()", "id in Ids"),
        ensures=("logit in Logits[0..Logits.Length]", "ValidTokensIdsLogits()"),
    )
    def IdToLogit(self, id: Id) -> Logit:
        return self.Logits[id]

    @dafny_spec(
        kind="function",
        reads=("this", "this.Logits"),
        requires=("ValidTokensIdsLogits()", "token in Tokens"),
        ensures=("ValidTokensIdsLogits()",),
    )
    def TokenToLogit(self, token: Token) -> Logit:
        return self.IdToLogit(self.TokenToId(token))

    @dafny_spec(
        kind="function",
        reads=("this", "this.Logits"),
        requires=(
            "ValidTokensIdsLogits()",
            "|tokens| > 0",
            "forall token: Token :: token in tokens ==> token in Tokens",
        ),
        ensures=("ValidTokensIdsLogits()",),
    )
    def TokensToLogits(self, tokens: Prefix) -> list[Logit]:
        return (
            [self.TokenToLogit(tokens[0])]
            if len(tokens) == 1
            else [self.TokenToLogit(tokens[0])] + self.TokensToLogits(tokens[1:])
        )

    @dafny_spec(
        kind="function",
        reads=("this", "this.Logits"),
        requires=(
            "ValidTokensIdsLogits()",
            "|ids| > 0",
            "forall id: Id :: id in ids ==> id in Ids",
        ),
        ensures=("ValidTokensIdsLogits()",),
    )
    def IdsToLogits(self, ids: list[Id]) -> list[Logit]:
        return [self.IdToLogit(ids[0])] if len(ids) == 1 else [self.IdToLogit(ids[0])] + self.IdsToLogits(ids[1:])

    # ── Hard Masking ──────────────────────────────────────────────────────

    @dafny_spec(
        kind="method",
        modifies=("this.Logits",),
        requires=("ValidTokensIdsLogits()", "token in Tokens"),
        ensures=(
            "ValidTokensIdsLogits()",
            "IsMasked(token)",
            "forall t: Token :: t in Tokens && t != token ==> Logits[TokenToId(t)] == old(Logits[TokenToId(t)])",
        ),
    )
    def MaskToken(self, token: Token) -> None:
        token_id = self.TokenToId(token)
        self.Logits[token_id] = -1e9

    @dafny_spec(
        kind="method",
        modifies=("this.Logits",),
        requires=(
            "ValidTokensIdsLogits()",
            "|tokens| > 0",
            "forall token :: token in tokens ==> token in Tokens",
        ),
        ensures=(
            "ValidTokensIdsLogits()",
            "forall t :: t in tokens ==> IsMasked(t)",
            "forall t :: t in Tokens && !(t in tokens) ==> Logits[TokenToId(t)] == old(Logits[TokenToId(t)])",
        ),
    )
    def MaskTokens(self, tokens: Prefix) -> None:
        n = len(tokens)
        i = 0
        # invariant 0 <= i <= N
        # invariant ValidTokensIdsLogits()
        # invariant forall j :: 0 <= j < i ==> IsMasked(tokens[j])
        # invariant forall t :: t in Tokens && !(t in tokens[..i]) ==> Logits[TokenToId(t)] == old(Logits[TokenToId(t)])
        while i < n:
            self.MaskToken(tokens[i])
            i += 1

    @dafny_spec(
        kind="method",
        modifies=("this.Logits",),
        requires=(
            "ValidTokensIdsLogits()",
            "|tokens| > 0",
            "forall token :: token in tokens ==> token in Tokens",
        ),
        ensures=(
            "ValidTokensIdsLogits()",
            "forall t :: t in Tokens && !(t in tokens) ==> IsMasked(t)",
            "forall t :: t in tokens ==> Logits[TokenToId(t)] == old(Logits[TokenToId(t)])",
        ),
    )
    def MaskTokensExcept(self, tokens: Prefix) -> None:
        to_mask: Prefix = []
        n = len(self.Tokens)
        i = 0
        # invariant 0 <= i <= N
        # invariant ValidTokensIdsLogits()
        # invariant forall j :: 0 <= j < i && !(Tokens[j] in tokens) ==> Tokens[j] in toMask
        # invariant forall j :: 0 <= j < i && Tokens[j] in tokens ==> !(Tokens[j] in toMask)
        # invariant forall t: Token :: t in toMask ==> t !in tokens && t in Tokens
        while i < n:
            if self.Tokens[i] not in tokens:
                to_mask = to_mask + [self.Tokens[i]]
            i += 1
        if len(to_mask) > 0:
            self.MaskTokens(to_mask)

    @dafny_spec(
        kind="predicate",
        reads=("this", "this.Logits"),
        requires=("ValidTokensIdsLogits()", "token in Tokens"),
        ensures=("ValidTokensIdsLogits()",),
    )
    def IsMasked(self, token: Token) -> bool:
        return self.Logits[self.TokenToId(token)] == -1e9

    @dafny_spec(
        kind="predicate",
        reads=("this", "this.Logits"),
        requires=("ValidTokensIdsLogits()",),
        ensures=("ValidTokensIdsLogits()",),
    )
    def HasUnmaskedToken(self) -> bool:
        return any(token in self.Tokens and not self.IsMasked(token) for token in self.Tokens)

    # ── Soft Logit Shaping ────────────────────────────────────────────────

    @dafny_spec(
        kind="method",
        modifies=("this.Logits",),
        requires=("ValidTokensIdsLogits()", "token in Tokens"),
        ensures=(
            "ValidTokensIdsLogits()",
            "-1e9 <= Logits[TokenToId(token)] <= 1e9",
            "Logits[TokenToId(token)] == if old(Logits[TokenToId(token)]) + delta > 1e9 then 1e9 else if old(Logits[TokenToId(token)]) + delta < -1e9 then -1e9 else old(Logits[TokenToId(token)]) + delta",
            "forall t: Token :: t in Tokens && t != token ==> Logits[TokenToId(t)] == old(Logits[TokenToId(t)])",
        ),
    )
    def BiasToken(self, token: Token, delta: Logit) -> None:
        token_id = self.TokenToId(token)
        raw = self.Logits[token_id] + delta
        if raw > 1e9:
            raw = 1e9
        if raw < -1e9:
            raw = -1e9
        self.Logits[token_id] = raw

    @dafny_spec(
        kind="method",
        modifies=("this.Logits",),
        requires=(
            "ValidTokensIdsLogits()",
            "|tokens| > 0",
            "forall token :: token in tokens ==> token in Tokens",
        ),
        ensures=(
            "ValidTokensIdsLogits()",
            "forall t :: t in Tokens && !(t in tokens) ==> Logits[TokenToId(t)] == old(Logits[TokenToId(t)])",
        ),
    )
    def BiasTokens(self, tokens: Prefix, delta: Logit) -> None:
        n = len(tokens)
        i = 0
        # invariant 0 <= i <= n
        # invariant ValidTokensIdsLogits()
        # invariant forall t :: t in Tokens && !(t in tokens[..i]) ==> Logits[TokenToId(t)] == old(Logits[TokenToId(t)])
        while i < n:
            self.BiasToken(tokens[i], delta)
            i += 1

    @dafny_spec(
        kind="method",
        modifies=("this.Logits",),
        requires=("ValidTokensIdsLogits()", "token in Tokens", "factor != 0.0"),
        ensures=(
            "ValidTokensIdsLogits()",
            "-1e9 <= Logits[TokenToId(token)] <= 1e9",
            "forall t: Token :: t in Tokens && t != token ==> Logits[TokenToId(t)] == old(Logits[TokenToId(t)])",
        ),
    )
    def ScaleToken(self, token: Token, factor: Logit) -> None:
        token_id = self.TokenToId(token)
        raw = self.Logits[token_id] * factor
        if raw > 1e9:
            raw = 1e9
        if raw < -1e9:
            raw = -1e9
        self.Logits[token_id] = raw

    @dafny_spec(
        kind="method",
        modifies=("this.Logits",),
        requires=(
            "ValidTokensIdsLogits()",
            "|tokens| > 0",
            "forall token :: token in tokens ==> token in Tokens",
            "factor != 0.0",
        ),
        ensures=(
            "ValidTokensIdsLogits()",
            "forall t :: t in Tokens && !(t in tokens) ==> Logits[TokenToId(t)] == old(Logits[TokenToId(t)])",
        ),
    )
    def ScaleTokens(self, tokens: Prefix, factor: Logit) -> None:
        n = len(tokens)
        i = 0
        # invariant 0 <= i <= n
        # invariant ValidTokensIdsLogits()
        # invariant forall t :: t in Tokens && !(t in tokens[..i]) ==> Logits[TokenToId(t)] == old(Logits[TokenToId(t)])
        while i < n:
            self.ScaleToken(tokens[i], factor)
            i += 1

    @dafny_spec(
        kind="method",
        modifies=("this.Logits",),
        requires=("ValidTokensIdsLogits()", "-1e9 <= low", "low <= high", "high <= 1e9"),
        ensures=(
            "ValidTokensIdsLogits()",
            "forall id :: 0 <= id < Logits.Length ==> low <= Logits[id] <= high",
        ),
    )
    def ClampLogits(self, low: Logit, high: Logit) -> None:
        n = len(self.Logits)
        i = 0
        # invariant 0 <= i <= n
        # invariant ValidTokensIdsLogits()
        # invariant forall j :: 0 <= j < i ==> low <= Logits[j] <= high
        # invariant forall j :: i <= j < n ==> Logits[j] == old(Logits[j])
        while i < n:
            if self.Logits[i] > high:
                self.Logits[i] = high
            if self.Logits[i] < low:
                self.Logits[i] = low
            i += 1

    # ── Filtering ─────────────────────────────────────────────────────────

    @dafny_spec(
        kind="method",
        modifies=("this.Logits",),
        requires=("ValidTokensIdsLogits()", "1 <= k <= |Tokens|"),
        ensures=(
            "ValidTokensIdsLogits()",
            "HasUnmaskedToken()",
            "forall t :: t in Tokens && !IsMasked(t) ==> !old(IsMasked(t))",
        ),
        axiom=True,
        extern=True,
    )
    def TopKFilter(self, k: int) -> None:
        n = len(self.Tokens)
        sorted_ids = sorted(range(n), key=lambda idx: self.Logits[idx], reverse=True)
        keep = set(sorted_ids[:k])
        i = 0
        while i < n:
            if i not in keep:
                self.Logits[i] = -1e9
            i += 1

    # ── Generation ────────────────────────────────────────────────────────

    @dafny_spec(
        kind="method",
        modifies=("this.Logits",),
        requires=("ValidTokensIdsLogits()",),
        ensures=("ValidTokensIdsLogits()",),
        axiom=True,
        extern=True,
    )
    def GenerateLogits(self, input: Prefix) -> None:
        if not self.ValidTokensIdsLogits():
            raise ValueError("LM invariant violated before GenerateLogits")

    @dafny_spec(
        kind="method",
        requires=("ValidTokensIdsLogits()",),
        ensures=("token in Tokens", "!IsMasked(token)", "ValidTokensIdsLogits()"),
        axiom=True,
        extern=True,
    )
    def ChooseNextToken(self) -> Token:
        best_token: Token | None = None
        best_logit: Logit | None = None
        for token, logit in zip(self.Tokens, self.Logits):
            if logit == -1e9:
                continue
            if best_logit is None or logit > best_logit:
                best_token = token
                best_logit = logit
        if best_token is None:
            raise ValueError("No unmasked token is available")
        return best_token


class Parser:
    @dafny_spec(
        kind="predicate",
        ensures=("forall k :: 0 <= k < |prefix| ==> IsValidPrefix(prefix[..k])",),
        axiom=True,
        extern=True,
    )
    def IsValidPrefix(self, prefix: Prefix) -> bool:
        raise NotImplementedError

    @dafny_spec(
        kind="lemma",
        ensures=("IsValidPrefix([])",),
        axiom=True,
    )
    def EmptyPrefixIsValid(self) -> None:
        assert self.IsValidPrefix([])

    @dafny_spec(
        kind="predicate",
        ensures=("IsValidPrefix(prefix)",),
        axiom=True,
        extern=True,
    )
    def IsCompletePrefix(self, prefix: Prefix) -> bool:
        raise NotImplementedError

    @dafny_spec(kind="predicate")
    def IsDeadPrefix(self, prefix: Prefix) -> bool:
        return (not self.IsCompletePrefix(prefix)) and len(self.ValidNextTokens(prefix)) == 0

    @dafny_spec(
        kind="predicate",
        requires=("IsValidPrefix(prefix)",),
    )
    def ValidNextToken(self, prefix: Prefix, token: Token) -> bool:
        return token in self.ValidNextTokens(prefix)

    @dafny_spec(
        kind="function",
        requires=("IsValidPrefix(prefix)",),
        ensures=(
            "forall t :: t in ValidNextTokens(prefix) ==> IsValidPrefix(prefix + [t])",
            "(IsCompletePrefix(prefix) || |ValidNextTokens(prefix)| > 0)",
        ),
        axiom=True,
        extern=True,
    )
    def ValidNextTokens(self, prefix: Prefix) -> Prefix:
        raise NotImplementedError

    @dafny_spec(
        kind="function",
        requires=("IsValidPrefix(prefix)",),
        ensures=(
            "result >= 0",
            "result == |ValidNextTokens(prefix)|",
            "result == 0 ==> (IsCompletePrefix(prefix) || IsDeadPrefix(prefix))",
        ),
    )
    def ValidContinuationCount(self, prefix: Prefix) -> int:
        return len(self.ValidNextTokens(prefix))

    @dafny_spec(
        kind="function",
        requires=("IsValidPrefix(prefix)",),
        ensures=(
            "result >= 0",
            "IsCompletePrefix(prefix) ==> result == 0",
            "!IsCompletePrefix(prefix) ==> result >= 1",
        ),
        axiom=True,
        extern=True,
    )
    def ParserDistanceToComplete(self, prefix: Prefix) -> int:
        return 0 if self.IsCompletePrefix(prefix) else 1


class Delimiter:
    Left: Token
    Right: Token

    @dafny_spec(
        kind="constructor",
        requires=("left != right",),
        ensures=("this.Left == left && this.Right == right", "this.Left != this.Right"),
    )
    def __init__(self, left: Token, right: Token) -> None:
        if left == right:
            raise ValueError("Delimiter endpoints must be distinct")
        self.Left = left
        self.Right = right

    @dafny_spec(
        kind="function",
        ensures=(
            "result <= |prefix|",
            "result < |prefix| ==> prefix[result] == this.Left",
            "result == |prefix| ==> forall i :: 0 <= i < |prefix| ==> prefix[i] != this.Left",
            "result < |prefix| ==> forall i :: result < i < |prefix| ==> prefix[i] != this.Left",
        ),
        decreases=("|prefix|",),
    )
    def LastLeftDelimiterIndex(self, prefix: Prefix) -> int:
        return (
            0
            if len(prefix) == 0
            else (len(prefix) - 1 if prefix[-1] == self.Left else self.LastLeftDelimiterIndex(prefix[:-1]))
        )

    @dafny_spec(
        kind="function",
        ensures=(
            "result <= |content|",
            "result < |content| ==> content[result] == this.Right",
            "forall i :: 0 <= i < result ==> content[i] != this.Right",
        ),
        decreases=("|content|",),
    )
    def FirstRightDelimiterIndex(self, content: Prefix) -> int:
        return 0 if len(content) == 0 or content[0] == self.Right else 1 + self.FirstRightDelimiterIndex(content[1:])

    @dafny_spec(
        kind="lemma",
        requires=("FirstRightDelimiterIndex(content) == |content|",),
        ensures=("!PrefixContains(content, this.Right)",),
    )
    def NoFirstRightDelimiterIndexMeansNoRight(self, content: Prefix) -> None:
        assert self.FirstRightDelimiterIndex(content) == len(content)
        assert not PrefixContains(content, self.Right)

    @dafny_spec(
        kind="function",
        ensures=(
            "|GetDelimitedContent(prefix)| <= |prefix|",
            "forall t: Token :: t in GetDelimitedContent(prefix) ==> t in prefix",
        ),
    )
    def GetDelimitedContent(self, prefix: Prefix) -> Prefix:
        start = self.LastLeftDelimiterIndex(prefix) + 1
        after_left = [] if start > len(prefix) else prefix[start:]
        end_idx = self.FirstRightDelimiterIndex(after_left)
        return after_left[:end_idx]

    @dafny_spec(kind="predicate")
    def InsideDelimitedWindow(self, prefix: Prefix) -> bool:
        start = self.LastLeftDelimiterIndex(prefix) + 1
        return start <= len(prefix) and self.FirstRightDelimiterIndex(prefix[start:]) == len(prefix[start:])

    @dafny_spec(
        kind="lemma",
        requires=("InsideDelimitedWindow(prefix)",),
        ensures=("!PrefixContains(GetDelimitedContent(prefix), this.Right)",),
    )
    def InsideDelimitedWindowNoRight(self, prefix: Prefix) -> None:
        start = self.LastLeftDelimiterIndex(prefix) + 1
        after_left = prefix[start:]
        self.NoFirstRightDelimiterIndexMeansNoRight(after_left)

    @dafny_spec(
        kind="lemma",
        requires=("InsideDelimitedWindow(prefix)", "next != Right", "next != Left"),
        ensures=(
            "GetDelimitedContent(prefix + [next]) == GetDelimitedContent(prefix) + [next]",
            "next != Right ==> InsideDelimitedWindow(prefix + [next])",
        ),
        axiom=True,
    )
    def GetDelimitedContentAppend(self, prefix: Prefix, next: Token) -> None:
        assert self.InsideDelimitedWindow(prefix)
        assert next != self.Right
        assert next != self.Left
        assert self.GetDelimitedContent(prefix + [next]) == self.GetDelimitedContent(prefix) + [next]
        assert self.InsideDelimitedWindow(prefix + [next])

    @dafny_spec(
        kind="lemma",
        ensures=(
            "InsideDelimitedWindow(prefix + [this.Left])",
            "GetDelimitedContent(prefix + [this.Left]) == []",
        ),
    )
    def AppendLeftEntersWindow(self, prefix: Prefix) -> None:
        assert self.InsideDelimitedWindow(prefix + [self.Left])
        assert self.GetDelimitedContent(prefix + [self.Left]) == []

    @dafny_spec(
        kind="lemma",
        requires=("FirstRightDelimiterIndex(content) == |content|",),
        ensures=("FirstRightDelimiterIndex(content + [this.Right]) == |content|",),
    )
    def FirstRightDelimiterAppendRight(self, content: Prefix) -> None:
        assert self.FirstRightDelimiterIndex(content) == len(content)
        assert self.FirstRightDelimiterIndex(content + [self.Right]) == len(content)

    @dafny_spec(
        kind="lemma",
        requires=("tok != this.Left",),
        ensures=(
            "var oldIdx := LastLeftDelimiterIndex(prefix); var newIdx := LastLeftDelimiterIndex(prefix + [tok]); if oldIdx < |prefix| then newIdx == oldIdx else newIdx == |prefix + [tok]|",
        ),
    )
    def LastLeftDelimiterAppendNonLeft(self, prefix: Prefix, tok: Token) -> None:
        assert tok != self.Left
        old_idx = self.LastLeftDelimiterIndex(prefix)
        new_idx = self.LastLeftDelimiterIndex(prefix + [tok])
        if old_idx < len(prefix):
            assert new_idx == old_idx
        else:
            assert new_idx == len(prefix + [tok])

    @dafny_spec(
        kind="lemma",
        requires=("InsideDelimitedWindow(prefix)", "this.Left != this.Right"),
        ensures=("!InsideDelimitedWindow(prefix + [this.Right])",),
    )
    def AppendRightExitsWindow(self, prefix: Prefix) -> None:
        assert self.InsideDelimitedWindow(prefix)
        assert self.Left != self.Right
        assert not self.InsideDelimitedWindow(prefix + [self.Right])


class CSDHelpers:
    lm: LM
    parser: Parser

    @dafny_spec(
        kind="constructor",
        requires=("lm.ValidTokensIdsLogits()",),
        ensures=(
            "this.lm == lm && this.parser == parser",
            "lm.ValidTokensIdsLogits()",
        ),
    )
    def __init__(self, lm: LM, parser: Parser) -> None:
        self.lm = lm
        self.parser = parser

    # ── Core Lemmas ───────────────────────────────────────────────────────

    @dafny_spec(
        kind="lemma",
        requires=("lm.ValidTokensIdsLogits()", "parser.IsValidPrefix(content)"),
        ensures=(
            "lm.ValidTokensIdsLogits()",
            "forall t: Token :: t in parser.ValidNextTokens(content) ==> t in lm.Tokens",
        ),
        axiom=True,
    )
    def AllValidNextTokensInLM(self, content: Prefix) -> None:
        assert self.lm.ValidTokensIdsLogits()
        assert self.parser.IsValidPrefix(content)
        for token in self.parser.ValidNextTokens(content):
            assert token in self.lm.Tokens

    @dafny_spec(
        kind="lemma",
        requires=(
            "lm.ValidTokensIdsLogits()",
            "parser.IsValidPrefix(content)",
            "!parser.IsCompletePrefix(content)",
            "forall t: Token :: t in parser.ValidNextTokens(content) ==> t in lm.Tokens",
            "parser.IsValidPrefix(content + [next])",
        ),
        ensures=("forall t: Token :: t in parser.ValidNextTokens(content + [next]) ==> t in lm.Tokens",),
        axiom=True,
    )
    def ValidNextTokensInLMAfterStep(self, content: Prefix, next: Token) -> None:
        assert self.lm.ValidTokensIdsLogits()
        assert self.parser.IsValidPrefix(content + [next])

    # ── Suffix-Based Grammar Alignment ────────────────────────────────────

    @dafny_spec(
        kind="function",
        reads=("this", "this.parser"),
        requires=("parser.IsValidPrefix([])",),
        ensures=(
            "parser.IsValidPrefix(result)",
            "|result| <= |prefix|",
            "|prefix| > 0 && parser.IsValidPrefix(prefix) ==> result == prefix",
            "|prefix| == 0 ==> result == []",
            "forall i :: 0 <= i < |result| ==> result[i] == prefix[|prefix| - |result| + i]",
        ),
        decreases=("|prefix|",),
        axiom=True,
    )
    def LongestValidSuffix(self, prefix: Prefix) -> Prefix:
        return (
            []
            if len(prefix) == 0
            else (prefix if self.parser.IsValidPrefix(prefix) else self.LongestValidSuffix(prefix[1:]))
        )

    @dafny_spec(
        kind="lemma",
        requires=(
            "parser.IsValidPrefix([])",
            "parser.IsValidPrefix(LongestValidSuffix(prefix))",
            "parser.ValidNextToken(LongestValidSuffix(prefix), next)",
        ),
        ensures=(
            "parser.IsValidPrefix(LongestValidSuffix(prefix) + [next])",
            "|LongestValidSuffix(prefix + [next])| >= |LongestValidSuffix(prefix)| + 1",
        ),
        axiom=True,
    )
    def LongestValidSuffixAppend(self, prefix: Prefix, next: Token) -> None:
        assert self.parser.IsValidPrefix(self.LongestValidSuffix(prefix + [next]))

    @dafny_spec(
        kind="lemma",
        requires=("parser.IsValidPrefix([])",),
        ensures=("parser.IsValidPrefix(LongestValidSuffix(prefix))",),
    )
    def LongestValidSuffixIsValid(self, prefix: Prefix) -> None:
        assert self.parser.IsValidPrefix(self.LongestValidSuffix(prefix))

    @dafny_spec(
        kind="lemma",
        requires=(
            "parser.IsValidPrefix([])",
            "parser.IsValidPrefix(LongestValidSuffix(prefix))",
        ),
        ensures=(
            "parser.IsCompletePrefix(LongestValidSuffix(prefix)) || |parser.ValidNextTokens(LongestValidSuffix(prefix))| > 0",
        ),
    )
    def LongestValidSuffixNotDead(self, prefix: Prefix) -> None:
        # Follows from ValidNextTokens's ensures on any valid prefix
        suffix = self.LongestValidSuffix(prefix)
        assert self.parser.IsCompletePrefix(suffix) or len(self.parser.ValidNextTokens(suffix)) > 0

    @dafny_spec(
        kind="predicate",
        reads=("this", "this.parser"),
        requires=("parser.IsValidPrefix([])",),
    )
    def CanConstrain(self, prefix: Prefix) -> bool:
        return not self.parser.IsCompletePrefix(self.LongestValidSuffix(prefix))

    # ── Step Functions ────────────────────────────────────────────────────

    @dafny_spec(
        kind="method",
        modifies=("this.lm.Logits",),
        requires=(
            "this.lm.ValidTokensIdsLogits()",
            "stepsLeft >= 1",
        ),
        ensures=(
            "this.lm.ValidTokensIdsLogits()",
            "stepsLeft' == stepsLeft - 1",
            "next in lm.Tokens",
            "!lm.IsMasked(next)",
        ),
    )
    def UnconstrainedStep(self, prompt: Prefix, generated: Prefix, stepsLeft: int) -> tuple[Token, int]:
        self.lm.ValidTokensIdsLogitsAlways()
        self.lm.GenerateLogits(prompt + generated)
        next_token = self.lm.ChooseNextToken()
        return next_token, stepsLeft - 1

    @dafny_spec(
        kind="method",
        modifies=("this.lm.Logits",),
        requires=(
            "this.lm.ValidTokensIdsLogits()",
            "parser.IsValidPrefix([])",
            "!parser.IsCompletePrefix(LongestValidSuffix(generated))",
            "stepsLeft >= 1",
        ),
        ensures=(
            "this.lm.ValidTokensIdsLogits()",
            "stepsLeft' == stepsLeft - 1",
            "stepsLeft' >= 0",
            "next in lm.Tokens",
            "!lm.IsMasked(next)",
            "parser.ValidNextToken(LongestValidSuffix(generated), next)",
            "parser.IsValidPrefix(LongestValidSuffix(generated) + [next])",
            "|LongestValidSuffix(generated + [next])| >= |LongestValidSuffix(generated)| + 1",
        ),
    )
    def ConstrainedStep(self, prompt: Prefix, generated: Prefix, stepsLeft: int) -> tuple[Token, int]:
        self.LongestValidSuffixIsValid(generated)
        suffix = self.LongestValidSuffix(generated)
        self.AllValidNextTokensInLM(suffix)
        self.lm.GenerateLogits(prompt + generated)
        self.lm.MaskTokensExcept(self.parser.ValidNextTokens(suffix))
        next_token = self.lm.ChooseNextToken()
        self.LongestValidSuffixAppend(generated, next_token)
        return next_token, stepsLeft - 1

    @dafny_spec(
        kind="method",
        modifies=("this.lm.Logits",),
        requires=(
            "this.lm.ValidTokensIdsLogits()",
            "parser.IsValidPrefix([])",
            "!parser.IsCompletePrefix(LongestValidSuffix(generated))",
            "stepsLeft >= 1",
            "penalty > 0.0",
        ),
        ensures=(
            "this.lm.ValidTokensIdsLogits()",
            "stepsLeft' == stepsLeft - 1",
            "stepsLeft' >= 0",
            "next in lm.Tokens",
            "!lm.IsMasked(next)",
        ),
    )
    def SoftConstrainedStep(self, prompt: Prefix, generated: Prefix, penalty: Logit, stepsLeft: int) -> tuple[Token, int]:
        self.LongestValidSuffixIsValid(generated)
        suffix = self.LongestValidSuffix(generated)
        self.AllValidNextTokensInLM(suffix)
        valid_tokens = self.parser.ValidNextTokens(suffix)
        invalid_tokens: Prefix = []
        n = len(self.lm.Tokens)
        i = 0
        # invariant 0 <= i <= n
        # invariant lm.ValidTokensIdsLogits()
        # invariant forall j :: 0 <= j < i && !(lm.Tokens[j] in valid_tokens) ==> lm.Tokens[j] in invalid_tokens
        # invariant forall t :: t in invalid_tokens ==> t in lm.Tokens && !(t in valid_tokens)
        while i < n:
            if self.lm.Tokens[i] not in valid_tokens:
                invalid_tokens = invalid_tokens + [self.lm.Tokens[i]]
            i += 1
        self.lm.GenerateLogits(prompt + generated)
        if len(invalid_tokens) > 0:
            self.lm.BiasTokens(invalid_tokens, -penalty)
        next_token = self.lm.ChooseNextToken()
        return next_token, stepsLeft - 1

    @dafny_spec(
        kind="method",
        modifies=("this.lm.Logits",),
        requires=(
            "this.lm.ValidTokensIdsLogits()",
            "parser.IsValidPrefix([])",
            "!parser.IsCompletePrefix(LongestValidSuffix(generated))",
            "stepsLeft >= 1",
            "1 <= k <= |lm.Tokens|",
        ),
        ensures=(
            "this.lm.ValidTokensIdsLogits()",
            "stepsLeft' == stepsLeft - 1",
            "stepsLeft' >= 0",
            "next in lm.Tokens",
            "!lm.IsMasked(next)",
            "parser.ValidNextToken(LongestValidSuffix(generated), next)",
            "parser.IsValidPrefix(LongestValidSuffix(generated) + [next])",
        ),
    )
    def TopKConstrainedStep(self, prompt: Prefix, generated: Prefix, k: int, stepsLeft: int) -> tuple[Token, int]:
        self.LongestValidSuffixIsValid(generated)
        self.lm.GenerateLogits(prompt + generated)
        self.lm.TopKFilter(k)
        suffix = self.LongestValidSuffix(generated)
        self.AllValidNextTokensInLM(suffix)
        self.lm.MaskTokensExcept(self.parser.ValidNextTokens(suffix))
        next_token = self.lm.ChooseNextToken()
        self.LongestValidSuffixAppend(generated, next_token)
        return next_token, stepsLeft - 1

    @dafny_spec(
        kind="method",
        requires=(
            "token in lm.Tokens",
            "stepsLeft >= 1",
        ),
        ensures=(
            "this.lm.ValidTokensIdsLogits()",
            "stepsLeft' == stepsLeft - 1",
            "stepsLeft' >= 0",
            "next == token",
            "next in lm.Tokens",
        ),
    )
    def ForcedTokenStep(self, prompt: Prefix, generated: Prefix, token: Token, stepsLeft: int) -> tuple[Token, int]:
        self.lm.ValidTokensIdsLogitsAlways()
        return token, stepsLeft - 1

    @dafny_spec(
        kind="method",
        modifies=("this.lm.Logits",),
        requires=(
            "this.lm.ValidTokensIdsLogits()",
            "parser.IsValidPrefix([])",
            "stepsLeft >= 1",
            "completionThreshold >= 1",
        ),
        ensures=(
            "this.lm.ValidTokensIdsLogits()",
            "stepsLeft' == stepsLeft - 1",
            "stepsLeft' >= 0",
            "next in lm.Tokens",
            "!lm.IsMasked(next)",
            "stepsLeft <= completionThreshold && !parser.IsCompletePrefix(LongestValidSuffix(generated)) ==> parser.ValidNextToken(LongestValidSuffix(generated), next)",
        ),
    )
    def BudgetAwareStep(self, prompt: Prefix, generated: Prefix, stepsLeft: int, completionThreshold: int) -> tuple[Token, int]:
        suffix = self.LongestValidSuffix(generated)
        next_token = self.lm.Tokens[0]
        steps_left_prime = stepsLeft - 1
        if stepsLeft <= completionThreshold and not self.parser.IsCompletePrefix(suffix):
            next_token, steps_left_prime = self.ConstrainedStep(prompt, generated, stepsLeft)
        else:
            next_token, steps_left_prime = self.UnconstrainedStep(prompt, generated, stepsLeft)
        return next_token, steps_left_prime

    # ── Ergonomic State-Transforming Wrappers ─────────────────────────────

    @dafny_spec(
        kind="method",
        modifies=("this.lm.Logits",),
        requires=(
            "this.lm.ValidTokensIdsLogits()",
            "stepsLeft >= 1",
        ),
        ensures=(
            "this.lm.ValidTokensIdsLogits()",
            "remainingSteps == stepsLeft - 1",
            "remainingSteps >= 0",
            "|updated| == |prefix| + 1",
            "updated[|prefix|] in lm.Tokens",
            "!lm.IsMasked(updated[|prefix|])",
        ),
    )
    def AppendUnconstrainedStep(self, prompt: Prefix, prefix: Prefix, stepsLeft: int) -> tuple[Prefix, int]:
        next_token, remaining_steps = self.UnconstrainedStep(prompt, prefix, stepsLeft)
        return prefix + [next_token], remaining_steps

    @dafny_spec(
        kind="method",
        modifies=("this.lm.Logits",),
        requires=(
            "this.lm.ValidTokensIdsLogits()",
            "parser.IsValidPrefix([])",
            "CanConstrain(prefix)",
            "stepsLeft >= 1",
        ),
        ensures=(
            "this.lm.ValidTokensIdsLogits()",
            "remainingSteps == stepsLeft - 1",
            "remainingSteps >= 0",
            "|updated| == |prefix| + 1",
            "updated[|prefix|] in lm.Tokens",
            "!lm.IsMasked(updated[|prefix|])",
            "parser.ValidNextToken(LongestValidSuffix(prefix), updated[|prefix|])",
            "parser.IsValidPrefix(LongestValidSuffix(prefix) + [updated[|prefix|]])",
        ),
    )
    def AppendConstrainedStep(self, prompt: Prefix, prefix: Prefix, stepsLeft: int) -> tuple[Prefix, int]:
        next_token, remaining_steps = self.ConstrainedStep(prompt, prefix, stepsLeft)
        return prefix + [next_token], remaining_steps

    @dafny_spec(
        kind="method",
        modifies=("this.lm.Logits",),
        requires=(
            "this.lm.ValidTokensIdsLogits()",
            "parser.IsValidPrefix([])",
            "CanConstrain(prefix)",
            "stepsLeft >= 1",
            "penalty > 0.0",
        ),
        ensures=(
            "this.lm.ValidTokensIdsLogits()",
            "remainingSteps == stepsLeft - 1",
            "remainingSteps >= 0",
            "|updated| == |prefix| + 1",
            "updated[|prefix|] in lm.Tokens",
            "!lm.IsMasked(updated[|prefix|])",
        ),
    )
    def AppendSoftConstrainedStep(
        self, prompt: Prefix, prefix: Prefix, penalty: Logit, stepsLeft: int
    ) -> tuple[Prefix, int]:
        next_token, remaining_steps = self.SoftConstrainedStep(prompt, prefix, penalty, stepsLeft)
        return prefix + [next_token], remaining_steps

    @dafny_spec(
        kind="method",
        modifies=("this.lm.Logits",),
        requires=(
            "this.lm.ValidTokensIdsLogits()",
            "parser.IsValidPrefix([])",
            "CanConstrain(prefix)",
            "stepsLeft >= 1",
            "1 <= k <= |lm.Tokens|",
        ),
        ensures=(
            "this.lm.ValidTokensIdsLogits()",
            "remainingSteps == stepsLeft - 1",
            "remainingSteps >= 0",
            "|updated| == |prefix| + 1",
            "updated[|prefix|] in lm.Tokens",
            "!lm.IsMasked(updated[|prefix|])",
            "parser.ValidNextToken(LongestValidSuffix(prefix), updated[|prefix|])",
            "parser.IsValidPrefix(LongestValidSuffix(prefix) + [updated[|prefix|]])",
        ),
    )
    def AppendTopKConstrainedStep(self, prompt: Prefix, prefix: Prefix, k: int, stepsLeft: int) -> tuple[Prefix, int]:
        next_token, remaining_steps = self.TopKConstrainedStep(prompt, prefix, k, stepsLeft)
        return prefix + [next_token], remaining_steps

    @dafny_spec(
        kind="method",
        modifies=("this.lm.Logits",),
        requires=(
            "this.lm.ValidTokensIdsLogits()",
            "parser.IsValidPrefix([])",
            "stepsLeft >= 1",
            "completionThreshold >= 1",
        ),
        ensures=(
            "this.lm.ValidTokensIdsLogits()",
            "remainingSteps == stepsLeft - 1",
            "remainingSteps >= 0",
            "|updated| == |prefix| + 1",
            "updated[|prefix|] in lm.Tokens",
            "!lm.IsMasked(updated[|prefix|])",
            "stepsLeft <= completionThreshold && CanConstrain(prefix) ==> parser.ValidNextToken(LongestValidSuffix(prefix), updated[|prefix|])",
        ),
    )
    def AppendBudgetAwareStep(
        self, prompt: Prefix, prefix: Prefix, stepsLeft: int, completionThreshold: int
    ) -> tuple[Prefix, int]:
        next_token, remaining_steps = self.BudgetAwareStep(prompt, prefix, stepsLeft, completionThreshold)
        return prefix + [next_token], remaining_steps

    @dafny_spec(
        kind="method",
        requires=(
            "token in lm.Tokens",
            "stepsLeft >= 1",
        ),
        ensures=(
            "this.lm.ValidTokensIdsLogits()",
            "remainingSteps == stepsLeft - 1",
            "remainingSteps >= 0",
            "updated == prefix + [token]",
        ),
    )
    def AppendForcedToken(self, prefix: Prefix, token: Token, stepsLeft: int) -> tuple[Prefix, int]:
        next_token, remaining_steps = self.ForcedTokenStep([], prefix, token, stepsLeft)
        return prefix + [next_token], remaining_steps

    @dafny_spec(
        kind="method",
        requires=(
            "LeftDelimiter in lm.Tokens",
            "stepsLeft >= 1",
        ),
        ensures=(
            "this.lm.ValidTokensIdsLogits()",
            "remainingSteps == stepsLeft - 1",
            "remainingSteps >= 0",
            "updated == prefix + [LeftDelimiter]",
        ),
    )
    def AppendLeftDelimiter(self, prefix: Prefix, stepsLeft: int) -> tuple[Prefix, int]:
        updated, remainingSteps = self.AppendForcedToken(prefix, LeftDelimiter, stepsLeft)
        return updated, remainingSteps

    @dafny_spec(
        kind="method",
        requires=(
            "RightDelimiter in lm.Tokens",
            "stepsLeft >= 1",
        ),
        ensures=(
            "this.lm.ValidTokensIdsLogits()",
            "remainingSteps == stepsLeft - 1",
            "remainingSteps >= 0",
            "updated == prefix + [RightDelimiter]",
        ),
    )
    def AppendRightDelimiter(self, prefix: Prefix, stepsLeft: int) -> tuple[Prefix, int]:
        updated, remainingSteps = self.AppendForcedToken(prefix, RightDelimiter, stepsLeft)
        return updated, remainingSteps

    # ── Repair and Salvage ────────────────────────────────────────────────

    @dafny_spec(
        kind="method",
        requires=("parser.IsValidPrefix([])",),
        ensures=(
            "parser.IsValidPrefix(repaired)",
            "|repaired| <= |generated|",
            "!parser.IsDeadPrefix(repaired)",
            "|repaired| == 0 ==> parser.IsValidPrefix([])",
            "forall i :: 0 <= i < |repaired| ==> repaired[i] == generated[i]",
        ),
    )
    def RollbackToValidPrefix(self, generated: Prefix) -> Prefix:
        repaired = list(generated)
        # invariant |repaired| <= |generated|
        # invariant parser.IsValidPrefix(repaired) || |repaired| > 0
        # invariant forall i :: 0 <= i < |repaired| ==> repaired[i] == generated[i]
        while len(repaired) > 0 and (
            (not self.parser.IsValidPrefix(repaired)) or self.parser.IsDeadPrefix(repaired)
        ):
            repaired = repaired[:-1]
        return repaired

    @dafny_spec(
        kind="method",
        requires=("parser.IsValidPrefix([])",),
        ensures=(
            "parser.IsValidPrefix(result)",
            "|result| <= |generated|",
            "forall t :: t in result ==> t in generated",
            "|generated| > 0 ==> |result| >= 0",
        ),
        axiom=True,
    )
    def FindLongestValidSpan(self, generated: Prefix) -> Prefix:
        best: Prefix = []
        n = len(generated)
        i = 0
        # invariant 0 <= i <= n
        # invariant parser.IsValidPrefix(best)
        # invariant |best| <= n
        # invariant forall t :: t in best ==> t in generated
        while i < n:
            j = i + 1
            # invariant i < j <= n + 1
            # invariant parser.IsValidPrefix(best)
            while j <= n:
                span = generated[i:j]
                if self.parser.IsValidPrefix(span) and len(span) > len(best):
                    best = span
                j = j + 1
            i = i + 1
        return best

    @dafny_spec(
        kind="method",
        requires=("parser.IsValidPrefix([])",),
        ensures=(
            "forall span :: span in result ==> parser.IsValidPrefix(span)",
            "forall span :: span in result ==> |span| > 0",
            "forall span :: span in result ==> (forall t :: t in span ==> t in generated)",
        ),
        axiom=True,
    )
    def ExtractAllValidSpans(self, generated: Prefix) -> list[Prefix]:
        spans: list[Prefix] = []
        n = len(generated)
        start = 0
        # invariant 0 <= start <= n
        # invariant forall span :: span in spans ==> parser.IsValidPrefix(span) && |span| > 0
        while start < n:
            best_end = start
            end = start + 1
            # invariant start < end <= n + 1
            while end <= n:
                if self.parser.IsValidPrefix(generated[start:end]):
                    best_end = end
                end = end + 1
            if best_end > start:
                spans = spans + [generated[start:best_end]]
                start = best_end
            else:
                start = start + 1
        return spans

    @dafny_spec(
        kind="method",
        modifies=("this.lm.Logits",),
        requires=(
            "this.lm.ValidTokensIdsLogits()",
            "parser.IsValidPrefix([])",
            "maxRetries >= 1",
            "stepsLeft >= maxRetries",
        ),
        ensures=(
            "this.lm.ValidTokensIdsLogits()",
            "parser.IsValidPrefix(LongestValidSuffix(result))",
            "|result| >= 0",
            "remainingSteps >= 0",
            "remainingSteps >= stepsLeft - maxRetries",
        ),
        axiom=True,
    )
    def RepairByRetry(self, prompt: Prefix, generated: Prefix, maxRetries: int, stepsLeft: int) -> tuple[Prefix, int]:
        repaired = self.RollbackToValidPrefix(generated)
        retries = 0
        steps_remaining = stepsLeft
        # invariant 0 <= retries <= maxRetries
        # invariant lm.ValidTokensIdsLogits()
        # invariant parser.IsValidPrefix(LongestValidSuffix(repaired))
        # invariant steps_remaining >= stepsLeft - retries
        # invariant steps_remaining >= 0
        while retries < maxRetries and steps_remaining > 0 and not self.parser.IsCompletePrefix(self.LongestValidSuffix(repaired)):
            next_tok, steps_remaining = self.ConstrainedStep(prompt, repaired, steps_remaining)
            repaired = repaired + [next_tok]
            retries = retries + 1
        return repaired, steps_remaining

    # ── Budget Utilities ──────────────────────────────────────────────────

    @dafny_spec(
        kind="predicate",
    )
    def HasBudget(self, stepsLeft: int, needed: int) -> bool:
        return stepsLeft >= needed

    @dafny_spec(
        kind="function",
        reads=("this", "this.parser"),
        requires=("parser.IsValidPrefix([])",),
        ensures=(
            "result >= 0",
            "parser.IsCompletePrefix(LongestValidSuffix(prefix)) ==> result == 0",
            "!parser.IsCompletePrefix(LongestValidSuffix(prefix)) ==> result >= 1",
        ),
    )
    def MinStepsToComplete(self, prefix: Prefix) -> int:
        return self.parser.ParserDistanceToComplete(self.LongestValidSuffix(prefix))

    # ── Composite Logit Shaping ───────────────────────────────────────────

    @dafny_spec(
        kind="method",
        modifies=("this.lm.Logits",),
        requires=(
            "this.lm.ValidTokensIdsLogits()",
            "parser.IsValidPrefix([])",
            "penalty > 0.0",
        ),
        ensures=(
            "this.lm.ValidTokensIdsLogits()",
            "forall t :: t in parser.ValidNextTokens(LongestValidSuffix(prefix)) && t in lm.Tokens ==> lm.Logits[lm.TokenToId(t)] == old(lm.Logits[lm.TokenToId(t)])",
        ),
    )
    def SoftConstrainToGrammar(self, prefix: Prefix, penalty: Logit) -> None:
        self.LongestValidSuffixIsValid(prefix)
        suffix = self.LongestValidSuffix(prefix)
        if self.parser.IsCompletePrefix(suffix):
            return
        valid_tokens = self.parser.ValidNextTokens(suffix)
        invalid_tokens: Prefix = []
        n = len(self.lm.Tokens)
        i = 0
        # invariant 0 <= i <= n
        # invariant lm.ValidTokensIdsLogits()
        # invariant forall j :: 0 <= j < i && !(lm.Tokens[j] in valid_tokens) ==> lm.Tokens[j] in invalid_tokens
        # invariant forall t :: t in invalid_tokens ==> t in lm.Tokens && !(t in valid_tokens)
        while i < n:
            if self.lm.Tokens[i] not in valid_tokens:
                invalid_tokens = invalid_tokens + [self.lm.Tokens[i]]
            i += 1
        if len(invalid_tokens) > 0:
            self.lm.BiasTokens(invalid_tokens, -penalty)

    @dafny_spec(
        kind="method",
        modifies=("this.lm.Logits",),
        requires=(
            "this.lm.ValidTokensIdsLogits()",
            "parser.IsValidPrefix([])",
        ),
        ensures=(
            "this.lm.ValidTokensIdsLogits()",
            "!parser.IsCompletePrefix(LongestValidSuffix(prefix)) ==> forall t :: t in lm.Tokens && !(t in parser.ValidNextTokens(LongestValidSuffix(prefix))) ==> lm.IsMasked(t)",
            "!parser.IsCompletePrefix(LongestValidSuffix(prefix)) ==> forall t :: t in parser.ValidNextTokens(LongestValidSuffix(prefix)) && t in lm.Tokens ==> lm.Logits[lm.TokenToId(t)] == old(lm.Logits[lm.TokenToId(t)])",
            "parser.IsCompletePrefix(LongestValidSuffix(prefix)) ==> forall t :: t in lm.Tokens ==> lm.Logits[lm.TokenToId(t)] == old(lm.Logits[lm.TokenToId(t)])",
        ),
    )
    def IntersectWithGrammar(self, prefix: Prefix) -> None:
        self.LongestValidSuffixIsValid(prefix)
        suffix = self.LongestValidSuffix(prefix)
        if self.parser.IsCompletePrefix(suffix):
            return
        valid_tokens = self.parser.ValidNextTokens(suffix)
        self.AllValidNextTokensInLM(suffix)
        self.lm.MaskTokensExcept(valid_tokens)

    @dafny_spec(
        kind="method",
        modifies=("this.lm.Logits",),
        requires=(
            "this.lm.ValidTokensIdsLogits()",
            "parser.IsValidPrefix([])",
            "bonus > 0.0",
        ),
        ensures=(
            "this.lm.ValidTokensIdsLogits()",
            "forall t :: t in lm.Tokens && !(exists ct :: ct in parser.ValidNextTokens(LongestValidSuffix(prefix)) && parser.IsCompletePrefix(LongestValidSuffix(prefix) + [ct]) && ct == t) ==> lm.Logits[lm.TokenToId(t)] == old(lm.Logits[lm.TokenToId(t)])",
        ),
    )
    def BiasForCompletion(self, prefix: Prefix, bonus: Logit) -> None:
        self.LongestValidSuffixIsValid(prefix)
        suffix = self.LongestValidSuffix(prefix)
        if self.parser.IsCompletePrefix(suffix):
            return
        valid_next = self.parser.ValidNextTokens(suffix)
        self.AllValidNextTokensInLM(suffix)
        n = len(valid_next)
        i = 0
        # invariant 0 <= i <= n
        # invariant lm.ValidTokensIdsLogits()
        # invariant forall t :: t in lm.Tokens && !(t in valid_next[..i] && parser.IsCompletePrefix(suffix + [t])) ==> lm.Logits[lm.TokenToId(t)] == old(lm.Logits[lm.TokenToId(t)])
        while i < n:
            if self.parser.IsCompletePrefix(suffix + [valid_next[i]]):
                self.lm.BiasToken(valid_next[i], bonus)
            i += 1


class CheckpointStack:
    stack: list[Prefix]

    @dafny_spec(
        kind="constructor",
        ensures=("Depth() == 0", "IsEmpty()"),
    )
    def __init__(self) -> None:
        self.stack = []

    @dafny_spec(
        kind="method",
        modifies=("this",),
        requires=(),
        ensures=(
            "Depth() == old(Depth()) + 1",
            "Peek() == prefix",
            "!IsEmpty()",
            "Depth() >= 1",
        ),
    )
    def Push(self, prefix: Prefix) -> None:
        self.stack = self.stack + [prefix]

    @dafny_spec(
        kind="method",
        modifies=("this",),
        requires=("Depth() > 0", "!IsEmpty()"),
        ensures=(
            "Depth() == old(Depth()) - 1",
            "Depth() >= 0",
            "|result| >= 0",
        ),
    )
    def Pop(self) -> Prefix:
        if len(self.stack) == 0:
            raise ValueError("Cannot Pop from empty CheckpointStack")
        result = self.stack[len(self.stack) - 1]
        self.stack = self.stack[:-1]
        return result

    @dafny_spec(
        kind="function",
        reads=("this",),
        requires=("Depth() > 0",),
        ensures=("|result| >= 0",),
    )
    def Peek(self) -> Prefix:
        return self.stack[len(self.stack) - 1]

    @dafny_spec(
        kind="function",
        reads=("this",),
        ensures=(
            "result >= 0",
            "result == 0 <==> IsEmpty()",
        ),
    )
    def Depth(self) -> int:
        return len(self.stack)

    @dafny_spec(
        kind="predicate",
        reads=("this",),
    )
    def IsEmpty(self) -> bool:
        return len(self.stack) == 0


class RepetitionTracker:
    ngramSize: int

    @dafny_spec(
        kind="constructor",
        requires=("ngramSize >= 1",),
        ensures=("this.ngramSize == ngramSize",),
    )
    def __init__(self, ngramSize: int) -> None:
        self.ngramSize = ngramSize
        self._ngramSize = ngramSize
        self._buffer: list[Token] = []
        self._counts: dict = {}

    @dafny_spec(
        kind="method",
        modifies=("this",),
        axiom=True,
        extern=True,
    )
    def RecordToken(self, token: Token) -> None:
        self._buffer.append(token)
        if len(self._buffer) >= self._ngramSize:
            ngram = tuple(self._buffer[-self._ngramSize:])
            self._counts[ngram] = self._counts.get(ngram, 0) + 1

    @dafny_spec(
        kind="function",
        reads=("this",),
        requires=("|ngram| == this.ngramSize",),
        ensures=(
            "result >= 0",
        ),
        axiom=True,
        extern=True,
    )
    def GetCount(self, ngram: Prefix) -> int:
        return self._counts.get(tuple(ngram), 0)

    @dafny_spec(
        kind="function",
        reads=("this",),
        ensures=(
            "result >= 0.0",
        ),
        axiom=True,
        extern=True,
    )
    def GetRepetitionPenalty(self, token: Token) -> Logit:
        if self._ngramSize == 1:
            return float(self._counts.get((token,), 0))
        if len(self._buffer) < self._ngramSize - 1:
            return 0.0
        recent = tuple(self._buffer[-(self._ngramSize - 1):]) + (token,)
        return float(self._counts.get(recent, 0))

    @dafny_spec(
        kind="method",
        modifies=("this",),
        requires=("lm.ValidTokensIdsLogits()",),
        ensures=("lm.ValidTokensIdsLogits()",),
        axiom=True,
        extern=True,
    )
    def ApplyRepetitionPenalties(self, lm: LM) -> None:
        for token in lm.Tokens:
            penalty = self.GetRepetitionPenalty(token)
            if penalty > 0.0:
                lm.BiasToken(token, -penalty)


__all__ = [
    "MODULE_NAME",
    "Token",
    "Prefix",
    "Id",
    "Logit",
    "DafnySpec",
    "dafny_spec",
    "Contains",
    "PrefixContains",
    "DelimitedAnswerValidForParser",
    "LeftDelimiter",
    "RightDelimiter",
    "LM",
    "Parser",
    "Delimiter",
    "CSDHelpers",
    "CheckpointStack",
    "RepetitionTracker",
]
