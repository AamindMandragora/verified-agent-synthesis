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
        if self.Tokens[offset] == token:
            return offset
        return self.TokenToIdRecursive(token, offset + 1)

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
        if len(tokens) == 1:
            return [self.TokenToLogit(tokens[0])]
        return [self.TokenToLogit(tokens[0])] + self.TokensToLogits(tokens[1:])

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
        if len(ids) == 1:
            return [self.IdToLogit(ids[0])]
        return [self.IdToLogit(ids[0])] + self.IdsToLogits(ids[1:])

    @dafny_spec(
        kind="method",
        modifies=("this.Logits",),
        requires=("ValidTokensIdsLogits()", "token in Tokens"),
        ensures=(
            "ValidTokensIdsLogits()",
            "Tokens[TokenToId(token)] == token",
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
        for i in range(len(prefix) - 1, -1, -1):
            if prefix[i] == self.Left:
                return i
        return len(prefix)

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
        for i, token in enumerate(content):
            if token == self.Right:
                return i
        return len(content)

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
        if start > len(prefix):
            return []
        after_left = prefix[start:]
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
    delimiter: Delimiter

    @dafny_spec(
        kind="constructor",
        requires=("delimiter.Left != delimiter.Right",),
        ensures=(
            "this.lm == lm && this.parser == parser && this.delimiter == delimiter",
            "this.delimiter.Left != this.delimiter.Right",
        ),
    )
    def __init__(self, lm: LM, parser: Parser, delimiter: Delimiter) -> None:
        if delimiter.Left == delimiter.Right:
            raise ValueError("delimiter.Left and delimiter.Right must differ")
        self.lm = lm
        self.parser = parser
        self.delimiter = delimiter

    @dafny_spec(
        kind="predicate",
        reads=("this", "this.delimiter", "this.lm", "this.lm.Logits"),
    )
    def DelimitersInLM(self) -> bool:
        return (
            self.lm.ValidTokensIdsLogits()
            and self.delimiter.Left in self.lm.Tokens
            and self.delimiter.Right in self.lm.Tokens
        )

    @dafny_spec(
        kind="lemma",
        ensures=("DelimitersInLM()",),
        axiom=True,
    )
    def DelimitersInLMAlways(self) -> None:
        assert self.DelimitersInLM()

    @dafny_spec(
        kind="function",
        reads=("this", "this.delimiter"),
        ensures=("result == this.delimiter.Left",),
    )
    def LeftDelimiter(self) -> Token:
        return self.delimiter.Left

    @dafny_spec(
        kind="function",
        reads=("this", "this.delimiter"),
        ensures=("result == this.delimiter.Right",),
    )
    def RightDelimiter(self) -> Token:
        return self.delimiter.Right

    @dafny_spec(
        kind="function",
        reads=("this", "this.delimiter", "this.lm", "this.lm.Logits"),
        requires=("lm.ValidTokensIdsLogits()",),
        ensures=("lm.ValidTokensIdsLogits()", "result == this.delimiter.GetDelimitedContent(prefix)"),
    )
    def GetDelimitedContent(self, prefix: Prefix) -> Prefix:
        return self.delimiter.GetDelimitedContent(prefix)

    @dafny_spec(
        kind="predicate",
        reads=("this", "this.delimiter"),
    )
    def InsideDelimitedWindow(self, prefix: Prefix) -> bool:
        return self.delimiter.InsideDelimitedWindow(prefix)

    @dafny_spec(
        kind="predicate",
        reads=("this", "this.delimiter", "this.parser"),
    )
    def ConstrainedWindowValid(self, prefix: Prefix) -> bool:
        return (not self.delimiter.InsideDelimitedWindow(prefix)) or self.parser.IsValidPrefix(
            self.delimiter.GetDelimitedContent(prefix)
        )

    @dafny_spec(
        kind="predicate",
        reads=("this", "this.delimiter", "this.parser", "this.lm", "this.lm.Logits"),
        requires=("lm.ValidTokensIdsLogits()",),
        ensures=("lm.ValidTokensIdsLogits()",),
    )
    def CompletedDelimitedAnswer(self, prefix: Prefix) -> bool:
        return (
            PrefixContains(prefix, self.LeftDelimiter())
            and PrefixContains(prefix, self.RightDelimiter())
            and (not self.InsideDelimitedWindow(prefix))
            and self.parser.IsCompletePrefix(self.GetDelimitedContent(prefix))
        )

    @dafny_spec(
        kind="predicate",
        reads=("this", "this.delimiter", "this.parser", "this.lm", "this.lm.Logits"),
        requires=("lm.ValidTokensIdsLogits()",),
        ensures=("lm.ValidTokensIdsLogits()",),
    )
    def DelimitedAnswerValid(self, prefix: Prefix) -> bool:
        return DelimitedAnswerValidForParser(self.parser, prefix)

    @dafny_spec(
        kind="lemma",
        requires=(
            "InsideDelimitedWindow(prefix)",
            "lm.ValidTokensIdsLogits()",
            "ConstrainedWindowValid(prefix)",
        ),
        ensures=("parser.IsValidPrefix(GetDelimitedContent(prefix))", "lm.ValidTokensIdsLogits()"),
        axiom=True,
    )
    def InDelimitedWindowThenContentValid(self, prefix: Prefix) -> None:
        assert self.InsideDelimitedWindow(prefix)
        assert self.lm.ValidTokensIdsLogits()
        assert self.ConstrainedWindowValid(prefix)
        assert self.parser.IsValidPrefix(self.GetDelimitedContent(prefix))

    @dafny_spec(
        kind="lemma",
        requires=(
            "lm.ValidTokensIdsLogits()",
            "InsideDelimitedWindow(prefix)",
            "ConstrainedWindowValid(prefix)",
            "parser.ValidNextToken(GetDelimitedContent(prefix), next)",
            "next != RightDelimiter()",
            "next != LeftDelimiter()",
        ),
        ensures=(
            "lm.ValidTokensIdsLogits()",
            "GetDelimitedContent(prefix + [next]) == GetDelimitedContent(prefix) + [next]",
            "ConstrainedWindowValid(prefix + [next])",
            "InsideDelimitedWindow(prefix + [next])",
        ),
    )
    def GetDelimitedContentAppend(self, prefix: Prefix, next: Token) -> None:
        self.delimiter.GetDelimitedContentAppend(prefix, next)

    @dafny_spec(
        kind="lemma",
        requires=("lm.ValidTokensIdsLogits()",),
        ensures=(
            "lm.ValidTokensIdsLogits()",
            "InsideDelimitedWindow(prefix + [delimiter.Left])",
            "GetDelimitedContent(prefix + [delimiter.Left]) == []",
            "parser.IsValidPrefix([])",
        ),
    )
    def EnterDelimitedWindow(self, prefix: Prefix) -> None:
        self.delimiter.AppendLeftEntersWindow(prefix)
        self.parser.EmptyPrefixIsValid()

    @dafny_spec(
        kind="lemma",
        requires=("InsideDelimitedWindow(prefix)", "this.delimiter.Left != this.delimiter.Right"),
        ensures=(
            "!InsideDelimitedWindow(prefix + [delimiter.Right])",
            "ConstrainedWindowValid(prefix + [delimiter.Right])",
        ),
    )
    def ExitDelimitedWindow(self, prefix: Prefix) -> None:
        self.delimiter.AppendRightExitsWindow(prefix)

    @dafny_spec(
        kind="lemma",
        requires=("lm.ValidTokensIdsLogits()", "ConstrainedWindowValid(prefix)"),
        ensures=("lm.ValidTokensIdsLogits()", "ConstrainedWindowValid(prefix + [next])"),
        axiom=True,
    )
    def UnconstrainedStepPreservesWindowValid(self, prefix: Prefix, next: Token) -> None:
        assert self.lm.ValidTokensIdsLogits()
        assert self.ConstrainedWindowValid(prefix)

    @dafny_spec(
        kind="method",
        modifies=("this.lm.Logits",),
        requires=(
            "this.lm.ValidTokensIdsLogits()",
            "ConstrainedWindowValid(generated)",
            "stepsLeft >= 1",
        ),
        ensures=(
            "this.lm.ValidTokensIdsLogits()",
            "stepsLeft' == stepsLeft - 1",
            "ConstrainedWindowValid(generated + [next])",
        ),
    )
    def UnconstrainedStep(self, prompt: Prefix, generated: Prefix, stepsLeft: int) -> tuple[Token, int]:
        self.lm.ValidTokensIdsLogitsAlways()
        self.lm.GenerateLogits(prompt + generated)
        next_token = self.lm.ChooseNextToken()
        steps_left_prime = stepsLeft - 1
        self.UnconstrainedStepPreservesWindowValid(generated, next_token)
        return next_token, steps_left_prime

    @dafny_spec(
        kind="method",
        modifies=("this.lm.Logits",),
        requires=(
            "this.lm.ValidTokensIdsLogits()",
            "ConstrainedWindowValid(generated)",
            "stepsLeft >= 1",
            "DelimitersInLM()",
        ),
        ensures=(
            "next in lm.Tokens",
            "next != LeftDelimiter()",
            "next != RightDelimiter()",
            "this.lm.ValidTokensIdsLogits()",
            "!this.lm.IsMasked(next)",
            "stepsLeft' == stepsLeft - 1",
            "ConstrainedWindowValid(generated + [next])",
        ),
    )
    def ExpressiveStep(self, prompt: Prefix, generated: Prefix, stepsLeft: int) -> tuple[Token, int]:
        self.lm.ValidTokensIdsLogitsAlways()
        self.lm.GenerateLogits(prompt + generated)
        self.lm.MaskToken(self.LeftDelimiter())
        self.lm.MaskToken(self.RightDelimiter())
        next_token = self.lm.ChooseNextToken()
        steps_left_prime = stepsLeft - 1
        self.UnconstrainedStepPreservesWindowValid(generated, next_token)
        return next_token, steps_left_prime

    @dafny_spec(
        kind="method",
        modifies=("this.lm.Logits",),
        requires=(
            "this.lm.ValidTokensIdsLogits()",
            "InsideDelimitedWindow(generated)",
            "ConstrainedWindowValid(generated)",
            "!parser.IsCompletePrefix(GetDelimitedContent(generated))",
            "stepsLeft >= 1",
            "DelimitersInLM()",
        ),
        ensures=(
            "next in lm.Tokens",
            "next != LeftDelimiter()",
            "this.lm.ValidTokensIdsLogits()",
            "parser.ValidNextToken(GetDelimitedContent(generated), next)",
            "!this.lm.IsMasked(next)",
            "stepsLeft' == stepsLeft - 1",
            "forall t: Token :: t in parser.ValidNextTokens(GetDelimitedContent(generated) + [next]) ==> t in this.lm.Tokens",
            "parser.IsValidPrefix(GetDelimitedContent(generated) + [next])",
            "next != RightDelimiter() ==> GetDelimitedContent(generated + [next]) == GetDelimitedContent(generated) + [next]",
            "next != RightDelimiter() ==> ConstrainedWindowValid(generated + [next])",
            "next != RightDelimiter() ==> InsideDelimitedWindow(generated + [next])",
            "ConstrainedWindowValid(generated + [next])",
        ),
    )
    def ConstrainedStep(self, prompt: Prefix, generated: Prefix, stepsLeft: int) -> tuple[Token, int]:
        self.ContentIsValidInWindow(generated)
        content = self.GetDelimitedContent(generated)
        self.ValidNextTokensInLM(content)
        self.lm.GenerateLogits(prompt + generated)
        self.lm.MaskTokensExcept(self.parser.ValidNextTokens(content))
        self.lm.MaskToken(self.LeftDelimiter())
        next_token = self.lm.ChooseNextToken()
        self.ConstrainedStepNextValid(content, next_token)
        steps_left_prime = stepsLeft - 1
        if next_token != self.RightDelimiter():
            self.delimiter.GetDelimitedContentAppend(generated, next_token)
        else:
            self.delimiter.AppendRightExitsWindow(generated)
        return next_token, steps_left_prime

    @dafny_spec(
        kind="method",
        modifies=("this.lm.Logits",),
        requires=(
            "this.lm.ValidTokensIdsLogits()",
            "parser.IsValidPrefix(answer)",
            "!parser.IsCompletePrefix(answer)",
            "stepsLeft >= 1",
        ),
        ensures=(
            "next in lm.Tokens",
            "this.lm.ValidTokensIdsLogits()",
            "parser.ValidNextToken(answer, next)",
            "!this.lm.IsMasked(next)",
            "stepsLeft' == stepsLeft - 1",
            "parser.IsValidPrefix(answer + [next])",
            "forall t: Token :: t in parser.ValidNextTokens(answer + [next]) ==> t in this.lm.Tokens",
        ),
    )
    def ConstrainedAnswerStep(
        self,
        prompt: Prefix,
        freeform: Prefix,
        answer: Prefix,
        stepsLeft: int,
    ) -> tuple[Token, int]:
        self.ValidNextTokensInLM(answer)
        self.lm.GenerateLogits(prompt + freeform + [self.LeftDelimiter()] + answer)
        self.lm.MaskTokensExcept(self.parser.ValidNextTokens(answer))
        next_token = self.lm.ChooseNextToken()
        self.ConstrainedStepNextValid(answer, next_token)
        steps_left_prime = stepsLeft - 1
        return next_token, steps_left_prime

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
    def ConstrainedStepNextValid(self, content: Prefix, next: Token) -> None:
        assert self.lm.ValidTokensIdsLogits()
        assert self.parser.IsValidPrefix(content)
        assert not self.parser.IsCompletePrefix(content)
        assert self.parser.IsValidPrefix(content + [next])

    @dafny_spec(
        kind="lemma",
        requires=(
            "InsideDelimitedWindow(generated)",
            "lm.ValidTokensIdsLogits()",
            "ConstrainedWindowValid(generated)",
        ),
        ensures=("lm.ValidTokensIdsLogits()", "parser.IsValidPrefix(GetDelimitedContent(generated))"),
    )
    def ContentIsValidInWindow(self, generated: Prefix) -> None:
        self.InDelimitedWindowThenContentValid(generated)

    @dafny_spec(
        kind="lemma",
        requires=("lm.ValidTokensIdsLogits()", "parser.IsValidPrefix(content)"),
        ensures=(
            "lm.ValidTokensIdsLogits()",
            "forall t: Token :: t in parser.ValidNextTokens(content) ==> t in lm.Tokens",
        ),
        axiom=True,
    )
    def ValidNextTokensInLM(self, content: Prefix) -> None:
        assert self.lm.ValidTokensIdsLogits()
        assert self.parser.IsValidPrefix(content)
        for token in self.parser.ValidNextTokens(content):
            assert token in self.lm.Tokens

    @dafny_spec(
        kind="lemma",
        requires=("lm.ValidTokensIdsLogits()", "ConstrainedWindowValid(prefix)"),
        ensures=(
            "InsideDelimitedWindow(prefix) ==> (forall t: Token :: t in parser.ValidNextTokens(GetDelimitedContent(prefix)) ==> t in lm.Tokens)",
        ),
        axiom=True,
    )
    def RollbackPreservesTokenInvariant(self, prefix: Prefix) -> None:
        assert self.lm.ValidTokensIdsLogits()
        assert self.ConstrainedWindowValid(prefix)

    @dafny_spec(
        kind="lemma",
        requires=(
            "lm.ValidTokensIdsLogits()",
            "parser.IsValidPrefix(answer)",
            "|answer| > 0",
        ),
        ensures=(
            "GetDelimitedContent(freeform + [LeftDelimiter()] + answer + [RightDelimiter()]) == answer",
            "!InsideDelimitedWindow(freeform + [LeftDelimiter()] + answer + [RightDelimiter()])",
            "ConstrainedWindowValid(freeform + [LeftDelimiter()] + answer + [RightDelimiter()])",
            "DelimitedAnswerValidForParser(parser, freeform + [LeftDelimiter()] + answer + [RightDelimiter()])",
        ),
        axiom=True,
    )
    def FinalizeDelimitedAnswer(self, freeform: Prefix, answer: Prefix) -> None:
        assert self.lm.ValidTokensIdsLogits()
        assert self.parser.IsValidPrefix(answer)
        assert len(answer) > 0

    @dafny_spec(
        kind="method",
        requires=("parser.IsValidPrefix([])",),
        ensures=("parser.IsValidPrefix(repaired)", "|repaired| <= |generated|"),
    )
    def RollbackToValidPrefix(self, generated: Prefix) -> Prefix:
        repaired = list(generated)

        # invariant |repaired| <= |generated|
        # invariant parser.IsValidPrefix(repaired) || |repaired| > 0
        while len(repaired) > 0 and (
            (not self.parser.IsValidPrefix(repaired)) or self.parser.IsDeadPrefix(repaired)
        ):
            repaired = repaired[:-1]
        return repaired


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
    "LeftDelimiter",
    "RightDelimiter",
    "LM",
    "Parser",
    "Delimiter",
    "CSDHelpers",
]
