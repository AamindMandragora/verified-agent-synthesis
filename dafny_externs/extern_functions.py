"""
Python implementations of Dafny {:extern} functions.

These implementations provide the runtime behavior for the {:extern} functions
defined in VerifiedAgentSynthesis.dfy. They are used when executing compiled
Dafny strategies in Python.
"""

from typing import Optional
import random


# =============================================================================
# Type Aliases (matching Dafny types)
# =============================================================================

Token = str
Prefix = list[Token]
Id = int
Logit = float


# =============================================================================
# Language Model (LM) Class
# =============================================================================

class LM:
    """Language Model wrapper for constrained decoding."""

    def __init__(
        self,
        tokens: Optional[list[Token]] = None,
        vocab_size: int = 1000,
    ):
        if tokens is None:
            self.Tokens = self._generate_default_vocab(vocab_size)
        else:
            self.Tokens = tokens

        self.Ids = list(range(len(self.Tokens)))
        self.Logits = [0.0] * len(self.Tokens)
        self._token_to_id = {t: i for i, t in enumerate(self.Tokens)}

    def _generate_default_vocab(self, size: int) -> list[Token]:
        vocab = [
            "<EOS>", "<PAD>", " ", "\n", "\t",
            "(", ")", "[", "]", "{", "}",
            "+", "-", "*", "/", "=", "<", ">",
            "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
            "<<", ">>",  # delimiters
        ]
        for c in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ":
            vocab.append(c)
        while len(vocab) < size:
            vocab.append(f"<T{len(vocab)}>")
        return vocab[:size]

    # --- Dafny predicates / lemmas ---

    def ValidTokensIdsLogits(self) -> bool:
        return (
            len(self.Tokens) == len(self.Ids) == len(self.Logits)
            and len(self.Ids) > 0
            and self.Ids[0] == 0
        )

    def ValidTokensIdsLogitsAlways(self) -> None:
        """Axiom: always holds; no-op in Python."""
        pass

    # --- Token / Id / Logit conversions ---

    def IdToToken(self, id: Id) -> Token:
        return self.Tokens[id]

    def TokenToId(self, token: Token) -> Id:
        return self._token_to_id[token]

    def IdToLogit(self, id: Id) -> Logit:
        return self.Logits[id]

    def TokenToLogit(self, token: Token) -> Logit:
        return self.Logits[self.TokenToId(token)]

    def TokensToLogits(self, tokens: list[Token]) -> list[Logit]:
        return [self.TokenToLogit(t) for t in tokens]

    def IdsToLogits(self, ids: list[Id]) -> list[Logit]:
        return [self.IdToLogit(i) for i in ids]

    # --- Masking ---

    def MaskToken(self, token: Token) -> None:
        self.Logits[self.TokenToId(token)] = -1e9

    def MaskTokens(self, tokens: list[Token]) -> None:
        for token in tokens:
            self.MaskToken(token)

    def MaskTokensExcept(self, tokens: list[Token]) -> None:
        allowed = set(tokens)
        for i, t in enumerate(self.Tokens):
            if t not in allowed:
                self.Logits[i] = -1e9

    def IsMasked(self, token: Token) -> bool:
        return self.Logits[self.TokenToId(token)] == -1e9

    def HasUnmaskedToken(self) -> bool:
        return any(not self.IsMasked(t) for t in self.Tokens)

    # --- Generation ---

    def GenerateLogits(self, input: Prefix) -> None:
        self.Logits = [random.gauss(0, 1) for _ in self.Tokens]

    def ChooseNextToken(self) -> Token:
        max_idx = max(range(len(self.Logits)), key=lambda i: self.Logits[i])
        return self.Tokens[max_idx]

    def ChooseNextTokenUnconstrained(self) -> Token:
        return self.Tokens[random.randint(0, len(self.Tokens) - 1)]


# =============================================================================
# Parser Class
# =============================================================================

class Parser:
    """Grammar parser for validating token sequences."""

    def __init__(
        self,
        grammar: Optional[str] = None,
        valid_tokens: Optional[set[Token]] = None,
    ):
        self.grammar = grammar
        self._valid_tokens = valid_tokens or set()
        self._always_valid = True

    def EmptyPrefixIsValid(self) -> None:
        """Axiom: no-op in Python."""
        pass

    def IsValidPrefix(self, prefix: Prefix) -> bool:
        if self._always_valid:
            return True
        return all(t in self._valid_tokens for t in prefix)

    def IsCompletePrefix(self, prefix: Prefix) -> bool:
        if not prefix:
            return False
        return prefix[-1] == ">>"

    def IsDeadPrefix(self, prefix: Prefix) -> bool:
        return not self.IsCompletePrefix(prefix) and len(self.ValidNextTokens(prefix)) == 0

    def ValidNextToken(self, prefix: Prefix, token: Token) -> bool:
        return token in self.ValidNextTokens(prefix)

    def ValidNextTokens(self, prefix: Prefix) -> list[Token]:
        return list(self._valid_tokens) if self._valid_tokens else []


# =============================================================================
# Delimiter Class
# =============================================================================

class Delimiter:
    """Tracks the constrained window delimited by Left/Right tokens."""

    def __init__(self, left: Token, right: Token):
        self.Left = left
        self.Right = right

    def LastLeftDelimiterIndex(self, prefix: Prefix) -> int:
        """Index of the last Left token in prefix, or len(prefix) if none."""
        for i in range(len(prefix) - 1, -1, -1):
            if prefix[i] == self.Left:
                return i
        return len(prefix)

    def FirstRightDelimiterIndex(self, content: Prefix) -> int:
        """Index of the first Right token in content, or len(content) if none."""
        for i, t in enumerate(content):
            if t == self.Right:
                return i
        return len(content)

    def GetDelimitedContent(self, prefix: Prefix) -> Prefix:
        """Tokens strictly between the last Left and the next Right (or end)."""
        start = self.LastLeftDelimiterIndex(prefix) + 1
        if start > len(prefix):
            return []
        after_left = prefix[start:]
        end_idx = self.FirstRightDelimiterIndex(after_left)
        return after_left[:end_idx]

    def InsideDelimitedWindow(self, prefix: Prefix) -> bool:
        """True iff we have seen Left and not yet seen a matching Right."""
        start = self.LastLeftDelimiterIndex(prefix) + 1
        if start > len(prefix):
            return False
        after_left = prefix[start:]
        return self.FirstRightDelimiterIndex(after_left) == len(after_left)


# =============================================================================
# CSDHelpers Class
# =============================================================================

class CSDHelpers:
    """Holds lm, parser, and delimiter; implements all step/helper methods."""

    def __init__(self, lm: LM, parser: Parser, delimiter: Delimiter):
        self.lm = lm
        self.parser = parser
        self.delimiter = delimiter

    # --- Delimiter conveniences ---

    def LeftDelimiter(self) -> Token:
        return self.delimiter.Left

    def RightDelimiter(self) -> Token:
        return self.delimiter.Right

    def GetDelimitedContent(self, prefix: Prefix) -> Prefix:
        return self.delimiter.GetDelimitedContent(prefix)

    def InsideDelimitedWindow(self, prefix: Prefix) -> bool:
        return self.delimiter.InsideDelimitedWindow(prefix)

    def ConstrainedWindowValid(self, prefix: Prefix) -> bool:
        if not self.delimiter.InsideDelimitedWindow(prefix):
            return True
        return self.parser.IsValidPrefix(self.delimiter.GetDelimitedContent(prefix))

    # --- Lemmas (no-ops in Python) ---

    def InDelimitedWindowThenContentValid(self, prefix: Prefix) -> None:
        pass

    def GetDelimitedContentAppend(self, prefix: Prefix, next: Token) -> None:
        pass

    def ConstrainedStepNextValid(self, content: Prefix, next: Token) -> None:
        pass

    def RollbackPreservesTokenInvariant(self, prefix: Prefix) -> None:
        pass

    # --- Step methods ---

    def UnconstrainedStep(self, prompt: Prefix, generated: Prefix, stepsLeft: int) -> tuple[Token, int]:
        self.lm.ValidTokensIdsLogitsAlways()
        self.lm.GenerateLogits(prompt + generated)
        next_token = self.lm.ChooseNextTokenUnconstrained()
        return next_token, stepsLeft - 1

    def ConstrainedStep(self, prompt: Prefix, generated: Prefix, stepsLeft: int) -> tuple[Token, int]:
        self.lm.ValidTokensIdsLogitsAlways()
        content = self.GetDelimitedContent(generated)
        self.lm.GenerateLogits(prompt + generated)
        valid = self.parser.ValidNextTokens(content)
        self.lm.MaskTokensExcept(valid)
        next_token = self.lm.ChooseNextToken()
        self.ConstrainedStepNextValid(content, next_token)
        return next_token, stepsLeft - 1

    def RollbackToValidPrefix(self, generated: Prefix) -> Prefix:
        repaired = list(generated)
        while repaired and (
            not self.parser.IsValidPrefix(repaired)
            or self.parser.IsDeadPrefix(repaired)
        ):
            repaired = repaired[:-1]
        return repaired
