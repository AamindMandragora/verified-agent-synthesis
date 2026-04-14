# `VerifiedAgentSynthesis.py` Function Reference

This document summarizes every function and method defined in `VerifiedAgentSynthesis.py`.

The file mixes:
- runtime behavior,
- abstract interfaces marked `extern`,
- and verification-oriented helper lemmas/predicates used to express invariants.

## Top-Level Functions

| Name | Function |
| --- | --- |
| `dafny_spec(...)` | Decorator factory that attaches Dafny-style metadata (`kind`, `requires`, `ensures`, etc.) to functions and methods. |
| `decorate(obj)` | Inner helper returned by `dafny_spec`; writes the constructed `DafnySpec` to `obj.__dafny_spec__`. |
| `Contains(s, sub)` | Predicate wrapper around Python substring membership (`sub in s`). |
| `PrefixContains(p, t)` | Returns whether a token appears anywhere in a token prefix/list. |
| `DelimitedAnswerValidForParser(parser, prefix)` | Checks whether `prefix` contains a finished delimited answer: it must contain both delimiters, be outside the delimiter window, and the extracted content must be a non-empty valid parser prefix. |

## `LM` Methods

| Name | Function |
| --- | --- |
| `LM.__init__()` | Initializes the language-model token/id/logit tables with the default left and right delimiter tokens. |
| `LM.ValidTokensIdsLogits()` | Core invariant check for the model vocabulary: token/id/logit arrays must align, ids must be consistent, tokens must be unique, and logits must be in range. |
| `LM.ValidTokensIdsLogitsAlways()` | Lemma-style helper asserting that the `LM` invariant holds. |
| `LM.IdToToken(id)` | Converts a token id to its corresponding token string. |
| `LM.TokenToId(token)` | Converts a token string to its corresponding id by delegating to the recursive search helper. |
| `LM.TokenToIdRecursive(token, offset)` | Recursively scans the token list from `offset` until it finds the requested token. |
| `LM.IdToLogit(id)` | Returns the logit stored for a given token id. |
| `LM.TokenToLogit(token)` | Looks up a token's logit by composing `TokenToId` and `IdToLogit`. |
| `LM.TokensToLogits(tokens)` | Recursively maps a non-empty list of tokens to their logits. |
| `LM.IdsToLogits(ids)` | Recursively maps a non-empty list of ids to their logits. |
| `LM.MaskToken(token)` | Masks one token by setting its logit to `-1e9`, making it effectively unselectable. |
| `LM.MaskTokens(tokens)` | Masks every token in a provided token list. |
| `LM.MaskTokensExcept(tokens)` | Masks all model tokens except the provided allowlist. |
| `LM.IsMasked(token)` | Returns whether a token is currently masked. |
| `LM.HasUnmaskedToken()` | Returns whether the model still has at least one selectable token. |
| `LM.GenerateLogits(input)` | External/placeholder hook where logits for the next decoding step are expected to be produced from the prompt/prefix. |
| `LM.ChooseNextToken()` | Selects the highest-logit unmasked token, raising an error if all tokens are masked. |

## `Parser` Methods

| Name | Function |
| --- | --- |
| `Parser.IsValidPrefix(prefix)` | Abstract predicate for whether `prefix` is syntactically valid so far. |
| `Parser.EmptyPrefixIsValid()` | Lemma asserting that the empty prefix is valid. |
| `Parser.IsCompletePrefix(prefix)` | Abstract predicate for whether `prefix` is a complete parse/answer. |
| `Parser.IsDeadPrefix(prefix)` | Returns whether `prefix` is neither complete nor extendable by any valid next token. |
| `Parser.ValidNextToken(prefix, token)` | Convenience predicate for membership in `ValidNextTokens(prefix)`. |
| `Parser.ValidNextTokens(prefix)` | Abstract function returning the allowed next tokens for a valid prefix. |

## `Delimiter` Methods

| Name | Function |
| --- | --- |
| `Delimiter.__init__(left, right)` | Builds a delimiter pair and rejects equal left/right endpoints. |
| `Delimiter.LastLeftDelimiterIndex(prefix)` | Returns the index of the last left delimiter in `prefix`, or `len(prefix)` if none exists. |
| `Delimiter.FirstRightDelimiterIndex(content)` | Returns the index of the first right delimiter in `content`, or `len(content)` if none exists. |
| `Delimiter.NoFirstRightDelimiterIndexMeansNoRight(content)` | Lemma showing that if no right delimiter index is found, then the content truly contains no right delimiter. |
| `Delimiter.GetDelimitedContent(prefix)` | Extracts the tokens after the last left delimiter and before the next right delimiter. |
| `Delimiter.InsideDelimitedWindow(prefix)` | Returns whether decoding is currently inside an open delimiter span that has not yet been closed. |
| `Delimiter.InsideDelimitedWindowNoRight(prefix)` | Lemma showing that content inside an open delimiter window contains no right delimiter yet. |
| `Delimiter.GetDelimitedContentAppend(prefix, next)` | Lemma showing that appending a non-delimiter token inside the window appends to the extracted delimited content and keeps the window open. |
| `Delimiter.AppendLeftEntersWindow(prefix)` | Lemma showing that appending a left delimiter opens a new delimited window with empty content. |
| `Delimiter.FirstRightDelimiterAppendRight(content)` | Lemma showing how appending the right delimiter sets the first-right-delimiter position when none existed before. |
| `Delimiter.LastLeftDelimiterAppendNonLeft(prefix, tok)` | Lemma describing how appending a non-left token affects the last-left-delimiter index. |
| `Delimiter.AppendRightExitsWindow(prefix)` | Lemma showing that appending the right delimiter closes an open delimiter window. |

## `CSDHelpers` Methods

| Name | Function |
| --- | --- |
| `CSDHelpers.__init__(lm, parser, delimiter)` | Stores the LM, parser, and delimiter objects and checks that the delimiter endpoints differ. |
| `CSDHelpers.DelimitersInLM()` | Verifies that both delimiter tokens exist in the language model vocabulary. |
| `CSDHelpers.DelimitersInLMAlways()` | Lemma asserting that delimiter tokens are present in the LM. |
| `CSDHelpers.LeftDelimiter()` | Returns the configured left delimiter token. |
| `CSDHelpers.RightDelimiter()` | Returns the configured right delimiter token. |
| `CSDHelpers.GetDelimitedContent(prefix)` | Convenience wrapper around `Delimiter.GetDelimitedContent`. |
| `CSDHelpers.InsideDelimitedWindow(prefix)` | Convenience wrapper around `Delimiter.InsideDelimitedWindow`. |
| `CSDHelpers.ConstrainedWindowValid(prefix)` | Ensures that if decoding is inside the delimited window, the extracted content is still a parser-valid prefix. |
| `CSDHelpers.CompletedDelimitedAnswer(prefix)` | Returns whether `prefix` contains a finished delimited answer whose content is parser-complete. |
| `CSDHelpers.DelimitedAnswerValid(prefix)` | Returns whether `prefix` contains a non-empty finished delimited answer that is parser-valid. |
| `CSDHelpers.InDelimitedWindowThenContentValid(prefix)` | Lemma deriving that open-window content is parser-valid when the constrained-window invariant holds. |
| `CSDHelpers.GetDelimitedContentAppend(prefix, next)` | Wrapper lemma delegating the append-inside-window reasoning to `Delimiter`. |
| `CSDHelpers.EnterDelimitedWindow(prefix)` | Lemma for transitioning into constrained decoding after emitting the left delimiter. |
| `CSDHelpers.ExitDelimitedWindow(prefix)` | Lemma for transitioning out of constrained decoding after emitting the right delimiter. |
| `CSDHelpers.UnconstrainedStepPreservesWindowValid(prefix, next)` | Lemma stating that an unconstrained decoding step preserves the window-validity invariant. |
| `CSDHelpers.UnconstrainedStep(prompt, generated, stepsLeft)` | Performs one normal LM decoding step with no delimiter-specific masking and decrements the remaining step counter. |
| `CSDHelpers.ExpressiveStep(prompt, generated, stepsLeft)` | Performs one unconstrained/freeform decoding step while masking both delimiters so they cannot be emitted. |
| `CSDHelpers.ConstrainedStep(prompt, generated, stepsLeft)` | Performs one decoding step inside the delimiter window, allowing only parser-valid next tokens and disallowing the left delimiter. |
| `CSDHelpers.ConstrainedAnswerStep(prompt, freeform, answer, stepsLeft)` | Similar to `ConstrainedStep`, but operates directly on a separated `freeform + left-delimiter + answer` decomposition. |
| `CSDHelpers.ConstrainedStepNextValid(content, next)` | Lemma that carries the valid-next-token invariant forward after a constrained step. |
| `CSDHelpers.ContentIsValidInWindow(generated)` | Convenience lemma that re-establishes parser validity for the currently extracted delimited content. |
| `CSDHelpers.ValidNextTokensInLM(content)` | Lemma asserting that every parser-allowed next token is present in the LM vocabulary. |
| `CSDHelpers.RollbackPreservesTokenInvariant(prefix)` | Lemma stating that rollback under the constrained-window invariant preserves the token-availability assumption. |
| `CSDHelpers.FinalizeDelimitedAnswer(freeform, answer)` | Lemma showing that wrapping a non-empty valid answer with delimiters produces a valid, closed delimited answer segment. |
| `CSDHelpers.RollbackToValidPrefix(generated)` | Trims tokens from the end of `generated` until the remaining prefix is parser-valid and not dead. |

## Notes

- Methods marked as abstract/`extern` are interface points or proof placeholders rather than fully implemented runtime logic.
- Methods named like lemmas primarily exist to support reasoning about invariants and transitions in constrained decoding.
- The runtime decoding flow is mainly centered around `UnconstrainedStep`, `ExpressiveStep`, `ConstrainedStep`, `ConstrainedAnswerStep`, and `RollbackToValidPrefix`.
