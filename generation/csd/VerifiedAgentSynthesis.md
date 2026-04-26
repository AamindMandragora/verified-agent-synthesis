# CSD Helper Library — Function Reference

This document specifies the complete set of composable library functions available for synthesizing constrained decoding strategies (CSDs).

The library is built on two primitives:
- **LM**: a language model that accepts a token prefix, produces logits over the vocabulary, and selects a token.
- **Parser**: a grammar-backed oracle that can validate prefixes, report completeness, identify dead ends, and enumerate valid continuations.

Everything else — step functions, logit shapers, repair utilities, parser queries, and state structures — is composed from these two primitives.

The file mixes:
- runtime behavior,
- abstract interfaces marked `extern`,
- and verification-oriented helper lemmas/predicates used to express invariants.

---

## Top-Level Functions

| Name | Signature | Function |
|------|-----------|----------|
| `dafny_spec(...)` | `(**kwargs) → decorator` | Decorator factory that attaches Dafny-style metadata (`kind`, `requires`, `ensures`, etc.) to functions and methods. |
| `decorate(obj)` | `(obj) → obj` | Inner helper returned by `dafny_spec`; writes the constructed `DafnySpec` to `obj.__dafny_spec__`. |
| `Contains(s, sub)` | `(str, str) → bool` | Predicate wrapper around Python substring membership (`sub in s`). |
| `PrefixContains(p, t)` | `(Prefix, Token) → bool` | Returns whether a token appears anywhere in a token prefix/list. |

---

## `LM` — Language Model Interface

Manages the token vocabulary, id mapping, logit array, and all logit-shaping operations. The logit array is mutable; every shaping method modifies it in place.

### Vocabulary and Invariant

| Name | Signature | Function |
|------|-----------|----------|
| `__init__()` | `() → LM` | Initializes the language-model token/id/logit tables. Extern constructor. |
| `ValidTokensIdsLogits()` | `() → bool` | Core invariant predicate: token/id/logit arrays must align in length, ids must be sequential from 0, tokens must be unique, and all logits must be in `[-1e9, 1e9]`. Nearly every other method requires this as a precondition. |
| `ValidTokensIdsLogitsAlways()` | `() → None` | Lemma asserting that the LM invariant holds unconditionally. Used to re-establish the invariant at proof boundaries. |

### Token / Id / Logit Conversions

| Name | Signature | Function |
|------|-----------|----------|
| `IdToToken(id)` | `(Id) → Token` | Converts a token id to its corresponding token string. Requires `id in Ids`. |
| `TokenToId(token)` | `(Token) → Id` | Converts a token string to its corresponding id by delegating to the recursive search helper. Requires `token in Tokens`. |
| `TokenToIdRecursive(token, offset)` | `(Token, int) → Id` | Recursively scans the token list from `offset` until it finds the requested token. Decreases on `|Tokens| - offset`. |
| `IdToLogit(id)` | `(Id) → Logit` | Returns the logit stored for a given token id. |
| `TokenToLogit(token)` | `(Token) → Logit` | Looks up a token's logit by composing `TokenToId` and `IdToLogit`. |
| `TokensToLogits(tokens)` | `(Prefix) → list[Logit]` | Recursively maps a non-empty list of tokens to their logits. |
| `IdsToLogits(ids)` | `(list[Id]) → list[Logit]` | Recursively maps a non-empty list of ids to their logits. |

### Hard Masking

These set logits to `-1e9`, making tokens effectively unselectable.

| Name | Signature | Function |
|------|-----------|----------|
| `MaskToken(token)` | `(Token) → None` | Masks one token by setting its logit to `-1e9`. Ensures all other logits are unchanged. |
| `MaskTokens(tokens)` | `(Prefix) → None` | Masks every token in the provided list. Ensures all tokens outside the list are unchanged. |
| `MaskTokensExcept(tokens)` | `(Prefix) → None` | Masks all vocabulary tokens *except* the provided allowlist. Ensures allowlisted logits are unchanged. |
| `IsMasked(token)` | `(Token) → bool` | Returns whether a token's logit is currently at `-1e9`. |
| `HasUnmaskedToken()` | `() → bool` | Returns whether the model still has at least one selectable (non-masked) token. |

### Soft Logit Shaping

These modify logits without fully zeroing them, enabling graduated preference rather than binary allow/deny.

| Name | Signature | Function |
|------|-----------|----------|
| `BiasToken(token, delta)` | `(Token, Logit) → None` | Adds `delta` to the logit of `token`, clamped to `[-1e9, 1e9]`. Ensures: `Logits[TokenToId(token)] == clamp(old(Logits[TokenToId(token)]) + delta, -1e9, 1e9)`. All other logits are unchanged. |
| `BiasTokens(tokens, delta)` | `(Prefix, Logit) → None` | Applies `BiasToken(t, delta)` for every `t` in `tokens`. Ensures: all tokens outside the list are unchanged. |
| `ScaleToken(token, factor)` | `(Token, Logit) → None` | Multiplies the logit of `token` by `factor`, clamped to `[-1e9, 1e9]`. Requires `factor != 0.0` (to keep the operation invertible in principle). Ensures: `Logits[TokenToId(token)] == clamp(old(Logits[TokenToId(token)]) * factor, -1e9, 1e9)`. All other logits are unchanged. |
| `ScaleTokens(tokens, factor)` | `(Prefix, Logit) → None` | Applies `ScaleToken(t, factor)` for every `t` in `tokens`. Requires `factor != 0.0`. Ensures: all tokens outside the list are unchanged. |
| `ClampLogits(low, high)` | `(Logit, Logit) → None` | Clips every logit in the vocabulary to `[low, high]`. Requires `-1e9 <= low <= high <= 1e9`. Ensures: `forall id :: low <= Logits[id] <= high`. Useful as a normalization step after multiple bias/scale operations to prevent runaway values. |

### Filtering

| Name | Signature | Function |
|------|-----------|----------|
| `TopKFilter(k)` | `(int) → None` | Masks all tokens except the `k` with the highest logits. Requires `1 <= k <= |Tokens|`. Ensures: the number of unmasked tokens is `<= k`, and every surviving token's logit is `>=` every masked token's pre-masking logit. Ties are broken arbitrarily. This is an LM-confidence filter, orthogonal to grammar filtering. |

### Generation

| Name | Signature | Function |
|------|-----------|----------|
| `GenerateLogits(input)` | `(Prefix) → None` | Extern hook. Populates the logit array with the LM's next-token distribution conditioned on `input`. Modifies `Logits`. Ensures the LM invariant is preserved. |
| `ChooseNextToken()` | `() → Token` | Returns the highest-logit unmasked token. Raises an error if all tokens are masked. Extern. Ensures: `token in Tokens`, `!IsMasked(token)`. |

---

## `Parser` — Grammar Oracle Interface

All methods are abstract/extern except `IsDeadPrefix` and `ValidNextToken`, which are defined in terms of the other methods. The parser is stateless: all state is carried in the `prefix` argument.

| Name | Signature | Function |
|------|-----------|----------|
| `IsValidPrefix(prefix)` | `(Prefix) → bool` | Abstract predicate for whether `prefix` is a syntactically valid partial parse under the grammar. Ensures that every proper prefix of a valid prefix is also valid. Extern. |
| `EmptyPrefixIsValid()` | `() → None` | Lemma asserting that the empty prefix `[]` is always valid. |
| `IsCompletePrefix(prefix)` | `(Prefix) → bool` | Abstract predicate for whether `prefix` is a complete, finished parse under the grammar. Ensures `IsValidPrefix(prefix)`. Extern. |
| `IsDeadPrefix(prefix)` | `(Prefix) → bool` | Returns whether `prefix` is neither complete nor extendable by any valid next token. Defined as `!IsCompletePrefix(prefix) && |ValidNextTokens(prefix)| == 0`. |
| `ValidNextToken(prefix, token)` | `(Prefix, Token) → bool` | Convenience predicate: returns `token in ValidNextTokens(prefix)`. Requires `IsValidPrefix(prefix)`. |
| `ValidNextTokens(prefix)` | `(Prefix) → Prefix` | Abstract function returning the set of tokens that can validly extend `prefix`. Requires `IsValidPrefix(prefix)`. Ensures: every returned token produces a valid extension, and either the prefix is complete or at least one valid next token exists. Extern. |

### Extended Parser Queries

These provide richer information about the grammar state to enable more sophisticated strategy decisions.

| Name | Signature | Function |
|------|-----------|----------|
| `ValidContinuationCount(prefix)` | `(Prefix) → int` | Returns `|ValidNextTokens(prefix)|`. Requires `IsValidPrefix(prefix)`. Ensures `result >= 0`, and `result == 0 ==> IsCompletePrefix(prefix) \|\| IsDeadPrefix(prefix)`. Cheap to compute but lets strategies detect bottlenecks (count == 1 means forced move) versus wide-open states. |
| `PrefixMatchesSubgrammar(prefix, label)` | `(Prefix, str) → bool` | Returns whether the current partial parse is inside a named grammar rule identified by `label` (e.g., `"number"`, `"expression"`, `"identifier"`). This lets strategies apply different constraint strengths depending on which part of the grammar is active. Extern axiom. |
| `ParserDistanceToComplete(prefix)` | `(Prefix) → int` | Returns a lower bound on the minimum number of additional tokens needed to reach a complete parse from `prefix`. Requires `IsValidPrefix(prefix)`. Ensures `result >= 0` and `result == 0 ==> IsCompletePrefix(prefix)`. Extern axiom (the actual computation depends on grammar structure). |

---

## `CSDHelpers` — Strategy Building Blocks

The central helper class that composes the LM and Parser into reusable decoding step functions, repair utilities, and state management tools.

### Constructor and Core Predicates

| Name | Signature | Function |
|------|-----------|----------|
| `__init__(lm, parser)` | `(LM, Parser) → CSDHelpers` | Stores the LM and parser references. Ensures `lm.ValidTokensIdsLogits()`. |
| `AllValidNextTokensInLM(content)` | `(Prefix) → None` | Lemma asserting that every token returned by `parser.ValidNextTokens(content)` is present in the LM vocabulary. Requires `lm.ValidTokensIdsLogits()` and `parser.IsValidPrefix(content)`. Extern axiom. |

### Suffix-Based Grammar Alignment

Instead of requiring explicit delimiter management, these functions find the grammar-relevant portion of any prefix by scanning for the longest suffix that constitutes a valid parser prefix.

| Name | Signature | Function |
|------|-----------|----------|
| `LongestValidSuffix(prefix)` | `(Prefix) → Prefix` | Returns the longest suffix of `prefix` such that `parser.IsValidPrefix(suffix)` holds. If no suffix is valid (including the empty suffix, which is always valid by `EmptyPrefixIsValid`), returns `[]`. Ensures: `parser.IsValidPrefix(result)`, `result` is a suffix of `prefix`, and no longer suffix of `prefix` is a valid parser prefix. Decreases on `|prefix|`. |
| `LongestValidSuffixAppend(prefix, next)` | `(Prefix, Token) → None` | Lemma relating `LongestValidSuffix(prefix + [next])` to `LongestValidSuffix(prefix)`. If `next` is a valid continuation of `LongestValidSuffix(prefix)`, then the longest valid suffix of `prefix + [next]` is at least `LongestValidSuffix(prefix) + [next]`. Axiom. |
| `LongestValidSuffixIsValid(prefix)` | `(Prefix) → None` | Convenience lemma re-establishing `parser.IsValidPrefix(LongestValidSuffix(prefix))`. |
| `CanConstrain(prefix)` | `(Prefix) → bool` | Convenience predicate for `!parser.IsCompletePrefix(LongestValidSuffix(prefix))`. This is the preferred guard before calling constrained step helpers. |

### Raw Step Functions

Each step function performs one LM decoding step with a specific constraint policy. All step functions consume one unit from `stepsLeft` and return `(next_token, stepsLeft - 1)`.

| Name | Signature | Function |
|------|-----------|----------|
| `UnconstrainedStep(prompt, generated, stepsLeft)` | `(Prefix, Prefix, int) → (Token, int)` | Generates logits for `prompt + generated`, then chooses the next token with no masking or shaping. The vanilla baseline. Requires `stepsLeft >= 1`. Ensures `next in lm.Tokens`. |
| `ConstrainedStep(prompt, generated, stepsLeft)` | `(Prefix, Prefix, int) → (Token, int)` | Generates logits for `prompt + generated`, computes `LongestValidSuffix(generated)` to determine the current grammar state, then masks all tokens except `parser.ValidNextTokens(suffix)`. Chooses from the survivors. Requires `stepsLeft >= 1` and that the derived suffix is not already complete. Ensures: `next` is a valid continuation of the grammar state derived from the suffix, `parser.IsValidPrefix(LongestValidSuffix(generated) + [next])`. |
| `SoftConstrainedStep(prompt, generated, penalty, stepsLeft)` | `(Prefix, Prefix, Logit, int) → (Token, int)` | Like `ConstrainedStep`, but instead of masking invalid tokens, applies a negative bias to invalid tokens before choosing. The LM can still select a grammar-invalid token if its raw preference is strong enough to overcome the penalty. Requires `stepsLeft >= 1`, `penalty > 0.0`, and that the derived suffix is not already complete. The verified contract focuses on budget/invariant preservation; the sharper logit-shaping guarantee lives on `SoftConstrainToGrammar`, which operates after logits already exist. |
| `TopKConstrainedStep(prompt, generated, k, stepsLeft)` | `(Prefix, Prefix, int, int) → (Token, int)` | Generates logits, applies `TopKFilter(k)` to keep only the `k` highest-logit tokens, *then* intersects with grammar-valid tokens via `MaskTokensExcept`. This gives "confident *and* grammar-valid" token selection. Requires `stepsLeft >= 1`, `1 <= k <= |lm.Tokens|`, and that the derived suffix is not already complete. Ensures: `next` is grammar-valid. |
| `ForcedTokenStep(prompt, generated, token, stepsLeft)` | `(Prefix, Prefix, Token, int) → (Token, int)` | Skips LM generation entirely and returns `token` as the next token. Requires `token in lm.Tokens` and `stepsLeft >= 1`. Useful for emitting known structural tokens (delimiters, separators, keywords) without consuming an LM call. |
| `BudgetAwareStep(prompt, generated, stepsLeft, completionThreshold)` | `(Prefix, Prefix, int, int) → (Token, int)` | If `stepsLeft <= completionThreshold` and `parser.IsCompletePrefix(LongestValidSuffix(generated))` is false, switches to `ConstrainedStep` to force grammar compliance before the budget runs out. Otherwise delegates to `UnconstrainedStep`. Requires `stepsLeft >= 1` and `completionThreshold >= 1`. |

### Ergonomic State-Transforming Wrappers

These wrap the raw step helpers but return the updated prefix directly. They are meant to be easier for synthesis models to use because they avoid the repeated proof pattern of unpacking `(next_token, new_steps)`, appending `next_token`, and synchronizing the budget manually.

| Name | Signature | Function |
|------|-----------|----------|
| `AppendUnconstrainedStep(prompt, prefix, stepsLeft)` | `(Prefix, Prefix, int) → (Prefix, int)` | Wrapper around `UnconstrainedStep`. Appends the chosen token to `prefix` and returns `(prefix + [next], stepsLeft - 1)`. Preferred for free-form generation. |
| `AppendConstrainedStep(prompt, prefix, stepsLeft)` | `(Prefix, Prefix, int) → (Prefix, int)` | Wrapper around `ConstrainedStep`. Requires `CanConstrain(prefix)`. Appends one grammar-valid token to `prefix` and returns the updated prefix plus remaining budget. Preferred for constrained answer growth. |
| `AppendSoftConstrainedStep(prompt, prefix, penalty, stepsLeft)` | `(Prefix, Prefix, Logit, int) → (Prefix, int)` | Wrapper around `SoftConstrainedStep`. Requires `CanConstrain(prefix)` and `penalty > 0.0`. |
| `AppendTopKConstrainedStep(prompt, prefix, k, stepsLeft)` | `(Prefix, Prefix, int, int) → (Prefix, int)` | Wrapper around `TopKConstrainedStep`. Requires `CanConstrain(prefix)` and `1 <= k <= |lm.Tokens|`. |
| `AppendBudgetAwareStep(prompt, prefix, stepsLeft, completionThreshold)` | `(Prefix, Prefix, int, int) → (Prefix, int)` | Wrapper around `BudgetAwareStep`. Useful when the strategy wants updated-prefix ergonomics but still wants the budget-aware switching policy. |
| `AppendForcedToken(prefix, token, stepsLeft)` | `(Prefix, Token, int) → (Prefix, int)` | Wrapper around `ForcedTokenStep`. Appends the forced token directly and returns the updated prefix and remaining budget. |
| `AppendLeftDelimiter(prefix, stepsLeft)` | `(Prefix, int) → (Prefix, int)` | Specialized wrapper for appending `LeftDelimiter`. Preferred over spelling out `AppendForcedToken(prefix, LeftDelimiter, stepsLeft)`. |
| `AppendRightDelimiter(prefix, stepsLeft)` | `(Prefix, int) → (Prefix, int)` | Specialized wrapper for appending `RightDelimiter`. Preferred over spelling out `AppendForcedToken(prefix, RightDelimiter, stepsLeft)`. |

### Repair and Salvage

Functions for recovering from invalid or dead-end prefixes.

| Name | Signature | Function |
|------|-----------|----------|
| `RollbackToValidPrefix(generated)` | `(Prefix) → Prefix` | Trims tokens from the end of `generated` until the remaining prefix is parser-valid and not dead. Requires `parser.IsValidPrefix([])`. Ensures: `parser.IsValidPrefix(result)` and `|result| <= |generated|`. |
| `FindLongestValidSpan(generated)` | `(Prefix) → Prefix` | Scans `generated` for the longest contiguous subsequence (not necessarily a suffix) that is a valid parser prefix. Returns that span. Ensures: `parser.IsValidPrefix(result)` and `|result| <= |generated|` and every token in `result` appears in `generated`. This is strictly more powerful than `RollbackToValidPrefix` because it can salvage valid content from the middle of a corrupted sequence. |
| `ExtractAllValidSpans(generated)` | `(Prefix) → list[Prefix]` | Returns all maximal contiguous substrings of `generated` that are valid parser prefixes, ordered by their starting position. Each span is maximal in the sense that it cannot be extended left or right while remaining valid. Ensures: every returned span satisfies `parser.IsValidPrefix(span)` and `|span| > 0`. |
| `RepairByRetry(prompt, generated, maxRetries, stepsLeft)` | `(Prefix, Prefix, int, int) → (Prefix, int)` | Rolls back to the longest valid suffix, then takes up to `maxRetries` constrained steps to try to extend past the point of failure. Returns the repaired prefix and remaining steps. Requires `maxRetries >= 1` and `stepsLeft >= maxRetries`. Ensures `parser.IsValidPrefix(LongestValidSuffix(result))`. |

### Checkpoint Stack

Enables branching/backtracking strategies that save and restore known-good states.

| Name | Signature | Function |
|------|-----------|----------|
| `CheckpointStack.__init__()` | `() → CheckpointStack` | Creates an empty checkpoint stack. Ensures `Depth() == 0`. |
| `CheckpointStack.Push(prefix)` | `(Prefix) → None` | Pushes `prefix` onto the stack. Requires `parser.IsValidPrefix(LongestValidSuffix(prefix))`. Ensures `Depth() == old(Depth()) + 1` and `Peek() == prefix`. |
| `CheckpointStack.Pop()` | `() → Prefix` | Removes and returns the top prefix. Requires `Depth() > 0`. Ensures `Depth() == old(Depth()) - 1`. |
| `CheckpointStack.Peek()` | `() → Prefix` | Returns the top prefix without removing it. Requires `Depth() > 0`. |
| `CheckpointStack.Depth()` | `() → int` | Returns the number of saved checkpoints. Ensures `result >= 0`. |
| `CheckpointStack.IsEmpty()` | `() → bool` | Returns `Depth() == 0`. |

### Repetition Tracking

Tracks n-gram frequencies in generated output to enable repetition-aware strategies.

| Name | Signature | Function |
|------|-----------|----------|
| `RepetitionTracker.__init__(ngramSize)` | `(int) → RepetitionTracker` | Creates a tracker for n-grams of the given size. Requires `ngramSize >= 1`. |
| `RepetitionTracker.RecordToken(token)` | `(Token) → None` | Appends `token` to the internal buffer and updates n-gram frequency counts. |
| `RepetitionTracker.GetCount(ngram)` | `(Prefix) → int` | Returns how many times `ngram` has been observed. Requires `|ngram| == ngramSize`. Ensures `result >= 0`. |
| `RepetitionTracker.GetRepetitionPenalty(token)` | `(Token) → Logit` | Returns a non-negative penalty value proportional to how often the n-gram ending with `token` has appeared. Ensures `result >= 0.0`. The penalty can be fed directly into `BiasToken(token, -penalty)`. |
| `RepetitionTracker.ApplyRepetitionPenalties(lm)` | `(LM) → None` | For every token in the vocabulary, computes `GetRepetitionPenalty(token)` and applies `lm.BiasToken(token, -penalty)`. Convenience method that composes tracking with logit shaping in one call. |

### Token Budget Utilities

Help strategies make budget-aware decisions.

| Name | Signature | Function |
|------|-----------|----------|
| `HasBudget(stepsLeft, needed)` | `(int, int) → bool` | Returns `stepsLeft >= needed`. Pure convenience predicate for readable strategy conditions. |
| `MinStepsToComplete(prefix)` | `(Prefix) → int` | Wrapper around `parser.ParserDistanceToComplete(LongestValidSuffix(prefix))`. Returns a lower bound on the steps needed to finish the grammar from the current state. |

### Composite Step Helpers

Higher-level operations that compose multiple primitives for common patterns.

| Name | Signature | Function |
|------|-----------|----------|
| `SoftConstrainToGrammar(prefix, penalty)` | `(Prefix, Logit) → None` | Computes `LongestValidSuffix(prefix)`, gets `parser.ValidNextTokens(suffix)`, and applies `lm.BiasTokens(invalidTokens, -penalty)` for every vocabulary token not in the valid set. Grammar-valid tokens are untouched. Requires `penalty > 0.0`. If the derived suffix is already complete, this helper is a no-op. This is the fundamental "reduce without zeroing" operation. |
| `IntersectWithGrammar(prefix)` | `(Prefix) → None` | Computes `LongestValidSuffix(prefix)`, gets `parser.ValidNextTokens(suffix)`, and calls `lm.MaskTokensExcept(validTokens)`. Hard grammar masking without any delimiter or window management. If the derived suffix is already complete, this helper is a no-op instead of masking everything. |
| `BiasForCompletion(prefix, bonus)` | `(Prefix, Logit) → None` | If `parser.IsCompletePrefix(LongestValidSuffix(prefix))` is reachable within one token (i.e., there exists a token `t` in `ValidNextTokens` such that appending `t` yields a complete prefix), biases those completion tokens by `+bonus`. Encourages the LM to finish when the grammar allows it. |

---

## Design Principles

### Composability

Every logit-shaping operation is designed to compose with every other:

- **Additive biases compose**: two `BiasToken` calls sum their deltas.
- **Multiplicative scales compose**: two `ScaleToken` calls multiply their factors.
- **Hard masking is a floor**: once masked, a token stays masked regardless of subsequent bias/scale operations (since `-1e9 + delta` is still effectively `-1e9` for reasonable `delta` values, and `ClampLogits` respects the lower bound).
- **`IntersectWithGrammar` then `TopKFilter`** gives "grammar-valid top-k" — a combined policy that is strictly more selective than either alone.
- **`SoftConstrainToGrammar` then `TopKFilter`** gives "grammar-biased top-k" — top-k selection where grammar-invalid tokens have been penalized but not eliminated.

### Suffix-Based Grammar Alignment

Instead of explicit delimiter windows that assume CRANE-style interleaved constrained/unconstrained regions, `LongestValidSuffix` lets any strategy apply grammar awareness at any point. The strategy author simply calls a constrained step and the helper determines the grammar state from the suffix. This means:

- No delimiter bookkeeping in the strategy body.
- Grammar constraints can be applied intermittently, gradually, or continuously.
- The same helper works whether the strategy is doing free-form generation, structured output, or a hybrid.

### Separation of Concerns

- **Logit shaping** (bias, scale, mask, clamp, top-k) lives on `LM` — it is grammar-agnostic.
- **Grammar queries** (valid, complete, dead, next tokens, distance, subgrammar) live on `Parser` — they are LM-agnostic.
- **Composed operations** (constrained steps, soft constraints, repair) live on `CSDHelpers` — they bridge the two.
- **State structures** (checkpoint stack, repetition tracker) are standalone — strategies opt into them.
- **Token selection policy** (argmax, sampling, temperature) is external — not in scope for this library.

---

## Notes

- Methods marked as abstract/`extern` are interface points or proof placeholders rather than fully implemented runtime logic.
- Methods named as lemmas primarily exist to support reasoning about invariants and transitions in constrained decoding; they are proof artifacts, not runtime operations.
- The runtime decoding flow is centered around the step functions (`UnconstrainedStep`, `ConstrainedStep`, `SoftConstrainedStep`, `TopKConstrainedStep`, `ForcedTokenStep`, `BudgetAwareStep`), the append-style wrappers (`AppendUnconstrainedStep`, `AppendConstrainedStep`, `AppendForcedToken`, `AppendLeftDelimiter`, `AppendRightDelimiter`, etc.), the repair utilities (`RollbackToValidPrefix`, `FindLongestValidSpan`, `RepairByRetry`), and the logit-shaping primitives on `LM`.
- `CheckpointStack` and `RepetitionTracker` are opt-in state structures. A strategy that doesn't need backtracking or repetition awareness simply doesn't instantiate them.
- All logit values are clamped to `[-1e9, 1e9]`. The sentinel value `-1e9` means "masked / unselectable."
