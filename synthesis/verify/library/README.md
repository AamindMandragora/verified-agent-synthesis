# Verify Library (Dafny)

This directory contains the Dafny files that define the formal substrate for synthesized strategies.
These files are the foundation for both verification and runtime compilation.

## Files

- Verified baseline-style examples live under `../reference/` (`crane.dfy`, `itergen.dfy`, `cars.dfy`); see `../reference/README.md`. Copy `MyCSDStrategy` into `GeneratedCSD.dfy` to run them in the pipeline.
- `GeneratedCSD.dfy`
  - Template file containing insertion markers where generated strategy logic is placed.
  - Treated as a reusable scaffold; synthesis should not permanently overwrite template semantics.
- `VerifiedAgentSynthesis.dfy`
  - Module `VerifiedDecoderAgent`: types, `LM`, `Parser`, and `CSDHelpers` used by generated strategies.
  - Per-member summaries below; pre/postconditions in the `.dfy` file are authoritative.

## `VerifiedAgentSynthesis.dfy` — member summaries

Types: **`Token`** is `string`; **`Prefix`** is `seq<Token>`; **`Id`** is `nat`; **`Logit`** is `real`.

### `LM` (extern-backed language model)

- **`Tokens` / `Ids` / `Logits`** — Immutable vocabulary and id sequences plus mutable per-step logit array (lengths tied by `ValidTokensIdsLogits`).
- **`ValidTokensIdsLogits`** — Predicate: vocabulary, ids, and logits lengths align, ids are 0..n-1, tokens are unique, every token has a logit in a fixed finite range.
- **Constructor** — Establishes `ValidTokensIdsLogits()` for a fresh model instance.
- **`IdToToken` / `TokenToId` / `TokenToIdRecursive`** — Bijective lookup between ids and vocabulary tokens (`Recursive` is the spec implementation helper).
- **`IdToLogit` / `TokenToLogit` / `TokensToLogits` / `IdsToLogits`** — Read current logits for ids or tokens (sequences require non-empty input and in-vocabulary members).
- **`MaskToken` / `MaskTokens` / `MaskTokensExcept`** — Set selected logits to the hard-mask sentinel so those tokens cannot be chosen.
- **`IsMasked` / `HasUnmaskedToken`** — Query whether a token is masked or any vocabulary choice remains unmasked.
- **`GenerateLogits`** — Recompute next-step logits from the given prefix (extern).
- **`AppendTaskGuidance`** — Append a CSD-authored evaluator prompt guidance
  block in the runtime wrapper; first non-empty call wins and the Dafny cost is
  unchanged. The guidance is additive and must not contradict, weaken, or
  replace earlier task instructions, examples, schema, or output-format
  requirements.
- **`ChooseNextToken`** — Sample/argmax one **unmasked** token (extern).
- **`ChooseNextTokenUnconstrained`** — Sample from the full vocabulary without respecting masks (extern).
- **`GenerateUnconstrainedChunk`** — Emit up to `maxNewTokens` tokens without grammar masking, stopping on open-span or EOS (extern).
- **`MaskValidNextAndEos`** — Hard-mask everything that is neither grammar-valid next nor EOS (extern; may use DFA masks).
- **`BoostValidNextAndEos`** — Add a non-negative bump to logits of grammar-valid next tokens and EOS (extern).

### `Parser` (extern-backed grammar)

- **`IsValidPrefix` / `IsCompletePrefix`** — Prefix is syntactically valid or complete under the grammar.
- **`ValidNextTokenCountUpTo` / `ValidNextToken` / `ValidNextTokens`** — Count (exactly, stopping at a cap) or enumerate admissible next tokens at a valid prefix.
- **`IsDeadPrefix`** — Valid but incomplete prefix with no legal continuations.
- **`ParseG`** — Run the grammar on a raw string and report success (extern).

### Module-level

- **`Contains(s, sub)`** — True iff `sub` occurs as a contiguous substring of `s`.

### `CSDHelpers` (verified strategy API)

Instance field **`cost`** — Accumulated token-step budget; the constructor sets it to 0.

**Unconstrained and chunking**

- **`AppendTaskGuidance`** — Start-of-CSD prompt policy hook; appends guidance
  before generation begins, keeps `cost` unchanged, and should not be used as a
  mid-generation control action. The guidance must not contradict, weaken, or
  replace earlier task instructions, examples, schema, or output-format
  requirements.
- **`UnconstrainedStep`** — One full-vocabulary step: forward pass + `ChooseNextTokenUnconstrained`; increments `cost` by 1.
- **`UnconstrainedChunk`** — One call that extends output with a short unconstrained continuation (EOS/open-span semantics per contract); increments `cost` by `stepsUsed`.

**Constrained span bookkeeping**

- **`OpenConstrainedSpan`** — Appends `"<<"`, enters constrained mode, resets active constrained suffix; +1 cost.
- **`EnterObservedConstrainedSpan`** — Same state as open-span but without appending (delimiter already in `generated`); cost unchanged.
- **`AppendConstrainedToken`** — Append a parser-valid token to both full output and active constrained suffix; cost unchanged.
- **`CloseConstrainedSpan`** — Close a complete constrained span (append `">>"` if needed), exit constrained mode; +1 cost.

**Core constrained decoding**

- **`ConstrainedStep`** — Forward pass, mask to valid-next ∪ EOS, choose unmasked token; +1 cost.
- **`ConstrainedSymbol`** — Unconstrained chunk with `"<<"` chunk delimiter, then keep longest parser-valid prefix of the chunk; `cost` += `stepsUsed`.
- **`ConstrainedSymbolInGenerated`** — Same as `ConstrainedSymbol` but also rebuilds full `generated` from stable prefix + new constrained suffix.

**Group boosts and penalties**

- **`GroupHasValidMember`** — Whether any token in a group is a valid next token at the prefix (no LM call).
- **`BoostValidGroups`** — For each group with a valid member, boost all tokens in that group (safe via intersection for vocab).
- **`GroupBoostedConstrainedStep`** — Forward pass, optional group boosts, then hard mask and choose; +1 cost.
- **`AdaptiveConstrainedStep`** — Like group-boosted step but boosts only when at most `narrowThreshold` tokens can come next; +1 cost.
- **`AdaptiveConstrainedStepWithPenalties`** — Adaptive boosts plus safe penalties before hard mask; +1 cost.
- **`PenalizedConstrainedStep` / `BoostedConstrainedStep`** — Single-step constrained decode after penalizing or boosting an explicit token list (callers must prove tokens ∈ `lm.Tokens`); +1 cost.
- **`SafeBoostedConstrainedStep` / `SafePenalizedConstrainedStep`** — Same with non-vocabulary tokens ignored; +1 cost.
- **`SafeBoostTokenLogits` / `SafePenalizeTokenLogits`** — Filter token list to vocabulary, then add/subtract logit deltas; no LM call, `cost` unchanged.
- **`BoostTokenLogits` / `PenalizeTokenLogits`** — Same without filtering (requires membership proofs); no LM call.

**Soft and hybrid steps**

- **`SoftConstrainedStep`** — Boost valid-next and EOS, unconstrained sample, report whether result kept prefix valid; +1 cost.
- **`SafeSoftConstrainedStep`** — Soft sample first; if invalid, fall back to hard-masked constrained choice; +1 cost.
- **`ConfidenceGatedStep`** — Use top logit token if EOS or parser-valid; else hard-mask and choose; +1 cost.

**Logit inspection and shaping (no extra forward pass unless noted)**

- **`GetHighestLogitToken`** — Argmax over vocabulary on current logits.
- **`GetLogitGap`** — Difference between top two **unmasked** logits (0 if fewer than two).
- **`GetTopKTokens`** — K highest-logit distinct tokens, lower index tie-break; `cost` unchanged.
- **`GetTokenLogit`** — Logit for one vocabulary token.
- **`MaskTokensInPrefix`** — Hard-mask every vocab token that appears in a prefix sequence; `cost` unchanged.
- **`ScaleAllLogits`** — Multiply all logits by a positive scalar with clamping; no LM call.

**Parser metrics and candidates**

- **`DeadEndDetection`** — True iff valid-next count is strictly below a threshold.
- **`ValidTokenCount`** — Returns the exact count of next tokens for a prefix, stopping at 64.
- **`IsTokenValidNext`** — Boolean `ValidNextToken` for one token.
- **`TopValidCandidates`** — One forward pass, then up to K highest-logit tokens from valid-next ∪ EOS; +1 cost.

**Rollback and repair**

- **`RollbackToValidPrefix`** — Trim suffix tokens until the prefix is valid and not dead (static).
- **`RollbackConstrainedSpan`** — Roll back only the constrained part given explicit stable prefix equality.
- **`RollbackConstrainedSuffix`** — Roll back constrained suffix using length split from full `generated`.

**Token-set utilities (mostly static)**

- **`ExtractAfterKeyword`** — Tokens immediately following each occurrence of a keyword in a prefix.
- **`IntersectTokenSets` / `SubtractTokenSets`** — Sequence-based set intersection / difference.
- **`FlattenTokenGroups` / `GroupContaining`** — Flatten nested token groups or find index of group containing a token.
- **`LastTokenBefore`** — Token before the last occurrence of a separator in a prefix.

**String and prefix helpers**

- **`PrefixToString`** — Concatenate prefix tokens into one string.
- **`ExtractContentBetweenDelimiters`** — Extern-specified extraction of substring between delimiters.
- **`CountSubstring`** — Count non-overlapping occurrences in a string.
- **`CountTokenOccurrences` / `OccurrencesInRange`** — Count token occurrences in a prefix (function + inductive helper).
- **`TokensSinceLastOccurrence`** — Tokens from end back to last occurrence of a target (or full length if absent).

**Repetition, temperature, rollout**

- **`RepetitionPenaltyStep` / `SafeRepetitionPenaltyStep`** — Penalize logits for tokens already in a history prefix, then constrained step; +1 cost.
- **`TemperatureConstrainedStep` / `SafeTemperatureConstrainedStep`** — Scale logits by 1/temperature (clamped), then constrained step; +1 cost.
- **`RolloutConstrainedWithPenalties`** — Loop of safe penalized constrained steps until budget, completion, or EOS; `cost` += `stepsUsed`.
- **`SpeculativeConstrainedRollout`** — Run up to N `ConstrainedStep`s from a snapshot, restore logits after; `cost` includes speculative steps.

**Snapshots**

- **`SaveLogitsSnapshot` / `RestoreLogitsSnapshot`** — Copy logits to/from a sequence for branching or speculation; `cost` unchanged on restore path aside from inner helpers.

**Specification-only lemmas (axioms, not runtime calls)**

- **`ConstrainedStepNextValid`**, **`RollbackPreservesTokenInvariant`**, **`UnchosenIndexExists`**, **`DistinctChosenSeq`** — Proof support for vocabulary/parser invariants and `GetTopKTokens`.

## Role in the Pipeline

1. Generation creates a candidate strategy body.
2. The body is injected into the template contract context.
3. Dafny verifies the assembled program.
4. Verified code is compiled to Python for evaluation.

## Editing Guidance

- Changes here can invalidate synthesis assumptions and proof obligations.
- Treat edits as high-impact; update `synthesis/generate/prompts.py` and this README when the strategy-facing API changes.
- Keep method signatures and contract semantics stable unless intentionally evolving the synthesis interface.
