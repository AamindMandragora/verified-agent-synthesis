"""
Prompt templates for Qwen-based CSD strategy generation.

This project synthesizes *constrained decoding strategies* (CSDs), not the final
task output itself. The LLM should choose an appropriate verified strategy
primitive and parameters based on the *use-case* described by the task.

The generator expects these entrypoints:
- build_initial_prompt(task_description)
- build_verification_error_prompt(task_description, previous_strategy, error_message)
- build_runtime_error_prompt(previous_strategy, error_traceback)
- build_compilation_error_prompt(previous_strategy, error_message)
- build_format_repair_prompt(previous_strategy)
"""

import re

# NOTE:
# The synthesized output is injected into
# `synthesis/verify/library/GeneratedCSD.dfy` as the BODY
# of method `MyCSDStrategy(...)`.
#
# The output is a multi-line Dafny method body.
# It receives the full generated prefix so far plus explicit state for the
# currently active constrained segment, if any.


SYSTEM_PROMPT = """\
You are generating a *constrained decoding strategy* (CSD) that composes verified tools to produce valid output.

You must output ONLY the Dafny method body for:

  method MyCSDStrategy(
    lm: LM,
    parser: Parser,
    prompt: Prefix,
    generatedPrefix: Prefix,
    insideConstrained: bool,
    currentConstrained: Prefix,
    maxSteps: nat,
    stepTokenBudget: nat,
    validTokenGroups: seq<seq<Token>>,
    eosToken: Token
  ) returns (
    generated: Prefix,
    insideConstrainedOut: bool,
    currentConstrainedOut: Prefix,
    cost: int
  )
    modifies lm.Logits
    requires lm.ValidTokensIdsLogits()
    requires parser.IsValidPrefix([])
    requires !insideConstrained ==> currentConstrained == []
    requires insideConstrained ==> parser.IsValidPrefix(currentConstrained)
    requires insideConstrained ==> |currentConstrained| <= |generatedPrefix|
    requires eosToken in lm.Tokens
    ensures lm.ValidTokensIdsLogits()
    ensures |generated| <= |generatedPrefix| + maxSteps
    ensures !insideConstrainedOut ==> currentConstrainedOut == []
    ensures insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)
    ensures cost <= maxSteps
    ensures maxSteps == 0 || cost > 0 || generated != generatedPrefix ||
            insideConstrainedOut != insideConstrained ||
            currentConstrainedOut != currentConstrained
    decreases maxSteps

## Output rules
- Start with: `// CSD_RATIONALE_BEGIN\n// ...\n// CSD_RATIONALE_END`
- Immediately after the rationale, emit a proof sketch block:
  `// CSD_PROOF_SKETCH_BEGIN\n// ...\n// CSD_PROOF_SKETCH_END`.
- Initialize all out-parameters before any loop/recursion.
- Assign `cost` before returning.
- Assign the existing out-parameters directly; they are already in scope.
- Use the provided `helpers` instance (type `CSDHelpers`).
- Call static entries listed as `CSDHelpers.<Method>` with that qualifier.
- Call instance helper methods as `helpers.<Method>`.

## API Guidance

- `Token` is type `string`.
- `prompt` is type `Prefix` (= `seq<Token>`).
- `generated` / `generatedPrefix` contain the full answer text, including delimiter tokens.
- `currentConstrained` / `currentConstrainedOut` track only the active constrained segment contents between delimiters.
- EOS is terminal.
- Visible delimiters such as `"<<"` and `">>"` are task-contract artifacts.
  Use visible delimiters only when the task or evaluator requires visible constrained spans.
  For hidden constrained chunks, fully constrained objects, or another structured-output surface,
  emit the task-native surface.

## Available Tools

### Runtime inputs
```
validTokenGroups: seq<seq<Token>>
```
`validTokenGroups` is caller-supplied contextual vocabulary. It may be empty,
inner groups may be empty, groups may overlap, and groups may contain tokens
outside `lm.Tokens`; safe helpers internally ignore non-vocabulary tokens.

### Helper methods
```
helpers.AppendTaskGuidance(lm, guidance);
var next := helpers.UnconstrainedStep(lm, prompt, generated);
var generated, insideConstrainedOut, currentConstrainedOut := helpers.OpenConstrainedSpan(lm, generated);
var generated, insideConstrainedOut, currentConstrainedOut := helpers.EnterObservedConstrainedSpan(lm, generated);
var generated, insideConstrainedOut, currentConstrainedOut := helpers.AppendConstrainedToken(lm, parser, generated, currentConstrained, next);
var generated, insideConstrainedOut, currentConstrainedOut := helpers.CloseConstrainedSpan(lm, parser, generated, currentConstrained);
var generated, insideConstrainedOut, currentConstrainedOut, closed := helpers.CloseSpanIfComplete(lm, parser, generated, currentConstrained);
var next := helpers.ConstrainedStep(lm, parser, prompt, currentConstrained, eosToken);
var next, wasConstrained := helpers.ConfidenceGatedStep(lm, parser, prompt, currentConstrained, eosToken);
helpers.SafeBoostTokenLogits(lm, tokens, amount);
helpers.SafePenalizeTokenLogits(lm, tokens, amount);
var next := helpers.SafeBoostedConstrainedStep(lm, parser, prompt, currentConstrained, tokensToBoost, 4.0, eosToken);
var next := helpers.SafePenalizedConstrainedStep(lm, parser, prompt, currentConstrained, tokensToPenalize, 4.0, eosToken);
var next := helpers.SafeRepetitionPenaltyStep(lm, parser, prompt, currentConstrained, generated, 2.0, eosToken);
var next := helpers.SafeTemperatureConstrainedStep(lm, parser, prompt, currentConstrained, 0.8, eosToken);
var next, usedFallback := helpers.SafeSoftConstrainedStep(lm, parser, prompt, currentConstrained, 8.0, eosToken);
var next := helpers.GroupBoostedConstrainedStep(lm, parser, prompt, currentConstrained, validTokenGroups, 4.0, eosToken);
helpers.BoostValidGroups(lm, parser, currentConstrained, groups, amount);
var next := helpers.AdaptiveConstrainedStep(lm, parser, prompt, currentConstrained, validTokenGroups, 4.0, 12, eosToken);
var nextPen := helpers.AdaptiveConstrainedStepWithPenalties(lm, parser, prompt, currentConstrained, validTokenGroups, 4.0, penaltyTokens, 4.0, 12, eosToken);
var gap := helpers.GetLogitGap(lm);
var topK := helpers.GetTopKTokens(lm, k);
helpers.MaskTokensInPrefix(lm, generated);
var snap := helpers.SaveLogitsSnapshot(lm);
helpers.RestoreLogitsSnapshot(lm, snap);
var candTok, candPre, hitComplete, hitEos, stepsUsed := helpers.SpeculativeConstrainedRollout(lm, parser, prompt, currentConstrained, numSpecSteps, eosToken);
var generatedOut, stoppedOnOpenSpan, stoppedOnEos, stepsUsed := helpers.UnconstrainedChunk(lm, prompt, generated, maxChunkTokens, openSpanToken, eosToken);
var currentOut, hitEos, stepsUsed := helpers.ConstrainedSymbol(lm, parser, constrainedPrompt, currentConstrained, stepTokenBudget, eosToken);
var generatedOut, currentOut, hitEos, stepsUsed := helpers.ConstrainedSymbolInGenerated(lm, parser, constrainedPrompt, generated, currentConstrained, stepTokenBudget, eosToken);
var nextSoft, softOk := helpers.SoftConstrainedStep(lm, parser, prompt, currentConstrained, boostAmount, eosToken);
var nextPenRaw := helpers.PenalizedConstrainedStep(lm, parser, prompt, currentConstrained, tokensToPenalize, penaltyAmount, eosToken);
var nextBoostRaw := helpers.BoostedConstrainedStep(lm, parser, prompt, currentConstrained, tokensToBoost, boostAmount, eosToken);
var nextRep := helpers.RepetitionPenaltyStep(lm, parser, prompt, currentConstrained, generated, penaltyAmount, eosToken);
var nextTemp := helpers.TemperatureConstrainedStep(lm, parser, prompt, currentConstrained, temperature, eosToken);
var rolloutGen, rolloutSteps, rolloutEos := helpers.RolloutConstrainedWithPenalties(lm, parser, prompt, startPrefix, budget, penalties, penaltyAmount, eosToken);
var freeGenerated := helpers.UnconstrainedGeneration(lm, prompt, maxSteps);
var constrainedGenerated, terminatedByEos := helpers.ConstrainedGeneration(lm, parser, prompt, maxSteps, eosToken);
var craneGenerated := helpers.CraneGeneration(lm, parser, prompt, maxSteps, minReasoningSteps, eosToken);
var topTok := helpers.GetHighestLogitToken(lm);
var logitOne := helpers.GetTokenLogit(lm, token);
helpers.ScaleAllLogits(lm, scalar);
helpers.BoostTokenLogits(lm, tokensInVocab, amount);
helpers.PenalizeTokenLogits(lm, tokensInVocab, amount);
var subCount := CSDHelpers.CountSubstring(text, sub);
var s := CSDHelpers.PrefixToString(prefix);
var between := CSDHelpers.ExtractContentBetweenDelimiters(text, startDelim, endDelim);
var seenInPrompt := helpers.PrefixAppearsInPrompt(lm, prefix);
var promptResemblance: real := helpers.PrefixResemblesPromptExamples(lm, prefix);
var anyInGroup := helpers.GroupHasValidMember(parser, prefix, group);
var rolledGen, rolledCurrent := helpers.RollbackConstrainedSpan(parser, stablePrefix, generated, currentConstrained);
```
`OpenConstrainedSpan` appends a new `"<<"` token and costs 1 step. If `"<<"`
was already emitted by `UnconstrainedStep` or `UnconstrainedChunk`, use
`EnterObservedConstrainedSpan` to update span state without appending another
delimiter or consuming additional token budget.

### Parser queries
```
var narrow := helpers.DeadEndDetection(parser, currentConstrained, minValidCount);
var count := helpers.ValidTokenCount(parser, currentConstrained);
var valid := helpers.IsTokenValidNext(parser, currentConstrained, token);
var candidates := helpers.TopValidCandidates(lm, parser, prompt, currentConstrained, maxCandidates, eosToken);
var rolled := CSDHelpers.RollbackToValidPrefix(parser, constrainedPrefix);
var rolled := CSDHelpers.RollbackToCompletePrefix(parser, constrainedPrefix);
var generatedOut, currentOut := helpers.RollbackConstrainedSpan(parser, stablePrefix, generated, currentConstrained);
var generatedOut, currentOut := helpers.RollbackConstrainedSuffix(parser, generated, currentConstrained);
var generatedOut, currentOut := helpers.RollbackConstrainedToComplete(parser, generated, currentConstrained);
var generatedOut, currentOut := helpers.RollbackAndContinue(lm, parser, prompt, generated, currentConstrained, eosToken, maxSteps, closeReserve, maxRetries);
var flat := CSDHelpers.FlattenTokenGroups(validTokenGroups);
var groupIdx := CSDHelpers.GroupContaining(validTokenGroups, token);
var prevTok, foundPrev := helpers.LastTokenBefore(generated, ">>");
var occ := CSDHelpers.CountTokenOccurrences(generated, tok);
var occPrefix := CSDHelpers.OccurrencesInRange(generated, tok, hi);
var since := CSDHelpers.TokensSinceLastOccurrence(generated, tok);
var following := CSDHelpers.ExtractAfterKeyword(prefix, keyword);
var intersection := CSDHelpers.IntersectTokenSets(a, b);
var difference := CSDHelpers.SubtractTokenSets(a, b);
```

## Tool API Reference

Use this as API documentation for what each helper accomplishes in a constrained
decoding strategy. These descriptions define library behavior, not task-specific
revision rules.

`cost` is token-step budget use, not wall-clock time and not a count of helper
calls. LM-generated tokens count even if the token is EOS or later rejected from
visible output. A delimiter token appended by a helper counts as one token-step.
Bookkeeping, parser queries, token-set transforms, and logit edits do not
consume token budget by themselves.

### Task prompt guidance

- `helpers.AppendTaskGuidance(lm, guidance)`
  Role: split ownership. The constrained decode owns the answer's form — which
  spans exist, where they sit, what surface syntax is legal — and the parser
  reads that form back. Neither owns meaning: what the task's language refers to
  and the conventions it obeys. No grammar can encode that, so this is the only
  channel that can carry it.

  The call is first-call-wins and the string it takes rides every later attempt,
  so treat the first emission as final rather than as a draft to repair.

  Compatibility: this guidance is additive. It must not contradict, weaken, or
  replace earlier task instructions, examples, schema, or output-format
  requirements. If wording conflicts, the earlier task contract remains
  authoritative.

  Say what the task's language means wherever the grammar leaves meaning open —
  a named quantity, an operator or relation, or a convention holding across the
  whole problem. To give a sense of the shape: in a math word problem this might
  be that a quantity is a whole count rather than a fraction, or what "per"
  distributes over; in text-to-SQL, which schema object an ambiguous phrase
  picks out, or which direction "best" orders by; in a molecular string, which
  bond or valence a token stands for. These are illustrations, not a checklist —
  what a given task leaves open is yours to find.

  Do not restate delimiters, output format, or syntax; the parser already
  enforces those, so spending the sentence there changes nothing.
  Mechanics: forwards `guidance` to the runtime LM wrapper. The evaluator keeps
  its normal prompt, examples, schema, question, and output contract; the
  guidance is appended as an extra block. Runtime semantics are append-only and
  first-call-wins for the current CSD invocation; empty guidance is ignored.
  Cost: +0.
  Control profile: prompt policy only. Call it once at method start, after
  output initialization and before the first LM generation helper.

### Outside-span generation

- `helpers.UnconstrainedStep(lm, prompt, generated)`
  Role: one free LM token outside parser control.
  Mechanics: calls the LM on `prompt + generated` and returns one token; the
  strategy appends non-EOS tokens itself.
  Cost: +1 token-step, including EOS.
  Control profile: maximum free-LM continuation, no parser control.

- `helpers.UnconstrainedChunk(lm, prompt, generated, maxChunkTokens, "<<", eosToken)`
  Role: a short free-LM continuation that can stop when an opening delimiter is
  naturally emitted.
  Mechanics: returns `generatedOut` already extended by the non-EOS chunk. If
  `stoppedOnOpenSpan` is true, `generatedOut` already ends with `"<<"`.
  Cost: +`stepsUsed`; EOS counts in `stepsUsed` but is not appended.
  Control profile: free-LM continuation with delimiter observation.

- `helpers.UnconstrainedGeneration(lm, prompt, maxSteps)`
  Role: bounded free-LM generation helper.
  Mechanics: returns exactly `maxSteps` tokens sampled without parser control.
  Cost: +`maxSteps`.
  Control profile: full free generation with no EOS special case.

- `helpers.CraneGeneration(lm, parser, prompt, maxSteps, minReasoningSteps, eosToken)`
  Role: CRANE-style baseline generation with free text outside constrained spans
  and parser-aware decoding inside observed spans.
  Mechanics: emits free tokens until constrained-span state is reached, then
  uses confidence-gated parser control inside that span.
  Cost: at most +`maxSteps`.
  Control profile: free outer continuation plus parser fallback inside spans.

- `helpers.OpenConstrainedSpan(lm, generated)`
  Role: explicit transition from free generation into constrained generation.
  Mechanics: appends visible `"<<"`, sets `insideOut := true`, and resets
  `currentOut := []`.
  Cost: +1 token-step for the forced delimiter.
  Control profile: direct delimiter/state control, no LM sampling.

- `helpers.EnterObservedConstrainedSpan(lm, generated)`
  Role: state transition after `"<<"` is already present in visible output.
  Mechanics: leaves `generated` unchanged, sets `insideOut := true`, and resets
  `currentOut := []`.
  Cost: +0.
  Control profile: bookkeeping only.

### Inside-span generation and state updates

- `helpers.ConstrainedStep(lm, parser, prompt, currentConstrained, eosToken)`
  Role: one parser-valid token choice inside a constrained span.
  Mechanics: calls the LM, hard-masks to parser-valid next tokens plus EOS, and
  returns one token without appending it.
  Cost: +1 token-step, including EOS.
  Control profile: strongest token-level parser control.

- `helpers.ConfidenceGatedStep(lm, parser, prompt, currentConstrained, eosToken)`
  Role: one inside-span token choice for non-exact spans that uses hard parser
  control only when the LM's current top token would not preserve parser
  validity.
  Mechanics: calls the LM, reads the highest-logit token, returns it directly
  if it is EOS or keeps `currentConstrained + [token]` parser-valid; otherwise
  hard-masks to parser-valid next tokens plus EOS and samples from that mask.
  Returns `wasConstrained == true` only on the hard-mask fallback path.
  Cost: +1 token-step, including EOS.
  Control profile: LM-preferred continuation when already parser-compatible,
  hard parser fallback otherwise; avoid using this helper for exact visible
  spans that must remain fully hard-controlled.

- `helpers.AppendConstrainedToken(lm, parser, generated, currentConstrained, next)`
  Role: commit a previously selected parser-valid token into visible and
  constrained state.
  Mechanics: appends `next` to both `generated` and `currentConstrained`.
  Cost: +0; the token was already counted by the generation helper.
  Control profile: state synchronization only.

- `helpers.ConstrainedSymbol(lm, parser, constrainedPrompt, currentConstrained, maxSymbolTokens, eosToken)`
  Role: multi-token constrained progress from one LM chunk.
  Mechanics: generates up to `maxSymbolTokens`, then accepts only the longest
  parser-valid prefix of that chunk.
  Cost: +`stepsUsed`; rejected suffix tokens and EOS still count.
  Control profile: chunk-level LM continuation with parser-prefix acceptance.

- `helpers.ConstrainedSymbolInGenerated(lm, parser, constrainedPrompt, generated, currentConstrained, maxSymbolTokens, eosToken)`
  Role: multi-token constrained progress while updating full visible output.
  Mechanics: computes the stable prefix, calls `ConstrainedSymbol`, then returns
  `generatedOut := stablePrefix + currentOut`.
  Cost: +`stepsUsed`, which may exceed visible output growth.
  Control profile: chunk-level LM continuation plus generated/current state
  reconstruction.

- `helpers.ConstrainedGeneration(lm, parser, prompt, maxSteps, eosToken)`
  Role: bounded parser-valid generation from an empty constrained prefix.
  Mechanics: loops `ConstrainedStep` until parser completeness, EOS, or budget.
  Cost: +`|generated|`, plus one additional step when terminated by EOS.
  Control profile: full hard-parser generation.

- `helpers.CloseConstrainedSpan(lm, parser, generated, currentConstrained)`
  Role: exit a complete constrained span.
  Mechanics: requires `parser.IsCompletePrefix(currentConstrained)`, appends
  visible `">>"` unless already emitted by the constrained grammar, exits
  constrained mode, and clears `currentOut`.
  Cost: +1 token-step for the close action.
  Control profile: direct delimiter/state control gated by parser completeness.

- `helpers.CloseSpanIfComplete(lm, parser, generated, currentConstrained)`
  Role: close the constrained span only if it already holds a complete parse.
  Mechanics: checks `parser.IsCompletePrefix(currentConstrained)` internally; when
  complete it delegates to `CloseConstrainedSpan` (appends `">>"` unless already
  present, exits constrained mode, clears `currentOut`) and returns `closed == true`;
  when not yet complete it leaves state unchanged and returns `closed == false`.
  Cost: +1 token-step when it closes; +0 when it leaves the span open.
  Control profile: completeness-gated close that needs no caller-side proof of completeness; safe to call speculatively each step (no-op until the span parses) and branch on `closed`.

- `helpers.CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrained, eosToken, budget)`
  Role: bring an open constrained span to a completable state and close it within a step budget.
  Mechanics: generates forward inside the span (dead-end-aware), tracking the longest prefix that
  parses as complete; reserves one step and emits `">>"` at that longest complete point. If no
  completable state is reached within `budget`, the span is left open.
  Requires: `lm.ValidTokensIdsLogits()`, `parser.IsValidPrefix(currentConstrained)`,
  `|currentConstrained| <= |generated|`, `eosToken in lm.Tokens`, `">>" in lm.Tokens`.
  Returns: `(generatedOut, insideOut, currentOut)` with `!insideOut ==> currentOut == []`,
  `insideOut ==> parser.IsValidPrefix(currentOut)`, `|generatedOut| <= |generated| + budget`,
  `cost <= old(cost) + budget`, and `cost >= old(cost)`.
  Cost: ≤ `budget` token-steps.
  Control profile: budget-bounded completeness-tracking close that needs no caller-side proof of completeness.

- `helpers.ManagedStep(lm, parser, prompt, generated, insideConstrained, currentConstrained, validTokenGroups, boostAmount, narrowThreshold, eosToken)`
  Role: one self-contained free-or-constrained decode step with delimiter/state
  management built in.
  Mechanics: outside a span, takes one `UnconstrainedStep` and enters constrained
  mode if it emits `"<<"`; inside a span, closes when `currentConstrained` is
  complete, otherwise takes one `AdaptiveConstrainedStep` and appends it.
  Returns `(generatedOut, insideOut, currentOut, done)` where `done` means EOS or
  span close was reached.
  Requires: same parser-valid/current-state shape as the lower-level span helpers,
  `"<<", ">>", eosToken in lm.Tokens`, and bounded non-negative `boostAmount`.
  Cost: exactly +1 token-step on every path.
  Control profile: proof-friendly single-step state machine for strategies that
  need to manage spans without hand-writing delimiter and parser-state branches.

- `helpers.GenerateWithManagedSpan(lm, parser, prompt, generatedPrefix, insideConstrained, currentConstrained, maxSteps, validTokenGroups, boostAmount, narrowThreshold, eosToken)`
  Role: bounded full decode loop with free preamble, constrained span handling,
  completeness-gated close, and proof obligations discharged inside the helper.
  Mechanics: loops up to `maxSteps`; outside a span it samples freely until EOS
  or `"<<"`; inside a span it closes when complete, otherwise advances with
  `AdaptiveConstrainedStep`. The caller receives `(generated,
  insideConstrainedOut, currentConstrainedOut)` and does not need to prove a
  custom loop.
  Requires: same parser-valid/current-state shape as `ManagedStep`, `"<<", ">>",
  eosToken in lm.Tokens`, and bounded non-negative `boostAmount`.
  Cost: at most +`maxSteps`.
  Control profile: high-level span manager for cold strategies that are losing
  budget or correctness in hand-written free/inside-span loops.

- `helpers.GenerateWithPrefixAndManagedSpan(lm, parser, prompt, generatedPrefix, insideConstrained, currentConstrained, maxSteps, prefixBudget, validTokenGroups, boostAmount, narrowThreshold, eosToken)`
  Role: bounded full decode loop that hard-caps the unconstrained preamble before
  forcing entry into a constrained span.
  Mechanics: like `GenerateWithManagedSpan`, but while outside a span it allows
  at most `prefixBudget` free steps; after that it calls `OpenConstrainedSpan`
  so the remaining budget can be spent on parser-controlled content and closing.
  Requires: same requirements as `GenerateWithManagedSpan`, plus
  `prefixBudget <= maxSteps`.
  Cost: at most +`maxSteps`.
  Control profile: output-budget guard for failures where the model stays in
  free text too long or enters constrained mode too late.

### Soft preferences and group-aware constrained decoding

- `helpers.GroupBoostedConstrainedStep(lm, parser, prompt, currentConstrained, validTokenGroups, amount, eosToken)`
  Role: one parser-valid token choice with caller-supplied group preferences.
  Mechanics: calls the LM, boosts groups containing parser-valid members, then
  hard-masks to parser-valid next tokens plus EOS.
  Cost: +1 token-step, including EOS.
  Control profile: hard parser control with soft preference among legal choices.

- `helpers.BoostValidGroups(lm, parser, prefix, groups, amount)`
  Role: apply the same per-group soft boost used inside group-boosted steps, without sampling.
  Mechanics: for each `groups[i]` such that `GroupHasValidMember` is true at `prefix`, calls `SafeBoostTokenLogits(lm, groups[i], amount)` (non-vocabulary tokens in the group are ignored).
  Cost: +0 (no `GenerateLogits` / `ChooseNextToken` in this helper).
  Control profile: logit shaping only; call after `GenerateLogits` if logits must reflect the current prefix.

- `helpers.AdaptiveConstrainedStep(lm, parser, prompt, currentConstrained, validTokenGroups, amount, narrowThreshold, eosToken)`
  Role: one parser-valid token choice with group preferences applied only at
  narrower parser states.
  Mechanics: same hard mask as `ConstrainedStep`; group boosts are applied only
  when `parser.ValidNextTokenCount(currentConstrained) <= narrowThreshold`.
  Cost: +1 token-step, including EOS.
  Control profile: hard parser control with conditional soft preference.

- `helpers.AdaptiveConstrainedStepWithPenalties(lm, parser, prompt, currentConstrained, boostGroups, boostAmount, penaltyTokens, penaltyAmount, narrowThreshold, eosToken)`
  Role: same adaptive group boosts as `AdaptiveConstrainedStep`, plus safe token
  penalties before the hard mask.
  Mechanics: `GenerateLogits`, conditional `BoostValidGroups`, `SafePenalizeTokenLogits`,
  `MaskValidNextAndEos`, `ChooseNextToken`.
  Cost: +1 token-step, including EOS.
  Control profile: hard parser control with conditional boosts and penalties.

- `helpers.SafeBoostedConstrainedStep(lm, parser, prompt, currentConstrained, tokens, amount, eosToken)`
  Role: one parser-valid token choice with a soft boost for a caller-supplied
  token set.
  Mechanics: calls the LM, ignores any `tokens` outside `lm.Tokens`, boosts the
  remaining listed tokens, then hard-masks to parser-valid next tokens plus EOS.
  Cost: +1 token-step, including EOS.
  Control profile: hard parser control with token-list soft preference.

- `helpers.SafePenalizedConstrainedStep(lm, parser, prompt, currentConstrained, tokens, amount, eosToken)`
  Role: one parser-valid token choice with a soft penalty for a caller-supplied
  token set.
  Mechanics: calls the LM, ignores any `tokens` outside `lm.Tokens`, penalizes
  the remaining listed tokens, then hard-masks to parser-valid next tokens plus
  EOS.
  Cost: +1 token-step, including EOS.
  Control profile: hard parser control with token-list soft avoidance.

- `helpers.SafeRepetitionPenaltyStep(lm, parser, prompt, currentConstrained, generated, amount, eosToken)`
  Role: one parser-valid token choice that discourages reusing tokens already
  present in the visible output.
  Mechanics: calls the LM, filters `generated` through `lm.Tokens`, penalizes
  those tokens, then hard-masks to parser-valid next tokens plus EOS.
  Cost: +1 token-step, including EOS.
  Control profile: hard parser control with repetition avoidance.

- `helpers.SafeTemperatureConstrainedStep(lm, parser, prompt, currentConstrained, temperature, eosToken)`
  Role: one parser-valid token choice with local sampling sharpness adjusted.
  Mechanics: calls the LM, clamps `temperature` to a safe range, scales logits,
  then hard-masks to parser-valid next tokens plus EOS.
  Cost: +1 token-step, including EOS.
  Control profile: hard parser control with sharper or flatter token sampling.

- `helpers.SafeSoftConstrainedStep(lm, parser, prompt, currentConstrained, boostAmount, eosToken)`
  Role: one inside-span token choice that first tries a soft grammar preference,
  then falls back to hard parser control if the soft choice would not preserve
  parser validity.
  Mechanics: calls the LM, boosts parser-valid next tokens plus EOS, samples one
  soft choice, and returns it if it preserves parser validity; otherwise it
  hard-masks to parser-valid next tokens plus EOS and returns the fallback token.
  Returns `usedFallback == true` only when the hard fallback path is used.
  Cost: +1 token-step by helper contract, including EOS.
  Control profile: soft grammar preference with hard parser fallback.

- `helpers.SafeBoostTokenLogits(lm, tokens, amount)`
  Role: raise soft preference for a caller-supplied token set in the current logits.
  Mechanics: filters `tokens` through `lm.Tokens`, then adds to their existing
  logits. It does not call the LM, sample, append output, or inspect the parser.
  Cost: +0.
  Control profile: soft logit preference only; relevant only to later choices
  that read the modified logits rather than regenerating fresh logits first.

- `helpers.SafePenalizeTokenLogits(lm, tokens, amount)`
  Role: lower soft preference for a caller-supplied token set in the current logits.
  Mechanics: filters `tokens` through `lm.Tokens`, then subtracts from their
  existing logits. It does not call the LM, sample, append output, or inspect
  the parser.
  Cost: +0.
  Control profile: soft logit preference only; relevant only to later choices
  that read the modified logits rather than regenerating fresh logits first.

- `helpers.GetLogitGap(lm)`
  Role: measure spread between the top two **unmasked** vocabulary logits.
  Mechanics: single scan over `lm.Logits` with `lm.IsMasked`; returns `0.0` if
  fewer than two unmasked positions exist.
  Cost: +0.

- `helpers.GetTopKTokens(lm, k)`
  Role: return the `k` vocabulary tokens with highest current logits (no LM call).
  Mechanics: greedy index selection with lower-index tie-break; requires
  `1 <= k <= |lm.Tokens|`.
  Cost: +0.

- `helpers.MaskTokensInPrefix(lm, prefix)`
  Role: hard-mask every vocabulary token that appears anywhere in `prefix`.
  Mechanics: walks `prefix` and calls `lm.MaskToken` for in-vocabulary entries.
  Cost: +0.

- `helpers.SpeculativeConstrainedRollout(lm, parser, prompt, constrainedPrefix, numTokens, eosToken)`
  Role: run up to `numTokens` `ConstrainedStep` calls from `constrainedPrefix`, then
  restore logits from a snapshot so the caller can inspect the candidate without
  committing logits state (cost still includes the speculative forward steps).
  Mechanics: internal `SaveLogitsSnapshot` / `RestoreLogitsSnapshot`.
  Cost: +`stepsUsed` token-steps (`stepsUsed <= numTokens`).

### Parser queries, repair, and context extraction

- `helpers.ValidTokenCount(parser, currentConstrained)`
  Role: inspect parser branching at the current constrained prefix.
  Mechanics: returns the valid-next-token count; no LM call and no state change.
  Cost: +0.
  Control profile: parser information only.

- `helpers.DeadEndDetection(parser, currentConstrained, minValidCount)`
  Role: detect whether parser branching is below a caller-supplied threshold.
  Mechanics: returns a thresholded boolean from the valid-next-token count; no
  LM call and no state change.
  Cost: +0.
  Control profile: parser information only.

- `helpers.IsTokenValidNext(parser, currentConstrained, token)`
  Role: test one candidate token against the parser.
  Mechanics: returns whether `token` is valid next; no state change.
  Cost: +0.
  Control profile: parser information only.

- `helpers.TopValidCandidates(lm, parser, prompt, currentConstrained, maxCandidates, eosToken)`
  Role: inspect the LM's ranking among legal next tokens.
  Mechanics: calls the LM once and returns up to `maxCandidates` high-logit
  parser-valid candidates, with EOS admissible. It does not append a candidate.
  Cost: +1 token-step.
  Control profile: LM-ranked parser-valid candidate information.

- `helpers.RollbackConstrainedSuffix(parser, generated, currentConstrained)`
  Role: repair active constrained state by shortening the constrained suffix.
  Mechanics: computes the stable prefix from the current suffix length, rolls
  back only `currentConstrained` until parser-valid, and reconstructs
  `generatedOut`.
  Cost: +0.
  Control profile: parser repair by deletion.

- `CSDHelpers.RollbackToCompletePrefix(parser, prefix)` (static)
  Role: shorten `prefix` from the end until it is a complete parse or empty.
  Mechanics: drops trailing tokens while the prefix is not `IsCompletePrefix`;
  returns the longest complete prefix, or empty if none exists.
  Cost: +0.
  Control profile: parser repair by deletion.

- `helpers.RollbackConstrainedToComplete(parser, generated, currentConstrained)`
  Role: repair active constrained state by shortening the constrained suffix to
  a complete parse.
  Mechanics: computes the stable prefix from the current suffix length, rolls
  back only `currentConstrained` until `IsCompletePrefix` (or empty), and
  reconstructs `generatedOut`.
  Cost: +0.
  Control profile: parser repair by deletion.

- `helpers.RollbackAndContinue(lm, parser, prompt, generated, currentConstrained, eosToken, maxSteps, closeReserve, maxRetries)`
  Role: repair active constrained state by rolling the constrained suffix back to
  a complete parse and then regenerating forward from that point.
  Mechanics: computes the stable prefix from the current suffix length, rolls
  `currentConstrained` back to its last `IsCompletePrefix` (or empty), then steps
  the LM forward, avoiding dead-end tokens (up to `maxRetries` retries per step)
  and tracking the longest complete prefix reached. Stops after at most
  `maxSteps - closeReserve` steps. `currentOut` is `IsCompletePrefix` or empty;
  `generatedOut` is the stable prefix followed by `currentOut`.
  The constrained span is left open — no closing delimiter is emitted — and the
  `closeReserve` steps withheld from the step budget remain available for closing
  it afterward.
  Requires `closeReserve <= maxSteps` and `|currentConstrained| <= |generated|`.
  Cost: at most +`(maxSteps - closeReserve)`.
  Control profile: parser repair by deletion plus dead-end-avoiding regeneration.

- `helpers.RegenerateUnitOnCheckFailure(lm, parser, prompt, currentConstrained, eosToken, maxStepsPerUnit, maxRetries, maxRollbackBudget, allowedUnits)`
  Role: generate a constrained span unit-by-unit, rewinding and resampling any
  unit whose rendered text is not in a caller-supplied allowed set.
  When to use: when evaluation shows outputs that are syntax-valid but score
  incorrect — i.e. the failure is in WHICH units the model chose, not in
  structure. If outputs are already well-formed (grammar-valid) yet wrong, the
  problem is unit selection; this helper directly addresses that by checking each
  completed grammar unit against a set of acceptable values and resampling on
  mismatch.
  How to use: build `allowedUnits` from the identifiers or names that appear
  in the per-example prompt context (e.g. scan `prompt` for recognizable tokens
  using `CSDHelpers.PrefixToString` or by extracting from the rendered input
  string). Pass this as a `seq<string>`. Start with conservative retry values
  such as `maxStepsPerUnit := 20`, `maxRetries := 3`, `maxRollbackBudget := 10`.
  Example call shape:
    var allowed: seq<string> := /* names extracted from prompt context */;
    var result := helpers.RegenerateUnitOnCheckFailure(
        lm, parser, prompt, currentConstrained, eosToken,
        20, 3, 10, allowed);
  If `allowedUnits` is empty (e.g. because no names were found in context), the
  check is disabled and the helper degrades gracefully to plain dead-end-avoiding
  generation — so it is always safe to call.
  Mechanics: generates tokens via `DeadEndAvoidingStep`. At each grammar-unit
  boundary (`parser.IsCompletePrefix` becomes true), it renders the new unit text
  and checks it against `allowedUnits`. On a failed check and with remaining retry
  budget, it rolls back to the last accepted checkpoint, penalizes the rejected
  first token, and regenerates from there. After `maxRetries` failures on one
  unit, the unit is accepted to preserve termination. Returns `resultConstrained:
  Prefix`, which is always parser-valid.
  Cost: bounded by `(maxRetries + 1) * maxStepsPerUnit` total steps; degrades to
  plain generation when `allowedUnits` is empty or the budget is exhausted.
  Control profile: unit-level rollback-and-resample with bounded retry and
  budget; grammar-driven boundary detection; caller-supplied allowed-unit set.

- `helpers.RegenerateUnitOnGroundingFailure(lm, parser, prompt, currentConstrained, eosToken, maxStepsPerUnit, maxRetries, maxRollbackBudget)`
  Role: generate a constrained span unit-by-unit, rewinding and resampling any
  completed unit whose identifier-like tokens are not grounded in the prompt
  context.
  Mechanics: generates tokens via dead-end-avoiding steps; at each grammar-unit
  boundary (`parser.IsCompletePrefix` becomes true) it renders the new unit and
  calls `lm.SpanGrounded` on it. On a false result and with remaining budget it
  rolls back to the last accepted checkpoint, penalizes the rejected first token,
  and regenerates from there; after `maxRetries` failures on one unit the unit is
  accepted to preserve termination. Returns `resultConstrained: Prefix`, always
  parser-valid.
  Cost: bounded by `(maxRetries + 1) * maxStepsPerUnit` total steps.
  Control profile: unit-level rollback-and-resample with bounded retry and
  budget; grammar-driven boundary detection; prompt-grounded acceptance test.

- `lm.SpanGrounded(text)` (predicate)
  Returns true iff every identifier-like token in `text` appears in the support
  set extracted from the per-example prompt context; returns true when the prompt
  contains no recognizable support set (so it is always safe to call). Identifier-
  like tokens exclude quoted string-literal contents and short alias-like tokens
  (a single letter, or letters followed by digits). Pure with respect to the
  Dafny heap. Fair: the support set is derived only from prompt text, never from
  execution feedback or gold answers.

- `helpers.PrefixAppearsInPrompt(lm, prefix)`
  Role: no-gold duplicate/exemplar check for a generated prefix.
  Mechanics: renders `prefix` and asks whether the normalized span already
  appears in the current prompt context, including prompt examples and rolling
  suffix text visible to the model. It uses exact normalized span matching, not
  substring search.
  Cost: +0.
  Fair: reads only prompt-visible text; never reads gold labels, scorer state,
  evaluator results, or class-specific win rules.

- `helpers.PrefixResemblesPromptExamples(lm, prefix)`
  Role: no-gold resemblance score for a generated prefix.
  Mechanics: renders `prefix` and returns a real number in [0,1] measuring how
  structurally similar it is to the example spans shown in the current prompt
  (the same prompt-visible examples `PrefixAppearsInPrompt` inspects). Similarity
  is computed with generic tooling. Returns 0.0 when the prompt shows no examples
  or the prefix cannot be parsed. The strategy chooses its own threshold.
  Cost: +0.
  Fair: reads only prompt-visible examples; never reads gold labels, scorer
  state, held-out data, evaluator results, or class-specific strategy advice.

- `helpers.LastTokenBefore(generated, sep)` and
  `CSDHelpers.ExtractAfterKeyword(prefix, keyword)`
  Role: read lightweight context from existing generated tokens.
  Mechanics: scan token sequences and return matching context tokens; no LM
  call and no state change.
  Cost: +0.
  Control profile: context information only.

- `CSDHelpers.CountTokenOccurrences(prefix, target)` (static)
  Role: count how many times `target` appears as a token element of `prefix`.
  Cost: +0.

- `CSDHelpers.OccurrencesInRange(prefix, target, hi)` (static function)
  Role: count how many times `target` appears before index `hi`.
  Contract: requires `hi <= |prefix|`.
  Cost: +0.

- `CSDHelpers.TokensSinceLastOccurrence(prefix, target)` (static)
  Role: distance in tokens from the end of `prefix` back to the last occurrence
  of `target`; returns `|prefix|` if `target` never occurs.
  Cost: +0.

- `CSDHelpers.FlattenTokenGroups`, `CSDHelpers.GroupContaining`,
  `CSDHelpers.IntersectTokenSets`, and `CSDHelpers.SubtractTokenSets`
  Role: transform token sets or groups.
  Mechanics: operate on sequences only; no LM call, parser query, output append,
  or state transition.
  Cost: +0.
  Control profile: token-set bookkeeping only.

- `helpers.GroupHasValidMember(parser, prefix, group)`
  Role: test whether any token in `group` is parser-valid at `prefix`.
  Mechanics: linear scan over `group`; no LM call or logit change.
  Cost: +0.

- `helpers.RollbackConstrainedSpan(parser, stablePrefix, generated, currentConstrained)`
  Role: repair constrained text when `generated == stablePrefix + currentConstrained`.
  Mechanics: rolls back `currentConstrained` with `RollbackToValidPrefix`, then
  reattaches `stablePrefix`.
  Cost: +0.

- `helpers.SoftConstrainedStep(lm, parser, prompt, constrainedPrefix, boostAmount, eosToken)`
  Role: one step that boosts grammar-valid logits (and EOS) then samples without a hard mask.
  Mechanics: `BoostValidNextAndEos`, `ChooseNextTokenUnconstrained`; `isValid` reports
  EOS or parser-valid extension.
  Cost: +1 token-step.

- `helpers.PenalizedConstrainedStep`
  Role: `ConstrainedStep` with an explicit penalize list that must already be in `lm.Tokens`.
  Mechanics: same hard mask and postconditions as `ConstrainedStep` after the
  penalty edit.
  Cost: +1 token-step.

- `helpers.BoostedConstrainedStep`
  Role: `ConstrainedStep` with an explicit boost list that must already be in `lm.Tokens`.
  Mechanics: same hard mask and postconditions as `ConstrainedStep` after the
  boost edit.
  Cost: +1 token-step.

- `helpers.BoostTokenLogits`
  Role: direct positive logit nudge for a vocabulary-known token list.
  Mechanics: requires `forall t in list :: t in lm.Tokens`; clamped arithmetic; no LM call.
  Cost: +0.

- `helpers.PenalizeTokenLogits`
  Role: direct logit nudge for a vocabulary-known token list.
  Mechanics: requires `forall t in list :: t in lm.Tokens`; clamped arithmetic; no LM call.
  Cost: +0.

- `helpers.RepetitionPenaltyStep`
  Role: constrained step with repetition penalty on a token bag before masking.
  Mechanics: non-safe variant imposes the same membership proofs as the
  underlying penalize helper.
  Cost: +1 token-step.

- `helpers.TemperatureConstrainedStep`
  Role: constrained step with temperature scaling before masking.
  Mechanics: non-safe variant imposes the same range proofs as the underlying
  scale helper.
  Cost: +1 token-step.

- `helpers.RolloutConstrainedWithPenalties(lm, parser, prompt, startPrefix, totalBudget, penalties, penaltyAmount, eosToken)`
  Role: bounded loop of `SafePenalizedConstrainedStep` until completion, EOS, or budget.
  Mechanics: extends `generatedOut` token by token; `cost` increases by `stepsUsed`.
  Cost: +`stepsUsed` token-steps (each inner step +1).

- `helpers.GetHighestLogitToken(lm)`
  Role: read the argmax from the current logit vector.
  Mechanics: no forward pass; assumes logits already match the intended prefix.
  Cost: +0.

- `helpers.GetTokenLogit(lm, token)`
  Role: read one coordinate from the current logit vector.
  Mechanics: no forward pass; assumes logits already match the intended prefix.
  Cost: +0.

- `helpers.ScaleAllLogits(lm, scalar)`
  Role: multiply every vocabulary logit by a positive scalar with bounds clamping.
  Mechanics: no LM call; use before a sampling helper that reads the same logits.
  Cost: +0.

- `helpers.SaveLogitsSnapshot(lm)`
  Role: copy logits for branching or speculation.
  Mechanics: full-array read; use with a later restore when a speculative path
  should not commit logit state.
  Cost: +0 for snapshot ops alone.

- `helpers.RestoreLogitsSnapshot(lm, snapshot)`
  Role: restore copied logits after branching or speculation.
  Mechanics: full-array write; restores prior logits without changing `cost` by itself.
  Cost: +0 for snapshot ops alone.

- `CSDHelpers.CountSubstring(s, sub)` (static function; same class as instance helpers)
  Role: count non-overlapping occurrences of `sub` in string `s`.
  Cost: +0.

- `CSDHelpers.PrefixToString(prefix)` / `CSDHelpers.ExtractContentBetweenDelimiters(input, startDelim, endDelim)` (static)
  Role: stringify a token prefix or extract delimited substring content per contract.
  Cost: +0.

## LM and `Parser` surface (for calls inside the strategy body)

Strategy code may also invoke `lm` and `parser` members directly when proofs permit.
Summaries align with `synthesis/verify/library/README.md`; the `.dfy` file states full contracts.

- **`lm`:** `GenerateLogits`, `ChooseNextToken`, `ChooseNextTokenUnconstrained`, `GenerateUnconstrainedChunk`,
  `MaskValidNextAndEos`, `BoostValidNextAndEos`, `MaskToken` / `MaskTokens` / `MaskTokensExcept`,
  `IdToToken`, `TokenToId`, logit readers, `IsMasked`, `HasUnmaskedToken`.
- **`parser`:** `IsValidPrefix`, `IsCompletePrefix`, `IsDeadPrefix`, `ValidNextTokenCount`, `ValidNextToken`,
  `ValidNextTokens`, `ParseG`.

## Proof sketch discipline

Your output must include a `// CSD_PROOF_SKETCH_BEGIN ... // CSD_PROOF_SKETCH_END`
block between the rationale and the method body.

For each of the following two non-trivial loop invariants, explain in one or
two sentences per branch why that branch preserves the invariant:

1. `parser_validity`:
   `insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)`
2. `progress`:
   `|generated| <= |generatedPrefix| + steps`
   Each branch should increment `steps` by the token budget consumed by its
   generation/forced-delimiter helper. Some helpers may consume tokens that are
   not appended visibly, for example EOS or a rejected suffix from
   `ConstrainedSymbol`; in those cases, the output-length bound is still
   preserved because visible growth is at most the consumed token budget.

The proof sketch should explain preservation of the listed invariants.

"""


_TOOL_REFERENCE_START = "\n## Available Tools\n"
_PROOF_DISCIPLINE_START = "\n## Proof sketch discipline\n"
_TOOL_REFERENCE_START_INDEX = SYSTEM_PROMPT.index(_TOOL_REFERENCE_START)
_PROOF_DISCIPLINE_START_INDEX = SYSTEM_PROMPT.index(_PROOF_DISCIPLINE_START)
TOOL_REFERENCE = SYSTEM_PROMPT[
    _TOOL_REFERENCE_START_INDEX:_PROOF_DISCIPLINE_START_INDEX
].strip()
SYSTEM_PROMPT = (
    SYSTEM_PROMPT[:_TOOL_REFERENCE_START_INDEX].rstrip()
    + "\n\n"
    + SYSTEM_PROMPT[_PROOF_DISCIPLINE_START_INDEX:].lstrip()
)

_HELPER_CALL_RE = re.compile(
    r"\b(?:helpers|CSDHelpers)\.([A-Za-z_][A-Za-z0-9_]*)\s*\("
)
_HELPER_REF_RE = re.compile(
    r"\b(?:helpers|CSDHelpers)\.([A-Za-z_][A-Za-z0-9_]*)\b"
)
_MENU_PRUNED_HELPERS = {
    "RollbackToValidPrefix",
    "RollbackToCompletePrefix",
    "RollbackConstrainedSpan",
    "RollbackAndRegenerate",
    "RollbackAndContinue",
    "SaveLogitsSnapshot",
    "RestoreLogitsSnapshot",
    "RolloutConstrainedWithPenalties",
    "SpeculativeConstrainedRollout",
    "RegenerateUnitOnCheckFailure",
}
_ALL_HELPER_NAMES = (
    set(_HELPER_REF_RE.findall(TOOL_REFERENCE))
    - _MENU_PRUNED_HELPERS
    | {"GenerateWithManagedSpan", "GenerateWithPrefixAndManagedSpan", "ManagedStep"}
)


def _filter_code_fence_lines(text: str, allowed: set[str]) -> str:
    """Drop helper signatures that are outside the active helper contract."""
    lines = []
    for line in text.splitlines():
        helper_names = set(_HELPER_REF_RE.findall(line))
        if helper_names and helper_names.isdisjoint(allowed):
            continue
        lines.append(line)
    return "\n".join(lines).rstrip()


def _iter_helper_doc_blocks(section_body: str):
    block: list[str] = []
    for line in section_body.splitlines():
        if line.startswith("- `"):
            if block:
                yield block
            block = [line]
        elif block:
            block.append(line)
    if block:
        yield block


def _filter_tool_api_reference(text: str, allowed: set[str]) -> str:
    """Keep API reference bullets only for helpers visible this attempt."""
    lm_parser_surface = ""
    lm_parser_header = "\n## LM and `Parser` surface"
    if lm_parser_header in text:
        text, lm_parser_surface = text.split(lm_parser_header, 1)
        lm_parser_surface = lm_parser_header.strip() + lm_parser_surface.rstrip()

    lines = text.splitlines()
    section_starts = [
        index for index, line in enumerate(lines) if line.startswith("### ")
    ]
    if not section_starts:
        parts = [text.rstrip()] if text.rstrip() else []
        if lm_parser_surface:
            parts.append(lm_parser_surface)
        return "\n\n".join(parts)

    intro = "\n".join(lines[:section_starts[0]]).rstrip()
    rendered_sections = []

    for start_index, next_start_index in zip(
        section_starts,
        section_starts[1:] + [len(lines)],
    ):
        header = lines[start_index]
        body = "\n".join(lines[start_index + 1:next_start_index])
        kept_blocks = []
        for block in _iter_helper_doc_blocks(body):
            helper_names = set(_HELPER_REF_RE.findall("\n".join(block)))
            if helper_names and helper_names.issubset(allowed):
                kept_blocks.append("\n".join(block).rstrip())
        if kept_blocks:
            rendered_sections.append(
                header + "\n\n" + "\n\n".join(kept_blocks)
            )

    parts = [intro] if intro else []
    parts.extend(rendered_sections)
    if lm_parser_surface:
        parts.append(lm_parser_surface)
    return "\n\n".join(parts).rstrip()


def _filter_tool_reference(allowed_helpers: list[str]) -> str:
    """Render a reduced tool catalog for the currently allowed helper set."""
    allowed = set(allowed_helpers)
    if "\n## Tool API Reference\n" not in TOOL_REFERENCE:
        return _filter_code_fence_lines(TOOL_REFERENCE, allowed)

    available_tools, api_reference = TOOL_REFERENCE.split(
        "\n## Tool API Reference\n",
        1,
    )
    return (
        _filter_code_fence_lines(available_tools, allowed)
        + "\n\n## Tool API Reference\n\n"
        + _filter_tool_api_reference(api_reference, allowed)
    ).rstrip()


INITIAL_GENERATION_PROMPT = """\
Generate a complete Dafny method body for this use-case.

Task:
{task_description}
{allowed_helpers_block}{tool_reference_block}{decoding_surface_block}

Return exactly the Dafny method body: rationale block, proof sketch block, then body statements.
Begin your output with `// CSD_RATIONALE_BEGIN` — no prose, no preamble, no Markdown fence above it.

## Verified Examples

The verified examples below are pattern demonstrations, not task-specific recommendations.
Use them as a palette of mechanisms: span entry, constrained progression,
closing/termination, repair, chunking, and preference shaping. Adapt or combine
only the parts whose control behavior matches the current task contract and
measured failures.

When the output contract uses visible delimiter spans, keep each span as short
as possible, open it only when the exact content is known, and close it
immediately after the final token of that exact content.
Keep all intermediate reasoning outside visible spans; a visible span should
contain only the final exact expression or answer content, not setup or prose.
If the task allows it, use only one visible span for the final answer rather
than emitting multiple visible spans throughout the solution.
For visible spans that contain exact content, prefer parser-valid token steps
over free-form chunks, and avoid repetition penalties that can amplify valid
symbol loops.
For exact visible spans, use hard parser-controlled helpers such as
`ConstrainedStep`; do not use `ConfidenceGatedStep` inside exact spans.

```dafny
// CSD_RATIONALE_BEGIN
// Guided-adaptive CSD. The strategy uses an append-only prompt-guidance
// block to steer the model, then adapts the decoder: it uses group boosting
// while the active prefix is narrow, and switches to penalty-aware constrained
// decoding once the prefix has grown past the initial region. The keyword
// groups, penalty tokens, and narrow threshold are caller-provided parameters.
// CSD_RATIONALE_END
// CSD_PROOF_SKETCH_BEGIN
// parser_validity: AppendTaskGuidance leaves generated state unchanged.
//   Outside the span, the implication is vacuous unless "<<" is observed, which
//   resets currentConstrainedOut to the valid empty prefix. CloseConstrainedSpan
//   exits constrained mode. GroupBoostedConstrainedStep and
//   AdaptiveConstrainedStepWithPenalties return EOS or parser-valid tokens, and
//   AppendConstrainedToken preserves parser validity.
// progress: AppendTaskGuidance costs 0. Every later branch consumes one step
//   and appends at most one visible token, so the output-length bound remains
//   linear in steps. The adaptive branch only changes which parser-valid helper
//   is used.
// CSD_PROOF_SKETCH_END
generated := generatedPrefix;
insideConstrainedOut := insideConstrained;
currentConstrainedOut := currentConstrained;
cost := 0;

var guidance: string := "Generate exactly one task-appropriate output. No explanation or Markdown. Follow the task's declared output contract exactly.";
helpers.AppendTaskGuidance(lm, guidance);

// Caller-provided parameters (in practice these would be passed via the task description or synthesis context)
var keywordGroups: seq<seq<Token>> := validTokenGroups; // Default to caller-supplied groups
var penaltyTokens: seq<Token> := []; // Caller-provided tokens to penalize (e.g., early terminators)
var narrowThreshold: nat := 10;
var steps: nat := 0;
var phase: nat := 0; // 0 = initial narrow phase, 1 = post-penalty phase

while steps < maxSteps
  invariant 0 <= steps <= maxSteps
  invariant lm.ValidTokensIdsLogits()
  invariant !insideConstrainedOut ==> currentConstrainedOut == []
  invariant insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)
  invariant insideConstrainedOut ==> |currentConstrainedOut| <= |generated|
  invariant |generated| <= |generatedPrefix| + steps
  invariant phase == 0 || phase == 1
  decreases maxSteps - steps
{{
  if !insideConstrainedOut {{
    var next := helpers.UnconstrainedStep(lm, prompt, generated);
    steps := steps + 1;
    if next == eosToken {{
      break;
    }} else {{
      generated := generated + [next];
      if next == "<<" {{
        insideConstrainedOut := true;
        currentConstrainedOut := [];
        phase := 0;
      }}
    }}
  }} else if parser.IsCompletePrefix(currentConstrainedOut) {{
    var closedGenerated, closedInside, closedCurrent := helpers.CloseConstrainedSpan(
      lm, parser, generated, currentConstrainedOut
    );
    generated := closedGenerated;
    insideConstrainedOut := closedInside;
    currentConstrainedOut := closedCurrent;
    steps := steps + 1;
  }} else {{
    var constrainedPrompt := prompt + generated[..|generated| - |currentConstrainedOut|];
    var validCount := helpers.ValidTokenCount(parser, currentConstrainedOut);
    var next := eosToken;
    if phase == 0 && validCount <= narrowThreshold {{
      var groups := keywordGroups + validTokenGroups;
      next := helpers.GroupBoostedConstrainedStep(
        lm, parser, constrainedPrompt, currentConstrainedOut, groups, 6.0, eosToken
      );
      if validCount > narrowThreshold {{
        phase := 1;
      }}
    }} else if phase == 1 && |penaltyTokens| > 0 {{
      next := helpers.AdaptiveConstrainedStepWithPenalties(
        lm, parser, constrainedPrompt, currentConstrainedOut,
        validTokenGroups, 4.0, penaltyTokens, 5.0, 8, eosToken
      );
    }} else {{
      next := helpers.AdaptiveConstrainedStep(
        lm, parser, constrainedPrompt, currentConstrainedOut, validTokenGroups, 4.0, narrowThreshold, eosToken
      );
    }}
    steps := steps + 1;
    if next == eosToken {{
      break;
    }} else {{
      var appendedGenerated, appendedInside, appendedCurrent := helpers.AppendConstrainedToken(
        lm, parser, generated, currentConstrainedOut, next
      );
      generated := appendedGenerated;
      insideConstrainedOut := appendedInside;
      currentConstrainedOut := appendedCurrent;
    }}
  }}
}}

cost := steps;
```

```dafny
// CSD_RATIONALE_BEGIN
// Simple delimiter-triggered CSD. Generate freely until "<<" appears, then
// constrain; each constrained step calls CloseSpanIfComplete, which emits ">>"
// once the parser accepts the span and is a no-op otherwise.
// CSD_RATIONALE_END
// CSD_PROOF_SKETCH_BEGIN
// parser_validity: In the unconstrained branch we only flip insideConstrainedOut
//   to true when next == "<<", and we set currentConstrainedOut := [] which is
//   a valid prefix. In the constrained branch, CloseSpanIfComplete either closes
//   the span (sets insideConstrainedOut to false, making the implication vacuous,
//   and clears currentConstrainedOut to []) or leaves generated/inside/current
//   unchanged (no-op). In the no-op path, ConstrainedStep plus AppendConstrainedToken
//   preserves parser validity when IsTokenValidNext holds.
// progress: Every branch appends at most one token to generated and steps grows
//   by 1, so |generated| - |generatedPrefix| <= steps <= steps * stepTokenBudget.
// CSD_PROOF_SKETCH_END
generated := generatedPrefix;
insideConstrainedOut := insideConstrained;
currentConstrainedOut := currentConstrained;
cost := 0;

var steps: nat := 0;

while steps < maxSteps
  invariant 0 <= steps <= maxSteps
  invariant lm.ValidTokensIdsLogits()
  invariant !insideConstrainedOut ==> currentConstrainedOut == []
  invariant insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)
  invariant insideConstrainedOut ==> |currentConstrainedOut| <= |generated|
  invariant |generated| <= |generatedPrefix| + steps
  decreases maxSteps - steps
{{
  if !insideConstrainedOut {{
    var next := helpers.UnconstrainedStep(lm, prompt, generated);
    steps := steps + 1;
    if next == eosToken {{
      break;
    }} else {{
      generated := generated + [next];
      if next == "<<" {{
        insideConstrainedOut := true;
        currentConstrainedOut := [];
      }}
    }}
  }} else {{
    var cg, ci, cc, closed := helpers.CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut);
    steps := steps + 1;
    if closed {{
      generated := cg;
      insideConstrainedOut := ci;
      currentConstrainedOut := cc;
    }} else {{
      var constrainedPrompt := prompt + generated[..|generated| - |currentConstrainedOut|];
      var next := helpers.ConstrainedStep(lm, parser, constrainedPrompt, currentConstrainedOut, eosToken);
      if next == eosToken {{
        break;
      }} else {{
        var valid := helpers.IsTokenValidNext(parser, currentConstrainedOut, next);
        if valid {{
          var appendedGenerated, appendedInside, appendedCurrent := helpers.AppendConstrainedToken(
            lm, parser, generated, currentConstrainedOut, next
          );
          generated := appendedGenerated;
          insideConstrainedOut := appendedInside;
          currentConstrainedOut := appendedCurrent;
        }}
      }}
    }}
  }}
}}

cost := steps;
```

```dafny
// CSD_RATIONALE_BEGIN
// Context-triggered CSD. The strategy tracks whether a neutral local marker has
// recently appeared in the free text. When that marker is seen, the next
// outside-span action opens a constrained span before returning to ordinary
// delimiter-triggered behavior. Inside the span, parser validity remains the
// hard authority.
// CSD_RATIONALE_END
// CSD_PROOF_SKETCH_BEGIN
// parser_validity: Outside the span, the implication is vacuous. The
//   context-triggered open branch uses OpenConstrainedSpan, which returns
//   currentConstrainedOut == [], valid by parser.IsValidPrefix([]). If the
//   free token itself is "<<", we enter with currentConstrainedOut := [].
//   CloseConstrainedSpan exits constrained mode. ConstrainedStep plus
//   AppendConstrainedToken preserves validity for non-EOS tokens.
// progress: UnconstrainedStep, OpenConstrainedSpan, CloseConstrainedSpan, and
//   ConstrainedStep each consume one step and append at most one token, so
//   |generated| <= |generatedPrefix| + steps is preserved.
// CSD_PROOF_SKETCH_END
generated := generatedPrefix;
insideConstrainedOut := insideConstrained;
currentConstrainedOut := currentConstrained;
cost := 0;

var steps: nat := 0;
var markerArmed: bool := false;
var markerToken: Token := ":";

while steps < maxSteps
  invariant 0 <= steps <= maxSteps
  invariant lm.ValidTokensIdsLogits()
  invariant !insideConstrainedOut ==> currentConstrainedOut == []
  invariant insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)
  invariant insideConstrainedOut ==> |currentConstrainedOut| <= |generated|
  invariant |generated| <= |generatedPrefix| + steps
  decreases maxSteps - steps
{{
  if !insideConstrainedOut {{
    if markerArmed {{
      var openedGenerated, openedInside, openedCurrent := helpers.OpenConstrainedSpan(lm, generated);
      generated := openedGenerated;
      insideConstrainedOut := openedInside;
      currentConstrainedOut := openedCurrent;
      markerArmed := false;
      steps := steps + 1;
    }} else {{
      var next := helpers.UnconstrainedStep(lm, prompt, generated);
      steps := steps + 1;
      if next == eosToken {{
        break;
      }} else {{
        generated := generated + [next];
        if next == "<<" {{
          insideConstrainedOut := true;
          currentConstrainedOut := [];
          markerArmed := false;
        }} else if next == markerToken {{
          markerArmed := true;
        }}
      }}
    }}
  }} else if parser.IsCompletePrefix(currentConstrainedOut) {{
    var closedGenerated, closedInside, closedCurrent := helpers.CloseConstrainedSpan(
      lm, parser, generated, currentConstrainedOut
    );
    generated := closedGenerated;
    insideConstrainedOut := closedInside;
    currentConstrainedOut := closedCurrent;
    steps := steps + 1;
  }} else {{
    var constrainedPrompt := prompt + generated[..|generated| - |currentConstrainedOut|];
    var next := helpers.ConstrainedStep(lm, parser, constrainedPrompt, currentConstrainedOut, eosToken);
    steps := steps + 1;
    if next == eosToken {{
      break;
    }} else {{
      var appendedGenerated, appendedInside, appendedCurrent := helpers.AppendConstrainedToken(
        lm, parser, generated, currentConstrainedOut, next
      );
      generated := appendedGenerated;
      insideConstrainedOut := appendedInside;
      currentConstrainedOut := appendedCurrent;
    }}
  }}
}}

cost := steps;
```

```dafny
// CSD_RATIONALE_BEGIN
// Group-aware constrained CSD. Generate freely until "<<" appears, then use
// caller-supplied token groups as a soft preference while the parser remains
// the hard validity authority.
// CSD_RATIONALE_END
// CSD_PROOF_SKETCH_BEGIN
// parser_validity: Outside the span, the implication is vacuous unless next is
//   "<<", in which case currentConstrainedOut := [] is valid by precondition.
//   CloseConstrainedSpan flips insideConstrainedOut to false. In the active
//   constrained branch, GroupBoostedConstrainedStep returns either EOS or a
//   parser-valid next token, and AppendConstrainedToken preserves validity.
// progress: Every branch appends at most one token and steps grows by 1, so
//   |generated| <= |generatedPrefix| + steps is preserved.
// CSD_PROOF_SKETCH_END
generated := generatedPrefix;
insideConstrainedOut := insideConstrained;
currentConstrainedOut := currentConstrained;
cost := 0;

var steps: nat := 0;

while steps < maxSteps
  invariant 0 <= steps <= maxSteps
  invariant lm.ValidTokensIdsLogits()
  invariant !insideConstrainedOut ==> currentConstrainedOut == []
  invariant insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)
  invariant insideConstrainedOut ==> |currentConstrainedOut| <= |generated|
  invariant |generated| <= |generatedPrefix| + steps
  decreases maxSteps - steps
{{
  if !insideConstrainedOut {{
    var next := helpers.UnconstrainedStep(lm, prompt, generated);
    steps := steps + 1;
    if next == eosToken {{
      break;
    }} else {{
      generated := generated + [next];
      if next == "<<" {{
        insideConstrainedOut := true;
        currentConstrainedOut := [];
      }}
    }}
  }} else if parser.IsCompletePrefix(currentConstrainedOut) {{
    var closedGenerated, closedInside, closedCurrent := helpers.CloseConstrainedSpan(
      lm, parser, generated, currentConstrainedOut
    );
    generated := closedGenerated;
    insideConstrainedOut := closedInside;
    currentConstrainedOut := closedCurrent;
    steps := steps + 1;
  }} else {{
    var constrainedPrompt := prompt + generated[..|generated| - |currentConstrainedOut|];
    var next := helpers.GroupBoostedConstrainedStep(
      lm, parser, constrainedPrompt, currentConstrainedOut, validTokenGroups, 4.0, eosToken
    );
    steps := steps + 1;
    if next == eosToken {{
      break;
    }} else {{
      var appendedGenerated, appendedInside, appendedCurrent := helpers.AppendConstrainedToken(
        lm, parser, generated, currentConstrainedOut, next
      );
      generated := appendedGenerated;
      insideConstrainedOut := appendedInside;
      currentConstrainedOut := appendedCurrent;
    }}
  }}
}}

cost := steps;
```

```dafny
// CSD_RATIONALE_BEGIN
// Top-candidate constrained CSD. Inside a span, query a small ranked set of
// parser-valid candidates and append the first non-EOS candidate if available.
// CSD_RATIONALE_END
// CSD_PROOF_SKETCH_BEGIN
// parser_validity: Enter only via the unconstrained branch when next == "<<",
//   at which point currentConstrainedOut := [] (valid). Complete-prefix branch
//   flips insideConstrainedOut to false via CloseConstrainedSpan. TopValidCandidates
//   returns only EOS or valid-next tokens; after excluding EOS, AppendConstrainedToken
//   preserves validity.
// progress: Every branch appends at most one token; attempts grows by 1, so
//   |generated| <= |generatedPrefix| + attempts * stepTokenBudget holds.
// CSD_PROOF_SKETCH_END
generated := generatedPrefix;
insideConstrainedOut := insideConstrained;
currentConstrainedOut := currentConstrained;
cost := 0;

var attempts: nat := 0;

while attempts < maxSteps
  invariant 0 <= attempts <= maxSteps
  invariant lm.ValidTokensIdsLogits()
  invariant !insideConstrainedOut ==> currentConstrainedOut == []
  invariant insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)
  invariant insideConstrainedOut ==> |currentConstrainedOut| <= |generated|
  invariant |generated| <= |generatedPrefix| + attempts
  decreases maxSteps - attempts
{{
  if !insideConstrainedOut {{
    var next := helpers.UnconstrainedStep(lm, prompt, generated);
    attempts := attempts + 1;
    if next == eosToken {{
      break;
    }} else {{
      generated := generated + [next];
      if next == "<<" {{
        insideConstrainedOut := true;
        currentConstrainedOut := [];
      }}
    }}
  }} else if parser.IsCompletePrefix(currentConstrainedOut) {{
    var closedGenerated, closedInside, closedCurrent := helpers.CloseConstrainedSpan(
      lm, parser, generated, currentConstrainedOut
    );
    generated := closedGenerated;
    insideConstrainedOut := closedInside;
    currentConstrainedOut := closedCurrent;
    attempts := attempts + 1;
  }} else {{
    var constrainedPrompt := prompt + generated[..|generated| - |currentConstrainedOut|];
    var candidates := helpers.TopValidCandidates(
      lm, parser, constrainedPrompt, currentConstrainedOut, 4, eosToken
    );
    var next := candidates[0];
    if next == eosToken && |candidates| > 1 {{
      next := candidates[1];
    }}
    attempts := attempts + 1;
    if next == eosToken {{
      break;
    }} else {{
      assert next in candidates;
      assert next in parser.ValidNextTokens(currentConstrainedOut);
      assert parser.IsValidPrefix(currentConstrainedOut + [next]);
      var appendedGenerated, appendedInside, appendedCurrent := helpers.AppendConstrainedToken(
        lm, parser, generated, currentConstrainedOut, next
      );
      generated := appendedGenerated;
      insideConstrainedOut := appendedInside;
      currentConstrainedOut := appendedCurrent;
    }}
  }}
}}

cost := attempts;
```

```dafny
// CSD_RATIONALE_BEGIN
// Chunked-outside CSD. Outside a constrained span we generate unconstrained
// tokens in a single multi-token call (`UnconstrainedChunk`) that breaks early
// on EOS or on the open-span delimiter `"<<"`. Keep the outside chunk budget
// large enough that the model can emit the task's surface prefix before the
// span opens, but still bounded so it does not run away. Inside a
// span we decode token by token using the parser the same way the simple
// delimiter-triggered strategy does.
// Multi-token chunking amortizes per-token dispatch overhead across the
// unconstrained region without starving the prefix.
// CSD_RATIONALE_END
// CSD_PROOF_SKETCH_BEGIN
// parser_validity: Outside the span, insideConstrainedOut stays false unless
//   UnconstrainedChunk reports stoppedOnOpenSpan, in which case we set
//   currentConstrainedOut := [], a valid prefix by the method precondition
//   parser.IsValidPrefix([]). In the complete-prefix branch, CloseConstrainedSpan
//   flips insideConstrainedOut to false, making the implication vacuous. In the
//   constrained-step branch, AppendConstrainedToken is only invoked after
//   ConstrainedStep returned a parser-valid next token (or EOS, which breaks).
// progress: Chunk branch: |generatedOut| <= |generated| + stepsUsed and
//   steps := steps + stepsUsed, so |new_generated| <= |generatedPrefix| +
//   steps + stepsUsed = |generatedPrefix| + new_steps. Other branches append
//   ≤1 token and steps += 1. Linear arithmetic throughout. ✓
// CSD_PROOF_SKETCH_END
generated := generatedPrefix;
insideConstrainedOut := insideConstrained;
currentConstrainedOut := currentConstrained;
cost := 0;

var steps: nat := 0;

while steps < maxSteps
  invariant 0 <= steps <= maxSteps
  invariant lm.ValidTokensIdsLogits()
  invariant !insideConstrainedOut ==> currentConstrainedOut == []
  invariant insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)
  invariant insideConstrainedOut ==> |currentConstrainedOut| <= |generated|
  invariant |generated| <= |generatedPrefix| + steps
  decreases maxSteps - steps
{{
  if !insideConstrainedOut {{
    var chunkBudget: nat := maxSteps - steps;
    if chunkBudget > 32 {{
      chunkBudget := 32;
    }}
    var chunkedG, stoppedOpen, stoppedEos, stepsUsed := helpers.UnconstrainedChunk(
      lm, prompt, generated, chunkBudget, "<<", eosToken
    );
    generated := chunkedG;
    steps := steps + stepsUsed;
    if stoppedEos {{
      break;
    }} else if stoppedOpen {{
      insideConstrainedOut := true;
      currentConstrainedOut := [];
    }}
  }} else if parser.IsCompletePrefix(currentConstrainedOut) {{
    var closedGenerated, closedInside, closedCurrent := helpers.CloseConstrainedSpan(
      lm, parser, generated, currentConstrainedOut
    );
    generated := closedGenerated;
    insideConstrainedOut := closedInside;
    currentConstrainedOut := closedCurrent;
    steps := steps + 1;
  }} else {{
    var constrainedPrompt := prompt + generated[..|generated| - |currentConstrainedOut|];
    var next := helpers.ConstrainedStep(lm, parser, constrainedPrompt, currentConstrainedOut, eosToken);
    steps := steps + 1;
    if next == eosToken {{
      break;
    }} else {{
      var appendedGenerated, appendedInside, appendedCurrent := helpers.AppendConstrainedToken(
        lm, parser, generated, currentConstrainedOut, next
      );
      generated := appendedGenerated;
      insideConstrainedOut := appendedInside;
      currentConstrainedOut := appendedCurrent;
    }}
  }}
}}

cost := steps;
```

```dafny
// CSD_RATIONALE_BEGIN
// Symbol-step CSD. Each outer loop iteration is one "symbol step" — the model
// is called for up to stepTokenBudget tokens at once, and the longest valid
// parser prefix of the result is accepted. This aligns generation granularity
// with the task's natural units: SQL keywords, arithmetic expressions, or
// multi-subword identifiers can be emitted as a unit instead of being forced
// token by token. Outside a constrained span the strategy generates one token
// freely. Inside the span it uses ConstrainedSymbol, passing stepTokenBudget
// as the per-step token allowance. Close as soon as the parser reports the
// prefix is complete.
// CSD_RATIONALE_END
// CSD_PROOF_SKETCH_BEGIN
// parser_validity: Opening a span sets currentConstrainedOut := [], valid by
//   precondition. CloseConstrainedSpan flips insideConstrainedOut to false.
//   ConstrainedSymbol postcondition: parser.IsValidPrefix(symbolOut).
// progress: steps advances by exactly 1 for UnconstrainedStep and
//   CloseConstrainedSpan (each appends ≤1 token), and by stepsUsed ≥ 1 for
//   ConstrainedSymbol (postcondition). So decreases maxSteps - steps always
//   decreases. The invariant |generated| <= |generatedPrefix| + steps is
//   linear: each branch adds at most the consumed token budget to visible
//   output and advances steps by that budget (or 1 for single-token branches). ✓
// CSD_PROOF_SKETCH_END
generated := generatedPrefix;
insideConstrainedOut := insideConstrained;
currentConstrainedOut := currentConstrained;
cost := 0;

var steps: nat := 0;

while steps < maxSteps
  invariant 0 <= steps <= maxSteps
  invariant lm.ValidTokensIdsLogits()
  invariant !insideConstrainedOut ==> currentConstrainedOut == []
  invariant insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)
  invariant insideConstrainedOut ==> |currentConstrainedOut| <= |generated|
  invariant |generated| <= |generatedPrefix| + steps
  decreases maxSteps - steps
{{
  if !insideConstrainedOut {{
    var next := helpers.UnconstrainedStep(lm, prompt, generated);
    steps := steps + 1;
    if next == eosToken {{
      break;
    }} else {{
      generated := generated + [next];
      if next == "<<" {{
        insideConstrainedOut := true;
        currentConstrainedOut := [];
      }}
    }}
  }} else if parser.IsCompletePrefix(currentConstrainedOut) {{
    var closedGenerated, closedInside, closedCurrent := helpers.CloseConstrainedSpan(
      lm, parser, generated, currentConstrainedOut
    );
    generated := closedGenerated;
    insideConstrainedOut := closedInside;
    currentConstrainedOut := closedCurrent;
    steps := steps + 1;
  }} else {{
    var stablePrefix := generated[..|generated| - |currentConstrainedOut|];
    var constrainedPrompt := prompt + stablePrefix;
    var symbolBudget: nat := maxSteps - steps;
    var symbolGenerated, symbolOut, hitEos, stepsUsed := helpers.ConstrainedSymbolInGenerated(
      lm, parser, constrainedPrompt, generated, currentConstrainedOut, symbolBudget, eosToken
    );
    generated := symbolGenerated;
    currentConstrainedOut := symbolOut;
    steps := steps + stepsUsed;
    if hitEos {{
      break;
    }}
  }}
}}

cost := steps;
```

```dafny
// CSD_RATIONALE_BEGIN
// Adaptive-narrowness CSD. Inside a constrained span, query the parser's
// valid-continuation count and choose between different constrained decoding
// strategies based on branch factor. Uses caller-provided keyword groups and
// penalty tokens for the adaptive phases.
// CSD_RATIONALE_END
// CSD_PROOF_SKETCH_BEGIN
// parser_validity: CloseConstrainedSpan makes implication vacuous. In the
//   tight branch, GroupBoostedConstrainedStep returns a parser-valid next token
//   and AppendConstrainedToken preserves validity. In the penalty branch,
//   AdaptiveConstrainedStepWithPenalties returns EOS or parser-valid tokens.
//   In the default branch, AdaptiveConstrainedStep returns EOS or parser-valid
//   tokens. All preserve validity via AppendConstrainedToken.
// progress: Every branch consumes one step and appends at most one visible
//   token, so |generated| <= |generatedPrefix| + steps is preserved.
// CSD_PROOF_SKETCH_END
generated := generatedPrefix;
insideConstrainedOut := insideConstrained;
currentConstrainedOut := currentConstrained;
cost := 0;

var narrowThreshold: nat := 20;
var steps: nat := 0;
var phase: nat := 0; // 0 = narrow (boost), 1 = penalty phase, 2 = default
var keywordGroups: seq<seq<Token>> := validTokenGroups; // Caller-provided
var penaltyTokens: seq<Token> := []; // Caller-provided tokens to penalize

while steps < maxSteps
  invariant 0 <= steps <= maxSteps
  invariant lm.ValidTokensIdsLogits()
  invariant !insideConstrainedOut ==> currentConstrainedOut == []
  invariant insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)
  invariant insideConstrainedOut ==> |currentConstrainedOut| <= |generated|
  invariant |generated| <= |generatedPrefix| + steps
  invariant phase <= 2
  decreases maxSteps - steps
{{
  if !insideConstrainedOut {{
    var next := helpers.UnconstrainedStep(lm, prompt, generated);
    steps := steps + 1;
    if next == eosToken {{
      break;
    }} else {{
      generated := generated + [next];
      if next == "<<" {{
        insideConstrainedOut := true;
        currentConstrainedOut := [];
        phase := 0;
      }}
    }}
  }} else if parser.IsCompletePrefix(currentConstrainedOut) {{
    var closedGenerated, closedInside, closedCurrent := helpers.CloseConstrainedSpan(
      lm, parser, generated, currentConstrainedOut
    );
    generated := closedGenerated;
    insideConstrainedOut := closedInside;
    currentConstrainedOut := closedCurrent;
    steps := steps + 1;
  }} else {{
    var constrainedPrompt := prompt + generated[..|generated| - |currentConstrainedOut|];
    var validCount := helpers.ValidTokenCount(parser, currentConstrainedOut);
    var next := eosToken;
    if phase == 0 && validCount <= narrowThreshold {{
      var groups := keywordGroups + validTokenGroups;
      next := helpers.GroupBoostedConstrainedStep(
        lm, parser, constrainedPrompt, currentConstrainedOut, groups, 6.0, eosToken
      );
      if validCount > narrowThreshold {{
        phase := 1;
      }}
    }} else if phase == 1 && |penaltyTokens| > 0 {{
      next := helpers.AdaptiveConstrainedStepWithPenalties(
        lm, parser, constrainedPrompt, currentConstrainedOut,
        validTokenGroups, 4.0, penaltyTokens, 5.0, 8, eosToken
      );
      if validCount <= narrowThreshold {{
        phase := 0;
      }} else {{
        phase := 2;
      }}
    }} else {{
      next := helpers.AdaptiveConstrainedStep(
        lm, parser, constrainedPrompt, currentConstrainedOut, validTokenGroups, 4.0, narrowThreshold, eosToken
      );
    }}
    steps := steps + 1;
    if next == eosToken {{
      break;
    }} else {{
      var appendedGenerated, appendedInside, appendedCurrent := helpers.AppendConstrainedToken(
        lm, parser, generated, currentConstrainedOut, next
      );
      generated := appendedGenerated;
      insideConstrainedOut := appendedInside;
      currentConstrainedOut := appendedCurrent;
    }}
  }}
}}

cost := steps;
```

```dafny
// CSD_RATIONALE_BEGIN
// Context-tracking CSD. Maintains a strategy-local seq<Token> across loop
// iterations. After each constrained token append, it queries the span for
// tokens following a caller-provided keyword via ExtractAfterKeyword. At
// candidate-selection positions, it intersects parser-valid candidates with
// that context set and boosts the intersection.
// CSD_RATIONALE_END
// CSD_PROOF_SKETCH_BEGIN
// parser_validity: GroupBoostedConstrainedStep returns either EOS or a
//   parser-valid next token; the EOS branch breaks. AppendConstrainedToken
//   preserves validity. CloseConstrainedSpan makes the implication vacuous.
//   The context variable is never passed to the parser as authority, so it
//   cannot affect parser_validity.
// progress: Every branch increments steps by 1 and appends at most one token,
//   so |generated| <= |generatedPrefix| + steps is preserved throughout.
// CSD_PROOF_SKETCH_END
generated := generatedPrefix;
insideConstrainedOut := insideConstrained;
currentConstrainedOut := currentConstrained;
cost := 0;

var semanticContext: seq<Token> := [];
var scopeKeyword: Token := ""; // Caller-provided keyword to track (e.g., "FROM", "=", "(")
var steps: nat := 0;

while steps < maxSteps
  invariant 0 <= steps <= maxSteps
  invariant lm.ValidTokensIdsLogits()
  invariant !insideConstrainedOut ==> currentConstrainedOut == []
  invariant insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)
  invariant insideConstrainedOut ==> |currentConstrainedOut| <= |generated|
  invariant |generated| <= |generatedPrefix| + steps
  decreases maxSteps - steps
{{
  if !insideConstrainedOut {{
    var next := helpers.UnconstrainedStep(lm, prompt, generated);
    steps := steps + 1;
    if next == eosToken {{
      break;
    }} else {{
      generated := generated + [next];
      if next == "<<" {{
        insideConstrainedOut := true;
        currentConstrainedOut := [];
      }}
    }}
  }} else if parser.IsCompletePrefix(currentConstrainedOut) {{
    var closedGenerated, closedInside, closedCurrent := helpers.CloseConstrainedSpan(
      lm, parser, generated, currentConstrainedOut
    );
    generated := closedGenerated;
    insideConstrainedOut := closedInside;
    currentConstrainedOut := closedCurrent;
    steps := steps + 1;
  }} else {{
    // Update semantic context from accumulated span content
    if |scopeKeyword| > 0 {{
      semanticContext := CSDHelpers.ExtractAfterKeyword(currentConstrainedOut, scopeKeyword);
    }}

    var constrainedPrompt := prompt + generated[..|generated| - |currentConstrainedOut|];
    var groups := validTokenGroups;
    if |semanticContext| > 0 {{
      groups := [semanticContext] + validTokenGroups;
    }}
    var next := helpers.GroupBoostedConstrainedStep(
      lm, parser, constrainedPrompt, currentConstrainedOut, groups, 6.0, eosToken
    );
    steps := steps + 1;
    if next == eosToken {{
      break;
    }} else {{
      var appendedGenerated, appendedInside, appendedCurrent := helpers.AppendConstrainedToken(
        lm, parser, generated, currentConstrainedOut, next
      );
      generated := appendedGenerated;
      insideConstrainedOut := appendedInside;
      currentConstrainedOut := appendedCurrent;
    }}
  }}
}}

cost := steps;
```

```dafny
// CSD_RATIONALE_BEGIN
// Rollback-on-stall CSD. Generate freely until "<<" appears. Inside the span,
// generate parser-valid tokens, but if the active constrained content grows
// beyond a local rollback limit before becoming complete, roll back only the
// constrained suffix to the nearest valid non-dead prefix and continue from
// that repaired state.
// CSD_RATIONALE_END
// CSD_PROOF_SKETCH_BEGIN
// parser_validity: Opening a span sets currentConstrainedOut := [], which is
//   valid. Closing a complete span exits constrained mode. ConstrainedStep plus
//   AppendConstrainedToken preserves validity on non-EOS tokens. The rollback
//   branch uses RollbackConstrainedSuffix, whose postcondition gives a valid
//   repaired currentConstrainedOut.
// progress: UnconstrainedStep, CloseConstrainedSpan, and ConstrainedStep each
//   consume one step and append at most one token. RollbackConstrainedSuffix
//   shrinks or preserves generated and we still increment steps by 1, so the
//   output-length bound remains true while the loop metric decreases.
// CSD_PROOF_SKETCH_END
generated := generatedPrefix;
insideConstrainedOut := insideConstrained;
currentConstrainedOut := currentConstrained;
cost := 0;

var steps: nat := 0;
var rollbackLimit: nat := 24;

while steps < maxSteps
  invariant 0 <= steps <= maxSteps
  invariant lm.ValidTokensIdsLogits()
  invariant !insideConstrainedOut ==> currentConstrainedOut == []
  invariant insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)
  invariant insideConstrainedOut ==> |currentConstrainedOut| <= |generated|
  invariant |generated| <= |generatedPrefix| + steps
  decreases maxSteps - steps
{{
  if !insideConstrainedOut {{
    var next := helpers.UnconstrainedStep(lm, prompt, generated);
    steps := steps + 1;
    if next == eosToken {{
      break;
    }} else {{
      generated := generated + [next];
      if next == "<<" {{
        insideConstrainedOut := true;
        currentConstrainedOut := [];
      }}
    }}
  }} else if parser.IsCompletePrefix(currentConstrainedOut) {{
    var closedGenerated, closedInside, closedCurrent := helpers.CloseConstrainedSpan(
      lm, parser, generated, currentConstrainedOut
    );
    generated := closedGenerated;
    insideConstrainedOut := closedInside;
    currentConstrainedOut := closedCurrent;
    steps := steps + 1;
  }} else if |currentConstrainedOut| >= rollbackLimit {{
    var rolledGenerated, rolledCurrent := helpers.RollbackConstrainedSuffix(
      parser, generated, currentConstrainedOut
    );
    generated := rolledGenerated;
    insideConstrainedOut := true;
    currentConstrainedOut := rolledCurrent;
    steps := steps + 1;
  }} else {{
    var constrainedPrompt := prompt + generated[..|generated| - |currentConstrainedOut|];
    var next := helpers.ConstrainedStep(lm, parser, constrainedPrompt, currentConstrainedOut, eosToken);
    steps := steps + 1;
    if next == eosToken {{
      break;
    }} else {{
      var appendedGenerated, appendedInside, appendedCurrent := helpers.AppendConstrainedToken(
        lm, parser, generated, currentConstrainedOut, next
      );
      generated := appendedGenerated;
      insideConstrainedOut := appendedInside;
      currentConstrainedOut := appendedCurrent;
    }}
  }}
}}

cost := steps;
```

```dafny
// CSD_RATIONALE_BEGIN
// Logit-shaped constrained CSD. The strategy uses ordinary free generation
// outside spans. Inside a span, parser validity remains hard; the only extra
// policy is a local soft preference: avoid closing very short constrained
// prefixes (using caller-provided penalty tokens) and softly prefer operator
// tokens once the prefix has begun (using caller-provided boost tokens). The
// safe helpers filter literal token lists internally, so the strategy does not
// need separate vocabulary-membership state for those lists.
// CSD_RATIONALE_END
// CSD_PROOF_SKETCH_BEGIN
// parser_validity: Outside the span the implication is vacuous unless a "<<"
//   token is observed, in which case currentConstrainedOut becomes [].
//   CloseConstrainedSpan exits constrained mode. Both safe constrained-step
//   helpers return EOS or a token preserving parser validity, and
//   AppendConstrainedToken carries that validity into currentConstrainedOut.
// progress: UnconstrainedStep, CloseConstrainedSpan, and each safe constrained
//   step consume one token-step and append at most one visible token, so
//   |generated| <= |generatedPrefix| + steps is preserved.
// CSD_PROOF_SKETCH_END
generated := generatedPrefix;
insideConstrainedOut := insideConstrained;
currentConstrainedOut := currentConstrained;
cost := 0;

var steps: nat := 0;
var minPrefixLength: nat := 2; // Minimum constrained prefix length before switching to boost mode
var penaltyTokens: seq<Token> := []; // Caller-provided tokens to penalize (e.g., early terminators like ">>")
var boostTokens: seq<Token> := []; // Caller-provided tokens to boost (e.g., operators)

while steps < maxSteps
  invariant 0 <= steps <= maxSteps
  invariant lm.ValidTokensIdsLogits()
  invariant !insideConstrainedOut ==> currentConstrainedOut == []
  invariant insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)
  invariant insideConstrainedOut ==> |currentConstrainedOut| <= |generated|
  invariant |generated| <= |generatedPrefix| + steps
  decreases maxSteps - steps
{{
  if !insideConstrainedOut {{
    var next := helpers.UnconstrainedStep(lm, prompt, generated);
    steps := steps + 1;
    if next == eosToken {{
      break;
    }} else {{
      generated := generated + [next];
      if next == "<<" {{
        insideConstrainedOut := true;
        currentConstrainedOut := [];
      }}
    }}
  }} else if parser.IsCompletePrefix(currentConstrainedOut) {{
    var closedGenerated, closedInside, closedCurrent := helpers.CloseConstrainedSpan(
      lm, parser, generated, currentConstrainedOut
    );
    generated := closedGenerated;
    insideConstrainedOut := closedInside;
    currentConstrainedOut := closedCurrent;
    steps := steps + 1;
  }} else {{
    var constrainedPrompt := prompt + generated[..|generated| - |currentConstrainedOut|];
    var next := eosToken;
    if |currentConstrainedOut| < minPrefixLength && |penaltyTokens| > 0 {{
      next := helpers.SafePenalizedConstrainedStep(
        lm, parser, constrainedPrompt, currentConstrainedOut, penaltyTokens, 6.0, eosToken
      );
    }} else if |boostTokens| > 0 {{
      next := helpers.SafeBoostedConstrainedStep(
        lm, parser, constrainedPrompt, currentConstrainedOut, boostTokens, 2.0, eosToken
      );
    }} else {{
      next := helpers.ConstrainedStep(lm, parser, constrainedPrompt, currentConstrainedOut, eosToken);
    }}
    steps := steps + 1;
    if next == eosToken {{
      break;
    }} else {{
      var appendedGenerated, appendedInside, appendedCurrent := helpers.AppendConstrainedToken(
        lm, parser, generated, currentConstrainedOut, next
      );
      generated := appendedGenerated;
      insideConstrainedOut := appendedInside;
      currentConstrainedOut := appendedCurrent;
    }}
  }}
}}

cost := steps;
```

```dafny
// CSD_RATIONALE_BEGIN
// Grounded-and-closed constrained CSD. Phase 1 generates unconstrained until
// "<<" opens a visible span, reserving step budget. Phase 2 fills the span with
// RegenerateUnitOnGroundingFailure using HALF the remaining budget, so a reserve
// is kept for closing; the helper rewinds and resamples any completed unit whose
// identifier-like tokens are not grounded in the prompt context. Phase 3 calls
// CloseSpanWithinBudget on the remaining budget to bring the span to a completable
// state and emit the closing ">>". Per-phase budgets keep the length/cost bounds.
// CSD_RATIONALE_END
// CSD_PROOF_SKETCH_BEGIN
// parser_validity: Phase 1 sets insideConstrainedOut true only on next == "<<",
//   clearing currentConstrainedOut to [] (a valid prefix). Phase 2 stores the
//   helper's result (parser-valid by its postcondition) into currentConstrainedOut.
//   Phase 3 stores CloseSpanWithinBudget's result: on close, insideConstrainedOut is
//   false and currentConstrainedOut == []; otherwise currentConstrainedOut stays a
//   valid prefix. So !insideConstrainedOut ==> currentConstrainedOut == [] and
//   insideConstrainedOut ==> IsValidPrefix(currentConstrainedOut) both hold.
// length/cost: Phase 1 grows generated and steps by 1 per iteration, keeping
//   |generated| <= |generatedPrefix| + steps. Phase 2 runs the helper with
//   maxStepsPerUnit = fillBudget/4 (fillBudget = rem/2), maxRetries = 3; by its
//   length/cost postconditions generated and cost grow by at most 4*(fillBudget/4)
//   <= rem, added to steps. Phase 3 runs CloseSpanWithinBudget with closeBudget =
//   maxSteps - steps; by its length/cost postconditions |generated| <=
//   |generatedPrefix| + maxSteps, and we set steps := maxSteps.
// progress: maxSteps > 0 ==> Phase 1 takes a step (cost > 0), or the span was open
//   on entry and Phase 3 flips insideConstrainedOut.
// CSD_PROOF_SKETCH_END
generated := generatedPrefix;
insideConstrainedOut := insideConstrained;
currentConstrainedOut := currentConstrained;
cost := 0;

var steps: nat := 0;

while steps < maxSteps && !insideConstrainedOut
  invariant 0 <= steps <= maxSteps
  invariant lm.ValidTokensIdsLogits()
  invariant !insideConstrainedOut ==> currentConstrainedOut == []
  invariant insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)
  invariant insideConstrainedOut ==> |currentConstrainedOut| <= |generated|
  invariant |generated| <= |generatedPrefix| + steps
  decreases maxSteps - steps
{{
  var next := helpers.UnconstrainedStep(lm, prompt, generated);
  steps := steps + 1;
  if next == eosToken {{
    break;
  }}
  generated := generated + [next];
  if next == "<<" {{
    insideConstrainedOut := true;
    currentConstrainedOut := [];
  }}
}}

if insideConstrainedOut && steps < maxSteps {{
  var rem := maxSteps - steps;
  var fillBudget := rem / 2;
  if fillBudget >= 4 {{
    var perUnit := fillBudget / 4;
    assert fillBudget == 4 * (fillBudget / 4) + fillBudget % 4;
    assert 4 * perUnit <= fillBudget;
    assert 4 * perUnit <= rem;
    var stable := generated[..|generated| - |currentConstrainedOut|];
    var filled := helpers.RegenerateUnitOnGroundingFailure(
      lm, parser, prompt + stable, currentConstrainedOut, eosToken, perUnit, 3, perUnit
    );
    generated := stable + filled;
    currentConstrainedOut := filled;
    steps := steps + 4 * perUnit;
  }}
}}

if insideConstrainedOut && steps < maxSteps {{
  var closeBudget := maxSteps - steps;
  var cg, ci, cc := helpers.CloseSpanWithinBudget(
    lm, parser, prompt, generated, currentConstrainedOut, eosToken, closeBudget
  );
  generated := cg;
  insideConstrainedOut := ci;
  currentConstrainedOut := cc;
  steps := maxSteps;
}}

cost := steps;
```

```dafny
// CSD_RATIONALE_BEGIN
// Open-then-reliably-close CSD. Phase 1 generates unconstrained until "<<" opens
// a visible span. Phase 2 calls CloseSpanWithinBudget on the entire remaining
// budget to advance the span to a completable state and emit the closing ">>",
// leaving the span open only if no completable state is reachable within budget.
// CSD_RATIONALE_END
// CSD_PROOF_SKETCH_BEGIN
// parser_validity: Phase 1 flips insideConstrainedOut true only on next == "<<",
//   setting currentConstrainedOut := [] (valid). Phase 2's CloseSpanWithinBudget
//   returns a closed span (insideConstrainedOut false, currentConstrainedOut [])
//   or a still-open valid prefix, so both span invariants hold.
// length/cost: Phase 1 grows generated and steps by 1 per iteration
//   (|generated| <= |generatedPrefix| + steps). Phase 2 runs CloseSpanWithinBudget
//   with closeBudget = maxSteps - steps; by its length/cost postconditions
//   |generated| <= |generatedPrefix| + maxSteps, and we set steps := maxSteps.
// progress: maxSteps > 0 ==> Phase 1 takes a step (cost > 0), or the span was open
//   on entry and Phase 2 flips insideConstrainedOut.
// CSD_PROOF_SKETCH_END
generated := generatedPrefix;
insideConstrainedOut := insideConstrained;
currentConstrainedOut := currentConstrained;
cost := 0;

var steps: nat := 0;

while steps < maxSteps && !insideConstrainedOut
  invariant 0 <= steps <= maxSteps
  invariant lm.ValidTokensIdsLogits()
  invariant !insideConstrainedOut ==> currentConstrainedOut == []
  invariant insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)
  invariant insideConstrainedOut ==> |currentConstrainedOut| <= |generated|
  invariant |generated| <= |generatedPrefix| + steps
  decreases maxSteps - steps
{{
  var next := helpers.UnconstrainedStep(lm, prompt, generated);
  steps := steps + 1;
  if next == eosToken {{
    break;
  }}
  generated := generated + [next];
  if next == "<<" {{
    insideConstrainedOut := true;
    currentConstrainedOut := [];
  }}
}}

if insideConstrainedOut && steps < maxSteps {{
  var closeBudget := maxSteps - steps;
  var cg, ci, cc := helpers.CloseSpanWithinBudget(
    lm, parser, prompt, generated, currentConstrainedOut, eosToken, closeBudget
  );
  generated := cg;
  insideConstrainedOut := ci;
  currentConstrainedOut := cc;
  steps := maxSteps;
}}

cost := steps;
```

```dafny
// CSD_RATIONALE_BEGIN
// Token-0 grounded constrained CSD. The eval surface is grammar-constrained from
// the very first token and carries NO visible << >> delimiters, so this strategy
// never emits "<<". If it is not already inside a constrained span it ENTERS one
// silently via EnterObservedConstrainedSpan (sets inside=true, current=[], costs 0
// tokens) instead of opening a visible span. It then GROUNDS the constrained
// content unit-by-unit with RegenerateUnitOnGroundingFailure, which rewinds and
// resamples any completed identifier whose tokens are unsupported by the prompt
// context. Half the remaining step budget goes to grounding; the other half is
// reserved so CloseSpanWithinBudget can always advance the span to a completable
// state. Every budget is charged into `steps`, so the length and cost bounds hold.
// CSD_RATIONALE_END
// CSD_PROOF_SKETCH_BEGIN
// parser_validity: EnterObservedConstrainedSpan returns current == [] (valid);
//   RegenerateUnitOnGroundingFailure returns a valid prefix; CloseSpanWithinBudget
//   returns either a closed span (current == []) or a valid open prefix. So both
//   span invariants hold after every phase.
// length/cost: steps starts at 0. The grounding phase spends fillBudget =
//   (maxSteps - steps)/2 and raises steps by that amount; |generated| grows by at
//   most fillBudget because the regenerated unit replaces a suffix of length
//   |currentConstrainedOut| and RegenerateUnitOnGroundingFailure bounds the result
//   by |currentConstrainedOut| + fillBudget. The close phase spends the rest
//   (closeBudget = maxSteps - steps) and sets steps := maxSteps; by
//   CloseSpanWithinBudget's |generatedOut| <= |generated| + closeBudget we get
//   |generated| <= |generatedPrefix| + maxSteps. cost := steps <= maxSteps.
// CSD_PROOF_SKETCH_END
generated := generatedPrefix;
insideConstrainedOut := insideConstrained;
currentConstrainedOut := currentConstrained;
cost := 0;

var steps: nat := 0;

// (1) Enter a constrained span with NO visible "<<" if not already inside.
if !insideConstrainedOut {{
  generated, insideConstrainedOut, currentConstrainedOut :=
    helpers.EnterObservedConstrainedSpan(lm, generated);
}}
assert insideConstrainedOut;
assert parser.IsValidPrefix(currentConstrainedOut);
assert |currentConstrainedOut| <= |generated|;
assert |generated| <= |generatedPrefix| + steps;

// (2) Ground the constrained content unit-by-unit on half the remaining budget.
if steps < maxSteps {{
  var rem: nat := maxSteps - steps;
  var fillBudget: nat := rem / 2;
  if fillBudget >= 1 {{
    assert |currentConstrainedOut| <= |generated|;
    var stable := generated[..|generated| - |currentConstrainedOut|];
    var filled := helpers.RegenerateUnitOnGroundingFailure(
      lm, parser, prompt + stable, currentConstrainedOut, eosToken, fillBudget, 3, fillBudget);
    generated := stable + filled;
    currentConstrainedOut := filled;
    steps := steps + fillBudget;
    assert |generated| <= |generatedPrefix| + steps;
  }}
}}
assert insideConstrainedOut;
assert parser.IsValidPrefix(currentConstrainedOut);
assert |currentConstrainedOut| <= |generated|;
assert |generated| <= |generatedPrefix| + steps;
assert steps <= maxSteps;

// (3) Advance the span to a completable state on the reserved budget.
if steps < maxSteps {{
  var closeBudget: nat := maxSteps - steps;
  generated, insideConstrainedOut, currentConstrainedOut :=
    helpers.CloseSpanWithinBudget(
      lm, parser, prompt, generated, currentConstrainedOut, eosToken, closeBudget);
  steps := maxSteps;
}}
assert |generated| <= |generatedPrefix| + steps;
assert steps <= maxSteps;

cost := steps;
```

## Requirements
- Output a COMPLETE method body with concrete Dafny statements in every required branch.
- The rationale block should briefly explain:
  - what state the strategy tracks
  - what conditions or observations it uses to choose its next action
  - how the tracked state is meant to support valid generation for this task
- Include the proof sketch block described in the system prompt. Dafny will
  check your proof for you — your sketch just has to convince a careful reader
  that each non-trivial loop invariant is preserved in every branch.
- If you call helpers with preconditions, track enough verifier-friendly state to make those preconditions provable at each call site.
- Include a real decoding procedure, EOS handling, and state updates when needed by your design.
"""


_VERIFIED_EXAMPLES_HEADER = "## Verified Examples\n\n"
_REQUIREMENTS_HEADER = "\n\n## Requirements"
VERIFIED_EXAMPLES = INITIAL_GENERATION_PROMPT.split(_VERIFIED_EXAMPLES_HEADER, 1)[1].split(
    _REQUIREMENTS_HEADER,
    1,
)[0]
INITIAL_GENERATION_PROMPT = INITIAL_GENERATION_PROMPT.replace(
    VERIFIED_EXAMPLES,
    "{verified_examples}",
    1,
)

_VERIFIED_EXAMPLE_PREFIXES = (
    "// Guided-adaptive CSD.",
    "// Simple delimiter-triggered CSD.",
    "// Group-aware constrained CSD.",
    "// Adaptive-narrowness CSD.",
    "// Open-then-reliably-close CSD.",
)


VERIFICATION_ERROR_REFINEMENT_PROMPT = """\
Your previous method body failed Dafny verification.

Task:
{task_description}
{allowed_helpers_block}{tool_reference_block}{decoding_surface_block}

## Verified Examples

These are verified reference CSD patterns available for reuse during verification repair.
Use them as examples of valid helper usage, state tracking, loop structure, and proof style.

{verified_examples}

{search_memory_block}
Previous attempt:
```dafny
{previous_strategy}
```
{strategy_context_block}

Verification error:
```
{error_message}
```
{structured_feedback_block}{error_history_block}{behavioral_context_block}
Revise the method body so it verifies.

Dafny constraint reminder:
- Methods cannot be called directly inside expression contexts.
- Call the method first, bind its result to a local variable, then use that variable in the condition.

Return exactly the corrected full Dafny method body: rationale block, proof sketch block, then body statements.
Begin your output with `// CSD_RATIONALE_BEGIN` — no prose, no preamble, no Markdown fence above it.
Use only the contracts and tools provided above.
"""


RUNTIME_ERROR_REFINEMENT_PROMPT = """\
Your method body passed Dafny verification but failed at runtime.

Task:
{task_description}
{allowed_helpers_block}{tool_reference_block}{decoding_surface_block}

{search_memory_block}
Previous attempt:
```dafny
{previous_strategy}
```

Runtime error:
```
{error_traceback}
```

Fix the runtime error. If needed, rewrite the method body instead of making only local edits.
Return exactly the corrected method body: rationale block, proof sketch block, then body statements.
Begin your output with `// CSD_RATIONALE_BEGIN` — no prose, no preamble, no Markdown fence above it.
"""


COMPILATION_ERROR_REFINEMENT_PROMPT = """\
Your method body passed Dafny verification but failed during Dafny-to-Python compilation.

{allowed_helpers_block}{tool_reference_block}{decoding_surface_block}
{search_memory_block}
Previous attempt:
```dafny
{previous_strategy}
```

Compilation error:
```
{error_message}
```

Fix the compilation error. If needed, rewrite the method body instead of making only local edits.
Return exactly the corrected method body: rationale block, proof sketch block, then body statements.
Begin your output with `// CSD_RATIONALE_BEGIN` — no prose, no preamble, no Markdown fence above it.
"""


FORMAT_REPAIR_PROMPT = """Your output must be a Dafny method body and is missing the required rationale block markers.

Rewrite the following content into a valid Dafny method body that preserves the same strategy semantics and outputs ONLY the method body.

{allowed_helpers_block}{tool_reference_block}{decoding_surface_block}
{search_memory_block}
Content to rewrite:
```dafny
{previous_strategy}
```

The corrected body must include the required rationale and proof sketch blocks.
"""




EVALUATION_FAILURE_REFINEMENT_PROMPT = """\
Your method body passed verification and compilation, then was evaluated on the task,
but did not meet evaluation thresholds.

Task:
{task_description}
{allowed_helpers_block}{tool_reference_block}{decoding_surface_block}

## Verified Dafny patterns (proof-style reference only)

{verified_examples}

{search_memory_block}## Previous attempt

```dafny
{previous_strategy}
```

Result: accuracy {previous_accuracy:.1%}, syntax {previous_syntax_rate:.1%} on {num_examples} examples.
Goal:   accuracy ≥ {goal_accuracy:.1%}, syntax ≥ {goal_syntax_rate:.1%}.
{eval_budget_block}Gap to goal accuracy: {accuracy_gap_pp:.1f}pp.
Gap to goal syntax:   {syntax_gap_pp:.1f}pp.

Evaluation feedback:
```
{evaluation_feedback}
```
{attempt_outcome_ledger_block}{mode_examples_block}{best_so_far_block}
## How to revise

Choose the change size that matches your diagnosis. Numeric tweaks, helper swaps, and structural rewrites are all on the table — pick whatever the failure pattern most directly calls for.
If the task contract includes exact content or visible delimiter spans, treat syntax failures as structural: prefer parser-controlled token generation and immediate span closure over longer unconstrained chunks.
When a visible span is left unterminated or malformed, repair the active constrained state so the span returns to a point the parser can close cleanly, and emit the closing delimiter as soon as it can — before the step budget runs out.
If the output uses visible spans, avoid opening them until the content is settled; once open, keep them to the smallest exact unit the contract allows.
Do not place reasoning text inside a visible span; reserve visible spans for the final exact content only.
When a visible span contains exact content, use parser-guided token selection and keep the span content to a single finished expression rather than a loose sequence of partial fragments.
If a visible span is exact, keep it under hard parser control until the parser
can close it cleanly, without falling back to `ConfidenceGatedStep`.

Return exactly the corrected Dafny method body: rationale block, proof sketch block, then body statements.
Begin your output with `// CSD_RATIONALE_BEGIN` — no prose, no preamble, no Markdown fence above it.
"""


def _build_allowed_helpers_block(allowed_helpers: list[str] | None) -> str:
    """Build a hard helper-call contract block for the current attempt."""
    if not allowed_helpers:
        return ""
    helper_names = ", ".join(f"`{name}`" for name in sorted(set(allowed_helpers)))
    return (
        "Helper-call contract for this attempt:\n"
        "Only these `helpers.<Method>(...)` and `CSDHelpers.<Method>(...)` calls are allowed:\n"
        f"{helper_names}\n"
        "Calls to helper or CSDHelpers methods outside this set are invalid for this attempt.\n\n"
    )


def _build_tool_reference_block(allowed_helpers: list[str] | None) -> str:
    """Build the helper/API reference the model sees for this attempt."""
    if not allowed_helpers:
        return _filter_tool_reference(sorted(_ALL_HELPER_NAMES)) + "\n\n"
    return _filter_tool_reference(allowed_helpers) + "\n\n"


def _build_decoding_surface_block(force_open_span: bool) -> str:
    """State where in the output this run starts.

    Two different runs hand the strategy different starting points: one starts
    outside the span and must open it; the other starts inside a span the
    runtime already opened. Without this block a strategy written for one
    starting point silently does nothing on the other, e.g. one that waits for
    `next == "<<"` before constraining never constrains anything on a run whose
    `<<` is already in the output.
    """
    if force_open_span:
        return (
            "## Decoding surface\n\n"
            "This run starts inside a span the runtime already opened: the "
            "output already contains a visible `<<` and you are inside the "
            "span. Call `CloseConstrainedSpan` to append `>>` and leave the "
            "span once its content is complete. If you are not inside a span, "
            "open one with `OpenConstrainedSpan`, which appends a literal "
            "`<<`. Do not write logic that waits for a `<<` to arrive; the "
            "opening `<<` is already there.\n\n"
        )
    return (
        "## Decoding surface\n\n"
        "This run starts outside the constrained region. Enter constrained "
        "mode by calling `OpenConstrainedSpan`, which appends a literal "
        "`<<` to the output. Call `CloseConstrainedSpan` to append `>>` and "
        "exit constrained mode.\n\n"
    )


def _build_verified_examples_block(allowed_helpers: list[str] | None) -> str:
    """Keep only verified examples compatible with the active helper contract."""
    allowed = set(allowed_helpers) if allowed_helpers is not None else None
    chunks = re.split(r"(?=// CSD_RATIONALE_BEGIN)", VERIFIED_EXAMPLES)
    kept_chunks = []
    for chunk in chunks:
        if "// CSD_RATIONALE_BEGIN" not in chunk:
            continue
        chunk = chunk.strip()
        if not any(prefix in chunk for prefix in _VERIFIED_EXAMPLE_PREFIXES):
            continue
        if allowed is not None:
            helper_names = set(_HELPER_CALL_RE.findall(chunk))
            if not helper_names.issubset(allowed):
                continue
            kept_chunks.append(chunk)
        else:
            kept_chunks.append(chunk)

    if kept_chunks:
        return "\n\n".join(kept_chunks).strip()
    return (
        "// No verified examples are compatible with the active helper-call "
        "contract for this attempt."
    )


def build_initial_prompt(
    task_description: str,
    allowed_helpers: list[str] | None = None,
    force_open_span: bool = False,
) -> tuple[str, str]:
    user_prompt = INITIAL_GENERATION_PROMPT.format(
        task_description=task_description,
        allowed_helpers_block=_build_allowed_helpers_block(allowed_helpers),
        tool_reference_block=_build_tool_reference_block(allowed_helpers),
        decoding_surface_block=_build_decoding_surface_block(force_open_span),
        verified_examples=_build_verified_examples_block(allowed_helpers),
    )
    return SYSTEM_PROMPT, user_prompt


def build_verification_error_prompt(
    task_description: str,
    previous_strategy: str,
    error_message: str,
    behavioral_context: str = "",
    structured_feedback: str = "",
    error_history: str = "",
    strategy_context: str = "",
    search_memory: str = "",
    allowed_helpers: list[str] | None = None,
    force_open_span: bool = False,
) -> tuple[str, str]:
    behavioral_context_block = ""
    structured_feedback_block = ""
    error_history_block = ""
    strategy_context_block = ""
    search_memory_block = ""
    if search_memory:
        search_memory_block = f"{search_memory}\n"
    if strategy_context:
        strategy_context_block = (
            "\nStrategy context from evaluated attempts before this verification failure:\n"
            f"{strategy_context}\n"
        )
    if behavioral_context:
        behavioral_context_block = (
            "\nRecent behavioral context from evaluation:\n```\n"
            f"{behavioral_context}\n"
            "```\n\n"
        )
    if structured_feedback:
        structured_feedback_block = (
            "\nStructured verifier analysis:\n```\n"
            f"{structured_feedback}\n"
            "```\n\n"
        )
    if error_history:
        error_history_block = (
            "\nRecent verification history across this run:\n```\n"
            f"{error_history}\n"
            "```\n\n"
        )
    user_prompt = VERIFICATION_ERROR_REFINEMENT_PROMPT.format(
        task_description=task_description,
        allowed_helpers_block=_build_allowed_helpers_block(allowed_helpers),
        tool_reference_block=_build_tool_reference_block(allowed_helpers),
        decoding_surface_block=_build_decoding_surface_block(force_open_span),
        previous_strategy=previous_strategy,
        error_message=error_message,
        strategy_context_block=strategy_context_block,
        structured_feedback_block=structured_feedback_block,
        error_history_block=error_history_block,
        behavioral_context_block=behavioral_context_block,
        verified_examples=_build_verified_examples_block(allowed_helpers),
        search_memory_block=search_memory_block,
    )
    return SYSTEM_PROMPT, user_prompt


def build_runtime_error_prompt(
    previous_strategy: str,
    error_traceback: str,
    task_description: str = "Unknown task",
    search_memory: str = "",
    allowed_helpers: list[str] | None = None,
    force_open_span: bool = False,
) -> tuple[str, str]:
    search_memory_block = f"{search_memory}\n" if search_memory else ""
    user_prompt = RUNTIME_ERROR_REFINEMENT_PROMPT.format(
        task_description=task_description,
        allowed_helpers_block=_build_allowed_helpers_block(allowed_helpers),
        tool_reference_block=_build_tool_reference_block(allowed_helpers),
        decoding_surface_block=_build_decoding_surface_block(force_open_span),
        previous_strategy=previous_strategy,
        error_traceback=error_traceback,
        search_memory_block=search_memory_block,
    )
    return SYSTEM_PROMPT, user_prompt


def build_compilation_error_prompt(
    previous_strategy: str,
    error_message: str,
    search_memory: str = "",
    allowed_helpers: list[str] | None = None,
    force_open_span: bool = False,
) -> tuple[str, str]:
    search_memory_block = f"{search_memory}\n" if search_memory else ""
    user_prompt = COMPILATION_ERROR_REFINEMENT_PROMPT.format(
        allowed_helpers_block=_build_allowed_helpers_block(allowed_helpers),
        tool_reference_block=_build_tool_reference_block(allowed_helpers),
        decoding_surface_block=_build_decoding_surface_block(force_open_span),
        previous_strategy=previous_strategy,
        error_message=error_message,
        search_memory_block=search_memory_block,
    )
    return SYSTEM_PROMPT, user_prompt


def build_format_repair_prompt(
    previous_strategy: str,
    search_memory: str = "",
    allowed_helpers: list[str] | None = None,
    force_open_span: bool = False,
) -> tuple[str, str]:
    search_memory_block = f"{search_memory}\n" if search_memory else ""
    user_prompt = FORMAT_REPAIR_PROMPT.format(
        allowed_helpers_block=_build_allowed_helpers_block(allowed_helpers),
        tool_reference_block=_build_tool_reference_block(allowed_helpers),
        decoding_surface_block=_build_decoding_surface_block(force_open_span),
        previous_strategy=previous_strategy,
        search_memory_block=search_memory_block,
    )
    return SYSTEM_PROMPT, user_prompt


def _build_best_so_far_block(
    best_strategy: str | None,
    best_accuracy: float | None,
    best_syntax_rate: float | None,
) -> str:
    """Render the best-so-far strategy as a positive anchor.

    Shown in refinement mode only when the previous attempt is NOT the
    best-so-far — i.e. the previous attempt was a regression and the model
    should build from the better-scoring lineage instead. Empty otherwise,
    so the prompt stays minimal when there is only one strategy to discuss.
    """
    if (
        not best_strategy
        or best_accuracy is None
        or best_syntax_rate is None
    ):
        return ""
    return (
        "\n## Best result so far\n\n"
        f"```dafny\n{best_strategy}\n```\n\n"
        f"Result: accuracy {best_accuracy:.1%}, "
        f"syntax {best_syntax_rate:.1%}.\n"
        "The previous attempt regressed from this; consider building from "
        "this strategy instead.\n"
    )


def _build_eval_budget_block(eval_max_seconds_per_example: float | None) -> str:
    if eval_max_seconds_per_example is None:
        return ""
    return (
        f"Each example: {eval_max_seconds_per_example:.0f}s wall-clock budget; "
        "over-budget examples score 0.\n"
    )


def build_evaluation_failure_prompt(
    task_description: str,
    previous_strategy: str,
    previous_accuracy: float,
    previous_syntax_rate: float,
    num_examples: int,
    goal_accuracy: float,
    goal_syntax_rate: float,
    evaluation_feedback: str,
    best_strategy: str | None = None,
    best_accuracy: float | None = None,
    best_syntax_rate: float | None = None,
    search_memory: str = "",
    allowed_helpers: list[str] | None = None,
    eval_max_seconds_per_example: float | None = None,
    mode_examples: str = "",
    attempt_outcome_ledger: str = "",
    force_open_span: bool = False,
) -> tuple[str, str]:
    search_memory_block = f"{search_memory}\n" if search_memory else ""
    best_so_far_block = _build_best_so_far_block(
        best_strategy=best_strategy,
        best_accuracy=best_accuracy,
        best_syntax_rate=best_syntax_rate,
    )
    mode_examples_block = (
        f"\n## Concrete failing rollouts from prior attempt\n\n{mode_examples}\n"
        if mode_examples
        else ""
    )
    attempt_outcome_ledger_block = (
        f"\n## Attempt outcome ledger\n\n{attempt_outcome_ledger}\n"
        if attempt_outcome_ledger
        else ""
    )
    accuracy_gap_pp = max(0.0, (goal_accuracy - previous_accuracy) * 100.0)
    syntax_gap_pp = max(0.0, (goal_syntax_rate - previous_syntax_rate) * 100.0)
    user_prompt = EVALUATION_FAILURE_REFINEMENT_PROMPT.format(
        task_description=task_description,
        allowed_helpers_block=_build_allowed_helpers_block(allowed_helpers),
        tool_reference_block=_build_tool_reference_block(allowed_helpers),
        decoding_surface_block=_build_decoding_surface_block(force_open_span),
        previous_strategy=previous_strategy,
        previous_accuracy=previous_accuracy,
        previous_syntax_rate=previous_syntax_rate,
        num_examples=num_examples,
        goal_accuracy=goal_accuracy,
        goal_syntax_rate=goal_syntax_rate,
        accuracy_gap_pp=accuracy_gap_pp,
        syntax_gap_pp=syntax_gap_pp,
        evaluation_feedback=evaluation_feedback,
        attempt_outcome_ledger_block=attempt_outcome_ledger_block,
        mode_examples_block=mode_examples_block,
        verified_examples=_build_verified_examples_block(allowed_helpers),
        search_memory_block=search_memory_block,
        best_so_far_block=best_so_far_block,
        eval_budget_block=_build_eval_budget_block(eval_max_seconds_per_example),
    )
    return SYSTEM_PROMPT, user_prompt
