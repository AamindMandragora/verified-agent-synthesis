// =============================================================================
// GeneratedCSD.dfy - Template for QWEN-generated Constrained Decoding (CSD) Strategies
// =============================================================================
//
// Purpose:
//   This file is a **template** where QWEN generates the body of a single
//   constrained decoding strategy. The Dafny verifier will check correctness.
//
// =============================================================================

include "VerifiedAgentSynthesis.dfy"

module GeneratedCSD {
  import opened VerifiedDecoderAgent

  // =============================================================================
  // === Method Skeleton ===
  //
  // QWEN must ONLY generate the **contents of this method**. Do NOT redefine
  // the method signature, preconditions, or postconditions.
  //
  // Requirements for your code inside this method:
  //   1. Maintain lm.ValidTokensIdsLogits() at all times.
  //   2. All generated sequences must satisfy parser.IsValidPrefix(generated).
  //   3. Sequence must be complete if |generated| < maxSteps:
  //        |generated| == maxSteps || parser.IsCompletePrefix(generated)
  //   4. Only valid tokens at each step:
  //        parser.ValidNextToken(generated[..i], generated[i])
  //   5. Length of sequence must not exceed maxSteps.
  //   6. Use only provided helpers functions.
  //
  // =============================================================================

  // =============================================================================
  // === Available Helper Functions for QWEN CSD Synthesis ===
  // =============================================================================
  //
  // LM class (language model):
  //   - lm.ValidTokensIdsLogits(): predicate ensuring token/logit consistency
  //   - lm.IdToToken(id: Id) -> Token
  //   - lm.TokenToId(token: Token) -> Id
  //   - lm.IdToLogit(id: Id) -> Logit
  //   - lm.TokenToLogit(token: Token) -> Logit
  //   - lm.IdsToLogits(ids: seq<Id>) -> seq<Logit>
  //   - lm.TokensToLogits(tokens: seq<Token>) -> seq<Logit>
  //   - lm.MaskToken(token: Token)
  //   - lm.MaskTokens(tokens: seq<Token>)
  //   - lm.MaskTokensExcept(tokens: seq<Token>)
  //   - lm.IsMasked(token: Token): predicate
  //   - lm.HasUnmaskedToken(): predicate
  //   - lm.GenerateLogits(input: Prefix)
  //   - lm.ChooseNextToken() -> Token
  //
  // Parser class:
  //   - parser.IsValidPrefix(prefix: Prefix): predicate
  //   - parser.IsCompletePrefix(prefix: Prefix): predicate
  //   - parser.IsDeadPrefix(prefix: Prefix): predicate
  //   - parser.ValidNextToken(prefix: Prefix, token: Token): predicate
  //   - parser.ValidNextTokens(prefix: Prefix) -> seq<Token>  // extern
  //
  // CSDHelpers class:
  //   - CSDHelpers.UnconstrainedStep(lm, prompt, generated) -> Token
  //   - CSDHelpers.ConstrainedStep(lm, parser, prompt, generated) -> Token
  //   - CSDHelpers.UnconstrainedGeneration(lm, prompt, maxSteps) -> seq<Token>
  //   - CSDHelpers.ConstrainedGeneration(lm, parser, prompt, maxSteps) -> seq<Token>
  //
  // =============================================================================
  // Notes:
  //   - Only use these helpers for manipulating tokens, logits, and generating sequences.
  //   - Do NOT modify lm.Logits directly outside of LM methods or CSDHelpers.
  //   - All generated sequences must maintain lm.ValidTokensIdsLogits() and respect parser constraints.
  // =============================================================================

  method MyCSDStrategy(lm: LM, parser: Parser, prompt: Prefix, maxSteps: nat) returns (generated: Prefix)
    modifies lm.Logits
    requires lm.ValidTokensIdsLogits()
    requires parser.IsValidPrefix([])
    requires forall t: Token :: t in parser.ValidNextTokens([]) ==> t in lm.Tokens
    ensures lm.ValidTokensIdsLogits()
    ensures |generated| <= maxSteps
    ensures parser.IsValidPrefix(generated)
    ensures |generated| == maxSteps || parser.IsCompletePrefix(generated)
  {
    // QWEN_INSERT_STRATEGY_HERE
  }
}