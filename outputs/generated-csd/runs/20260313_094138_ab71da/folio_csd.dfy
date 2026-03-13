include "VerifiedAgentSynthesis.dfy"

module GeneratedCSD {
  import opened VerifiedDecoderAgent

  method MyCSDStrategy(lm: LM, parser: Parser, prompt: Prefix, maxSteps: nat, eosToken: Token) returns (generated: Prefix, remainingSteps: nat)
    modifies lm.Logits
    requires lm.ValidTokensIdsLogits()
    requires parser.IsValidPrefix([])
    requires forall t: Token :: t in parser.ValidNextTokens([]) ==> t in lm.Tokens
    requires "<<" in lm.Tokens && ">>" in lm.Tokens
    ensures lm.ValidTokensIdsLogits()
    ensures |generated| <= maxSteps
    ensures remainingSteps >= 0 && remainingSteps <= maxSteps
  {
    var helpers := new CSDHelpers();
    var stepsLeft := maxSteps;
    // CSD_RATIONALE_BEGIN
// This CSD strategy generates FOL formulas using a constrained decoding approach.
// It uses {forall} and {exists} for quantifiers, predicates with uppercase names,
// and logical connectives. The strategy ensures that each generated prefix is
// valid according to the FOL grammar and that the prefix is complete after each step.
// The constrained windows contain complete FOL statements (premises and conclusion).
// The strategy alternates between unconstrained and constrained steps to ensure
// that the generated prefix remains valid and complete.
// CSD_RATIONALE_END

generated := [];

while stepsLeft > 0 && !parser.IsCompletePrefix(generated)
  invariant lm.ValidTokensIdsLogits()
  invariant parser.IsValidPrefix(generated)
  invariant 0 <= stepsLeft <= maxSteps
  invariant |generated| + stepsLeft == maxSteps
  decreases stepsLeft
{
  // Always try constrained step first
  CSDHelpers.RollbackPreservesTokenInvariant(lm, parser, generated);
  var next, newSteps := helpers.ConstrainedStep(lm, parser, prompt, generated, stepsLeft);

  // If constrained step was successful, append it to generated and update stepsLeft
  generated := generated + [next];
  stepsLeft := newSteps;
}
    remainingSteps := stepsLeft;
  }
}
