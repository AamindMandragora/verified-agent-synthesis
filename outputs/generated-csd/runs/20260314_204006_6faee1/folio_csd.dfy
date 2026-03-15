include "VerifiedAgentSynthesis.dfy"

module GeneratedCSD {
  import opened VerifiedDecoderAgent

  method MyCSDStrategy(lm: LM, parser: Parser, prompt: Prefix, maxSteps: nat, eosToken: Token) returns (generated: Prefix, remainingSteps: nat)
    modifies lm.Logits
    requires lm.ValidTokensIdsLogits()
    requires parser.IsValidPrefix([])
    requires forall t: Token :: t in parser.ValidNextTokens([]) ==> t in lm.Tokens
    requires "«" in lm.Tokens && "»" in lm.Tokens
    ensures lm.ValidTokensIdsLogits()
    ensures |generated| <= maxSteps
    ensures remainingSteps >= 0 && remainingSteps <= maxSteps
  {
    var helpers := new CSDHelpers();
    var stepsLeft := maxSteps;
    // CSD_RATIONALE_BEGIN
// This strategy uses a combination of UnconstrainedStep and ConstrainedStep to generate a prefix that starts with plain text,
// followed by the left delimiter, a single Prover9-formula, and then the right delimiter. UnconstrainedStep is used for the plain text
// part, and ConstrainedStep is used for the formula segment to enforce the Prover9 grammar.
// CSD_RATIONALE_END

generated := [];
while stepsLeft > 0 && !parser.IsCompletePrefix(generated)
  invariant lm.ValidTokensIdsLogits()
  invariant parser.IsValidPrefix(generated)
  invariant 0 <= stepsLeft <= maxSteps
  invariant |generated| + stepsLeft == maxSteps
  decreases stepsLeft
{
  if (parser.IsCompletePrefix(generated))
  {
    // If the generated prefix is complete, start a new unconstrained step to add more plain text
    var next, newSteps := helpers.UnconstrainedStep(lm, prompt, generated, stepsLeft);
    generated := generated + [next];
    stepsLeft := newSteps;
  }
  else
  {
    // If the generated prefix is not complete, continue with a constrained step to add a formula
    CSDHelpers.RollbackPreservesTokenInvariant(lm, parser, generated);
    var next, newSteps := helpers.ConstrainedStep(lm, parser, prompt, generated, stepsLeft);
    generated := generated + [next];
    stepsLeft := newSteps;
  }
}
    remainingSteps := stepsLeft;
  }
}
