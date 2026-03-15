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
// This CSD strategy generates a valid prefix that starts with optional plain text,
// followed by a Prover9-style first-order logic formula, and ends with " >>".
// It uses the parser to ensure the generated prefix is grammatically correct within the specified constraints.
// The strategy alternates between unconstrained and constrained steps, depending on the presence of a complete formula.
// The loop continues until the maximum number of steps is reached or a complete formula is generated.
// The strategy ensures that << >> segments are constrained and the rest is unconstrained.
// CSD_RATIONALE_END

generated := [];

while stepsLeft > 0 && !parser.IsCompletePrefix(generated)
  invariant lm.ValidTokensIdsLogits()
  invariant parser.IsValidPrefix(generated)
  invariant 0 <= stepsLeft <= maxSteps
  invariant |generated| + stepsLeft == maxSteps
  decreases stepsLeft
{
  var validTokens := parser.ValidNextTokens(generated);
  if !parser.IsPermissive(generated) || stepsLeft == 0 {
    // Constrained step for << >> segments
    CSDHelpers.RollbackPreservesTokenInvariant(lm, parser, generated);
    var next, newSteps := helpers.ConstrainedStep(lm, parser, prompt, generated, stepsLeft);
    generated := generated + [next];
    stepsLeft := newSteps;
  } else {
    // Unconstrained step for plain text
    CSDHelpers.RollbackPreservesTokenInvariant(lm, parser, generated);
    var next, newSteps := helpers.UnconstrainedStep(lm, prompt, generated, stepsLeft);
    CSDHelpers.UnconstrainedPreservesValidWhenPermissive(parser, generated, next);
    generated := generated + [next];
    stepsLeft := newSteps;
  }
}
remainingSteps := stepsLeft;
    remainingSteps := stepsLeft;
  }
}
