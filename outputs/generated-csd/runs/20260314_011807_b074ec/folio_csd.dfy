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
// The strategy for the FOLIO task is to generate plain text first using UnconstrainedStep,
// then insert the left delimiter " << ", followed by exactly one first-order logic formula
// using ConstrainedStep, and finally insert the right delimiter " >> ". We use UnconstrainedStep
// for the plain text part and ConstrainedStep within the << >> segment to ensure the formula
// adheres to the Prover9 grammar. The loop continues until the maximum number of steps is reached
// or the complete prefix is generated.
// CSD_RATIONALE_END

generated := [];
var stepCounter := 0;

while stepsLeft > 0 && !parser.IsCompletePrefix(generated)
  invariant lm.ValidTokensIdsLogits()
  invariant parser.IsValidPrefix(generated)
  invariant 0 <= stepsLeft <= maxSteps
  invariant |generated| + stepsLeft == maxSteps
  decreases stepsLeft
{
  if (parser.IsCompletePrefix(generated))
  {
    // Plain text generation using UnconstrainedStep
    var next, newSteps := helpers.UnconstrainedStep(lm, prompt, generated, stepsLeft);
    generated := generated + [next];
    stepsLeft := newSteps;
    stepCounter := stepCounter + 1;
  }
  else
  {
    // Constrained generation of the first-order logic formula
    CSDHelpers.RollbackPreservesTokenInvariant(lm, parser, generated);
    var next, newSteps := helpers.ConstrainedStep(lm, parser, prompt, generated, stepsLeft);
    generated := generated + [next];
    stepsLeft := newSteps;
    stepCounter := stepCounter + 1;
  }
}

    remainingSteps := stepsLeft;
  }
}
