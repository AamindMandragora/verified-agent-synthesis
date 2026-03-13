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
// This implementation uses a constrained decoding strategy to generate a response to a logical entailment question. 
// It starts with an empty generated sequence and continues to generate tokens until a complete prefix is formed or the maximum number of steps is reached.
// The strategy uses ConstrainedStep to ensure that the generated sequence remains a valid prefix under the grammar and can be completed by the parser.
// The loop invariant ensures that the current state of the sequence and stepsLeft are consistent with the requirements.
// The output is constrained to one of three possible answers: True, False, or Uncertain, as determined by the parser's validation functions.
// CSD_RATIONALE_END

generated := [];
while stepsLeft > 0 && !parser.IsCompletePrefix(generated)
  invariant lm.ValidTokensIdsLogits()
  invariant parser.IsValidPrefix(generated)
  invariant 0 <= stepsLeft <= maxSteps
  invariant |generated| + stepsLeft == maxSteps
  decreases stepsLeft
{
  CSDHelpers.RollbackPreservesTokenInvariant(lm, parser, generated);
  var next, newSteps := helpers.ConstrainedStep(lm, parser, prompt, generated, stepsLeft);
  generated := generated + [next];
  stepsLeft := newSteps;
}
// template then assigns remainingSteps := stepsLeft;
    remainingSteps := stepsLeft;
  }
}
