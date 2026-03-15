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
// This strategy uses a Crane-style approach where we first generate plain text using UnconstrainedStep
// until the delimiter is encountered. Once the delimiter is found, we switch to ConstrainedStep
// to generate the formula within the << >> segment. We continue this process until all steps are used
// or the parser indicates that the prefix is complete.
// CSD_RATIONALE_END

generated := [];

while stepsLeft > 0 && !parser.IsCompletePrefix(generated)
  invariant lm.ValidTokensIdsLogits()
  invariant parser.IsValidPrefix(generated)
  invariant 0 <= stepsLeft <= maxSteps
  invariant |generated| + stepsLeft == maxSteps
  decreases stepsLeft
{
  var next: Token; var newSteps: nat;
  if parser.IsCompletePrefix(generated)
  {
    // If we are already past the delimiter, use UnconstrainedStep to continue generating plain text
    next, newSteps := helpers.UnconstrainedStep(lm, prompt, generated, stepsLeft);
  }
  else
  {
    // Otherwise, use ConstrainedStep to generate the formula within the << >> segment
    CSDHelpers.RollbackPreservesTokenInvariant(lm, parser, generated);
    next, newSteps := helpers.ConstrainedStep(lm, parser, prompt, generated, stepsLeft);
  }

  generated := generated + [next];
  stepsLeft := newSteps;
}
    remainingSteps := stepsLeft;
  }
}
