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
// This strategy uses a constrained decoding approach to generate output that includes plain text,
// followed by a Prover9 grammar formula enclosed in "<< >>". The constrained decoder respects the grammar
// only between the delimiters. It uses a combination of unconstrained and constrained steps to ensure the
// grammar is maintained and the correct number of steps is taken.
// CSD_RATIONALE_END

generated := [];
while stepsLeft > 0 && !parser.IsCompletePrefix(generated)
  invariant lm.ValidTokensIdsLogits()
  invariant parser.IsValidPrefix(generated)
  invariant 0 <= stepsLeft <= maxSteps
  invariant |generated| + stepsLeft == maxSteps
  decreases stepsLeft
{
  // Rollback to a valid prefix before starting a constrained step
// Determine if we can use a constrained step
  if !parser.IsCompletePrefix(generated)
  {
    // Call ConstrainedStep to generate a token that respects the grammar
    CSDHelpers.RollbackPreservesTokenInvariant(lm, parser, generated);
    var next, newSteps := helpers.ConstrainedStep(lm, parser, prompt, generated, stepsLeft);
    generated := generated + [next];
    stepsLeft := newSteps;
  }
  else
  {
    // If we reach here, it means the generated prefix is complete
    // We need to append the closing delimiter " >>" and terminate the loop
    generated := generated + [" >>"];
    stepsLeft := stepsLeft - 1; // Consume the final step
  }
}
// template then assigns remainingSteps := stepsLeft;
    remainingSteps := stepsLeft;
  }
}
