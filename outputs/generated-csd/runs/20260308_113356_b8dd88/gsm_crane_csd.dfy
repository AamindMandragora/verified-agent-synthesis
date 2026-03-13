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
// This CSD strategy aims to generate short, valid arithmetic expressions using the given parser and constraints.
// It ensures that every expression contains at least one variable, follows the grammar rules, and is complete.
// The strategy uses ConstrainedStep to ensure that the prefix is complete and follows the grammar constraints.
// It also uses RollbackToValidPrefix to repair the prefix if necessary.
// The loop continues until the maximum number of steps is reached or the prefix is complete.
// CSD_RATIONALE_END

generated := [];

while stepsLeft > 0 && !parser.IsCompletePrefix(generated)
  invariant lm.ValidTokensIdsLogits()
  invariant parser.IsValidPrefix(generated)
  invariant 0 <= stepsLeft <= maxSteps
  invariant |generated| + stepsLeft == maxSteps
  decreases stepsLeft
{
  // Rollback to a valid prefix if the current one is invalid
  if !parser.IsValidPrefix(generated) {
    generated := helpers.RollbackToValidPrefix(parser, generated);
  }

  // Generate the next token using ConstrainedStep if the prefix is incomplete
  if !parser.IsCompletePrefix(generated) {
    CSDHelpers.RollbackPreservesTokenInvariant(lm, parser, generated);
    var next, newSteps := helpers.ConstrainedStep(lm, parser, prompt, generated, stepsLeft);
    generated := generated + [next];
    stepsLeft := newSteps;
  } else {
    // If the prefix is complete, stop the loop
    break;
  }
}
    remainingSteps := stepsLeft;
  }
}
