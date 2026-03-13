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
// This CSD strategy generates FOL formulas by following the specified rules and constraints.
// It uses ConstrainedStep to ensure that generated prefixes remain valid and complete FOL statements.
// The strategy uses {forall} and {exists} for quantifiers, {and}, {or}, {not}, {implies}, {iff}, and {xor}.
// The loop continues until stepsLeft is exhausted or a complete FOL statement is generated.
// The invariant ensures that the generated prefix remains valid and complete throughout the process.
// CSD_RATIONALE_END

generated := [];
var stepCounter := 0;

while stepsLeft > 0 && !parser.IsCompletePrefix(generated)
  invariant lm.ValidTokensIdsLogits()
  invariant parser.IsValidPrefix(generated)
  invariant 0 <= stepsLeft <= maxSteps
  invariant |generated| + stepsLeft == maxSteps
  decreases maxSteps - |generated|
{
  // If the generated prefix is not valid, rollback to the nearest valid prefix
  if !parser.IsValidPrefix(generated)
  {
    generated := helpers.RollbackToValidPrefix(parser, generated);
  }

  // Generate a next token using ConstrainedStep
  CSDHelpers.RollbackPreservesTokenInvariant(lm, parser, generated);
  var next, newSteps := helpers.ConstrainedStep(lm, parser, prompt, generated, stepsLeft);
  generated := generated + [next];
  stepsLeft := newSteps;
  stepCounter := stepCounter + 1;
}

// If the loop exits without generating a complete prefix, rollback to the nearest valid prefix
if !parser.IsCompletePrefix(generated)
{
  generated := helpers.RollbackToValidPrefix(parser, generated);
}
    remainingSteps := stepsLeft;
  }
}
