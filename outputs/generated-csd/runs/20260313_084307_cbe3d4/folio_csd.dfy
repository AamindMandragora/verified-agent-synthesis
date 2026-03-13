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
// This CSD strategy generates first-order logic (FOL) formulas using constrained decoding.
// It uses quantifiers {forall} and {exists} for variables, predicates starting with uppercase,
// and logical connectives {and}, {or}, {not}, {implies}, {iff}, {xor}.
// The constrained windows ensure complete FOL statements as premises and conclusions.
// Parentheses are used for grouping with operator precedence: iff < implies < xor < or < and < not.
// The strategy uses ConstrainedStep to generate the next token, ensuring the grammar is respected.
// The loop continues until the prefix is complete or stepsLeft is exhausted.
// CSD_RATIONALE_END

generated := [];
while stepsLeft > 0 && !parser.IsCompletePrefix(generated)
  invariant lm.ValidTokensIdsLogits()
  invariant parser.IsValidPrefix(generated)
  invariant 0 <= stepsLeft <= maxSteps
  invariant |generated| + stepsLeft == maxSteps
  decreases stepsLeft
{
  // Rollback to a valid prefix before taking a constrained step
// Use ConstrainedStep to generate the next token
  CSDHelpers.RollbackPreservesTokenInvariant(lm, parser, generated);
  var next, newSteps := helpers.ConstrainedStep(lm, parser, prompt, generated, stepsLeft);
  
  // Append the next token to the generated prefix
  generated := generated + [next];
  
  // Update the number of remaining steps
  stepsLeft := newSteps;
}
    remainingSteps := stepsLeft;
  }
}
