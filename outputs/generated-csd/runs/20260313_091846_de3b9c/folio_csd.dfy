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
// This CSD strategy generates FOL formulas by using quantifiers, predicates, and logical connectives.
// It ensures that the generated prefix is a valid FOL statement and follows the specified rules.
// The strategy uses a constrained approach where it alternates between unconstrained and constrained steps.
// The unconstrained steps generate random tokens, while constrained steps ensure the prefix remains a valid FOL statement.
// The strategy terminates when the maximum number of steps is reached or a complete FOL statement is generated.
// CSD_RATIONALE_END

generated := [];

while stepsLeft > 0 && !parser.IsCompletePrefix(generated)
  invariant lm.ValidTokensIdsLogits()
  invariant parser.IsValidPrefix(generated)
  invariant 0 <= stepsLeft <= maxSteps
  invariant |generated| + stepsLeft == maxSteps
  decreases stepsLeft
{
  // Rollback to a valid prefix to ensure the next step maintains the valid next tokens invariant
// Generate the next token based on the current generated prefix and the remaining steps
  CSDHelpers.RollbackPreservesTokenInvariant(lm, parser, generated);
  var next, newSteps := helpers.ConstrainedStep(lm, parser, prompt, generated, stepsLeft);

  // Add the next token to the generated prefix
  generated := generated + [next];

  // Update the remaining steps
  stepsLeft := newSteps;
}
// The template then assigns remainingSteps := stepsLeft;
    remainingSteps := stepsLeft;
  }
}
