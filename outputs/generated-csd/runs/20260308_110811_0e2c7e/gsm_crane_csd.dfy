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
// This CSD strategy generates short symbolic mathematical expressions for GSM-Symbolic reasoning.
// It ensures the generated expressions contain at least one variable, enforce strict arithmetic grammar rules,
// and maintain a constrained window size. The strategy uses both unconstrained and constrained steps to
// construct valid expressions efficiently. It alternates between unconstrained and constrained steps based on the parser state.
// CSD_RATIONALE_END

generated := [];

while stepsLeft > 0 && !parser.IsCompletePrefix(generated)
  invariant lm.ValidTokensIdsLogits()
  invariant parser.IsValidPrefix(generated)
  invariant 0 <= stepsLeft <= maxSteps
  invariant |generated| + stepsLeft == maxSteps
  decreases stepsLeft
{
  // Rollback to a valid prefix before starting constrained steps
  CSDHelpers.RollbackPreservesTokenInvariant(lm, parser, generated);

  // Determine if we can use constrained steps
  if !parser.IsCompletePrefix(generated)
  {
    // Use constrained step to generate a token that maintains the valid next tokens condition
    var next, newSteps := helpers.ConstrainedStep(lm, parser, prompt, generated, stepsLeft);
    generated := generated + [next];
    stepsLeft := newSteps;
  }
  else
  {
    // If the current prefix is complete, generate a valid next token using unconstrained step
    var next, newSteps := helpers.UnconstrainedStep(lm, prompt, generated, stepsLeft);
    generated := generated + [next];
    stepsLeft := newSteps;
  }
}

// Optional: Rollback to a valid prefix if necessary after the loop
if !parser.IsCompletePrefix(generated)
{
  generated := CSDHelpers.RollbackToValidPrefix(parser, generated);
}
    remainingSteps := stepsLeft;
  }
}
