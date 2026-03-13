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
// This strategy uses a constrained decoding approach to generate FOL formulas. 
// It alternates between unconstrained and constrained steps based on the parser's ability to 
// complete a statement and the presence of << >> segments. The constrained window ensures that 
// each premise and conclusion are valid and well-formed, while unconstrained steps allow for free text.
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

  if (!parser.IsDeadPrefix(generated))
  {
    // Use ConstrainedStep for << >> segments
    CSDHelpers.RollbackPreservesTokenInvariant(lm, parser, generated);
    var next, newSteps := helpers.ConstrainedStep(lm, parser, prompt, generated, stepsLeft);
    generated := generated + [next];
    stepsLeft := newSteps;
  }
  else
  {
    // Use UnconstrainedStep for free text
    CSDHelpers.RollbackPreservesTokenInvariant(lm, parser, generated);
    var next, newSteps := helpers.UnconstrainedStep(lm, prompt, generated, stepsLeft);
    generated := generated + [next];
    stepsLeft := newSteps;
  }
}

remainingSteps := stepsLeft;
    remainingSteps := stepsLeft;
  }
}
