include "VerifiedAgentSynthesis.dfy"

module GeneratedCSD {
  import opened VerifiedDecoderAgent

  method MyCSDStrategy(lm: LM, parser: Parser, prompt: Prefix, maxSteps: nat, eosToken: Token) returns (generated: Prefix, remainingSteps: nat)
    modifies lm.Logits
    requires lm.ValidTokensIdsLogits()
    requires parser.IsValidPrefix([])
    requires forall t: Token :: t in parser.ValidNextTokens([]) ==> t in lm.Tokens
    requires LeftDelimiter in lm.Tokens && RightDelimiter in lm.Tokens
    ensures lm.ValidTokensIdsLogits()
    ensures |generated| <= maxSteps
    ensures remainingSteps >= 0 && remainingSteps <= maxSteps
  {
    var delim := new Delimiter(LeftDelimiter, RightDelimiter);
    var helpers := new CSDHelpers(lm, parser, delim);
    helpers.DelimitersInLMAlways();
    var stepsLeft := maxSteps;
    generated := [];
// CSD_RATIONALE_BEGIN
// Use a combination of unconstrained and constrained steps. Start with unconstrained
// steps for the initial part of the prompt, then switch to constrained steps once
// the delimited window is entered. If a valid prefix is found, stop early.
// This strategy ensures that the constrained part is correctly handled within the
// delim, and the parser enforces structural validity.
// CSD_RATIONALE_END

while (stepsLeft > 0 && !parser.IsCompletePrefix(generated))
  invariant lm.ValidTokensIdsLogits()
  invariant 0 <= stepsLeft <= maxSteps
  invariant |generated| + stepsLeft == maxSteps
  decreases stepsLeft
{
  // If we're inside the delimited window, use ConstrainedStep
  if (helpers.InsideDelimitedWindow(generated) && !parser.IsCompletePrefix(helpers.GetDelimitedContent(generated)))
  {
    var next, newSteps := helpers.ConstrainedStep(prompt, generated, stepsLeft);
    generated := generated + [next];
    stepsLeft := newSteps;
  }
  else
  {
    // Otherwise, use UnconstrainedStep
    var next, newSteps := helpers.UnconstrainedStep(prompt, generated, stepsLeft);
    generated := generated + [next];
    stepsLeft := newSteps;
  }
}
    remainingSteps := stepsLeft;
  }
}
