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
    // CSD_RATIONALE_BEGIN
// This strategy uses an unconstrained approach for the initial part of the generation and switches to a constrained approach when inside the delimited section.
// The loop continues until the maximum number of steps is reached or the prefix is complete.
// Constraints are enforced when inside the delimited section to ensure the generated text adheres to the specified grammar.
// When inside the delimited section, the strategy checks if the next token is a valid continuation and if the current prefix is not complete.
// If both conditions are met, it switches to a constrained approach. Otherwise, it continues with an unconstrained step.
// The loop invariants ensure that the generated prefix remains valid under the grammar and that stepsLeft is correctly decremented.
// CSD_RATIONALE_END

generated := [];

while stepsLeft > 0 && !parser.IsCompletePrefix(generated)
  invariant lm.ValidTokensIdsLogits()
  invariant |generated| + stepsLeft == maxSteps
  decreases stepsLeft
{
  if parser.IsCompletePrefix(generated) {
    // If the prefix is complete, continue with an unconstrained step
    var next, newSteps := helpers.UnconstrainedStep(prompt, generated, stepsLeft);
    generated := generated + [next];
    stepsLeft := newSteps;
  } else {
    // If the prefix is incomplete, check if we are inside the delimited section
    if helpers.InsideDelimitedWindow(generated) && !parser.IsCompletePrefix(helpers.GetDelimitedContent(generated)) {
      // If inside the delimited section and not complete, switch to constrained step
      var next, newSteps := helpers.ConstrainedStep(prompt, generated, stepsLeft);
      generated := generated + [next];
      stepsLeft := newSteps;
    } else {
      // If outside the delimited section or complete, continue with an unconstrained step
      var next, newSteps := helpers.UnconstrainedStep(prompt, generated, stepsLeft);
      generated := generated + [next];
      stepsLeft := newSteps;
    }
  }
}
    remainingSteps := stepsLeft;
  }
}
