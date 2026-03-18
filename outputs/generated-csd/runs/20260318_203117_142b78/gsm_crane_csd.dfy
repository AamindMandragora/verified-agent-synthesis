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
    lm.ValidTokensIdsLogitsAlways();
    generated := [];
    var stepsLeft := maxSteps;
    // CSD_RATIONALE_BEGIN
// This strategy uses a simple loop to generate tokens based on whether the generated text is inside a delimited window or not. It switches to constrained generation when inside the window and uses unconstrained generation otherwise. The loop continues until a valid prefix is found or the maximum number of steps is reached.
// CSD_RATIONALE_END

lm.ValidTokensIdsLogitsAlways();
generated := [];

var hasValidPrefix := false;

while stepsLeft > 0 && !parser.IsCompletePrefix(generated)
  invariant lm.ValidTokensIdsLogits()
  invariant 0 <= stepsLeft <= maxSteps
  invariant |generated| + stepsLeft <= maxSteps
  invariant helpers.ConstrainedWindowValid(generated)
  decreases stepsLeft
{
  if helpers.InsideDelimitedWindow(generated) && !parser.IsCompletePrefix(helpers.GetDelimitedContent(generated))
  {
    // Constrained window: generate tokens within the delimited window
    var next: Token;
    var newSteps: nat;
    next, newSteps := helpers.ConstrainedStep(prompt, generated, stepsLeft);
    generated := generated + [next];
    stepsLeft := newSteps;
    hasValidPrefix := true;
  }
  else
  {
    // Unconstrained window: generate tokens outside the delimited window
    var next: Token;
    var newSteps: nat;
    next, newSteps := helpers.UnconstrainedStep(prompt, generated, stepsLeft);
    generated := generated + [next];
    stepsLeft := newSteps;
  }

  // Loop invariants
}
if !hasValidPrefix
{
  generated := helpers.RollbackToValidPrefix(generated);
}
    remainingSteps := stepsLeft;
  }
}
