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
    while (stepsLeft > 0 && !parser.IsCompletePrefix(generated))
      invariant lm.ValidTokensIdsLogits()
      invariant 0 <= stepsLeft <= maxSteps
      invariant |generated| + stepsLeft <= maxSteps
      invariant helpers.ConstrainedWindowValid(generated)
      decreases stepsLeft
    {
      var next_token, new_steps := helpers.UnconstrainedStep(prompt, generated, stepsLeft);
      generated := (generated + [next_token]);
      stepsLeft := new_steps;
    }
    remainingSteps := stepsLeft;
  }

}