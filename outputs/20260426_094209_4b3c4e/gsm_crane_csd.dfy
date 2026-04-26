include "VerifiedAgentSynthesis.dfy"

module GeneratedCSD {
  import opened VerifiedDecoderAgent

  method MyCSDStrategy(lm: LM, parser: Parser, prompt: Prefix, maxSteps: nat, eosToken: Token) returns (generated: Prefix, remainingSteps: nat)
    modifies lm.Logits
    requires lm.ValidTokensIdsLogits()
    requires parser.IsValidPrefix([])
    requires !parser.IsCompletePrefix([])
    requires forall t: Token :: t in parser.ValidNextTokens([]) ==> t in lm.Tokens
    requires maxSteps >= 2
    requires LeftDelimiter in lm.Tokens
    requires RightDelimiter in lm.Tokens
    ensures lm.ValidTokensIdsLogits()
    ensures |generated| <= maxSteps
    ensures remainingSteps >= 0 && remainingSteps <= maxSteps
  {
    var helpers := new CSDHelpers(lm, parser);
    lm.ValidTokensIdsLogitsAlways();
    generated := [];
    var stepsLeft := maxSteps;
    var phase := 0;
    var reasoning_tokens := 0;
    var constrained_tokens := 0;
    var reasoning_budget := 8;
    if stepsLeft < 16 {
      reasoning_budget := (stepsLeft / 4);
    }
    while ((stepsLeft > 0) && (phase < 3))
      invariant lm.ValidTokensIdsLogits()
      invariant 0 <= stepsLeft <= maxSteps
      invariant 0 <= reasoning_tokens
      invariant 0 <= constrained_tokens
      invariant 0 <= phase <= 3
      invariant |generated| + stepsLeft <= maxSteps
      decreases stepsLeft
    {
      if ((phase == 0) && (reasoning_tokens < reasoning_budget) && (stepsLeft > 2)) {
        generated, stepsLeft := helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft);
        reasoning_tokens := (reasoning_tokens + 1);
        if ((reasoning_tokens >= reasoning_budget) || (stepsLeft <= 2)) {
          phase := 1;
        }
      } else {
        if phase == 0 {
          generated, stepsLeft := helpers.AppendLeftDelimiter(generated, stepsLeft);
          phase := 2;
        } else {
          if phase == 1 {
            generated, stepsLeft := helpers.AppendLeftDelimiter(generated, stepsLeft);
            phase := 2;
          } else {
            if ((phase == 2) && (helpers.CanConstrain(generated))) {
              generated, stepsLeft := helpers.AppendConstrainedStep(prompt, generated, stepsLeft);
              constrained_tokens := (constrained_tokens + 1);
            } else {
              if ((phase == 2) && (!helpers.CanConstrain(generated))) {
                generated, stepsLeft := helpers.AppendRightDelimiter(generated, stepsLeft);
                phase := 3;
              }
            }
          }
        }
      }
    }
    remainingSteps := stepsLeft;
  }

}