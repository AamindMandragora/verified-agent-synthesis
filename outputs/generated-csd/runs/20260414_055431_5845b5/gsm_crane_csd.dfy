include "VerifiedAgentSynthesis.dfy"

module GeneratedCSD {
  import opened VerifiedDecoderAgent

  method MyCSDStrategy(lm: LM, parser: Parser, prompt: Prefix, maxSteps: nat, eosToken: Token) returns (generated: Prefix, remainingSteps: nat)
    modifies lm.Logits
    requires lm.ValidTokensIdsLogits()
    requires parser.IsValidPrefix([])
    requires !parser.IsCompletePrefix([])
    requires forall t: Token :: t in parser.ValidNextTokens([]) ==> t in lm.Tokens
    requires LeftDelimiter in lm.Tokens && RightDelimiter in lm.Tokens
    requires maxSteps >= 4
    ensures lm.ValidTokensIdsLogits()
    ensures |generated| <= maxSteps
    ensures remainingSteps >= 0 && remainingSteps <= maxSteps
    ensures DelimitedAnswerValidForParser(parser, generated)
  {
    var delim := new Delimiter(LeftDelimiter, RightDelimiter);
    var helpers := new CSDHelpers(lm, parser, delim);
    helpers.DelimitersInLMAlways();
    lm.ValidTokensIdsLogitsAlways();
    generated := [];
    var answer := [];
    var stepsLeft := (maxSteps - 2);
    var phase := 0;
    var preamble_tokens := 0;
    var exploration_budget := 1;
    var answer_tokens := 0;
    while ((stepsLeft > 0) && (!parser.IsCompletePrefix(answer)))
      invariant lm.ValidTokensIdsLogits()
      invariant 0 <= stepsLeft <= maxSteps - 2
      invariant 0 <= preamble_tokens <= 1
      invariant 0 <= exploration_budget <= 1
      invariant 0 <= answer_tokens
      invariant helpers.ConstrainedWindowValid(generated)
      invariant parser.IsValidPrefix(answer)
      invariant |generated| + |answer| + stepsLeft <= maxSteps - 2
      invariant |answer| == 0 ==> exploration_budget < stepsLeft
      decreases stepsLeft
    {
      var next_token := eosToken;
      var new_steps := stepsLeft;
      var spend_freeform := ((phase < 2) && (exploration_budget > 0) && (preamble_tokens < 1) && (stepsLeft > 1));
      if spend_freeform {
        next_token, new_steps := helpers.ExpressiveStep(prompt, generated, stepsLeft);
        generated := (generated + [next_token]);
        stepsLeft := new_steps;
        preamble_tokens := (preamble_tokens + 1);
        exploration_budget := (exploration_budget - 1);
        if preamble_tokens >= 1 {
          phase := 1;
        }
        if ((preamble_tokens >= 1) || (stepsLeft <= 1)) {
          phase := 2;
        }
      } else {
        next_token, new_steps := helpers.ConstrainedAnswerStep(prompt, generated, answer, stepsLeft);
        answer := (answer + [next_token]);
        stepsLeft := new_steps;
        answer_tokens := (answer_tokens + 1);
        if answer_tokens >= 1 {
          phase := 3;
        }
      }
    }
    helpers.FinalizeDelimitedAnswer(generated, answer);
    generated := (((generated + [LeftDelimiter]) + answer) + [RightDelimiter]);
    remainingSteps := stepsLeft;
  }

}