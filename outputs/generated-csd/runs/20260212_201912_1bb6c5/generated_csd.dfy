include "VerifiedAgentSynthesis.dfy"

module GeneratedCSD {
  import opened VerifiedDecoderAgent

  method MyCSDStrategy(lm: LM, parser: Parser, prompt: Prefix, maxSteps: nat) returns (generated: Prefix, cost: int)
    modifies lm.Logits
    requires lm.ValidTokensIdsLogits()
    requires parser.IsValidPrefix([])
    requires forall t: Token :: t in parser.ValidNextTokens([]) ==> t in lm.Tokens
    requires "<<" in lm.Tokens && ">>" in lm.Tokens
    ensures lm.ValidTokensIdsLogits()
    ensures |generated| <= maxSteps
    ensures parser.IsValidPrefix(generated)
    ensures |generated| == maxSteps || parser.IsCompletePrefix(generated)
    
  {
    var helpers := new CSDHelpers();
    // CSD_RATIONALE_BEGIN
// I chose HybridGeneration for a math problem because it balances the need for strict parsing
// with occasional creative exploration to potentially find a solution quickly.
// HybridGeneration allows the model to generate reasoning segments within << and >>,
// providing a structured approach while still allowing for some freedom if necessary.
// CSD_RATIONALE_END

generated := helpers.HybridGeneration(lm, parser, prompt, maxSteps);
cost := helpers.cost;
    cost := helpers.cost;
  }
}
