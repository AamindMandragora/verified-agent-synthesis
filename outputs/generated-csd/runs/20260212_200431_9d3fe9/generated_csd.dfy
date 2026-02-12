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
    ensures cost <= 2 * maxSteps
  {
    var helpers := new CSDHelpers();
    // CSD_RATIONALE_BEGIN
// I chose HybridGeneration to enable interleaved reasoning within << and >> delimiters while maintaining overall validity.
// This approach allows the model to generate creative and coherent responses that adhere to the problem's structure.
// CSD_RATIONALE_END
generated := helpers.HybridGeneration(lm, parser, prompt, maxSteps);
cost := helpers.cost;
    cost := helpers.cost;
  }
}
