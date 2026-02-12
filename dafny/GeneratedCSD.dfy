include "VerifiedAgentSynthesis.dfy"

module GeneratedCSD {
  import opened VerifiedDecoderAgent

  method MyCSDStrategy(lm: LM, parser: Parser, prompt: Prefix, maxSteps: nat, eosToken: Token) returns (generated: Prefix, cost: int)
    modifies lm.Logits
    requires lm.ValidTokensIdsLogits()
    requires parser.IsValidPrefix([])
    requires forall t: Token :: t in parser.ValidNextTokens([]) ==> t in lm.Tokens
    requires "<<" in lm.Tokens && ">>" in lm.Tokens
    ensures lm.ValidTokensIdsLogits()
    ensures |generated| <= maxSteps
    
  {
    var helpers := new CSDHelpers();
    // CSD_RATIONALE_BEGIN
    // I chose CraneGeneration for the GSM-Symbolic dataset because it matches the 
    // CRANE-style windowing requirement: unconstrained reasoning segments outside
    // of << and >> delimiters, and constrained math expressions inside them.
    // CSD_RATIONALE_END

    generated := helpers.CraneGeneration(lm, parser, prompt, maxSteps, 50, eosToken);
    cost := helpers.cost;
  }
}
