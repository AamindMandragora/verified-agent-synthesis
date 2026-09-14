include "../library/VerifiedAgentSynthesis.dfy"

// Faithful CRANE baseline: IterGen-unit adaptive GSM (unconstrained until <<,
// then forward(VARIABLE) / view / valid_vars membership / backward), matching
// CRANE generate_gsm_symbolic_with_itergen. Distinct from crane.dfy (exact
// token equality for << entry).
module ReferenceCraneFaithfulCSD {
  import opened VerifiedDecoderAgent

  method MyCSDStrategy(
    lm: LM,
    parser: Parser,
    prompt: Prefix,
    generatedPrefix: Prefix,
    insideConstrained: bool,
    currentConstrained: Prefix,
    maxSteps: nat,
    stepTokenBudget: nat,
    validTokenGroups: seq<seq<Token>>,
    eosToken: Token,
    helpers: CSDHelpers
  ) returns (
    generated: Prefix,
    insideConstrainedOut: bool,
    currentConstrainedOut: Prefix,
    cost: int
  )
    modifies lm.Logits, helpers
    requires lm.ValidTokensIdsLogits()
    requires parser.IsValidPrefix([])
    requires !insideConstrained ==> currentConstrained == []
    requires insideConstrained ==> parser.IsValidPrefix(currentConstrained)
    requires insideConstrained ==> |currentConstrained| <= |generatedPrefix|
    requires "<<" in lm.Tokens && ">>" in lm.Tokens
    requires eosToken in lm.Tokens
    requires helpers.cost == 0
    ensures lm.ValidTokensIdsLogits()
    ensures |generated| <= |generatedPrefix| + maxSteps
    ensures !insideConstrainedOut ==> currentConstrainedOut == []
    ensures insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)
    ensures cost <= maxSteps
    ensures cost == helpers.cost
  {
    generated := helpers.CraneGeneration(lm, parser, prompt, maxSteps, 10, validTokenGroups, eosToken);
    insideConstrainedOut := false;
    currentConstrainedOut := [];
    cost := helpers.cost;
  }
}
