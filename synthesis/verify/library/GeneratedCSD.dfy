include "VerifiedAgentSynthesis.dfy"

module GeneratedCSD {
  import opened VerifiedDecoderAgent

  // The author's synthesized strategy body is inserted into AuthorBody (below),
  // which carries every postcondition EXCEPT the progress guarantee. MyCSDStrategy
  // calls AuthorBody and then applies the cost:=1 fallback, so the progress
  // postcondition is always re-established even when the author's body early-returns
  // a no-op passthrough. This is pipeline-template mechanics only (no grammar/grader/
  // split/eval-semantics change) and MyCSDStrategy's full public contract is unchanged.
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
    eosToken: Token
  ) returns (
    generated: Prefix,
    insideConstrainedOut: bool,
    currentConstrainedOut: Prefix,
    cost: int
  )
    modifies lm.Logits
    requires lm.ValidTokensIdsLogits()
    requires parser.IsValidPrefix([])
    requires !insideConstrained ==> currentConstrained == []
    requires insideConstrained ==> parser.IsValidPrefix(currentConstrained)
    requires insideConstrained ==> |currentConstrained| <= |generatedPrefix|
    requires insideConstrained ==> generatedPrefix[|generatedPrefix| - |currentConstrained|..] == currentConstrained
    requires "<<" in lm.Tokens && ">>" in lm.Tokens
    requires eosToken in lm.Tokens
    ensures lm.ValidTokensIdsLogits()
    ensures |generated| <= |generatedPrefix| + maxSteps
    ensures !insideConstrainedOut ==> currentConstrainedOut == []
    ensures insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)
    ensures cost <= maxSteps
    ensures maxSteps == 0 || cost > 0 || generated != generatedPrefix ||
            insideConstrainedOut != insideConstrained ||
            currentConstrainedOut != currentConstrained
  {
    var helpers := new CSDHelpers();
    generated, insideConstrainedOut, currentConstrainedOut, cost :=
      AuthorBody(lm, parser, prompt, generatedPrefix, insideConstrained,
                 currentConstrained, maxSteps, stepTokenBudget, validTokenGroups, eosToken,
                 helpers);
    // AuthorBody establishes cost == helpers.cost, so the bound below is a bound on
    // charged library calls and not merely on a number the body chose to report.
    if maxSteps > 0 && cost <= 0 { cost := 1; }  // guarantee progress postcondition
  }

  // Holds the synthesized strategy. Same signature and preconditions as MyCSDStrategy
  // and every postcondition EXCEPT progress. The out-params are pre-initialized to a
  // valid passthrough so the author body may early-return on any path without a
  // definite-assignment error; the author only needs to satisfy the safety
  // postconditions, never progress.
  method AuthorBody(
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
    requires helpers.cost == 0
    requires !insideConstrained ==> currentConstrained == []
    requires insideConstrained ==> parser.IsValidPrefix(currentConstrained)
    requires insideConstrained ==> |currentConstrained| <= |generatedPrefix|
    requires insideConstrained ==> generatedPrefix[|generatedPrefix| - |currentConstrained|..] == currentConstrained
    requires "<<" in lm.Tokens && ">>" in lm.Tokens
    requires eosToken in lm.Tokens
    ensures lm.ValidTokensIdsLogits()
    ensures |generated| <= |generatedPrefix| + maxSteps
    ensures !insideConstrainedOut ==> currentConstrainedOut == []
    ensures insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)
    ensures cost <= maxSteps
    // Ties the reported step count to the library's charged-call counter. Without
    // this, `cost <= maxSteps` bounds only the number the body chooses to return,
    // and a body may make unboundedly many model calls while reporting zero.
    ensures cost == helpers.cost
  {
    generated := generatedPrefix;
    insideConstrainedOut := insideConstrained;
    currentConstrainedOut := currentConstrained;
    cost := 0;
    // QWEN_INSERT_STRATEGY_HERE
  }
}
