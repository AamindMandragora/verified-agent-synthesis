include "../library/VerifiedAgentSynthesis.dfy"

// Reference reconstruction: hard-masked constrained steps inside << ... >> (CRANE-style).
module ReferenceCraneCSD {
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
    requires insideConstrained ==> generatedPrefix[|generatedPrefix| - |currentConstrained|..] == currentConstrained
    requires "<<" in lm.Tokens && ">>" in lm.Tokens
    requires eosToken in lm.Tokens
    requires helpers.cost == 0
    ensures lm.ValidTokensIdsLogits()
    ensures |generated| <= |generatedPrefix| + maxSteps
    ensures !insideConstrainedOut ==> currentConstrainedOut == []
    ensures insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)
    ensures cost <= maxSteps
    ensures cost == helpers.cost
    ensures maxSteps == 0 || cost > 0 || generated != generatedPrefix ||
            insideConstrainedOut != insideConstrained ||
            currentConstrainedOut != currentConstrained

  {
    var g := generatedPrefix;
    var inside := insideConstrained;
    var cur := currentConstrained;

    if maxSteps == 0 {
      generated := g;
      insideConstrainedOut := inside;
      currentConstrainedOut := if inside then cur else [];
      cost := helpers.cost;
      return;
    }

    while helpers.cost < maxSteps
      invariant lm.ValidTokensIdsLogits()
      invariant |g| <= |generatedPrefix| + helpers.cost
      invariant !inside ==> cur == []
      invariant inside ==> parser.IsValidPrefix(cur)
      invariant inside ==> |cur| <= |g|
      invariant inside ==> g[|g| - |cur|..] == cur
      invariant 0 <= helpers.cost <= maxSteps
      decreases maxSteps - helpers.cost
    {
      if inside && parser.IsCompletePrefix(cur) {
        g, inside, cur := helpers.CloseConstrainedSpan(lm, parser, g, cur);
      } else if !inside {
        var next := helpers.UnconstrainedStep(lm, prompt, g);
        g := g + [next];
        if next == eosToken {
          break;
        } else if next == "<<" {
          inside := true;
          cur := [];
        }
      } else {
        var next :=
          helpers.GroupBoostedConstrainedStep(lm, parser, prompt, cur, [], 0.0, eosToken);
        if next == eosToken {
          break;
        }
        g, inside, cur := helpers.AppendConstrainedToken(lm, parser, g, cur, next);
      }
    }

    generated := g;
    insideConstrainedOut := inside;
    currentConstrainedOut := if inside then cur else [];
    cost := helpers.cost;
  }
}
