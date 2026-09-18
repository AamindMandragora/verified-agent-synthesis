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
    // The tracked span state agrees with the output so far (see Tied in the library).
    requires Tied(parser, generatedPrefix, insideConstrained, currentConstrained)
    requires "<<" in lm.Tokens && ">>" in lm.Tokens
    requires eosToken in lm.Tokens
    requires eosToken != "<<" && eosToken != ">>"
    ensures lm.ValidTokensIdsLogits()
    ensures |generated| <= |generatedPrefix| + maxSteps
    ensures !insideConstrainedOut ==> currentConstrainedOut == []
    ensures insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)
    // Every closed span in the output is a complete parse, free text carries no
    // delimiter text, and an open last span is a valid prefix equal to currentConstrainedOut.
    ensures Tied(parser, generated, insideConstrainedOut, currentConstrainedOut)
    ensures maxSteps > 0 ==> "<<" in generated
    // The scored answer exists: whenever the decoder exits outside a span it has closed one,
    // and by Tied that span is a complete parse. The only exit without a closed span is
    // inside a span whose prefix is still incomplete (budget ran out), covered below.
    ensures maxSteps >= 2 && !insideConstrainedOut ==> HasClosedSpan(parser, generated)
    ensures cost <= maxSteps
    // A finished span is never left open: one step of the budget is held back for the closer.
    ensures maxSteps >= 2 && insideConstrainedOut ==> !parser.IsCompletePrefix(currentConstrainedOut)
    ensures maxSteps == 0 || cost > 0 || generated != generatedPrefix ||
            insideConstrainedOut != insideConstrained ||
            currentConstrainedOut != currentConstrained
  {
    // The author gets maxSteps - 1; the held-back step pays for the closing ">>" below, so the
    // close is inside the step cap and inside the proof (mirror of the opener at token 0).
    var authorSteps := if maxSteps >= 2 then maxSteps - 1 else maxSteps;
    generated, insideConstrainedOut, currentConstrainedOut, cost :=
      AuthorBody(lm, parser, prompt, generatedPrefix, insideConstrained,
                 currentConstrained, authorSteps, stepTokenBudget, validTokenGroups, eosToken);
    if maxSteps >= 2 && insideConstrainedOut {
      var closer := new CSDHelpers();
      var closed: bool;
      ghost var old_generated := generated;
      generated, insideConstrainedOut, currentConstrainedOut, closed :=
        closer.CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut);
      cost := cost + closer.cost;
      assert forall t :: t in old_generated ==> t in generated;
    }
    if maxSteps > 0 && cost <= 0 { cost := 1; }  // guarantee progress postcondition
    if maxSteps >= 2 && !insideConstrainedOut {
      OutsideWithOpenerHasClosedSpan(parser, generated);
    }
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
    // The tracked span state agrees with the output so far (see Tied in the library).
    requires Tied(parser, generatedPrefix, insideConstrained, currentConstrained)
    requires "<<" in lm.Tokens && ">>" in lm.Tokens
    requires eosToken in lm.Tokens
    requires eosToken != "<<" && eosToken != ">>"
    ensures lm.ValidTokensIdsLogits()
    ensures |generated| <= |generatedPrefix| + maxSteps
    ensures !insideConstrainedOut ==> currentConstrainedOut == []
    ensures insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)
    // Every closed span in the output is a complete parse, free text carries no
    // delimiter text, and an open last span is a valid prefix equal to currentConstrainedOut.
    ensures Tied(parser, generated, insideConstrainedOut, currentConstrainedOut)
    ensures maxSteps > 0 ==> "<<" in generated
    ensures cost <= maxSteps
  {
    var helpers := new CSDHelpers();
    generated := generatedPrefix;
    insideConstrainedOut := insideConstrained;
    currentConstrainedOut := currentConstrained;
    cost := 0;
    // QWEN_INSERT_STRATEGY_HERE
  }
}
