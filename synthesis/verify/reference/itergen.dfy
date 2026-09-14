include "../library/VerifiedAgentSynthesis.dfy"

// Reference reconstruction: faithful IterGen constrained decoding (v3).
//
// Models the real IterGen eval loop (itergen/case_studies/sql/
// eval_sql_seed334_test300.py:79-108 + itergen/main.py forward/backward):
//
//   1. Greedy grammar-masked decoding. Every constrained token is chosen
//      greedily from grammar-masked logits. No unconstrained exploration
//      (contrast with CARS).
//
//   2. SCHEMA-grounded symbol-boundary backtrack — the real trigger. After
//      each `column_name`/`table_name` grammar symbol completes, real IterGen
//      checks the identifier against the schema parsed from the prompt
//      (exists_column/exists_table); on a miss it calls backward() to that
//      symbol and regenerates. v2 of this file used parser.IsDeadPrefix as
//      the trigger, which never fires under masked decoding — the rollback
//      was inert and the reference behaved as plain grammar-greedy (official
//      rescore 2026-07-03: 0.057 acc vs the 0.3767 IterGen 2B target, with
//      114/300 execution "Other Error" = exactly the ungrounded-identifier
//      failure class). v3 delegates the whole constrained phase to
//      RegenerateUnitOnGroundingFailure, which implements the faithful
//      mechanism: DeadEndAvoidingStep decode, CompletedSchemaSymbolCount
//      unit-boundary detection (= IterGen's SymbolPosMap boundary),
//      FirstUngroundedIdentifierTokenIdx ground-check against the
//      prompt-derived support set, rollback to the last grounded checkpoint,
//      and a persistent penalty on the out-of-schema identifier's own token.
//
//   3. Recurrence penalty 0.3. PenalizeTriedTokenAt's host implementation
//      applies ln(CSD_RECURRENCE_PENALTY=0.3) to the penalized token's logit
//      on every regeneration at that prefix — the multiplicative x0.3 of real
//      IterGen (recurrence_penalty=0.3). Run parity legs with
//      CSD_RECURRENCE_FLAT=1 (flat, once per distinct token — IterGen's
//      semantics) rather than the cumulative default.
//
//   4. Bounded retries. Real IterGen: backwards_limit=10 backtracks per
//      example (global), max_iter=20 forward/backward rounds. Mapped to
//      maxRollbackBudget=10 (global backtrack cap) and maxRetries=10 (per
//      symbol; real IterGen has no per-symbol cap, so the global 10 always
//      binds first and the per-symbol value never changes behaviour).
//
//   Known approximations (documented, not fixable without new helpers):
//   - DeadEndAvoidingStep uses lookahead-8 dead-end avoidance; real IterGen
//     is plain masked greedy and relies on backtracking to escape dead ends.
//   - Real IterGen caps unit-completions at max_iter=20 per query; the
//     helper caps total tokens (budget) instead.
//
// NOTE on the contract: unlike v2 this file does NOT carry the "progress"
// postcondition (cost > 0 || generated changed || ...).
// RegenerateUnitOnGroundingFailure intentionally exposes no lower bound on
// cost, so progress is not provable here. The Pattern-A template's
// MyCSDStrategy wrapper supplies progress via its cost:=1 fallback; the body
// below is inserted into AuthorBody, which (like this file) does not carry
// the progress postcondition.
module ReferenceIterGenCSD {
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

  {
    var g := generatedPrefix;
    var inside := insideConstrained;
    var cur := currentConstrained;

    // Position in g where the current constrained span started.
    var spanEntryLen := if inside then |g| - |cur| else 0;

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
      invariant inside ==> 0 <= spanEntryLen <= |g|
      invariant inside ==> spanEntryLen == |g| - |cur|
      invariant inside ==> spanEntryLen <= |generatedPrefix| + helpers.cost
      decreases maxSteps - helpers.cost
    {
      if !inside {
        // Outside a span: unconstrained decoding until "<<" or eos. Dead code
        // in spider token-0 mode (start_inside_constrained=True) but keeps the
        // strategy total over all inputs.
        var next := helpers.UnconstrainedStep(lm, prompt, g);
        g := g + [next];
        if next == eosToken {
          break;
        } else if next == "<<" {
          inside := true;
          cur := [];
          spanEntryLen := |g|;
        }
      } else {
        // The whole constrained phase is one grounded-regeneration call:
        // greedy masked decode; on each completed column_name/table_name
        // symbol, ground-check against the prompt schema; on a miss, roll
        // back to the last grounded checkpoint with a persistent x0.3
        // penalty on the offending token (real IterGen's backward() +
        // recurrence_penalty). One step of the budget is reserved so the
        // closing ">>" always fits.
        var budget := maxSteps - helpers.cost;
        var newCur := helpers.RegenerateUnitOnGroundingFailure(
          lm, parser, prompt, cur, eosToken, budget - 1, 10, 10
        );
        g := g[..spanEntryLen] + newCur;
        cur := newCur;
        // IterGen's stopping rule: the model stopped (eos) or budget ran
        // out. Close the span (cosmetic ">>") when the query parses.
        if parser.IsCompletePrefix(cur) && helpers.cost < maxSteps {
          g, inside, cur := helpers.CloseConstrainedSpan(lm, parser, g, cur);
        }
        break;
      }
    }

    generated := g;
    insideConstrainedOut := inside;
    currentConstrainedOut := if inside then cur else [];
    cost := helpers.cost;
  }
}
