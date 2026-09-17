    // Reference reconstruction: faithful IterGen constrained decoding (v3).
    //
    // Models the real IterGen eval loop (itergen/case_studies/sql/
    // eval_sql_seed334_test300.py:79-108 + itergen/main.py forward/backward):
    //
    //   1. Greedy grammar-masked decoding. Every constrained token is chosen
    //      greedily from grammar-masked logits. No unconstrained exploration
    //      (contrast with CARS).
    //
    //   2. SCHEMA-grounded symbol-boundary backtrack - the real trigger. After
    //      each column_name/table_name grammar symbol completes, real IterGen
    //      checks the identifier against the schema parsed from the prompt and on
    //      a miss calls backward() to that symbol and regenerates. The whole
    //      constrained phase is delegated to RegenerateUnitOnGroundingFailure,
    //      which implements that mechanism: DeadEndAvoidingStep decode,
    //      CompletedSchemaSymbolCount unit-boundary detection, ground-check
    //      against the prompt-derived support set, rollback to the last grounded
    //      checkpoint, and a persistent penalty on the out-of-schema identifier.
    //
    //   3. Recurrence penalty 0.3 (PenalizeTriedTokenAt; CSD_RECURRENCE_FLAT=1
    //      for parity legs).
    //
    //   4. Bounded retries: maxRollbackBudget=10 (global backtrack cap, IterGen's
    //      backwards_limit) and maxRetries=10 per symbol.
    //
    //   Known approximations (documented, not fixable without new helpers):
    //   - DeadEndAvoidingStep uses lookahead-8 dead-end avoidance; real IterGen is
    //     plain masked greedy and relies on backtracking to escape dead ends.
    //   - Real IterGen caps unit-completions at max_iter=20 per query; the helper
    //     caps total tokens (budget) instead.
    if maxSteps == 0 { return; }

    // One step is held back so a span can still be opened at the end if the model
    // never emitted "<<" (the contract requires an opener in the output).
    while helpers.cost + 1 < maxSteps
      invariant lm.ValidTokensIdsLogits()
      invariant |generated| <= |generatedPrefix| + helpers.cost
      invariant Tied(parser, generated, insideConstrainedOut, currentConstrainedOut)
      invariant !insideConstrainedOut ==> currentConstrainedOut == []
      invariant insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)
      invariant 0 <= helpers.cost && helpers.cost + 1 <= maxSteps
      decreases maxSteps - helpers.cost
    {
      if !insideConstrainedOut {
        // Outside a span: unconstrained decoding until "<<" or eos. Dead code in
        // spider token-0 mode (start_inside_constrained=True) but keeps the
        // strategy total over all inputs.
        var next := helpers.UnconstrainedStep(lm, prompt, generated);
        generated := generated + [next];
        if next == "<<" {
          insideConstrainedOut := true;
          currentConstrainedOut := [];
        } else if next == eosToken {
          break;
        }
      } else {
        // The whole constrained phase is one grounded-regeneration call: greedy
        // masked decode; on each completed column_name/table_name symbol,
        // ground-check against the prompt schema; on a miss, roll back to the last
        // grounded checkpoint with a persistent x0.3 penalty on the offending
        // token (real IterGen's backward() + recurrence_penalty). One step of the
        // budget is reserved so the closing ">>" always fits.
        var budget := maxSteps - helpers.cost;
        var newCur := helpers.RegenerateUnitOnGroundingFailure(
          lm, parser, prompt, currentConstrainedOut, eosToken, budget - 1, 10, 10);
        ReplaceTied(parser, generated, currentConstrainedOut, newCur);
        generated := generated[..|generated| - |currentConstrainedOut|] + newCur;
        currentConstrainedOut := newCur;
        // IterGen's stopping rule: the model stopped (eos) or the budget ran out.
        // Close the span (cosmetic ">>") when the query parses.
        if parser.IsCompletePrefix(currentConstrainedOut) && helpers.cost < maxSteps {
          generated, insideConstrainedOut, currentConstrainedOut :=
            helpers.CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut);
        }
        break;
      }
    }

    if !insideConstrainedOut && "<<" !in generated {
      generated, insideConstrainedOut, currentConstrainedOut :=
        helpers.OpenConstrainedSpan(lm, generated);
    }
    cost := helpers.cost;
