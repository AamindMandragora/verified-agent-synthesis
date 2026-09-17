    // Faithful CRANE baseline: IterGen-unit adaptive GSM (unconstrained until "<<",
    // then forward(start) / view / valid_vars membership / backward), matching
    // CRANE generate_gsm_symbolic_with_itergen. Distinct from crane.dfy, which takes
    // one grammar-masked token at a time inside the span.
    //
    // This is the loop CSDHelpers.CraneGeneration runs, lifted into the body so the
    // span state it tracks is the contract's own (generated / insideConstrainedOut /
    // currentConstrainedOut). CraneGeneration itself generates from an empty prefix
    // and exposes no Tied postcondition, so it cannot be used under the span
    // contract. Decoding behaviour is unchanged: same ConfidenceGatedStep forward
    // over one "start" unit, same maxIter / backwardsLimit, same var grounding check.
    if maxSteps == 0 { return; }

    var unitIters := 0;
    var numBackwards := 0;
    var maxIter := 80;
    var backwardsLimit := 20;

    // One step is held back so a span can still be opened at the end if the model
    // never emitted "<<" (the contract requires an opener in the output).
    while helpers.cost + 1 < maxSteps
      invariant lm.ValidTokensIdsLogits()
      invariant |generated| <= |generatedPrefix| + helpers.cost
      invariant Tied(parser, generated, insideConstrainedOut, currentConstrainedOut)
      invariant !insideConstrainedOut ==> currentConstrainedOut == []
      invariant insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)
      invariant 0 <= helpers.cost && helpers.cost + 1 <= maxSteps
      invariant 0 <= unitIters <= maxIter + 1
      decreases maxSteps - helpers.cost, (if insideConstrainedOut then 1 else 0), (maxIter + 1 - unitIters)
    {
      if !insideConstrainedOut {
        var next := helpers.UnconstrainedStep(lm, prompt, generated);
        generated := generated + [next];
        if next == "<<" {
          // CRANE: `start_symbol in unconstrained_gen`.
          insideConstrainedOut := true;
          currentConstrainedOut := [];
          unitIters := 0;
          numBackwards := 0;
        } else if next == eosToken {
          break;
        }
      } else if RenderedEndsWith(currentConstrainedOut, ">>") {
        // The span's own content carried the closer (GSM grammar includes ">>").
        CloserIsTerminal(parser, currentConstrainedOut);
        CloseTied(parser, generated, currentConstrainedOut);
        insideConstrainedOut := false;
        currentConstrainedOut := [];
      } else if unitIters >= maxIter {
        // CRANE parity: do not flip to unconstrained mid-<<...>>; stop.
        break;
      } else {
        var spanStart := |generated| - |currentConstrainedOut|;
        var constrainedPrompt := prompt + generated[..spanStart];
        var budgetLeft := maxSteps - 1 - helpers.cost;
        var beforeCost := helpers.cost;
        // CRANE gsm_symbolic_constraints: forward(num=1) with default_unit=start
        // (one full <<expr>>), not per-var.
        var newCur := helpers.ForwardUntilSymbol(
          lm, parser, constrainedPrompt, currentConstrainedOut, eosToken, "start", 1, budgetLeft);
        unitIters := unitIters + 1;
        // ForwardUntilSymbol may emit more tokens than it charged steps for, so crop
        // the span to what the remaining length budget allows (CraneGeneration does
        // the same crop against maxSteps).
        assert helpers.cost >= beforeCost;
        if spanStart + |newCur| > |generatedPrefix| + helpers.cost {
          var keep := |generatedPrefix| + helpers.cost - spanStart;
          assert 0 <= keep <= |newCur|;
          ValidPrefixesAreClosed(parser, newCur, keep);
          newCur := newCur[..keep];
        }
        ReplaceTied(parser, generated, currentConstrainedOut, newCur);
        generated := generated[..|generated| - |currentConstrainedOut|] + newCur;
        currentConstrainedOut := newCur;
        if helpers.cost == beforeCost && !RenderedEndsWith(currentConstrainedOut, ">>") {
          // No progress: stop rather than going unconstrained mid-span.
          break;
        }
        if !RenderedEndsWith(currentConstrainedOut, ">>") {
          var lastVar := helpers.ViewLastSymbol(parser, currentConstrainedOut, "var");
          var allowed := helpers.IsAllowedVarText(validTokenGroups, lastVar);
          if lastVar != "" && !allowed {
            if numBackwards < backwardsLimit {
              var backCur := helpers.BackwardToSymbol(parser, currentConstrainedOut, "var", 1);
              ReplaceTied(parser, generated, currentConstrainedOut, backCur);
              generated := generated[..|generated| - |currentConstrainedOut|] + backCur;
              currentConstrainedOut := backCur;
              numBackwards := numBackwards + 1;
            } else {
              numBackwards := 0;
            }
          }
        }
      }
    }

    if !insideConstrainedOut && "<<" !in generated {
      generated, insideConstrainedOut, currentConstrainedOut :=
        helpers.OpenConstrainedSpan(lm, generated);
    }
    cost := helpers.cost;
