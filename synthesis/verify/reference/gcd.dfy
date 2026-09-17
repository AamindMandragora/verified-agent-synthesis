    // Reference reconstruction: greedy constrained decoding (GCD / SynCode-style).
    // Every model-chosen token is grammar-constrained via hard masking. There is no
    // unconstrained reasoning: the strategy immediately opens a constrained span and
    // stays there until the parse is complete. Delimiters (<< / >>) are emitted by
    // helpers (not by the model), keeping extraction uniform across benchmarks.
    if maxSteps == 0 { return; }

    // Token 0: open a span if the runtime did not already start us inside one.
    if !insideConstrainedOut {
      generated, insideConstrainedOut, currentConstrainedOut :=
        helpers.OpenConstrainedSpan(lm, generated);
    } else {
      OpenSpanHasOpener(parser, generated, currentConstrainedOut);
    }
    assert "<<" in generated;

    while helpers.cost < maxSteps
      invariant lm.ValidTokensIdsLogits()
      invariant |generated| <= |generatedPrefix| + helpers.cost
      invariant Tied(parser, generated, insideConstrainedOut, currentConstrainedOut)
      invariant !insideConstrainedOut ==> currentConstrainedOut == []
      invariant insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)
      invariant "<<" in generated
      invariant 0 <= helpers.cost <= maxSteps
      decreases maxSteps - helpers.cost
    {
      if !insideConstrainedOut { break; }
      if parser.IsCompletePrefix(currentConstrainedOut) {
        // Parse is complete - close the span (emits >>) and stop.
        ghost var before := generated;
        generated, insideConstrainedOut, currentConstrainedOut :=
          helpers.CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut);
        assert "<<" in before;
        assert generated == before || generated == before + [">>"];
        break;
      }
      var next := helpers.ConstrainedStep(lm, parser, prompt, currentConstrainedOut, eosToken);
      if next == eosToken { break; }
      generated, insideConstrainedOut, currentConstrainedOut :=
        helpers.AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, next);
    }
    cost := helpers.cost;
