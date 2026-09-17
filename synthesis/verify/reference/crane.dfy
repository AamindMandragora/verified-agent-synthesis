    // Reference reconstruction: free-text reasoning with hard-masked constrained
    // steps inside << ... >> (CRANE-style). Outside a span the model decodes
    // freely; the token "<<" switches it into a grammar-masked span, which closes
    // as soon as the parse is complete.
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
      if insideConstrainedOut && parser.IsCompletePrefix(currentConstrainedOut) {
        generated, insideConstrainedOut, currentConstrainedOut :=
          helpers.CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut);
      } else if !insideConstrainedOut {
        var next := helpers.UnconstrainedStep(lm, prompt, generated);
        generated := generated + [next];
        if next == "<<" {
          insideConstrainedOut := true;
          currentConstrainedOut := [];
        } else if next == eosToken {
          break;
        }
      } else {
        var next := helpers.GroupBoostedConstrainedStep(
          lm, parser, prompt, currentConstrainedOut, [], 0.0, eosToken);
        if next == eosToken { break; }
        generated, insideConstrainedOut, currentConstrainedOut :=
          helpers.AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, next);
      }
    }

    // The model never opened a span: emit the opener with the held-back step so the
    // output still carries one.
    if !insideConstrainedOut && "<<" !in generated {
      generated, insideConstrainedOut, currentConstrainedOut :=
        helpers.OpenConstrainedSpan(lm, generated);
    }
    cost := helpers.cost;
