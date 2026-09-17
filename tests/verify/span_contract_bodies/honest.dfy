    if maxSteps == 0 { return; }
    while helpers.cost + 1 < maxSteps
      invariant lm.ValidTokensIdsLogits()
      invariant |generated| <= |generatedPrefix| + helpers.cost
      invariant Tied(parser, generated, insideConstrainedOut, currentConstrainedOut)
      invariant 0 <= helpers.cost && helpers.cost + 1 <= maxSteps
      decreases maxSteps - helpers.cost
    {
      if insideConstrainedOut && parser.IsCompletePrefix(currentConstrainedOut) {
        generated, insideConstrainedOut, currentConstrainedOut := helpers.CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut);
      } else if !insideConstrainedOut {
        var next := helpers.UnconstrainedStep(lm, prompt, generated);
        if next == eosToken { break; }
        generated := generated + [next];
        if next == "<<" { insideConstrainedOut := true; currentConstrainedOut := []; }
      } else {
        var next := helpers.GroupBoostedConstrainedStep(lm, parser, prompt, currentConstrainedOut, [], 0.0, eosToken);
        if next == eosToken { break; }
        generated, insideConstrainedOut, currentConstrainedOut := helpers.AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, next);
      }
    }
    if !insideConstrainedOut && "<<" !in generated {
      generated, insideConstrainedOut, currentConstrainedOut := helpers.OpenConstrainedSpan(lm, generated);
    }
    cost := helpers.cost;
