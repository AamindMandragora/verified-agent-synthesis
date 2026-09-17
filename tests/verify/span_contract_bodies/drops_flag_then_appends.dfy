    if maxSteps < 3 || insideConstrainedOut { return; }
    generated, insideConstrainedOut, currentConstrainedOut := helpers.OpenConstrainedSpan(lm, generated);
    // Debangshu's counterexample: drop the span flag and append anything.
    insideConstrainedOut := false;
    currentConstrainedOut := [];
    var next := helpers.UnconstrainedStep(lm, prompt, generated);
    generated := generated + [next];
    cost := helpers.cost;
