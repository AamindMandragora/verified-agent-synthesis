    // Reference reconstruction: pure unconstrained decoding (no grammar enforcement).
    //
    // Two honest concessions to the span contract, neither of which changes what the
    // model is allowed to sample:
    //   * On Spider/SMILES the runtime starts the body already inside a span
    //     (generatedPrefix == ["<<"]). This baseline enforces no grammar, so it drops
    //     that opener (rolling the span's content back to the entry point first) and
    //     decodes free text from there. Nothing the model produced is discarded: at
    //     token 0 the span content is empty.
    //   * The contract requires "<<" somewhere in the output whenever maxSteps > 0,
    //     so if the model never emits one, a single held-back step opens a span at the
    //     very end. It carries no content and no grammar mask is ever applied.
    if maxSteps == 0 { return; }

    if insideConstrainedOut {
      // Roll the open span back to its entry point, then drop the opener itself.
      var stable := generated[..|generated| - |currentConstrainedOut|];
      ReplaceTied(parser, generated, currentConstrainedOut, []);
      assert stable + [] == stable;
      generated := stable;
      currentConstrainedOut := [];
      assert Tied(parser, generated, true, []);
      generated := generated[..|generated| - 1];
      insideConstrainedOut := false;
    }
    assert Tied(parser, generated, false, []);

    while helpers.cost + 1 < maxSteps
      invariant lm.ValidTokensIdsLogits()
      invariant |generated| <= |generatedPrefix| + helpers.cost
      invariant Tied(parser, generated, insideConstrainedOut, currentConstrainedOut)
      invariant !insideConstrainedOut ==> currentConstrainedOut == []
      invariant insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)
      invariant 0 <= helpers.cost && helpers.cost + 1 <= maxSteps
      decreases maxSteps - helpers.cost
    {
      if insideConstrainedOut { break; }
      var next := helpers.UnconstrainedStep(lm, prompt, generated);
      generated := generated + [next];
      if next == "<<" {
        // The model wrote an opener itself; free-running past it would break the
        // span contract, so stop here and let the template close the span.
        insideConstrainedOut := true;
        currentConstrainedOut := [];
        break;
      }
      if next == eosToken { break; }
    }

    if !insideConstrainedOut && "<<" !in generated {
      generated, insideConstrainedOut, currentConstrainedOut :=
        helpers.OpenConstrainedSpan(lm, generated);
    }
    cost := helpers.cost;
