    // Reference reconstruction: constrained adaptive rejection sampling (CARS).
    //
    // Models the full CARS algorithm within a single strategy invocation:
    //   1.  constrain_first - the first constrained token is hard-masked.
    //   2.  Exploration - subsequent tokens are sampled unconstrained
    //       (SoftConstrainedStep with zero boost, matching CARS first-pass
    //       behaviour on new trie nodes where the grammar bitmask is stored but
    //       not applied).
    //   3.  Rejection - if the unconstrained token violates the grammar the
    //       attempt is rejected, the failing token is recorded, and the
    //       constrained suffix is rolled back to the span entry point
    //       (models CARS raising ValueError and restarting from the trie root).
    //   4.  Exploitation - on retries the accumulated rejected tokens are
    //       penalised and generation is hard-masked (SafePenalizedConstrainedStep),
    //       modelling CARS revisited trie nodes where log_theta carries both
    //       the grammar bitmask and accumulated failure penalties.
    //   5.  Termination - the span closes only when the MODEL signals a stop
    //       (samples eos or ">>") over a complete molecule; eos over an
    //       incomplete molecule is a rejected sample. CARS never force-stops a
    //       molecule.
    if maxSteps == 0 { return; }

    var rejectedTokens: seq<Token> := [];
    // Set when the model has signalled a stop (eos or ">>") over a complete
    // molecule; the span is closed at the top of the next iteration so each
    // iteration still costs exactly one step.
    var closeRequested := false;
    // eos ends the whole sample (CARS: accepted sample -> done); ">>" only
    // closes the span and generation continues unconstrained.
    var stopAfterClose := false;

    // One step is held back so a span can still be opened at the end if the model
    // never emitted "<<" (the contract requires an opener in the output).
    while helpers.cost + 1 < maxSteps
      invariant lm.ValidTokensIdsLogits()
      invariant |generated| <= |generatedPrefix| + helpers.cost
      invariant Tied(parser, generated, insideConstrainedOut, currentConstrainedOut)
      invariant !insideConstrainedOut ==> currentConstrainedOut == []
      invariant insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)
      invariant closeRequested ==> insideConstrainedOut && parser.IsCompletePrefix(currentConstrainedOut)
      invariant 0 <= helpers.cost && helpers.cost + 1 <= maxSteps
      decreases maxSteps - helpers.cost
    {
      if insideConstrainedOut && closeRequested {
        // The model signalled a stop over a complete molecule last step.
        generated, insideConstrainedOut, currentConstrainedOut :=
          helpers.CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut);
        closeRequested := false;
        if stopAfterClose { break; }
      } else if !insideConstrainedOut {
        var next := helpers.UnconstrainedStep(lm, prompt, generated);
        generated := generated + [next];
        if next == "<<" {
          insideConstrainedOut := true;
          currentConstrainedOut := [];
          rejectedTokens := [];
        } else if next == eosToken {
          break;
        }
      } else if |currentConstrainedOut| == 0 {
        // ---- constrain_first ------------------------------------------------
        // CARS always grammar-masks the first constrained token.
        var next := helpers.ConstrainedStep(lm, parser, prompt, currentConstrainedOut, eosToken);
        if next == eosToken { break; }
        generated, insideConstrainedOut, currentConstrainedOut :=
          helpers.AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, next);
      } else if |rejectedTokens| == 0 {
        // ---- exploration (first pass) ----------------------------------------
        // On new trie nodes CARS stores the grammar bitmask in log_theta but does
        // NOT apply it to scores. Model this with SoftConstrainedStep(boost=0):
        // sample from the raw LM distribution, then check grammar validity.
        var next: Token;
        var isValid: bool;
        next, isValid := helpers.SoftConstrainedStep(lm, parser, prompt, currentConstrainedOut, 0.0, eosToken);
        if next == eosToken {
          if parser.IsCompletePrefix(currentConstrainedOut) {
            closeRequested := true;
            stopAfterClose := true;
          } else {
            rejectedTokens := rejectedTokens + [next];
            var stable := generated[..|generated| - |currentConstrainedOut|];
            ReplaceTied(parser, generated, currentConstrainedOut, []);
            assert stable + [] == stable;
            generated := stable;
            currentConstrainedOut := [];
          }
        } else if isValid {
          generated, insideConstrainedOut, currentConstrainedOut :=
            helpers.AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, next);
        } else if next == ">>" && parser.IsCompletePrefix(currentConstrainedOut) {
          // The model closed the span itself over a complete molecule.
          closeRequested := true;
        } else {
          // ---- rejection (trie learning) ------------------------------------
          // CARS sets log_theta[failing_token] = -inf at the current trie node and
          // raises ValueError to abort the sample. Model the token-level penalty as
          // an addition to rejectedTokens and the sample abort as a rollback to the
          // span entry point.
          rejectedTokens := rejectedTokens + [next];
          var stable := generated[..|generated| - |currentConstrainedOut|];
          ReplaceTied(parser, generated, currentConstrainedOut, []);
          assert stable + [] == stable;
          generated := stable;
          currentConstrainedOut := [];
        }
      } else {
        // ---- exploitation (revisited trie nodes) ----------------------------
        // On revisited nodes scores += log_theta, which contains both the grammar
        // bitmask (-inf for invalid tokens) and accumulated failure penalties.
        var next := helpers.SafePenalizedConstrainedStep(
          lm, parser, prompt, currentConstrainedOut, rejectedTokens, 100000000.0, eosToken);
        if next == eosToken {
          if parser.IsCompletePrefix(currentConstrainedOut) {
            closeRequested := true;
            stopAfterClose := true;
          } else {
            rejectedTokens := rejectedTokens + [next];
            var stable := generated[..|generated| - |currentConstrainedOut|];
            ReplaceTied(parser, generated, currentConstrainedOut, []);
            assert stable + [] == stable;
            generated := stable;
            currentConstrainedOut := [];
          }
        } else {
          generated, insideConstrainedOut, currentConstrainedOut :=
            helpers.AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, next);
        }
      }
    }

    if !insideConstrainedOut && "<<" !in generated {
      generated, insideConstrainedOut, currentConstrainedOut :=
        helpers.OpenConstrainedSpan(lm, generated);
    }
    cost := helpers.cost;
