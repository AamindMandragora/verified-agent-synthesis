include "../library/VerifiedAgentSynthesis.dfy"

// Reference reconstruction: constrained adaptive rejection sampling (CARS).
//
// Models the full CARS algorithm within a single strategy invocation:
//   1.  constrain_first — the first constrained token is hard-masked.
//   2.  Exploration — subsequent tokens are sampled unconstrained
//       (SoftConstrainedStep with zero boost, matching CARS first-pass
//       behaviour on new trie nodes where grammar bitmask is stored but
//       not applied).
//   3.  Rejection — if the unconstrained token violates the grammar the
//       attempt is rejected, the failing token is recorded, and the
//       constrained suffix is rolled back to the span entry point
//       (models CARS raising ValueError and restarting from the trie root).
//   4.  Exploitation — on retries the accumulated rejected tokens are
//       penalised and generation is hard-masked (SafePenalizedConstrainedStep),
//       modelling CARS revisited trie nodes where log_theta carries both
//       the grammar bitmask and accumulated failure penalties.
//   5.  Termination — the span closes only when the MODEL signals a stop
//       (samples eos or ">>") over a complete molecule; eos over an
//       incomplete molecule is a rejected sample. CARS never force-stops a
//       molecule: sample length is the model's choice, and completion is
//       checked at the model's own eos. (An earlier version of this file
//       closed the span at the first complete prefix, which collapsed every
//       answer to the shortest valid molecule — 35/50 answers were "<<C>>".)
module ReferenceCarsCSD {
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
    ensures maxSteps == 0 || cost > 0 || generated != generatedPrefix ||
            insideConstrainedOut != insideConstrained ||
            currentConstrainedOut != currentConstrained

  {
    var g := generatedPrefix;
    var inside := insideConstrained;
    var cur := currentConstrained;

    var rejectedTokens: seq<Token> := [];
    var spanEntryLen := if inside then |g| - |cur| else 0;
    // Set when the model has signalled a stop (eos or ">>") over a complete
    // molecule; the span is closed at the top of the next iteration so each
    // iteration still costs exactly one step.
    var closeRequested := false;
    // eos ends the whole sample (CARS: accepted sample -> done); ">>" only
    // closes the span and generation continues unconstrained.
    var stopAfterClose := false;

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
      invariant inside ==> spanEntryLen <= |generatedPrefix| + helpers.cost
      invariant closeRequested ==> inside && parser.IsCompletePrefix(cur)
      decreases maxSteps - helpers.cost
    {
      if inside && closeRequested {
        // The model signalled a stop over a complete molecule last step.
        g, inside, cur := helpers.CloseConstrainedSpan(lm, parser, g, cur);
        closeRequested := false;
        if stopAfterClose {
          break;
        }
      } else if !inside {
        var next := helpers.UnconstrainedStep(lm, prompt, g);
        g := g + [next];
        if next == eosToken {
          break;
        } else if next == "<<" {
          inside := true;
          cur := [];
          spanEntryLen := |g|;
          rejectedTokens := [];
        }
      } else if |cur| == 0 {
        // ---- constrain_first ------------------------------------------------
        // CARS always grammar-masks the first constrained token (root trie node
        // with constrain_first=True on first visit, grammar bitmask in log_theta
        // on revisits).
        var next := helpers.ConstrainedStep(lm, parser, prompt, cur, eosToken);
        if next == eosToken {
          break;
        }
        g, inside, cur := helpers.AppendConstrainedToken(lm, parser, g, cur, next);
      } else if |rejectedTokens| == 0 {
        // ---- exploration (first pass) ----------------------------------------
        // On new trie nodes at learn_level >= 3, CARS stores the grammar bitmask
        // in log_theta but does NOT apply it to scores (adjust_scores is False
        // for non-root new nodes).  Generation is effectively unconstrained.
        // Model this with SoftConstrainedStep(boost=0): sample from the raw LM
        // distribution, then check grammar validity.
        var next: Token;
        var isValid: bool;
        next, isValid := helpers.SoftConstrainedStep(
          lm, parser, prompt, cur, 0.0, eosToken
        );
        if next == eosToken {
          // The model wants to stop. CARS accepts the sample iff the
          // accumulated molecule parses completely; otherwise the sample is
          // rejected and eos is penalised at this node.
          if parser.IsCompletePrefix(cur) {
            closeRequested := true;
            stopAfterClose := true;
          } else {
            rejectedTokens := rejectedTokens + [next];
            g := g[..spanEntryLen];
            cur := [];
          }
        } else if isValid {
          g, inside, cur := helpers.AppendConstrainedToken(lm, parser, g, cur, next);
        } else if next == ">>" && parser.IsCompletePrefix(cur) {
          // The model closed the span itself over a complete molecule.
          closeRequested := true;
        } else {
          // ---- rejection (trie learning) ------------------------------------
          // CARS sets log_theta[failing_token] = -inf at the current trie node
          // and back-propagates through _recompute_in_trie, then raises
          // ValueError to abort the sample.  Model the token-level penalty as
          // an addition to rejectedTokens and the sample abort as a rollback
          // to the span entry point.
          rejectedTokens := rejectedTokens + [next];
          g := g[..spanEntryLen];
          cur := [];
        }
      } else {
        // ---- exploitation (revisited trie nodes) ----------------------------
        // On revisited nodes adjust_scores is True and scores += log_theta,
        // which contains both the grammar bitmask (-inf for invalid tokens)
        // and accumulated failure penalties.  SafePenalizedConstrainedStep
        // replicates this: hard grammar mask + penalty on rejected tokens.
        var next := helpers.SafePenalizedConstrainedStep(
          lm, parser, prompt, cur, rejectedTokens, 100000000.0, eosToken
        );
        if next == eosToken {
          // Same stop rule as exploration: accept only a complete molecule,
          // otherwise reject the sample and penalise eos.
          if parser.IsCompletePrefix(cur) {
            closeRequested := true;
            stopAfterClose := true;
          } else {
            rejectedTokens := rejectedTokens + [next];
            g := g[..spanEntryLen];
            cur := [];
          }
        } else {
          g, inside, cur := helpers.AppendConstrainedToken(lm, parser, g, cur, next);
        }
      }
    }

    generated := g;
    insideConstrainedOut := inside;
    currentConstrainedOut := if inside then cur else [];
    cost := helpers.cost;
  }
}
