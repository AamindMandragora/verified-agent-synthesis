from synthesis.feedback_loop import repair_verification_strategy


def test_repair_replaces_done_state_with_break():
    strategy = """current_step = "explore"
while stepsLeft > 0 and not parser.IsCompletePrefix(answer):
    if current_step == "explore":
        current_step = "done"
"""

    repaired, changed = repair_verification_strategy(
        strategy,
        "Dafny verification failed with 1 error(s):\n\n(Line 1, Column 1): Error: decreases expression might not decrease",
    )

    assert changed
    assert 'current_step = "done"' not in repaired
    assert repaired.endswith("        break") or "\n        break\n" in repaired


def test_repair_removes_unprovable_current_step_invariant():
    strategy = """# invariant current_step in ["explore", "generate"]
while stepsLeft > 0 and not parser.IsCompletePrefix(answer):
    pass
"""

    repaired, changed = repair_verification_strategy(
        strategy,
        "Dafny verification failed with 1 error(s):\n\n(Line 1, Column 1): Error: this invariant could not be proved to be maintained by the loop\ncurrent_step",
    )

    assert changed
    assert 'current_step in ["explore", "generate"]' not in repaired


def test_repair_injects_nonempty_answer_guard_for_finalize():
    strategy = """while stepsLeft > 0 and not parser.IsCompletePrefix(answer):
    break
"""

    repaired, changed = repair_verification_strategy(
        strategy,
        "Dafny verification failed with 2 error(s):\n\n"
        "(Line 1, Column 1): Error: a precondition for this call could not be proved\n"
        "helpers.FinalizeDelimitedAnswer(generated, answer);\n"
        "VerifiedAgentSynthesis.dfy(671,35): Related location: this is the precondition that could not be proved\n"
        "requires parser.IsValidPrefix(answer)\n"
        "VerifiedAgentSynthesis.dfy(672,24): Related location: this is the precondition that could not be proved\n"
        "requires |answer| > 0",
    )

    assert changed
    assert "if len(answer) == 0 and stepsLeft > 0 and not parser.IsCompletePrefix(answer):" in repaired
    assert "helpers.ConstrainedAnswerStep(prompt, generated, answer, stepsLeft)" in repaired


def test_repair_rewrites_prefix_truthiness_and_pop():
    strategy = """while stepsLeft > 0:
    if not helpers.LongestValidSuffix(generated):
        next_token, new_steps = helpers.UnconstrainedStep(prompt, generated, stepsLeft)
    else:
        next_token, new_steps = helpers.ConstrainedStep(prompt, generated, stepsLeft)
    if helpers.LongestValidSuffix(generated):
        generated.pop()
"""

    repaired, changed = repair_verification_strategy(
        strategy,
        "Dafny verification failed with 7 error(s):\n\n"
        "(Line 49, Column 20): Error: type seq<Token> does not have a member pop\n"
        "(Line 31, Column 9): Error: condition is expected to be of type bool, but is Prefix\n"
        "(Line 31, Column 36): Error: logical/bitwise negation expects a boolean or bitvector argument (instead got Prefix)\n",
    )

    assert changed
    assert "if len(helpers.LongestValidSuffix(generated)) == 0:" in repaired
    assert "if len(helpers.LongestValidSuffix(generated)) > 0:" in repaired
    assert "generated = generated[:-1]" in repaired


def test_repair_rewrites_budget_aware_threshold_keyword_to_positional():
    strategy = """if expressionDepth > 2:
    next_token, new_steps = helpers.BudgetAwareStep(prompt, generated, stepsLeft, threshold=5)
"""

    repaired, changed = repair_verification_strategy(
        strategy,
        "Dafny verification failed with 1 error(s):\n\n"
        "(Line 50, Column 56): Error: wrong number of arguments (got 3, but method 'BudgetAwareStep' expects 4: "
        "(prompt: Prefix, generated: Prefix, stepsLeft: nat, completionThreshold: nat))\n",
    )

    assert changed
    assert "helpers.BudgetAwareStep(prompt, generated, stepsLeft, 5)" in repaired


def test_repair_predeclares_next_token_for_mixed_tuple_step_assignment():
    strategy = """# CSD_RATIONALE_BEGIN
# Keep a short free-form phase before the constrained answer.
# CSD_RATIONALE_END
while stepsLeft > 0 and len(generated) < 3:
    next_token, stepsLeft = helpers.UnconstrainedStep(prompt, generated, stepsLeft)
    generated = generated + [next_token]
"""

    repaired, changed = repair_verification_strategy(
        strategy,
        "Dafny verification failed with 2 error(s):\n\n"
        "(Line 4, Column 5): Error: unresolved identifier: next_token\n"
        "(Line 5, Column 29): Error: unresolved identifier: next_token\n",
    )

    assert changed
    assert "# CSD_RATIONALE_END\nnext_token = eosToken\nwhile" in repaired


def test_repair_predeclares_discard_target_for_forced_token_steps():
    strategy = """# CSD_RATIONALE_BEGIN
# Emit delimiters explicitly.
# CSD_RATIONALE_END
new_steps = stepsLeft
_, new_steps = helpers.ForcedTokenStep(prompt, generated, LeftDelimiter, new_steps)
generated = generated + [LeftDelimiter]
"""

    repaired, changed = repair_verification_strategy(
        strategy,
        "Dafny verification failed with 1 error(s):\n\n"
        "(Line 5, Column 1): Error: unresolved identifier: _\n",
    )

    assert changed
    assert "# CSD_RATIONALE_END\n_ = eosToken\nnew_steps = stepsLeft" in repaired


def test_repair_adds_incomplete_prefix_guard_to_budgeted_loop():
    strategy = """while stepsLeft > 0:
    next_token, new_steps = helpers.ConstrainedStep(prompt, generated, stepsLeft)
    generated = generated + [next_token]
"""

    repaired, changed = repair_verification_strategy(
        strategy,
        "Dafny verification failed with 1 error(s):\n\n"
        "VerifiedAgentSynthesis.dfy(598,15): Related location: this is the precondition that could not be proved\n"
        "requires !parser.IsCompletePrefix(LongestValidSuffix(generated))\n",
    )

    assert changed
    assert (
        "while stepsLeft > 0 and not parser.IsCompletePrefix(helpers.LongestValidSuffix(generated)):"
        in repaired
    )


def test_repair_wraps_forced_token_step_with_budget_guard():
    strategy = """next_token, new_steps = helpers.ForcedTokenStep(prompt, generated, RightDelimiter, stepsLeft)
generated = generated + [next_token]
stepsLeft = new_steps
"""

    repaired, changed = repair_verification_strategy(
        strategy,
        "Dafny verification failed with 1 error(s):\n\n"
        "helpers.ForcedTokenStep(prompt, generated, RightDelimiter, stepsLeft);\n"
        "VerifiedAgentSynthesis.dfy(692,25): Related location: this is the precondition that could not be proved\n"
        "requires stepsLeft >= 1\n",
    )

    assert changed
    assert "if new_steps > 0:" in repaired or "if stepsLeft > 0:" in repaired
    assert "next_token, new_steps = helpers.ForcedTokenStep(prompt, generated, RightDelimiter, stepsLeft)" in repaired
    assert "generated = generated + [next_token]" in repaired
    assert "stepsLeft = new_steps" in repaired


def test_repair_rewrites_bare_forced_token_step_calls():
    strategy = """helpers.ForcedTokenStep(prompt, generated, LeftDelimiter, stepsLeft)
generated = generated + [LeftDelimiter]
"""

    repaired, changed = repair_verification_strategy(
        strategy,
        "Dafny verification failed with 1 error(s):\n\n"
        "(Line 1, Column 1): Error: the method returns 2 values but is assigned to 0 variable\n"
        "helpers.ForcedTokenStep(prompt, generated, LeftDelimiter, stepsLeft)\n",
    )

    assert changed
    assert "next_token, new_steps = helpers.ForcedTokenStep(prompt, generated, LeftDelimiter, stepsLeft)" in repaired
    assert "generated = generated + [next_token]" in repaired
    assert "stepsLeft = new_steps" in repaired
    assert "generated = generated + [LeftDelimiter]" not in repaired


def test_repair_budget_bounds_sentinel_while_loops_for_termination():
    strategy = """while not right_delimiter_emitted:
    next_token, stepsLeft = helpers.SoftConstrainedStep(prompt, generated, 0.5, stepsLeft)
"""

    repaired, changed = repair_verification_strategy(
        strategy,
        "Dafny verification failed with 1 error(s):\n\n"
        "(Line 1, Column 1): Error: cannot prove termination; try supplying a decreases clause for the loop\n",
    )

    assert changed
    assert "while stepsLeft > 0 and not right_delimiter_emitted:" in repaired


def test_repair_rewrites_soft_constrained_keywords_to_positional():
    strategy = """next_token, stepsLeft = helpers.SoftConstrainedStep(prompt, generated, penalty=0.1, stepsLeft=stepsLeft)
"""

    repaired, changed = repair_verification_strategy(
        strategy,
        "Dafny verification failed with 1 error(s):\n\n"
        "(Line 1, Column 1): Error: wrong number of arguments (got 2, but method 'SoftConstrainedStep' expects 4: "
        "(prompt: Prefix, generated: Prefix, penalty: Logit, stepsLeft: nat))\n",
    )

    assert changed
    assert "helpers.SoftConstrainedStep(" in repaired
    assert "penalty=" not in repaired
    assert "stepsLeft=" not in repaired


def test_repair_adds_can_constrain_guard_to_constrained_append_branch():
    strategy = """while stepsLeft > 0 and phase < 3:
    if phase == 1:
        generated, stepsLeft = helpers.AppendConstrainedStep(prompt, generated, stepsLeft)
"""

    repaired, changed = repair_verification_strategy(
        strategy,
        "Dafny verification failed with 1 error(s):\n\n"
        "VerifiedAgentSynthesis.dfy(760,27): Related location: this is the precondition that could not be proved\n"
        "requires CanConstrain(prefix)\n",
    )

    assert changed
    assert "if phase == 1 and helpers.CanConstrain(generated):" in repaired
    assert "helpers.AppendConstrainedStep(prompt, generated, stepsLeft)" in repaired


def test_repair_injects_standard_loop_invariants_for_lm_logit_preconditions():
    strategy = """while stepsLeft > 0:
    next_token, stepsLeft = helpers.UnconstrainedStep(prompt, generated, stepsLeft)
"""

    repaired, changed = repair_verification_strategy(
        strategy,
        "Dafny verification failed with 1 error(s):\n\n"
        "VerifiedAgentSynthesis.dfy(579,43): Related location: this is the precondition that could not be proved\n"
        "requires this.lm.ValidTokensIdsLogits()\n",
    )

    assert changed
    assert "# invariant lm.ValidTokensIdsLogits()" in repaired
    assert "# invariant 0 <= stepsLeft <= maxSteps" in repaired
    assert "# invariant |generated| + stepsLeft <= maxSteps" in repaired
    assert "# decreases stepsLeft" in repaired


def test_repair_replaces_unprovable_soft_penalty_with_positive_literal():
    strategy = """next_token, stepsLeft = helpers.SoftConstrainedStep(prompt, generated, constraintStrength, stepsLeft)
"""

    repaired, changed = repair_verification_strategy(
        strategy,
        "Dafny verification failed with 1 error(s):\n\n"
        "VerifiedAgentSynthesis.dfy(627,23): Related location: this is the precondition that could not be proved\n"
        "requires penalty > 0.0\n",
    )

    assert changed
    assert "helpers.SoftConstrainedStep(prompt, generated, 0.1, stepsLeft)" in repaired
