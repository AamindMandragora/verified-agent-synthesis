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
