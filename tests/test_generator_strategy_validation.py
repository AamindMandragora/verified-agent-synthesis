from synthesis.generator import StrategyGenerator


def test_normalize_rationale_block_comments_plain_lines():
    generator = StrategyGenerator.__new__(StrategyGenerator)
    strategy = """# CSD_RATIONALE_BEGIN
This rationale line is not commented.
# CSD_RATIONALE_END
flag = False
"""

    normalized = generator._normalize_rationale_block(strategy)

    assert "# This rationale line is not commented." in normalized


def test_structural_issue_rejects_unknown_helper_methods():
    generator = StrategyGenerator.__new__(StrategyGenerator)
    strategy = """# CSD_RATIONALE_BEGIN
# test
# CSD_RATIONALE_END
state_a = False
state_b = False
while stepsLeft > 0 and not parser.IsCompletePrefix(answer):
    next_token = eosToken
    new_steps = stepsLeft
    if not state_a:
        next_token, new_steps = helpers.ExpressiveStep(prompt, generated, stepsLeft)
        generated = generated + [next_token]
        if helpers.VariableFound(answer):
            state_a = True
    else:
        next_token, new_steps = helpers.ConstrainedAnswerStep(prompt, generated, answer, stepsLeft)
        answer = answer + [next_token]
        state_b = True
    stepsLeft = new_steps
"""

    issue = generator._structural_issue(strategy)

    assert issue is not None
    assert "VariableFound" in issue


def test_structural_issue_rejects_unknown_parser_methods():
    generator = StrategyGenerator.__new__(StrategyGenerator)
    strategy = """# CSD_RATIONALE_BEGIN
# test
# CSD_RATIONALE_END
flag_a = False
flag_b = False
while stepsLeft > 0 and not parser.IsCompletePrefix(answer):
    next_token = eosToken
    new_steps = stepsLeft
    if not flag_a:
        next_token, new_steps = helpers.ExpressiveStep(prompt, generated, stepsLeft)
        generated = generated + [next_token]
        if parser.PotentialConstrainedSegment(generated):
            flag_a = True
    else:
        next_token, new_steps = helpers.ConstrainedAnswerStep(prompt, generated, answer, stepsLeft)
        answer = answer + [next_token]
        flag_b = True
    stepsLeft = new_steps
"""

    issue = generator._structural_issue(strategy)

    assert issue is not None
    assert "PotentialConstrainedSegment" in issue


def test_structural_issue_rejects_parser_methods_on_generated():
    generator = StrategyGenerator.__new__(StrategyGenerator)
    strategy = """# CSD_RATIONALE_BEGIN
# test
# CSD_RATIONALE_END
flag_a = False
flag_b = False
while stepsLeft > 0 and not parser.IsCompletePrefix(answer):
    next_token = eosToken
    new_steps = stepsLeft
    if not flag_a:
        next_token, new_steps = helpers.ExpressiveStep(prompt, generated, stepsLeft)
        generated = generated + [next_token]
        if parser.IsValidPrefix(generated):
            flag_a = True
    else:
        next_token, new_steps = helpers.ConstrainedAnswerStep(prompt, generated, answer, stepsLeft)
        answer = answer + [next_token]
        flag_b = True
    stepsLeft = new_steps
"""

    issue = generator._structural_issue(strategy)

    assert issue is not None
    assert "generated" in issue
    assert "IsValidPrefix" in issue


def test_structural_issue_rejects_finalize_in_body():
    generator = StrategyGenerator.__new__(StrategyGenerator)
    strategy = """# CSD_RATIONALE_BEGIN
# test
# CSD_RATIONALE_END
flag_a = False
flag_b = False
while stepsLeft > 0 and not parser.IsCompletePrefix(answer):
    next_token = eosToken
    new_steps = stepsLeft
    if not flag_a:
        next_token, new_steps = helpers.ExpressiveStep(prompt, generated, stepsLeft)
        generated = generated + [next_token]
        flag_a = True
    else:
        next_token, new_steps = helpers.ConstrainedAnswerStep(prompt, generated, answer, stepsLeft)
        answer = answer + [next_token]
        flag_b = True
    stepsLeft = new_steps
helpers.FinalizeDelimitedAnswer(generated, answer)
"""

    issue = generator._structural_issue(strategy)

    assert issue is not None
    assert "FinalizeDelimitedAnswer" in issue
