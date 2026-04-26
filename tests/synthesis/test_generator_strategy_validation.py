from generation.generator import StrategyGenerator


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
phase = 0
constrained_count = 0
# invariant 0 <= stepsLeft <= maxSteps
# decreases stepsLeft
while stepsLeft > 0 and phase < 3:
    next_token = eosToken
    new_steps = stepsLeft
    if phase == 0:
        next_token, new_steps = helpers.UnconstrainedStep(prompt, generated, stepsLeft)
        generated = generated + [next_token]
        stepsLeft = new_steps
        if helpers.VariableFound(generated):
            phase = 1
    elif phase == 1:
        next_token, new_steps = helpers.ForcedTokenStep(prompt, generated, LeftDelimiter, stepsLeft)
        generated = generated + [next_token]
        stepsLeft = new_steps
        phase = 2
    elif phase == 2 and not parser.IsCompletePrefix(helpers.LongestValidSuffix(generated)):
        next_token, new_steps = helpers.ConstrainedStep(prompt, generated, stepsLeft)
        generated = generated + [next_token]
        stepsLeft = new_steps
        constrained_count = constrained_count + 1
    elif phase == 2:
        next_token, new_steps = helpers.ForcedTokenStep(prompt, generated, RightDelimiter, stepsLeft)
        generated = generated + [next_token]
        stepsLeft = new_steps
        phase = 3
"""

    issue = generator._structural_issue(strategy)

    assert issue is not None
    assert "VariableFound" in issue


def test_structural_issue_rejects_parser_methods_called_on_helpers():
    generator = StrategyGenerator.__new__(StrategyGenerator)
    strategy = """# CSD_RATIONALE_BEGIN
# test
# CSD_RATIONALE_END
phase = 0
reasoning_tokens = 0
constraint_mode = 0
# invariant 0 <= stepsLeft <= maxSteps
# decreases stepsLeft
while stepsLeft > 0 and phase < 3:
    if phase == 0 and reasoning_tokens < 1:
        generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
        reasoning_tokens = reasoning_tokens + 1
        if helpers.ValidContinuationCount(helpers.LongestValidSuffix(generated)) > 0:
            phase = 1
    elif phase == 1:
        generated, stepsLeft = helpers.AppendLeftDelimiter(generated, stepsLeft)
        phase = 2
    elif phase == 2 and helpers.CanConstrain(generated):
        generated, stepsLeft = helpers.AppendConstrainedStep(prompt, generated, stepsLeft)
        constraint_mode = constraint_mode + 1
    else:
        generated, stepsLeft = helpers.AppendRightDelimiter(generated, stepsLeft)
        phase = 3
"""

    issue = generator._structural_issue(strategy)

    assert issue is not None
    assert "parser methods" in issue
    assert "ValidContinuationCount" in issue


def test_structural_issue_rejects_unknown_parser_methods():
    generator = StrategyGenerator.__new__(StrategyGenerator)
    strategy = """# CSD_RATIONALE_BEGIN
# test
# CSD_RATIONALE_END
phase = 0
constrained_count = 0
# invariant 0 <= stepsLeft <= maxSteps
# decreases stepsLeft
while stepsLeft > 0 and phase < 3:
    next_token = eosToken
    new_steps = stepsLeft
    if phase == 0:
        next_token, new_steps = helpers.UnconstrainedStep(prompt, generated, stepsLeft)
        generated = generated + [next_token]
        stepsLeft = new_steps
        if parser.PotentialConstrainedSegment(generated):
            phase = 1
    elif phase == 1:
        next_token, new_steps = helpers.ForcedTokenStep(prompt, generated, LeftDelimiter, stepsLeft)
        generated = generated + [next_token]
        stepsLeft = new_steps
        phase = 2
    elif phase == 2 and not parser.IsCompletePrefix(helpers.LongestValidSuffix(generated)):
        next_token, new_steps = helpers.ConstrainedStep(prompt, generated, stepsLeft)
        generated = generated + [next_token]
        stepsLeft = new_steps
        constrained_count = constrained_count + 1
    elif phase == 2:
        next_token, new_steps = helpers.ForcedTokenStep(prompt, generated, RightDelimiter, stepsLeft)
        generated = generated + [next_token]
        stepsLeft = new_steps
        phase = 3
"""

    issue = generator._structural_issue(strategy)

    assert issue is not None
    assert "PotentialConstrainedSegment" in issue


def test_structural_issue_rejects_parser_methods_on_generated():
    generator = StrategyGenerator.__new__(StrategyGenerator)
    strategy = """# CSD_RATIONALE_BEGIN
# test
# CSD_RATIONALE_END
phase = 0
constrained_count = 0
# invariant 0 <= stepsLeft <= maxSteps
# decreases stepsLeft
while stepsLeft > 0 and phase < 3:
    next_token = eosToken
    new_steps = stepsLeft
    if phase == 0:
        next_token, new_steps = helpers.UnconstrainedStep(prompt, generated, stepsLeft)
        generated = generated + [next_token]
        stepsLeft = new_steps
        if parser.IsValidPrefix(generated):
            phase = 1
    elif phase == 1:
        next_token, new_steps = helpers.ForcedTokenStep(prompt, generated, LeftDelimiter, stepsLeft)
        generated = generated + [next_token]
        stepsLeft = new_steps
        phase = 2
    elif phase == 2 and not parser.IsCompletePrefix(helpers.LongestValidSuffix(generated)):
        next_token, new_steps = helpers.ConstrainedStep(prompt, generated, stepsLeft)
        generated = generated + [next_token]
        stepsLeft = new_steps
        constrained_count = constrained_count + 1
    elif phase == 2:
        next_token, new_steps = helpers.ForcedTokenStep(prompt, generated, RightDelimiter, stepsLeft)
        generated = generated + [next_token]
        stepsLeft = new_steps
        phase = 3
"""

    issue = generator._structural_issue(strategy)

    assert issue is not None
    assert "generated" in issue
    assert "IsValidPrefix" in issue


def test_structural_issue_rejects_old_api_calls():
    generator = StrategyGenerator.__new__(StrategyGenerator)
    strategy = """# CSD_RATIONALE_BEGIN
# test
# CSD_RATIONALE_END
phase = 0
answer_tokens = 0
# invariant 0 <= stepsLeft <= maxSteps
# decreases stepsLeft
while stepsLeft > 0 and phase < 2:
    next_token = eosToken
    new_steps = stepsLeft
    if phase == 0:
        next_token, new_steps = helpers.ExpressiveStep(prompt, generated, stepsLeft)
        generated = generated + [next_token]
        stepsLeft = new_steps
        phase = 1
    else:
        next_token, new_steps = helpers.ConstrainedAnswerStep(prompt, generated, answer, stepsLeft)
        answer = answer + [next_token]
        stepsLeft = new_steps
        answer_tokens = answer_tokens + 1
"""

    issue = generator._structural_issue(strategy)

    assert issue is not None
    assert "old" in issue.lower() or "replaced" in issue.lower()


def test_structural_issue_accepts_valid_new_api_strategy():
    generator = StrategyGenerator.__new__(StrategyGenerator)
    strategy = """# CSD_RATIONALE_BEGIN
# Hybrid strategy using the new suffix-based API.
# CSD_RATIONALE_END
phase = 0
reasoning_tokens = 0
constrained_count = 0
# invariant lm.ValidTokensIdsLogits()
# invariant 0 <= stepsLeft <= maxSteps
# decreases stepsLeft
while stepsLeft > 0 and phase < 3:
    next_token = eosToken
    new_steps = stepsLeft
    if phase == 0 and reasoning_tokens < 3 and stepsLeft > 2:
        next_token, new_steps = helpers.UnconstrainedStep(prompt, generated, stepsLeft)
        generated = generated + [next_token]
        stepsLeft = new_steps
        reasoning_tokens = reasoning_tokens + 1
        if reasoning_tokens >= 3:
            phase = 1
    elif phase == 1:
        next_token, new_steps = helpers.ForcedTokenStep(prompt, generated, LeftDelimiter, stepsLeft)
        generated = generated + [next_token]
        stepsLeft = new_steps
        phase = 2
    elif phase == 2 and not parser.IsCompletePrefix(helpers.LongestValidSuffix(generated)):
        next_token, new_steps = helpers.ConstrainedStep(prompt, generated, stepsLeft)
        generated = generated + [next_token]
        stepsLeft = new_steps
        constrained_count = constrained_count + 1
    elif phase == 2 and parser.IsCompletePrefix(helpers.LongestValidSuffix(generated)):
        next_token, new_steps = helpers.ForcedTokenStep(prompt, generated, RightDelimiter, stepsLeft)
        generated = generated + [next_token]
        stepsLeft = new_steps
        phase = 3
"""

    issue = generator._structural_issue(strategy)

    assert issue is None


def test_structural_issue_rejects_missing_standard_loop_invariants():
    generator = StrategyGenerator.__new__(StrategyGenerator)
    strategy = """# CSD_RATIONALE_BEGIN
# test
# CSD_RATIONALE_END
phase = 0
reasoning_tokens = 0
constrained_count = 0
while stepsLeft > 0 and phase < 3:
    if phase == 0 and reasoning_tokens < 2 and stepsLeft > 2:
        generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
        reasoning_tokens = reasoning_tokens + 1
        if reasoning_tokens >= 2:
            phase = 1
    elif phase == 1:
        generated, stepsLeft = helpers.AppendLeftDelimiter(generated, stepsLeft)
        phase = 2
    elif phase == 2 and helpers.CanConstrain(generated):
        generated, stepsLeft = helpers.AppendConstrainedStep(prompt, generated, stepsLeft)
        constrained_count = constrained_count + 1
    else:
        generated, stepsLeft = helpers.AppendRightDelimiter(generated, stepsLeft)
        phase = 3
"""

    issue = generator._structural_issue(strategy)

    assert issue is not None
    assert "standard loop invariant" in issue or "standard decreases clause" in issue


def test_structural_issue_accepts_append_helper_strategy():
    generator = StrategyGenerator.__new__(StrategyGenerator)
    strategy = """# CSD_RATIONALE_BEGIN
# Hybrid strategy using the append-style helper wrappers.
# CSD_RATIONALE_END
phase = 0
reasoning_tokens = 0
constraint_mode = 0
# invariant lm.ValidTokensIdsLogits()
# invariant 0 <= stepsLeft <= maxSteps
# decreases stepsLeft
while stepsLeft > 0 and phase < 3:
    if phase == 0 and reasoning_tokens < 2 and stepsLeft > 2:
        generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
        reasoning_tokens = reasoning_tokens + 1
        if reasoning_tokens >= 2:
            phase = 1
    elif phase == 1:
        generated, stepsLeft = helpers.AppendLeftDelimiter(generated, stepsLeft)
        phase = 2
    elif phase == 2 and helpers.CanConstrain(generated):
        if constraint_mode == 0:
            generated, stepsLeft = helpers.AppendSoftConstrainedStep(prompt, generated, 0.5, stepsLeft)
        else:
            generated, stepsLeft = helpers.AppendConstrainedStep(prompt, generated, stepsLeft)
        constraint_mode = constraint_mode + 1
    else:
        generated, stepsLeft = helpers.AppendRightDelimiter(generated, stepsLeft)
        phase = 3
"""

    issue = generator._structural_issue(strategy)

    assert issue is None


def test_structural_issue_rejects_bare_forced_token_step_calls():
    generator = StrategyGenerator.__new__(StrategyGenerator)
    strategy = """# CSD_RATIONALE_BEGIN
# test
# CSD_RATIONALE_END
phase = 0
reasoning_tokens = 0
constrained_count = 0
# invariant 0 <= stepsLeft <= maxSteps
# decreases stepsLeft
while stepsLeft > 0 and phase < 3:
    next_token = eosToken
    new_steps = stepsLeft
    if phase == 0 and reasoning_tokens < 2:
        next_token, new_steps = helpers.UnconstrainedStep(prompt, generated, stepsLeft)
        generated = generated + [next_token]
        stepsLeft = new_steps
        reasoning_tokens = reasoning_tokens + 1
        if reasoning_tokens >= 2:
            phase = 1
    elif phase == 1:
        helpers.ForcedTokenStep(prompt, generated, LeftDelimiter, stepsLeft)
        generated = generated + [LeftDelimiter]
        phase = 2
    elif phase == 2 and not parser.IsCompletePrefix(helpers.LongestValidSuffix(generated)):
        next_token, new_steps = helpers.ConstrainedStep(prompt, generated, stepsLeft)
        generated = generated + [next_token]
        stepsLeft = new_steps
        constrained_count = constrained_count + 1
    elif phase == 2:
        next_token, new_steps = helpers.ForcedTokenStep(prompt, generated, RightDelimiter, stepsLeft)
        generated = generated + [next_token]
        stepsLeft = new_steps
        phase = 3
"""

    issue = generator._structural_issue(strategy)

    assert issue is not None
    assert "ForcedTokenStep" in issue
    assert "bare statement" in issue


def test_structural_issue_rejects_bare_append_helper_calls():
    generator = StrategyGenerator.__new__(StrategyGenerator)
    strategy = """# CSD_RATIONALE_BEGIN
# test
# CSD_RATIONALE_END
phase = 0
reasoning_tokens = 0
constraint_mode = 0
# invariant 0 <= stepsLeft <= maxSteps
# decreases stepsLeft
while stepsLeft > 0 and phase < 3:
    if phase == 0 and reasoning_tokens < 1:
        generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
        reasoning_tokens = reasoning_tokens + 1
        phase = 1
    elif phase == 1:
        helpers.AppendLeftDelimiter(generated, stepsLeft)
        phase = 2
    elif phase == 2 and helpers.CanConstrain(generated):
        generated, stepsLeft = helpers.AppendConstrainedStep(prompt, generated, stepsLeft)
        constraint_mode = constraint_mode + 1
    else:
        generated, stepsLeft = helpers.AppendRightDelimiter(generated, stepsLeft)
        phase = 3
"""

    issue = generator._structural_issue(strategy)

    assert issue is not None
    assert "Append*" in issue
    assert "must not be used as bare statements" in issue


def test_structural_issue_rejects_unguarded_append_constrained_step():
    generator = StrategyGenerator.__new__(StrategyGenerator)
    strategy = """# CSD_RATIONALE_BEGIN
# test
# CSD_RATIONALE_END
phase = 0
budget_for_reasoning = stepsLeft // 3
budget_for_constrained = stepsLeft - budget_for_reasoning
# invariant 0 <= stepsLeft <= maxSteps
# decreases stepsLeft
while stepsLeft > 0 and phase < 3:
    if phase == 0 and len(generated) < budget_for_reasoning:
        generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
        if len(generated) >= budget_for_reasoning:
            phase = 1
    elif phase == 0:
        generated, stepsLeft = helpers.AppendLeftDelimiter(generated, stepsLeft)
        phase = 2
    elif phase == 1:
        generated, stepsLeft = helpers.AppendConstrainedStep(prompt, generated, stepsLeft)
    elif phase == 2 and helpers.CanConstrain(generated):
        generated, stepsLeft = helpers.AppendConstrainedStep(prompt, generated, stepsLeft)
    else:
        generated, stepsLeft = helpers.AppendRightDelimiter(generated, stepsLeft)
        phase = 3
"""

    issue = generator._structural_issue(strategy)

    assert issue is not None
    assert "Every constrained helper call" in issue
    assert "AppendConstrainedStep" in issue


def test_structural_issue_rejects_unbounded_while_loops():
    generator = StrategyGenerator.__new__(StrategyGenerator)
    strategy = """# CSD_RATIONALE_BEGIN
# test
# CSD_RATIONALE_END
phase = 0
reasoning_tokens = 0
constrained_count = 0
delim_phase = 0
# invariant 0 <= stepsLeft <= maxSteps
# decreases stepsLeft
while not parser.IsCompletePrefix(helpers.LongestValidSuffix(generated)):
    next_token = eosToken
    new_steps = stepsLeft
    if delim_phase == 0:
        next_token, new_steps = helpers.ForcedTokenStep(prompt, generated, LeftDelimiter, stepsLeft)
        generated = generated + [next_token]
        stepsLeft = new_steps
        delim_phase = 1
    elif delim_phase == 1:
        next_token, new_steps = helpers.ConstrainedStep(prompt, generated, stepsLeft)
        generated = generated + [next_token]
        stepsLeft = new_steps
        constrained_count = constrained_count + 1
        if constrained_count > 2:
            delim_phase = 2
    else:
        next_token, new_steps = helpers.ForcedTokenStep(prompt, generated, RightDelimiter, stepsLeft)
        generated = generated + [next_token]
        stepsLeft = new_steps
        phase = 1
"""

    issue = generator._structural_issue(strategy)

    assert issue is not None
    assert "budget-bounded" in issue


def test_structural_issue_rejects_string_methods_on_generated():
    generator = StrategyGenerator.__new__(StrategyGenerator)
    strategy = """# CSD_RATIONALE_BEGIN
# test
# CSD_RATIONALE_END
phase = 0
repair_mode = False
delimiter_round = 0
# invariant lm.ValidTokensIdsLogits()
# invariant 0 <= stepsLeft <= maxSteps
# decreases stepsLeft
while stepsLeft > 0 and phase < 3:
    if phase == 0 and not generated.endswith(LeftDelimiter):
        generated, stepsLeft = helpers.AppendLeftDelimiter(generated, stepsLeft)
        phase = 1
    elif phase == 1 and helpers.CanConstrain(generated):
        generated, stepsLeft = helpers.AppendConstrainedStep(prompt, generated, stepsLeft)
        delimiter_round = delimiter_round + 1
        if delimiter_round > 1:
            phase = 2
    elif phase == 2 and not generated.startswith(LeftDelimiter):
        generated, stepsLeft = helpers.AppendRightDelimiter(generated, stepsLeft)
        repair_mode = True
        phase = 3
"""

    issue = generator._structural_issue(strategy)

    assert issue is not None
    assert "generated" in issue
    assert "startswith" in issue
    assert "endswith" in issue


def test_structural_issue_rejects_string_methods_on_longest_valid_suffix():
    generator = StrategyGenerator.__new__(StrategyGenerator)
    strategy = """# CSD_RATIONALE_BEGIN
# test
# CSD_RATIONALE_END
phase = 0
repair_mode = False
delimiter_round = 0
# invariant lm.ValidTokensIdsLogits()
# invariant 0 <= stepsLeft <= maxSteps
# decreases stepsLeft
while stepsLeft > 0 and phase < 3:
    if phase == 0:
        generated, stepsLeft = helpers.AppendLeftDelimiter(generated, stepsLeft)
        phase = 1
    elif phase == 1 and helpers.CanConstrain(generated):
        generated, stepsLeft = helpers.AppendConstrainedStep(prompt, generated, stepsLeft)
        delimiter_round = delimiter_round + 1
        if delimiter_round > 1:
            phase = 2
    elif phase == 2 and not helpers.LongestValidSuffix(generated).endswith(RightDelimiter):
        generated, stepsLeft = helpers.AppendRightDelimiter(generated, stepsLeft)
        repair_mode = True
        phase = 3
"""

    issue = generator._structural_issue(strategy)

    assert issue is not None
    assert "LongestValidSuffix" in issue
    assert "endswith" in issue


def test_structural_issue_rejects_append_helper_assigned_to_next_token():
    generator = StrategyGenerator.__new__(StrategyGenerator)
    strategy = """# CSD_RATIONALE_BEGIN
# test
# CSD_RATIONALE_END
phase = 0
reasoning_tokens = 0
constraint_mode = 0
# invariant lm.ValidTokensIdsLogits()
# invariant 0 <= stepsLeft <= maxSteps
# decreases stepsLeft
while stepsLeft > 0 and phase < 3:
    if phase == 0 and reasoning_tokens < 1:
        next_token, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
        reasoning_tokens = reasoning_tokens + 1
        phase = 1
    elif phase == 1:
        generated, stepsLeft = helpers.AppendLeftDelimiter(generated, stepsLeft)
        phase = 2
    elif phase == 2 and helpers.CanConstrain(generated):
        next_token, stepsLeft = helpers.AppendConstrainedStep(prompt, generated, stepsLeft)
        constraint_mode = constraint_mode + 1
    else:
        generated, stepsLeft = helpers.AppendRightDelimiter(generated, stepsLeft)
        phase = 3
"""

    issue = generator._structural_issue(strategy)

    assert issue is not None
    assert "Append*" in issue
    assert "next_token" in issue
