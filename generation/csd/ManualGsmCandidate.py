from __future__ import annotations

from VerifiedAgentSynthesis import (
    CSDHelpers,
    LM,
    LeftDelimiter,
    RightDelimiter,
    Parser,
    Prefix,
    Token,
    dafny_spec,
)


DAFNY_INCLUDE = "VerifiedAgentSynthesis.dfy"
MODULE_NAME = "ManualGsmCandidate"
DAFNY_OPEN_IMPORT = "VerifiedDecoderAgent"


@dafny_spec(
    kind="method",
    modifies=("lm.Logits",),
    requires=(
        "lm.ValidTokensIdsLogits()",
        "parser.IsValidPrefix([])",
        "!parser.IsCompletePrefix([])",
        "forall t: Token :: t in parser.ValidNextTokens([]) ==> t in lm.Tokens",
        "maxSteps >= 2",
        "LeftDelimiter in lm.Tokens",
        "RightDelimiter in lm.Tokens",
    ),
    ensures=(
        "lm.ValidTokensIdsLogits()",
        "|generated| <= maxSteps",
        "remainingSteps >= 0 && remainingSteps <= maxSteps",
    ),
)
def MyCSDStrategy(
    lm: LM,
    parser: Parser,
    prompt: Prefix,
    maxSteps: int,
    eosToken: Token,
) -> tuple[Prefix, int]:
    helpers = CSDHelpers(lm, parser)
    lm.ValidTokensIdsLogitsAlways()
    generated = []
    stepsLeft = maxSteps

    # CSD_RATIONALE_BEGIN
    # Manual rewrite of a promising soft-to-hard constrained strategy.
    # It allows a short free-form prefix, then emits << and adapts from
    # soft grammar pressure to hard masking as the continuation space narrows.
    # CSD_RATIONALE_END
    phase = 0
    constraint_level = 0
    reasoning_tokens = 0
    max_reasoning_tokens = 4

    # invariant lm.ValidTokensIdsLogits()
    # invariant 0 <= stepsLeft <= maxSteps
    # invariant |generated| + stepsLeft <= maxSteps
    # invariant 0 <= phase <= 2
    # invariant 0 <= constraint_level <= 1
    # invariant 0 <= reasoning_tokens <= max_reasoning_tokens
    # decreases stepsLeft
    while stepsLeft > 0 and phase < 2:
        next_token = eosToken
        new_steps = stepsLeft
        suffix = helpers.LongestValidSuffix(generated)

        if phase == 0 and reasoning_tokens < max_reasoning_tokens and len(suffix) == 0 and stepsLeft > 2:
            next_token, new_steps = helpers.UnconstrainedStep(prompt, generated, stepsLeft)
            generated = generated + [next_token]
            stepsLeft = new_steps
            reasoning_tokens = reasoning_tokens + 1
            if reasoning_tokens >= max_reasoning_tokens or stepsLeft <= 2:
                phase = 1
        elif phase == 0:
            next_token, new_steps = helpers.ForcedTokenStep(prompt, generated, LeftDelimiter, stepsLeft)
            generated = generated + [next_token]
            stepsLeft = new_steps
            phase = 1
        elif phase == 1 and not parser.IsCompletePrefix(suffix):
            if len(suffix) > 0 and parser.ValidContinuationCount(suffix) <= 4:
                constraint_level = 1
            if constraint_level == 0:
                next_token, new_steps = helpers.SoftConstrainedStep(prompt, generated, 0.5, stepsLeft)
            else:
                next_token, new_steps = helpers.ConstrainedStep(prompt, generated, stepsLeft)
            generated = generated + [next_token]
            stepsLeft = new_steps
        elif phase == 1:
            next_token, new_steps = helpers.ForcedTokenStep(prompt, generated, RightDelimiter, stepsLeft)
            generated = generated + [next_token]
            stepsLeft = new_steps
            phase = 2

    remainingSteps = stepsLeft
    return generated, remainingSteps
