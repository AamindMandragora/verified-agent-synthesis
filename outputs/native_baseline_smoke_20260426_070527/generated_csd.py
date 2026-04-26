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
MODULE_NAME = "GeneratedCSD"
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
    # Fallback starter strategy: budget-split between unconstrained reasoning and grammar-constrained answer inside << >> delimiters.
    # CSD_RATIONALE_END
    phase = 0
    reasoning_tokens = 0
    constrained_tokens = 0
    reasoning_budget = stepsLeft // 2
    # invariant lm.ValidTokensIdsLogits()
    # invariant 0 <= stepsLeft <= maxSteps
    # invariant 0 <= reasoning_tokens
    # invariant 0 <= constrained_tokens
    # invariant 0 <= phase <= 3
    # invariant |generated| + stepsLeft <= maxSteps
    # decreases stepsLeft
    while stepsLeft > 0 and phase < 3:
        next_token = eosToken
        new_steps = stepsLeft
        if phase == 0 and reasoning_tokens < reasoning_budget and stepsLeft > 2:
            next_token, new_steps = helpers.UnconstrainedStep(prompt, generated, stepsLeft)
            generated = generated + [next_token]
            stepsLeft = new_steps
            reasoning_tokens = reasoning_tokens + 1
            if reasoning_tokens >= reasoning_budget or stepsLeft <= 2:
                phase = 1
        elif phase == 0:
            next_token, new_steps = helpers.ForcedTokenStep(prompt, generated, LeftDelimiter, stepsLeft)
            generated = generated + [next_token]
            stepsLeft = new_steps
            phase = 2
        elif phase == 1:
            next_token, new_steps = helpers.ForcedTokenStep(prompt, generated, LeftDelimiter, stepsLeft)
            generated = generated + [next_token]
            stepsLeft = new_steps
            phase = 2
        elif phase == 2 and not parser.IsCompletePrefix(helpers.LongestValidSuffix(generated)):
            next_token, new_steps = helpers.ConstrainedStep(prompt, generated, stepsLeft)
            generated = generated + [next_token]
            stepsLeft = new_steps
            constrained_tokens = constrained_tokens + 1
        elif phase == 2 and parser.IsCompletePrefix(helpers.LongestValidSuffix(generated)):
            next_token, new_steps = helpers.ForcedTokenStep(prompt, generated, RightDelimiter, stepsLeft)
            generated = generated + [next_token]
            stepsLeft = new_steps
            phase = 3
    remainingSteps = stepsLeft
    return generated, remainingSteps
