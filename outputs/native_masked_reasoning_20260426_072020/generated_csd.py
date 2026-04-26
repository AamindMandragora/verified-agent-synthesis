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
    # Take a short free-form reasoning pass while masking delimiter tokens, then emit one final constrained GSM answer segment.
    # CSD_RATIONALE_END
    next_token = eosToken
    reasoning_steps = 0
    # invariant lm.ValidTokensIdsLogits()
    # invariant 0 <= stepsLeft <= maxSteps
    # invariant |generated| + stepsLeft <= maxSteps
    # decreases stepsLeft
    while stepsLeft > 2 and reasoning_steps < 24:
        lm.GenerateLogits(prompt + generated)
        if LeftDelimiter in lm.Tokens:
            lm.MaskToken(LeftDelimiter)
        if RightDelimiter in lm.Tokens:
            lm.MaskToken(RightDelimiter)
        next_token = lm.ChooseNextTokenUnconstrained()
        generated = generated + [next_token]
        stepsLeft = stepsLeft - 1
        reasoning_steps = reasoning_steps + 1
    if stepsLeft > 0:
        next_token, stepsLeft = helpers.ForcedTokenStep(prompt, generated, LeftDelimiter, stepsLeft)
        generated = generated + [next_token]
    # invariant lm.ValidTokensIdsLogits()
    # invariant 0 <= stepsLeft <= maxSteps
    # invariant |generated| + stepsLeft <= maxSteps
    # decreases stepsLeft
    while stepsLeft > 0 and not parser.IsCompletePrefix(helpers.LongestValidSuffix(generated)):
        next_token, stepsLeft = helpers.ConstrainedStep(prompt, generated, stepsLeft)
        generated = generated + [next_token]
    if stepsLeft > 0:
        next_token, stepsLeft = helpers.ForcedTokenStep(prompt, generated, RightDelimiter, stepsLeft)
        generated = generated + [next_token]
    remainingSteps = stepsLeft
    return generated, remainingSteps
