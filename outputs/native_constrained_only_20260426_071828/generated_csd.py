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
    # Emit a single constrained GSM answer segment without free-form reasoning so the output has only one << >> span.
    # CSD_RATIONALE_END
    next_token = eosToken
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
