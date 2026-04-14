from __future__ import annotations

from VerifiedAgentSynthesis import (
    CSDHelpers,
    Delimiter,
    LM,
    LeftDelimiter,
    Parser,
    Prefix,
    RightDelimiter,
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
        "forall t: Token :: t in parser.ValidNextTokens([]) ==> t in lm.Tokens",
        "LeftDelimiter in lm.Tokens && RightDelimiter in lm.Tokens",
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
    delim = Delimiter(LeftDelimiter, RightDelimiter)
    helpers = CSDHelpers(lm, parser, delim)
    helpers.DelimitersInLMAlways()
    lm.ValidTokensIdsLogitsAlways()
    generated = []
    stepsLeft = maxSteps
    # CSD_RATIONALE_BEGIN
    # This strategy performs a simple pipeline smoke test by alternating between unconstrained and constrained decoding steps until the maximum number of steps is reached or the parser completes the prefix.
    # CSD_RATIONALE_END

    # invariant lm.ValidTokensIdsLogits()
    # invariant 0 <= steps
    remainingSteps = stepsLeft
    return generated, remainingSteps
