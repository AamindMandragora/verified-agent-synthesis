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
    # This strategy uses unconstrained decoding outside the delimited window defined by '<<' and '>>',
    # and constrained decoding inside the window to ensure that expressions within delimiters are parser-valid.
    # CSD_RATIONALE_END

    # invariant lm.ValidTokensIdsLogits()
    # invariant 0 <= stepsLeft <= maxSteps
    # invariant len(generated) + stepsLeft <= maxSteps
    # invariant helpers.ConstrainedWindowValid(generated)
    # decreases stepsLeft
    while stepsLeft > 0 and not parser.IsCompletePrefix(generated):
        next_token, new_steps = helpers.UnconstrainedStep(prompt, generated, stepsLeft)
        generated = generated + [next_token]
        stepsLeft = new_steps
    remainingSteps = stepsLeft
    return generated, remainingSteps
