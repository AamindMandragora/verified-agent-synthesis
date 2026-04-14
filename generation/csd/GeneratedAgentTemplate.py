from __future__ import annotations

from VerifiedAgentSynthesis import (
    CSDHelpers,
    DelimitedAnswerValidForParser,
    Delimiter,
    LM,
    LeftDelimiter,
    Parser,
    PrefixContains,
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
        "!parser.IsCompletePrefix([])",
        "forall t: Token :: t in parser.ValidNextTokens([]) ==> t in lm.Tokens",
        "LeftDelimiter in lm.Tokens && RightDelimiter in lm.Tokens",
        "maxSteps >= 4",
    ),
    ensures=(
        "lm.ValidTokensIdsLogits()",
        "|generated| <= maxSteps",
        "remainingSteps >= 0 && remainingSteps <= maxSteps",
        "DelimitedAnswerValidForParser(parser, generated)",
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
    answer = []
    stepsLeft = maxSteps - 2
    # QWEN_INSERT_STRATEGY_BEGIN
    raise NotImplementedError("QWEN_INSERT_STRATEGY_HERE")
    # QWEN_INSERT_STRATEGY_END
    helpers.FinalizeDelimitedAnswer(generated, answer)
    generated = generated + [LeftDelimiter] + answer + [RightDelimiter]
    remainingSteps = stepsLeft
    return generated, remainingSteps
