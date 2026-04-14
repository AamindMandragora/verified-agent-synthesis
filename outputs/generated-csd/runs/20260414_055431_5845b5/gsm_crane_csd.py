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
    # CSD_RATIONALE_BEGIN
    # Fallback starter strategy: keep delimiter control explicit, spend a small controlled budget on expressive free-form text, then drive a separate constrained answer channel until it becomes complete.
    # CSD_RATIONALE_END
    phase = 0
    preamble_tokens = 0
    exploration_budget = 1
    answer_tokens = 0
    # invariant lm.ValidTokensIdsLogits()
    # invariant 0 <= stepsLeft <= maxSteps - 2
    # invariant 0 <= preamble_tokens <= 1
    # invariant 0 <= exploration_budget <= 1
    # invariant 0 <= answer_tokens
    # invariant helpers.ConstrainedWindowValid(generated)
    # invariant parser.IsValidPrefix(answer)
    # invariant |generated| + |answer| + stepsLeft <= maxSteps - 2
    # invariant |answer| == 0 ==> exploration_budget < stepsLeft
    # decreases stepsLeft
    while stepsLeft > 0 and not parser.IsCompletePrefix(answer):
        next_token = eosToken
        new_steps = stepsLeft
        spend_freeform = phase < 2 and exploration_budget > 0 and preamble_tokens < 1 and stepsLeft > 1
        if spend_freeform:
            next_token, new_steps = helpers.ExpressiveStep(prompt, generated, stepsLeft)
            generated = generated + [next_token]
            stepsLeft = new_steps
            preamble_tokens = preamble_tokens + 1
            exploration_budget = exploration_budget - 1
            if preamble_tokens >= 1:
                phase = 1
            if preamble_tokens >= 1 or stepsLeft <= 1:
                phase = 2
        else:
            next_token, new_steps = helpers.ConstrainedAnswerStep(prompt, generated, answer, stepsLeft)
            answer = answer + [next_token]
            stepsLeft = new_steps
            answer_tokens = answer_tokens + 1
            if answer_tokens >= 1:
                phase = 3
    helpers.FinalizeDelimitedAnswer(generated, answer)
    generated = generated + [LeftDelimiter] + answer + [RightDelimiter]
    remainingSteps = stepsLeft
    return generated, remainingSteps
