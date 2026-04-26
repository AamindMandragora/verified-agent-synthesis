"""
Tests for the updated VerifiedAgentSynthesis library functions.

Covers: new LM methods, CSDHelpers suffix-based design,
CheckpointStack, RepetitionTracker, and end-to-end strategy smoke test.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "generation" / "csd"))

from VerifiedAgentSynthesis import (
    CSDHelpers,
    CheckpointStack,
    LM,
    LeftDelimiter,
    Parser,
    Prefix,
    RepetitionTracker,
    RightDelimiter,
    Token,
)


# ---------------------------------------------------------------------------
# Minimal concrete LM for testing
# ---------------------------------------------------------------------------

class SimpleLM(LM):
    def __init__(self, tokens: list[str]) -> None:
        self.Tokens = tokens
        self.Ids = list(range(len(tokens)))
        self.Logits = [0.0] * len(tokens)

    def GenerateLogits(self, input: Prefix) -> None:
        # Give each token a distinct logit based on its index
        for i in range(len(self.Tokens)):
            self.Logits[i] = float(i)

    def ChooseNextToken(self) -> Token:
        best_i = -1
        best_l = -2e9
        for i in range(len(self.Tokens)):
            if self.Logits[i] > best_l and self.Logits[i] != -1e9:
                best_l = self.Logits[i]
                best_i = i
        if best_i == -1:
            raise ValueError("All tokens masked")
        return self.Tokens[best_i]


# ---------------------------------------------------------------------------
# Minimal concrete Parser for testing
# ---------------------------------------------------------------------------

GRAMMAR_TOKENS = {"a", "b", "c"}

class SimpleParser(Parser):
    """Accepts sequences of 'a', 'b', 'c'; complete after exactly 3 tokens."""

    def IsValidPrefix(self, prefix: Prefix) -> bool:
        return len(prefix) <= 3 and all(t in GRAMMAR_TOKENS for t in prefix)

    def IsCompletePrefix(self, prefix: Prefix) -> bool:
        return len(prefix) == 3 and all(t in GRAMMAR_TOKENS for t in prefix)

    def ValidNextTokens(self, prefix: Prefix) -> Prefix:
        if len(prefix) >= 3:
            return []
        return list(GRAMMAR_TOKENS)


# ---------------------------------------------------------------------------
# LM: BiasToken / BiasTokens
# ---------------------------------------------------------------------------

def test_bias_token_adds_delta():
    lm = SimpleLM(["x", "y", "z"])
    lm.Logits = [1.0, 2.0, 3.0]
    lm.BiasToken("x", 5.0)
    assert lm.Logits[0] == 6.0
    assert lm.Logits[1] == 2.0  # unchanged


def test_bias_token_clamps_high():
    lm = SimpleLM(["x"])
    lm.Logits = [9e8]
    lm.BiasToken("x", 9e8)
    assert lm.Logits[0] == 1e9


def test_bias_token_clamps_low():
    lm = SimpleLM(["x"])
    lm.Logits = [-9e8]
    lm.BiasToken("x", -9e8)
    assert lm.Logits[0] == -1e9


def test_bias_tokens_applies_to_all():
    lm = SimpleLM(["x", "y", "z"])
    lm.Logits = [1.0, 2.0, 3.0]
    lm.BiasTokens(["x", "z"], 10.0)
    assert lm.Logits[0] == 11.0
    assert lm.Logits[1] == 2.0   # unchanged
    assert lm.Logits[2] == 13.0


# ---------------------------------------------------------------------------
# LM: ScaleToken / ScaleTokens
# ---------------------------------------------------------------------------

def test_scale_token_multiplies():
    lm = SimpleLM(["x", "y"])
    lm.Logits = [4.0, 2.0]
    lm.ScaleToken("x", 3.0)
    assert lm.Logits[0] == 12.0
    assert lm.Logits[1] == 2.0


def test_scale_tokens_multiplies_all():
    lm = SimpleLM(["x", "y"])
    lm.Logits = [2.0, 3.0]
    lm.ScaleTokens(["x", "y"], 2.0)
    assert lm.Logits[0] == 4.0
    assert lm.Logits[1] == 6.0


# ---------------------------------------------------------------------------
# LM: ClampLogits
# ---------------------------------------------------------------------------

def test_clamp_logits():
    lm = SimpleLM(["x", "y", "z"])
    lm.Logits = [-500.0, 0.0, 500.0]
    lm.ClampLogits(-100.0, 100.0)
    assert lm.Logits[0] == -100.0
    assert lm.Logits[1] == 0.0
    assert lm.Logits[2] == 100.0


# ---------------------------------------------------------------------------
# LM: TopKFilter
# ---------------------------------------------------------------------------

def test_top_k_filter_keeps_highest():
    lm = SimpleLM(["a", "b", "c", "d"])
    lm.Logits = [1.0, 4.0, 3.0, 2.0]
    lm.TopKFilter(2)
    assert lm.Logits[1] != -1e9  # "b" highest
    assert lm.Logits[2] != -1e9  # "c" second
    assert lm.Logits[0] == -1e9  # "a" masked
    assert lm.Logits[3] == -1e9  # "d" masked


# ---------------------------------------------------------------------------
# Parser: ValidContinuationCount
# ---------------------------------------------------------------------------

def test_valid_continuation_count():
    p = SimpleParser()
    assert p.ValidContinuationCount([]) == 3
    assert p.ValidContinuationCount(["a", "b", "c"]) == 0


# ---------------------------------------------------------------------------
# CSDHelpers: LongestValidSuffix
# ---------------------------------------------------------------------------

def make_helpers(tokens=None):
    if tokens is None:
        tokens = ["a", "b", "c", "<<", ">>", "x"]
    lm = SimpleLM(tokens)
    parser = SimpleParser()
    return CSDHelpers(lm, parser)


def test_longest_valid_suffix_empty_prefix():
    h = make_helpers()
    assert h.LongestValidSuffix([]) == []


def test_longest_valid_suffix_fully_valid():
    h = make_helpers()
    assert h.LongestValidSuffix(["a", "b"]) == ["a", "b"]


def test_longest_valid_suffix_strips_invalid_front():
    h = make_helpers()
    # "x" is not in GRAMMAR_TOKENS, so ["x", "a", "b"] prefix:
    # ["x","a","b"] — not valid (x not in grammar)
    # ["a","b"] — valid!
    result = h.LongestValidSuffix(["x", "a", "b"])
    assert result == ["a", "b"]


def test_longest_valid_suffix_all_invalid():
    h = make_helpers()
    result = h.LongestValidSuffix(["x", "x", "x"])
    assert result == []


def test_longest_valid_suffix_after_delimiter():
    h = make_helpers()
    # After emitting "<<", the suffix should reset to [] because "<<" is not a grammar token
    result = h.LongestValidSuffix(["a", "<<"])
    assert result == []


def test_longest_valid_suffix_tracking_through_constrained():
    h = make_helpers()
    # Simulates: prefix after << then constrained tokens
    result = h.LongestValidSuffix(["some", "<<", "a", "b"])
    assert result == ["a", "b"]


# ---------------------------------------------------------------------------
# CSDHelpers: UnconstrainedStep / ConstrainedStep
# ---------------------------------------------------------------------------

def test_unconstrained_step_returns_token_and_decrements():
    lm = SimpleLM(["a", "b", "c", "<<", ">>"])
    lm.Logits = [1.0, 2.0, 3.0, 0.0, 0.0]
    parser = SimpleParser()
    h = CSDHelpers(lm, parser)
    tok, steps = h.UnconstrainedStep([], [], 10)
    assert tok in lm.Tokens
    assert steps == 9


def test_constrained_step_produces_grammar_valid_token():
    lm = SimpleLM(["a", "b", "c", "<<", ">>", "x"])
    parser = SimpleParser()
    h = CSDHelpers(lm, parser)
    # generated = ["<<"] → LongestValidSuffix = [] → grammar start
    tok, steps = h.ConstrainedStep([], ["<<"], 10)
    assert tok in GRAMMAR_TOKENS
    assert steps == 9


# ---------------------------------------------------------------------------
# CSDHelpers: ForcedTokenStep / ergonomic wrappers
# ---------------------------------------------------------------------------

def test_forced_token_step_returns_exact_token():
    lm = SimpleLM(["a", "<<", ">>"])
    parser = SimpleParser()
    h = CSDHelpers(lm, parser)
    tok, steps = h.ForcedTokenStep([], [], "<<", 5)
    assert tok == "<<"
    assert steps == 4


def test_can_constrain_matches_suffix_completion():
    h = make_helpers()
    assert h.CanConstrain(["<<"])
    assert not h.CanConstrain(["<<", "a", "b", "c"])


def test_append_left_delimiter_appends_exact_token():
    lm = SimpleLM(["a", "<<", ">>"])
    parser = SimpleParser()
    h = CSDHelpers(lm, parser)
    generated, steps = h.AppendLeftDelimiter([], 5)
    assert generated == ["<<"]
    assert steps == 4


def test_append_constrained_step_appends_grammar_valid_token():
    lm = SimpleLM(["a", "b", "c", "<<", ">>", "x"])
    parser = SimpleParser()
    h = CSDHelpers(lm, parser)
    generated, steps = h.AppendConstrainedStep([], ["<<"], 10)
    assert generated[:1] == ["<<"]
    assert generated[-1] in GRAMMAR_TOKENS
    assert steps == 9


def test_append_right_delimiter_appends_exact_token():
    lm = SimpleLM(["a", "<<", ">>"])
    parser = SimpleParser()
    h = CSDHelpers(lm, parser)
    generated, steps = h.AppendRightDelimiter(["a"], 3)
    assert generated == ["a", ">>"]
    assert steps == 2


# ---------------------------------------------------------------------------
# CSDHelpers: SoftConstrainedStep
# ---------------------------------------------------------------------------

def test_soft_constrained_step_penalizes_invalid():
    lm = SimpleLM(["a", "b", "c", "<<", ">>"])
    lm.Logits = [0.0, 0.0, 0.0, 100.0, 100.0]
    parser = SimpleParser()
    h = CSDHelpers(lm, parser)
    # Without soft constraint, "<<" or ">>" would win (logit=100)
    # With SoftConstrainedStep(penalty=200), invalid tokens get -200 bias → ~-100
    tok, steps = h.SoftConstrainedStep([], ["<<"], 200.0, 10)
    assert tok in GRAMMAR_TOKENS
    assert steps == 9


# ---------------------------------------------------------------------------
# CSDHelpers: BudgetAwareStep
# ---------------------------------------------------------------------------

def test_budget_aware_step_switches_to_constrained_near_end():
    lm = SimpleLM(["a", "b", "c", "<<", ">>"])
    parser = SimpleParser()
    h = CSDHelpers(lm, parser)
    # stepsLeft=2, threshold=3 → stepsLeft <= threshold, grammar incomplete → ConstrainedStep
    # generated=["<<"] so LongestValidSuffix=[] which is not complete
    tok, steps = h.BudgetAwareStep([], ["<<"], 2, 3)
    assert tok in GRAMMAR_TOKENS
    assert steps == 1


def test_append_budget_aware_step_updates_prefix():
    lm = SimpleLM(["a", "b", "c", "<<", ">>"])
    parser = SimpleParser()
    h = CSDHelpers(lm, parser)
    generated, steps = h.AppendBudgetAwareStep([], ["<<"], 2, 3)
    assert generated[:1] == ["<<"]
    assert generated[-1] in GRAMMAR_TOKENS
    assert steps == 1


# ---------------------------------------------------------------------------
# CSDHelpers: RollbackToValidPrefix
# ---------------------------------------------------------------------------

def test_rollback_to_valid_prefix():
    lm = SimpleLM(["a", "b", "c", "x"])
    parser = SimpleParser()
    h = CSDHelpers(lm, parser)
    # "a", "b", "x" — "x" is invalid, so roll back to ["a", "b"]
    result = h.RollbackToValidPrefix(["a", "b", "x"])
    assert parser.IsValidPrefix(result)
    assert result == ["a", "b"]


def test_rollback_fully_invalid_returns_empty():
    lm = SimpleLM(["x", "y"])
    parser = SimpleParser()
    h = CSDHelpers(lm, parser)
    result = h.RollbackToValidPrefix(["x", "y"])
    assert result == []


# ---------------------------------------------------------------------------
# CSDHelpers: end-to-end with delimiters
# ---------------------------------------------------------------------------

def test_full_strategy_produces_delimited_output():
    """Smoke test: simulate a strategy that emits << constrained >> output."""
    tokens = list(GRAMMAR_TOKENS) + [LeftDelimiter, RightDelimiter, "free"]
    lm = SimpleLM(tokens)
    parser = SimpleParser()
    h = CSDHelpers(lm, parser)

    generated: Prefix = []
    stepsLeft = 20

    # Free-form phase
    tok, stepsLeft = h.UnconstrainedStep([], generated, stepsLeft)
    generated = generated + [tok]

    # Emit <<
    tok, stepsLeft = h.ForcedTokenStep([], generated, LeftDelimiter, stepsLeft)
    generated = generated + [tok]

    # Constrained phase
    while stepsLeft > 1 and not parser.IsCompletePrefix(h.LongestValidSuffix(generated)):
        tok, stepsLeft = h.ConstrainedStep([], generated, stepsLeft)
        generated = generated + [tok]

    # Emit >>
    tok, stepsLeft = h.ForcedTokenStep([], generated, RightDelimiter, stepsLeft)
    generated = generated + [tok]

    output = "".join(generated)
    assert "<<" in output
    assert ">>" in output
    suffix = h.LongestValidSuffix(generated[:-1])  # suffix before >>
    assert parser.IsCompletePrefix(suffix)


def test_full_strategy_with_append_helpers_produces_delimited_output():
    tokens = list(GRAMMAR_TOKENS) + [LeftDelimiter, RightDelimiter, "free"]
    lm = SimpleLM(tokens)
    parser = SimpleParser()
    h = CSDHelpers(lm, parser)

    generated: Prefix = []
    stepsLeft = 20

    generated, stepsLeft = h.AppendUnconstrainedStep([], generated, stepsLeft)
    generated, stepsLeft = h.AppendLeftDelimiter(generated, stepsLeft)

    while stepsLeft > 1 and h.CanConstrain(generated):
        generated, stepsLeft = h.AppendConstrainedStep([], generated, stepsLeft)

    generated, stepsLeft = h.AppendRightDelimiter(generated, stepsLeft)

    output = "".join(generated)
    assert "<<" in output
    assert ">>" in output
    suffix = h.LongestValidSuffix(generated[:-1])
    assert parser.IsCompletePrefix(suffix)


# ---------------------------------------------------------------------------
# CheckpointStack
# ---------------------------------------------------------------------------

def test_checkpoint_stack_push_pop():
    stack = CheckpointStack()
    assert stack.IsEmpty()
    stack.Push(["a", "b"])
    assert stack.Depth() == 1
    result = stack.Pop()
    assert result == ["a", "b"]
    assert stack.IsEmpty()


def test_checkpoint_stack_peek():
    stack = CheckpointStack()
    stack.Push(["x"])
    assert stack.Peek() == ["x"]
    assert stack.Depth() == 1  # peek does not remove


def test_checkpoint_stack_multiple():
    stack = CheckpointStack()
    stack.Push(["a"])
    stack.Push(["b"])
    stack.Push(["c"])
    assert stack.Depth() == 3
    assert stack.Pop() == ["c"]
    assert stack.Pop() == ["b"]
    assert stack.Depth() == 1


# ---------------------------------------------------------------------------
# RepetitionTracker
# ---------------------------------------------------------------------------

def test_repetition_tracker_records_and_counts():
    rt = RepetitionTracker(2)
    rt.RecordToken("a")
    rt.RecordToken("b")
    rt.RecordToken("a")
    rt.RecordToken("b")
    assert rt.GetCount(["a", "b"]) == 2


def test_repetition_tracker_penalty_zero_before_ngram_filled():
    rt = RepetitionTracker(3)
    rt.RecordToken("a")
    assert rt.GetRepetitionPenalty("b") == 0.0


def test_repetition_tracker_penalty_increases():
    rt = RepetitionTracker(2)
    rt.RecordToken("a")
    rt.RecordToken("b")
    rt.RecordToken("a")
    # After ["a","b","a"]: the bigram ("a","b") appeared once. GetRepetitionPenalty("b") checks recent=("a","b").
    penalty = rt.GetRepetitionPenalty("b")
    assert penalty >= 1.0


def test_apply_repetition_penalties_biases_lm():
    lm = SimpleLM(["a", "b"])
    lm.Logits = [5.0, 5.0]
    rt = RepetitionTracker(1)
    rt.RecordToken("a")
    rt.RecordToken("a")
    rt.ApplyRepetitionPenalties(lm)
    assert lm.Logits[0] < 5.0   # "a" penalized
    assert lm.Logits[1] == 5.0  # "b" unchanged
