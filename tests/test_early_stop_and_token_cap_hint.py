"""Tests for the CRANE-style early-stop-on-answer harness and the
token-cap-exhaustion author hint (both flag-gated, default OFF).

Written FIRST (TDD red) — the names under test do not exist yet.
"""

import pytest


# ---------------------------------------------------------------------------
# 1. _answer_complete predicate (model_utils)
# ---------------------------------------------------------------------------

def _predicate():
    from synthesis.evaluate.benchmarks.common.model_utils import _answer_complete
    return _answer_complete


def test_answer_complete_false_without_phrase():
    assert _predicate()("some reasoning <<x + 1>> more text") is False


def test_answer_complete_false_phrase_but_no_span():
    assert _predicate()("The final answer is 42") is False


def test_answer_complete_false_unclosed_span():
    assert _predicate()("The final answer is <<x + 1") is False


def test_answer_complete_true_complete_span():
    assert _predicate()("Let's think. <<a*2>> The final answer is <<x + 1>>.") is True


def test_answer_complete_case_insensitive():
    assert _predicate()("the FINAL ANSWER is <<3//2>>\n") is True


def test_answer_complete_false_bare_closed_span_needs_lookahead():
    # Span just closed; we cannot yet know whether the expression continues
    # (e.g. "<<n1>> + <<mult>>"), so do NOT stop until the next token shows
    # a non-continuation character.
    assert _predicate()("The final answer is <<x + 1>>") is False


def test_answer_complete_false_expression_continues_after_span():
    assert _predicate()("The final answer is <<n1>> + ") is False
    assert _predicate()("The final answer is <<n1>> * <<mult>>") is False
    assert _predicate()("The final answer is <<n1>> <<") is False


def test_answer_complete_true_multi_span_then_terminator():
    assert _predicate()("The final answer is <<n1>> + <<n2>>.") is True


def test_answer_complete_false_span_only_before_phrase():
    # A span that closed BEFORE the phrase does not count as the answer span.
    assert _predicate()("<<x+1>> The final answer is ") is False


# ---------------------------------------------------------------------------
# 2. _check_answer_early_stop hook behavior (duck-typed fake self)
# ---------------------------------------------------------------------------

class _FakeLM:
    def __init__(self, enabled):
        self._answer_early_stop_enabled = enabled
        self._early_stop_tokens = None

    def _to_str(self, obj):
        return obj if isinstance(obj, str) else "".join(obj)


def _hook():
    from synthesis.evaluate.benchmarks.common.model_utils import _TensorizedLMBase
    return _TensorizedLMBase._check_answer_early_stop


def test_hook_noop_when_disabled():
    fake = _FakeLM(enabled=False)
    _hook()(fake, ["The final answer is <<x+1>>"])  # must not raise
    assert fake._early_stop_tokens is None


def test_hook_noop_when_answer_incomplete():
    fake = _FakeLM(enabled=True)
    _hook()(fake, ["The final answer is <<x+1"])  # must not raise
    assert fake._early_stop_tokens is None


def test_hook_raises_and_stashes_when_answer_complete():
    from synthesis.evaluate.benchmarks.common.model_utils import AnswerCompleteStop
    fake = _FakeLM(enabled=True)
    tokens = ["The final answer", " is <<x", "+1>>", "."]
    with pytest.raises(AnswerCompleteStop):
        _hook()(fake, tokens)
    assert fake._early_stop_tokens == tokens


def test_hook_noop_when_span_closed_but_expression_may_continue():
    fake = _FakeLM(enabled=True)
    _hook()(fake, ["The final answer is <<n1>>"])  # must not raise yet
    assert fake._early_stop_tokens is None


# ---------------------------------------------------------------------------
# 3. _token_cap_exhaustion_hint (feedback_loop)
# ---------------------------------------------------------------------------

def _hint():
    from synthesis.evaluate.feedback_loop import _token_cap_exhaustion_hint
    return _token_cap_exhaustion_hint


def _samples(n_capped, n_total):
    return (
        [{"hit_max_steps": True} for _ in range(n_capped)]
        + [{"hit_max_steps": False} for _ in range(n_total - n_capped)]
    )


def test_hint_empty_for_no_samples():
    assert _hint()([], 900) == ""
    assert _hint()(None, 900) == ""


def test_hint_empty_below_half_capped():
    assert _hint()(_samples(2, 10), 900) == ""


def test_hint_fires_at_half_capped_with_counts():
    text = _hint()(_samples(5, 10), 900)
    assert "5/10" in text
    assert "900" in text
    # Causal explanation must be present: truncation -> last-span grading.
    assert "cut off" in text.lower() or "truncat" in text.lower()


# ---------------------------------------------------------------------------
# 4. run_crane_csd early-stop path returns output-so-far NORMALLY (the crux)
# ---------------------------------------------------------------------------

class _FakeSeq(list):
    """Stands in for _dafny.Seq / SeqWithoutIsStrInference results."""


class _FakeDafny:
    @staticmethod
    def Seq(x):
        return x

    @staticmethod
    def SeqWithoutIsStrInference(x):
        return _FakeSeq(x)


class _FakeTokenizer:
    eos_token = "<|eos|>"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kw):
        return messages[-1]["content"]


class _StoppingLM:
    """Fake LM whose strategy call ends via AnswerCompleteStop."""

    def __init__(self):
        self.tokenizer = _FakeTokenizer()
        self.instruction_text = ""
        self._answer_early_stop_enabled = False
        self._early_stop_tokens = None
        self.task_guidance = None

    def SetAnswerEarlyStop(self, enabled):
        self._answer_early_stop_enabled = bool(enabled)
        if enabled:
            self._early_stop_tokens = None

    def ClearRuntimeDeadline(self):
        pass

    def SetRuntimeDeadline(self, deadline):
        pass

    def ResetTaskGuidance(self):
        pass

    def set_chat_messages(self, messages):
        pass


def _make_env(lm, strategy):
    class _GeneratedDefault:
        MyCSDStrategy = strategy

    class _GeneratedCSD:
        default__ = _GeneratedDefault

    return {"_dafny": _FakeDafny, "GeneratedCSD": _GeneratedCSD, "lm": lm, "parser": object()}


def test_run_crane_csd_returns_output_so_far_on_answer_complete_stop(tmp_path):
    from synthesis.evaluate.benchmarks.common.model_utils import AnswerCompleteStop
    from synthesis.evaluate.benchmarks.gsm_symbolic.generation import run_crane_csd

    lm = _StoppingLM()
    stashed = ["The final answer is ", "<<x+1>>"]

    def strategy(lm_arg, parser, seq0, prefix, start_inside, cur, max_steps, step_budget, eos):
        # Simulate the per-step hook firing mid-generation.
        lm_arg._early_stop_tokens = list(stashed)
        raise AnswerCompleteStop("answer span complete")

    env = _make_env(lm, strategy)
    output_text, token_count, gen_time, segments, helper_trace, constrained_work = run_crane_csd(
        env=env,
        prompt_text="solve it",
        max_steps=900,
        grammar_file=tmp_path / "unused.lark",
        early_stop_on_answer=True,
    )
    assert output_text == "The final answer is <<x+1>>"
    assert token_count == len(stashed)
    assert constrained_work == len(stashed)
    # Flag must be cleared after the call so it cannot leak across examples.
    assert lm._answer_early_stop_enabled is False


def test_run_crane_csd_without_flag_does_not_enable(tmp_path):
    from synthesis.evaluate.benchmarks.gsm_symbolic.generation import run_crane_csd

    lm = _StoppingLM()

    def strategy(lm_arg, parser, seq0, prefix, start_inside, cur, max_steps, step_budget, eos):
        assert lm_arg._answer_early_stop_enabled is False
        return (_FakeSeq(["ok"]), False, _FakeSeq([]), 7)

    env = _make_env(lm, strategy)
    output_text, token_count, _, _, _, constrained_work = run_crane_csd(
        env=env,
        prompt_text="solve it",
        max_steps=900,
        grammar_file=tmp_path / "unused.lark",
    )
    assert output_text == "ok"
    assert constrained_work == 7
