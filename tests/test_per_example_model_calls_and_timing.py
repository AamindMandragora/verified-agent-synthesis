"""Per-example cost accounting (2026-09-25).

1. The [STEP_BREAKDOWN] line must describe ONE example. The timing dicts in
   model_utils/parser_utils are cumulative for the whole process, so the
   per-example numbers have to be the change across the example, not the
   running total (the running total made every example after the first print
   0.0% dafny/uninstrumented).
2. Every per-example result records how many model calls (GenerateLogits calls
   that reached the model) that example used, on both LM backends.
3. The author feedback carries a short, task-agnostic model-call cost section.

CPU only: the model, engine and strategy are fakes.
"""
from __future__ import annotations

import re
from collections import defaultdict

import pytest


# ---------------------------------------------------------------------------
# 1. per-example STEP_BREAKDOWN
# ---------------------------------------------------------------------------


class _Seq(list):
    pass


class _FakeDafny:
    @staticmethod
    def Seq(x):
        return x

    @staticmethod
    def SeqWithoutIsStrInference(x):
        return _Seq(x)


class _FakeTokenizer:
    eos_token = "<|eos|>"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kw):
        return messages[-1]["content"]


class _PlainLM:
    def __init__(self):
        self.tokenizer = _FakeTokenizer()
        self.instruction_text = ""
        self.task_guidance = None

    def SetRuntimeDeadline(self, deadline):
        pass

    def ClearRuntimeDeadline(self):
        pass

    def ResetTaskGuidance(self):
        pass

    def set_chat_messages(self, messages):
        pass


def _env(lm, strategy):
    class _Default:
        MyCSDStrategy = strategy

    class _Generated:
        default__ = _Default

    return {"_dafny": _FakeDafny, "GeneratedCSD": _Generated, "lm": lm, "parser": object()}


_BREAKDOWN = re.compile(
    r"\[STEP_BREAKDOWN\] wall=([\d.]+)s\s+lm_callbacks=([\d.]+)s\s+"
    r"parser_callbacks=([\d.]+)s\s+dafny_internal=([\d.]+)s"
)


def test_step_breakdown_is_per_example_not_cumulative(monkeypatch, capsys, tmp_path):
    from synthesis.evaluate.benchmarks.common import model_utils, parser_utils
    from synthesis.evaluate.benchmarks.gsm_symbolic.generation import run_crane_csd

    monkeypatch.setattr(model_utils, "_TIMINGS_ENABLED", True)
    monkeypatch.setattr(parser_utils, "_PARSER_TIMINGS_ENABLED", True)
    # Earlier examples in the same process already accumulated a lot of time.
    lm_timings = defaultdict(lambda: [0.0, 0])
    lm_timings["GenerateLogits.total"] = [500.0, 4000]
    parser_timings = defaultdict(lambda: [0.0, 0])
    parser_timings["accept_mask"] = [200.0, 9000]
    monkeypatch.setattr(model_utils, "_TIMINGS", lm_timings)
    monkeypatch.setattr(parser_utils, "_PARSER_TIMINGS", parser_timings)

    import time

    def strategy(lm_arg, parser, seq0, prefix, start_inside, cur, max_steps, step_budget, eos):
        with model_utils._timed("GenerateLogits.total"):
            time.sleep(0.05)
        time.sleep(0.25)  # strategy-side work no callback timer covers
        return (_Seq(["ok"]), False, _Seq([]), 1)

    run_crane_csd(
        env=_env(_PlainLM(), strategy),
        prompt_text="q",
        max_steps=10,
        grammar_file=tmp_path / "unused.lark",
    )
    match = _BREAKDOWN.search(capsys.readouterr().out)
    assert match, "no [STEP_BREAKDOWN] line printed"
    wall, lm_cb, parser_cb, dafny = (float(g) for g in match.groups())
    assert 0.04 <= lm_cb < 0.5, lm_cb
    assert parser_cb < 0.5, parser_cb
    assert dafny >= 0.2, dafny
    assert wall >= 0.29


# ---------------------------------------------------------------------------
# 2a. both LM backends count model calls per example
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", ["huggingface", "vllm"])
def test_lm_counts_model_calls_per_example(monkeypatch, backend):
    from tests.test_spider_output_contract import _factory_entry_lm

    events = []
    lm = _factory_entry_lm(monkeypatch, backend, events)

    lm.ResetModelCallCount()
    assert lm.ModelCallCount() == 0

    lm.GenerateLogits([])
    assert lm.ModelCallCount() == 1
    # Same prompt, logits still fresh: served from the prefix cache, no model call.
    lm.GenerateLogits([])
    assert lm.ModelCallCount() == 1
    # Logits invalidated (what a rollback does): the model is called again.
    lm._logits_dirty = True
    lm.GenerateLogits([])
    assert lm.ModelCallCount() == 2
    assert events.count("backend") == 2

    lm.ResetModelCallCount()
    assert lm.ModelCallCount() == 0


# ---------------------------------------------------------------------------
# 2b. the evaluator records model_calls in every per-example result
# ---------------------------------------------------------------------------


class _CountingLM:
    task_guidance = None
    _last_generation_evidence = None
    _last_prompt_contract = None

    def __init__(self, stale=0):
        self._calls = stale

    def ResetModelCallCount(self):
        self._calls = 0

    def ModelCallCount(self):
        return self._calls


class _Logic:
    def build_dynamic_parser(self, evaluator, env, example):
        return None

    def accuracy_applicable(self, aux):
        return True


def _bare_evaluator():
    from synthesis.evaluate.evaluator import Evaluator

    ev = Evaluator.__new__(Evaluator)
    ev.dataset_name = "gsm_symbolic"
    ev.max_steps = 900
    ev.step_token_budget = 1
    ev.max_seconds_per_example = None
    ev.early_stop_on_answer = False
    ev._format_prompt = lambda example: "prompt"
    ev._get_expected_answer = lambda example: "42"
    ev._get_grammar_file = lambda: None
    ev._accuracy_applicable_for_example = lambda aux: True
    return ev


def test_timed_out_example_records_its_model_calls():
    from synthesis.evaluate.evaluator import Evaluator, PerExampleTimeout

    lm = _CountingLM(stale=999)  # left over from the previous example

    def run(**kwargs):
        lm._calls += 37
        raise PerExampleTimeout("out of time")

    sample = Evaluator._evaluate_one_example(
        _bare_evaluator(), 0, {"question": "q"}, 1, {"lm": lm}, _Logic(), run, {}
    )
    assert sample["timed_out"] is True
    assert sample["model_calls"] == 37


def test_failed_example_does_not_inherit_previous_examples_calls():
    from synthesis.evaluate.evaluator import Evaluator

    lm = _CountingLM(stale=999)

    def run(**kwargs):
        raise RuntimeError("strategy fell over before any model call")

    sample = Evaluator._evaluate_one_example(
        _bare_evaluator(), 0, {"question": "q"}, 1, {"lm": lm}, _Logic(), run, {}
    )
    assert sample["model_calls"] == 0


def test_completed_example_records_its_model_calls(monkeypatch):
    from synthesis.evaluate.benchmarks.sql_spider import eval_logic as sql_eval_logic
    from synthesis.evaluate.benchmarks.sql_spider import executor
    from synthesis.evaluate.evaluator import Evaluator
    from tests.test_spider_output_contract import _GRAMMAR_PATH, _example

    evaluator = Evaluator(
        dataset_name="spider",
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
        backend="huggingface",
        device="cpu",
        sample_size=1,
        max_steps=8,
    )
    evaluator._base_grammar_text = _GRAMMAR_PATH.read_text()
    monkeypatch.setattr(
        executor, "prediction_matches_gold", lambda actual, row: actual == row.get("query")
    )
    lm = _CountingLM(stale=999)

    def run(**kwargs):
        lm._calls += 12
        return "<<SELECT name FROM singer>>", 4, 0.01, [], []

    sample = evaluator._evaluate_one_example(
        0, _example(), 1, {"lm": lm, "tokenizer": None}, sql_eval_logic, run, {}
    )
    assert sample["is_correct"] is True
    assert sample["model_calls"] == 12


# ---------------------------------------------------------------------------
# 3. author-facing model-call cost section
# ---------------------------------------------------------------------------


def _result(samples):
    from synthesis.evaluate.evaluator import EvaluationResult

    return EvaluationResult(
        success=False,
        accuracy=0.2,
        contains_delimiters=True,
        syntax_rate=0.8,
        num_examples=len(samples),
        num_correct=1,
        total_time_seconds=100.0,
        sample_outputs=samples,
    )


def _sample(idx, calls, tokens, timed_out=False):
    return {
        "example_index": idx,
        "question": "SECRET QUESTION TEXT",
        "full_output": "SECRET OUTPUT TEXT",
        "model_calls": calls,
        "token_count": tokens,
        "timed_out": timed_out,
        "runtime_budget_exceeded": timed_out,
        "is_correct": False,
        "syntax_rate": 0.0,
    }


def _cost_section(text):
    start = text.index("Model Call Cost")
    rest = text[start:]
    end = rest.find("\n\n")
    return rest if end < 0 else rest[:end]


def test_feedback_summary_has_model_call_cost_section():
    samples = [
        _sample(0, 380, 300),
        _sample(1, 120, 90),
        _sample(2, 5000, 0, timed_out=True),
        _sample(3, 900, 410),
        _sample(4, 60, 40),
    ]
    section = _cost_section(_result(samples).get_feedback_summary(require_delimiters=True))

    assert "median 380" in section
    assert "max 5000" in section
    assert "time limit: 1/5" in section
    # three worst, most calls first, 1-based example numbers, calls + output tokens
    worst = section[section.index("Most calls"):]
    assert worst.index("#3") < worst.index("#4") < worst.index("#1")
    assert "5000 calls, 0 output tokens" in worst
    assert "900 calls, 410 output tokens" in worst
    assert "#2 " not in worst and "#5 " not in worst
    # a few lines, mechanism facts only: nothing from the task leaks in
    assert len(section.splitlines()) <= 5
    assert "SECRET" not in section


def test_cost_section_absent_when_samples_have_no_call_counts():
    samples = [_sample(0, None, 10), _sample(1, None, 20)]
    for s in samples:
        s.pop("model_calls")
    text = _result(samples).get_feedback_summary(require_delimiters=True)
    assert "Model Call Cost" not in text
