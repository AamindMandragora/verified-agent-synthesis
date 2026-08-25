from dataclasses import FrozenInstanceError

import pytest

from synthesis.evaluate.benchmarks.sql_spider import eval_logic as sql_eval_logic
from synthesis.evaluate.benchmarks.sql_spider import generation as sql_generation


def _example():
    return {
        "db_id": "concert_singer",
        "db_info": "# singer ( singer_id , name )",
        "question": "How many singers do we have?",
    }


def _prompt():
    evaluator = type("PromptEvaluator", (), {"model_name": "Qwen/Qwen2.5-7B-Instruct"})()
    return sql_eval_logic.format_prompt(evaluator, _example())


def test_spider_format_prompt_exposes_shared_renderer_state():
    prompt = _prompt()

    assert callable(getattr(prompt, "render_for_model", None))
    assert prompt.raw_text == str(prompt)
    assert str(prompt).endswith("SQL:")


def test_spider_shared_renderer_matches_qwen35_fixed_adapter():
    calls = []

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            calls.append((messages, kwargs))
            return "<|user|>SPIDER<|assistant|>"

    prompt = _prompt()
    rendered = prompt.render_for_model(Tokenizer(), model_name="Qwen/Qwen3.5-2B")

    assert rendered == "<|user|>SPIDER<|assistant|>"
    assert calls[-1][1]["enable_thinking"] is False


def test_spider_csd_entry_keeps_structured_prompt_and_no_raw_completion_mode(monkeypatch):
    observed = {}

    def fake_run_crane_csd(*args, **kwargs):
        observed.update(kwargs)
        return "", 0, 0.0, [], []

    monkeypatch.setattr(sql_generation, "run_crane_csd", fake_run_crane_csd)
    runner = sql_eval_logic.get_generation_runner()
    runner(
        env={},
        prompt_text=_prompt(),
        max_steps=1,
        step_token_budget=1,
        grammar_file=None,
    )

    assert callable(getattr(observed["prompt_text"], "render_for_model", None))
    assert observed.get("completion_mode", False) is False
    assert observed["start_inside_constrained"] is True


def test_spider_guidance_is_before_the_final_sql_cue():
    prompt = _prompt()
    guided = prompt.with_guidance("Use only the supplied tables.")

    assert guided.raw_text.index("Additional task guidance from CSD:") < guided.raw_text.rindex("SQL:")
    assert guided.raw_text.endswith("SQL:")


class _Parser:
    def parse(self, text):
        if text not in {"SELECT name FROM singer", "SELECT 'SQL:' FROM singer"}:
            raise ValueError("invalid SQL")
        return object()



class _Evaluator:
    def _get_syntax_parser(self, example):
        return _Parser()


def test_spider_existing_entry_rejects_sql_label_wrapper():
    actual, source, aux = sql_eval_logic.extract_actual(
        _Evaluator(), "SQL: SELECT name FROM singer", _example()
    )

    assert actual is None
    assert source == "spider_output_contract_rejected"
    assert aux["output_contract_valid"] is False


def test_spider_existing_entry_keeps_supported_string_marker_as_sql():
    actual, source, aux = sql_eval_logic.extract_actual(
        _Evaluator(), "SELECT 'SQL:' FROM singer", _example()
    )

    assert actual == "SELECT 'SQL:' FROM singer"
    assert source == "bare_sql"
    assert aux["output_contract_valid"] is True


def test_spider_existing_entry_rejects_prose_wrapped_sql():
    actual, source, aux = sql_eval_logic.extract_actual(
        _Evaluator(),
        "Here is the query: SELECT name FROM singer",
        _example(),
    )

    assert actual is None


def test_guidance_without_registered_prompt_fails_closed():
    from synthesis.evaluate.benchmarks.common.model_utils import (
        _TaskGuidanceState,
        _TensorizedLMBase,
    )

    lm = object.__new__(_TensorizedLMBase)
    lm._task_guidance = _TaskGuidanceState()
    lm._structured_prompt = None
    lm._chat_messages = None

    with pytest.raises(RuntimeError, match="registered structured or chat prompt"):
        lm.AppendTaskGuidance("Use only supplied tables")


class _TraceTokenizer:
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return "<chat>" + messages[0]["content"] + "<assistant>"

    def encode(self, text, add_special_tokens=False):
        return [ord(char) for char in text]


def _model_evaluator(model_name):
    return type("ModelEvaluator", (), {"model_name": model_name})()


def test_qwen35_csd_template_disables_thinking_without_retry():
    calls = []

    class FailingTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            calls.append(kwargs)
            raise TypeError("thinking argument is required")

    prompt = sql_eval_logic.format_prompt(
        _model_evaluator("Qwen/Qwen3.5-4B"), _example()
    )
    from synthesis.evaluate.benchmarks.sql_spider.prompts import SpiderPromptRenderError

    with pytest.raises(SpiderPromptRenderError, match="chat-template rendering failed"):
        prompt.render_for_model(FailingTokenizer(), model_name="Qwen/Qwen3.5-4B")
    assert len(calls) == 1
    assert calls[0]["enable_thinking"] is False


def test_qwen25_baseline_and_csd_keep_historical_raw_prompt():
    tokenizer = _TraceTokenizer()
    prompt = sql_eval_logic.format_prompt(
        _model_evaluator("Qwen/Qwen2.5-7B-Instruct"), _example()
    )

    assert prompt.render_for_model(tokenizer, model_name="Qwen/Qwen2.5-7B-Instruct") == prompt.raw_text
    assert tokenizer.calls == []


def test_no_guidance_prompt_bytes_match_historical_contract():
    prompt = sql_eval_logic.format_prompt(
        _model_evaluator("Qwen/Qwen3.5-2B"), _example()
    )
    expected = (
        "db_id: concert_singer\n"
        "db_info: # singer ( singer_id , name )\n"
        "question: How many singers do we have? Only output the SQL quey. \n"
        "SQL:"
    )

    assert prompt.raw_text == expected
    assert str(prompt) == expected


def test_spider_prompt_parts_are_immutable():
    prompt = _prompt()

    with pytest.raises(FrozenInstanceError):
        prompt.task_text = "changed"


def test_token0_guidance_rebuilds_before_final_sql_cue():
    from synthesis.evaluate.benchmarks.common.model_utils import (
        _TaskGuidanceState,
        _TensorizedLMBase,
    )

    prompt = _prompt()
    lm = object.__new__(_TensorizedLMBase)
    lm._task_guidance = _TaskGuidanceState()
    lm._structured_prompt = None
    lm._chat_messages = None
    lm.tokenizer = _TraceTokenizer()
    lm._tried_token_penalties = {}
    lm._penalty_instruction_key = None
    lm._grounding_cache_key = None
    lm._grounding_cache_val = set()
    lm._last_generation_evidence = None
    lm._last_full_prompt = None
    lm._logits_dirty = False
    lm.set_structured_prompt(prompt, model_name="Qwen/Qwen2.5-7B-Instruct")
    lm.instruction_text = prompt.raw_text

    lm.AppendTaskGuidance("Use only supplied tables; SQL: is part of this guidance")
    guided = lm._structured_prompt.raw_text

    assert guided.count("Additional task guidance from CSD:") == 1
    assert guided.index("Additional task guidance from CSD:") < guided.rindex("SQL:")
    assert guided.endswith("SQL:")
    assert lm.instruction_text == guided


def test_guidance_first_call_wins_and_reset_prevents_leakage():
    from synthesis.evaluate.benchmarks.common.model_utils import (
        _TaskGuidanceState,
        _TensorizedLMBase,
    )

    lm = object.__new__(_TensorizedLMBase)
    lm._task_guidance = _TaskGuidanceState()
    lm._structured_prompt = None
    lm._chat_messages = None
    lm.tokenizer = _TraceTokenizer()
    lm._tried_token_penalties = {}
    lm._penalty_instruction_key = None
    lm._grounding_cache_key = None
    lm._grounding_cache_val = set()
    lm._last_generation_evidence = None
    lm._last_full_prompt = None
    lm._logits_dirty = False
    first = _prompt()
    lm.set_structured_prompt(first, model_name="Qwen/Qwen2.5-1.5B-Instruct")
    lm.instruction_text = first.raw_text

    lm.AppendTaskGuidance("")
    lm.AppendTaskGuidance("first guidance")
    lm.AppendTaskGuidance("second guidance")
    assert lm.task_guidance == "first guidance"
    assert "second guidance" not in lm.instruction_text

    lm.ResetTaskGuidance()
    assert lm.task_guidance is None
    assert lm._structured_prompt is None
    assert lm._chat_messages is None
    assert lm.instruction_text == ""

    second = sql_eval_logic.format_prompt(
        _model_evaluator("Qwen/Qwen2.5-1.5B-Instruct"),
        {**_example(), "question": "What is singer_id?"},
    )
    lm.set_structured_prompt(second, model_name="Qwen/Qwen2.5-1.5B-Instruct")
    lm.instruction_text = second.raw_text
    lm.AppendTaskGuidance("new guidance")
    assert "first guidance" not in lm.instruction_text
    assert "new guidance" in lm.instruction_text


def test_guidance_without_rebuild_state_fails_without_mutation():
    from synthesis.evaluate.benchmarks.common.model_utils import (
        _TaskGuidanceState,
        _TensorizedLMBase,
    )

    lm = object.__new__(_TensorizedLMBase)
    lm._task_guidance = _TaskGuidanceState()
    lm._structured_prompt = None
    lm._chat_messages = None
    lm.instruction_text = "historical prompt"

    with pytest.raises(RuntimeError, match="registered structured or chat prompt"):
        lm.AppendTaskGuidance("Use only supplied tables")

    assert lm.instruction_text == "historical prompt"
    assert lm.task_guidance is None


def test_gsm_runner_registers_chat_guidance_and_resets_between_examples(tmp_path):
    from synthesis.evaluate.benchmarks.common.model_utils import _TensorizedLMBase
    from synthesis.evaluate.benchmarks.gsm_symbolic import eval_logic as gsm_eval_logic

    class Dafny:
        @staticmethod
        def Seq(value):
            return value

        @staticmethod
        def SeqWithoutIsStrInference(values):
            return list(values)

    class LegacyChatTokenizer:
        eos_token = "<eos>"
        eos_token_id = 99
        all_special_ids = {99}

        def __init__(self):
            self.calls = []

        def apply_chat_template(self, messages, **kwargs):
            snapshot = [dict(message) for message in messages]
            self.calls.append((snapshot, dict(kwargs)))
            if "enable_thinking" in kwargs:
                raise TypeError("legacy tokenizer has no thinking option")
            return "\n".join(
                f"{message['role']}:{message.get('content', '')}"
                for message in messages
            )

        def decode(self, token_ids, skip_special_tokens=False):
            del skip_special_tokens
            return "".join("ok" if int(token_id) == 1 else "<eos>" for token_id in token_ids)

        def encode(self, text, add_special_tokens=False):
            del text, add_special_tokens
            return [1]

    class Parser:
        @staticmethod
        def is_complete(text):
            return text == "ok"

    class GeneratedDefault:
        @staticmethod
        def MyCSDStrategy(
            lm_arg,
            parser,
            seq0,
            generated_prefix,
            start_inside,
            current_constrained,
            max_steps,
            step_budget,
            eos_token,
        ):
            del parser, seq0, generated_prefix, start_inside, current_constrained
            del max_steps, step_budget, eos_token
            if len(captures) == 0:
                lm_arg.AppendTaskGuidance("")
                lm_arg.AppendTaskGuidance("first guidance")
                lm_arg.AppendTaskGuidance("second guidance")
            captures.append(
                {
                    "messages": [dict(message) for message in lm_arg._chat_messages],
                    "instruction_text": lm_arg.instruction_text,
                    "task_guidance": lm_arg.task_guidance,
                }
            )
            return (["ok"], False, [], 1)

    class GeneratedCSD:
        default__ = GeneratedDefault

    tokenizer = LegacyChatTokenizer()
    lm = _TensorizedLMBase(Dafny(), tokenizer, ["ok"], [1])
    env = {
        "_dafny": Dafny,
        "GeneratedCSD": GeneratedCSD,
        "lm": lm,
        "parser": Parser(),
    }
    captures = []
    runner = gsm_eval_logic.get_generation_runner()

    first_prompt = gsm_eval_logic.format_prompt(
        object(), {"question": "How many singers?"}
    )
    runner(
        env=env,
        prompt_text=first_prompt,
        max_steps=8,
        grammar_file=tmp_path / "unused.lark",
    )

    second_prompt = gsm_eval_logic.format_prompt(
        object(), {"question": "How many albums?"}
    )
    runner(
        env=env,
        prompt_text=second_prompt,
        max_steps=8,
        grammar_file=tmp_path / "unused.lark",
    )

    assert len(captures) == 2
    first_user = [message for message in captures[0]["messages"] if message["role"] == "user"][-1]
    assert "first guidance" in first_user["content"]
    assert "second guidance" not in first_user["content"]
    assert captures[0]["task_guidance"] == "first guidance"
    assert "first guidance" not in captures[1]["instruction_text"]
    assert "second guidance" not in captures[1]["instruction_text"]
    assert captures[1]["task_guidance"] is None

    thinking_calls = [kwargs for _, kwargs in tokenizer.calls if "enable_thinking" in kwargs]
    legacy_calls = [kwargs for _, kwargs in tokenizer.calls if "enable_thinking" not in kwargs]
    assert len(thinking_calls) == 3
    assert len(legacy_calls) == 3
