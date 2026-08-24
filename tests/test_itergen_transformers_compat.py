import argparse
import sys
import types

import pytest

from synthesis.evaluate import run_legacy_fixed_strategy as legacy_runner
from synthesis.evaluate.benchmarks.sql_spider import eval_logic as sql_eval_logic
from synthesis.evaluate.benchmarks.sql_spider.prompts import SpiderPromptParts
from synthesis.evaluate.run_legacy_fixed_strategy import (
    _crane_adaptive_surface,
    _crane_stop_words,
    _itergen_generation_kwargs,
    _itergen_generate_spider,
    _install_itergen_transformers_compat,
    _legacy_fixed_max_new_tokens,
)


class _GenerationConfig:
    def __init__(self, *, do_sample: bool):
        self.do_sample = do_sample

    def update(self, **values):
        for key, value in values.items():
            setattr(self, key, value)


class _ModernModel:
    def __init__(self):
        self.processor_calls = []

    def _get_logits_processor(self, **kwargs):
        self.processor_calls.append(kwargs)
        return "transformers-5-processors"


class _OldModel:
    def __init__(self):
        self.calls = []

    def _get_logits_warper(self, generation_config, *, device):
        self.calls.append((generation_config, device))
        return "legacy-warper"


class _LegacyIterGen:
    def update_gen_args(self, **gen_args):
        self.generation_config.update(**gen_args)
        self.logit_warper = self.model._get_logits_warper(
            self.generation_config,
            device=self.device,
        )


def _instance(*, do_sample: bool):
    value = _LegacyIterGen()
    value.generation_config = _GenerationConfig(do_sample=do_sample)
    value.model = _ModernModel()
    value.device = "cuda:0"
    return value


def test_modern_transformers_greedy_itergen_uses_identity_warper():
    _install_itergen_transformers_compat(_LegacyIterGen)
    value = _instance(do_sample=False)

    value.update_gen_args(max_new_tokens=128)

    assert value.generation_config.max_new_tokens == 128
    assert list(value.logit_warper) == []


def test_modern_transformers_restores_legacy_greedy_beam_defaults():
    _install_itergen_transformers_compat(_LegacyIterGen)
    value = _instance(do_sample=False)
    value.generation_config.num_beams = None
    value.generation_config.num_beam_groups = None

    value.update_gen_args(max_new_tokens=128)

    assert value.generation_config.num_beams == 1
    assert value.generation_config.num_beam_groups == 1


def test_modern_transformers_sampling_uses_transformers5_processors():
    _install_itergen_transformers_compat(_LegacyIterGen)
    value = _instance(do_sample=True)

    value.update_gen_args(max_new_tokens=128, temperature=0.7)

    assert value.logit_warper == "transformers-5-processors"
    assert len(value.model.processor_calls) == 1
    call = value.model.processor_calls[0]
    assert call["generation_config"] is value.generation_config
    assert call["input_ids_seq_length"] == 0
    assert call["device"] == "cuda:0"


def test_older_transformers_keeps_the_original_logits_warper_path():
    class OldIterGen:
        def update_gen_args(self, **gen_args):
            raise AssertionError("the compatibility wrapper should replace this method")

    _install_itergen_transformers_compat(OldIterGen)
    value = OldIterGen()
    value.generation_config = _GenerationConfig(do_sample=False)
    value.model = _OldModel()
    value.device = "cuda:2"

    value.update_gen_args(max_new_tokens=64)

    assert value.logit_warper == "legacy-warper"
    assert value.model.calls == [(value.generation_config, "cuda:2")]


def test_qwen35_itergen_rebuilds_its_cache_with_linear_attention_layers():
    class TextConfig:
        num_hidden_layers = 2
        layer_types = ["linear_attention", "full_attention"]
        sliding_window = None
        attention_chunk_size = None

    class ModelConfig:
        def get_text_config(self, *, decoder):
            assert decoder is True
            return TextConfig()

    class HybridModel:
        config = ModelConfig()

    class HybridIterGen:
        def update_gen_args(self, **gen_args):
            raise AssertionError("not used by this test")

        def start(self, prompt):
            self.model_kwargs = {"past_key_values": "legacy-lazy-cache"}
            return prompt

    _install_itergen_transformers_compat(HybridIterGen)
    value = HybridIterGen()
    value.model = HybridModel()

    assert value.start("prompt") == "prompt"
    assert value.model_kwargs["past_key_values"].get_seq_length() == 0


def test_full_attention_itergen_keeps_its_existing_cache_path():
    class TextConfig:
        layer_types = ["full_attention"]

    class ModelConfig:
        def get_text_config(self, *, decoder):
            return TextConfig()

    class FullAttentionModel:
        config = ModelConfig()

    class FullAttentionIterGen:
        def update_gen_args(self, **gen_args):
            raise AssertionError("not used by this test")

        def start(self, prompt):
            self.model_kwargs = {"past_key_values": "legacy-lazy-cache"}

    _install_itergen_transformers_compat(FullAttentionIterGen)
    value = FullAttentionIterGen()
    value.model = FullAttentionModel()

    value.start("prompt")

    assert value.model_kwargs["past_key_values"] == "legacy-lazy-cache"


def test_spider_itergen_uses_the_checked_in_upstream_search_settings():
    kwargs = _itergen_generation_kwargs(
        dataset="spider",
        max_tokens=8192,
        max_new_tokens=176,
    )

    assert kwargs["do_sample"] is False
    assert kwargs["recurrence_penalty"] == 0.3

def test_smiles_itergen_samples_at_the_approved_temperature():
    kwargs = _itergen_generation_kwargs(
        dataset="smiles",
        max_tokens=8192,
        max_new_tokens=400,
    )

    assert kwargs["do_sample"] is True
    assert kwargs["temperature"] == 0.7
    assert "recurrence_penalty" not in kwargs



def test_spider_itergen_advances_by_schema_units_and_backtracks_invalid_names():
    class FakeIterGen:
        def __init__(self):
            self.started_with = None
            self.forward_calls = []
            self.backward_calls = []
            self.iteration = 0

        def start(self, prompt):
            self.started_with = prompt

        def finished(self):
            return self.iteration >= 2

        def forward(self, **kwargs):
            self.forward_calls.append(kwargs)
            self.iteration += 1
            return ["SELECT invented FROM singer" if self.iteration == 1 else "SELECT name FROM singer"]

        def view(self, unit):
            if unit == "column_name":
                return [["invented"]] if self.iteration == 1 else [["name"]]
            if unit == "table_name":
                return [["singer"]]
            raise AssertionError(unit)

        def backward(self, unit):
            self.backward_calls.append(unit)

    value = FakeIterGen()
    result = _itergen_generate_spider(
        value,
        "SQL:",
        {
            "db_info": "# singer ( singer_id , name )",
        },
    )

    assert value.started_with == "SQL:"
    assert value.forward_calls == [
        {"units": ["column_name", "table_name"], "num": 1},
        {"units": ["column_name", "table_name"], "num": 1},
    ]
    assert value.backward_calls == ["column_name"]
    assert result == "SELECT name FROM singer"



def test_spider_itergen_timeout_does_not_restore_session_evidence_after_empty_completion(
    monkeypatch,
    tmp_path,
):
    observed = {}
    written = {}

    class Tokenizer:
        all_special_ids = {2}
        eos_token_id = 2

        def apply_chat_template(self, messages, **kwargs):
            return "<chat>" + messages[0]["content"] + "</chat>"

    class FakeIterGen:
        def __init__(self, **kwargs):
            self.tokenizer = Tokenizer()

    class FakeEvaluator:
        def __init__(self, **kwargs):
            pass

        def _check_syntax_validity(self, scored_output, *, example):
            return scored_output == "SELECT name FROM singer", []

    class FakeLogic:
        def load_dataset_sample(self, evaluator):
            return [{
                "id": "spider-timeout",
                "db_info": "# singer ( singer_id , name )",
                "query": "SELECT name FROM singer",
            }]

        def expected_answer(self, evaluator, example):
            return example["query"]

        def extract_actual(self, evaluator, scored_output, example):
            observed["scored_output"] = scored_output
            if scored_output == "SELECT name FROM singer":
                return (
                    scored_output,
                    "bare_sql",
                    {"syntax_valid": True, "output_contract_valid": True, "output_rejection_reason": None},
                )
            return (
                None,
                "spider_output_contract_rejected",
                {"syntax_valid": False, "output_contract_valid": False, "output_rejection_reason": "prompt_or_wrapper"},
            )

        def is_correct(self, evaluator, actual, expected, example, aux, scored_output):
            return actual == expected

    fake_itergen_package = types.ModuleType("itergen")
    fake_itergen_package.__path__ = []
    fake_itergen_main = types.ModuleType("itergen.main")
    fake_itergen_main.IterGen = FakeIterGen
    monkeypatch.setitem(sys.modules, "itergen", fake_itergen_package)
    monkeypatch.setitem(sys.modules, "itergen.main", fake_itergen_main)

    from synthesis.evaluate import evaluator as evaluator_module
    from synthesis.evaluate.benchmarks import registry as registry_module

    monkeypatch.setattr(evaluator_module, "Evaluator", FakeEvaluator)
    monkeypatch.setattr(registry_module, "get_logic", lambda dataset: FakeLogic())
    monkeypatch.setattr(legacy_runner, "_itergen_add_import_paths", lambda root: None)
    monkeypatch.setattr(legacy_runner, "_install_itergen_transformers_compat", lambda cls: None)
    monkeypatch.setattr(legacy_runner, "_configure_fixed_eval_runtime", lambda *args: None)
    monkeypatch.setattr(legacy_runner, "_legacy_local_cuda_device", lambda device: "cuda:0")
    monkeypatch.setattr(
        legacy_runner,
        "_legacy_benchmark_prompt",
        lambda *args: SpiderPromptParts("RAW SPIDER PROMPT", answer_cue=""),
    )
    monkeypatch.setattr(legacy_runner, "_baseline_row_question", lambda *args: "question")
    monkeypatch.setattr(legacy_runner, "_ITERGEN_PER_EXAMPLE_TIMEOUT_SECONDS", 1)
    monkeypatch.setattr(legacy_runner.Path, "exists", lambda self: True)
    monkeypatch.setattr(
        legacy_runner,
        "_itergen_generate_with_timeout",
        lambda *args, **kwargs: ("", True),
    )
    monkeypatch.setattr(
        legacy_runner,
        "_itergen_generation_token_evidence",
        lambda iter_gen: {
            "raw_token_ids": [10, 11, 2],
            "raw_decoded_text": "SELECT name FROM singer<eos>",
            "removed_terminal_token_ids": [2],
            "decoded_text": "SELECT name FROM singer",
        },
    )
    monkeypatch.setattr(
        legacy_runner,
        "_build_minimal_json",
        lambda rows, *args, **kwargs: written.update(rows=rows),
    )

    args = argparse.Namespace(
        dataset="spider",
        eval_model="Qwen/Qwen2.5-7B-Instruct",
        eval_backend="vllm",
        device="cuda",
        eval_sample_size=1,
        eval_max_steps=64,
        eval_step_token_budget=1,
        vllm_gpu_memory_utilization=0.1,
        vllm_tensor_parallel_size=1,
        gsm_split_file=None,
        gsm_split_name="eval",
        spider_split_file=None,
        spider_split_name="eval",
        smiles_classes=None,
        output_json=str(tmp_path / "baseline.json"),
    )

    assert legacy_runner._run_itergen_legacy_adapter_inner(args) == 0
    assert observed["scored_output"] == ""
    assert written["rows"][0]["llm_response"] == ""
    assert written["rows"][0]["actual"] is None
    assert written["rows"][0]["syntax_valid"] is False
    assert written["rows"][0]["timed_out"] is True


def test_spider_qwen35_itergen_renders_chat_template_without_thinking():
    calls = []

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            calls.append((messages, kwargs))
            return "<|user|>SQL prompt<|assistant|>"

    rendered = SpiderPromptParts("SQL prompt", answer_cue="").render_for_model(
        Tokenizer(),
        model_name="Qwen/Qwen3.5-2B",
    )

    assert rendered == "<|user|>SQL prompt<|assistant|>"
    assert calls == [
        (
            [{"role": "user", "content": "SQL prompt"}],
            {
                "add_generation_prompt": True,
                "tokenize": False,
                "enable_thinking": False,
            },
        )
    ]


@pytest.mark.parametrize(
    "model_name",
    [
        "Qwen/Qwen2.5-1.5B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct",
        "Qwen/Qwen3.5-2B",
        "Qwen/Qwen3.5-4B",
    ],
)
def test_spider_renderer_uses_exact_manifest_model_names(model_name):
    calls = []

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            calls.append((messages, kwargs))
            return "rendered"

    prompt = SpiderPromptParts("raw prompt", answer_cue="")
    rendered = prompt.render_for_model(Tokenizer(), model_name=model_name)
    if "Qwen3.5" in model_name:
        assert rendered == "rendered"
        assert calls[0][1]["enable_thinking"] is False
    else:
        assert rendered == "raw prompt"
        assert calls == []



def test_crane_uses_the_closing_delimiter_as_its_stop_word():
    assert _crane_stop_words("smiles") == [">>"]
    assert _crane_stop_words("gsm_symbolic") == [">>"]
    assert _crane_stop_words("spider") == [">>"]


def test_qwen35_spider_adapter_keeps_raw_prompt_for_scoring_and_evidence(
    monkeypatch,
    tmp_path,
):
    observed = {}
    written = {}

    class Tokenizer:
        all_special_ids = (2, 99)
        eos_token_id = 2

        def apply_chat_template(self, messages, **kwargs):
            observed["template_call"] = (messages, kwargs)
            return "<|user|>RAW SPIDER PROMPT<|assistant|>"

        def decode(self, token_ids, skip_special_tokens=False):
            pieces = {
                10: "RAW SPIDER PROMPT",
                11: "SELECT name FROM singer",
                99: "<|assistant|>",
                2: "<eos>",
            }
            return "".join(pieces[int(token_id)] for token_id in token_ids)

    class Config:
        model_type = "qwen3_5"

    class Model:
        config = Config()

    class SessionTokens:
        def __getitem__(self, key):
            observed["session_slice"] = key
            return [10, 11, 99, 2]

    class FakeIterGen:
        def __init__(self, **kwargs):
            observed["constructor_kwargs"] = kwargs
            self.model = Model()
            self.tokenizer = Tokenizer()
            self.session_tokens = SessionTokens()
            self.start_from = 3
            self.steps = 0

        def start(self, prompt):
            observed["generation_prompt"] = prompt

        def finished(self):
            return self.steps >= 1

        def forward(self, **kwargs):
            self.steps += 1
            return ["SQL: SELECT name FROM singer"]

        def view(self, unit):
            return {
                "column_name": [["name"]],
                "table_name": [["singer"]],
            }[unit]

        def backward(self, unit):
            raise AssertionError(f"unexpected backtrack: {unit}")

    class FakeEvaluator:
        def __init__(self, **kwargs):
            observed["evaluator_kwargs"] = kwargs

        def _check_syntax_validity(self, scored_output, *, example):
            return True, []

    class FakeLogic:
        def load_dataset_sample(self, evaluator):
            return [
                {
                    "id": "spider-1",
                    "db_info": "# singer ( singer_id , name )",
                    "query": "SELECT name FROM singer",
                }
            ]

        def expected_answer(self, evaluator, example):
            return example["query"]

        def extract_actual(self, evaluator, scored_output, example):
            observed["scored_output"] = scored_output
            if scored_output == "SELECT name FROM singer":
                return (
                    scored_output,
                    "bare_sql",
                    {"syntax_valid": True, "output_contract_valid": True, "output_rejection_reason": None},
                )
            return (
                None,
                "spider_output_contract_rejected",
                {"syntax_valid": False, "output_contract_valid": False, "output_rejection_reason": "prompt_or_wrapper"},
            )

        def is_correct(self, evaluator, actual, expected, example, aux, scored_output):
            return actual == expected

    fake_itergen_package = types.ModuleType("itergen")
    fake_itergen_package.__path__ = []
    fake_itergen_main = types.ModuleType("itergen.main")
    fake_itergen_main.IterGen = FakeIterGen
    monkeypatch.setitem(sys.modules, "itergen", fake_itergen_package)
    monkeypatch.setitem(sys.modules, "itergen.main", fake_itergen_main)

    from synthesis.evaluate import evaluator as evaluator_module
    from synthesis.evaluate.benchmarks import registry as registry_module

    monkeypatch.setattr(evaluator_module, "Evaluator", FakeEvaluator)
    monkeypatch.setattr(registry_module, "get_logic", lambda dataset: FakeLogic())
    monkeypatch.setattr(legacy_runner, "_itergen_add_import_paths", lambda root: None)
    monkeypatch.setattr(legacy_runner, "_install_itergen_transformers_compat", lambda cls: None)
    monkeypatch.setattr(legacy_runner, "_configure_fixed_eval_runtime", lambda *args: None)
    monkeypatch.setattr(legacy_runner, "_legacy_local_cuda_device", lambda device: "cuda:0")
    monkeypatch.setattr(
        legacy_runner,
        "_legacy_benchmark_prompt",
        lambda *args: SpiderPromptParts("RAW SPIDER PROMPT", answer_cue=""),
    )
    monkeypatch.setattr(legacy_runner, "_baseline_row_question", lambda *args: "question")
    monkeypatch.setattr(legacy_runner, "_ITERGEN_PER_EXAMPLE_TIMEOUT_SECONDS", 0)
    monkeypatch.setattr(legacy_runner.Path, "exists", lambda self: True)

    def score_completion(prompt, raw_completion):
        raise AssertionError("Spider must score generated text without prefix stripping")

    monkeypatch.setattr(legacy_runner, "completion_for_scoring", score_completion)
    monkeypatch.setattr(
        legacy_runner,
        "_build_minimal_json",
        lambda rows, *args, **kwargs: written.update(rows=rows),
    )

    args = argparse.Namespace(
        dataset="spider",
        eval_model="Qwen/Qwen3.5-2B",
        eval_backend="vllm",
        device="cuda",
        eval_sample_size=1,
        eval_max_steps=64,
        eval_step_token_budget=1,
        vllm_gpu_memory_utilization=0.1,
        vllm_tensor_parallel_size=1,
        gsm_split_file=None,
        gsm_split_name="eval",
        spider_split_file=None,
        spider_split_name="eval",
        smiles_classes=None,
        output_json=str(tmp_path / "baseline.json"),
    )

    assert legacy_runner._run_itergen_legacy_adapter_inner(args) == 0
    assert observed["generation_prompt"] == "<|user|>RAW SPIDER PROMPT<|assistant|>"
    assert observed["scored_output"] == "RAW SPIDER PROMPTSELECT name FROM singer<|assistant|>"
    assert "scoring_prompt" not in observed
    assert written["rows"][0]["prompt_used"] == "RAW SPIDER PROMPT"
    assert written["rows"][0]["llm_response"] == "RAW SPIDER PROMPTSELECT name FROM singer<|assistant|>"
    assert written["rows"][0]["actual"] is None
    assert written["rows"][0]["answer_source"] == "spider_output_contract_rejected"
    assert written["rows"][0]["output_contract_valid"] is False
    evidence = written["rows"][0]["generation_token_evidence"]
    assert observed["session_slice"] == (Ellipsis, slice(3, None))
    assert evidence["raw_token_ids"] == [10, 11, 99, 2]
    assert evidence["removed_terminal_token_ids"] == [2]
    assert evidence["raw_decoded_text"] == "RAW SPIDER PROMPTSELECT name FROM singer<|assistant|><eos>"
    assert evidence["decoded_text"] == "RAW SPIDER PROMPTSELECT name FROM singer<|assistant|>"


@pytest.mark.parametrize("strategy", ["gcd", "itergen"])
def test_smiles_fixed_decoders_honor_the_requested_token_budget(strategy):
    assert _legacy_fixed_max_new_tokens("smiles", 400, strategy=strategy) == 400


def test_crane_smiles_uses_reasoning_before_a_delimited_constrained_span():
    surface = _crane_adaptive_surface("smiles", "start: /[A-Z]+/")

    assert surface == {
        "grammar": 'start: "<<" crane_body ">>"\ncrane_body: /[A-Z]+/',
        "start_symbol": "<<",
        "start_in_grammar": True,
        "end_symbol": ">>",
        "end_in_grammar": True,
        "start_inside_constrained": False,
    }


def test_crane_smiles_samples_with_neutral_reasoning_and_scores_only_inner_span(
    monkeypatch,
    tmp_path,
):
    observed = {}
    written = {}

    class FakeAdaptiveSynCode:
        def __init__(self, **kwargs):
            observed["constructor_kwargs"] = kwargs

        def infer(self, prompt, stop_words=None):
            observed["generation_prompt"] = prompt
            observed["stop_words"] = stop_words
            return ["Reasoning about the class. <<C=CC(=O)OCC>> ignored tail"]

    class FakeEvaluator:
        def __init__(self, **kwargs):
            observed["evaluator_kwargs"] = kwargs

        def _check_syntax_validity(self, scored_output, *, example):
            observed["syntax_input"] = scored_output
            return scored_output == "C=CC(=O)OCC", []

    class FakeLogic:
        def load_dataset_sample(self, evaluator):
            return [
                {
                    "class_name": "acrylates",
                    "grammar_text": "start: /[A-Z0-9=()]+/",
                    "prompt": "Generate a molecule.\nMolecule:",
                }
            ]

        def expected_answer(self, evaluator, example):
            return None

        def extract_actual(self, evaluator, scored_output, example):
            observed["extract_input"] = scored_output
            return scored_output, "completion", {"syntax_valid": True}

        def is_correct(self, evaluator, actual, expected, example, aux, scored_output):
            return actual == "C=CC(=O)OCC"

    fake_syncode_package = types.ModuleType("syncode")
    fake_syncode_package.__path__ = []
    fake_syncode_infer = types.ModuleType("syncode.infer")
    fake_syncode_infer.AdaptiveSynCode = FakeAdaptiveSynCode
    monkeypatch.setitem(sys.modules, "syncode", fake_syncode_package)
    monkeypatch.setitem(sys.modules, "syncode.infer", fake_syncode_infer)

    from synthesis.evaluate import evaluator as evaluator_module
    from synthesis.evaluate.benchmarks import registry as registry_module

    monkeypatch.setattr(evaluator_module, "Evaluator", FakeEvaluator)
    monkeypatch.setattr(registry_module, "get_logic", lambda dataset: FakeLogic())
    monkeypatch.setattr(legacy_runner, "_configure_fixed_eval_runtime", lambda *args: None)
    monkeypatch.setattr(legacy_runner, "_legacy_local_cuda_device", lambda device: "cuda:0")
    monkeypatch.setattr(legacy_runner, "_legacy_benchmark_prompt", lambda *args: "Generate a molecule.\nMolecule:")
    monkeypatch.setattr(legacy_runner, "_baseline_row_question", lambda *args: "question")
    monkeypatch.setattr(legacy_runner, "_build_minimal_json", lambda rows, *args, **kwargs: written.update(rows=rows))

    args = argparse.Namespace(
        dataset="smiles", eval_model="Qwen/Qwen2.5-Coder-7B-Instruct",
        eval_backend="vllm", device="cuda", eval_sample_size=1, eval_max_steps=64,
        eval_step_token_budget=1, vllm_gpu_memory_utilization=0.1,
        vllm_tensor_parallel_size=1, gsm_split_file=None, gsm_split_name="eval",
        spider_split_file=None, spider_split_name="eval", smiles_classes="acrylates",
        output_json=str(tmp_path / "baseline.json"),
    )

    assert legacy_runner._crane_via_adaptive_syncode(args, "smiles") == 0

    constructor = observed["constructor_kwargs"]
    assert constructor["do_sample"] is True
    assert constructor["temperature"] == 0.7
    assert constructor["start_symbol"] == "<<"
    assert constructor["end_symbol"] == ">>"
    assert constructor["start_inside_constrained"] is False
    assert observed["generation_prompt"] == (
        "Generate a molecule.\nMolecule:\n\n"
        "Think through the requested molecular class, then put only the final SMILES "
        "between << and >>."
    )
    assert observed["stop_words"] == [">>"]
    assert observed["extract_input"] == "C=CC(=O)OCC"
    assert observed["syntax_input"] == "C=CC(=O)OCC"
    assert written["rows"][0]["llm_response"] == "C=CC(=O)OCC"


def test_spider_qwen35_prompt_state_is_shared_with_the_csd_entry():
    evaluator = type("PromptEvaluator", (), {"model_name": "Qwen/Qwen3.5-2B"})()
    prompt = sql_eval_logic.format_prompt(evaluator, {
        "db_id": "concert_singer",
        "db_info": "# singer ( singer_id , name )",
        "question": "How many singers do we have?",
    })

    assert callable(getattr(prompt, "render_for_model", None))
    assert prompt.raw_text == str(prompt)



@pytest.mark.parametrize(
    "model_name",
    [
        "Qwen/Qwen2.5-1.5B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct",
        "Qwen/Qwen3.5-2B",
        "Qwen/Qwen3.5-4B",
    ],
)
def test_real_fixed_and_csd_entries_match_rendered_strings_and_token_ids(
    model_name,
    tmp_path,
    monkeypatch,
):
    from synthesis.evaluate.benchmarks.gsm_symbolic.generation import run_crane_csd

    example = {
        "db_id": "concert_singer",
        "db_info": "# singer ( singer_id , name )",
        "question": "How many singers do we have?",
    }

    class Tokenizer:
        eos_token = "<eos>"
        eos_token_id = 2

        def __init__(self):
            self.template_calls = []
            self.encode_calls = []

        def apply_chat_template(self, messages, **kwargs):
            self.template_calls.append((messages, kwargs))
            return "<chat>" + messages[0]["content"] + "</chat>"

        def encode(self, text, add_special_tokens=False):
            self.encode_calls.append((text, add_special_tokens))
            return [ord(char) % 97 for char in text]

    def fixed_entry(tokenizer):
        evaluator = types.SimpleNamespace(model_name=model_name)
        parts = legacy_runner._legacy_benchmark_prompt(
            sql_eval_logic,
            evaluator,
            example,
            "expression_only",
        )

        class FixedIterGen:
            def __init__(self):
                self.tokenizer = tokenizer
                self.steps = 0
                self.tokenized_prompt_ids = None

            def start(self, prompt):
                self.prompt = prompt
                self.tokenized_prompt_ids = self.tokenizer.encode(
                    prompt, add_special_tokens=False
                )

            def finished(self):
                return self.steps >= 1

            def forward(self, **kwargs):
                self.steps += 1
                return ["SELECT name FROM singer"]

            def view(self, unit):
                return {"column_name": [["name"]], "table_name": [["singer"]]}[unit]

            def backward(self, unit):
                raise AssertionError(f"unexpected backtrack: {unit}")

        iter_gen = FixedIterGen()
        rendered = parts.render_for_model(tokenizer, model_name=model_name)
        generated = legacy_runner._itergen_generate_spider(
            iter_gen, rendered, example
        )
        assert generated == "SELECT name FROM singer"
        return rendered, iter_gen.tokenized_prompt_ids

    class FakeSeq(list):
        pass

    class Dafny:
        @staticmethod
        def Seq(value):
            return value

        @staticmethod
        def SeqWithoutIsStrInference(values):
            return FakeSeq(values)

    class LM:
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer
            self.model_name = model_name
            self._last_generation_evidence = None
            self.task_guidance = None

        def ResetTaskGuidance(self):
            pass

        def set_structured_prompt(self, prompt, *, model_name=None):
            pass

        def SetRuntimeDeadline(self, deadline):
            pass

        def ClearRuntimeDeadline(self):
            pass

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
            lm_arg.csd_tokenized_ids = lm_arg.tokenizer.encode(
                lm_arg.instruction_text, add_special_tokens=False
            )
            return FakeSeq([]), False, FakeSeq([]), 0

    class GeneratedCSD:
        default__ = GeneratedDefault

    class Parser:
        def is_complete(self, text):
            return False

    fixed_tokenizer = Tokenizer()
    fixed_string, fixed_ids = fixed_entry(fixed_tokenizer)

    csd_tokenizer = Tokenizer()
    evaluator = types.SimpleNamespace(model_name=model_name)
    csd_parts = sql_eval_logic.format_prompt(evaluator, example)
    lm = LM(csd_tokenizer)
    env = {
        "_dafny": Dafny,
        "GeneratedCSD": GeneratedCSD,
        "lm": lm,
        "parser": Parser(),
        "model_name": model_name,
    }
    run_crane_csd(
        env=env,
        prompt_text=csd_parts,
        max_steps=8,
        grammar_file=tmp_path / "unused.lark",
    )
    csd_string = lm.instruction_text
    csd_ids = lm.csd_tokenized_ids

    assert fixed_string == csd_string
    assert fixed_ids == csd_ids
    if "Qwen3.5" in model_name:
        assert len(fixed_tokenizer.template_calls) == 1
        assert len(csd_tokenizer.template_calls) == 1
        for calls in (fixed_tokenizer.template_calls, csd_tokenizer.template_calls):
            messages, kwargs = calls[0]
            assert messages == [{"role": "user", "content": str(csd_parts)}]
            assert kwargs["add_generation_prompt"] is True
            assert kwargs["enable_thinking"] is False
    else:
        assert fixed_tokenizer.template_calls == []
        assert csd_tokenizer.template_calls == []
