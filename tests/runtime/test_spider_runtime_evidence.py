from types import SimpleNamespace

import pytest


@pytest.mark.parametrize(
    ("model_name", "expected_family", "expected_mode"),
    [
        ("Qwen/Qwen2.5-7B-Instruct", "qwen2.5", "raw"),
        ("Qwen/Qwen3.5-4B", "qwen3.5", "chat"),
    ],
)
def test_run_crane_csd_records_actual_spider_prompt_render_branch(
    tmp_path, model_name, expected_family, expected_mode
):
    from synthesis.evaluate.benchmarks.common.model_utils import _TensorizedLMBase
    from synthesis.evaluate.benchmarks.gsm_symbolic.generation import run_crane_csd
    from synthesis.evaluate.benchmarks.sql_spider.prompts import SpiderPromptParts

    class Tokenizer:
        eos_token = "<eos>"
        eos_token_id = 99
        all_special_ids = {99}

        def __init__(self):
            self.template_calls = []

        def apply_chat_template(self, messages, **kwargs):
            self.template_calls.append(([dict(message) for message in messages], dict(kwargs)))
            return "<chat-rendered>"

        def decode(self, token_ids, skip_special_tokens=False):
            del skip_special_tokens
            return "".join("<eos>" if int(token_id) == 99 else "ok" for token_id in token_ids)

        def encode(self, text, add_special_tokens=False):
            del text, add_special_tokens
            return []

    class Dafny:
        @staticmethod
        def Seq(value):
            return value

        @staticmethod
        def SeqWithoutIsStrInference(values):
            return list(values)

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
            del lm_arg, parser, seq0, generated_prefix, start_inside
            del current_constrained, max_steps, step_budget, eos_token
            return (["ok"], False, [], 1)

    class GeneratedCSD:
        default__ = GeneratedDefault

    tokenizer = Tokenizer()
    lm = _TensorizedLMBase(Dafny(), tokenizer, ["ok"], [1])
    env = {
        "_dafny": Dafny,
        "GeneratedCSD": GeneratedCSD,
        "lm": lm,
        "parser": Parser(),
        "model_name": model_name,
    }

    run_crane_csd(
        env=env,
        prompt_text=SpiderPromptParts("db_id: x\nquestion: q\n", answer_cue="SQL:"),
        max_steps=8,
        grammar_file=tmp_path / "unused.lark",
    )

    contract = lm._last_prompt_contract
    assert contract["renderer"] == "spider"
    assert contract["family"] == expected_family
    assert contract["mode"] == expected_mode
    assert contract["render_succeeded"] is True
    assert contract["prompt_chars"] == len(lm.instruction_text)
    if expected_mode == "raw":
        assert contract["template_used"] is False
        assert contract["raw_prompt"] is True
        assert contract["chat_message_count"] == 0
        assert tokenizer.template_calls == []
    else:
        assert contract["template_used"] is True
        assert contract["raw_prompt"] is False
        assert contract["chat_message_count"] == 1
        assert contract["user_message_count"] == 1
        assert contract["add_generation_prompt"] is True
        assert contract["enable_thinking"] is False
        assert tokenizer.template_calls[0][0] == [
            {"role": "user", "content": "db_id: x\nquestion: q\nSQL:"}
        ]


@pytest.mark.parametrize(
    "indices",
    [
        [3],
        [3, 8, 9, 15, 20, 21, 23, 30, 35, 40],
    ],
)
def test_reevaluation_provenance_records_result_indices_in_order(tmp_path, indices):
    from synthesis.evaluate.evaluator import EvaluationResult
    from synthesis.scripts.reevaluate_compiled_csd import build_reevaluation_provenance

    compiled = tmp_path / "GeneratedCSD.py"
    compiled.write_text("# frozen strategy\n", encoding="utf-8")
    result = EvaluationResult(
        success=True,
        accuracy=0.0,
        contains_delimiters=False,
        syntax_rate=0.0,
        num_examples=len(indices),
        num_correct=0,
        total_time_seconds=0.1,
        sample_outputs=[
            {
                "example_index": position,
                "source_index": source_index,
                "spider_source_index": source_index,
            }
            for position, source_index in enumerate(indices)
        ],
    )
    args = type(
        "Args",
        (),
        {
            "dataset": "spider",
            "eval_model": "Qwen/Qwen2.5-7B-Instruct",
            "sample_size": len(indices),
            "sample_offset": 0,
            "max_steps": 400,
            "step_token_budget": 1,
            "smiles_classes": None,
            "spider_split_name": "train",
            "spider_split_file": "/splits/spider.json",
            "provenance_cell_id": "spider-qwen25-7b",
            "provenance_manifest_commit": "a" * 40,
            "claimed_evaluated_source_indices": [999],
        },
    )()

    provenance = build_reevaluation_provenance(
        args,
        compiled,
        evaluation_result=result,
    )

    assert provenance["sample_offset"] == 0
    assert provenance["spider_split_name"] == "train"
    assert provenance["evaluated_source_indices"] == indices




def test_evaluator_sample_carries_runtime_prompt_contract(monkeypatch):
    from pathlib import Path

    from synthesis.evaluate.benchmarks.sql_spider import executor
    from synthesis.evaluate.benchmarks.sql_spider import eval_logic
    from synthesis.evaluate.evaluator import Evaluator

    evaluator = Evaluator(
        dataset_name="spider",
        model_name="Qwen/Qwen2.5-7B-Instruct",
        backend="huggingface",
        device="cpu",
        sample_size=1,
        max_steps=8,
    )
    evaluator._base_grammar_text = (
        Path("synthesis/evaluate/grammars/sql.lark").read_text(encoding="utf-8")
    )
    monkeypatch.setattr(
        executor,
        "prediction_matches_gold",
        lambda actual, row: True,
    )

    class LM:
        _last_generation_evidence = None
        _last_prompt_contract = {
            "renderer": "spider",
            "family": "qwen2.5",
            "mode": "raw",
            "template_used": False,
            "raw_prompt": True,
            "render_succeeded": True,
            "prompt_chars": 12,
        }
        task_guidance = None

    def fake_run(**kwargs):
        return "SELECT name FROM singer", 4, 0.01, [], []

    sample = evaluator._evaluate_one_example(
        0,
        {
            "db_id": "concert_singer",
            "db_info": "# singer ( singer_id , name )",
            "question": "How many singers?",
            "query": "SELECT name FROM singer",
        },
        1,
        {"lm": LM(), "tokenizer": None},
        eval_logic,
        fake_run,
        {},
    )

    assert sample["prompt_contract"]["family"] == "qwen2.5"
    assert sample["prompt_contract"]["raw_prompt"] is True



def test_evaluator_sequential_rows_keep_resolved_spider_source_indices(monkeypatch):
    from pathlib import Path

    from synthesis.evaluate.benchmarks.sql_spider import executor
    from synthesis.evaluate.benchmarks.sql_spider import eval_logic
    from synthesis.evaluate.evaluator import Evaluator

    evaluator = Evaluator(
        dataset_name="spider",
        model_name="Qwen/Qwen2.5-7B-Instruct",
        backend="huggingface",
        device="cpu",
        sample_size=1,
        max_steps=8,
    )
    evaluator._base_grammar_text = Path(
        "synthesis/evaluate/grammars/sql.lark"
    ).read_text(encoding="utf-8")
    monkeypatch.setattr(
        executor,
        "prediction_matches_gold",
        lambda actual, row: True,
    )
    monkeypatch.setattr(
        eval_logic,
        "get_generation_runner",
        lambda: (lambda **kwargs: ("SELECT name FROM singer", 4, 0.01, [], [])),
    )

    class LM:
        _last_generation_evidence = None
        _last_prompt_contract = None
        task_guidance = None

    dataset = [
        {
            "spider_source_index": 3,
            "db_id": "concert_singer",
            "db_info": "# singer ( singer_id , name )",
            "question": "How many singers?",
            "query": "SELECT name FROM singer",
        }
    ]
    samples, reason = evaluator._evaluate_examples_sequential_with_early_stop(
        dataset,
        {"lm": LM(), "tokenizer": None},
        eval_logic,
        None,
        None,
        None,
    )

    assert reason is None
    assert samples[0]["example_index"] == 0
    assert samples[0]["spider_source_index"] == 3
    assert samples[0]["source_index"] == 3


def _source_alias_provenance_args(dataset, split_file):
    return SimpleNamespace(
        dataset=dataset,
        eval_model="Qwen/Qwen2.5-7B-Instruct",
        sample_size=1,
        sample_offset=0,
        max_steps=8,
        step_token_budget=1,
        smiles_classes=None,
        spider_split_name="test",
        spider_split_file=str(split_file),
        gsm_split_name="test",
        gsm_split_file=str(split_file),
        provenance_cell_id=f"{dataset}-test",
        provenance_manifest_commit="a" * 40,
    )


@pytest.mark.parametrize(
    ("dataset", "dataset_alias"),
    [("spider", "spider_source_index"), ("gsm_symbolic", "crane_source_index")],
)
def test_sequential_provenance_rejects_conflicting_source_aliases(
    tmp_path, dataset, dataset_alias
):
    from synthesis.scripts.reevaluate_compiled_csd import build_reevaluation_provenance

    compiled = tmp_path / "GeneratedCSD.py"
    compiled.write_text("# frozen\n")
    result = SimpleNamespace(
        sample_outputs=[{"source_index": 3, dataset_alias: 4}],
    )

    with pytest.raises(ValueError, match="source index aliases disagree"):
        build_reevaluation_provenance(
            _source_alias_provenance_args(dataset, tmp_path / "split.json"),
            compiled,
            evaluation_result=result,
        )


@pytest.mark.parametrize(
    ("dataset", "bad_key", "bad_value"),
    [
        ("spider", "source_index", "3"),
        ("spider", "spider_source_index", 3.0),
        ("spider", "spider_source_index", True),
        ("gsm_symbolic", "source_index", 3.0),
        ("gsm_symbolic", "crane_source_index", "3"),
        ("gsm_symbolic", "crane_source_index", True),
    ],
)
def test_sequential_provenance_rejects_non_integer_source_aliases(
    tmp_path, dataset, bad_key, bad_value
):
    from synthesis.scripts.reevaluate_compiled_csd import build_reevaluation_provenance

    compiled = tmp_path / "GeneratedCSD.py"
    compiled.write_text("# frozen\n")
    dataset_alias = "spider_source_index" if dataset == "spider" else "crane_source_index"
    row = {"source_index": 3, dataset_alias: 3}
    row[bad_key] = bad_value
    result = SimpleNamespace(sample_outputs=[row])

    with pytest.raises(ValueError, match="source alias|integer"):
        build_reevaluation_provenance(
            _source_alias_provenance_args(dataset, tmp_path / "split.json"),
            compiled,
            evaluation_result=result,
        )


@pytest.mark.parametrize(
    ("dataset", "dataset_alias"),
    [("spider", "spider_source_index"), ("gsm_symbolic", "crane_source_index")],
)
def test_sequential_provenance_accepts_equal_aliases_and_optional_none(
    tmp_path, dataset, dataset_alias
):
    from synthesis.scripts.reevaluate_compiled_csd import build_reevaluation_provenance

    compiled = tmp_path / "GeneratedCSD.py"
    compiled.write_text("# frozen\n")
    equal_result = SimpleNamespace(
        sample_outputs=[{"source_index": 3, dataset_alias: 3}],
    )
    equal_provenance = build_reevaluation_provenance(
        _source_alias_provenance_args(dataset, tmp_path / "split.json"),
        compiled,
        evaluation_result=equal_result,
    )
    assert equal_provenance["evaluated_source_indices"] == [3]

    optional_none_result = SimpleNamespace(
        sample_outputs=[{"source_index": 3, dataset_alias: None}],
    )
    optional_none_provenance = build_reevaluation_provenance(
        _source_alias_provenance_args(dataset, tmp_path / "split.json"),
        compiled,
        evaluation_result=optional_none_result,
    )
    assert optional_none_provenance["evaluated_source_indices"] == [3]
