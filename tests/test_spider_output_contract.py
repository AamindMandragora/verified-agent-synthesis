"""CPU integration tests for the Spider token-0 output contract."""

from pathlib import Path

import pytest

from synthesis.evaluate.benchmarks.sql_spider import eval_logic as sql_eval_logic


_GRAMMAR_PATH = Path("synthesis/evaluate/grammars/sql.lark")


def _validate_bare_sql(*args, **kwargs):
    from synthesis.evaluate.benchmarks.sql_spider.output_contract import validate_bare_sql
    return validate_bare_sql(*args, **kwargs)


def _strip_terminal_special_token_ids(*args, **kwargs):
    from synthesis.evaluate.benchmarks.sql_spider.output_contract import strip_terminal_special_token_ids
    return strip_terminal_special_token_ids(*args, **kwargs)


def _example() -> dict[str, str]:
    return {
        "db_id": "concert_singer",
        "db_info": "# singer ( singer_id , name )",
        "question": "How many singers do we have?",
        "query": "SELECT name FROM singer",
    }


class _CachedRealEvaluator:
    """Small evaluator stand-in using the checked-in SQL grammar and one parser."""

    def __init__(self) -> None:
        self._parser = None

    def _get_grammar_text(self) -> str:
        return _GRAMMAR_PATH.read_text()

    def _get_syntax_parser(self, example):
        if self._parser is None:
            self._parser = sql_eval_logic.get_syntax_parser(self, example)
        return self._parser


def _real_parser():
    return _CachedRealEvaluator()._get_syntax_parser(_example())


def test_existing_token0_entry_rejects_sql_label_instead_of_extracting_it():
    actual, source, aux = sql_eval_logic.extract_actual(
        _CachedRealEvaluator(),
        "SQL: SELECT name FROM singer",
        _example(),
    )

    assert actual is None
    assert source == "spider_output_contract_rejected"
    assert aux == {
        "syntax_valid": False,
        "removed_terminal_token_count": 0,
        "output_contract_valid": False,
        "output_rejection_reason": "prompt_or_wrapper",
    }


def test_existing_itergen_adapter_rejects_wrapped_sql_with_coherent_fields():
    actual, source, aux = sql_eval_logic.extract_actual(
        _CachedRealEvaluator(),
        "Here is the query: SELECT name FROM singer",
        _example(),
    )
    row = {
        "actual": actual,
        "answer_source": source,
        "output_contract_valid": aux["output_contract_valid"],
        "output_rejection_reason": aux["output_rejection_reason"],
        "syntax_valid": aux["syntax_valid"],
    }

    assert row == {
        "actual": None,
        "answer_source": "spider_output_contract_rejected",
        "output_contract_valid": False,
        "output_rejection_reason": "prompt_or_wrapper",
        "syntax_valid": False,
    }


def test_bare_sql_and_marker_like_literals_remain_valid():
    parser = _real_parser()
    cases = [
        ("SELECT name FROM singer", "SELECT name FROM singer"),
        ("  SELECT name  FROM singer;  ", "SELECT name  FROM singer"),
        (
            "SELECT 'SQL: <<not a marker>>;  still text' FROM singer",
            "SELECT 'SQL: <<not a marker>>;  still text' FROM singer",
        ),
        (
            "SELECT 'line  \nvalue; marker' FROM singer",
            "SELECT 'line  \nvalue; marker' FROM singer",
        ),
        (
            "SELECT name FROM singer -- SQL: <<comment; marker>>",
            "SELECT name FROM singer -- SQL: <<comment; marker>>",
        ),
    ]

    for output, expected_sql in cases:
        result = _validate_bare_sql(output, parser=parser)
        assert result.accepted, (output, result.rejection_reason)
        assert result.sql == expected_sql
        assert result.rejection_reason is None


def test_multiline_internal_whitespace_is_preserved_without_flattening():
    output = "SELECT 'left  \nright' FROM singer"
    result = _validate_bare_sql(output, parser=_real_parser())

    assert result.accepted is True
    assert result.sql == output


def test_outer_whitespace_multiline_text_is_preserved_by_live_parser():
    output = "  SELECT 'left  \nright' FROM singer  "
    result = _validate_bare_sql(output, parser=_real_parser())

    assert result.accepted is True
    assert result.sql == "SELECT 'left  \nright' FROM singer"


def test_clause_newline_not_supported_by_live_parser_is_rejected():
    result = _validate_bare_sql("SELECT name\nFROM singer", parser=_real_parser())

    assert result.accepted is False
    assert result.rejection_reason == "invalid_or_non_bare_sql"


def test_doubled_quote_and_semicolon_follow_live_parser_support():
    result = _validate_bare_sql(
        "SELECT 'it''s; still' FROM singer",
        parser=_real_parser(),
    )

    assert result.accepted is False
    assert result.rejection_reason == "invalid_or_non_bare_sql"


@pytest.mark.parametrize(
    "output",
    [
        "SELECT \"name\" FROM singer",
        "SELECT name FROM singer /* marker comment */",
    ],
)
def test_unsupported_quoted_identifier_and_block_comment_are_rejected(output):
    result = _validate_bare_sql(output, parser=_real_parser())

    assert result.accepted is False
    assert result.rejection_reason == "invalid_or_non_bare_sql"


@pytest.mark.parametrize(
    ("output", "reason"),
    [
        ("SQL: SELECT name FROM singer", "prompt_or_wrapper"),
        ("```sql\nSELECT name FROM singer\n```", "prompt_or_wrapper"),
        ("<<SELECT name FROM singer>>", "prompt_or_wrapper"),
        ("<think>plan</think> SELECT name FROM singer", "prompt_or_wrapper"),
        ("Here is the query: SELECT name FROM singer", "prompt_or_wrapper"),
        ("SELECT name FROM singer\nThis is the query.", "prompt_or_wrapper"),
        ("<|assistant|>SELECT name FROM singer", "prompt_or_wrapper"),
        (
            "db_id: concert_singer\ndb_info: singer\nquestion: q\nSQL: SELECT name FROM singer",
            "prompt_or_wrapper",
        ),
    ],
)
def test_bare_sql_validator_rejects_outer_prompt_and_prose(output, reason):
    result = _validate_bare_sql(output, parser=_real_parser())

    assert result.accepted is False
    assert result.sql is None
    assert result.rejection_reason == reason


def test_rejected_first_paragraph_is_not_rescued_by_later_sql():
    result = _validate_bare_sql(
        "Here is the query:\n\nSELECT name FROM singer",
        parser=_real_parser(),
    )

    assert result.accepted is False
    assert result.rejection_reason == "prompt_or_wrapper"


def test_valid_sql_followed_by_prose_is_not_truncated_to_first_paragraph():
    result = _validate_bare_sql(
        "SELECT name FROM singer\nExplanation: this is the result",
        parser=_real_parser(),
    )

    assert result.accepted is False
    assert result.rejection_reason == "prompt_or_wrapper"


def test_bare_sql_validator_rejects_multiple_and_trailing_statements():
    for output in (
        "SELECT name FROM singer; SELECT singer_id FROM singer",
        "SELECT name FROM singer;;",
        "SELECT name FROM singer; -- trailing comment",
    ):
        result = _validate_bare_sql(output, parser=_real_parser())
        assert result.accepted is False
        assert result.rejection_reason == "multiple_statements"


def test_malformed_sql_has_the_stable_invalid_reason():
    result = _validate_bare_sql("SELECT name", parser=_real_parser())

    assert result.accepted is False
    assert result.rejection_reason == "invalid_or_non_bare_sql"


def test_empty_output_has_the_stable_empty_reason():
    result = _validate_bare_sql(" \n\t ", parser=_real_parser())

    assert result.accepted is False
    assert result.rejection_reason == "empty"


class _Tokenizer:
    all_special_ids = {0, 2, 99}
    eos_token_id = 2


def test_terminal_token_removal_uses_only_exact_generation_stop_ids():
    assert _strip_terminal_special_token_ids(
        [10, 11, 99, 2], _Tokenizer(), terminal_stop_token_ids={2}
    ) == [10, 11, 99]
    assert _strip_terminal_special_token_ids(
        [10, 11, 3], _Tokenizer(), terminal_stop_token_ids={2}
    ) == [10, 11, 3]
    assert _strip_terminal_special_token_ids(
        [10, 0, 11], _Tokenizer(), terminal_stop_token_ids={2}
    ) == [10, 0, 11]


def test_real_unconstrained_decode_boundary_preserves_token_evidence():
    from synthesis.evaluate.benchmarks.common.model_utils import _TensorizedLMBase

    class BoundaryTokenizer:
        all_special_ids = {2, 99}
        eos_token_id = 2

        def decode(self, token_ids, skip_special_tokens=False):
            pieces = {10: "SELECT ", 11: "1", 99: "<special>", 2: "<eos>"}
            return "".join(pieces[int(token_id)] for token_id in token_ids)

        def encode(self, text, add_special_tokens=False):
            return []

    class Dafny:
        @staticmethod
        def Seq(text):
            return text

        @staticmethod
        def SeqWithoutIsStrInference(values):
            return values

    from synthesis.evaluate.benchmarks.sql_spider.prompts import SpiderPromptParts

    lm = object.__new__(_TensorizedLMBase)
    lm.tokenizer = BoundaryTokenizer()
    lm._dafny = Dafny()
    lm._token_id_to_str = {}
    lm._structured_prompt = SpiderPromptParts(
        "task\n", model_name="Qwen/Qwen2.5-7B-Instruct"
    )
    result = lm._build_unconstrained_chunk_result(
        [10, 11, 99, 2], "<<", "<eos>", 10
    )

    assert result[0] == ["SELECT ", "1", "<special>"]
    assert result[1:] == (False, True, 3)
    assert lm._last_generation_evidence == {
        "raw_token_ids": [10, 11, 99, 2],
        "raw_decoded_text": "SELECT 1<special><eos>",
        "removed_terminal_token_ids": [2],
        "decoded_text": "SELECT 1<special>",
    }


def test_non_spider_unconstrained_decode_preserves_eos_and_has_no_spider_evidence():
    from synthesis.evaluate.benchmarks.common.model_utils import _TensorizedLMBase

    class Tokenizer:
        all_special_ids = {2, 99}

        def decode(self, token_ids, skip_special_tokens=False):
            pieces = {10: "SELECT ", 2: "<eos>", 99: "<special>"}
            return "".join(pieces[int(token_id)] for token_id in token_ids)

        def encode(self, text, add_special_tokens=False):
            return []

    class Dafny:
        @staticmethod
        def Seq(text):
            return text

        @staticmethod
        def SeqWithoutIsStrInference(values):
            return values

    lm = object.__new__(_TensorizedLMBase)
    lm.tokenizer = Tokenizer()
    lm._dafny = Dafny()
    lm._token_id_to_str = {}
    lm._structured_prompt = None
    result = lm._build_unconstrained_chunk_result([10, 2], "<<", "<eos>", 10)

    assert result[0] == ["SELECT "]
    assert result[1:] == (False, True, 2)
    assert lm._last_generation_evidence is None


def test_itergen_generation_boundary_text_is_the_scored_text():
    from synthesis.evaluate.run_legacy_fixed_strategy import _itergen_generation_token_evidence

    class Tokenizer:
        all_special_ids = {2, 99}
        eos_token_id = 2

        def decode(self, token_ids, skip_special_tokens=False):
            pieces = {
                10: "SQL: ",
                11: "SELECT name FROM singer",
                99: "<|assistant|>",
                2: "<eos>",
            }
            return "".join(pieces[int(token_id)] for token_id in token_ids)

    class Session:
        def __getitem__(self, key):
            assert key == (Ellipsis, slice(3, None))
            return [10, 11, 99, 2]

    class IterGen:
        session_tokens = Session()
        start_from = 3
        tokenizer = Tokenizer()

    evidence = _itergen_generation_token_evidence(IterGen())
    actual, source, aux = sql_eval_logic.extract_actual(
        _CachedRealEvaluator(), evidence["decoded_text"], _example()
    )

    assert evidence["decoded_text"] == "SQL: SELECT name FROM singer<|assistant|>"
    assert actual is None
    assert source == "spider_output_contract_rejected"
    assert aux["output_rejection_reason"] == "prompt_or_wrapper"


def test_legacy_visible_span_opt_out_is_unchanged(monkeypatch):
    monkeypatch.setenv("SPIDER_TOKEN0_CONSTRAINED", "0")
    actual, source, aux = sql_eval_logic.extract_actual(
        _CachedRealEvaluator(),
        "<<SELECT name FROM singer>>",
        _example(),
    )

    assert actual == "SELECT name FROM singer"
    assert source == "last_visible_span"
    assert aux is None


def _evaluate_one_sample(
    monkeypatch,
    output: str,
    evidence: dict | None = None,
    max_seconds_per_example: float | None = None,
    prediction_matches_gold=None,
    fast_parser: bool = False,
) -> dict:
    from synthesis.evaluate.benchmarks.sql_spider import executor
    from synthesis.evaluate.evaluator import Evaluator

    evaluator = Evaluator(
        dataset_name="spider",
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
        backend="huggingface",
        device="cpu",
        sample_size=1,
        max_steps=8,
        max_seconds_per_example=max_seconds_per_example,
    )
    evaluator._base_grammar_text = _GRAMMAR_PATH.read_text()
    if fast_parser:
        class FastParser:
            def parse(self, text):
                return object()
        evaluator._get_syntax_parser = lambda example: FastParser()
    example = _example()

    class LM:
        _last_generation_evidence = evidence
        task_guidance = None

    monkeypatch.setattr(
        executor,
        "prediction_matches_gold",
        prediction_matches_gold or (lambda actual, row: actual == row.get("query")),
    )

    def fake_run(**kwargs):
        return output, 4, 0.01, [], []

    return evaluator._evaluate_one_example(
        0,
        example,
        1,
        {"lm": LM(), "tokenizer": None},
        sql_eval_logic,
        fake_run,
        {},
    )


def test_rejected_evaluator_sample_fields_are_coherent(monkeypatch):
    sample = _evaluate_one_sample(monkeypatch, "SQL: SELECT name FROM singer")

    assert sample["actual"] is None
    assert sample["answer_source"] == "spider_output_contract_rejected"
    assert sample["has_extracted_answer"] is False
    assert sample["is_syntax_valid"] is False
    assert sample["is_correct"] is False
    assert sample["accuracy_applicable"] is True
    assert sample["output_contract_valid"] is False
    assert sample["output_rejection_reason"] == "prompt_or_wrapper"


def test_evaluator_sample_carries_removed_terminal_token_count(monkeypatch):
    sample = _evaluate_one_sample(
        monkeypatch,
        "SELECT name FROM singer;",
        evidence={"removed_terminal_token_ids": [99, 2]},
    )

    assert sample["removed_terminal_token_count"] == 2


def test_accepted_evaluator_sample_fields_are_coherent(monkeypatch):
    sample = _evaluate_one_sample(monkeypatch, "SELECT name FROM singer;")

    assert sample["actual"] == "SELECT name FROM singer"
    assert sample["answer_source"] == "bare_sql"
    assert sample["has_extracted_answer"] is True
    assert sample["is_syntax_valid"] is True
    assert sample["is_correct"] is True
    assert sample["accuracy_applicable"] is True
    assert sample["output_contract_valid"] is True
    assert sample["output_rejection_reason"] is None



def test_evaluator_does_not_strip_raw_spider_prompt_echo(monkeypatch):
    from synthesis.evaluate.evaluator import Evaluator

    evaluator = Evaluator(
        dataset_name="spider",
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
        backend="huggingface",
        device="cpu",
        sample_size=1,
        max_steps=8,
    )
    evaluator._base_grammar_text = _GRAMMAR_PATH.read_text()
    example = _example()

    class LM:
        _last_generation_evidence = None
        task_guidance = None

    def fake_run(**kwargs):
        prompt = str(kwargs["prompt_text"])
        return prompt + "SELECT name FROM singer", 4, 0.01, [], []

    sample = evaluator._evaluate_one_example(
        0,
        example,
        1,
        {"lm": LM(), "tokenizer": None},
        sql_eval_logic,
        fake_run,
        {},
    )

    assert sample["actual"] is None
    assert sample["output_rejection_reason"] == "prompt_or_wrapper"
    assert sample["has_extracted_answer"] is False


def test_spider_prompt_renderer_failure_propagates_as_harness_error(monkeypatch):
    from synthesis.evaluate.evaluator import Evaluator

    try:
        from synthesis.evaluate.benchmarks.sql_spider.prompts import SpiderPromptRenderError
    except ImportError:
        class SpiderPromptRenderError(RuntimeError):
            pass

    evaluator = Evaluator(
        dataset_name="spider",
        model_name="Qwen/Qwen3.5-2B",
        backend="huggingface",
        device="cpu",
        sample_size=1,
        max_steps=8,
    )
    evaluator._base_grammar_text = _GRAMMAR_PATH.read_text()

    class LM:
        _last_generation_evidence = None
        task_guidance = None

    def fake_run(**kwargs):
        raise SpiderPromptRenderError("Spider chat template rendering failed")

    with pytest.raises(SpiderPromptRenderError, match="chat template rendering failed"):
        evaluator._evaluate_one_example(
            0,
            _example(),
            1,
            {"lm": LM(), "tokenizer": None},
            sql_eval_logic,
            fake_run,
            {},
        )


def test_generation_boundary_removes_only_actual_stop_ids():
    from synthesis.evaluate.benchmarks.common.model_utils import _TensorizedLMBase

    class Tokenizer:
        all_special_ids = (2, 99)

        def decode(self, token_ids, skip_special_tokens=False):
            pieces = {
                10: "SELECT ",
                11: "name FROM singer",
                99: "<|assistant|>",
                2: "<eos>",
            }
            return "".join(pieces[int(token_id)] for token_id in token_ids)

    lm = object.__new__(_TensorizedLMBase)
    lm.tokenizer = Tokenizer()
    lm._generation_stop_token_ids = {2}

    retained = lm._prepare_generated_token_ids([10, 11, 99, 2])

    assert retained == [10, 11, 99]
    assert lm._last_generation_evidence["removed_terminal_token_ids"] == [2]
    assert lm._last_generation_evidence["decoded_text"].endswith("<|assistant|>")
    result = _validate_bare_sql(lm._last_generation_evidence["decoded_text"], parser=_real_parser())
    assert result.accepted is False
    assert result.rejection_reason == "prompt_or_wrapper"


def _run_fake_spider_csd_with_ids(tmp_path, token_ids, token_texts):
    from synthesis.evaluate.benchmarks.gsm_symbolic.generation import run_crane_csd
    from synthesis.evaluate.benchmarks.sql_spider.prompts import SpiderPromptParts

    class FakeSeq(list):
        pass

    class Dafny:
        @staticmethod
        def Seq(value):
            return value

        @staticmethod
        def SeqWithoutIsStrInference(values):
            return FakeSeq(values)

    class Tokenizer:
        eos_token = "<eos>"
        eos_token_id = 2
        all_special_ids = (2,)

        def apply_chat_template(self, messages, **kwargs):
            return messages[-1]["content"]

        def decode(self, values, skip_special_tokens=False):
            return "".join(token_texts[int(value)] for value in values)

    class LM:
        def __init__(self):
            self.tokenizer = Tokenizer()
            self.model_name = "Qwen/Qwen2.5-1.5B-Instruct"
            self._last_generation_evidence = None
            self._generation_token_ids = []
            self.task_guidance = None

        def ResetTaskGuidance(self):
            pass

        def set_structured_prompt(self, prompt, *, model_name=None):
            self.structured_prompt = prompt

        def SetRuntimeDeadline(self, deadline):
            pass

        def ClearRuntimeDeadline(self):
            pass

    lm = LM()

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
            lm_arg._generation_token_ids = list(token_ids)
            return (
                FakeSeq([token_texts[int(value)] for value in token_ids if int(value) != 2]),
                False,
                FakeSeq([]),
                len(token_ids),
            )

    class GeneratedCSD:
        default__ = GeneratedDefault

    class Parser:
        def is_complete(self, text):
            return True

    env = {"_dafny": Dafny, "GeneratedCSD": GeneratedCSD, "lm": lm, "parser": Parser()}
    result = run_crane_csd(
        env=env,
        prompt_text=SpiderPromptParts("db_id: x\nquestion: q\n", model_name=lm.model_name),
        max_steps=32,
        grammar_file=tmp_path / "unused.lark",
        start_inside_constrained=True,
    )
    return lm, result


def test_spider_constrained_only_csd_records_full_generation_evidence(tmp_path):
    lm, result = _run_fake_spider_csd_with_ids(
        tmp_path,
        [10, 11, 2],
        {10: "SELECT ", 11: "name FROM singer", 2: "<eos>"},
    )

    assert result[0] == "SELECT name FROM singer"
    assert lm._last_generation_evidence["raw_token_ids"] == [10, 11, 2]
    assert lm._last_generation_evidence["raw_decoded_text"] == "SELECT name FROM singer<eos>"
    assert lm._last_generation_evidence["removed_terminal_token_ids"] == [2]
    assert lm._last_generation_evidence["decoded_text"] == "SELECT name FROM singer"


def test_spider_multi_chunk_csd_evidence_keeps_ordered_full_span(tmp_path):
    lm, _ = _run_fake_spider_csd_with_ids(
        tmp_path,
        [10, 11, 12, 2],
        {
            10: "SELECT ",
            11: "name ",
            12: "FROM singer",
            2: "<eos>",
        },
    )

    assert lm._last_generation_evidence["raw_token_ids"] == [10, 11, 12, 2]
    assert lm._last_generation_evidence["raw_decoded_text"] == "SELECT name FROM singer<eos>"
    assert lm._last_generation_evidence["decoded_text"] == "SELECT name FROM singer"


def test_spider_execution_comparison_is_inside_example_timer(monkeypatch):
    from synthesis.evaluate.benchmarks.sql_spider import executor

    def slow_executor(actual, row):
        import time
        time.sleep(0.30)
        return True

    monkeypatch.setattr(executor, "prediction_matches_gold", slow_executor)
    sample = _evaluate_one_sample(
        monkeypatch,
        "SELECT name FROM singer",
        max_seconds_per_example=0.10,
        prediction_matches_gold=slow_executor,
        fast_parser=True,
    )

    assert sample["timed_out"] is True
    assert sample["runtime_budget_exceeded"] is True
    assert sample["is_correct"] is False
