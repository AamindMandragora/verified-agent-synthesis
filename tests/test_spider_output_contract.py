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


def test_clause_newline_bare_sql_is_accepted_and_preserved():
    output = "SELECT name\nFROM singer"
    result = _validate_bare_sql(output, parser=_real_parser())

    assert result.accepted is True
    assert result.sql == output
    assert result.rejection_reason is None

    actual, source, aux = sql_eval_logic.extract_actual(
        _CachedRealEvaluator(), output, _example()
    )
    assert actual == output
    assert source == "bare_sql"
    assert aux["output_contract_valid"] is True
    assert aux["output_rejection_reason"] is None


def test_line_comment_newline_bare_sql_is_accepted_and_preserved():
    output = "SELECT name -- selected column\r\nFROM singer"
    result = _validate_bare_sql(output, parser=_real_parser())

    assert result.accepted is True
    assert result.sql == output
    assert result.rejection_reason is None

    actual, source, aux = sql_eval_logic.extract_actual(
        _CachedRealEvaluator(), output, _example()
    )
    assert actual == output
    assert source == "bare_sql"
    assert aux["output_contract_valid"] is True
    assert aux["output_rejection_reason"] is None


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


def test_spider_only_declared_stop_id_terminates_when_eos_text_is_duplicated():
    """A non-stop ID that decodes like EOS remains generated Spider content."""
    from synthesis.evaluate.benchmarks.common.model_utils import _TensorizedLMBase
    from synthesis.evaluate.benchmarks.sql_spider.prompts import SpiderPromptParts

    class Tokenizer:
        all_special_ids = {2, 99}
        eos_token_id = 2

        def decode(self, token_ids, skip_special_tokens=False):
            pieces = {
                10: "SELECT ",
                99: "<eos>",
                11: "name FROM singer",
                2: "<eos>",
            }
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
    lm._structured_prompt = SpiderPromptParts(
        "db_id: x\\nquestion: q\\n", model_name="Qwen/Qwen2.5-1.5B-Instruct"
    )
    lm._generation_stop_token_ids = {2}

    result = lm._build_unconstrained_chunk_result(
        [10, 99, 11, 2], "<<", "<eos>", 10
    )

    assert result[0] == ["SELECT ", "<eos>", "name FROM singer"]
    assert result[1:] == (False, True, 3)
    assert lm._last_generation_evidence["raw_token_ids"] == [10, 99, 11, 2]
    assert lm._last_generation_evidence["removed_terminal_token_ids"] == [2]
    assert lm._last_generation_evidence["decoded_text"] == (
        "SELECT <eos>name FROM singer"
    )


    contract = _validate_bare_sql(
        "SELECT <eos>name FROM singer", parser=_real_parser()
    )
    assert contract.accepted is False
    assert contract.rejection_reason == "invalid_or_non_bare_sql"

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


def _guidance_test_lm(tokenizer):
    from synthesis.evaluate.benchmarks.common.model_utils import _TensorizedLMBase

    class Dafny:
        @staticmethod
        def Seq(value):
            return value

        @staticmethod
        def SeqWithoutIsStrInference(values):
            return list(values)

    return _TensorizedLMBase(Dafny(), tokenizer, ["token"], [1])


def _run_guidance_failure_at_evaluator_boundary(lm):
    from synthesis.evaluate.evaluator import Evaluator

    evaluator = Evaluator(
        dataset_name="spider",
        model_name="Qwen/Qwen3.5-2B",
        backend="huggingface",
        device="cpu",
        sample_size=1,
        max_steps=8,
    )
    evaluator._base_grammar_text = _GRAMMAR_PATH.read_text()

    def fake_run(**kwargs):
        lm.AppendTaskGuidance("Use only names from the schema.")
        return "SELECT name FROM singer", 4, 0.01, [], []

    return evaluator._evaluate_one_example(
        0,
        _example(),
        1,
        {"lm": lm, "tokenizer": lm.tokenizer},
        sql_eval_logic,
        fake_run,
        {},
    )


def test_real_append_guidance_render_failure_propagates_typed_spider_error():
    from synthesis.evaluate.benchmarks.sql_spider.prompts import (
        SpiderPromptParts,
        SpiderPromptRenderError,
    )

    class Tokenizer:
        all_special_ids = {2}
        eos_token_id = 2
        eos_token = "<eos>"

        def decode(self, token_ids, skip_special_tokens=False):
            return "token"

        def encode(self, text, add_special_tokens=False):
            return [1]

        def apply_chat_template(self, messages, **kwargs):
            raise ValueError("second prompt render failed")

    lm = _guidance_test_lm(Tokenizer())
    prompt = SpiderPromptParts("task", answer_cue="", model_name="Qwen/Qwen3.5-2B")
    lm.set_structured_prompt(prompt, model_name="Qwen/Qwen3.5-2B")

    with pytest.raises(SpiderPromptRenderError, match="Qwen3.5"):
        _run_guidance_failure_at_evaluator_boundary(lm)


def test_real_append_guidance_missing_state_propagates_typed_spider_error():
    from synthesis.evaluate.benchmarks.sql_spider.prompts import SpiderPromptRenderError

    class Tokenizer:
        all_special_ids = {2}
        eos_token_id = 2
        eos_token = "<eos>"

        def decode(self, token_ids, skip_special_tokens=False):
            return "token"

        def encode(self, text, add_special_tokens=False):
            return [1]

    lm = _guidance_test_lm(Tokenizer())

    with pytest.raises(SpiderPromptRenderError, match="registered"):
        _run_guidance_failure_at_evaluator_boundary(lm)


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


def _recording_spider_lm(token_ids, token_texts):
    from synthesis.evaluate.benchmarks.common.model_utils import _TensorizedLMBase
    from synthesis.evaluate.benchmarks.sql_spider.prompts import SpiderPromptParts

    class Dafny:
        @staticmethod
        def Seq(value):
            return value

        @staticmethod
        def SeqWithoutIsStrInference(values):
            return list(values)

    class Tokenizer:
        eos_token = "<eos>"
        eos_token_id = 2
        all_special_ids = {2}

        def decode(self, values, skip_special_tokens=False):
            return "".join(token_texts[int(value)] for value in values)

        def encode(self, text, add_special_tokens=False):
            exact = {
                "SELECT ": [10],
                "SELECT name ": [10, 11],
                "SELECT name FROM singer": [10, 11, 12],
            }
            if text in exact:
                return exact[text]
            return [token_id for token_id in token_ids if token_texts[token_id] == text]

    lm = _TensorizedLMBase(
        Dafny(),
        Tokenizer(),
        [token_texts[token_id] for token_id in token_ids],
        token_ids,
    )
    lm._structured_prompt = SpiderPromptParts(
        "db_id: x\nquestion: q\n",
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
    )
    return lm


def test_spider_unconstrained_recorder_discards_post_marker_unused_ids():
    lm = _recording_spider_lm(
        [10, 11, 13, 14, 2],
        {
            10: "SELECT ",
            11: "name ",
            13: "<<",
            14: "unused-after-marker",
            2: "<eos>",
        },
    )

    result = lm._build_unconstrained_chunk_result([10, 11, 13, 14, 2], "<<", "<eos>", 10)

    assert result[0] == ["SELECT ", "name ", "<<"]
    assert result[1:] == (True, False, 3)
    assert lm._last_generation_evidence["raw_token_ids"] == [10, 11, 13]
    assert lm._last_generation_evidence["removed_terminal_token_ids"] == []
    assert lm._last_generation_evidence["decoded_text"] == "SELECT name <<"


def test_spider_rejected_unconstrained_retry_discards_speculative_id():
    import types
    import torch

    lm = _recording_spider_lm(
        [1, 2, 3],
        {1: "invented", 2: "name", 3: "<eos>"},
    )
    lm._generation_stop_token_ids = {3}
    lm._full_logits = torch.tensor([0.0, 5.0, 4.0, -1.0])
    lm.ChooseNextTokenUnconstrained()
    lm._oracle_node = types.SimpleNamespace(log_theta=torch.zeros(1, 4), raw_logprob=None)
    lm._oracle_depth = 0
    lm._oracle_context_ids = []
    lm._oracle_pending_reject_id = None
    lm._decode_trace_token_ids = set()
    lm.RejectLastInTrie()
    lm._full_logits = torch.tensor([0.0, -1.0, 5.0, -1.0])
    lm.ChooseNextTokenUnconstrained()

    evidence = lm._finalize_generation_evidence()
    assert evidence["raw_token_ids"] == [2]
    assert evidence["decoded_text"] == "name"


def test_spider_constrained_rollback_discards_committed_token():
    import torch

    lm = _recording_spider_lm(
        [1, 2, 3],
        {1: "invented", 2: "name", 3: "<eos>"},
    )
    lm._generation_stop_token_ids = {3}
    lm._logits_tensor = torch.tensor([0.0, 5.0, 4.0, -1.0])
    lm.ChooseNextToken()
    lm._oracle_node.log_theta = torch.zeros(1, 4)
    lm._oracle_pending_reject_id = None
    lm._oracle_context_ids = [1]
    lm._last_unconstrained_token_id = None
    lm.RejectLastInTrie()
    lm._logits_tensor = torch.tensor([0.0, -1.0, 5.0, -1.0])
    lm.ChooseNextToken()

    evidence = lm._finalize_generation_evidence()
    assert evidence["raw_token_ids"] == [2, 3]
    assert evidence["removed_terminal_token_ids"] == [3]
    assert evidence["decoded_text"] == "name"


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
            self._committed_token_ids = []
            self._generation_stop_token_ids = {2}
            self.task_guidance = None

        def _record_generated_token_ids(self, token_ids):
            self._committed_token_ids.extend(int(token_id) for token_id in token_ids)

        def _finalize_generation_evidence(self):
            from synthesis.evaluate.benchmarks.sql_spider.output_contract import generation_token_evidence

            self._last_generation_evidence = generation_token_evidence(
                self._committed_token_ids,
                self.tokenizer,
                terminal_stop_token_ids=self._generation_stop_token_ids,
            )
            return self._last_generation_evidence

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
            lm_arg._record_generated_token_ids(token_ids)
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

@pytest.fixture
def _verified_csd_helpers(monkeypatch):
    """Load the checked-in compiled CSD helper implementation with a tiny Dafny shim."""
    import contextlib
    import importlib
    import sys
    import types

    ref_dir = (
        Path(__file__).resolve().parents[1]
        / "outputs"
        / "compiled_references"
        / "crane_faithful"
        / "ref_crane_faithful"
    )
    dafny = types.ModuleType("_dafny")

    class _Array:
        def __init__(self, default, size):
            self.values = [default] * size

        def __getitem__(self, index):
            return self.values[index]

        def __setitem__(self, index, value):
            self.values[index] = value

        def length(self, _dimension):
            return len(self.values)

    class _TailCall(Exception):
        pass

    class _Break(Exception):
        def __init__(self, label):
            super().__init__(label)
            self.label = label

    dafny.Array = _Array
    dafny.TailCall = _TailCall
    dafny.Seq = lambda value: [] if isinstance(value, dict) else list(value)
    dafny.SeqWithoutIsStrInference = lambda values: list(values)
    dafny.CodePoint = lambda value: value
    dafny.BigRational = lambda value: float(value)
    dafny.IntegerRange = lambda start, end: range(start, end)
    dafny.quantifier = lambda *args: False

    @contextlib.contextmanager
    def _label(label):
        try:
            yield
        except _Break as exc:
            if exc.label != label:
                raise

    @contextlib.contextmanager
    def _c_label(*_args):
        yield

    dafny.Break = _Break
    dafny.label = _label
    dafny.c_label = _c_label
    monkeypatch.setitem(sys.modules, "_dafny", dafny)
    monkeypatch.setitem(sys.modules, "System_", types.ModuleType("System_"))
    monkeypatch.syspath_prepend(str(ref_dir))
    for module_name in ("module_", "GeneratedCSD", "VerifiedDecoderAgent"):
        sys.modules.pop(module_name, None)
    verified = importlib.import_module("VerifiedDecoderAgent")
    monkeypatch.setattr(
        verified.default__,
        "RenderPrefix",
        staticmethod(lambda prefix: list(prefix)),
    )
    return verified.CSDHelpers


class _HelperParser:
    def IsValidPrefix(self, prefix):
        return "bad" not in list(prefix)

    def IsDeadPrefix(self, prefix):
        return list(prefix) == ["bad"]

    def IsCompletePrefix(self, prefix):
        return len(prefix) >= 1

    def CompletedSchemaSymbolCount(self, prefix):
        return len(prefix)


def _spider_helper_lm():
    import types

    import torch

    from synthesis.evaluate.benchmarks.common.model_utils import _TensorizedLMBase
    from synthesis.evaluate.benchmarks.sql_spider.prompts import SpiderPromptParts

    class Dafny:
        @staticmethod
        def Seq(value):
            return value

        @staticmethod
        def SeqWithoutIsStrInference(values):
            return list(values)

    class Tokenizer:
        all_special_ids = {3}
        eos_token_id = 3
        eos_token = "eos"

        def decode(self, token_ids, skip_special_tokens=False):
            pieces = {1: "bad", 2: "good", 3: "eos"}
            return "".join(pieces[int(token_id)] for token_id in token_ids)

        def encode(self, text, add_special_tokens=False):
            return {"bad": [1], "good": [2], "eos": [3]}.get(text, [])

    lm = _TensorizedLMBase(Dafny(), Tokenizer(), ["bad", "good", "eos"], [1, 2, 3])
    lm.Tokens = lm._Tokens
    lm._structured_prompt = SpiderPromptParts(
        "db_id: x\nquestion: q\n", model_name="Qwen/Qwen2.5-1.5B-Instruct"
    )
    lm._generation_stop_token_ids = {3}

    def generate_logits(self, prefix):
        begin_transaction = getattr(self, "_begin_generation_transaction", None)
        if callable(begin_transaction):
            begin_transaction(prefix)
        calls = getattr(self, "_test_generate_calls", 0)
        self._test_generate_calls = calls + 1
        if calls == 0:
            self._full_logits = torch.tensor([0.0, 5.0, 4.0, 0.0])
        else:
            self._full_logits = torch.tensor([0.0, 4.0, 5.0, 0.0])
        self._logits_tensor = self._full_logits[self._token_ids_tensor]
        self.Logits.update_tensors(self._logits_tensor, self._full_logits)
        self._apply_recurrence_penalty(self.instruction_text + self._prefix_text(prefix))
        self._logits_dirty = False

    lm.GenerateLogits = types.MethodType(generate_logits, lm)
    lm.MaskValidNextAndEos = types.MethodType(lambda self, *args: None, lm)

    def first_ungrounded(self, unit_tokens):
        return (bool(unit_tokens) and self._to_str(unit_tokens[0]) == "bad", 0)

    lm.FirstUngroundedIdentifierTokenIdx = types.MethodType(first_ungrounded, lm)
    return lm


def _finalize_spider_scored_prefix(lm, scored_output="good"):
    from synthesis.evaluate.benchmarks.gsm_symbolic.generation import (
        _finalize_spider_generation_evidence,
    )

    _finalize_spider_generation_evidence(
        lm, spider_prompt_active=True, scored_output=scored_output
    )
    return lm._last_generation_evidence


def test_spider_dead_end_helper_reconciles_masked_retry_to_committed_prefix(
    _verified_csd_helpers,
):
    lm = _spider_helper_lm()
    helper = _verified_csd_helpers()
    helper.ctor__()

    next_token, success = helper.DeadEndAvoidingStep(
        lm, _HelperParser(), [], [], "eos", 1
    )

    assert next_token == "good" and success is True
    evidence = _finalize_spider_scored_prefix(lm)
    assert evidence["raw_token_ids"] == [2]
    assert evidence["decoded_text"] == "good"


def test_spider_check_failure_helper_reconciles_masked_retry_to_committed_prefix(
    _verified_csd_helpers,
):
    lm = _spider_helper_lm()
    helper = _verified_csd_helpers()
    helper.ctor__()

    result = helper.RegenerateUnitOnCheckFailure(
        lm,
        _HelperParser(),
        [],
        [],
        "eos",
        2,
        1,
        2,
        [["good"]],
    )

    assert result == ["good"]
    evidence = _finalize_spider_scored_prefix(lm)
    assert evidence["raw_token_ids"] == [2]
    assert evidence["decoded_text"] == "good"


def test_spider_grounding_failure_helper_reconciles_penalty_retry_to_committed_prefix(
    _verified_csd_helpers,
):
    lm = _spider_helper_lm()
    helper = _verified_csd_helpers()
    helper.ctor__()

    result = helper.RegenerateUnitOnGroundingFailure(
        lm, _HelperParser(), [], [], "eos", 2, 1, 2
    )

    assert result == ["good"]
    evidence = _finalize_spider_scored_prefix(lm)
    assert evidence["raw_token_ids"] == [2]
    assert evidence["decoded_text"] == "good"


def test_spider_evidence_contract_mismatch_propagates_as_typed_harness_error():
    from synthesis.evaluate.evaluator import Evaluator
    from synthesis.evaluate.benchmarks.gsm_symbolic.generation import (
        _finalize_spider_generation_evidence,
    )

    lm = _spider_helper_lm()
    lm._record_generated_token_ids([1])

    evaluator = Evaluator(
        dataset_name="spider",
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
        backend="huggingface",
        device="cpu",
        sample_size=1,
        max_steps=8,
    )
    evaluator._base_grammar_text = _GRAMMAR_PATH.read_text()

    def fake_run(**kwargs):
        _finalize_spider_generation_evidence(
            lm, spider_prompt_active=True, scored_output="different"
        )

    with pytest.raises(RuntimeError, match="does not match") as exc_info:
        evaluator._evaluate_one_example(
            0,
            _example(),
            1,
            {"lm": lm, "tokenizer": lm.tokenizer},
            sql_eval_logic,
            fake_run,
            {},
        )

    assert type(exc_info.value).__name__ == "SpiderEvidenceContractError"

def test_spider_reconciliation_rejects_unsampled_retokenization_and_aborts():
    from synthesis.evaluate.benchmarks.common.model_utils import _TensorizedLMBase
    from synthesis.evaluate.benchmarks.gsm_symbolic.generation import (
        _finalize_spider_generation_evidence,
    )
    from synthesis.evaluate.benchmarks.sql_spider.output_contract import (
        SpiderEvidenceContractError,
    )
    from synthesis.evaluate.benchmarks.sql_spider.prompts import SpiderPromptParts

    class ContextSensitiveTokenizer:
        all_special_ids = set()
        eos_token_id = 99

        def __init__(self):
            self.encode_calls = []

        def decode(self, token_ids, skip_special_tokens=False):
            values = {
                (): "",
                (1,): "x",
                (2,): "y",
                (1, 2): "abc",
                (3,): "ab",
            }
            return values[tuple(int(token_id) for token_id in token_ids)]

        def encode(self, text, add_special_tokens=False):
            self.encode_calls.append((text, add_special_tokens))
            return [3] if text == "ab" else []

    class Dafny:
        @staticmethod
        def Seq(value):
            return value

        @staticmethod
        def SeqWithoutIsStrInference(values):
            return list(values)

    tokenizer = ContextSensitiveTokenizer()
    lm = object.__new__(_TensorizedLMBase)
    lm.tokenizer = tokenizer
    lm._dafny = Dafny()
    lm._token_id_to_str = {}
    lm._structured_prompt = SpiderPromptParts(
        "db_id: x\nquestion: q\n",
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
    )
    lm._generation_stop_token_ids = {99}
    lm._generation_token_ids = [1, 2]

    assert lm._reconcile_generation_evidence("ab") is False
    assert lm._generation_token_ids == [1, 2]
    assert tokenizer.encode_calls == []

    with pytest.raises(SpiderEvidenceContractError):
        _finalize_spider_generation_evidence(
            lm,
            spider_prompt_active=True,
            scored_output="ab",
        )


def _spider_duplicate_text_helper_lm():
    import types

    import torch

    from synthesis.evaluate.benchmarks.common.model_utils import _TensorizedLMBase
    from synthesis.evaluate.benchmarks.sql_spider.prompts import SpiderPromptParts

    class Dafny:
        @staticmethod
        def Seq(value):
            return value

        @staticmethod
        def SeqWithoutIsStrInference(values):
            return list(values)

    class Tokenizer:
        all_special_ids = {99}
        eos_token_id = 99
        eos_token = "eos"

        def decode(self, token_ids, skip_special_tokens=False):
            pieces = {1: "x", 2: "x", 99: "eos"}
            return "".join(pieces[int(token_id)] for token_id in token_ids)

        def encode(self, text, add_special_tokens=False):
            return {"x": [1], "eos": [99]}.get(text, [])

    lm = _TensorizedLMBase(Dafny(), Tokenizer(), ["x", "x", "eos"], [1, 2, 99])
    lm.Tokens = lm._Tokens
    lm._structured_prompt = SpiderPromptParts(
        "db_id: x\nquestion: q\n",
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
    )
    lm._generation_stop_token_ids = {99}
    lm._callback_calls = {"mask": 0, "penalize": 0}

    def generate_logits(self, prefix):
        begin_transaction = getattr(self, "_begin_generation_transaction", None)
        if callable(begin_transaction):
            begin_transaction(prefix)
        calls = getattr(self, "_test_generate_calls", 0)
        self._test_generate_calls = calls + 1
        full_logits = torch.full((100,), -1e9)
        full_logits[99 if calls else 2] = 5.0
        self._full_logits = full_logits
        self._logits_tensor = self._full_logits[self._token_ids_tensor]
        self.Logits.update_tensors(self._logits_tensor, self._full_logits)
        self._apply_recurrence_penalty(self.instruction_text + self._prefix_text(prefix))
        self._logits_dirty = False

    lm.GenerateLogits = types.MethodType(generate_logits, lm)
    lm.MaskValidNextAndEos = types.MethodType(lambda self, *args: None, lm)

    original_mask = lm.MaskToken
    def mask_token(self, token):
        self._callback_calls["mask"] += 1
        return original_mask(token)

    original_penalize = lm.PenalizeTriedTokenAt
    def penalize_token(self, prefix, token):
        self._callback_calls["penalize"] += 1
        return original_penalize(prefix, token)

    lm.MaskToken = types.MethodType(mask_token, lm)
    lm.PenalizeTriedTokenAt = types.MethodType(penalize_token, lm)

    def first_ungrounded(self, unit_tokens):
        return (bool(unit_tokens) and self._to_str(unit_tokens[0]) == "x", 0)

    lm.FirstUngroundedIdentifierTokenIdx = types.MethodType(first_ungrounded, lm)
    return lm


def test_spider_check_failure_real_mask_retry_keeps_checkpoint_token_provenance(
    _verified_csd_helpers,
):
    lm = _spider_duplicate_text_helper_lm()
    helper = _verified_csd_helpers()
    helper.ctor__()
    lm._record_generated_token_ids([1])

    result = helper.RegenerateUnitOnCheckFailure(
        lm, _HelperParser(), [], ["x"], "eos", 2, 1, 1, [["allowed"]]
    )

    assert result == ["x"]
    assert lm._callback_calls["mask"] >= 1
    evidence = _finalize_spider_scored_prefix(lm, scored_output="x")
    assert evidence["raw_token_ids"] == [1, 99]
    assert evidence["decoded_text"] == "x"


def test_spider_grounding_failure_real_penalty_retry_keeps_checkpoint_token_provenance(
    _verified_csd_helpers,
):
    lm = _spider_duplicate_text_helper_lm()
    helper = _verified_csd_helpers()
    helper.ctor__()
    lm._record_generated_token_ids([1])

    result = helper.RegenerateUnitOnGroundingFailure(
        lm, _HelperParser(), [], ["x"], "eos", 2, 1, 1
    )

    assert result == ["x"]
    assert lm._callback_calls["penalize"] >= 1
    evidence = _finalize_spider_scored_prefix(lm, scored_output="x")
    assert evidence["raw_token_ids"] == [1, 99]
    assert evidence["decoded_text"] == "x"


def test_spider_finalizer_propagates_false_reconcile_before_matching_full_decode():
    from synthesis.evaluate.benchmarks.common.model_utils import _TensorizedLMBase
    from synthesis.evaluate.benchmarks.gsm_symbolic.generation import (
        _finalize_spider_generation_evidence,
    )
    from synthesis.evaluate.benchmarks.sql_spider.output_contract import (
        SpiderEvidenceContractError,
    )
    from synthesis.evaluate.benchmarks.sql_spider.prompts import SpiderPromptParts

    class ContextTokenizer:
        all_special_ids = set()
        eos_token_id = 99

        def decode(self, token_ids, skip_special_tokens=False):
            values = {
                (1,): "left",
                (2,): "right",
                (1, 2): "ab",
            }
            return values[tuple(int(token_id) for token_id in token_ids)]

        def encode(self, text, add_special_tokens=False):
            return []

    class Dafny:
        @staticmethod
        def Seq(value):
            return value

        @staticmethod
        def SeqWithoutIsStrInference(values):
            return list(values)

    lm = object.__new__(_TensorizedLMBase)
    lm.tokenizer = ContextTokenizer()
    lm._dafny = Dafny()
    lm._token_id_to_str = {}
    lm._structured_prompt = SpiderPromptParts(
        "db_id: x\nquestion: q\n",
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
    )
    lm._generation_stop_token_ids = {99}
    lm._generation_token_ids = [1, 2]
    lm._reconcile_generation_evidence = lambda _scored_output: False

    assert lm._reconcile_generation_evidence("ab") is False
    with pytest.raises(SpiderEvidenceContractError):
        _finalize_spider_generation_evidence(
            lm,
            spider_prompt_active=True,
            scored_output="ab",
        )

def _spider_two_token_duplicate_helper_lm():
    import types

    import torch

    from synthesis.evaluate.benchmarks.common.model_utils import _TensorizedLMBase
    from synthesis.evaluate.benchmarks.sql_spider.prompts import SpiderPromptParts

    class Dafny:
        @staticmethod
        def Seq(value):
            return value

        @staticmethod
        def SeqWithoutIsStrInference(values):
            return list(values)

    class Tokenizer:
        all_special_ids = {99}
        eos_token_id = 99
        eos_token = "eos"

        def decode(self, token_ids, skip_special_tokens=False):
            pieces = {1: "x", 2: "x", 3: "bad", 4: "good", 99: "eos"}
            return "".join(pieces[int(token_id)] for token_id in token_ids)

        def encode(self, text, add_special_tokens=False):
            return {"x": [1], "bad": [3], "good": [4], "eos": [99]}.get(text, [])

    lm = _TensorizedLMBase(
        Dafny(),
        Tokenizer(),
        ["x", "x", "bad", "good", "eos"],
        [1, 2, 3, 4, 99],
    )
    lm.Tokens = lm._Tokens
    lm._structured_prompt = SpiderPromptParts(
        "db_id: x\nquestion: q\n",
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
    )
    lm._generation_stop_token_ids = {99}
    lm._test_generate_calls = 0

    # First unit: [1, 3] -> x,bad. After rollback, retry unit: [2, 4] -> x,good.
    sequence = [1, 3, 3, 2, 4, 99]

    def generate_logits(self, prefix):
        self._begin_generation_transaction(prefix)
        call = self._test_generate_calls
        self._test_generate_calls += 1
        full_logits = torch.full((100,), -1e9)
        full_logits[sequence[min(call, len(sequence) - 1)]] = 5.0
        self._full_logits = full_logits
        self._logits_tensor = self._full_logits[self._token_ids_tensor]
        self.Logits.update_tensors(self._logits_tensor, self._full_logits)
        self._apply_recurrence_penalty(self.instruction_text + self._prefix_text(prefix))
        self._logits_dirty = False

    lm.GenerateLogits = types.MethodType(generate_logits, lm)
    lm.MaskValidNextAndEos = types.MethodType(lambda self, *args: None, lm)

    def first_ungrounded(self, unit_tokens):
        values = list(unit_tokens)
        return (len(values) >= 2 and self._to_str(values[1]) == "bad", 1)

    lm.FirstUngroundedIdentifierTokenIdx = types.MethodType(first_ungrounded, lm)
    return lm


class _TwoTokenParser(_HelperParser):
    def IsCompletePrefix(self, prefix):
        return len(prefix) >= 2

    def CompletedSchemaSymbolCount(self, prefix):
        return 2 if len(prefix) >= 2 else 0


def test_spider_check_failure_two_token_retry_keeps_accepted_id_provenance(
    _verified_csd_helpers,
):
    lm = _spider_two_token_duplicate_helper_lm()
    helper = _verified_csd_helpers()
    helper.ctor__()

    result = helper.RegenerateUnitOnCheckFailure(
        lm,
        _TwoTokenParser(),
        [],
        [],
        "eos",
        6,
        1,
        1,
        [["x", "good"]],
    )

    assert result == ["x", "good"]
    evidence = _finalize_spider_scored_prefix(lm, scored_output="xgood")
    assert evidence["raw_token_ids"] == [2, 4, 99]
    assert evidence["decoded_text"] == "xgood"


def test_spider_grounding_failure_two_token_retry_keeps_accepted_id_provenance(
    _verified_csd_helpers,
):
    lm = _spider_two_token_duplicate_helper_lm()
    helper = _verified_csd_helpers()
    helper.ctor__()

    result = helper.RegenerateUnitOnGroundingFailure(
        lm,
        _TwoTokenParser(),
        [],
        [],
        "eos",
        6,
        1,
        1,
    )

    assert result == ["x", "good"]
    evidence = _finalize_spider_scored_prefix(lm, scored_output="xgood")
    assert evidence["raw_token_ids"] == [2, 4, 99]
    assert evidence["decoded_text"] == "xgood"


def _factory_entry_lm(monkeypatch, backend, events):
    import sys
    import types

    import torch

    from synthesis.evaluate.benchmarks.common import model_utils
    from synthesis.evaluate.benchmarks.sql_spider.prompts import SpiderPromptParts

    class Tokenizer:
        all_special_ids = {99}
        eos_token_id = 99
        eos_token = "eos"
        generation_stop_token_ids = {99}

        def __len__(self):
            return 2

        def decode(self, token_ids, skip_special_tokens=False):
            return "".join({1: "x", 99: "eos"}[int(token_id)] for token_id in token_ids)

        def encode(self, text, add_special_tokens=False):
            return []

    class Dafny:
        @staticmethod
        def Seq(value):
            return value

        @staticmethod
        def SeqWithoutIsStrInference(values):
            return list(values)

    class VerifiedDecoderAgent:
        class LM:
            def __init__(self):
                pass

    tokenizer = Tokenizer()

    if backend == "huggingface":
        class FakeHFModel:
            def eval(self):
                return self

            def __call__(self, input_ids=None, **kwargs):
                events.append("backend")
                seq_len = max(1, int(input_ids.shape[-1]))
                logits = torch.zeros((1, seq_len, 100))
                logits[:, -1, 1] = 1.0
                return types.SimpleNamespace(logits=logits)

            def generate(self, **kwargs):
                events.append("backend")
                input_ids = kwargs["input_ids"]
                return torch.cat(
                    [input_ids, torch.tensor([[1]], dtype=torch.long)],
                    dim=1,
                )

        monkeypatch.setattr(model_utils, "load_runtime_tokenizer", lambda *args, **kwargs: tokenizer)
        monkeypatch.setattr(
            model_utils.AutoModelForCausalLM,
            "from_pretrained",
            lambda *args, **kwargs: FakeHFModel(),
        )
        monkeypatch.setattr(model_utils, "get_max_input_length", lambda *args, **kwargs: 64)
        lm = model_utils.create_huggingface_lm(
            model_name="Qwen/Qwen2.5-1.5B-Instruct",
            device="cpu",
            VerifiedDecoderAgent=VerifiedDecoderAgent,
            _dafny=Dafny(),
            token_ids=[1, 99],
        )
    else:
        class _CudaNamedCpu(str):
            def startswith(self, prefix):
                return prefix == "cuda" or super().startswith(prefix)

        class SamplingParams:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        class FakeEngine:
            def generate(self, prompts, sampling_params, use_tqdm=False):
                events.append("backend")
                output = types.SimpleNamespace(
                    token_ids=[1],
                    logprobs=[{1: types.SimpleNamespace(logprob=0.0)}],
                )
                return [types.SimpleNamespace(outputs=[output])]

        fake_vllm = types.ModuleType("vllm")
        fake_vllm.SamplingParams = SamplingParams
        monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
        monkeypatch.setattr(model_utils, "_configure_vllm_multiprocessing", lambda: None)
        monkeypatch.setattr(model_utils, "resolve_vllm_tensor_parallel_size", lambda value: 1)
        monkeypatch.setattr(
            model_utils,
            "_get_cached_vllm_engine",
            lambda **kwargs: (FakeEngine(), tokenizer),
        )
        lm = model_utils.create_vllm_lm(
            model_name="Qwen/Qwen2.5-1.5B-Instruct",
            device=_CudaNamedCpu("cpu"),
            VerifiedDecoderAgent=VerifiedDecoderAgent,
            _dafny=Dafny(),
            token_ids=[1, 99],
        )

    lm._structured_prompt = SpiderPromptParts(
        "db_id: x\nquestion: q\n",
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
    )
    lm.instruction_text = "db_id: x\nquestion: q\n"
    return lm


def _assert_backend_entry_transaction_order(monkeypatch, backend, method_name):
    events = []
    lm = _factory_entry_lm(monkeypatch, backend, events)
    original_begin = lm._begin_generation_transaction

    def tracked_begin(prefix):
        events.append("begin")
        return original_begin(prefix)

    monkeypatch.setattr(lm, "_begin_generation_transaction", tracked_begin)

    if method_name == "GenerateLogits":
        lm.GenerateLogits([])
    else:
        lm.GenerateUnconstrainedChunk([], 1, "<open>", "eos")

    assert events[:2] == ["begin", "backend"]

    if method_name == "GenerateLogits":
        events.clear()
        lm._last_full_prompt = lm.instruction_text
        lm._logits_dirty = False
        lm.GenerateLogits([])
        assert events == ["begin"]


def test_huggingface_constrained_entry_begins_transaction_before_cache(
    monkeypatch,
):
    _assert_backend_entry_transaction_order(monkeypatch, "huggingface", "GenerateLogits")


def test_huggingface_unconstrained_entry_begins_transaction_before_generation(
    monkeypatch,
):
    _assert_backend_entry_transaction_order(
        monkeypatch, "huggingface", "GenerateUnconstrainedChunk"
    )


def test_vllm_constrained_entry_begins_transaction_before_cache(monkeypatch):
    _assert_backend_entry_transaction_order(monkeypatch, "vllm", "GenerateLogits")


def test_vllm_unconstrained_entry_begins_transaction_before_generation(monkeypatch):
    _assert_backend_entry_transaction_order(
        monkeypatch, "vllm", "GenerateUnconstrainedChunk"
    )


def _strategy_mutation_lm():
    """Small tensorized LM whose sampled IDs are distinct from strategy text."""
    import types

    import torch

    from synthesis.evaluate.benchmarks.common.model_utils import _TensorizedLMBase
    from synthesis.evaluate.benchmarks.sql_spider.prompts import SpiderPromptParts

    class Dafny:
        @staticmethod
        def Seq(value):
            return value

        @staticmethod
        def SeqWithoutIsStrInference(values):
            return list(values)

    class Tokenizer:
        all_special_ids = {99}
        eos_token_id = 99
        eos_token = "eos"

        def decode(self, token_ids, skip_special_tokens=False):
            del skip_special_tokens
            pieces = {1: "a", 2: "b", 3: "c", 99: "eos"}
            return "".join(pieces[int(token_id)] for token_id in token_ids)

        def encode(self, text, add_special_tokens=False):
            del add_special_tokens
            return {"a": [1], "b": [2], "c": [3], "eos": [99]}.get(text, [])

    lm = _TensorizedLMBase(
        Dafny(), Tokenizer(), ["a", "b", "c", "eos"], [1, 2, 3, 99]
    )
    lm.Tokens = lm._Tokens
    lm._structured_prompt = SpiderPromptParts(
        "db_id: x\nquestion: q\n",
        model_name="Qwen/Qwen2.5-7B-Instruct",
    )
    lm._generation_stop_token_ids = {99}

    def generate_logits(self, prefix):
        self._begin_generation_transaction(prefix)
        full_logits = torch.full((100,), -1e9)
        full_logits[3] = 5.0
        self._full_logits = full_logits
        self._logits_tensor = self._full_logits[self._token_ids_tensor]
        self.Logits.update_tensors(self._logits_tensor, self._full_logits)
        self._logits_dirty = False

    lm.GenerateLogits = types.MethodType(generate_logits, lm)
    lm.MaskValidNextAndEos = types.MethodType(lambda self, *args: None, lm)
    # The preserved generated strategy calls this before its mutation-only path;
    # prompt rendering is not part of these evidence tests.
    lm.AppendTaskGuidance = types.MethodType(lambda self, guidance: None, lm)
    return lm


def _strategy_origin_alias_lm():
    """Small Spider LM where a sampled close marker aliases strategy output."""
    import types

    from synthesis.evaluate.benchmarks.common.model_utils import _TensorizedLMBase
    from synthesis.evaluate.benchmarks.sql_spider.prompts import SpiderPromptParts

    class Dafny:
        @staticmethod
        def Seq(value):
            return value

        @staticmethod
        def SeqWithoutIsStrInference(values):
            return list(values)

    class Tokenizer:
        all_special_ids = {99}
        eos_token_id = 99
        eos_token = "eos"

        def decode(self, token_ids, skip_special_tokens=False):
            del skip_special_tokens
            pieces = {1: "a", 2: "b", 3: ">>", 99: "eos"}
            return "".join(pieces[int(token_id)] for token_id in token_ids)

        def encode(self, text, add_special_tokens=False):
            del add_special_tokens
            return {"a": [1], "b": [2], ">>": [3], "eos": [99]}.get(text, [])

    lm = _TensorizedLMBase(
        Dafny(), Tokenizer(), ["a", "b", ">>", "eos"], [1, 2, 3, 99]
    )
    lm.Tokens = lm._Tokens
    lm._structured_prompt = SpiderPromptParts(
        "db_id: x\nquestion: q\n",
        model_name="Qwen/Qwen2.5-7B-Instruct",
    )
    lm._generation_stop_token_ids = {99}
    lm.AppendTaskGuidance = types.MethodType(lambda self, guidance: None, lm)
    return lm


class _StrategyCompleteParser:
    def IsCompletePrefix(self, prefix):
        return list(prefix) == ["a"]


class _StrategyRegenerateParser:
    def IsValidPrefix(self, prefix):
        return "bad" not in list(prefix)

    def IsDeadPrefix(self, prefix):
        return False

    def IsCompletePrefix(self, prefix):
        return list(prefix) == ["a", "c"]


def test_strategy_rollback_to_complete_keeps_raw_ids_and_scores_returned_text(
    _verified_csd_helpers,
):
    """Strategy removal is separate from unchanged sampled-ID evidence."""
    from synthesis.evaluate.benchmarks.gsm_symbolic.generation import (
        _finalize_spider_generation_evidence,
    )
    from synthesis.evaluate.benchmarks.sql_spider.output_contract import (
        SpiderEvidenceContractError,
    )

    lm = _strategy_mutation_lm()
    helper = _verified_csd_helpers()
    helper.ctor__()
    lm._record_generated_token_ids([1, 2])

    generated, current = helper.RollbackConstrainedToComplete(
        _StrategyCompleteParser(), ["a", "b"], ["a", "b"]
    )

    assert generated == ["a"]
    assert current == ["a"]
    assert lm._generation_token_ids == [1, 2]
    strategy_output = "".join(generated)
    try:
        _finalize_spider_generation_evidence(
            lm, spider_prompt_active=True, scored_output=strategy_output,
            strategy_token_sequence=generated,
        )
    except SpiderEvidenceContractError as exc:
        pytest.fail(f"strategy-authored rollback text aborted evidence finalization: {exc}")
    assert lm._last_generation_evidence["strategy_output_text"] == strategy_output
    assert lm._last_generation_evidence["strategy_output_relation"] == "mixed"
    assert lm._last_generation_evidence["strategy_mutation"] is True
    assert lm._last_generation_evidence["raw_token_ids"] == [1, 2]


def test_strategy_rollback_and_regenerate_without_callback_discards_removed_id(
    _verified_csd_helpers,
):
    """Callback-free strategy rollback must not treat removed ID 2 as committed."""
    from synthesis.evaluate.benchmarks.gsm_symbolic.generation import (
        _finalize_spider_generation_evidence,
    )
    from synthesis.evaluate.benchmarks.sql_spider.output_contract import (
        SpiderEvidenceContractError,
    )

    lm = _strategy_mutation_lm()
    helper = _verified_csd_helpers()
    helper.ctor__()
    lm._record_generated_token_ids([1, 2])

    result = helper.RollbackAndRegenerate(
        lm,
        _StrategyRegenerateParser(),
        [],
        ["a", "bad"],
        "eos",
        1,
        0,
    )

    assert result == ["a", "c"]
    assert lm._generation_token_ids == [1, 3]
    try:
        _finalize_spider_generation_evidence(
            lm, spider_prompt_active=True, scored_output="ac",
            strategy_token_sequence=result,
        )
    except SpiderEvidenceContractError as exc:
        pytest.fail(f"callback-free strategy rollback aborted evidence finalization: {exc}")
    assert lm._last_generation_evidence["raw_token_ids"] == [1, 3]
    assert lm._last_generation_evidence["strategy_output_text"] == "ac"
    assert lm._last_generation_evidence["strategy_output_relation"] == "mixed"
    assert lm._last_generation_evidence["strategy_mutation"] is True


def test_strategy_close_span_keeps_sampled_ids_and_reaches_strict_wrapper_rejection(
    _verified_csd_helpers,
):
    """CloseConstrainedSpan output is scored strictly after raw-ID preservation."""
    from synthesis.evaluate.benchmarks.common.dafny_tokens import dafny_seq_to_str
    from synthesis.evaluate.benchmarks.gsm_symbolic.generation import (
        _finalize_spider_generation_evidence,
    )
    from synthesis.evaluate.benchmarks.sql_spider import eval_logic as sql_eval_logic
    from synthesis.evaluate.benchmarks.sql_spider.output_contract import (
        SpiderEvidenceContractError,
    )

    lm = _strategy_mutation_lm()
    helper = _verified_csd_helpers()
    helper.ctor__()
    lm._record_generated_token_ids([1])

    generated, inside, current = helper.CloseConstrainedSpan(
        lm, _StrategyCompleteParser(), ["a"], ["a"]
    )

    assert inside is False
    assert current == []
    strategy_output = "".join(dafny_seq_to_str(token) for token in generated)
    assert strategy_output == "a>>"
    assert lm._generation_token_ids == [1]
    try:
        _finalize_spider_generation_evidence(
            lm, spider_prompt_active=True, scored_output=strategy_output,
            strategy_token_sequence=generated,
        )
    except SpiderEvidenceContractError as exc:
        pytest.fail(f"strategy-authored close marker aborted evidence finalization: {exc}")

    actual, source, aux = sql_eval_logic.extract_actual(
        _CachedRealEvaluator(), strategy_output, _example()
    )
    assert actual is None
    assert source == "spider_output_contract_rejected"
    assert aux["output_rejection_reason"] == "prompt_or_wrapper"
    assert lm._last_generation_evidence["strategy_output_text"] == strategy_output
    assert lm._last_generation_evidence["strategy_output_relation"] == "mixed"
    assert lm._last_generation_evidence["strategy_mutation"] is True


def test_reevaluation_export_preserves_static_close_strategy_evidence(
    tmp_path, _verified_csd_helpers
):
    """The real evaluator/exporter must retain a compiled close-marker mutation."""
    import json
    import sys
    import types

    import torch

    from synthesis.evaluate.baseline_store import save_minimal_baseline_json
    from synthesis.evaluate.benchmarks.common.model_utils import _TensorizedLMBase
    from synthesis.evaluate.benchmarks.gsm_symbolic.environment import (
        _attach_helper_trace,
    )
    from synthesis.evaluate.benchmarks.gsm_symbolic.generation import run_crane_csd
    from synthesis.evaluate.evaluator import EvaluationResult, Evaluator

    class Dafny:
        @staticmethod
        def Seq(value):
            return value

        @staticmethod
        def SeqWithoutIsStrInference(values):
            return list(values)

    class Tokenizer:
        eos_token = "<eos>"
        eos_token_id = 99
        all_special_ids = {99}

        def decode(self, token_ids, skip_special_tokens=False):
            del skip_special_tokens
            pieces = {1: "a", 3: ">>", 99: "<eos>"}
            return "".join(pieces[int(token_id)] for token_id in token_ids)

        def encode(self, text, add_special_tokens=False):
            del add_special_tokens
            return {"a": [1], ">>": [3], "<eos>": [99]}.get(text, [])

    class Parser:
        @staticmethod
        def IsCompletePrefix(prefix):
            return list(prefix) == ["a"]

        @staticmethod
        def is_complete(text):
            return text == "a>>"

    tokenizer = Tokenizer()
    lm = _TensorizedLMBase(Dafny(), tokenizer, ["a", ">>", "<eos>"], [1, 3, 99])
    lm.Tokens = lm._Tokens
    lm.model_name = "Qwen/Qwen2.5-1.5B-Instruct"

    def generate_logits(self, prefix):
        self._begin_generation_transaction(prefix)
        full_logits = torch.full((100,), -1e9)
        full_logits[1] = 5.0
        self._finalize_full_logits(full_logits)
        self._logits_dirty = False

    lm.GenerateLogits = types.MethodType(generate_logits, lm)

    helpers_cls = _verified_csd_helpers
    verified = sys.modules["VerifiedDecoderAgent"]
    trace_state = {"events": []}
    _attach_helper_trace(verified, trace_state)

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
            del seq0, generated_prefix, start_inside, current_constrained
            del max_steps, step_budget, eos_token
            helper = helpers_cls()
            helper.ctor__()
            lm_arg.GenerateLogits([])
            sampled = lm_arg.ChooseNextToken()
            generated, inside, current = helper.CloseConstrainedSpan(
                lm_arg, parser, [sampled], [sampled]
            )
            return generated, inside, current, 1

    class GeneratedCSD:
        default__ = GeneratedDefault

    env = {
        "_dafny": Dafny,
        "GeneratedCSD": GeneratedCSD,
        "lm": lm,
        "parser": Parser(),
        "model_name": lm.model_name,
        "tokenizer": tokenizer,
        "csd_trace": trace_state,
    }

    evaluator = Evaluator(
        dataset_name="spider",
        model_name=lm.model_name,
        backend="huggingface",
        device="cpu",
        sample_size=1,
        max_steps=8,
    )
    evaluator._base_grammar_text = _GRAMMAR_PATH.read_text(encoding="utf-8")

    def real_runner(**kwargs):
        return run_crane_csd(
            env=env,
            prompt_text=kwargs["prompt_text"],
            max_steps=kwargs["max_steps"],
            step_token_budget=kwargs.get("step_token_budget", 1),
            grammar_file=kwargs["grammar_file"],
            dynamic_parser=kwargs.get("dynamic_parser"),
            start_inside_constrained=True,
        )

    sample = evaluator._evaluate_one_example(
        0,
        _example(),
        1,
        env,
        sql_eval_logic,
        real_runner,
        {},
    )
    assert sample["full_output"] == "a>>"
    assert sample["output_rejection_reason"] == "prompt_or_wrapper"
    assert sample["generation_token_evidence"]["raw_token_ids"] == [1]
    assert any(event["helper"] == "CloseConstrainedSpan" for event in sample["helper_trace"])

    result = EvaluationResult(
        success=True,
        accuracy=0.0,
        contains_delimiters=False,
        syntax_rate=0.0,
        num_examples=1,
        num_correct=0,
        total_time_seconds=sample["time_seconds"],
        sample_outputs=[sample],
    )
    output_path = save_minimal_baseline_json(result, tmp_path / "smoke.json")
    exported = json.loads(output_path.read_text(encoding="utf-8"))
    evidence = exported["reevaluation_sample_evidence"][0]

    assert evidence["strategy_mutation"] is True
    assert evidence["strategy_output_relation"] == "mixed"
    assert evidence["strategy_removed_sampled_token_ids"] == []
    assert evidence["generation_token_evidence"]["raw_token_ids"] == [1]
    assert evidence["generation_token_evidence"]["decoded_text"] == "a"


def test_strategy_origin_alias_discards_sampled_marker_before_authored_close(
    _verified_csd_helpers,
):
    """A traced rollback then close must distinguish discarded and authored markers."""
    import sys

    from synthesis.evaluate.benchmarks.common.dafny_tokens import (
        dafny_seq_to_str,
    )
    from synthesis.evaluate.benchmarks.gsm_symbolic.environment import (
        _attach_helper_trace,
    )
    from synthesis.evaluate.benchmarks.gsm_symbolic.generation import (
        _finalize_spider_generation_evidence,
    )
    from synthesis.evaluate.benchmarks.sql_spider import eval_logic as sql_eval_logic
    from synthesis.evaluate.benchmarks.sql_spider.output_contract import (
        SpiderEvidenceContractError,
    )

    lm = _strategy_origin_alias_lm()
    verified = sys.modules["VerifiedDecoderAgent"]
    trace_state = {"events": []}
    _attach_helper_trace(verified, trace_state)
    helper = _verified_csd_helpers()
    helper.ctor__()
    lm._record_generated_token_ids([1, 3])

    generated, current = helper.RollbackConstrainedToComplete(
        _StrategyCompleteParser(), ["a", ">>"], ["a", ">>"]
    )
    generated, inside, current = helper.CloseConstrainedSpan(
        lm, _StrategyCompleteParser(), generated, current
    )

    helper_names = [event["helper"] for event in trace_state["events"]]
    assert helper_names[-2:] == ["RollbackConstrainedToComplete", "CloseConstrainedSpan"]
    rollback_event = trace_state["events"][-2]
    assert rollback_event["generated_len_before"] == 2
    assert rollback_event["generated_len_after"] == 1
    assert rollback_event["current_len_before"] == 2
    assert rollback_event["current_len_after"] == 1
    assert dafny_seq_to_str(generated[0]) == "a"
    assert dafny_seq_to_str(generated[1]) == ">>"
    assert inside is False
    assert current == []
    strategy_output = "".join(dafny_seq_to_str(token) for token in generated)
    assert strategy_output == "a>>"
    try:
        _finalize_spider_generation_evidence(
            lm,
            spider_prompt_active=True,
            scored_output=strategy_output,
            strategy_token_sequence=generated,
        )
    except SpiderEvidenceContractError as exc:
        pytest.fail(f"origin-alias strategy output aborted evidence finalization: {exc}")

    evidence = lm._last_generation_evidence
    assert evidence["raw_token_ids"] == [1]
    assert evidence["raw_decoded_text"] == "a"
    assert evidence["strategy_removed_sampled_token_ids"] == [3]
    assert evidence["strategy_output_relation"] == "mixed"
    assert evidence["strategy_mutation"] is True
    actual, source, aux = sql_eval_logic.extract_actual(
        _CachedRealEvaluator(), strategy_output, _example()
    )
    assert actual is None
    assert source == "spider_output_contract_rejected"
    assert aux["output_rejection_reason"] == "prompt_or_wrapper"


def test_spider_public_rollback_to_complete_prefix_trace_preserves_origin(
    _verified_csd_helpers,
):
    """The public static rollback alias must trace removal before authored close."""
    import sys

    from synthesis.evaluate.benchmarks.common.dafny_tokens import dafny_seq_to_str
    from synthesis.evaluate.benchmarks.gsm_symbolic.environment import (
        _attach_helper_trace,
    )
    from synthesis.evaluate.benchmarks.gsm_symbolic.generation import (
        _finalize_spider_generation_evidence,
    )
    from synthesis.evaluate.benchmarks.sql_spider import eval_logic as sql_eval_logic
    from synthesis.evaluate.benchmarks.sql_spider.output_contract import (
        SpiderEvidenceContractError,
    )

    lm = _strategy_origin_alias_lm()
    helper = _verified_csd_helpers()
    verified = sys.modules["VerifiedDecoderAgent"]
    trace_state = {"events": []}
    _attach_helper_trace(verified, trace_state)
    helper_instance = helper
    helper_instance.ctor__()
    lm._record_generated_token_ids([1, 3])

    parser = _StrategyCompleteParser()
    generated = type(helper).RollbackToCompletePrefix(parser, ["a", ">>"])
    generated, inside, current = helper_instance.CloseConstrainedSpan(
        lm, parser, generated, generated
    )

    assert [dafny_seq_to_str(token) for token in generated] == ["a", ">>"]
    assert inside is False
    assert current == []
    strategy_output = "".join(dafny_seq_to_str(token) for token in generated)
    assert strategy_output == "a>>"
    helper_names = [event["helper"] for event in trace_state["events"]]
    assert helper_names[-2:] == [
        "RollbackToCompletePrefix",
        "CloseConstrainedSpan",
    ]
    rollback_event = trace_state["events"][-2]
    assert rollback_event["generated_len_before"] == 2
    assert rollback_event["generated_len_after"] == 1
    assert set(rollback_event) == {
        "helper",
        "detail",
        "cost_before",
        "cost_after",
        "generated_len_before",
        "generated_len_after",
    }
    assert all(value not in (["a", ">>"], ["a"]) for value in rollback_event.values())

    try:
        _finalize_spider_generation_evidence(
            lm,
            spider_prompt_active=True,
            scored_output=strategy_output,
            strategy_token_sequence=generated,
        )
    except SpiderEvidenceContractError as exc:
        pytest.fail(f"public rollback strategy output aborted finalization: {exc}")

    evidence = lm._last_generation_evidence
    assert evidence["raw_token_ids"] == [1]
    assert evidence["raw_decoded_text"] == "a"
    assert evidence["strategy_removed_sampled_token_ids"] == [3]
    assert evidence["strategy_output_relation"] == "mixed"
    assert evidence["strategy_mutation"] is True
    actual, source, aux = sql_eval_logic.extract_actual(
        _CachedRealEvaluator(), strategy_output, _example()
    )
    assert actual is None
    assert source == "spider_output_contract_rejected"
    assert aux["output_rejection_reason"] == "prompt_or_wrapper"


def test_spider_static_rollback_to_valid_prefix_preserves_descriptor_and_trace(
    _verified_csd_helpers,
):
    """Static rollback remains callable through both class and instance access."""
    import sys

    from synthesis.evaluate.benchmarks.gsm_symbolic.environment import (
        _attach_helper_trace,
    )

    class PrefixParser:
        def IsValidPrefix(self, prefix):
            return list(prefix) == ["a"]

        def IsDeadPrefix(self, prefix):
            return False

    helper = _verified_csd_helpers()
    verified = sys.modules["VerifiedDecoderAgent"]
    trace_state = {"events": []}
    _attach_helper_trace(verified, trace_state)
    parser = PrefixParser()
    generated = ["a", "bad"]

    class_result = type(helper).RollbackToValidPrefix(parser, generated)
    helper.ctor__()
    instance_result = helper.RollbackToValidPrefix(parser, generated)

    assert class_result == ["a"]
    assert instance_result == ["a"]
    events = [
        event
        for event in trace_state["events"]
        if event["helper"] == "RollbackToValidPrefix"
    ]
    assert len(events) == 2
    for event in events:
        assert event["generated_len_before"] == 2
        assert event["generated_len_after"] == 1
        assert all(value not in (["a", "bad"], ["a"]) for value in event.values())


def test_preserved_qwen25_7b_control_flow_rolls_back_then_closes_span(
    monkeypatch,
):
    """Run the preserved Qwen2.5-7B GeneratedCSD rollback/close control flow."""
    import hashlib
    import importlib
    import sys

    from synthesis.evaluate.benchmarks.gsm_symbolic.environment import (
        _attach_helper_trace,
    )
    from synthesis.evaluate.benchmarks.common.dafny_tokens import dafny_seq_to_str
    from synthesis.evaluate.benchmarks.gsm_symbolic.generation import (
        _finalize_spider_generation_evidence,
    )
    from synthesis.evaluate.benchmarks.sql_spider import eval_logic as sql_eval_logic
    from synthesis.evaluate.benchmarks.sql_spider.output_contract import (
        SpiderEvidenceContractError,
    )

    artifact = Path(
        "/home/aadivyar/csd-generation/outputs/generated/"
        "coldq_spider-qwen25-7b_20260724/"
        "coldq_spider-qwen25-7b_20260724_20260730_201000_b5cb23/python/"
        "coldq_spider-qwen25-7b_20260724_20260731_010805_0ff5e0/GeneratedCSD.py"
    )
    assert artifact.exists()
    assert hashlib.sha256(artifact.read_bytes()).hexdigest() == (
        "47a74f7243792f3da68996733316ffcf668ab150c738ba40c96296d306cf6c30"
    )
    monkeypatch.syspath_prepend(str(artifact.parent))
    for module_name in (
        "_dafny",
        "module_",
        "GeneratedCSD",
        "VerifiedDecoderAgent",
        "System_",
    ):
        sys.modules.pop(module_name, None)
    generated_csd = importlib.import_module("GeneratedCSD")

    trace_state = {"events": []}
    _attach_helper_trace(generated_csd.VerifiedDecoderAgent, trace_state)

    dafny = generated_csd._dafny

    def dtext(text):
        return dafny.SeqWithoutIsStrInference(map(dafny.CodePoint, text))

    def dseq(tokens):
        return dafny.SeqWithoutIsStrInference([dtext(token) for token in tokens])

    class PreservedParser:
        def IsCompletePrefix(self, prefix):
            return (
                len(prefix) == 1
                and len(prefix[0]) == 1
                and str(prefix[0][0]) == "a"
            )

    lm = _strategy_mutation_lm()
    lm._record_generated_token_ids([1, 2])
    result = generated_csd.default__.MyCSDStrategy(
        lm,
        PreservedParser(),
        dseq([]),
        dseq(["a", "b"]),
        True,
        dseq(["a", "b"]),
        1,
        1,
        [],
        "eos",
    )
    strategy_output = "".join(dafny_seq_to_str(token) for token in result[0])

    events = [event["helper"] for event in trace_state["events"]]
    assert events[-2:] == ["RollbackConstrainedToComplete", "CloseConstrainedSpan"]
    assert strategy_output == "a>>"
    assert lm._generation_token_ids == [1]
    try:
        _finalize_spider_generation_evidence(
            lm, spider_prompt_active=True, scored_output=strategy_output,
            strategy_token_sequence=result[0],
        )
    except SpiderEvidenceContractError as exc:
        pytest.fail(f"preserved strategy mutation aborted evidence finalization: {exc}")
    actual, source, aux = sql_eval_logic.extract_actual(
        _CachedRealEvaluator(), strategy_output, _example()
    )
    assert actual is None
    assert source == "spider_output_contract_rejected"
    assert aux["output_rejection_reason"] == "prompt_or_wrapper"
    assert lm._last_generation_evidence["strategy_output_text"] == strategy_output
    assert lm._last_generation_evidence["strategy_removed_sampled_token_ids"] == [2]
    assert lm._last_generation_evidence["strategy_output_relation"] == "mixed"
    assert lm._last_generation_evidence["strategy_mutation"] is True


def test_spider_alignment_preserves_multi_piece_csd_chunks_and_drops_branch():
    """One CSD chunk may contain several sampled token pieces."""
    lm = _strategy_mutation_lm()
    lm._generation_token_ids = [1, 2, 3]
    lm._begin_generation_transaction(["abc"])
    assert lm._generation_token_ids == [1, 2, 3]

    lm._reset_generation_transactions()
    lm._token_id_to_str[4] = "branch"
    lm._generation_token_ids = [1, 4, 2, 3]
    lm._begin_generation_transaction(["abc"])
    assert lm._generation_token_ids == [1, 2, 3]
    assert lm._generation_alignment_removed_token_ids == [4]


def test_spider_rollback_and_continue_nested_lifecycle_preserves_stable_ids(
    _verified_csd_helpers,
):
    """Nested complete rollback must flush the outer full-prefix occurrence history."""
    import sys
    import types

    from synthesis.evaluate.benchmarks.common.model_utils import _TensorizedLMBase
    from synthesis.evaluate.benchmarks.gsm_symbolic.environment import (
        _attach_helper_trace,
    )
    from synthesis.evaluate.benchmarks.gsm_symbolic.generation import (
        _finalize_spider_generation_evidence,
    )
    from synthesis.evaluate.benchmarks.sql_spider.output_contract import (
        SpiderEvidenceContractError,
    )
    from synthesis.evaluate.benchmarks.sql_spider.prompts import SpiderPromptParts

    class Dafny:
        @staticmethod
        def Seq(value):
            return value

        @staticmethod
        def SeqWithoutIsStrInference(values):
            return list(values)

    class Tokenizer:
        all_special_ids = {99}
        eos_token_id = 99
        eos_token = "eos"

        def decode(self, token_ids, skip_special_tokens=False):
            del skip_special_tokens
            return "".join({1: "a", 2: "b", 3: "c", 99: "eos"}[int(token_id)] for token_id in token_ids)

        def encode(self, text, add_special_tokens=False):
            del add_special_tokens
            return {"a": [1], "b": [2], "c": [3], "eos": [99]}.get(text, [])

    class NestedParser:
        def IsCompletePrefix(self, prefix):
            return list(prefix) == ["b"]

        def IsValidPrefix(self, prefix):
            return True

        def IsDeadPrefix(self, prefix):
            return False

    lm = _TensorizedLMBase(Dafny(), Tokenizer(), ["a", "b", "c", "eos"], [1, 2, 3, 99])
    lm.Tokens = lm._Tokens
    lm._structured_prompt = SpiderPromptParts(
        "db_id: x\nquestion: q\n", model_name="Qwen/Qwen2.5-7B-Instruct"
    )
    lm._generation_stop_token_ids = {99}

    def generate_logits(self, prefix):
        begin_transaction = getattr(self, "_begin_generation_transaction", None)
        if callable(begin_transaction):
            begin_transaction(prefix)

    lm.GenerateLogits = types.MethodType(generate_logits, lm)
    lm.MaskValidNextAndEos = types.MethodType(lambda self, *args: None, lm)
    lm.ChooseNextToken = types.MethodType(lambda self: "eos", lm)

    helper = _verified_csd_helpers()
    helper.ctor__()
    verified = sys.modules["VerifiedDecoderAgent"]
    trace_state = {"events": []}
    _attach_helper_trace(verified, trace_state)
    lm._record_generated_token_ids([1, 2, 3])

    generated, current = helper.RollbackAndContinue(
        lm,
        NestedParser(),
        ["a"],
        ["a", "b", "c"],
        ["b", "c"],
        "eos",
        2,
        1,
        1,
    )

    assert generated == ["a", "b"]
    assert current == ["b"]
    helper_names = [event["helper"] for event in trace_state["events"]]
    assert helper_names[-3:] == [
        "RollbackToCompletePrefix",
        "DeadEndAvoidingStep",
        "RollbackAndContinue",
    ]
    nested_event = trace_state["events"][-3]
    assert nested_event["generated_len_before"] == 2
    assert nested_event["generated_len_after"] == 1
    outer_event = trace_state["events"][-1]
    assert outer_event["generated_len_before"] == 3
    assert outer_event["generated_len_after"] == 2
    assert outer_event["current_len_before"] == 2
    assert outer_event["current_len_after"] == 1

    pending_prefix = trace_state.pop("_pending_spider_rollback_prefix", None)
    assert pending_prefix == generated
    lm._align_generation_history_to_prefix(pending_prefix)
    try:
        _finalize_spider_generation_evidence(
            lm,
            spider_prompt_active=True,
            scored_output="ab",
            strategy_token_sequence=generated,
        )
    except SpiderEvidenceContractError as exc:
        pytest.fail(f"nested rollback strategy output aborted finalization: {exc}")

    evidence = lm._last_generation_evidence
    assert evidence["raw_token_ids"] == [1, 2]
    assert evidence["raw_decoded_text"] == "ab"
    assert evidence["strategy_removed_sampled_token_ids"] == [3]


def test_helper_trace_preserves_classmethod_cost_transitions():
    """Classmethod wrappers must record pre-call and post-call class cost."""
    import types

    from synthesis.evaluate.benchmarks.gsm_symbolic.environment import (
        _attach_helper_trace,
    )

    class ClassMethodHelpers:
        cost = 0

        @classmethod
        def RollbackToCompletePrefix(cls, parser, generated):
            del parser
            cls.cost += 1
            return list(generated[:-1])

    verified = types.SimpleNamespace(CSDHelpers=ClassMethodHelpers)
    trace_state = {"events": []}
    _attach_helper_trace(verified, trace_state)
    parser = object()

    class_result = ClassMethodHelpers.RollbackToCompletePrefix(parser, ["a", "b"])
    instance_result = ClassMethodHelpers().RollbackToCompletePrefix(parser, ["a", "b"])

    assert class_result == ["a"]
    assert instance_result == ["a"]
    events = [
        event
        for event in trace_state["events"]
        if event["helper"] == "RollbackToCompletePrefix"
    ]
    assert [(event["cost_before"], event["cost_after"]) for event in events] == [
        (0, 1),
        (1, 2),
    ]





def test_spider_crane_budget_exhaustion_flushes_outer_full_prefix(
    _verified_csd_helpers,
    monkeypatch,
):
    """Crane's outer result owns the full rollback alignment after nested rewinds."""
    import json
    import sys
    import types

    import torch

    from synthesis.evaluate.benchmarks.common.model_utils import _TensorizedLMBase
    from synthesis.evaluate.benchmarks.gsm_symbolic.environment import (
        _attach_helper_trace,
    )
    from synthesis.evaluate.benchmarks.gsm_symbolic.generation import (
        _finalize_spider_generation_evidence,
    )
    from synthesis.evaluate.benchmarks.sql_spider.prompts import SpiderPromptParts

    class Dafny:
        @staticmethod
        def Seq(value):
            return list(value) if isinstance(value, str) else value

        @staticmethod
        def SeqWithoutIsStrInference(values):
            return list(values)

        @staticmethod
        def CodePoint(value):
            return value

    class Tokenizer:
        all_special_ids = {99}
        eos_token_id = 99
        eos_token = "eos"

        def decode(self, token_ids, skip_special_tokens=False):
            del skip_special_tokens
            pieces = {1: "<<", 2: "bad", 99: "eos"}
            return "".join(pieces[int(token_id)] for token_id in token_ids)

        def encode(self, text, add_special_tokens=False):
            del add_special_tokens
            return {"<<": [1], "bad": [2], "eos": [99]}.get(text, [])

    class CraneParser:
        def IsValidPrefix(self, prefix):
            del prefix
            return True

        def IsDeadPrefix(self, prefix):
            del prefix
            return False

        def CompletedSymbolCount(self, prefix, unit, baseline=0):
            del unit, baseline
            return 1 if prefix else 0

        def SymbolStartTokenIndex(self, prefix, unit, index):
            del prefix, unit, index
            return 0

    lm = _TensorizedLMBase(
        Dafny(), Tokenizer(), [list("<<"), list("bad"), list("eos")], [1, 2, 99]
    )
    lm.Tokens = lm._Tokens
    lm._structured_prompt = SpiderPromptParts(
        "db_id: x\nquestion: q\n", model_name="Qwen/Qwen2.5-7B-Instruct"
    )
    lm._generation_stop_token_ids = {99}

    helpers_cls = _verified_csd_helpers()
    helper = helpers_cls() if isinstance(helpers_cls, type) else helpers_cls
    helper.ctor__()
    helper_type = type(helper)

    def unconstrained_step(self, lm_arg, prompt, generated):
        del prompt, generated
        lm_arg._record_generated_token_ids([1])
        self.cost += 1
        return list("<<")

    def forward_until_symbol(
        self,
        lm_arg,
        parser,
        prompt,
        cur,
        eos_token,
        unit,
        num,
        budget,
    ):
        del parser, prompt, cur, eos_token, unit, num, budget
        lm_arg._record_generated_token_ids([2])
        self.cost += 1
        return [list("bad")]

    def view_last_symbol(self, parser, cur, unit):
        del parser, cur, unit
        return list("bad")

    def is_allowed_var_text(self, groups, text):
        del groups, text
        return False

    monkeypatch.setattr(
        helper_type,
        "UnconstrainedStep",
        unconstrained_step,
    )
    monkeypatch.setattr(
        helper_type,
        "ForwardUntilSymbol",
        forward_until_symbol,
    )
    monkeypatch.setattr(helper_type, "ViewLastSymbol", view_last_symbol)
    monkeypatch.setattr(helper_type, "IsAllowedVarText", is_allowed_var_text)

    verified = sys.modules["VerifiedDecoderAgent"]
    monkeypatch.setattr(
        verified.default__,
        "Contains",
        staticmethod(
            lambda value, needle: any(
                list(value)[index : index + len(needle)] == list(needle)
                for index in range(max(0, len(value) - len(needle) + 1))
            )
        ),
    )
    monkeypatch.setattr(
        verified.default__,
        "RenderedEndsWith",
        staticmethod(
            lambda value, suffix: len(value) >= len(suffix)
            and list(value)[-len(suffix) :] == list(suffix),
        ),
    )

    trace_state = {"events": []}
    _attach_helper_trace(verified, trace_state)

    result = helper.CraneGeneration(
        lm,
        CraneParser(),
        [],
        3,
        0,
        [],
        list("eos"),
    )
    strategy_text = "".join("".join(token) for token in result)
    assert strategy_text == "<<"

    pending = trace_state.pop("_pending_spider_rollback_prefix", None)
    if pending is not None:
        lm._align_generation_history_to_prefix(pending)
    _finalize_spider_generation_evidence(
        lm,
        spider_prompt_active=True,
        scored_output=strategy_text,
        strategy_token_sequence=result,
    )
    evidence = lm._last_generation_evidence
    event_names = [event["helper"] for event in trace_state["events"]]
    public_trace = json.dumps(trace_state["events"], default=str)

    assert {
        "outer_pending_matches": pending == result,
        "event_suffix": event_names[-2:],
        "cost": helper.cost,
        "raw_token_ids": evidence["raw_token_ids"],
        "removed_sampled_ids": evidence["strategy_removed_sampled_token_ids"],
        "alignment_removed": getattr(lm, "_generation_alignment_removed_token_ids", None),
        "trace_has_backward": "BackwardToSymbol" in event_names,
        "trace_has_raw_body": any(
            leaked in public_trace.lower() for leaked in ("bad", "select", "sql")
        ),
    } == {
        "outer_pending_matches": True,
        "event_suffix": ["BackwardToSymbol", "CraneGeneration"],
        "cost": 3,
        "raw_token_ids": [1],
        "removed_sampled_ids": [2, 2],
        "alignment_removed": [2, 2],
        "trace_has_backward": True,
        "trace_has_raw_body": False,
    }
