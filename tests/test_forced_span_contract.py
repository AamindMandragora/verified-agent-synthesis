"""The span contract every benchmark now shares.

There is no hidden-span mode any more. Spider and SMILES used to start
generation already inside a constrained region with no `<<` in the output;
now the runtime opens the span itself by seeding the output with a literal
`<<`, and closes it with `>>` when the strategy hands back a span whose
content is a complete parse. GSM was always like this.

What that buys, and what these tests pin down:

  * the entry state is the same object for every task: output `["<<"]`,
    inside-constrained true, empty current-span;
  * the closer is a decision, not a habit -- it is appended only when the
    span content parses as complete;
  * the syntax metric reads spans out of the OUTPUT TEXT for every task
    (walk_spans), never out of runtime bookkeeping;
  * scoring reads the span content, so `<<X>>` scores exactly like `X`;
  * the Spider token-evidence check does not fail closed just because the
    delimiters were inserted by the runtime instead of sampled.
"""

from __future__ import annotations

import pytest


# --------------------------------------------------------------------------
# Fakes: the smallest thing that can stand in for the compiled Dafny strategy.
# --------------------------------------------------------------------------


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

    def apply_chat_template(self, messages, **kwargs):
        del messages, kwargs
        return "<chat-rendered>"

    def decode(self, token_ids, skip_special_tokens=False):
        del skip_special_tokens
        return "".join("<eos>" if int(t) == 99 else "ok" for t in token_ids)

    def encode(self, text, add_special_tokens=False):
        del text, add_special_tokens
        return []


class _Parser:
    """`complete_texts` are the span bodies this parser calls complete."""

    def __init__(self, complete_texts=()):
        self._complete = set(complete_texts)

    def IsCompletePrefix(self, prefix):
        return "".join(str(t) for t in prefix) in self._complete

    def is_complete(self, text):
        return text in self._complete


def _strategy_returning(output_tokens, inside, current_span):
    """A compiled-strategy stand-in that also records how it was entered."""
    seen = {}

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
            del lm_arg, parser, seq0, max_steps, step_budget, eos_token
            seen["generated_prefix"] = list(generated_prefix)
            seen["inside"] = start_inside
            seen["current_constrained"] = list(current_constrained)
            return (list(output_tokens), inside, list(current_span), 1)

    class GeneratedCSD:
        default__ = GeneratedDefault

    return GeneratedCSD, seen


def _env(generated_csd, parser):
    from synthesis.evaluate.benchmarks.common.model_utils import _TensorizedLMBase

    lm = _TensorizedLMBase(Dafny(), Tokenizer(), ["ok"], [1])
    return {
        "_dafny": Dafny,
        "GeneratedCSD": generated_csd,
        "lm": lm,
        "parser": parser,
        "model_name": "Qwen/Qwen2.5-7B-Instruct",
    }


def _run(env, *, force_open_span, max_steps=8):
    from synthesis.evaluate.benchmarks.gsm_symbolic.generation import run_crane_csd

    return run_crane_csd(
        env=env,
        prompt_text="question",
        max_steps=max_steps,
        grammar_file=None,
        force_open_span=force_open_span,
    )


# --------------------------------------------------------------------------
# (1) Entry state
# --------------------------------------------------------------------------


@pytest.mark.parametrize("dataset", ["spider", "smiles"])
def test_spider_and_smiles_runners_force_the_span_open(dataset, monkeypatch):
    from synthesis.evaluate.benchmarks.registry import get_logic

    logic = get_logic(dataset)
    seen = {}

    module_name = {
        "spider": "synthesis.evaluate.benchmarks.sql_spider.generation",
        "smiles": "synthesis.evaluate.benchmarks.smiles.generation",
    }[dataset]
    import importlib

    module = importlib.import_module(module_name)

    def fake(*args, **kwargs):
        seen.update(kwargs)
        return "", 0, 0.0, [], []

    monkeypatch.setattr(module, "run_crane_csd", fake, raising=False)
    monkeypatch.setattr(module, "_run_crane_csd", fake, raising=False)

    logic.get_generation_runner()(env={}, prompt_text="p", max_steps=1, grammar_file=None)

    assert seen["force_open_span"] is True


def test_gsm_runner_does_not_force_the_span_open():
    from synthesis.evaluate.benchmarks.gsm_symbolic import eval_logic, generation

    assert eval_logic.get_generation_runner() is generation.run_crane_csd


def test_forced_open_span_entry_state_is_open_bracket_inside_and_empty():
    generated_csd, seen = _strategy_returning(["<<", "ok"], False, [])
    _run(_env(generated_csd, _Parser()), force_open_span=True)

    assert seen["generated_prefix"] == ["<<"]
    assert seen["inside"] is True
    assert seen["current_constrained"] == []


def test_unforced_entry_state_starts_with_an_empty_output_and_outside():
    generated_csd, seen = _strategy_returning(["ok"], False, [])
    _run(_env(generated_csd, _Parser()), force_open_span=False)

    assert seen["generated_prefix"] == []
    assert seen["inside"] is False
    assert seen["current_constrained"] == []


# --------------------------------------------------------------------------
# (2) The runtime never writes the closer; the strategy (verified template) does
# --------------------------------------------------------------------------


def test_runtime_never_appends_a_closer_of_its_own():
    # The closer is the verified template's job (it holds one step back for it), so the Python
    # runtime must not add tokens after the strategy returns: that would break the step cap.
    generated_csd, _ = _strategy_returning(
        ["<<", "SELECT 1"], True, ["SELECT 1"]
    )
    parser = _Parser(complete_texts={"SELECT 1"})
    output_text = _run(_env(generated_csd, parser), force_open_span=True)[0]

    assert output_text == "<<SELECT 1"


def test_runtime_leaves_an_incomplete_span_open():
    generated_csd, _ = _strategy_returning(["<<", "SELECT"], True, ["SELECT"])
    parser = _Parser(complete_texts={"SELECT 1"})
    output_text = _run(_env(generated_csd, parser), force_open_span=True)[0]

    assert output_text == "<<SELECT"


def test_runtime_does_not_double_close_a_span_the_strategy_already_closed():
    generated_csd, _ = _strategy_returning(
        ["<<", "SELECT 1", ">>"], False, []
    )
    parser = _Parser(complete_texts={"SELECT 1"})
    output_text = _run(_env(generated_csd, parser), force_open_span=True)[0]

    assert output_text == "<<SELECT 1>>"


# --------------------------------------------------------------------------
# (3) Segments come from the output text, for every task
# --------------------------------------------------------------------------


def test_generation_returns_no_runtime_invented_segments():
    generated_csd, _ = _strategy_returning(["<<", "SELECT 1"], True, ["SELECT 1"])
    parser = _Parser(complete_texts={"SELECT 1"})
    segments = _run(_env(generated_csd, parser), force_open_span=True)[3]

    assert segments == []


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("<<SELECT 1>>", ["SELECT 1"]),
        ("reason <<a>> then <<b>>", ["a", "b"]),
        ("<<unclosed", []),
        ("no span here", []),
    ],
)
def test_constrained_content_is_the_closed_spans_of_the_text(text, expected):
    from synthesis.evaluate.benchmarks.common.delimiter_hygiene import walk_spans
    from synthesis.evaluate.evaluator import Evaluator

    extracted = Evaluator._extract_constrained_content(None, text)
    assert extracted == expected
    assert extracted == [
        span.content.strip() for span in walk_spans(text).spans if span.closed
    ]


def test_walk_spans_has_no_start_inside_parameter():
    import inspect

    from synthesis.evaluate.benchmarks.common.delimiter_hygiene import walk_spans

    assert list(inspect.signature(walk_spans).parameters) == ["text"]


# --------------------------------------------------------------------------
# (4) Scoring reads the span content
# --------------------------------------------------------------------------


class _SyntaxOnlyEvaluator:
    """Enough of an Evaluator for Spider's extract_actual."""

    def _get_syntax_parser(self, example):
        del example
        return None


@pytest.mark.parametrize(
    "sql",
    ["SELECT name FROM singer", "SELECT count(*) FROM singer WHERE age > 30"],
)
def test_spider_scores_a_delimited_output_exactly_like_the_bare_content(sql):
    from synthesis.evaluate.benchmarks.sql_spider import eval_logic

    evaluator = _SyntaxOnlyEvaluator()
    example = {"db_id": "concert_singer"}

    delimited = eval_logic.extract_actual(evaluator, f"<<{sql}>>", example)
    bare = eval_logic.extract_actual(evaluator, sql, example)

    assert delimited[0] == sql
    assert delimited[0] == bare[0]


@pytest.mark.parametrize("smiles", ["CC(=O)OC=C", "CCOC(=O)C=C"])
def test_smiles_scores_a_delimited_output_exactly_like_the_bare_content(smiles):
    from synthesis.evaluate.benchmarks.smiles.metrics import clean_smiles_output

    assert clean_smiles_output(f"<<{smiles}>>") == clean_smiles_output(smiles)
    assert clean_smiles_output(f"<<{smiles}>>") == smiles


# --------------------------------------------------------------------------
# (5) The Spider evidence check survives runtime-inserted delimiters
# --------------------------------------------------------------------------


def _lm_with_committed_evidence():
    from synthesis.evaluate.benchmarks.common.model_utils import _TensorizedLMBase
    from synthesis.evaluate.benchmarks.sql_spider.prompts import SpiderPromptParts

    lm = _TensorizedLMBase(Dafny(), Tokenizer(), ["ok"], [1])
    lm._structured_prompt = SpiderPromptParts("db_id: x\nquestion: q\n", answer_cue="SQL:")
    lm._generation_stop_token_ids = frozenset()
    lm._generation_token_ids = [1]  # decodes to "ok"
    return lm


def test_evidence_check_accepts_runtime_inserted_delimiters():
    lm = _lm_with_committed_evidence()
    lm.SetForcedSpanDelimiters(opener="<<", closer=">>")

    assert lm._reconcile_generation_evidence("<<ok>>") is True


def test_evidence_check_accepts_a_span_the_runtime_opened_but_never_closed():
    lm = _lm_with_committed_evidence()
    lm.SetForcedSpanDelimiters(opener="<<", closer=">>")

    assert lm._reconcile_generation_evidence("<<ok") is True


def test_evidence_check_still_rejects_text_the_model_never_produced():
    lm = _lm_with_committed_evidence()
    lm.SetForcedSpanDelimiters(opener="<<", closer=">>")

    assert lm._reconcile_generation_evidence("<<not what was sampled>>") is False


def test_evidence_check_rejects_delimiters_when_none_were_forced():
    lm = _lm_with_committed_evidence()

    assert lm._reconcile_generation_evidence("<<ok>>") is False
    assert lm._reconcile_generation_evidence("ok") is True


# --------------------------------------------------------------------------
# (6) The removed mode leaves nothing behind
# --------------------------------------------------------------------------


@pytest.mark.parametrize("dataset", ["gsm_symbolic", "spider", "smiles"])
def test_no_benchmark_declares_a_hidden_or_delimiterless_surface(dataset):
    from synthesis.evaluate.benchmarks.registry import get_logic

    logic = get_logic(dataset)
    for gone in (
        "uses_hidden_chunks",
        "emits_visible_delimiters",
        "starts_inside_constrained",
        "_token0_enabled",
    ):
        assert not hasattr(logic, gone), f"{dataset}.{gone} should have been removed"


def test_require_delimiters_follows_the_cli_for_every_benchmark():
    from synthesis.evaluate.benchmarks.registry import resolve_require_delimiters

    for dataset in ("gsm_symbolic", "spider", "smiles"):
        assert resolve_require_delimiters(dataset, cli_value=True) is True
        assert resolve_require_delimiters(dataset, cli_value=False) is False


def test_forced_open_run_registers_the_closer_up_front():
    # A strategy that closes its own span writes ">>" without sampling it. The evidence check
    # must already know that closer, or every strategy-closed Spider output fails closed.
    generated_csd, _ = _strategy_returning(["<<", "SELECT 1", ">>"], False, [])
    env = _env(generated_csd, _Parser(complete_texts={"SELECT 1"}))
    _run(env, force_open_span=True)

    assert env["lm"]._forced_span_open_text == "<<"
    assert env["lm"]._forced_span_close_text == ">>"


def test_early_entry_complaint_is_not_raised_when_the_runtime_forced_the_opener():
    from synthesis.evaluate.evaluator import EvaluationResult

    result = EvaluationResult.__new__(EvaluationResult)
    forced = {"full_output": "<<SELECT 1>>", "actual": "SELECT 1", "span_forced_open": True,
              "contains_delimiters": True, "num_visible_spans": 1, "num_valid_visible_spans": 1}
    chosen = dict(forced, span_forced_open=False)
    assert "entered_constrained_mode_too_early" not in [m for m, _ in result._classify_sample_failure_modes(forced)]
    assert "entered_constrained_mode_too_early" in [m for m, _ in result._classify_sample_failure_modes(chosen)]
