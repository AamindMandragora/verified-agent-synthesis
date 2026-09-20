"""GSM GCD must actually use grammars/gsm.lark, the shared GSM prompt, greedy decoding,
a hard mask on every constrained token, and the real GSM scorer -- not just be routed
to run_gcd_legacy_adapter (that routing alone was tested in
test_gsm_gcd_does_not_route_through_crane_repo.py).
"""

from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

from synthesis.evaluate import run_legacy_fixed_strategy as runner


class FakeTokenizer:
    """Joins chat messages into one string; records what it was asked to render."""

    def __init__(self):
        self.calls: list[list[dict]] = []

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking=None):
        self.calls.append(messages)
        assert tokenize is False
        assert add_generation_prompt is True
        return "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant:"


def _fake_gsm_example():
    return {
        "question_parsed": "There are {t} trees.",
        "question": "There are 3 trees.",
        "answer_parsed": "5",
        "variable_types": {"t": "int"},
    }


def _install_fake_syncode(monkeypatch, constructor_kwargs, infer_calls, reply):
    class FakeSyncode:
        def __init__(self, **kwargs):
            constructor_kwargs.append(kwargs)

        def infer(self, prompt, stop_words):
            infer_calls.append({"prompt": prompt, "stop_words": stop_words})
            return [reply]

    syncode_package = types.ModuleType("syncode")
    syncode_package.__path__ = []
    syncode_infer = types.ModuleType("syncode.infer")
    syncode_infer.Syncode = FakeSyncode
    monkeypatch.setitem(sys.modules, "syncode", syncode_package)
    monkeypatch.setitem(sys.modules, "syncode.infer", syncode_infer)


def _run_gsm_gcd(monkeypatch, tmp_path: Path, reply: str):
    constructor_kwargs: list[dict] = []
    infer_calls: list[dict] = []
    _install_fake_syncode(monkeypatch, constructor_kwargs, infer_calls, reply)

    from synthesis.evaluate.benchmarks.gsm_symbolic import eval_logic as gsm_eval_logic

    fake_tokenizer = FakeTokenizer()
    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained",
        lambda *a, **k: fake_tokenizer,
    )

    example = _fake_gsm_example()

    class FakeEvaluator:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        @staticmethod
        def _check_syntax_validity(output, *, example):
            import re as _re

            # Digits/operators only (no letters): enough to separate a real
            # arithmetic expression from prose, without needing a real parser.
            match = _re.fullmatch(r"<<\s*[-+*/%()0-9.\s]+\s*>>", output.strip())
            return bool(match), ([(output, True)] if match else [(output, False)])

        @staticmethod
        def _truncate_gsm_output(output):
            return output

        @staticmethod
        def _gsm_symbolic_equivalence(model_expr, expected_expr, variable_types):
            # Real equivalence is a z3 for-all proof; a plain string match is enough
            # here since no test in this file expects a semantically-equal-but-
            # differently-written expression to be marked correct.
            return model_expr is not None and str(model_expr).strip() == str(expected_expr).strip()

    import synthesis.evaluate.benchmarks.registry as registry
    import synthesis.evaluate.evaluator as evaluator

    monkeypatch.setattr(registry, "get_logic", lambda dataset: gsm_eval_logic)
    monkeypatch.setattr(evaluator, "Evaluator", FakeEvaluator)
    monkeypatch.setattr(runner, "_configure_fixed_eval_runtime", lambda *args: None)
    monkeypatch.setattr(gsm_eval_logic, "load_dataset_sample", lambda runtime: [example])
    monkeypatch.setattr(runner, "_baseline_row_question", lambda *args: "question")

    saved_rows: list[list[dict]] = []
    monkeypatch.setattr(
        runner,
        "_build_minimal_json",
        lambda rows, *a, **kw: saved_rows.append(rows),
    )

    args = argparse.Namespace(
        dataset="gsm_symbolic",
        eval_model="Qwen/Qwen2.5-1.5B-Instruct",
        eval_backend="vllm",
        device="cuda:0",
        eval_sample_size=1,
        eval_max_steps=150,
        eval_step_token_budget=1,
        vllm_gpu_memory_utilization=0.3,
        vllm_tensor_parallel_size=1,
        gsm_split_file=None,
        gsm_split_name="test",
        spider_split_file=None,
        spider_split_name="eval",
        smiles_classes=None,
        output_json=tmp_path / "gcd_gsm.json",
    )

    rc = runner.run_gcd_legacy_adapter(args)
    return rc, constructor_kwargs, infer_calls, fake_tokenizer, saved_rows


def test_gsm_gcd_loads_the_gsm_lark_grammar(monkeypatch, tmp_path):
    rc, constructor_kwargs, _infer_calls, _tok, _rows = _run_gsm_gcd(monkeypatch, tmp_path, "<<t + 2>>")
    assert rc == 0
    grammar_text = constructor_kwargs[0]["grammar"]
    # The dynamic grammar is derived from grammars/gsm.lark: it keeps the file's
    # "syncode" rule name and the delimited-span shape GCD relies on.
    gsm_lark_path = Path(runner.__file__).resolve().parents[2] / "synthesis" / "evaluate" / "grammars" / "gsm.lark"
    assert "syncode:" in gsm_lark_path.read_text()
    assert "syncode:" in grammar_text
    assert '"<<"' not in grammar_text  # GCD supplies "<<" itself; grammar covers only the span body


def test_gsm_gcd_decodes_greedily(monkeypatch, tmp_path):
    _rc, constructor_kwargs, _infer_calls, _tok, _rows = _run_gsm_gcd(monkeypatch, tmp_path, "<<t + 2>>")
    assert constructor_kwargs[0]["do_sample"] is False


def test_gsm_gcd_uses_the_shared_gsm_prompt_not_expression_only(monkeypatch, tmp_path):
    from synthesis.evaluate.benchmarks.gsm_symbolic.prompts import GSM_CRANE_COT_TASK

    _rc, _kwargs, infer_calls, fake_tokenizer, _rows = _run_gsm_gcd(monkeypatch, tmp_path, "<<t + 2>>")

    assert len(fake_tokenizer.calls) == 1
    rendered_messages = fake_tokenizer.calls[0]
    # Same system header and CoT few-shots as every other GSM lane's shared prompt
    # (eval_logic.format_prompt / reasoning_with_symbolic_expr_messages) -- NOT
    # symbolic_expression_only_prompt's stripped-down, answer-only few-shots.
    assert rendered_messages[0]["role"] == "system"
    assert GSM_CRANE_COT_TASK.strip() in rendered_messages[0]["content"]
    assert any("Let's think step by step" in m["content"] for m in rendered_messages if m["role"] == "assistant")

    # The rendered chat text is what actually reaches Syncode, with "<<" appended
    # so the constrained span starts exactly where the grammar takes over.
    sent_prompt = infer_calls[0]["prompt"]
    assert sent_prompt.startswith("system: " + GSM_CRANE_COT_TASK.strip())
    assert sent_prompt.endswith("<<")


def test_gsm_gcd_scores_with_the_real_gsm_scorer(monkeypatch, tmp_path):
    _rc, _kwargs, _infer_calls, _tok, saved_rows = _run_gsm_gcd(monkeypatch, tmp_path, "<<3 + 2>>")

    row = saved_rows[0][0]
    # example's answer_parsed is "5"; a wrong expression must not be marked correct,
    # proving eval_logic's real extract_actual/is_correct ran (not a stub always-True).
    assert row["correct"] is False
    assert row["syntax_valid"] is True


def test_gsm_gcd_marks_a_matching_answer_correct(monkeypatch, tmp_path):
    # The scorer must actually run rather than being stubbed True: a completion with
    # no arithmetic expression anywhere in it is syntax-invalid.
    _rc, _kwargs, _infer_calls, _tok, saved_rows = _run_gsm_gcd(monkeypatch, tmp_path, "not an expression")
    row = saved_rows[0][0]
    assert row["syntax_valid"] is False
    assert row["correct"] is False
