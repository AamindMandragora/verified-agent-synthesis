"""Each baseline must leave enough in its log for a reader to tell, from the log
alone, whether the step that defines it actually ran: IterGen's forward/check/backward,
CRANE's mode switches, GCD's mask on every token, CARS's rejection loop, and the
absence of any grammar object in the unconstrained lanes.

All of it is off unless CSD_TRACE_LOG is set, because the harness never attached a
log handler before and every LOGGER.info call was silently dropped.
"""

from __future__ import annotations

import argparse
import logging
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from synthesis.evaluate import run_legacy_fixed_strategy as runner
from synthesis.evaluate.benchmarks.common.constraint_audit import ConstraintAudit
from synthesis.evaluate.benchmarks.common.generation_trace import describe_generation_end
from tests import test_cars_settings as cars_fixture
from tests import test_gsm_gcd_full_path as gcd_fixture

TRACE_LOGGER = "csd.baseline_trace"


def _messages(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.name == TRACE_LOGGER]


def _only(messages: list[str], needle: str) -> str:
    hits = [m for m in messages if needle in m]
    assert hits, f"no log line containing {needle!r} in:\n" + "\n".join(messages)
    return hits[0]


# ---- the switch ----------------------------------------------------------------


def test_trace_log_env_var_makes_info_lines_reach_stdout(monkeypatch, capsys):
    monkeypatch.setenv("CSD_TRACE_LOG", "1")
    csd_logger = logging.getLogger("csd")
    before = list(csd_logger.handlers)
    old_level = csd_logger.level
    try:
        runner._configure_trace_logging()
        runner.LOGGER.info("[legacy-test] hello %d", 7)
        out = capsys.readouterr().out
        assert "[legacy-test] hello 7" in out
        assert TRACE_LOGGER in out
    finally:
        for handler in list(csd_logger.handlers):
            if handler not in before:
                csd_logger.removeHandler(handler)
        csd_logger.setLevel(old_level)


def test_trace_log_stays_silent_without_the_env_var(monkeypatch, capsys):
    monkeypatch.delenv("CSD_TRACE_LOG", raising=False)
    before = list(logging.getLogger("csd").handlers)
    runner._configure_trace_logging()
    assert logging.getLogger("csd").handlers == before
    runner.LOGGER.info("[legacy-test] should not print")
    assert "should not print" not in capsys.readouterr().out


def test_long_text_is_clipped_with_a_marker_that_says_how_much_was_cut():
    clipped = runner._clip("x" * 1000)
    assert clipped.startswith("x" * 300)
    assert len(clipped) < 340
    assert "+700 chars" in clipped
    assert runner._clip("short") == "short"


def test_lane_header_logs_the_whole_prompt_end_tokens_and_budget(caplog):
    caplog.set_level(logging.INFO, logger=TRACE_LOGGER)
    prompt = "P" * 5000
    runner._log_lane_header(
        "legacy-gcd", "spider", prompt=prompt, end_token_ids=[7, 9], max_new_tokens=150
    )
    line = _only(_messages(caplog), "lane-header")
    assert "[legacy-gcd]" in line and "dataset=spider" in line
    assert "end_token_ids=[7, 9]" in line
    assert "max_new_tokens=150" in line
    assert prompt in line


# ---- why a generation ended ----------------------------------------------------


def test_generation_end_names_the_end_token():
    end = describe_generation_end(
        generated_token_ids=[5, 6, 2], stop_token_ids={2}, max_new_tokens=10,
        completion="ab", stop_words=None,
    )
    assert end == {"stop_reason": "end_token", "generated_tokens": 3, "last_token_id": 2}


def test_generation_end_names_the_token_budget():
    end = describe_generation_end(
        generated_token_ids=[5, 6, 7], stop_token_ids={2}, max_new_tokens=3,
        completion="abc", stop_words=[";"],
    )
    assert end["stop_reason"] == "max_new_tokens"


def test_generation_end_names_the_stop_word():
    end = describe_generation_end(
        generated_token_ids=[5, 6], stop_token_ids={2}, max_new_tokens=10,
        completion="SELECT 1;", stop_words=[";"],
    )
    assert end["stop_reason"] == "stop_word"


def test_generation_end_says_unknown_rather_than_guessing():
    end = describe_generation_end(
        generated_token_ids=[5, 6], stop_token_ids={2}, max_new_tokens=10,
        completion="ab", stop_words=None,
    )
    assert end["stop_reason"] == "unknown"


def test_audit_counts_masked_steps_next_to_unmasked_ones():
    audit = ConstraintAudit()
    audit.record_masked_step()
    audit.record_masked_step()
    audit.record_unconstrained_step("parse_error", "boom")
    assert audit.total_masked_steps == 2
    assert audit.total_unconstrained_steps == 1


def test_the_vendored_decoder_feeds_the_trace():
    """torch-free wiring check, same approach as test_unconstrained_fallback_is_recorded."""
    syncode = Path(runner.__file__).resolve().parent / "syncode" / "syncode"
    decoder = (syncode / "grammar_decoder.py").read_text()
    assert "self.constraint_audit.record_masked_step()" in decoder
    model = (syncode / "language_model.py").read_text()
    assert "describe_generation_end(" in model
    assert "self.last_generation_trace" in model
    assert model.count("self._mode_switches.append(") == 2  # into a span, and back out


# ---- GCD -----------------------------------------------------------------------


def _install_traced_fake_syncode(audit, trace):
    def install(monkeypatch, constructor_kwargs, infer_calls, reply):
        class FakeSyncode:
            def __init__(self, **kwargs):
                constructor_kwargs.append(kwargs)
                self.grammar_decoder = SimpleNamespace(constraint_audit=audit)
                self.model = SimpleNamespace(_stop_token_ids={11, 12}, last_generation_trace=None)

            def infer(self, prompt, stop_words):
                infer_calls.append({"prompt": prompt, "stop_words": stop_words})
                for _ in range(trace["generated_tokens"]):
                    audit.record_masked_step()
                self.model.last_generation_trace = dict(trace)
                return [reply]

        package = types.ModuleType("syncode")
        package.__path__ = []
        infer_module = types.ModuleType("syncode.infer")
        infer_module.Syncode = FakeSyncode
        monkeypatch.setitem(sys.modules, "syncode", package)
        monkeypatch.setitem(sys.modules, "syncode.infer", infer_module)

    return install


def test_gcd_logs_masked_token_count_settings_and_stop_reason(monkeypatch, tmp_path, caplog):
    caplog.set_level(logging.INFO, logger=TRACE_LOGGER)
    trace = {"stop_reason": "stop_word", "generated_tokens": 4, "last_token_id": 99, "mode_switches": []}
    monkeypatch.setattr(
        gcd_fixture, "_install_fake_syncode", _install_traced_fake_syncode(ConstraintAudit(), trace)
    )
    gcd_fixture._run_gsm_gcd(monkeypatch, tmp_path, "<<3 + 2>>")
    messages = _messages(caplog)

    settings = _only(messages, "[legacy-gcd] settings")
    assert "do_sample=False" in settings and "max_new_tokens=96" in settings
    assert "stop_words=['>>']" in settings

    header = _only(messages, "[legacy-gcd] lane-header")
    assert "end_token_ids=[11, 12]" in header and "max_new_tokens=96" in header

    example = _only(messages, "[legacy-gcd] example=0")
    assert "generated_tokens=4" in example
    assert "masked_steps=4" in example
    assert "unmasked_steps=0" in example
    assert "every_token_masked=True" in example
    assert "stop_reason=stop_word" in example


def test_gcd_log_exposes_a_token_that_skipped_the_mask(monkeypatch, tmp_path, caplog):
    caplog.set_level(logging.INFO, logger=TRACE_LOGGER)
    audit = ConstraintAudit()
    trace = {"stop_reason": "end_token", "generated_tokens": 3, "last_token_id": 11, "mode_switches": []}
    install = _install_traced_fake_syncode(audit, trace)

    def install_with_a_leak(monkeypatch, constructor_kwargs, infer_calls, reply):
        install(monkeypatch, constructor_kwargs, infer_calls, reply)
        fake_cls = sys.modules["syncode.infer"].Syncode
        original_infer = fake_cls.infer

        def leaky_infer(self, prompt, stop_words):
            audit.record_unconstrained_step("parse_error", "boom")
            return original_infer(self, prompt, stop_words)

        fake_cls.infer = leaky_infer

    monkeypatch.setattr(gcd_fixture, "_install_fake_syncode", install_with_a_leak)
    gcd_fixture._run_gsm_gcd(monkeypatch, tmp_path, "<<3 + 2>>")
    example = _only(_messages(caplog), "[legacy-gcd] example=0")
    assert "unmasked_steps=1" in example
    assert "every_token_masked=False" in example
    assert "parse_error" in example


# ---- CRANE ---------------------------------------------------------------------


def test_crane_logs_switches_spans_stop_reason_and_the_audit(monkeypatch, tmp_path, caplog):
    caplog.set_level(logging.INFO, logger=TRACE_LOGGER)
    audit = ConstraintAudit()
    trace = {
        "stop_reason": "stop_word",
        "generated_tokens": 20,
        "last_token_id": 99,
        "constrained_steps": 6,
        "free_steps": 14,
        "mode_switches": [
            {"to": "constrained", "token_index": 9},
            {"to": "unconstrained", "token_index": 15},
        ],
    }
    reply = "Think. <<1 + 1>> no wait <<3 + 2>>"

    def install(monkeypatch, constructor_kwargs, infer_calls, _reply):
        class FakeAdaptive:
            def __init__(self, **kwargs):
                constructor_kwargs.append(kwargs)
                self.grammar_decoder = SimpleNamespace(constraint_audit=audit)
                self.model = SimpleNamespace(_stop_token_ids={11}, last_generation_trace=None)

            def infer(self, prompt, stop_words):
                audit.record_unconstrained_step("no_valid_tokens", "tail")
                self.model.last_generation_trace = dict(trace)
                return [reply]

        package = types.ModuleType("syncode")
        package.__path__ = []
        infer_module = types.ModuleType("syncode.infer")
        infer_module.AdaptiveSynCode = FakeAdaptive
        monkeypatch.setitem(sys.modules, "syncode", package)
        monkeypatch.setitem(sys.modules, "syncode.infer", infer_module)

    monkeypatch.setattr(gcd_fixture, "_install_fake_syncode", install)
    monkeypatch.setattr(
        runner, "run_gcd_legacy_adapter", lambda args: runner._crane_via_adaptive_syncode(args, "gsm_symbolic")
    )
    gcd_fixture._run_gsm_gcd(monkeypatch, tmp_path, reply)
    messages = _messages(caplog)

    header = _only(messages, "[legacy-crane] lane-header")
    assert "end_token_ids=[11]" in header and "max_new_tokens=150" in header

    switches = [m for m in messages if "[legacy-crane] switch" in m]
    assert len(switches) == 2
    assert "to=constrained token_index=9" in switches[0]
    assert "to=unconstrained token_index=15" in switches[1]

    example = _only(messages, "[legacy-crane] example=0")
    assert "stop_reason=stop_word" in example
    assert "switches=2" in example
    assert "spans_closed=2" in example
    assert "picked_span=last_closed(index=1)" in example
    assert "'3 + 2'" in example

    audit_line = _only(messages, "[legacy-crane] constraint-audit example=0")
    assert "unmasked_steps=1" in audit_line and "no_valid_tokens" in audit_line


def test_the_crane_audit_line_carries_the_constrained_step_count(monkeypatch, tmp_path, caplog):
    """masked_steps=0 beside constrained_steps=13 reads as "the constraint did nothing".

    It does not mean that. CRANE runs opportunistic masking: on a constrained step the
    decoder checks the model's own pick against the grammar and reaches for the mask
    only when that pick is illegal. masked_steps=0 therefore says every constrained
    token was already legal. A reader can only see that when both numbers are on the
    same line, so the audit line carries the constrained count too.
    """
    caplog.set_level(logging.INFO, logger=TRACE_LOGGER)
    audit = ConstraintAudit()
    trace = {
        "stop_reason": "stop_word",
        "generated_tokens": 20,
        "last_token_id": 99,
        "constrained_steps": 13,
        "free_steps": 7,
        "mode_switches": [],
    }
    reply = "Think. <<1 + 1>>"

    def install(monkeypatch, constructor_kwargs, infer_calls, _reply):
        class FakeAdaptive:
            def __init__(self, **kwargs):
                constructor_kwargs.append(kwargs)
                self.grammar_decoder = SimpleNamespace(constraint_audit=audit)
                self.model = SimpleNamespace(_stop_token_ids={11}, last_generation_trace=None)

            def infer(self, prompt, stop_words):
                self.model.last_generation_trace = dict(trace)
                return [reply]

        package = types.ModuleType("syncode")
        package.__path__ = []
        infer_module = types.ModuleType("syncode.infer")
        infer_module.AdaptiveSynCode = FakeAdaptive
        monkeypatch.setitem(sys.modules, "syncode", package)
        monkeypatch.setitem(sys.modules, "syncode.infer", infer_module)

    monkeypatch.setattr(gcd_fixture, "_install_fake_syncode", install)
    monkeypatch.setattr(
        runner, "run_gcd_legacy_adapter", lambda args: runner._crane_via_adaptive_syncode(args, "gsm_symbolic")
    )
    gcd_fixture._run_gsm_gcd(monkeypatch, tmp_path, reply)
    messages = _messages(caplog)

    audit_line = _only(messages, "[legacy-crane] constraint-audit example=0")
    assert "constrained_steps=13" in audit_line
    assert "masked_steps=0" in audit_line
    assert "unmasked_steps=0" in audit_line


# ---- CARS ----------------------------------------------------------------------


def test_cars_logs_draws_rejections_and_the_10x_cap(monkeypatch, tmp_path, caplog):
    caplog.set_level(logging.INFO, logger=TRACE_LOGGER)
    examples = [{"class_name": "isocyanates", "grammar_text": "", "prompt": "Molecule:"}]
    cars_fixture._run_cars_smiles(
        monkeypatch, tmp_path, examples, molecule_sequence=["N=C=O"], invalid_molecules={"N=C=O"}
    )
    messages = _messages(caplog)

    settings = _only(messages, "[legacy-cars] settings")
    assert "max_new_tokens=256" in settings and "search_steps_cap=200" in settings

    header = _only(messages, "[legacy-cars] lane-header")
    assert "max_new_tokens=256" in header and "Molecule:" in header

    draws = [m for m in messages if "[legacy-cars] draw=" in m]
    assert len(draws) == 10
    assert "tries=1" in draws[0] and "accepted=False" in draws[0]

    summary = _only(messages, "[legacy-cars] class=isocyanates")
    assert "draws=10" in summary and "rejected=10" in summary
    assert "stop_reason=draw_cap_10x" in summary


def test_cars_logs_target_reached(monkeypatch, tmp_path, caplog):
    caplog.set_level(logging.INFO, logger=TRACE_LOGGER)
    examples = [{"class_name": "acrylates", "grammar_text": "", "prompt": "Molecule:"}] * 2
    cars_fixture._run_cars_smiles(
        monkeypatch, tmp_path, [dict(e) for e in examples], molecule_sequence=["C=C", "C=CC"]
    )
    summary = _only(_messages(caplog), "[legacy-cars] class=acrylates")
    assert "draws=2" in summary and "rejected=0" in summary
    assert "stop_reason=target_reached" in summary


def test_cars_try_cap_is_told_apart_from_an_accepted_sample():
    assert runner._cars_draw_stop_reason(tries=200, search_steps_cap=200, produced_text=False) == "try_cap_200"
    assert runner._cars_draw_stop_reason(tries=3, search_steps_cap=200, produced_text=True) == "sample_accepted"


def test_cars_logs_the_sampling_settings_of_the_first_real_generate_call(caplog):
    caplog.set_level(logging.INFO, logger=TRACE_LOGGER)
    calls = []

    class FakeHFModel:
        generation_config = SimpleNamespace(temperature=0.7, top_p=0.8, top_k=20)

        def generate(self, input_ids, **kwargs):
            calls.append(kwargs)
            return "ids"

    cars_model = SimpleNamespace(
        model=FakeHFModel(), tokenizer=SimpleNamespace(eos_token_id=5)
    )
    runner._cars_log_first_generate_call(cars_model, dataset="spider")
    passed = SimpleNamespace(temperature=1.0, top_p=1.0, top_k=None, do_sample=True, max_new_tokens=512, eos_token_id=5)
    assert cars_model.model.generate([1], generation_config=passed) == "ids"
    cars_model.model.generate([1], generation_config=passed)

    lines = [m for m in _messages(caplog) if "[legacy-cars] sampling" in m]
    assert len(lines) == 1
    assert "temperature=1.0 top_p=1.0 top_k=None" in lines[0]
    assert "do_sample=True" in lines[0] and "max_new_tokens=512" in lines[0]
    assert "model_default_temperature=0.7" in lines[0]
    assert len(calls) == 2


# ---- IterGen -------------------------------------------------------------------


class FakeTokenTensor:
    """The slice of a torch tensor the harness uses: ``tokens[..., start:]`` and ``tolist()``."""

    def __init__(self, rows):
        self.rows = rows

    def __getitem__(self, key):
        _ellipsis, token_slice = key
        return FakeTokenTensor([row[token_slice] for row in self.rows])

    def tolist(self):
        return self.rows


class FakeSpiderIterGen:
    """Emits a wrong column first, then the right one once it has been backed out."""

    def __init__(self):
        self.columns: list[str] = []
        self._script = ["SELECT nope", "SELECT name", "SELECT name FROM singer"]
        self._step = -1
        self.session_tokens = FakeTokenTensor([[1, 2, 3]])
        self.start_from = 3
        self.generation_config = SimpleNamespace(max_new_tokens=64, eos_token_id=5)
        self.tokenizer = SimpleNamespace(eos_token_id=5)
        self.recurrence_penalty = 0.3

    def start(self, prompt):
        pass

    def forward(self, unit=None, units=None, num=1, **gen_args):
        self._step += 1
        text = self._script[self._step]
        ids = [1, 2, 3] + list(range(10, 10 + len(text.split())))
        self.session_tokens = FakeTokenTensor([ids + [5] if self._step == 2 else ids])
        self.columns = [text.split()[1]]
        return [text]

    def view(self, unit=None):
        if unit == "column_name":
            return [list(self.columns)]
        return [["singer"] if self._step == 2 else []]

    def backward(self, unit=None, num=1):
        self.session_tokens = FakeTokenTensor([self.session_tokens.rows[0][:-1]])
        self.columns = []

    def finished(self):
        return self._step >= 2


def test_spider_itergen_logs_forward_view_check_backward_and_end(caplog):
    caplog.set_level(logging.INFO, logger=TRACE_LOGGER)
    example = {"db_info": "# singer ( name , age )"}
    runner._itergen_generate_spider(FakeSpiderIterGen(), "prompt", example)
    messages = _messages(caplog)
    tag = "[legacy-itergen-spider]"

    start = _only(messages, f"{tag} start")
    assert "recurrence_penalty=0.3" in start

    forwards = [m for m in messages if f"{tag} forward" in m]
    assert len(forwards) == 3
    assert "units=['column_name', 'table_name'] num=1" in forwards[0]
    assert "'SELECT nope'" in forwards[0]

    views = [m for m in messages if f"{tag} view" in m]
    assert "unit=column_name raw=[['nope']]" in views[0]

    checks = [m for m in messages if f"{tag} check" in m]
    assert "unit=column_name value='nope' passed=False reason=not_in_schema" in checks[0]
    assert any("unit=column_name value='name' passed=True" in m for m in checks)
    assert any("unit=table_name value='singer' passed=True" in m for m in checks)

    backward = _only(messages, f"{tag} backward")
    assert "unit=column_name" in backward and "tokens_removed=1" in backward
    assert "reason=not_in_schema" in backward

    finish = _only(messages, f"{tag} finish")
    assert "end_reason=end_token(id=5)" in finish


def test_spider_itergen_tells_the_iteration_limit_apart(caplog):
    caplog.set_level(logging.INFO, logger=TRACE_LOGGER)
    fake = FakeSpiderIterGen()
    runner._itergen_generate_spider(fake, "prompt", {"db_info": "# singer ( name )"}, max_iterations=1)
    assert "end_reason=iteration_limit" in _only(_messages(caplog), "finish")


def test_smiles_itergen_logs_the_failed_check_and_the_backward(caplog):
    from tests.test_itergen_smiles_loop import FakeIterGen, _INVALID_TEXT, _VALID_NEW_MOLECULE, _EXAMPLE

    caplog.set_level(logging.INFO, logger=TRACE_LOGGER)
    runner._itergen_generate_smiles(
        FakeIterGen([_INVALID_TEXT, _VALID_NEW_MOLECULE]), "prompt", dict(_EXAMPLE), seen=set()
    )
    messages = _messages(caplog)
    tag = "[legacy-itergen-smiles]"
    assert len([m for m in messages if f"{tag} forward" in m]) == 2
    checks = [m for m in messages if f"{tag} check" in m]
    assert "passed=False" in checks[0] and "reason=" in checks[0]
    assert "passed=True" in checks[1]
    assert "unit=token" in _only(messages, f"{tag} backward")
    assert "end_reason=molecule_accepted" in _only(messages, f"{tag} finish")


def test_plain_itergen_is_logged_like_its_siblings(caplog):
    caplog.set_level(logging.INFO, logger=TRACE_LOGGER)

    class OneShot:
        session_tokens = FakeTokenTensor([[1, 2, 9, 9]])
        start_from = 2
        generation_config = SimpleNamespace(max_new_tokens=2, eos_token_id=5)
        tokenizer = SimpleNamespace(eos_token_id=5)

        def start(self, prompt):
            pass

        def forward(self, **kwargs):
            return ["3 + 2>>"]

        def finished(self):
            return True

    assert runner._itergen_generate(OneShot(), "prompt") == "3 + 2>>"
    messages = _messages(caplog)
    assert "'3 + 2>>'" in _only(messages, "[legacy-itergen] forward")
    finish = _only(messages, "[legacy-itergen] finish")
    assert "end_reason=token_budget" in finish and "backtracks=0" in finish


def test_itergen_settings_line_carries_the_recurrence_penalty(caplog):
    caplog.set_level(logging.INFO, logger=TRACE_LOGGER)
    kwargs = runner._itergen_generation_kwargs(dataset="spider", max_tokens=4096, max_new_tokens=150)
    runner._log_itergen_settings("spider", kwargs)
    line = _only(_messages(caplog), "[legacy-itergen] settings")
    assert "recurrence_penalty=0.3" in line and "do_sample=False" in line and "max_new_tokens=150" in line

    caplog.clear()
    runner._log_itergen_settings(
        "smiles", runner._itergen_generation_kwargs(dataset="smiles", max_tokens=4096, max_new_tokens=64)
    )
    assert "recurrence_penalty=0.3" in _only(_messages(caplog), "settings")


# ---- unconstrained -------------------------------------------------------------


def _install_fake_vllm(monkeypatch, text, finish_reason, stop_reason, sampling_seen):
    class FakeLLM:
        def __init__(self, **kwargs):
            pass

        def get_tokenizer(self):
            from transformers import AutoTokenizer

            return AutoTokenizer.from_pretrained("Qwen/Qwen3.5-2B", trust_remote_code=True)

        def generate(self, prompts, sampling_params):
            completion = SimpleNamespace(
                text=text, token_ids=[1, 2, 3], finish_reason=finish_reason, stop_reason=stop_reason
            )
            return [SimpleNamespace(outputs=[completion])]

    def sampling_params(**kwargs):
        sampling_seen.append(kwargs)
        return SimpleNamespace(**kwargs)

    vllm = types.ModuleType("vllm")
    vllm.LLM = FakeLLM
    vllm.SamplingParams = sampling_params
    monkeypatch.setitem(sys.modules, "vllm", vllm)


def _unconstrained_args(tmp_path, **overrides):
    base = dict(
        dataset="smiles", strategy="unconstrained", eval_model="fake/model", eval_backend="vllm",
        device="cuda:0", eval_sample_size=1, eval_start_index=0, eval_end_index=None,
        eval_max_steps=64, eval_step_token_budget=1, vllm_gpu_memory_utilization=0.3,
        vllm_tensor_parallel_size=1, spider_split_file=None, spider_split_name="eval",
        smiles_classes="acrylates", smiles_samples_per_class=1, output_json=tmp_path / "out.json",
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def test_unconstrained_smiles_logs_no_grammar_stop_reason_and_both_texts(monkeypatch, tmp_path, caplog):
    pytest.importorskip("rdkit")
    caplog.set_level(logging.INFO, logger=TRACE_LOGGER)
    _install_fake_vllm(monkeypatch, " C=CC(=O)OCCCCC\n\nMolecule:", "stop", "\n\n", [])
    monkeypatch.setattr(runner, "_legacy_local_cuda_device", lambda device: "cpu")
    monkeypatch.setattr(runner, "_build_minimal_json", lambda *a, **k: None)
    runner.run_unconstrained_smiles_adapter(_unconstrained_args(tmp_path))
    messages = _messages(caplog)
    tag = "[legacy-unconstrained-smiles]"

    assert "no logits processor and no grammar object" in _only(messages, f"{tag} constraint")
    header = _only(messages, f"{tag} lane-header")
    assert "max_new_tokens=64" in header
    example = _only(messages, f"{tag} class=acrylates draw=0")
    assert "stop_reason=vllm:stop(" in example
    assert "raw=' C=CC(=O)OCCCCC" in example
    assert "scored='C=CC(=O)OCCCCC'" in example


def test_unconstrained_spider_logs_no_grammar_stop_reason_and_both_texts(monkeypatch, tmp_path, caplog):
    caplog.set_level(logging.INFO, logger=TRACE_LOGGER)
    _install_fake_vllm(monkeypatch, "SQL: SELECT 1\n\nSQL: again", "length", None, [])

    class FakeLogic:
        @staticmethod
        def load_dataset_sample(runtime):
            return [{"question": "q"}]

        @staticmethod
        def expected_answer(runtime, example):
            return "SELECT 1"

        @staticmethod
        def extract_actual(runtime, output, example):
            return "SELECT 1", "generated", {"syntax_valid": True}

        @staticmethod
        def is_correct(*args):
            return True

    import synthesis.evaluate.benchmarks.registry as registry
    import synthesis.evaluate.evaluator as evaluator

    monkeypatch.setattr(registry, "get_logic", lambda dataset: FakeLogic)
    monkeypatch.setattr(evaluator, "Evaluator", lambda **kwargs: SimpleNamespace())
    monkeypatch.setattr(runner, "_configure_fixed_eval_runtime", lambda *a: None)
    from synthesis.evaluate.benchmarks.sql_spider.prompts import SpiderPromptParts

    monkeypatch.setattr(
        runner, "_legacy_benchmark_prompt", lambda *a: SpiderPromptParts("the spider prompt\n")
    )
    monkeypatch.setattr(runner, "_baseline_row_question", lambda *a: "question")
    monkeypatch.setattr(runner, "_legacy_local_cuda_device", lambda device: "cpu")
    monkeypatch.setattr(runner, "_build_minimal_json", lambda *a, **k: None)

    runner.run_unconstrained_spider_adapter(_unconstrained_args(tmp_path, dataset="spider"))
    messages = _messages(caplog)
    tag = "[legacy-unconstrained-spider]"

    assert "no logits processor and no grammar object" in _only(messages, f"{tag} constraint")
    header = _only(messages, f"{tag} lane-header")
    assert "the spider prompt" in header and "max_new_tokens=64" in header
    example = _only(messages, f"{tag} example=0")
    assert "stop_reason=vllm:length" in example
    assert "raw='SQL: SELECT 1\\n\\nSQL: again'" in example
    assert "scored='SELECT 1'" in example
