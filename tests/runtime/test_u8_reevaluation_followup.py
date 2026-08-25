import json
from pathlib import Path
from types import SimpleNamespace

import pytest


class _ShardProcess:
    rows_by_shard = {0: [3], 1: [9, 15]}
    malformed_evidence = False
    source_mismatch = False
    outcome_mode = None
    provenance_mismatch_field = None
    provenance_mode = "present"
    missing_provenance_shard = None
    dataset = "spider"
    alias_mode = "equal"
    provenance_source_mode = None
    spawn_count = 0

    def __init__(self, command, stdout=None, stderr=None, env=None):
        type(self).spawn_count += 1
        del stdout, stderr, env
        output_path = Path(command[command.index("--output-json") + 1])
        shard_index = int(output_path.stem.removeprefix("part"))
        dataset = command[command.index("--dataset") + 1]
        split_flag = (
            "--spider-split-file" if dataset == "spider" else "--gsm-split-file"
        )
        split_path = Path(command[command.index(split_flag) + 1])
        split = json.loads(split_path.read_text())
        source_alias = (
            "spider_source_index" if dataset == "spider" else "crane_source_index"
        )
        rows = self.rows_by_shard[shard_index]
        answers = [
            {
                "example_index": local_index,
                "source_index": source_index,
                source_alias: source_index,
                "is_correct": source_index == 3,
                "is_syntax_valid": True,
                "question": f"q-{source_index}",
                "generated_answer": "SELECT 1",
            }
            for local_index, source_index in enumerate(rows)
        ]
        evidence = [
            {
                "evaluated_index": local_index,
                "source_index": source_index,
                source_alias: source_index,
                "is_correct": source_index == 3,
                "accuracy_applicable": True,
                "is_syntax_valid": True,
                "output_contract_valid": True,
            }
            for local_index, source_index in enumerate(rows)
        ]
        if self.malformed_evidence:
            evidence = evidence[:-1]
        if self.source_mismatch and evidence:
            evidence[0]["source_index"] += 1
            evidence[0][source_alias] += 1
        if self.outcome_mode == "missing":
            for row in answers + evidence:
                row.pop("is_correct", None)
                row.pop("is_syntax_valid", None)
        elif self.outcome_mode == "int":
            answers[0]["is_correct"] = 1
            evidence[0]["is_correct"] = 1
        elif self.outcome_mode == "string":
            answers[0]["is_syntax_valid"] = "true"
            evidence[0]["is_syntax_valid"] = "true"
        elif self.outcome_mode == "mismatch":
            evidence[0]["is_correct"] = not answers[0]["is_correct"]
        if self.alias_mode == "spider_conflict" and dataset == "spider":
            answers[0]["spider_source_index"] = answers[0]["source_index"] + 1
        elif self.alias_mode == "gsm_conflict" and dataset == "gsm_symbolic":
            answers[0]["crane_source_index"] = answers[0]["source_index"] + 1
        elif self.alias_mode == "invalid_string":
            answers[0]["source_index"] = "3"
        elif self.alias_mode == "invalid_float":
            answers[0]["source_index"] = 3.0
        elif self.alias_mode == "invalid_bool":
            answers[0]["source_index"] = True
        elif self.alias_mode == "none_optional":
            answers[0]["source_index"] = None
        elif self.alias_mode == "none_only":
            answers[0]["source_index"] = None
            answers[0][source_alias] = None
        split_provenance = {
            "gsm_split_file": str(split_path) if dataset == "gsm_symbolic" else None,
            "gsm_split_name": "test" if dataset == "gsm_symbolic" else None,
            "spider_split_file": str(split_path) if dataset == "spider" else None,
            "spider_split_name": "test" if dataset == "spider" else None,
            "bar_split_name": None,
        }
        provenance_indices = list(rows)
        if self.provenance_source_mode == "string":
            provenance_indices[0] = str(provenance_indices[0])
        elif self.provenance_source_mode == "float":
            provenance_indices[0] = float(provenance_indices[0])
        elif self.provenance_source_mode == "bool" and shard_index == 0:
            provenance_indices[0] = True
        output_path.write_text(
            json.dumps(
                {
                    "accuracy": sum(bool(row.get("is_correct")) for row in answers) / len(answers),
                    "syntax_rate": 1.0,
                    "metrics": {},
                    "answers": answers,
                    "eval_split": split_provenance,
                    "reevaluation_sample_evidence": evidence,
                    "reevaluation_provenance": (
                        None
                        if (
                            self.provenance_mode == "absent"
                            or (
                                self.provenance_mode == "mixed"
                                and shard_index == 1
                            )
                            or self.missing_provenance_shard == shard_index
                        )
                        else {
                            "dataset": dataset,
                            "gsm_split_file": str(split_path)
                            if dataset == "gsm_symbolic"
                            else None,
                            "gsm_split_name": "test"
                            if dataset == "gsm_symbolic"
                            else None,
                            "spider_split_file": str(split_path)
                            if dataset == "spider"
                            else None,
                            "spider_split_name": "test"
                            if dataset == "spider"
                            else None,
                            "evaluated_source_indices": provenance_indices,
                            "compiled_csd_path": "/compiled/GeneratedCSD.py",
                            "compiled_csd_sha256": "aaa",
                            "eval_model": "Qwen/Qwen2.5-7B-Instruct",
                            "cell_id": "spider-cell",
                            "manifest_commit": "m" * 40,
                            "sample_size": len(split["test_indices"]),
                            "max_steps": 8,
                            "step_token_budget": 1,
                            "smiles_class": None,
                        }
                    ),
                }
            )
        )
        if self.provenance_mismatch_field and shard_index == 1:
            output_payload = json.loads(output_path.read_text())
            output_payload["reevaluation_provenance"][self.provenance_mismatch_field] = {
                "compiled_csd_path": "/compiled/Other.py",
                "compiled_csd_sha256": "bbb",
                "eval_model": "Qwen/Qwen3.5-4B",
                "cell_id": "other-cell",
                "manifest_commit": "n" * 40,
                "max_steps": 9,
                "step_token_budget": 2,
                "smiles_class": "other",
            }[self.provenance_mismatch_field]
            output_path.write_text(json.dumps(output_payload))
        self.returncode = 0

    def wait(self):
        return self.returncode


def _run_sharded(
    monkeypatch,
    tmp_path,
    *,
    malformed=False,
    source_mismatch=False,
    rows_by_shard=None,
    outcome_mode=None,
    provenance_mismatch_field=None,
    provenance_mode="present",
    missing_provenance_shard=None,
    dataset="spider",
    alias_mode="equal",
    provenance_source_mode=None,
    planned_indices=None,
):
    from synthesis.scripts import sharded_eval_core

    split_path = tmp_path / "canonical_split.json"
    split_path.write_text(
        json.dumps(
            {
                "test_indices": (
                    planned_indices
                    if planned_indices is not None
                    else [3, 8, 9, 15]
                ),
                "test_size": 4,
                "train_indices": [100],
                "train_size": 1,
            }
        )
    )
    output_path = tmp_path / "merged.json"
    _ShardProcess.rows_by_shard = rows_by_shard or {0: [3], 1: [9, 15]}
    _ShardProcess.malformed_evidence = malformed
    _ShardProcess.source_mismatch = source_mismatch
    _ShardProcess.outcome_mode = outcome_mode
    _ShardProcess.provenance_mismatch_field = provenance_mismatch_field
    _ShardProcess.provenance_mode = provenance_mode
    _ShardProcess.missing_provenance_shard = missing_provenance_shard
    _ShardProcess.dataset = dataset
    _ShardProcess.alias_mode = alias_mode
    _ShardProcess.provenance_source_mode = provenance_source_mode
    _ShardProcess.spawn_count = 0
    monkeypatch.setattr(sharded_eval_core, "detect_gpu_slots", lambda *args, **kwargs: [0, 1])
    monkeypatch.setattr(sharded_eval_core.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(sharded_eval_core.subprocess, "Popen", _ShardProcess)
    return (
        sharded_eval_core.run_sharded_reevaluation(
            csd_path=str(tmp_path / "GeneratedCSD.py"),
            dataset=dataset,
            sample_size=4,
            output_json=str(output_path),
            split_file=str(split_path),
            split_name="test",
            passthrough=[],
            stagger_seconds=0,
        ),
        output_path,
        split_path,
    )


def test_run_sharded_reevaluation_merges_evidence_and_provenance_for_early_stop(
    monkeypatch, tmp_path
):
    rc, output_path, split_path = _run_sharded(monkeypatch, tmp_path)

    assert rc == 0
    payload = json.loads(output_path.read_text())
    assert [row["example_index"] for row in payload["answers"]] == [0, 1, 2]
    assert [row["evaluated_index"] for row in payload["reevaluation_sample_evidence"]] == [0, 1, 2]
    assert [row["source_index"] for row in payload["reevaluation_sample_evidence"]] == [3, 9, 15]
    assert payload["eval_split"]["spider_split_file"] == str(split_path.resolve())
    assert payload["eval_split"]["spider_split_name"] == "test"
    assert payload["reevaluation_provenance"]["evaluated_source_indices"] == [3, 9, 15]
    assert payload["reevaluation_provenance"]["spider_split_file"] == str(split_path.resolve())
    assert payload["reevaluation_provenance"]["sample_size"] == 4
    assert payload["reevaluation_provenance"]["planned_sample_size"] == 4
    assert payload["reevaluation_provenance"]["evaluated_count"] == 3
    assert payload["metrics"]["planned_sample_size"] == 4
    assert payload["metrics"]["evaluated_count"] == 3


def test_run_sharded_reevaluation_fails_closed_on_answer_evidence_misalignment(
    monkeypatch, tmp_path
):
    with pytest.raises(ValueError, match="evidence.*answers"):
        _run_sharded(monkeypatch, tmp_path, malformed=True)


@pytest.mark.parametrize(
    "bad_rows",
    [
        [8],
        [3, 9],
        [8, 3],
        [3, 8, 9],
    ],
)
def test_run_sharded_reevaluation_rejects_rows_outside_assigned_prefix(
    monkeypatch, tmp_path, bad_rows
):
    with pytest.raises(ValueError, match="assigned shard|prefix|canonical"):
        _run_sharded(
            monkeypatch,
            tmp_path,
            rows_by_shard={0: bad_rows, 1: [9, 15]},
        )


@pytest.mark.parametrize(
    "outcome_mode",
    ["missing", "int", "string", "mismatch"],
)
def test_run_sharded_reevaluation_requires_boolean_matching_outcomes(
    monkeypatch, tmp_path, outcome_mode
):
    with pytest.raises(ValueError, match="outcome|bool|match"):
        _run_sharded(monkeypatch, tmp_path, outcome_mode=outcome_mode)


@pytest.mark.parametrize(
    "provenance_mismatch_field",
    [
        "compiled_csd_path",
        "compiled_csd_sha256",
        "eval_model",
        "cell_id",
        "manifest_commit",
        "max_steps",
        "step_token_budget",
        "smiles_class",
    ],
)
def test_run_sharded_reevaluation_rejects_mixed_immutable_provenance(
    monkeypatch, tmp_path, provenance_mismatch_field
):
    with pytest.raises(ValueError, match="provenance|mismatch|shard"):
        _run_sharded(
            monkeypatch,
            tmp_path,
            provenance_mismatch_field=provenance_mismatch_field,
        )


def test_run_sharded_reevaluation_preserves_generic_outputs_without_provenance(
    monkeypatch, tmp_path
):
    rc, output_path, split_path = _run_sharded(
        monkeypatch, tmp_path, provenance_mode="absent"
    )

    assert rc == 0
    payload = json.loads(output_path.read_text())
    assert "reevaluation_provenance" not in payload
    assert [row["source_index"] for row in payload["answers"]] == [3, 9, 15]
    assert [row["source_index"] for row in payload["reevaluation_sample_evidence"]] == [
        3,
        9,
        15,
    ]
    assert payload["eval_split"]["spider_split_file"] == str(split_path.resolve())
    assert payload["metrics"]["planned_sample_size"] == 4
    assert payload["metrics"]["evaluated_count"] == 3


def test_run_sharded_reevaluation_rejects_mixed_provenance_presence(
    monkeypatch, tmp_path
):
    with pytest.raises(ValueError, match="provenance"):
        _run_sharded(monkeypatch, tmp_path, provenance_mode="mixed")


@pytest.mark.parametrize(
    "alias_mode",
    ["invalid_string", "invalid_float", "invalid_bool", "none_only"],
)
def test_run_sharded_reevaluation_rejects_non_integer_source_aliases(
    monkeypatch, tmp_path, alias_mode
):
    with pytest.raises(ValueError, match="source|alias|integer|int"):
        _run_sharded(monkeypatch, tmp_path, alias_mode=alias_mode)


def test_run_sharded_reevaluation_treats_none_alias_as_absent(
    monkeypatch, tmp_path
):
    rc, output_path, _ = _run_sharded(
        monkeypatch, tmp_path, alias_mode="none_optional"
    )

    assert rc == 0
    payload = json.loads(output_path.read_text())
    assert [row["source_index"] for row in payload["answers"]] == [None, None, 15]
    assert [row["spider_source_index"] for row in payload["answers"]] == [3, 9, 15]


def test_run_sharded_reevaluation_rejects_conflicting_spider_source_aliases(
    monkeypatch, tmp_path
):
    with pytest.raises(ValueError, match="source|alias|match|equal"):
        _run_sharded(monkeypatch, tmp_path, alias_mode="spider_conflict")


def test_run_sharded_reevaluation_rejects_conflicting_gsm_source_aliases(
    monkeypatch, tmp_path
):
    with pytest.raises(ValueError, match="source|alias|match|equal"):
        _run_sharded(
            monkeypatch,
            tmp_path,
            dataset="gsm_symbolic",
            alias_mode="gsm_conflict",
        )


def test_run_sharded_reevaluation_accepts_equal_multi_alias_rows(
    monkeypatch, tmp_path
):
    rc, output_path, _ = _run_sharded(monkeypatch, tmp_path, alias_mode="equal")

    assert rc == 0
    payload = json.loads(output_path.read_text())
    assert [row["source_index"] for row in payload["answers"]] == [3, 9, 15]
    assert [row["spider_source_index"] for row in payload["answers"]] == [3, 9, 15]
    assert [
        row["spider_source_index"] for row in payload["reevaluation_sample_evidence"]
    ] == [3, 9, 15]


@pytest.mark.parametrize("bad_value", ["3", 3.0, True])
def test_run_sharded_reevaluation_rejects_non_integer_planned_indices_before_spawn(
    monkeypatch, tmp_path, bad_value
):
    with pytest.raises(ValueError):
        _run_sharded(
            monkeypatch,
            tmp_path,
            planned_indices=[bad_value, 8, 9, 15],
        )
    assert _ShardProcess.spawn_count == 0


@pytest.mark.parametrize("provenance_source_mode", ["string", "float", "bool"])
def test_run_sharded_reevaluation_rejects_non_integer_provenance_indices(
    monkeypatch, tmp_path, provenance_source_mode
):
    kwargs = {"provenance_source_mode": provenance_source_mode}
    if provenance_source_mode == "bool":
        kwargs.update(
            rows_by_shard={0: [1], 1: [9, 15]},
            planned_indices=[1, 8, 9, 15],
        )
    with pytest.raises(ValueError):
        _run_sharded(monkeypatch, tmp_path, **kwargs)


class _GuidanceTokenizer:
    eos_token = "<eos>"
    eos_token_id = 99
    all_special_ids = {99}

    def __init__(self, *, reject_thinking=False):
        self.reject_thinking = reject_thinking
        self.template_calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.template_calls.append(([dict(message) for message in messages], dict(kwargs)))
        if self.reject_thinking and "enable_thinking" in kwargs:
            raise TypeError("legacy tokenizer has no enable_thinking")
        return "<rendered:" + "|".join(str(message.get("content", "")) for message in messages) + ">"

    def decode(self, token_ids, skip_special_tokens=False):
        del skip_special_tokens
        return "".join("<eos>" if int(token_id) == 99 else "ok" for token_id in token_ids)

    def encode(self, text, add_special_tokens=False):
        del text, add_special_tokens
        return []


class _GuidanceDafny:
    @staticmethod
    def Seq(value):
        return value

    @staticmethod
    def SeqWithoutIsStrInference(values):
        return list(values)


class _GuidanceParser:
    @staticmethod
    def is_complete(text):
        return text == "ok"


class _GuidanceDefault:
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
        lm_arg.AppendTaskGuidance("Use only the registered schema")
        return (["ok"], False, [], 1)


def test_run_sharded_reevaluation_fails_closed_on_source_misalignment(
    monkeypatch, tmp_path
):
    with pytest.raises(ValueError, match="answer/evidence source mismatch"):
        _run_sharded(monkeypatch, tmp_path, source_mismatch=True)


@pytest.mark.parametrize(
    ("model_name", "prompt_kind", "reject_thinking"),
    [
        ("Qwen/Qwen2.5-7B-Instruct", "structured", False),
        ("Qwen/Qwen3.5-4B", "structured", False),
        ("Qwen/Qwen2.5-7B-Instruct", "legacy-chat", True),
    ],
)
def test_run_crane_csd_guidance_refreshes_final_prompt_contract(
    monkeypatch, tmp_path, model_name, prompt_kind, reject_thinking
):
    from synthesis.evaluate.benchmarks.common.model_utils import _TensorizedLMBase
    from synthesis.evaluate.benchmarks.gsm_symbolic.generation import run_crane_csd
    from synthesis.evaluate.benchmarks.sql_spider.prompts import SpiderPromptParts

    tokenizer = _GuidanceTokenizer(reject_thinking=reject_thinking)
    lm = _TensorizedLMBase(_GuidanceDafny(), tokenizer, ["ok"], [1])
    env = {
        "_dafny": _GuidanceDafny,
        "GeneratedCSD": SimpleNamespace(default__=_GuidanceDefault),
        "lm": lm,
        "parser": _GuidanceParser(),
        "model_name": model_name,
    }
    prompt = (
        SpiderPromptParts("db_id: x\nquestion: q\n", answer_cue="SQL:")
        if prompt_kind == "structured"
        else [{"role": "user", "content": "question: q"}]
    )

    run_crane_csd(
        env=env,
        prompt_text=prompt,
        max_steps=8,
        grammar_file=tmp_path / "unused.lark",
    )

    contract = lm._last_prompt_contract
    assert contract["prompt_chars"] == len(lm.instruction_text)
    assert contract["render_succeeded"] is True
    assert "registered schema" in lm.instruction_text
    assert contract["chat_message_count"] == (1 if model_name.endswith("4B") or prompt_kind == "legacy-chat" else 0)
    if prompt_kind == "legacy-chat":
        assert contract["template_fallback"] is True
        assert contract["enable_thinking"] is None


def test_evaluator_late_generation_error_uses_published_token_evidence_for_count(monkeypatch):
    from synthesis.evaluate.evaluator import Evaluator

    evaluator = Evaluator(
        dataset_name="spider",
        model_name="Qwen/Qwen2.5-7B-Instruct",
        backend="huggingface",
        device="cpu",
        sample_size=1,
        max_steps=8,
    )
    monkeypatch.setattr(evaluator, "_format_prompt", lambda _example: "question")
    monkeypatch.setattr(evaluator, "_get_expected_answer", lambda _example: "SELECT 1")
    monkeypatch.setattr(evaluator, "_get_grammar_file", lambda: Path("sql.lark"))
    monkeypatch.setattr(evaluator, "_accuracy_applicable_for_example", lambda _aux: True)

    class Logic:
        @staticmethod
        def build_dynamic_parser(_evaluator, _env, _example):
            return None

    class LM:
        _last_generation_evidence = None
        _last_prompt_contract = None
        task_guidance = None

    env = {"lm": LM(), "tokenizer": None}

    def generation_error(**kwargs):
        kwargs["env"]["lm"]._last_generation_evidence = {
            "raw_token_ids": [10, 2],
            "raw_decoded_text": "SELECT 1<eos>",
            "removed_terminal_token_ids": [2],
            "decoded_text": "SELECT 1",
            "strategy_output_relation": "mixed",
            "strategy_mutation": True,
            "strategy_removed_sampled_token_ids": [],
        }
        raise RuntimeError("late generation failure")

    sample = evaluator._evaluate_one_example(
        0,
        {"question": "q", "query": "SELECT 1"},
        1,
        env,
        Logic,
        generation_error,
        {},
    )

    assert sample["generation_token_evidence"]["removed_terminal_token_ids"] == [2]
    assert sample["removed_terminal_token_count"] == 1
    assert sample["strategy_output_relation"] == "mixed"
    assert sample["strategy_mutation"] is True
    assert sample["strategy_removed_sampled_token_ids"] == []


def test_exporter_normalizes_inconsistent_removed_terminal_count():
    from synthesis.evaluate.baseline_store import build_reevaluation_sample_evidence

    rows = build_reevaluation_sample_evidence(
        [
            {
                "generation_token_evidence": {
                    "raw_token_ids": [10, 2],
                    "raw_decoded_text": "SELECT 1<eos>",
                    "removed_terminal_token_ids": [2],
                    "decoded_text": "SELECT 1",
                },
                "removed_terminal_token_count": 0,
            }
        ]
    )

    assert rows[0]["removed_terminal_token_count"] == 1


def test_default_spider_split_provenance_uses_canonical_file(monkeypatch, tmp_path):
    import synthesis.scripts.reevaluate_compiled_csd as reevaluate

    canonical = tmp_path / "canonical-spider.json"
    canonical.write_text("{}")
    monkeypatch.setitem(reevaluate.SPLIT_FILE_BY_DATASET, "spider", canonical)
    compiled = tmp_path / "GeneratedCSD.py"
    compiled.write_text("# frozen\n")
    args = SimpleNamespace(
        dataset="spider",
        eval_model="Qwen/Qwen2.5-7B-Instruct",
        sample_size=1,
        max_steps=8,
        step_token_budget=1,
        smiles_classes=None,
        provenance_cell_id="spider-test",
        provenance_manifest_commit="a" * 40,
        sample_offset=0,
        spider_split_file=None,
        spider_split_name="test",
    )
    result = SimpleNamespace(
        sample_outputs=[{"spider_source_index": 3}],
    )

    provenance = reevaluate.build_reevaluation_provenance(
        args, compiled, evaluation_result=result
    )

    assert provenance["spider_split_file"] == str(canonical)
    assert provenance["spider_split_name"] == "test"
