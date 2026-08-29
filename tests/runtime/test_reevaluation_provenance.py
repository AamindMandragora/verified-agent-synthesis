import hashlib
from pathlib import Path
from types import SimpleNamespace

from synthesis.scripts.reevaluate_compiled_csd import build_reevaluation_provenance


def test_reevaluation_provenance_binds_output_to_strategy_model_and_cell(tmp_path):
    csd = tmp_path / "GeneratedCSD.py"
    csd.write_text("# frozen strategy\n", encoding="utf-8")
    args = SimpleNamespace(
        dataset="smiles",
        eval_model="Qwen/Qwen3.5-9B",
        sample_size=100,
        max_steps=400,
        step_token_budget=1,
        smiles_classes="isocyanates",
        provenance_cell_id="smiles-qwen35-9b-isocyanates",
        provenance_manifest_commit="a" * 40,
    )

    provenance = build_reevaluation_provenance(args, csd)

    assert provenance == {
        "cell_id": "smiles-qwen35-9b-isocyanates",
        "manifest_commit": "a" * 40,
        "dataset": "smiles",
        "eval_model": "Qwen/Qwen3.5-9B",
        "compiled_csd_path": str(csd.resolve()),
        "compiled_csd_sha256": hashlib.sha256(csd.read_bytes()).hexdigest(),
        "sample_size": 100,
        "max_steps": 400,
        "step_token_budget": 1,
        "smiles_class": "isocyanates",
    }


def test_reevaluation_provenance_records_exact_eval_model_and_spider_data(tmp_path):
    csd = tmp_path / "GeneratedCSD.py"
    csd.write_text("# frozen strategy\n", encoding="utf-8")
    args = SimpleNamespace(
        dataset="spider",
        eval_model="Qwen/Qwen3.5-2B",
        sample_size=300,
        max_steps=900,
        step_token_budget=1,
        smiles_classes=None,
        provenance_cell_id="t5-opus5-spider",
        provenance_manifest_commit="a" * 40,
        provenance_eval_model_revision="1" * 40,
        provenance_eval_model_snapshot_path="/cache/snapshots/" + "1" * 40,
        provenance_eval_model_snapshot_sha256="3" * 64,
        provenance_eval_model_snapshot_file_count=10,
        provenance_spider_data_path="/data/spider",
        provenance_spider_data_sha256="2" * 64,
        provenance_spider_data_file_count=922,
    )

    provenance = build_reevaluation_provenance(args, csd)

    assert provenance["eval_model_revision"] == "1" * 40
    assert provenance["eval_model_snapshot_path"] == args.provenance_eval_model_snapshot_path
    assert provenance["eval_model_snapshot_sha256"] == "3" * 64
    assert provenance["eval_model_snapshot_file_count"] == 10
    assert provenance["spider_data_path"] == "/data/spider"
    assert provenance["spider_data_sha256"] == "2" * 64
    assert provenance["spider_data_file_count"] == 922


def test_babysitter_smoke_split_fallback_only_for_smoke_report_path():
    from synthesis.scripts.reevaluate_compiled_csd import babysitter_smoke_split_fallback

    smoke = Path("/repo/logs/zero_acc_babysitter/smoke_spider-qwen25-1p5b_x/smoke_report.json")
    assert babysitter_smoke_split_fallback(smoke) == "train"
    assert babysitter_smoke_split_fallback(None) is None
    assert babysitter_smoke_split_fallback(Path("/tmp/final_numbers.json")) is None
    assert babysitter_smoke_split_fallback(Path("/repo/logs/other/smoke_report.json")) is None
    assert babysitter_smoke_split_fallback(Path("/repo/logs/zero_acc_babysitter/x/report.json")) is None


def test_reevaluation_export_preserves_spider_sample_evidence(tmp_path):
    import json

    from synthesis.evaluate.baseline_store import save_minimal_baseline_json
    from synthesis.evaluate.evaluator import EvaluationResult

    result = EvaluationResult(
        success=True,
        accuracy=0.0,
        contains_delimiters=False,
        syntax_rate=0.0,
        num_examples=1,
        num_correct=0,
        total_time_seconds=0.25,
        sample_outputs=[
            {
                "example_index": 0,
                "source_index": 3,
                "spider_source_index": 3,
                "question": "How many singers?",
                "full_output": "SELECT name FROM singer",
                "scored_output": "SELECT name FROM singer",
                "is_correct": False,
                "accuracy_applicable": True,
                "is_syntax_valid": False,
                "answer_source": "spider_output_contract_rejected",
                "has_extracted_answer": False,
                "output_contract_valid": False,
                "output_rejection_reason": "prompt_or_wrapper",
                "timed_out": False,
                "constrained_work": 13,
                "error_type": None,
                "error_status": None,
                "removed_terminal_token_count": 1,
                "generation_token_evidence": {
                    "raw_token_ids": [10, 2],
                    "raw_decoded_text": "SELECT name FROM singer<eos>",
                    "removed_terminal_token_ids": [2],
                    "decoded_text": "SELECT name FROM singer",
                },
                "strategy_output_relation": "mixed",
                "strategy_mutation": True,
                "strategy_removed_sampled_token_ids": [10],
                "helper_trace": [
                    {
                        "helper": "RollbackToCompletePrefix",
                        "generated_len_before": 2,
                        "generated_len_after": 1,
                        "status": "completed",
                    }
                ],
                "provenance_tags": ["parser_repair_or_rollback"],
                "failure_location": "answer_extraction_or_completion",
                "prompt_contract": {
                    "renderer": "spider",
                    "family": "qwen35",
                    "mode": "chat",
                    "template_used": True,
                    "raw_prompt": False,
                    "chat_message_count": 1,
                    "user_message_count": 1,
                    "add_generation_prompt": True,
                    "enable_thinking": False,
                    "render_succeeded": True,
                    "prompt_chars": 42,
                },
            }
        ],
    )
    output = tmp_path / "reevaluation.json"

    save_minimal_baseline_json(
        result,
        output,
        eval_split={"spider_split_name": "train"},
        metadata={
            "reevaluation_provenance": {
                "dataset": "spider",
                "sample_size": 1,
                "sample_offset": 0,
                "evaluated_source_indices": [3],
            }
        },
    )

    payload = json.loads(output.read_text())
    assert payload["answers"][0]["source_index"] == 3
    assert payload["answers"][0]["generated_answer"] == "SELECT name FROM singer"
    assert "reevaluation_sample_evidence" in payload

    evidence = payload["reevaluation_sample_evidence"]
    assert len(evidence) == 1
    row = evidence[0]
    assert row["evaluated_index"] == 0
    assert row["source_index"] == 3
    assert row["is_correct"] is False
    assert row["accuracy_applicable"] is True
    assert row["is_syntax_valid"] is False
    assert row["answer_source"] == "spider_output_contract_rejected"
    assert row["has_extracted_answer"] is False
    assert row["output_contract_valid"] is False
    assert row["output_rejection_reason"] == "prompt_or_wrapper"
    assert row["timed_out"] is False
    assert row["constrained_work"] == 13
    assert payload["answers"][0]["constrained_work"] == 13
    assert payload["metrics"]["mean_constrained_work"] == 13.0
    assert row["error_type"] is None
    assert row["error_status"] is None
    assert row["removed_terminal_token_count"] == 1
    assert row["generation_token_evidence"] == {
        "raw_token_ids": [10, 2],
        "raw_decoded_text": "SELECT name FROM singer<eos>",
        "removed_terminal_token_ids": [2],
        "decoded_text": "SELECT name FROM singer",
    }
    assert row["strategy_output_relation"] == "mixed"
    assert row["strategy_mutation"] is True
    assert row["strategy_removed_sampled_token_ids"] == [10]
    assert row["helper_trace"][0]["helper"] == "RollbackToCompletePrefix"
    assert row["provenance_tags"] == ["parser_repair_or_rollback"]
    assert row["failure_location"] == "answer_extraction_or_completion"
    assert row["prompt_contract"]["enable_thinking"] is False


def test_constrained_work_metrics_and_rows_survive_all_dataset_exports():
    from synthesis.evaluate.baseline_store import build_minimal_baseline_record
    from synthesis.evaluate.evaluator import EvaluationResult

    for dataset in ("gsm_symbolic", "spider", "smiles"):
        samples = [
            {
                "question": f"{dataset} one",
                "full_output": "answer one",
                "token_count": 5,
                "time_seconds": 0.1,
                "constrained_work": 4,
                "is_correct": True,
                "is_syntax_valid": True,
            },
            {
                "question": f"{dataset} two",
                "full_output": "answer two",
                "token_count": 6,
                "time_seconds": 0.2,
                "constrained_work": 10,
                "is_correct": False,
                "is_syntax_valid": True,
            },
        ]
        record = build_minimal_baseline_record(
            EvaluationResult(
                success=True,
                accuracy=0.5,
                contains_delimiters=False,
                syntax_rate=1.0,
                num_examples=2,
                num_correct=1,
                total_time_seconds=0.3,
                sample_outputs=samples,
            )
        )
        assert record["metrics"]["total_constrained_work"] == 14
        assert record["metrics"]["mean_constrained_work"] == 7.0
        assert [row["constrained_work"] for row in record["answers"]] == [4, 10]
        assert [row["constrained_work"] for row in record["reevaluation_sample_evidence"]] == [4, 10]
