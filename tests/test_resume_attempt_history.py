import json

from synthesis.run_synthesis import _load_initial_attempt_history


def test_load_initial_attempt_history_restores_metrics_and_strategy(tmp_path):
    history = tmp_path / "history.json"
    history.write_text(
        json.dumps(
            [
                {
                    "attempt_number": 3,
                    "strategy_code": "method Main() {}",
                    "accuracy": 0.42,
                    "syntax_rate": 0.91,
                    "num_examples": 49,
                    "num_correct": 21,
                    "contains_delimiters": True,
                }
            ]
        )
    )

    attempts = _load_initial_attempt_history(history)

    assert len(attempts) == 1
    assert attempts[0].attempt_number == 3
    assert attempts[0].strategy_code == "method Main() {}"
    assert attempts[0].eval_result.accuracy == 0.42
    assert attempts[0].eval_result.syntax_rate == 0.91
    assert attempts[0].eval_result.num_correct == 21


def test_load_progress_report_restores_attempt_status_and_sample_outputs(tmp_path):
    history = tmp_path / "progress_report.json"
    history.write_text(
        json.dumps(
            {
                "attempts": [
                    {
                        "attempt_number": 12,
                        "strategy_code": "method Main() {}",
                        "timestamp": "2026-09-03T06:21:48",
                        "failed_at": "evaluation",
                        "error_summary": "below the new bar",
                        "verification": {"success": True, "error_count": 0},
                        "compilation": {
                            "success": True,
                            "output_dir": "/tmp/attempt12-compiled",
                        },
                        "evaluation": {
                            "success": True,
                            "accuracy": 8 / 49,
                            "contains_delimiters": False,
                            "syntax_rate": 45 / 49,
                            "num_examples": 49,
                            "num_correct": 8,
                            "accuracy_denominator": 49,
                            "accuracy_definition": "correct_examples_over_all_examples",
                            "invalid_outputs_excluded_from_accuracy": 0,
                            "total_time_seconds": 373.6,
                            "max_sample_time_seconds": 12.3,
                            "early_stopped": False,
                            "early_stop_reason": None,
                            "planned_num_examples": 49,
                            "error": None,
                            "sample_outputs": [
                                {"is_correct": False, "helper_trace": [{"helper": "A"}]}
                            ],
                            "aux_metrics": {"restored": True},
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    attempts = _load_initial_attempt_history(history)

    assert len(attempts) == 1
    attempt = attempts[0]
    assert attempt.failed_at.value == "evaluation"
    assert attempt.error_summary == "below the new bar"
    assert attempt.verification_result.success is True
    assert str(attempt.compilation_result.output_dir) == "/tmp/attempt12-compiled"
    assert attempt.eval_result.sample_outputs == [
        {"is_correct": False, "helper_trace": [{"helper": "A"}]}
    ]
    assert attempt.eval_result.aux_metrics == {"restored": True}
