from pathlib import Path

from synthesis.evaluate.evaluator import Evaluator, _print_realtime_completion


def test_realtime_completion_logging_preserves_multiline_sample_text(capsys):
    completion = "SQL: <<SELECT name\nFROM singer>>"

    _print_realtime_completion(2, 3, completion)

    assert capsys.readouterr().out == (
        "  [EVAL]   Sample 2/3 completion begin\n"
        "SQL: <<SELECT name\n"
        "FROM singer>>\n"
        "  [EVAL]   Sample 2/3 completion end\n"
    )


def test_evaluator_logs_the_completion_and_keeps_it_in_results(monkeypatch, capsys):
    completion = "SQL: <<SELECT name FROM singer>>"
    evaluator = Evaluator(dataset_name="spider", sample_size=1)

    class FakeLogic:
        def get_generation_runner(self):
            return lambda **_: (completion, 7, 0.25, [], [])

        def force_open_span(self):
            return True

        def constrained_temperature(self):
            return 0.0

        def build_dynamic_parser(self, *_):
            return None

        def accuracy_upper_bound(self, *_):
            return 1.0

        def final_accuracy_denominator(self, basis, _):
            return basis

        def accuracy_definition(self):
            return "test"

        def invalid_outputs_excluded(self, *_):
            return False

    monkeypatch.setattr(evaluator, "_ensure_smiles_rdkit_available", lambda: None)
    monkeypatch.setattr(evaluator, "_load_dataset_sample", lambda: [{"question": "q"}])
    monkeypatch.setattr(evaluator, "_setup_environment", lambda _: {})
    monkeypatch.setattr(evaluator, "_benchmark_logic", lambda: FakeLogic())
    monkeypatch.setattr(evaluator, "_format_prompt", lambda _: "prompt")
    monkeypatch.setattr(evaluator, "_get_expected_answer", lambda _: completion)
    monkeypatch.setattr(
        evaluator,
        "_extract_actual_for_example",
        lambda *_: ("SELECT name FROM singer", "test", None),
    )
    monkeypatch.setattr(evaluator, "_is_correct_for_example", lambda *_: True)
    monkeypatch.setattr(evaluator, "_check_syntax_validity", lambda *_args, **_kwargs: (True, [(completion, True)]))
    monkeypatch.setattr(evaluator, "_example_syntax_pass", lambda *_: True)
    monkeypatch.setattr(evaluator, "_accuracy_applicable_for_example", lambda *_: True)
    monkeypatch.setattr(evaluator, "_compute_smiles_aux_metrics", lambda _: {})

    result = evaluator.evaluate_sample(Path("unused/GeneratedCSD.py"))

    output = capsys.readouterr().out
    assert "Sample 1/1 completion begin" in output
    assert completion in output
    assert "Sample 1/1 completion end" in output
    assert result.sample_outputs[0]["full_output"] == completion
