import importlib.util
import sys
import types
from pathlib import Path


def _install_package(monkeypatch, name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__path__ = []
    monkeypatch.setitem(sys.modules, name, module)
    return module


def _load_eval_logic(monkeypatch):
    for name in [
        "synthesis",
        "synthesis.evaluate",
        "synthesis.evaluate.benchmarks",
        "synthesis.evaluate.benchmarks.smiles",
    ]:
        _install_package(monkeypatch, name)

    generation = types.ModuleType("synthesis.evaluate.benchmarks.smiles.generation")
    generation.run_crane_csd = lambda *args, **kwargs: None
    monkeypatch.setitem(
        sys.modules,
        "synthesis.evaluate.benchmarks.smiles.generation",
        generation,
    )

    path = (
        Path(__file__).resolve().parents[1]
        / "synthesis"
        / "evaluate"
        / "benchmarks"
        / "smiles"
        / "eval_logic.py"
    )
    spec = importlib.util.spec_from_file_location(
        "synthesis.evaluate.benchmarks.smiles.eval_logic",
        path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def _example():
    return {
        "prompt": "Generate a molecule.\nMolecule: CCO",
        "class_name": "acrylates",
        "grammar_text": 'start: "C"',
        "prompt_exemplars": [],
    }


def test_smiles_prompts_do_not_request_visible_delimiters(monkeypatch):
    eval_logic = _load_eval_logic(monkeypatch)
    example = _example()

    prompts = [
        eval_logic.format_prompt(None, example),
        eval_logic.format_prompt_expression_only(None, example),
        eval_logic.format_prompt_chain_of_thought(None, example),
    ]

    for prompt in prompts:
        assert "<<" not in prompt
        assert ">>" not in prompt
        assert "<<SMILES>>" not in prompt


def test_smiles_runs_with_a_runtime_opened_span(monkeypatch):
    eval_logic = _load_eval_logic(monkeypatch)

    assert eval_logic.force_open_span() is True


def test_smiles_generation_starts_inside_hidden_constrained_chunk(monkeypatch):
    eval_logic = _load_eval_logic(monkeypatch)
    captured = {}

    def fake_run_crane_csd(*args, **kwargs):
        captured.update(kwargs)
        return "C", 1, 0.0, [("C", True)], []

    generation = sys.modules["synthesis.evaluate.benchmarks.smiles.generation"]

    monkeypatch.setattr(generation, "run_crane_csd", fake_run_crane_csd)

    runner = eval_logic.get_generation_runner()
    runner(
        env={},
        prompt_text="Molecule:",
        max_steps=4,
        step_token_budget=1,
        grammar_file="dummy.lark",
        dynamic_parser=None,
    )

    assert captured["force_open_span"] is True
