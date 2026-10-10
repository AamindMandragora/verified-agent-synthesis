"""CARS and GCD must sample the same way on every model (user, 2026-10-09).

Transformers 5.5.4 fills each sampling value that a caller leaves unset (None) from the model's own
generation_config.json, then from the global defaults. Qwen2.5 ships temperature 0.7, top-p 0.8,
top-k 20 and repetition penalty 1.05 (7B) or 1.1 (1.5B). CARS builds a config with those values unset,
and GCD (SynCode with opp=False) calls generate() with keyword values only, so both picked up
Qwen2.5's values. Even greedy GCD on GSM and Spider got the repetition penalty. Qwen3.5 ships no such
file, so it got the global defaults: temperature 1.0, top-p 1.0, top-k 50, repetition penalty 1.0.
The user chose to give every model those global defaults. CRANE and IterGen build their config from
config.json, and the unconstrained lanes pass explicit vLLM SamplingParams, so neither reads the file.
"""
import inspect

from transformers import GenerationConfig

from synthesis.evaluate import run_legacy_fixed_strategy as runner
from synthesis.evaluate.run_legacy_fixed_strategy import _use_global_sampling_defaults


class _FakeModel:
    def __init__(self, generation_config):
        self.generation_config = generation_config


def _effective(model, **passed_values):
    """The values generate() uses: passed values first, then the model's file, then the global defaults."""
    passed = GenerationConfig(**passed_values)
    passed.update(**model.generation_config.to_dict(), defaults_only=True, allow_custom_entries=True)
    passed.update(**model.generation_config._get_default_generation_params(), defaults_only=True)
    return passed.temperature, passed.top_p, passed.top_k, passed.repetition_penalty


def _cars_effective(model):
    """The config CARS passes (cars/lib.py ConstrainedModel._generate)."""
    return _effective(model, max_new_tokens=8, num_return_sequences=1, do_sample=True, top_k=None)


def test_qwen25_cars_samples_with_the_global_defaults():
    model = _FakeModel(GenerationConfig.from_pretrained("Qwen/Qwen2.5-7B-Instruct"))
    assert _cars_effective(model) == (0.7, 0.8, 20, 1.05)  # the bug: Qwen2.5's own defaults
    _use_global_sampling_defaults(model, tag="legacy-cars")
    assert _cars_effective(model) == (1.0, 1.0, 50, 1.0)


def test_qwen25_gcd_greedy_has_no_repetition_penalty():
    model = _FakeModel(GenerationConfig.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct"))
    assert _effective(model, do_sample=False)[3] == 1.1  # the bug: greedy GCD got Qwen2.5's penalty
    _use_global_sampling_defaults(model, tag="legacy-gcd")
    assert _effective(model, do_sample=False)[3] == 1.0


def test_qwen25_gcd_smiles_samples_with_the_global_defaults():
    model = _FakeModel(GenerationConfig.from_pretrained("Qwen/Qwen2.5-7B-Instruct"))
    _use_global_sampling_defaults(model, tag="legacy-gcd")
    assert _effective(model, do_sample=True, temperature=1.0) == (1.0, 1.0, 50, 1.0)


def test_qwen25_keeps_its_end_tokens():
    model = _FakeModel(GenerationConfig.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct"))
    _use_global_sampling_defaults(model, tag="legacy-cars")
    assert model.generation_config.eos_token_id == [151645, 151643]


def test_model_without_sampling_defaults_is_unchanged():
    model = _FakeModel(GenerationConfig())  # Qwen3.5 has no generation_config.json
    before = _cars_effective(model)
    _use_global_sampling_defaults(model, tag="legacy-cars")
    assert _cars_effective(model) == before == (1.0, 1.0, 50, 1.0)


def test_cars_and_gcd_runners_clear_the_model_defaults():
    assert "_use_global_sampling_defaults(" in inspect.getsource(runner.run_cars_legacy_adapter)
    assert "_use_global_sampling_defaults(" in inspect.getsource(runner.run_gcd_legacy_adapter)
