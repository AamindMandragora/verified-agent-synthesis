import pytest

from synthesis.presets import get_synthesis_preset, resolve_model_name


def test_get_synthesis_preset_returns_gsm_defaults():
    preset = get_synthesis_preset("gsm_symbolic")

    assert preset.output_name == "gsm_crane_csd"
    assert preset.eval_max_steps == 2048
    assert preset.min_accuracy == 0.5
    assert preset.min_format_rate == 1.0
    assert preset.min_syntax_rate == 1.0
    assert "Pure arithmetic expressions are valid" in preset.task_description


def test_resolve_model_name_prefers_explicit_model():
    assert resolve_model_name(model="custom/model", model_preset="qwen3b") == "custom/model"


def test_unknown_model_preset_raises_helpful_error():
    with pytest.raises(ValueError, match="Unknown model preset"):
        resolve_model_name(model_preset="unknown")
