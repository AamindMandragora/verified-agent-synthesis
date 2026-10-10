"""The HuggingFace eval backend must not take Qwen2.5's sampling values (user, 2026-10-10).

Qwen2.5's generation_config.json sets repetition_penalty 1.1 / 1.05. On the HF backend,
GenerateUnconstrainedChunk calls model.generate(do_sample=False), which still applies that penalty.
create_huggingface_lm must clear the model's sampling values right after loading, with the same
helper the CARS and GCD baselines use.
"""
import inspect

from synthesis.evaluate.benchmarks.common import model_utils


def test_helper_lives_in_model_utils_and_baselines_reuse_it():
    from synthesis.evaluate import run_legacy_fixed_strategy as legacy

    assert legacy._use_global_sampling_defaults is model_utils._use_global_sampling_defaults


def test_create_huggingface_lm_clears_model_sampling_values_after_eval():
    src = inspect.getsource(model_utils.create_huggingface_lm)
    eval_at = src.index("model.eval()")
    clear_at = src.index("_use_global_sampling_defaults(model")
    assert clear_at > eval_at


def test_helper_clears_repetition_penalty():
    class Cfg:
        temperature = 0.7
        top_p = 0.8
        top_k = 20
        repetition_penalty = 1.1
        eos_token_id = [151645, 151643]

    class M:
        generation_config = Cfg()

    m = M()
    model_utils._use_global_sampling_defaults(m, tag="hf-eval")
    assert m.generation_config.repetition_penalty is None
    assert m.generation_config.top_k is None
    assert m.generation_config.eos_token_id == [151645, 151643]
