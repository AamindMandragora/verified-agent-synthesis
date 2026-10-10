"""AppendTaskGuidance must deliver the author's whole guidance to the eval model.

A silent 1200-character cut used to drop the tail (spider_2b attempt 35 lost
all of its worked examples), and the author was never told.
"""

from synthesis.evaluate.benchmarks.common.model_utils import (
    _TaskGuidanceState,
    _TensorizedLMBase,
)


class _EchoTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **_):
        return "\n".join(f"<{m['role']}>{m['content']}" for m in messages)


def _chat_lm():
    lm = object.__new__(_TensorizedLMBase)
    lm._task_guidance = _TaskGuidanceState()
    lm._structured_prompt = None
    lm._chat_messages = [{"role": "user", "content": "Question: how many singers?"}]
    lm._last_prompt_contract = {}
    lm.model_name = "Qwen/Qwen3.5-2B"
    lm.tokenizer = _EchoTokenizer()
    return lm


def test_long_guidance_reaches_the_rendered_prompt_whole():
    head = "Rule: 'how many' means count ( * ). "
    tail = "Worked reading: 'How many singers?' is select count ( * ) from singer. END_MARKER"
    guidance = head + ("x" * 3000) + tail

    lm = _chat_lm()
    lm.AppendTaskGuidance(guidance)

    assert lm.task_guidance == guidance
    assert tail in lm.instruction_text
    assert tail in lm._chat_messages[-1]["content"]
