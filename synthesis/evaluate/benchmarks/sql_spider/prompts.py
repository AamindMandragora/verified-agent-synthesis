"""Spider prompt templates shared across evaluator formatting modes."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, ClassVar


_PROMPT_LOG = logging.getLogger(__name__)


class SpiderPromptRenderError(RuntimeError):
    """Raised when a Spider model-specific prompt cannot be rendered."""


@dataclass(frozen=True)
class SpiderPromptParts:
    """Immutable Spider task content plus model-specific rendering state."""

    task_text: str
    answer_cue: str = "SQL:"
    guidance: str | None = None
    model_name: str | None = None

    GUIDANCE_HEADER: ClassVar[str] = "Additional task guidance from CSD:"

    @staticmethod
    def _compose(task_text: str, answer_cue: str, guidance: str | None) -> str:
        if not guidance:
            return task_text + answer_cue
        return (
            task_text.rstrip()
            + "\n\n"
            + SpiderPromptParts.GUIDANCE_HEADER
            + "\n"
            + guidance.strip()
            + "\n"
            + answer_cue
        )

    @property
    def raw_text(self) -> str:
        return self._compose(self.task_text, self.answer_cue, self.guidance)

    @property
    def user_content(self) -> str:
        return self.raw_text

    def __str__(self) -> str:
        return self.raw_text

    def with_guidance(self, guidance: str) -> "SpiderPromptParts":
        return type(self)(
            self.task_text,
            answer_cue=self.answer_cue,
            guidance=guidance,
            model_name=self.model_name,
        )

    def with_model_name(self, model_name: str | None) -> "SpiderPromptParts":
        """Return the same immutable task parts bound to an evaluator model."""
        return type(self)(
            self.task_text,
            answer_cue=self.answer_cue,
            guidance=self.guidance,
            model_name=model_name,
        )

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            return self.raw_text == other
        if isinstance(other, SpiderPromptParts):
            return (
                self.task_text,
                self.answer_cue,
                self.guidance,
                self.model_name,
            ) == (
                other.task_text,
                other.answer_cue,
                other.guidance,
                other.model_name,
            )
        return NotImplemented

    def render_for_model(
        self,
        tokenizer: Any,
        *,
        model_name: str | None = None,
    ) -> str:
        identity = (model_name or self.model_name or "").lower()
        identity = identity.replace("-", "_").replace(".", "_")
        is_qwen35 = "qwen3_5" in identity or "qwen35" in identity
        family = "qwen3.5" if is_qwen35 else (
            "qwen2.5" if "qwen2_5" in identity else "unknown"
        )
        if not is_qwen35:
            rendered = self.raw_text
            _PROMPT_LOG.debug(
                "[spider-prompt] family=%s mode=raw thinking_disabled=%s "
                "guidance_present=%s raw_chars=%d rendered_chars=%d",
                family,
                False,
                bool(self.guidance),
                len(self.raw_text),
                len(rendered),
            )
            return rendered
        try:
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": self.user_content}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except Exception as exc:
            _PROMPT_LOG.error(
                "[spider-prompt] render_failed family=%s error_type=%s",
                family,
                type(exc).__name__,
            )
            raise SpiderPromptRenderError(
                "Spider Qwen3.5 chat-template rendering failed"
            ) from exc
        if not isinstance(rendered, str):
            _PROMPT_LOG.error(
                "[spider-prompt] render_failed family=%s error_type=non_string_result",
                family,
            )
            raise SpiderPromptRenderError(
                "Spider chat template must return a string"
            )
        _PROMPT_LOG.debug(
            "[spider-prompt] family=%s mode=chat thinking_disabled=%s "
            "guidance_present=%s raw_chars=%d rendered_chars=%d",
            family,
            True,
            bool(self.guidance),
            len(self.raw_text),
            len(rendered),
        )
        return rendered

    def render_for_model_with_contract(
        self,
        tokenizer: Any,
        *,
        model_name: str | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Render and return safe metadata for the branch that actually succeeded."""
        identity = (model_name or self.model_name or "").lower()
        identity = identity.replace("-", "_").replace(".", "_")
        is_qwen35 = "qwen3_5" in identity or "qwen35" in identity
        family = "qwen3.5" if is_qwen35 else (
            "qwen2.5" if "qwen2_5" in identity else "unknown"
        )
        rendered = self.render_for_model(tokenizer, model_name=model_name)
        contract = {
            "renderer": "spider",
            "family": family,
            "mode": "chat" if is_qwen35 else "raw",
            "template_used": bool(is_qwen35),
            "raw_prompt": not is_qwen35,
            "chat_message_count": 1 if is_qwen35 else 0,
            "user_message_count": 1 if is_qwen35 else 0,
            "add_generation_prompt": True if is_qwen35 else False,
            "enable_thinking": False if is_qwen35 else None,
            "render_succeeded": True,
            "prompt_chars": len(rendered),
        }
        _PROMPT_LOG.debug(
            "[spider-prompt] contract family=%s mode=%s template=%s "
            "thinking=%s prompt_chars=%d",
            family,
            contract["mode"],
            contract["template_used"],
            contract["enable_thinking"],
            contract["prompt_chars"],
        )
        return rendered, contract


_SPIDER_FEW_SHOT = (
    "Example:\n"
    "db_id: concert_singer\n"
    "db_info: # singer ( singer_id , name , country , age )\n"
    "question: How many singers do we have?\n"
)


def format_spider_prompt(
    example: dict[str, Any],
    *,
    instruction: str,
    few_shot_answer_line: str,
) -> SpiderPromptParts:
    """Build a Spider task prompt from shared schema/question blocks."""
    db_id = example.get("db_id", "")
    db_info = example.get("db_info", "")
    question = example.get("question", "")
    task_text = (
        "You are given a database schema and a question. "
        f"{instruction}\n\n"
        f"{_SPIDER_FEW_SHOT}"
        f"{few_shot_answer_line}\n\n"
        f"db_id: {db_id}\n"
        f"db_info: {db_info}\n"
        f"question: {question}\n"
    )
    return SpiderPromptParts(task_text, answer_cue="SQL: ")


def format_spider_itergen_aligned_prompt(example: dict[str, Any]) -> SpiderPromptParts:
    """IterGen's EXACT Spider prompt for fair head-to-head comparison.

    IterGen feeds instruct models a single user turn whose content is built as
    ``f"db_id: {db_id}\\ndb_info: {db_info}\\nquestion: {question}"`` + the
    literal suffix ``" Only output the SQL quey. \\nSQL:"`` (the ``quey`` typo
    and leading space are IterGen's own and are preserved byte-for-byte).

    The shared renderer consumes these parts. Qwen3.5 receives one user turn
    through the tokenizer chat template with thinking disabled; Qwen2.5 and
    unknown model families receive the raw composed text. Keeping task wording
    separate from delivery lets the fixed adapter and evaluator use one source
    of truth.

    No few-shot example and no ``<< >>`` instruction: span opening must come from
    a FORCING strategy (OpenConstrainedSpan), not from the model emitting ``<<``.
    """
    db_id = example.get("db_id", "")
    db_info = example.get("db_info", "")
    question = example.get("question", "")
    task_text = (
        f"db_id: {db_id}\n"
        f"db_info: {db_info}\n"
        f"question: {question} Only output the SQL quey. \n"
    )
    return SpiderPromptParts(task_text)


def format_spider_messages(
    example: dict[str, Any],
    *,
    instruction: str,
    few_shot_answer_line: str,
) -> list[dict]:
    """Multi-turn chat delivery of the same Spider prompt.

    The single inline few-shot example is delivered as a user/assistant turn
    pair instead of being flattened into one user message. On Spider-1.5B
    unconstrained this lifted accuracy 38.0% -> 44.0% (+6pp) over the flattened
    form with zero content change. Mirrors the GSM multi-turn fix.
    """
    db_id = example.get("db_id", "")
    db_info = example.get("db_info", "")
    question = example.get("question", "")
    # The few-shot example schema/question as a clean user turn (strip the
    # "Example:\n" label and trailing newline used only in the flattened form).
    example_user = _SPIDER_FEW_SHOT.removeprefix("Example:\n").rstrip("\n")
    real_user = f"db_id: {db_id}\ndb_info: {db_info}\nquestion: {question}"
    return [
        {"role": "system", "content": "You are given a database schema and a question. " + instruction},
        {"role": "user", "content": example_user},
        {"role": "assistant", "content": few_shot_answer_line},
        {"role": "user", "content": real_user},
    ]
