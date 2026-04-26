"""
Shared synthesis presets for dataset-specific helper scripts.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SynthesisPreset:
    """Dataset-specific defaults for ``run_synthesis.py`` wrappers."""

    dataset: str
    output_name: str
    task_description: str
    min_accuracy: float
    min_format_rate: float
    min_syntax_rate: float
    eval_sample_size: int
    eval_max_steps: int

    def to_cli_args(
        self,
        *,
        model_name: str,
        max_iterations: int,
        temperature: float,
        device: str,
        output_name: str | None = None,
        min_accuracy: float | None = None,
        min_format_rate: float | None = None,
        min_syntax_rate: float | None = None,
        eval_sample_size: int | None = None,
        eval_max_steps: int | None = None,
    ) -> list[str]:
        """Build ``run_synthesis.py`` CLI arguments for this preset."""
        return [
            "--task",
            self.task_description,
            "--dataset",
            self.dataset,
            "--output-name",
            output_name or self.output_name,
            "--model",
            model_name,
            "--temperature",
            str(temperature),
            "--device",
            device,
            "--max-iterations",
            str(max_iterations),
            "--min-accuracy",
            str(self.min_accuracy if min_accuracy is None else min_accuracy),
            "--min-format-rate",
            str(self.min_format_rate if min_format_rate is None else min_format_rate),
            "--min-syntax-rate",
            str(self.min_syntax_rate if min_syntax_rate is None else min_syntax_rate),
            "--eval-sample-size",
            str(self.eval_sample_size if eval_sample_size is None else eval_sample_size),
            "--eval-max-steps",
            str(self.eval_max_steps if eval_max_steps is None else eval_max_steps),
        ]


MODEL_PRESETS = {
    "qwen3b": "Qwen/Qwen2.5-Coder-3B-Instruct",
    "qwen7b": "Qwen/Qwen2.5-Coder-7B-Instruct",
}


DATASET_PRESETS = {
    "gsm_symbolic": SynthesisPreset(
        dataset="gsm_symbolic",
        output_name="gsm_crane_csd",
        task_description=(
            "Generate short symbolic mathematical expressions for GSM-Symbolic "
            "reasoning. The parser enforces a strict arithmetic expression grammar "
            "with numeric constants and optional variables. CRITICAL RULES: "
            "1. Pure arithmetic expressions are valid; the final << >> segment may "
            "be either a single expression or a single equation. "
            "2. The evaluator computes the numeric value of the final constrained "
            "expression, so the model does not need to simplify all the way to a "
            "standalone numeral. "
            "3. Use variables only when they genuinely help; pure numeric "
            "expressions like 16 * 8.5 + 4 * 10.5 + 13 are allowed. "
            "4. Preserve numeric values exactly from the problem statement. Do "
            "not round or truncate decimals like 8.5 into 8. "
            "5. The constrained answer segment should stay short, compact, and "
            "mathematically meaningful. "
            "6. Prefer exactly one final << >> answer span; avoid emitting extra "
            "delimiter windows after the answer, and do not mention << or >> in "
            "the free-form reasoning text. "
            "7. The final constrained segment must be complete before the closing "
            "delimiter."
        ),
        min_accuracy=0.5,
        min_format_rate=1.0,
        min_syntax_rate=1.0,
        eval_sample_size=10,
        eval_max_steps=2048,
    ),
    "folio": SynthesisPreset(
        dataset="folio",
        output_name="folio_csd",
        task_description=(
            "Generate first-order logic formulas for FOLIO reasoning. The parser "
            "enforces a strict FOL grammar with quantifiers, predicates, constants, "
            "and logical connectives. CRITICAL RULES: "
            "1. Quantifiers use {forall} and {exists} with single lowercase "
            "variables. "
            "2. Predicates are uppercase/camel-case and constants are lowercase. "
            "3. Use simple well-typed formulas over overly deep nesting. "
            "4. Parentheses must stay balanced and formulas must be complete. "
            "5. The final constrained segment should contain the answer-bearing "
            "formula only."
        ),
        min_accuracy=0.5,
        min_format_rate=0.8,
        min_syntax_rate=0.8,
        eval_sample_size=10,
        eval_max_steps=1500,
    ),
    "pddl": SynthesisPreset(
        dataset="pddl",
        output_name="pddl_csd",
        task_description=(
            "Generate a PDDL planning strategy for Blocks World problems. The "
            "grammar enforces a strict sequence of actions: (pick-up X), "
            "(put-down X), (stack X Y), (unstack X Y). CRITICAL RULES: "
            "1. The constrained answer contains only actions. "
            "2. Actions must satisfy their preconditions. "
            "3. Plans are evaluated by simulation and should achieve the stated "
            "goal. "
            "4. Prefer short correct plans over verbose ones."
        ),
        min_accuracy=0.3,
        min_format_rate=0.5,
        min_syntax_rate=0.5,
        eval_sample_size=5,
        eval_max_steps=128,
    ),
    "sygus_slia": SynthesisPreset(
        dataset="sygus_slia",
        output_name="sygus_slia_csd",
        task_description=(
            "Generate a string-manipulation strategy for SyGuS SLIA problems. "
            "The grammar enforces a strict S-expression format using SLIA string "
            "operations and integer arithmetic. CRITICAL RULES: "
            "1. The constrained answer is a single complete S-expression. "
            "2. Variables are bare identifiers and string literals are quoted. "
            "3. Integer arguments must be valid integer expressions. "
            "4. Prefer compact correct expressions over unnecessary nesting."
        ),
        min_accuracy=0.3,
        min_format_rate=0.5,
        min_syntax_rate=0.5,
        eval_sample_size=5,
        eval_max_steps=256,
    ),
}


def get_synthesis_preset(dataset: str) -> SynthesisPreset:
    """Return the preset for a dataset or raise a helpful error."""
    try:
        return DATASET_PRESETS[dataset]
    except KeyError as exc:
        supported = ", ".join(sorted(DATASET_PRESETS))
        raise ValueError(f"Unknown synthesis preset '{dataset}'. Expected one of: {supported}") from exc


def resolve_model_name(model: str | None = None, model_preset: str | None = None) -> str:
    """Resolve a model alias, preferring an explicit model name."""
    if model:
        return model
    preset_name = model_preset or "qwen7b"
    try:
        return MODEL_PRESETS[preset_name]
    except KeyError as exc:
        supported = ", ".join(sorted(MODEL_PRESETS))
        raise ValueError(f"Unknown model preset '{preset_name}'. Expected one of: {supported}") from exc
