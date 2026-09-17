"""Snapshot test: synthesis prompts must not contain strategy guidance.

The project's constitutional rule (see ~/csd-generation/CLAUDE.md, "Critical
Rule: No Strategy Guidance in Synthesis Prompts") bans the prompt from
recommending tools, suggesting topologies, comparing to anchors, naming
patterns to avoid, or providing structural prescriptions. The model must
discover effective strategies from the task description and tool contracts
alone.

This test renders each prompt with stub inputs and asserts the rendered text
omits the historical guidance phrases that previously crept in.

Empirical motivation: trajectory analysis on prior runs showed that iter 1-2
produced winning strategies far more often than later iterations, because
later iterations carried structured guidance (anchor / single-axis delta /
optimization objective / win condition / families-tried / proven-helpers)
that biased the model away from helpers used in earlier failed attempts.
Stripping those blocks restores the iter-1 prompt shape on every iteration.

If you are adding new guidance, the right move is almost always not to add
it. See CLAUDE.md for the project's reasoning.
"""

from __future__ import annotations

import pytest

from synthesis.generate.prompts import (
    SYSTEM_PROMPT,
    build_initial_prompt,
    build_evaluation_failure_prompt,
    build_verification_error_prompt,
    build_runtime_error_prompt,
    build_compilation_error_prompt,
    build_format_repair_prompt,
)
from synthesis.evaluate.benchmarks.smiles import eval_logic as smiles_eval_logic
from synthesis.evaluate.benchmarks.sql_spider import eval_logic as sql_spider_eval_logic


# Phrases that have appeared in prior versions of the prompts as strategy
# guidance. Each one is a regression marker: if it reappears in a rendered
# prompt, the constitutional rule has been violated.
FORBIDDEN_PHRASES = [
    # Structural directives
    "single-axis delta",
    "Single-axis delta",
    "Build a single-axis delta",
    "do not introduce a new family",
    "Do not introduce a new family",
    "structurally different family",
    "structurally different change",
    "fundamentally different family",
    "differ in at least TWO positions",
    "5-tuple",
    "control topology",
    "control-topology",
    "Refinement discipline",
    "Restart discipline",
    # Anchor / objective framing
    "Pareto-best",
    "anchor for this refinement",
    "anchor on attempt",
    "Optimization objective",
    "optimization objective",
    "total_accuracy = syntax_rate",
    "decomposes as a product",
    # Win-condition predict-before-proposing
    "What counts as a winning delta",
    "win when ALL of",
    "Pre-evaluate",
    "predict its effect on each factor",
    # Working hypothesis / balanced-best
    "Working hypothesis state",
    "balanced-best",
    "Balanced-best",
    "Best-so-far is selected by balanced",
    # Avoid lists / families-tried / proven-helpers
    "Avoid small parameter tweaks",
    "Configurations that did not meet the goal",
    "Avoid repeating these specific patterns",
    "Families already explored",
    "Proven helpers not yet tried",
    "Boring helpers + new topology",
    "boring helpers",
    "exotic helpers",
    # Helper name recommendations (positive prescriptions)
    "Inside-span workhorse",
    "first-inside-token sharpener",
]


def _assert_no_forbidden(rendered: str, label: str) -> None:
    hits = [p for p in FORBIDDEN_PHRASES if p in rendered]
    assert not hits, (
        f"{label} contains forbidden strategy-guidance phrase(s): {hits}\n"
        "See CLAUDE.md 'No Strategy Guidance in Synthesis Prompts'."
    )


def test_system_prompt_has_no_strategy_guidance():
    _assert_no_forbidden(SYSTEM_PROMPT, "SYSTEM_PROMPT")


def test_initial_prompt_has_no_strategy_guidance():
    system, user = build_initial_prompt(
        task_description="Solve math word problems step by step.",
    )
    _assert_no_forbidden(system, "initial system prompt")
    _assert_no_forbidden(user, "initial user prompt")


def test_refinement_prompt_has_no_strategy_guidance():
    system, user = build_evaluation_failure_prompt(
        task_description="Solve math word problems step by step.",
        previous_strategy="// PREV_BODY\ngenerated := generatedPrefix;",
        previous_accuracy=0.12,
        previous_syntax_rate=0.56,
        num_examples=25,
        goal_accuracy=0.31,
        goal_syntax_rate=0.90,
        evaluation_feedback="Accuracy: 12.0%\nSyntax Rate: 56.0%",
    )
    _assert_no_forbidden(system, "refinement system prompt")
    _assert_no_forbidden(user, "refinement user prompt")


def test_refinement_prompt_renders_previous_strategy_and_scores():
    _system, user = build_evaluation_failure_prompt(
        task_description="Solve math word problems step by step.",
        previous_strategy="// PREV_BODY_MARKER",
        previous_accuracy=0.12,
        previous_syntax_rate=0.56,
        num_examples=25,
        goal_accuracy=0.31,
        goal_syntax_rate=0.90,
        evaluation_feedback="Accuracy: 12.0%\nSyntax Rate: 56.0%",
    )
    assert "// PREV_BODY_MARKER" in user
    # previous-attempt scores rendered
    assert "12.0%" in user or "12%" in user
    assert "56.0%" in user or "56%" in user
    # goal rendered
    assert "31.0%" in user or "31%" in user
    assert "90.0%" in user or "90%" in user
    # computed gap is rendered (31.0 - 12.0 = 19.0pp)
    assert "gap" in user.lower()
    assert "19.0pp" in user


def test_refinement_prompt_renders_best_so_far_when_different_from_previous():
    _system, user = build_evaluation_failure_prompt(
        task_description="Solve math word problems step by step.",
        previous_strategy="// PREV_BODY",
        previous_accuracy=0.04,
        previous_syntax_rate=0.20,
        num_examples=25,
        goal_accuracy=0.31,
        goal_syntax_rate=0.90,
        evaluation_feedback="...",
        best_strategy="// BEST_BODY",
        best_accuracy=0.16,
        best_syntax_rate=0.64,
    )
    assert "// PREV_BODY" in user
    assert "// BEST_BODY" in user
    # best scores rendered
    assert "16.0%" in user or "16%" in user
    assert "64.0%" in user or "64%" in user
    # framed as positive anchor, not negative blame
    assert "best" in user.lower()


def test_refinement_prompt_renders_attempt_outcome_ledger():
    _system, user = build_evaluation_failure_prompt(
        task_description="Solve math word problems step by step.",
        previous_strategy="// PREV_BODY",
        previous_accuracy=0.20,
        previous_syntax_rate=0.72,
        num_examples=25,
        goal_accuracy=0.31,
        goal_syntax_rate=0.90,
        evaluation_feedback="...",
        attempt_outcome_ledger=(
            "Use this as empirical search context, not as a recipe.\n"
            "- Attempt 3: accuracy 20.0%, syntax 72.0%; "
            "rationale claim: allowed int() and /; measured effect: syntax unchanged."
        ),
    )

    assert "## Attempt outcome ledger" in user
    assert "Use this as empirical search context" in user
    assert "Attempt 3" in user
    assert "allowed int() and /" in user


def test_refinement_prompt_omits_best_block_when_previous_is_best():
    """If the caller does not pass a separate best, don't render a duplicate."""
    _system, user = build_evaluation_failure_prompt(
        task_description="Solve math word problems step by step.",
        previous_strategy="// ONLY_BODY",
        previous_accuracy=0.16,
        previous_syntax_rate=0.64,
        num_examples=25,
        goal_accuracy=0.31,
        goal_syntax_rate=0.90,
        evaluation_feedback="...",
    )
    assert user.count("// ONLY_BODY") == 1


def test_refinement_prompt_renders_eval_budget_when_set():
    _system, user = build_evaluation_failure_prompt(
        task_description="Solve math word problems step by step.",
        previous_strategy="// PREV",
        previous_accuracy=0.12,
        previous_syntax_rate=0.56,
        num_examples=25,
        goal_accuracy=0.31,
        goal_syntax_rate=0.90,
        evaluation_feedback="...",
        eval_max_seconds_per_example=90.0,
    )
    assert "wall-clock budget" in user
    assert "90" in user


def test_refinement_prompt_omits_eval_budget_when_unset():
    _system, user = build_evaluation_failure_prompt(
        task_description="Solve math word problems step by step.",
        previous_strategy="// PREV",
        previous_accuracy=0.12,
        previous_syntax_rate=0.56,
        num_examples=25,
        goal_accuracy=0.31,
        goal_syntax_rate=0.90,
        evaluation_feedback="...",
    )
    assert "wall-clock budget" not in user


def test_verification_error_prompt_has_no_strategy_guidance():
    """Sibling prompt: same constitutional rule applies."""
    system, user = build_verification_error_prompt(
        task_description="Solve math word problems step by step.",
        previous_strategy="// prev",
        error_message="postcondition violation at line 42",
    )
    _assert_no_forbidden(system, "verification system prompt")
    _assert_no_forbidden(user, "verification user prompt")


def test_runtime_error_prompt_has_no_strategy_guidance():
    system, user = build_runtime_error_prompt(
        previous_strategy="// prev",
        error_traceback="IndexError: list index out of range",
    )
    _assert_no_forbidden(system, "runtime system prompt")
    _assert_no_forbidden(user, "runtime user prompt")


def test_compilation_error_prompt_has_no_strategy_guidance():
    system, user = build_compilation_error_prompt(
        previous_strategy="// prev",
        error_message="undefined identifier 'helpers'",
    )
    _assert_no_forbidden(system, "compilation system prompt")
    _assert_no_forbidden(user, "compilation user prompt")


def test_format_repair_prompt_has_no_strategy_guidance():
    system, user = build_format_repair_prompt(previous_strategy="// prev")
    _assert_no_forbidden(system, "format-repair system prompt")
    _assert_no_forbidden(user, "format-repair user prompt")


# Search-memory leakage guards. The FeedbackLoop previously injected a
# multi-line "Search memory:" block (built by _get_compact_search_memory)
# into the search_memory parameter of refinement/restart prompts. That
# block carried strategy guidance — balanced-best framing, useful-ingredients
# prose, mode-selection ("Near-win"/"Valid-basin"), dual-anchor recipes,
# broad-family preservation directives, and a "Revision check" cookbook.
# After removal, the loop must pass nothing (or an empty string) for
# search_memory, and the rendered prompt must show no trace of those
# blocks.

SEARCH_MEMORY_LEAK_PHRASES = [
    "Search memory:",
    "balanced-best",
    "Balanced-best",
    "useful ingredients",
    "Useful ingredients",
    "single causal axis",
    "single-axis",
    "one causal axis",
    "Near-win refinement mode",
    "Valid-basin refinement mode",
    "Preferred merge/repair",
    "broad family",
    "Broad family",
    "Repeated broad family",
    "Repeated outcome trap",
    "Revision check",
    "Hard delimiter contract",
    "Runtime contract",
    "Repair continuity",
    "Dual-anchor evidence",
    "accuracy anchor",
    "contract anchor",
    "contract/syntax anchor",
]


def _assert_no_search_memory_leakage(rendered: str, label: str) -> None:
    hits = [p for p in SEARCH_MEMORY_LEAK_PHRASES if p in rendered]
    assert not hits, (
        f"{label} contains search-memory leak phrase(s): {hits}\n"
        "_get_compact_search_memory and its callers were removed; "
        "the loop must not pass any 'Search memory:' block into the prompt."
    )


def test_refinement_prompt_default_has_no_search_memory_block():
    """With search_memory unset, no leaked block should appear in the prompt."""
    _system, user = build_evaluation_failure_prompt(
        task_description="Solve math word problems step by step.",
        previous_strategy="// PREV",
        previous_accuracy=0.12,
        previous_syntax_rate=0.56,
        num_examples=25,
        goal_accuracy=0.31,
        goal_syntax_rate=0.90,
        evaluation_feedback="Accuracy: 12.0%\nSyntax Rate: 56.0%",
    )
    _assert_no_search_memory_leakage(user, "refinement user prompt (default)")


def test_feedback_loop_does_not_construct_search_memory():
    """The FeedbackLoop class must not own a _get_compact_search_memory method.

    This test guards against regression: if someone re-introduces the
    search-memory builder, this fails immediately. We check by attribute
    presence rather than by rendering, because the loop ties to GPU/eval
    state that isn't available in unit tests.
    """
    from synthesis.evaluate.feedback_loop import SynthesisPipeline

    forbidden_methods = [
        "_get_compact_search_memory",
        "_get_useful_ingredients_summary",
        "_get_strategy_ingredient_parts",
        "_get_near_win_refinement_summary",
        "_get_valid_basin_refinement_summary",
        "_get_dual_anchor_summary",
        "_get_repeated_outer_structure_summary",
        "_describe_outer_structure_signature",
        "_get_outer_structure_signature",
        "_get_evaluation_history_summary",
        "_get_working_hypothesis_state",
        "_get_verification_refinement_context",
        "_get_repeated_strategy_profile_summary",
        "_get_best_so_far_comparison",
    ]
    present = [m for m in forbidden_methods if hasattr(SynthesisPipeline, m)]
    assert not present, (
        f"SynthesisPipeline still defines removed methods: {present}\n"
        "These were deleted because they injected strategy guidance into "
        "the synthesis prompt via the search_memory / strategy_context "
        "parameters."
    )


NEGATIVE_PROMPT_PHRASES = [
    "Do NOT redeclare out-parameters as locals",
    "Do NOT write `var helpers := new CSDHelpers();`",
    "Do NOT use `CSDHelpers.<Method>` for instance methods",
    "Do not invent visible delimiters",
    "Do not use it as a mid-generation control action",
    "Do NOT output a method signature",
    "no signature, no braces, no markdown fences",
    "do not copy an example shape just because it verifies",
    "Do not write explanations",
    "no reasoning or other text",
]


def test_model_facing_prompt_surfaces_use_positive_contract_language():
    """Model-facing prompt constraints should say what to emit/use."""
    rendered_surfaces = []
    rendered_surfaces.extend(
        [
            ("SYSTEM_PROMPT", SYSTEM_PROMPT),
            ("initial prompt", "\n".join(build_initial_prompt("Solve math."))),
            (
                "evaluation refinement prompt",
                "\n".join(
                    build_evaluation_failure_prompt(
                        task_description="Solve math.",
                        previous_strategy="// PREV",
                        previous_accuracy=0.12,
                        previous_syntax_rate=0.56,
                        num_examples=25,
                        goal_accuracy=0.31,
                        goal_syntax_rate=0.90,
                        evaluation_feedback="Accuracy low.",
                    )
                ),
            ),
            (
                "verification refinement prompt",
                "\n".join(
                    build_verification_error_prompt(
                        task_description="Solve math.",
                        previous_strategy="// PREV",
                        error_message="verification failed",
                    )
                ),
            ),
            (
                "runtime refinement prompt",
                "\n".join(
                    build_runtime_error_prompt(
                        previous_strategy="// PREV",
                        error_traceback="runtime failed",
                    )
                ),
            ),
            (
                "compilation refinement prompt",
                "\n".join(
                    build_compilation_error_prompt(
                        previous_strategy="// PREV",
                        error_message="compilation failed",
                    )
                ),
            ),
            ("format repair prompt", "\n".join(build_format_repair_prompt("// PREV"))),
        ]
    )

    benchmark_example = {
        "prompt": "Generate a molecule.",
        "db_id": "concert_singer",
        "db_info": "# singer ( singer_id , name )",
        "question": "How many singers do we have?",
    }
    spider_expression_only = sql_spider_eval_logic.format_prompt_expression_only(None, benchmark_example)
    assert isinstance(spider_expression_only, str)
    rendered_surfaces.extend(
        [
            (
                "SMILES expression-only prompt",
                smiles_eval_logic.format_prompt_expression_only(None, benchmark_example),
            ),
            (
                "Spider expression-only prompt",
                spider_expression_only,
            ),
        ]
    )

    hits = {
        label: [phrase for phrase in NEGATIVE_PROMPT_PHRASES if phrase in rendered]
        for label, rendered in rendered_surfaces
    }
    hits = {label: phrases for label, phrases in hits.items() if phrases}
    assert not hits

    combined = "\n".join(rendered for _label, rendered in rendered_surfaces)
    assert "Assign the existing out-parameters directly" in combined
    assert "Call instance helper methods as `helpers.<Method>`" in combined
    assert "Every constrained span is visible" in combined
    assert "Call it once at method start" in combined
    assert "Return exactly the Dafny method body" in combined
    assert "Return exactly one line containing the SMILES molecule" in combined
    assert "Only output the SQL quey." in combined
