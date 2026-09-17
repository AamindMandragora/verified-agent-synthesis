"""Test for EvaluationResult._render_mode_examples.

Verifies that the helper emits up to 3 verbatim-rollout blocks per distinct
failure mode (top-4 modes), with no truncation of the GSM prompt or Qwen
output, and no telemetry, counts, or narrative framing.

Selection rule within a mode bucket of size N (by `full_output` length):
  N == 1 → [only sample]
  N == 2 → [shortest, longest]
  N >= 3 → [shortest, median, longest]

Constitutional context: project CLAUDE.md ("No Strategy Guidance in
Synthesis Prompts") bans recommendations / hints / structural directives.
Raw rollouts (PROMPT, model output, extracted answer, correct answer) are
observational evidence, not guidance, and are permitted.
"""

from __future__ import annotations

from synthesis.evaluate.evaluator import EvaluationResult


def _base_sample(prompt_text: str, qwen_output: str, actual: str, expected: str) -> dict:
    return {
        "question": prompt_text[:200],          # mimics upstream 200-char truncation
        "question_full": prompt_text,           # new field — verbatim prompt
        "expected": expected,
        "actual": actual,
        "full_output": qwen_output,
        "is_correct": False,
        "accuracy_applicable": True,
        "contains_delimiters": False,
        "visible_delimiters": False,
        "used_constrained_chunk": False,
        "is_syntax_valid": False,
        "syntax_rate": 0.0,
        "runtime_budget_exceeded": False,
    }


def _make_eval_result(samples: list[dict]) -> EvaluationResult:
    return EvaluationResult(
        success=False,
        accuracy=0.0,
        contains_delimiters=False,
        syntax_rate=0.0,
        num_examples=len(samples),
        num_correct=0,
        total_time_seconds=float(len(samples)),
        sample_outputs=samples,
    )


def test_one_block_per_distinct_mode():
    # mode 1: unterminated_constrained_segment — `<<` present, `>>` absent
    prompt_a = "Janet's ducks lay 16 eggs per day. She eats 3 for breakfast and bakes muffins with 4. How many does she sell at the market?"
    qwen_a = "Janet has 16 eggs. She eats 3, so 13 left. She bakes with 4, so 9 left.\nFinal: <<x = 16 - 3 - 4"
    s_a = _base_sample(prompt_a, qwen_a, actual="13", expected="9")

    # mode 2: missing_constrained_segment — no `<<` at all
    prompt_b = "Sam buys 7 apples at $2 each and pays with a $20 bill. How much change does she get back?"
    qwen_b = "Sam buys 7 apples at $2 each = $14. Change from $20 is $6 plus tax of $1 = $7."
    s_b = _base_sample(prompt_b, qwen_b, actual="7", expected="6")

    # mode 3: malformed_constrained_content — `<<` and `>>` both present, syntax_rate=0
    prompt_c = "A pizza is cut into 8 slices. If 3 people each eat 2, how many slices are left over?"
    qwen_c = "Total slices: 8. Eaten: 3 * 2 = 6 slices.\nFinal: <<???>>"
    s_c = _base_sample(prompt_c, qwen_c, actual="???", expected="2")
    s_c["contains_delimiters"] = True
    s_c["visible_delimiters"] = True

    rendered = _make_eval_result([s_a, s_b, s_c])._render_mode_examples()

    # one block per mode
    assert isinstance(rendered, str) and rendered, f"expected non-empty render, got {rendered!r}"
    assert rendered.count("--- Example of ") == 3, (
        f"expected 3 example blocks (one per distinct mode), got render:\n{rendered}"
    )

    # required field labels appear once per block
    assert rendered.count("PROMPT:") == 3
    assert rendered.count("QWEN OUTPUT:") == 3
    assert rendered.count("STRATEGY EXTRACTED:") == 3
    assert rendered.count("CORRECT ANSWER:") == 3

    # full prompt and full Qwen output appear verbatim (no truncation)
    for prompt in (prompt_a, prompt_b, prompt_c):
        assert prompt in rendered, f"prompt missing verbatim: {prompt!r}"
    for sample in (s_a, s_b, s_c):
        assert sample["full_output"] in rendered, (
            f"qwen full_output missing verbatim: {sample['full_output']!r}"
        )

    # extracted and correct answer fields present, but only as labelled values
    assert "STRATEGY EXTRACTED: 13" in rendered
    assert "STRATEGY EXTRACTED: 7" in rendered
    assert "STRATEGY EXTRACTED: ???" in rendered
    assert "CORRECT ANSWER: 9" in rendered
    assert "CORRECT ANSWER: 6" in rendered
    assert "CORRECT ANSWER: 2" in rendered

    # no telemetry, no percentages, no aggregate counts, no narrative framing
    assert "%" not in rendered, "no percentages allowed (no telemetry)"
    lower = rendered.lower()
    for forbidden in (
        "examples)",        # "(3 examples)" style aggregate counts
        "diagnose",         # narrative framing
        "recommend",        # recommendations
        "we suggest",
        "you should",
        "your previous",    # comparisons to anchor
    ):
        assert forbidden not in lower, f"forbidden framing token present: {forbidden!r}"


def test_render_is_empty_when_no_failed_samples():
    result = EvaluationResult(
        success=True,
        accuracy=1.0,
        contains_delimiters=True,
        syntax_rate=1.0,
        num_examples=0,
        num_correct=0,
        total_time_seconds=0.0,
        sample_outputs=[],
    )
    rendered = result._render_mode_examples()
    assert rendered == "", f"empty samples should render as empty string, got {rendered!r}"


def test_renders_verbatim_full_prompt_not_truncated():
    """Regression: must use question_full (untruncated), not question[:200]."""
    long_prompt = "Janet's farm: " + ("she bought " + "carrots, " * 60) + "now what?"
    assert len(long_prompt) > 400, "test setup: prompt must be longer than upstream 200-char truncation"

    sample = _base_sample(long_prompt, qwen_output="no delim here at all", actual="x", expected="y")
    rendered = _make_eval_result([sample])._render_mode_examples()

    assert long_prompt in rendered, (
        "the full prompt (>400 chars) must appear verbatim; "
        "do not fall back to truncated `question` field"
    )


def test_caps_modes_to_top_four():
    """If more than 4 distinct modes are present, render rollouts from only the top 4 modes."""
    samples: list[dict] = []
    # 5 distinct modes: unterminated, missing, premature, malformed, repetition
    # Counts: unterminated x3, missing x2, premature x2, malformed x1, repetition x1
    for _ in range(3):
        s = _base_sample("p1 unterminated", "x <<incomplete", "a", "b")
        samples.append(s)
    for _ in range(2):
        s = _base_sample("p2 missing", "no delim", "a", "b")
        samples.append(s)
    for _ in range(2):
        s = _base_sample("p3 premature", ">> closed without open", "a", "b")
        samples.append(s)
    s4 = _base_sample("p4 malformed", "<<bogus>>", "a", "b")
    s4["contains_delimiters"] = True
    s4["visible_delimiters"] = True
    samples.append(s4)
    s5 = _base_sample("p5 repetition", "loop loop loop loop loop loop loop loop", "a", "b")
    samples.append(s5)

    rendered = _make_eval_result(samples)._render_mode_examples()
    # Count distinct mode headers (mode name appears after "--- Example of ").
    import re as _re
    distinct_modes = set(_re.findall(r"--- Example of (\S+) ---", rendered))
    assert len(distinct_modes) <= 4, (
        f"top-4 mode cap violated, found modes {distinct_modes}:\n{rendered}"
    )


def test_failure_mode_summary_lists_all_detected_modes():
    """Aggregate mode counts are cheap enough to show completely."""
    samples: list[dict] = []
    for _ in range(3):
        samples.append(_base_sample("p1 unterminated", "x <<incomplete", "a", "b"))
    for _ in range(2):
        samples.append(_base_sample("p2 missing", "no delim", "a", "b"))
    for _ in range(2):
        samples.append(_base_sample("p3 premature", ">> closed without open", "a", "b"))
    s4 = _base_sample("p4 malformed", "<<bogus>>", "a", "b")
    s4["contains_delimiters"] = True
    s4["visible_delimiters"] = True
    samples.append(s4)
    samples.append(_base_sample("p5 repetition", "loop loop loop loop loop loop loop loop", "a", "b"))

    modes = _make_eval_result(samples)._summarize_failure_modes()

    assert len(modes) > 4, f"failure mode summary should not top-k truncate: {modes}"


def test_other_failure_summary_does_not_slice_raw_output_preview():
    long_output = "ordinary wrong output " + " ".join(
        f"sample_specific_token_{idx}" for idx in range(40)
    )
    sample = _base_sample("p other", long_output, "wrong", "right")
    sample["contains_delimiters"] = True
    sample["visible_delimiters"] = True
    sample["is_syntax_valid"] = True
    sample["syntax_rate"] = 1.0

    modes = dict(
        (mode, detail)
        for mode, _count, detail in _make_eval_result([sample])._summarize_failure_modes()
    )

    detail = modes["other_observed_failure"]
    assert "representative rollout examples" in detail
    assert "sample_specific_token" not in detail
    assert long_output[:120] not in detail


# Prefix has >4 words so the early-constrained-entry classifier does NOT fire,
# keeping every sample in exactly one mode bucket (unterminated_constrained_segment).
_PREFIX = "one two three four five six seven "


def _outputs_for_mode(mode_unterminated_outputs: list[str]) -> list[dict]:
    """Helper: build N samples that all classify as `unterminated_constrained_segment`,
    each with a distinct `full_output` (and hence distinct length)."""
    return [
        _base_sample(f"prompt {i}", out, actual="a", expected="b")
        for i, out in enumerate(mode_unterminated_outputs)
    ]


def _picked_outputs_for_mode(rendered: str, mode: str) -> list[str]:
    """Extract the QWEN OUTPUT bodies for blocks rendered under a given mode."""
    import re as _re
    pattern = _re.compile(
        rf"--- Example of {_re.escape(mode)} ---\nPROMPT:\n.*?\n\nQWEN OUTPUT:\n(.*?)\n\nSTRATEGY EXTRACTED:",
        _re.DOTALL,
    )
    return pattern.findall(rendered)


def test_single_sample_mode_renders_one_block():
    samples = _outputs_for_mode([_PREFIX + "<<short"])
    rendered = _make_eval_result(samples)._render_mode_examples()
    assert rendered.count("--- Example of unterminated_constrained_segment ---") == 1


def test_two_sample_mode_renders_shortest_and_longest():
    short_out = _PREFIX + "<<a"
    long_out = _PREFIX + "<<" + ("y" * 200)
    samples = _outputs_for_mode([short_out, long_out])
    rendered = _make_eval_result(samples)._render_mode_examples()
    picked = _picked_outputs_for_mode(rendered, "unterminated_constrained_segment")
    assert len(picked) == 2, f"N=2 bucket should render exactly 2 rollouts, got {len(picked)}"
    assert short_out in picked
    assert long_out in picked


def test_three_sample_mode_renders_shortest_median_longest():
    short_out = _PREFIX + "<<a"
    median_out = _PREFIX + "<<" + ("m" * 50)
    long_out = _PREFIX + "<<" + ("L" * 200)
    samples = _outputs_for_mode([median_out, long_out, short_out])  # unsorted on purpose
    rendered = _make_eval_result(samples)._render_mode_examples()
    picked = _picked_outputs_for_mode(rendered, "unterminated_constrained_segment")
    assert len(picked) == 3, f"N=3 bucket should render exactly 3 rollouts, got {len(picked)}"
    for out in (short_out, median_out, long_out):
        assert out in picked, f"missing rollout: {out!r}"


def test_five_sample_mode_renders_only_three():
    outs = [
        _PREFIX + "<<" + ("c" * 10),
        _PREFIX + "<<" + ("c" * 30),
        _PREFIX + "<<" + ("c" * 50),
        _PREFIX + "<<" + ("c" * 70),
        _PREFIX + "<<" + ("c" * 90),
    ]
    samples = _outputs_for_mode(outs)
    rendered = _make_eval_result(samples)._render_mode_examples()
    picked = _picked_outputs_for_mode(rendered, "unterminated_constrained_segment")
    assert len(picked) == 3, f"N=5 bucket should still cap at 3 rollouts, got {len(picked)}"
    # shortest (10), median (50), longest (90) must be picked under this mode;
    # the two non-extremes (30, 70) must NOT.
    assert outs[0] in picked, "shortest must be picked"
    assert outs[2] in picked, "median must be picked"
    assert outs[4] in picked, "longest must be picked"
    assert outs[1] not in picked, "non-extremes (idx 1) should NOT be picked"
    assert outs[3] not in picked, "non-extremes (idx 3) should NOT be picked"
