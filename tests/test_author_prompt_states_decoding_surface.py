"""The author's prompt must say where in the output the run starts.

A strategy reaches the constrained region one of two ways:

  - it opens the span itself with `OpenConstrainedSpan`, which appends a
    literal `<<` to the output (`force_open_span=False`); or
  - the runtime already opened the span, so the output starts with a visible
    `<<` and the strategy is already inside it (`force_open_span=True`).

Spider and SMILES are the second kind. The author's prompt never used to say
so, and authors kept writing strategies whose only route into constrained mode
was waiting for a `<<` that had already gone by.
"""

from __future__ import annotations

from synthesis.generate.prompts import build_initial_prompt


def _render_user_prompt(force_open_span: bool) -> str:
    _system_prompt, user_prompt = build_initial_prompt(
        task_description="Write a syntactically valid SQL query for the question.",
        force_open_span=force_open_span,
    )
    return user_prompt


def test_runtime_opened_span_prompt_says_the_span_is_already_open():
    user = _render_user_prompt(force_open_span=True)

    assert "starts inside a span the runtime already opened" in user
    assert "CloseConstrainedSpan" in user
    # It must still say how to open one, for a strategy that closes and reopens.
    assert "OpenConstrainedSpan" in user
    assert "the opening `<<` is already there" in user


def test_self_opened_span_prompt_describes_the_open_constrained_span_route():
    user = _render_user_prompt(force_open_span=False)

    assert "OpenConstrainedSpan" in user
    assert "starts outside the constrained region" in user
    # The already-open warning must not leak into the other prompt.
    assert "already opened" not in user


def test_the_two_surface_renderings_differ():
    assert _render_user_prompt(force_open_span=True) != _render_user_prompt(
        force_open_span=False
    )
