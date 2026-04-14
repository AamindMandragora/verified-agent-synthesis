"""
Prompt templates for Python-first CSD strategy generation.

The model now writes a Python method body for `GeneratedAgentTemplate.py`.
That Python is later transpiled to Dafny for verification and compilation.
"""

SYSTEM_PROMPT = """\
You are an expert in formal verification and constrained decoding for language models.
You are generating the BODY of a Python function, not a full file.

The surrounding template is:

  def MyCSDStrategy(lm: LM, parser: Parser, prompt: Prefix, maxSteps: int, eosToken: Token) -> tuple[Prefix, int]:
      delim = Delimiter(LeftDelimiter, RightDelimiter)
      helpers = CSDHelpers(lm, parser, delim)
      helpers.DelimitersInLMAlways()
      lm.ValidTokensIdsLogitsAlways()
      generated = []   # free-form text outside the final constrained answer segment
      answer = []      # constrained answer content only, without delimiters
      stepsLeft = maxSteps - 2
      [YOUR BODY]
      helpers.FinalizeDelimitedAnswer(generated, answer)
      generated = generated + [LeftDelimiter] + answer + [RightDelimiter]
      remainingSteps = stepsLeft
      return generated, remainingSteps

Your output must therefore be ONLY Python statements for [YOUR BODY].
Do not write the function signature, imports, markdown fences, or a full file.
Do not redeclare `delim`, `helpers`, `generated`, `answer`, or `stepsLeft`.
Do not assign to `remainingSteps`; the template already does that.
Do not output only comments or only invariants. The body must execute real decoding steps.

Important semantic facts:
- `generated` is the expressive free-form prefix outside the final answer segment.
- `answer` is the constrained answer content only. The template will append `<<` and `>>` around it at the end.
- `prompt`, `generated`, and `answer` are Python lists of tokens.
- `Token` is a string-like token value.
- `helpers.ExpressiveStep(prompt, generated, stepsLeft)` returns `(next_token, new_steps)` for free-form output outside the final answer segment, while masking delimiter tokens.
- `helpers.ConstrainedAnswerStep(prompt, generated, answer, stepsLeft)` returns `(next_token, new_steps)` and extends the constrained answer channel.
- The parser governs `answer`, not the full `generated` output.
- The template requires `parser.IsValidPrefix(answer)` and `len(answer) > 0` by the time the body finishes, and then wraps `answer` as the final `<< ... >>` segment.
- A simple unconstrained-only loop is invalid because it cannot satisfy the final constrained-answer contract.
- The final `<< ... >>` segment is the answer-bearing segment used for grading; it should usually converge to a short arithmetic expression or equation whose right-most numeric value is the answer.
- Use `generated = generated + [next_token]` after each step call.
- Use `answer = answer + [next_token]` after constrained answer steps.
- Use `stepsLeft = new_steps` after each step call.

Python subset rules:
- Use normal Python syntax: `while ...:`, `if ...:`, `and`, `or`, `not`, `==`, `!=`.
- Use Python comments beginning with `#`.
- If you need loop invariants for the transpiler, put them as comments IMMEDIATELY above the `while`, e.g.:
    # invariant lm.ValidTokensIdsLogits()
    # invariant 0 <= stepsLeft <= maxSteps - 2
    # invariant helpers.ConstrainedWindowValid(generated)
    # invariant parser.IsValidPrefix(answer)
    # invariant |generated| + |answer| + stepsLeft <= maxSteps - 2
    # decreases stepsLeft
    while stepsLeft > 0 and not parser.IsCompletePrefix(answer):
        ...
- Important: the loop-invariant comments are passed directly to Dafny, so those comments must use Dafny syntax (`|generated|`, `==>`, etc.), even though the executable body uses Python syntax.
- Prefer `len(generated)` in Python; the transpiler will lower it to Dafny length syntax.
- Do not use Python `for` loops. Use `while` loops only.
- Do not use list comprehensions, lambdas, helper functions, or nested function definitions.
- Do not use `break`/`continue` unless truly necessary.
- Do not use exceptions for control flow.
- If you branch between different step choices, predeclare branch outputs before the `if`, e.g.:
    next_token = eosToken
    new_steps = stepsLeft
    if ...:
        next_token, new_steps = helpers.ConstrainedAnswerStep(prompt, generated, answer, stepsLeft)
    else:
        next_token, new_steps = helpers.ExpressiveStep(prompt, generated, stepsLeft)
- Every emitted token should come from a helper step call and should consume budget.
- The body is invalid unless it contains at least one call to `helpers.ConstrainedAnswerStep(...)`.

Required rationale block:
Your output MUST begin with:

# CSD_RATIONALE_BEGIN
# <short explanation>
# CSD_RATIONALE_END

Then write the Python statements for the body.
"""


INITIAL_GENERATION_PROMPT = """\
Generate a Python strategy body for this use-case:

Use-case description: {task_description}

Requirements:
- Output ONLY the Python body inserted into `MyCSDStrategy`.
- Start with the required rationale block using `#` comments.
- Use a `while` loop with these exact invariants as preceding comments:
    # invariant lm.ValidTokensIdsLogits()
    # invariant 0 <= stepsLeft <= maxSteps - 2
    # invariant helpers.ConstrainedWindowValid(generated)
    # invariant parser.IsValidPrefix(answer)
    # invariant |generated| + |answer| + stepsLeft <= maxSteps - 2
    # decreases stepsLeft
- The answer-bearing part is `answer`, not `generated`.
- You must ensure `parser.IsValidPrefix(answer)` at all times and leave `answer` nonempty before the body exits.
- The body must contain at least one executable `helpers.ConstrainedAnswerStep(...)` call and must update `answer`.
- If you emit expressive free-form text in `generated`, use `helpers.ExpressiveStep(...)`, not `helpers.UnconstrainedStep(...)`.
- If you branch between step types, write Python tuple-unpacking assignments such as:
    next_token = eosToken
    new_steps = stepsLeft
    if ...:
        next_token, new_steps = helpers.ConstrainedAnswerStep(prompt, generated, answer, stepsLeft)
- Append tokens with `generated = generated + [next_token]`.
- Append constrained answer tokens with `answer = answer + [next_token]`.
- Update remaining budget with `stepsLeft = new_steps`.
- Do NOT write `remainingSteps = stepsLeft`.
- Reserve enough budget so the body cannot spend all steps on free-form text before producing a nonempty constrained `answer`.
- If you keep a free-form budget or exploration budget, include a loop invariant that makes that reservation provable.
- Novelty requirements:
  - Do NOT produce unconstrained-only decoding.
  - Do NOT produce a basic CRANE-like window switch around `helpers.InsideDelimitedWindow(...)`.
  - Do NOT produce a simple rollback-to-valid-prefix strategy.
  - Do NOT use `helpers.UnconstrainedStep(...)`; use `helpers.ExpressiveStep(...)` for free-form output so delimiter control stays explicit.
  - Maintain at least two extra local state variables beyond `generated`, `answer`, `stepsLeft`, `next_token`, and `new_steps`.
  - Use multiple interacting progress signals to decide what to do next; do not drive the whole strategy from a single inside/outside test.
  - Favor control policies that evolve over time rather than a one-shot "free-form first, constrained later" shell.
- Do NOT use `parser.IsCompletePrefix(generated)`; the parser only governs `answer`.
"""


VERIFICATION_ERROR_REFINEMENT_PROMPT = """\
Your previous Python strategy body failed verification after being transpiled to Dafny.

Previous attempt:
```python
{previous_strategy}
```

Verification error:
```
{error_message}
```

Fix the Python body while preserving the overall strategy when possible.

Rules:
- Output ONLY a corrected Python body, not a full file.
- Start with the required rationale block using `#` comments.
- Keep the strategy within the supported Python subset from the system prompt.
- Use `# invariant ...` and `# decreases ...` comments immediately above `while` loops.
- Do not redeclare `delim`, `helpers`, `generated`, `answer`, or `stepsLeft`.
- Do not assign to `remainingSteps`.

Common fixes:
- If you used Dafny syntax like `:=`, replace it with Python `=`.
- If you used `&&`, `||`, or `!`, replace them with `and`, `or`, and `not`.
- If you used `//` comments, replace them with `#` comments.
- If the verifier complained that branch-local step outputs were undefined, predeclare `next_token` and `new_steps` before the `if`.
- The repaired body must preserve the final constrained-answer guarantee and remain nontrivial.
- Keep the answer-bearing segment in `answer`; if you emit free-form text, do it through `helpers.ExpressiveStep(...)`.
"""


RUNTIME_ERROR_REFINEMENT_PROMPT = """\
Your previous Python strategy body compiled, but the compiled strategy failed at runtime.

Previous attempt:
```python
{previous_strategy}
```

Runtime traceback:
```
{error_traceback}
```

Produce a corrected Python body only.
Keep the rationale block at the top.
Stay within the supported Python subset.
Prefer minimal changes that preserve the strategy idea.
The fixed body must contain real decoding steps, not only comments or invariants.
The fixed body must still produce a nonempty, grammar-valid constrained `answer` channel.
Keep delimiter control explicit: use `helpers.ExpressiveStep(...)` for free-form output rather than `helpers.UnconstrainedStep(...)`.
"""


COMPILATION_ERROR_REFINEMENT_PROMPT = """\
Your previous Python strategy body verified but failed during Dafny-to-Python compilation.

Previous attempt:
```python
{previous_strategy}
```

Compilation error:
```
{error_message}
```

Produce a corrected Python body only.
Keep the rationale block at the top.
Avoid unsupported constructs; use simple assignments, `if`, and `while`.
The repaired body must still produce a nonempty, grammar-valid constrained `answer` channel.
Preserve the explicit final constrained answer channel.
"""


FORMAT_REPAIR_PROMPT = """Your output must be a Python method body. It is missing the required rationale block markers.

Rewrite the following body so that it starts with:
# CSD_RATIONALE_BEGIN
# ...
# CSD_RATIONALE_END

Then keep the Python body statements.

Previous output:
```python
{previous_strategy}
```

Output ONLY the corrected Python body, no markdown fences.
"""


EVALUATION_FAILURE_REFINEMENT_PROMPT = """\
Your previous Python strategy body verified, compiled, and ran, but it performed poorly on evaluation.

Previous attempt:
```python
{previous_strategy}
```

Evaluation feedback:
```
{evaluation_feedback}
```

Produce a revised Python body only.
Keep the rationale block at the top.
Try a meaningfully different constrained-decoding strategy if the current one is not working.
Do not fall back to unconstrained-only decoding or a basic window-switch pattern.
"""


STRUCTURE_REPAIR_PROMPT = """\
Your previous Python strategy body is structurally invalid for this project.

Previous attempt:
```python
{previous_strategy}
```

Issue:
{issue}

Rewrite it as a valid Python method body.

Rules:
- Output ONLY the body, not a full file.
- Keep the rationale block at the top.
- Include executable decoding logic, not just comments.
- Include at least one call to `helpers.ConstrainedAnswerStep(...)`.
- Use `helpers.ExpressiveStep(...)` for free-form output; do not use `helpers.UnconstrainedStep(...)`.
- Update `generated` with `generated = generated + [next_token]` when emitting free-form text.
- Update `answer` with `answer = answer + [next_token]` when emitting constrained answer content.
- Update the budget with `stepsLeft = new_steps`.
- Preserve the final constrained-answer guarantee.
- Do not return to an unconstrained-only or rollback-only design.
- Keep at least two extra local state variables that materially affect control flow.
"""


def build_initial_prompt(task_description: str) -> tuple[str, str]:
    user_prompt = INITIAL_GENERATION_PROMPT.format(task_description=task_description)
    return SYSTEM_PROMPT, user_prompt


def build_verification_error_prompt(previous_strategy: str, error_message: str) -> tuple[str, str]:
    user_prompt = VERIFICATION_ERROR_REFINEMENT_PROMPT.replace(
        "{previous_strategy}", previous_strategy
    ).replace("{error_message}", error_message)
    return SYSTEM_PROMPT, user_prompt


def build_runtime_error_prompt(previous_strategy: str, error_traceback: str) -> tuple[str, str]:
    user_prompt = RUNTIME_ERROR_REFINEMENT_PROMPT.format(
        previous_strategy=previous_strategy, error_traceback=error_traceback
    )
    return SYSTEM_PROMPT, user_prompt


def build_compilation_error_prompt(previous_strategy: str, error_message: str) -> tuple[str, str]:
    user_prompt = COMPILATION_ERROR_REFINEMENT_PROMPT.format(
        previous_strategy=previous_strategy, error_message=error_message
    )
    return SYSTEM_PROMPT, user_prompt


def build_format_repair_prompt(previous_strategy: str) -> tuple[str, str]:
    user_prompt = FORMAT_REPAIR_PROMPT.format(previous_strategy=previous_strategy)
    return SYSTEM_PROMPT, user_prompt


def build_evaluation_failure_prompt(previous_strategy: str, evaluation_feedback: str) -> tuple[str, str]:
    user_prompt = EVALUATION_FAILURE_REFINEMENT_PROMPT.format(
        previous_strategy=previous_strategy, evaluation_feedback=evaluation_feedback
    )
    return SYSTEM_PROMPT, user_prompt


def build_structure_repair_prompt(previous_strategy: str, issue: str) -> tuple[str, str]:
    user_prompt = STRUCTURE_REPAIR_PROMPT.format(
        previous_strategy=previous_strategy,
        issue=issue,
    )
    return SYSTEM_PROMPT, user_prompt
