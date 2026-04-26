"""
Prompt templates for Python-first CSD strategy generation.

The model writes a Python method body for `generation/csd/GeneratedAgentTemplate.py`.
That Python is later transpiled to Dafny for verification.
"""

import os
from functools import lru_cache
from pathlib import Path


PROMPTS_DIR = Path(__file__).resolve().parent
HELPER_REFERENCE_PATH = PROMPTS_DIR / "csd" / "VerifiedAgentSynthesis.md"
CURATED_HELPER_REFERENCE = """\
# Curated Helper Mini-Reference

This mini-reference is distilled from `generation/csd/VerifiedAgentSynthesis.md`.
Use it as the high-signal helper guide when writing the strategy body.

## Core Facts

- `generated` is `list[str]`, not a Python string.
- `helpers.LongestValidSuffix(generated)` is also `list[str]`, not a Python string.
- Never call `generated.startswith(...)`, `generated.endswith(...)`, `generated.strip(...)`,
  or remove delimiters with string slicing. Track phases with booleans/counters instead.
- Never call `.startswith(...)`, `.endswith(...)`, or `.strip(...)` on
  `helpers.LongestValidSuffix(generated)` either.
- `prompt` stays unchanged. Every emitted token goes into `generated`.
- `LeftDelimiter` and `RightDelimiter` are full LM tokens.

## Preferred Step Surface

Prefer these helpers and always assign the tuple result back into `generated, stepsLeft`:

- `helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)`
- `helpers.AppendLeftDelimiter(generated, stepsLeft)`
- `helpers.AppendConstrainedStep(prompt, generated, stepsLeft)`
- `helpers.AppendSoftConstrainedStep(prompt, generated, penalty, stepsLeft)`
- `helpers.AppendTopKConstrainedStep(prompt, generated, k, stepsLeft)`
- `helpers.AppendRightDelimiter(generated, stepsLeft)`

Never use an `Append*` helper as a bare statement.
Never write `next_token, stepsLeft = helpers.AppendConstrainedStep(...)` or any other
`Append*` call into `next_token`; `Append*` returns an updated prefix, not a token.
If you use raw `ForcedTokenStep`, always write:
`next_token, new_steps = helpers.ForcedTokenStep(...)`,
then append `next_token`, then set `stepsLeft = new_steps`.

## Ownership Rules

Grammar-state queries live on `parser`, not on `helpers`.

- Correct: `helpers.CanConstrain(generated)`
- Correct: `parser.IsCompletePrefix(helpers.LongestValidSuffix(generated))`
- Correct: `parser.ValidContinuationCount(helpers.LongestValidSuffix(generated))`
- Wrong: `helpers.IsCompletePrefix(...)`
- Wrong: `helpers.ValidContinuationCount(...)`
- Wrong: `parser.IsCompletePrefix(generated)`

## Constrained-Call Rule

Every call to:
- `ConstrainedStep`
- `SoftConstrainedStep`
- `TopKConstrainedStep`
- `AppendConstrainedStep`
- `AppendSoftConstrainedStep`
- `AppendTopKConstrainedStep`

must be inside a branch or loop condition that explicitly mentions
`helpers.CanConstrain(generated)` or
`parser.IsCompletePrefix(helpers.LongestValidSuffix(generated))`.

Do not rely on a phase variable alone.

## Minimal Proof-Friendly Loop Pattern

Place these exact comments immediately above each decoding `while` loop:

```python
# invariant lm.ValidTokensIdsLogits()
# invariant 0 <= stepsLeft <= maxSteps
# invariant |generated| + stepsLeft <= maxSteps
# decreases stepsLeft
```

Every decoding loop must be budget-bounded, for example:
`while stepsLeft > 0 and phase < 3:`

## Delimiter Protocol

The graded answer is the final `<< ... >>` span.

Use this order:
1. Optional free-form reasoning with `helpers.AppendUnconstrainedStep(...)`
2. `generated, stepsLeft = helpers.AppendLeftDelimiter(generated, stepsLeft)`
3. While `helpers.CanConstrain(generated)`, append constrained grammar tokens
4. Only after the grammar suffix is complete, append `RightDelimiter`

Do not mention `<<` or `>>` in reasoning text outside the final answer span.
For GSM arithmetic, preserve decimals exactly; do not turn `8.5` into `8`.
"""


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip() in {"1", "true", "True", "yes", "on"}


def helper_reference_mode() -> str:
    explicit = os.environ.get("CSD_HELPER_REFERENCE_MODE", "").strip().lower()
    if explicit in {"curated", "mini", "compact"}:
        return "curated"
    if explicit in {"full", "markdown", "md"}:
        return "full"
    if explicit in {"none", "off", "0", "false"}:
        return "none"
    if _env_flag("CSD_INCLUDE_HELPER_REFERENCE_MD"):
        return "full"
    return "none"


@lru_cache(maxsize=1)
def _load_helper_reference_markdown() -> str:
    return HELPER_REFERENCE_PATH.read_text(encoding="utf-8").strip()


def _compose_system_prompt() -> str:
    mode = helper_reference_mode()
    if mode == "none":
        return SYSTEM_PROMPT
    if mode == "curated":
        return (
            SYSTEM_PROMPT
            + "\n\n## Additional Curated Helper Reference\n\n"
            + "The following mini-reference is distilled from `generation/csd/VerifiedAgentSynthesis.md`.\n"
            + "Use it as the authoritative short reference for helper names, object ownership, and proof-critical usage rules.\n\n"
            + "[BEGIN CURATED_HELPER_REFERENCE]\n"
            + CURATED_HELPER_REFERENCE.strip()
            + "\n[END CURATED_HELPER_REFERENCE]\n"
        )
    reference = _load_helper_reference_markdown()
    return (
        SYSTEM_PROMPT
        + "\n\n## Additional Authoritative Helper Reference\n\n"
        + "The following markdown is copied directly from `generation/csd/VerifiedAgentSynthesis.md`.\n"
        + "Use it as the authoritative reference for helper names, signatures, object ownership, and contracts.\n"
        + "If this reference conflicts with your memory, follow the reference.\n\n"
        + "[BEGIN VERIFIED_AGENT_SYNTHESIS_MD]\n"
        + reference
        + "\n[END VERIFIED_AGENT_SYNTHESIS_MD]\n"
    )

SYSTEM_PROMPT = """\
You are an expert in formal verification and constrained decoding for language models.
You are generating the BODY of a Python function, not a full file.

The surrounding template is:

  def MyCSDStrategy(lm: LM, parser: Parser, prompt: Prefix, maxSteps: int, eosToken: Token) -> tuple[Prefix, int]:
      helpers = CSDHelpers(lm, parser)
      lm.ValidTokensIdsLogitsAlways()
      generated = []   # all output tokens go here — free-form and constrained alike
      stepsLeft = maxSteps
      [YOUR BODY]
      remainingSteps = stepsLeft
      return generated, remainingSteps

Your output must therefore be ONLY Python statements for [YOUR BODY].
Do not write the function signature, imports, markdown fences, or a full file.
Do not redeclare `helpers`, `generated`, or `stepsLeft`.
Do not assign to `remainingSteps`; the template already does that.
Do not output only comments or only invariants. The body must execute real decoding steps.

## Library API

All tokens, prefixes, and logits are plain Python types (str, list[str], float).
`generated` is a list of token strings, not a Python string. Never call string methods like
`generated.startswith(...)` or `generated.endswith(...)`, and never strip delimiter characters
from it as if `<<` or `>>` were substrings. Delimiters are full tokens; emit them with
`helpers.AppendLeftDelimiter(...)` / `helpers.AppendRightDelimiter(...)` and track phase with
state variables instead of string slicing.

### Raw step functions — consume one step, return (next_token, new_stepsLeft)

- `helpers.UnconstrainedStep(prompt, generated, stepsLeft)` — generates next token with no grammar constraint.
- `helpers.ConstrainedStep(prompt, generated, stepsLeft)` — computes `LongestValidSuffix(generated)` to find the current grammar state, masks all invalid tokens, then generates. Use this whenever the model is inside a grammar-constrained segment.
- `helpers.SoftConstrainedStep(prompt, generated, penalty, stepsLeft)` — like ConstrainedStep but penalizes invalid tokens by `penalty` instead of hard-masking them.
- `helpers.TopKConstrainedStep(prompt, generated, k, stepsLeft)` — applies top-k filtering first, then grammar masking.
- `helpers.ForcedTokenStep(prompt, generated, token, stepsLeft)` — skips LM generation entirely; emits `token` directly. Use to emit structural tokens like `LeftDelimiter` and `RightDelimiter`. Always capture both return values as `next_token, new_steps = helpers.ForcedTokenStep(...)`, then append `next_token` and update `stepsLeft = new_steps`.
- `helpers.BudgetAwareStep(prompt, generated, stepsLeft, threshold)` — uses ConstrainedStep when `stepsLeft <= threshold` and the grammar is incomplete; otherwise UnconstrainedStep.

### Preferred append-style helpers — consume one step and return (updated_prefix, remaining_steps)

- `helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)` — preferred wrapper around `UnconstrainedStep`; appends the chosen token for you.
- `helpers.AppendConstrainedStep(prompt, generated, stepsLeft)` — preferred wrapper around `ConstrainedStep`; appends the grammar-valid token for you.
- `helpers.AppendSoftConstrainedStep(prompt, generated, penalty, stepsLeft)` — preferred wrapper around `SoftConstrainedStep`.
- `helpers.AppendTopKConstrainedStep(prompt, generated, k, stepsLeft)` — preferred wrapper around `TopKConstrainedStep`.
- `helpers.AppendBudgetAwareStep(prompt, generated, stepsLeft, threshold)` — preferred wrapper around `BudgetAwareStep`.
- `helpers.AppendForcedToken(generated, token, stepsLeft)` — preferred wrapper around `ForcedTokenStep`; appends `token` for you.
- `helpers.AppendLeftDelimiter(generated, stepsLeft)` — append `LeftDelimiter` in one call.
- `helpers.AppendRightDelimiter(generated, stepsLeft)` — append `RightDelimiter` in one call.

When possible, prefer the Append* helpers because they avoid the common proof mistakes around tuple unpacking, forgotten appends, and stale step budgets.

### Which object owns which method

Only a small set of names live on `helpers`. Grammar-state queries like completeness and continuation counts live on `parser`, not on `helpers`.

- Correct: `helpers.CanConstrain(generated)`
- Correct: `parser.IsCompletePrefix(helpers.LongestValidSuffix(generated))`
- Correct: `parser.ValidContinuationCount(helpers.LongestValidSuffix(generated))`
- Wrong: `helpers.IsCompletePrefix(...)`
- Wrong: `helpers.ValidContinuationCount(...)`

If you need parser information about the generated answer segment, first compute the grammar-relevant suffix with `helpers.LongestValidSuffix(generated)` and pass that suffix to `parser`.

### Grammar state queries

- `helpers.LongestValidSuffix(generated)` — returns the longest suffix of `generated` that the parser accepts as a valid prefix. Returns `[]` if no suffix is valid. Use this to check where the constrained segment begins.
- `helpers.CanConstrain(generated)` — shorthand for `not parser.IsCompletePrefix(helpers.LongestValidSuffix(generated))`. Prefer this guard before constrained helper calls.
- `parser.IsValidPrefix(prefix)` — True if `prefix` is a valid partial parse.
- `parser.IsCompletePrefix(prefix)` — True if `prefix` is a complete, finished parse.
- `parser.ValidNextTokens(prefix)` — returns list of valid next tokens from `prefix`.
- `parser.IsDeadPrefix(prefix)` — True if the prefix cannot be extended.
- `parser.ValidContinuationCount(prefix)` — number of valid continuations.
- `parser.ParserDistanceToComplete(prefix)` — lower bound on steps to complete the grammar.

### Logit shaping (on `lm`, not `helpers`)

- `lm.BiasToken(token, delta)` — add `delta` to one token's logit (clamped to [-1e9, 1e9]).
- `lm.BiasTokens(tokens, delta)` — add `delta` to a list of tokens.
- `lm.ScaleToken(token, factor)` — multiply one token's logit by `factor`.
- `lm.MaskToken(token)` — set one token's logit to -1e9.
- `lm.MaskTokensExcept(tokens)` — mask all tokens except the allowlist.
- `lm.TopKFilter(k)` — keep only the top-k logit tokens; mask the rest.
- `lm.ClampLogits(low, high)` — clamp all logits to [low, high].

### Composite helpers

- `helpers.SoftConstrainToGrammar(prefix, penalty)` — bias invalid tokens by -penalty (no LM call).
- `helpers.IntersectWithGrammar(prefix)` — hard-mask invalid tokens (no LM call).
- `helpers.BiasForCompletion(prefix, bonus)` — bias tokens that would complete the grammar by +bonus.
- `helpers.HasBudget(stepsLeft, needed)` — returns stepsLeft >= needed.
- `helpers.MinStepsToComplete(prefix)` — lower bound on steps to finish from current suffix.

### Repair utilities

- `helpers.RollbackToValidPrefix(generated)` — trim `generated` until it ends at a valid, non-dead grammar state.
- `helpers.FindLongestValidSpan(generated)` — find the longest grammar-valid contiguous span anywhere in `generated`.
- `helpers.RepairByRetry(prompt, generated, maxRetries, stepsLeft)` — rollback then take up to `maxRetries` constrained steps. Returns `(repaired_prefix, remaining_steps)`.

### State structures (opt-in)

- `CheckpointStack()` — push/pop/peek saved prefixes for backtracking.
- `RepetitionTracker(ngramSize)` — track n-gram frequencies; call `ApplyRepetitionPenalties(lm)` to bias logits.

### Constants

- `LeftDelimiter = "<<"` and `RightDelimiter = ">>"` are tokens in the LM vocabulary.

## Answer extraction

The evaluator extracts the answer from output matching `<< ... >>`. Your strategy MUST emit
`LeftDelimiter` (via `ForcedTokenStep`) before the constrained answer segment and `RightDelimiter`
after it. The grammar-constrained content between them is the answer. For GSM-style arithmetic
tasks, that final segment may be a compact expression like `<<16 * 8.5 + 4 * 10.5 + 13>>`;
the evaluator can compute its numeric value and does not require a standalone numeral.

## Workflow pattern

A typical strategy body:
1. Generate free-form reasoning with `helpers.AppendUnconstrainedStep(...)` into `generated`.
2. Emit `LeftDelimiter` with `helpers.AppendLeftDelimiter(generated, stepsLeft)`.
3. Loop `helpers.AppendConstrainedStep(prompt, generated, stepsLeft)` until `not helpers.CanConstrain(generated)`.
4. Emit `RightDelimiter` with `helpers.AppendRightDelimiter(generated, stepsLeft)`.

After step 2, `LongestValidSuffix` resets to `[]` (since `<<` is not a grammar token), so the
first `ConstrainedStep` will choose from `ValidNextTokens([])` — the grammar's starting tokens.
Do not jump directly from free-form reasoning to `AppendConstrainedStep` without first emitting
`LeftDelimiter`.

## Python subset rules

- Use normal Python syntax: `while ...:`, `if ...:`, `and`, `or`, `not`, `==`, `!=`.
- Use Python comments beginning with `#`.
- If you need loop invariants for the Dafny transpiler, put them as comments IMMEDIATELY above the `while`:
    # invariant lm.ValidTokensIdsLogits()
    # invariant 0 <= stepsLeft <= maxSteps
    # invariant |generated| + stepsLeft <= maxSteps
    # decreases stepsLeft
  These use Dafny syntax (`|generated|`, `==>`) even though the executable body uses Python.
- Prefer `len(generated)` in Python; the transpiler lowers it to Dafny length syntax.
- Do not use Python `for` loops. Use `while` loops only.
- Do not use list comprehensions, lambdas, helper functions, or nested function definitions.
- Do not use `break` unless truly necessary.
- If you branch between different step choices, predeclare branch outputs before the `if`:
    next_token = eosToken
    new_steps = stepsLeft
    if ...:
        next_token, new_steps = helpers.ConstrainedStep(prompt, generated, stepsLeft)
    else:
        next_token, new_steps = helpers.UnconstrainedStep(prompt, generated, stepsLeft)
- Every emitted token must come from a step call and must consume budget.
- Never call `helpers.ForcedTokenStep(...)` as a bare statement. Always write:
    next_token, new_steps = helpers.ForcedTokenStep(prompt, generated, LeftDelimiter, stepsLeft)
    generated = generated + [next_token]
    stepsLeft = new_steps
- Never call an `Append*` helper as a bare statement. Always assign its `(updated_prefix, remaining_steps)` result back into `generated` and `stepsLeft`.
- Do NOT call `parser.IsValidPrefix(generated)` or `parser.IsCompletePrefix(generated)` directly.
  Always route grammar queries through `helpers.LongestValidSuffix(generated)` first.

## Required rationale block

Your output MUST begin with:

# CSD_RATIONALE_BEGIN
# <short explanation of the strategy>
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
    # invariant 0 <= stepsLeft <= maxSteps
    # invariant |generated| + stepsLeft <= maxSteps
    # decreases stepsLeft
- Every decoding `while` loop must be budget-bounded, e.g. `while stepsLeft > 0 and ...:`.
  Do not write open-ended sentinel loops like `while not done:` without a `stepsLeft > 0` guard.
- Prefer the Append* wrappers unless you genuinely need the raw token return.
- Before any `ConstrainedStep`, `SoftConstrainedStep`, `TopKConstrainedStep`, `BudgetAwareStep`,
  `AppendConstrainedStep`, `AppendSoftConstrainedStep`, or `AppendTopKConstrainedStep` call,
  ensure the current grammar suffix is incomplete, preferably with `helpers.CanConstrain(generated)`.
- Put constrained helper calls inside a branch whose condition explicitly mentions
  `helpers.CanConstrain(generated)`. Do not rely on some earlier phase variable alone.
- Do not write `helpers.IsCompletePrefix(...)` or `helpers.ValidContinuationCount(...)`.
  Those are parser calls, not helper calls.
- The strategy MUST emit `LeftDelimiter` (prefer `helpers.AppendLeftDelimiter(...)`), followed by
  grammar-constrained tokens (prefer `helpers.AppendConstrainedStep(...)`), followed by
  `RightDelimiter` (prefer `helpers.AppendRightDelimiter(...)`).
- You must ensure `parser.IsCompletePrefix(helpers.LongestValidSuffix(generated))` before emitting
  `RightDelimiter`. The grammar-constrained content between delimiters is the graded answer.
- If you use raw `ForcedTokenStep`, always capture it as `next_token, new_steps = ...`, append
  `next_token`, and then update `stepsLeft = new_steps`. Do not append `LeftDelimiter` or
  `RightDelimiter` literals directly.
- The strategy may generate free-form reasoning before entering the constrained segment; use
  `helpers.AppendUnconstrainedStep(...)` or `helpers.UnconstrainedStep(...)` for that.
- If the task is GSM-style arithmetic, the final constrained segment may be a short expression
  or equation; it does not need to simplify all the way to a standalone numeral.
- For GSM-style arithmetic, preserve the numeric values from the problem exactly. Do not round or
  truncate decimals like `8.5` into `8`.
- Novelty requirements:
  - Do NOT produce a trivial two-phase "all unconstrained then all constrained" loop with no
    adaptive control.
  - Use multiple state variables to drive adaptive decisions across phases.
  - At least two interacting signals must affect what step type is chosen each iteration.
  - Favor strategies that evolve their constraint strength over time (e.g., soft → hard, or
    grammar-distance-aware budgeting).
- Maintain at least two extra local state variables beyond `generated`, `stepsLeft`, `next_token`,
  and `new_steps`.
- Do NOT call parser methods on `generated` directly; always use
  `helpers.LongestValidSuffix(generated)` to route grammar queries.
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
- Keep the standard proof-carrying loop lines unless there is a strong reason to strengthen them:
    # invariant lm.ValidTokensIdsLogits()
    # invariant 0 <= stepsLeft <= maxSteps
    # invariant |generated| + stepsLeft <= maxSteps
    # decreases stepsLeft
- Every decoding `while` loop must mention the decreasing budget in its condition, e.g.
  `while stepsLeft > 0 and ...:`.
- Do not call constrained helpers once `parser.IsCompletePrefix(helpers.LongestValidSuffix(generated))`
  is true.
- Do not redeclare `helpers`, `generated`, or `stepsLeft`.
- Do not assign to `remainingSteps`.

Common fixes:
- If you used Dafny syntax like `:=`, replace it with Python `=`.
- If you used `&&`, `||`, or `!`, replace them with `and`, `or`, and `not`.
- If you used `//` comments, replace them with `#` comments.
- If the verifier complained that branch-local step outputs were undefined, predeclare `next_token`
  and `new_steps` before the `if`.
- Do NOT call parser methods on `generated` directly; use `helpers.LongestValidSuffix(generated)`.
- The repaired body must still emit `LeftDelimiter`, constrained tokens, and `RightDelimiter`.
"""


RUNTIME_ERROR_REFINEMENT_PROMPT = """\
Your previous Python strategy body failed at runtime.

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
The fixed body must emit `LeftDelimiter`, grammar-constrained tokens, and `RightDelimiter`.
"""


EVALUATION_FAILURE_REFINEMENT_PROMPT = """\
Your previous Python strategy body verified and ran, but it performed poorly on evaluation.

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
If the task is GSM-style arithmetic, the final `<< >>` segment may stay as a short expression or
equation instead of simplifying to a standalone numeral.
The strategy must still emit `LeftDelimiter`, grammar-constrained tokens, and `RightDelimiter`.
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
- Every decoding `while` loop must be budget-bounded, e.g. `while stepsLeft > 0 and ...:`.
- Prefer `helpers.CanConstrain(generated)` over spelling out the full suffix-complete check.
- Include the standard proof-carrying loop lines:
  `# invariant lm.ValidTokensIdsLogits()`,
  `# invariant 0 <= stepsLeft <= maxSteps`,
  and `# decreases stepsLeft`.
- Prefer the Append* helpers:
  `helpers.AppendLeftDelimiter(generated, stepsLeft)`,
  `helpers.AppendConstrainedStep(prompt, generated, stepsLeft)`,
  `helpers.AppendRightDelimiter(generated, stepsLeft)`.
- Keep parser queries on `parser`, not on `helpers`. For example, write
  `parser.ValidContinuationCount(helpers.LongestValidSuffix(generated))`, not
  `helpers.ValidContinuationCount(...)`.
- Only call constrained helpers while `helpers.CanConstrain(generated)`.
- Put each constrained-helper call in a branch whose condition explicitly mentions
  `helpers.CanConstrain(generated)`.
- If you use raw forced-token calls, always capture them with `next_token, new_steps = ...`.
- Assign every Append* result back into `generated` and `stepsLeft`.
- If you use `helpers.RepairByRetry(prompt, generated, maxRetries, stepsLeft)`, only call it when
  `stepsLeft >= maxRetries`.
- If you use raw step calls, append every emitted token with `generated = generated + [next_token]`
  rather than appending delimiter literals directly, and update the budget with `stepsLeft = new_steps`.
- Do NOT call parser methods on `generated` directly.
- Maintain at least two extra local state variables that materially affect control flow.
"""


def build_initial_prompt(task_description: str) -> tuple[str, str]:
    user_prompt = INITIAL_GENERATION_PROMPT.format(task_description=task_description)
    return _compose_system_prompt(), user_prompt


def build_verification_error_prompt(previous_strategy: str, error_message: str) -> tuple[str, str]:
    user_prompt = VERIFICATION_ERROR_REFINEMENT_PROMPT.replace(
        "{previous_strategy}", previous_strategy
    ).replace("{error_message}", error_message)
    return _compose_system_prompt(), user_prompt


def build_runtime_error_prompt(previous_strategy: str, error_traceback: str) -> tuple[str, str]:
    user_prompt = RUNTIME_ERROR_REFINEMENT_PROMPT.format(
        previous_strategy=previous_strategy, error_traceback=error_traceback
    )
    return _compose_system_prompt(), user_prompt


def build_compilation_error_prompt(previous_strategy: str, error_message: str) -> tuple[str, str]:
    # Kept for backward compat; compilation no longer happens but this builder is still called
    user_prompt = EVALUATION_FAILURE_REFINEMENT_PROMPT.format(
        previous_strategy=previous_strategy,
        evaluation_feedback=f"(runtime error) {error_message}",
    )
    return _compose_system_prompt(), user_prompt


def build_format_repair_prompt(previous_strategy: str) -> tuple[str, str]:
    user_prompt = FORMAT_REPAIR_PROMPT.format(previous_strategy=previous_strategy)
    return _compose_system_prompt(), user_prompt


def build_evaluation_failure_prompt(previous_strategy: str, evaluation_feedback: str) -> tuple[str, str]:
    user_prompt = EVALUATION_FAILURE_REFINEMENT_PROMPT.format(
        previous_strategy=previous_strategy, evaluation_feedback=evaluation_feedback
    )
    return _compose_system_prompt(), user_prompt


def build_structure_repair_prompt(previous_strategy: str, issue: str) -> tuple[str, str]:
    user_prompt = STRUCTURE_REPAIR_PROMPT.format(
        previous_strategy=previous_strategy,
        issue=issue,
    )
    return _compose_system_prompt(), user_prompt
