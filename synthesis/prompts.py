"""
Prompt templates for Qwen-based CSD strategy generation.

This project synthesizes *constrained decoding strategies* (CSDs), not the final
task output itself. The LLM should choose an appropriate verified strategy
primitive and parameters based on the *use-case* described by the task.

The generator expects these entrypoints:
- build_initial_prompt(task_description)
- build_verification_error_prompt(previous_strategy, error_message)
- build_runtime_error_prompt(previous_strategy, error_traceback)
- build_compilation_error_prompt(previous_strategy, error_message)
- build_format_repair_prompt(previous_strategy)
"""

# NOTE:
# The synthesized output is injected into `dafny/GeneratedCSD.dfy` as the BODY
# of method `MyCSDStrategy(...) returns (generated: Prefix)`.
#
# The output may be a multi-line Dafny method body. It must assign the out-parameter
# `generated` and satisfy the method's contract. A common minimal shape is:
#   generated := helpers.<Strategy>(lm, parser, prompt, maxSteps, ...);
#   cost := helpers.cost;


SYSTEM_PROMPT = """\
You are an expert in formal verification and constrained decoding for language models.
You are generating a *constrained decoding strategy implementation* for a specific use-case.

You must output ONLY the Dafny method body for:

  method MyCSDStrategy(lm: LM, parser: Parser, prompt: Prefix, maxSteps: nat) returns (generated: Prefix, cost: int)
    modifies lm.Logits
    requires lm.ValidTokensIdsLogits()
    requires parser.IsValidPrefix([])
    requires forall t: Token :: t in parser.ValidNextTokens([]) ==> t in lm.Tokens
    ensures lm.ValidTokensIdsLogits()
    ensures |generated| <= maxSteps
    ensures parser.IsValidPrefix(generated)
    ensures |generated| == maxSteps || parser.IsCompletePrefix(generated)
    // PLUS a user-defined cost contract (ensures clause)
    // PLUS a default cost bound like `ensures cost <= 2 * maxSteps`

Important constraints:
- Output MUST be valid Dafny statements (end statements with ';').
- Initialize/assign the out-parameter `generated`.
- Initialize/assign the out-parameter `cost` (e.g., `cost := helpers.cost;`).
- Do NOT redeclare `generated` or `cost` as local variables. They are already method out-parameters.
- You MUST use the `helpers` instance (type `CSDHelpers`) which is already instantiated for you.
- Do NOT use `CSDHelpers.<Method>` (static call); use `helpers.<Method>` (instance call).
- `CSDHelpers` now tracks `cost: int`. Every helper call increments `helpers.cost`.
- Your generated code will be checked against the cost contract.
- Be constraint-driven: choose the strategy/parameters based on the parser strictness
  and the method contract, not based on any canned examples.

CRITICAL TYPE CONSTRAINTS:
- `prompt` is type `Prefix` (which is `seq<Token>`), NOT a string.
- Do NOT try to manually construct output by appending text. The CSDHelpers methods handle
  token generation.
- The `parser` enforces structural validity automatically during generation.

CRITICAL DAFNY SYNTAX:
- In Dafny, sequence concatenation uses `+` (e.g., `a + b`). Do NOT use `++`.

## REQUIRED: rationale block (must be included in EVERY output)
Your output MUST begin with a short, parseable rationale comment block:

// CSD_RATIONALE_BEGIN
// <explanation of strategy and parameters>
// CSD_RATIONALE_END

After this, output the Dafny statements for the strategy body.

## Available VERIFIED strategy helpers (use `helpers.<Method>`)

1) Pure constrained:
  generated := helpers.PureConstrainedGeneration(lm, parser, prompt, maxSteps);

2) Optimistic then fallback:
  generated := helpers.TryUnconstrainedThenConstrained(lm, parser, prompt, maxSteps, N);
N in [1..20], N <= maxSteps.

3) Interleaved hybrid (uses << >> for hybrid segments):
  generated := helpers.HybridGeneration(lm, parser, prompt, maxSteps);
  This strategy allows the model to produce reasoning/thought segments within << and >>.
  If the content between delimiters is invalid according to the parser, it is rolled back.

4) Speculative:
  generated := helpers.SpeculativeGeneration(lm, parser, prompt, maxSteps, N);
N in [2..8].

5) Max creativity with repair/complete:
  generated := helpers.UnconstrainedWithCompletion(lm, parser, prompt, maxSteps);

6) Completion of an existing valid prefix:
  generated := helpers.CompletePrefix(lm, parser, prompt, partial, maxSteps);

7) Generate with reasonable length:
  generated := helpers.GenerateWithReasonableLength(lm, parser, prompt, maxSteps, reasonableLength);

8) Generate until first complete:
  generated := helpers.GenerateUntilFirstComplete(lm, parser, prompt, maxSteps);

9) Generate multiple candidates and select best:
  generated := helpers.GenerateAndSelectBest(lm, parser, prompt, maxSteps, numCandidates, preferShorter);

## Utility Functions for Contracts (use in `ensures` clauses)

1) Extract content between delimiters:
  `helpers.ExtractContentBetweenDelimiters(helpers.PrefixToString(generated), "<<", ">>")`
  Returns the content found between the last occurrence of "<<" and ">>". Useful for asserting things about the reasoning trace.

2) Convert prefix to string:
  `helpers.PrefixToString(generated)`
  Converts a sequence of tokens into a single concatenated string. Useful for string-based contracts.

## Decision rubric (internal reasoning only)
Use the use-case description to decide:
- How strict the output format is (parser restrictive vs permissive)
- Whether you need early flexibility (drafting/planning) vs always-valid prefixes
- Whether you care about latency (speculation) vs simplicity/robustness
Then pick helper(s) and parameters accordingly, staying within the stated ranges.

Output format:
- Return ONLY the method body (no signature, no outer braces).
- Multi-line bodies are allowed and encouraged when useful (e.g., locals, branching, loops).
- Ensure `generated` is assigned along all paths.
"""


INITIAL_GENERATION_PROMPT = """\
Generate a CSD strategy implementation for this use-case:

Use-case description: {task_description}

Output MUST start with the required rationale block (see system prompt), then output ONLY the Dafny code.

Output ONLY the method body (no signature, no outer braces). Do NOT wrap output in markdown code fences (no ```dafny).
Multi-line bodies are allowed and encouraged when useful (locals, branching, loops), as long as `generated` is assigned on all paths.

Unless the use-case strongly demands the simplest baseline, prefer a **multi-statement** body (at least 2 statements),
using locals and/or if/else around verified helper calls. Avoid writing custom loops unless necessary.

To count as non-trivial multi-line, include an `if/else` with **meaningfully different** branches
(different helper and/or different parameters). Do NOT output only a temp wrapper around a single helper call.

Examples below are illustrative only - do NOT default to any one helper. Choose based on the use-case.

Example (fully constrained):
  // CSD_RATIONALE_BEGIN
  // I chose PureConstrainedGeneration because every prefix must remain valid under a strict parser.
  // CSD_RATIONALE_END
  generated := helpers.PureConstrainedGeneration(lm, parser, prompt, maxSteps);
  cost := helpers.cost;

Example (optimistic then fallback - MUST guard for precondition):
  // CSD_RATIONALE_BEGIN
  // I chose TryUnconstrainedThenConstrained to allow a short burst of free drafting, then guarantee validity.
  // N=5 is small to reduce waste if the parser is strict. I guard with maxSteps check to satisfy N<=maxSteps.
  // CSD_RATIONALE_END
  if maxSteps < 5 {{
    generated := helpers.PureConstrainedGeneration(lm, parser, prompt, maxSteps);
  }} else {{
    generated := helpers.TryUnconstrainedThenConstrained(lm, parser, prompt, maxSteps, 5);
  }}
  cost := helpers.cost;

Example (interleaved hybrid):
  // CSD_RATIONALE_BEGIN
  // I chose HybridGeneration to mostly stay constrained while occasionally attempting an unconstrained token when likely safe.
  // CSD_RATIONALE_END
  generated := helpers.HybridGeneration(lm, parser, prompt, maxSteps);
  cost := helpers.cost;

Example (speculative speed-oriented):
  // CSD_RATIONALE_BEGIN
  // I chose SpeculativeGeneration to reduce per-token overhead by validating batches, prioritizing latency.
  // Window N=4 balances speedups vs rejection waste.
  // CSD_RATIONALE_END
  generated := helpers.SpeculativeGeneration(lm, parser, prompt, maxSteps, 4);
  cost := helpers.cost;

Example (unconstrained with completion/repair):
  // CSD_RATIONALE_BEGIN
  // I chose UnconstrainedWithCompletion to allow maximum creativity early, then roll back to a valid prefix and finish constrained.
  // CSD_RATIONALE_END
  generated := helpers.UnconstrainedWithCompletion(lm, parser, prompt, maxSteps);
  cost := helpers.cost;

Example (multi-line composition/branching):
  // CSD_RATIONALE_BEGIN
  // I chose a simple conditional: for very small maxSteps, fully constrained is cheapest; otherwise use a faster batch-based strategy.
  // CSD_RATIONALE_END
  if maxSteps < 6 {{
    generated := helpers.PureConstrainedGeneration(lm, parser, prompt, maxSteps);
  }} else {{
    generated := helpers.SpeculativeGeneration(lm, parser, prompt, maxSteps, 4);
  }}
  cost := helpers.cost;

Example (preferred non-trivial multi-line: meaningful branching):
  // CSD_RATIONALE_BEGIN
  // I branch on maxSteps: for small budgets, fully constrained is simplest; for larger budgets, speculative can reduce overhead.
  // CSD_RATIONALE_END
  if maxSteps < 8 {{
    generated := helpers.PureConstrainedGeneration(lm, parser, prompt, maxSteps);
  }} else {{
    generated := helpers.SpeculativeGeneration(lm, parser, prompt, maxSteps, 4);
  }}
  cost := helpers.cost;
"""


VERIFICATION_ERROR_REFINEMENT_PROMPT = """\
Your previous method body failed Dafny verification.

Previous attempt:
```dafny
{previous_strategy}
```

Verification error:
```
{error_message}
```

Fix the issue while keeping the strategy non-trivial when appropriate.

Rules:
- Output ONLY a corrected method body (no signature, no braces). Do NOT wrap output in markdown code fences.
- The corrected body MUST include the required rationale block at the top (see system prompt). Update it if you change helpers/parameters.
- Preserve non-triviality when possible: keep a meaningful multi-statement structure (prefer `if/else` with different branches). Do not collapse to a single helper call unless required to make verification succeed.
- Ensure the body is valid Dafny. Statements must end with ';' where required.
- Ensure parameters are in-range:
  - TryUnconstrainedThenConstrained N: 1..20 and N <= maxSteps
  - HybridGeneration interval N: 2..10
  - SpeculativeGeneration window N: 2..8

Common fixes:
- Do NOT write `var generated: Prefix;` (duplicate/shadowing). Assign to the existing out-parameter `generated` instead.
- Avoid subtracting from `maxSteps` (e.g., `maxSteps - 5`) unless you guard with a proof-friendly condition and keep all variables initialized on all branches.
- **CRITICAL**: When using TryUnconstrainedThenConstrained with a constant N, you MUST guard it with `if maxSteps >= N` or use PureConstrainedGeneration when maxSteps is too small. Otherwise Dafny cannot prove the precondition `N <= maxSteps`.
  Example fix for "precondition could not be proved" error:
    WRONG: generated := CSDHelpers.TryUnconstrainedThenConstrained(lm, parser, prompt, maxSteps, 5);
    RIGHT: if maxSteps < 5 {{
             generated := CSDHelpers.PureConstrainedGeneration(lm, parser, prompt, maxSteps);
           }} else {{
             generated := CSDHelpers.TryUnconstrainedThenConstrained(lm, parser, prompt, maxSteps, 5);
           }}

CRITICAL: If the error mentions type mismatch with `string` or `seq<Token>`:
- `prompt` is type `Prefix` (seq<Token>), NOT a string. You CANNOT use `+` with strings.
- Do NOT try to manually construct output by appending strings like "\\n" or "[section]".
- Just call the appropriate CSDHelpers method - the parser handles structure automatically.

CRITICAL: If the error mentions `invalid UnaryExpression` and your code uses `++`:
- Replace `++` with `+` for sequence concatenation (e.g., `a + b`, not `a ++ b`).

If the same strategy keeps failing, switch to a different verified strategy helper (do not collapse to the most trivial unless the use-case demands strictness).
"""


RUNTIME_ERROR_REFINEMENT_PROMPT = """\
Your method body passed Dafny verification but failed at runtime.

Previous attempt:
```dafny
{previous_strategy}
```

Runtime error:
```
{error_traceback}
```

Adjust the strategy choice/parameters to avoid runtime pitfalls while staying aligned to the use-case.

Rules:
- Output ONLY a corrected method body (no signature, no braces).
- The corrected body MUST include the required rationale block at the top (see system prompt). Update it if you change helpers/parameters.
- Preserve non-triviality when possible: keep a meaningful multi-statement structure (prefer `if/else` with different branches). Do not collapse to a single helper call unless required to make runtime succeed.
- Keep parameters in-range (see verification prompt).
- If the runtime error suggests the environment is strict, move toward safer strategies:
  SpeculativeGeneration -> HybridGeneration -> TryUnconstrainedThenConstrained -> PureConstrainedGeneration
"""


COMPILATION_ERROR_REFINEMENT_PROMPT = """\
Your method body passed Dafny verification but failed during Dafny-to-Python compilation.

Previous attempt:
```dafny
{previous_strategy}
```

Compilation error:
```
{error_message}
```

Produce a new method body that is syntactically simple. Multi-line is allowed, but
avoid clever parsing tricks; prefer straightforward locals + verified helper calls.
Do NOT introduce new helper functions.

Output ONLY the corrected method body (no signature, no braces).
- The corrected body MUST include the required rationale block at the top (see system prompt). Update it if you change helpers/parameters.
- Preserve non-triviality when possible: keep a meaningful multi-statement structure (prefer `if/else` with different branches). Do not collapse to a single helper call unless required to make compilation succeed.
"""


FORMAT_REPAIR_PROMPT = """Your output must be a Dafny method body. It is missing the required rationale block markers.

Rewrite the following content into a valid Dafny method body that:
1. Starts with:
   // CSD_RATIONALE_BEGIN
   // <1-6 lines of explanation>
   // CSD_RATIONALE_END
2. Preserves the SAME strategy semantics (same helper choice(s) and key parameters) unless a change is required just to make it valid Dafny.
3. Outputs ONLY the method body (no signature, no outer braces).

Content to rewrite:
```dafny
{previous_strategy}
```
"""


def build_initial_prompt(task_description: str) -> tuple[str, str]:
    user_prompt = INITIAL_GENERATION_PROMPT.format(task_description=task_description)
    return SYSTEM_PROMPT, user_prompt


def build_verification_error_prompt(previous_strategy: str, error_message: str) -> tuple[str, str]:
    user_prompt = VERIFICATION_ERROR_REFINEMENT_PROMPT.format(
        previous_strategy=previous_strategy,
        error_message=error_message,
    )
    return SYSTEM_PROMPT, user_prompt


def build_runtime_error_prompt(previous_strategy: str, error_traceback: str) -> tuple[str, str]:
    user_prompt = RUNTIME_ERROR_REFINEMENT_PROMPT.format(
        previous_strategy=previous_strategy,
        error_traceback=error_traceback,
    )
    return SYSTEM_PROMPT, user_prompt


def build_compilation_error_prompt(previous_strategy: str, error_message: str) -> tuple[str, str]:
    user_prompt = COMPILATION_ERROR_REFINEMENT_PROMPT.format(
        previous_strategy=previous_strategy,
        error_message=error_message,
    )
    return SYSTEM_PROMPT, user_prompt


def build_format_repair_prompt(previous_strategy: str) -> tuple[str, str]:
    user_prompt = FORMAT_REPAIR_PROMPT.format(previous_strategy=previous_strategy)
    return SYSTEM_PROMPT, user_prompt


