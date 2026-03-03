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

  method MyCSDStrategy(lm: LM, parser: Parser, prompt: Prefix, maxSteps: nat, eosToken: Token) returns (generated: Prefix, cost: int)
    modifies lm.Logits
    requires lm.ValidTokensIdsLogits()
    requires parser.IsValidPrefix([])
    requires forall t: Token :: t in parser.ValidNextTokens([]) ==> t in lm.Tokens
    requires "<<" in lm.Tokens && ">>" in lm.Tokens
    ensures lm.ValidTokensIdsLogits()
    ensures |generated| <= maxSteps
    ensures cost <= maxSteps  // built-in: total LM calls must not exceed maxSteps

Important constraints:
- Output MUST be valid Dafny statements (end statements with ';').
- Initialize/assign the out-parameter `generated`.
- Initialize/assign the out-parameter `cost` (e.g., `cost := helpers.cost;`).
- Do NOT redeclare `generated` or `cost` as local variables. They are already method out-parameters.
- You MUST use the `helpers` instance (type `CSDHelpers`) which is already instantiated for you.
- Do NOT use `CSDHelpers.<Method>` (static call); use `helpers.<Method>` (instance call).
- `CSDHelpers` now tracks `cost: int`. Every helper call increments `helpers.cost`.
- The cost contract `ensures cost <= maxSteps` is built-in. Every strategy must satisfy it.
  All verified strategy helpers already guarantee `cost <= old(cost) + maxSteps`.
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

All helpers below satisfy `cost <= maxSteps` (the built-in cost contract).

1) Pure constrained (cost == |generated| <= maxSteps):
  generated := helpers.PureConstrainedGeneration(lm, parser, prompt, maxSteps);

2) Optimistic then fallback (cost <= maxSteps):
  generated := helpers.TryUnconstrainedThenConstrained(lm, parser, prompt, maxSteps, N);
  N in [1..20], N <= maxSteps. Uses N tokens unconstrained, then up to maxSteps-N constrained.

3) Interleaved hybrid (cost <= maxSteps):
  generated := helpers.HybridGeneration(lm, parser, prompt, maxSteps);
  Starts constrained. When << is generated (must be valid per grammar), switches to unconstrained.
  When >> is generated, validates and switches back. Shares a single cost budget = maxSteps.
  WARNING: Only use this when << is itself a valid first token in the grammar. If the grammar
  starts with an expression (e.g., s_expr), << will never be generated in constrained mode, so
  the strategy just generates a short expression and exits — it will NOT produce reasoning text.

4) CRANE-style generation (cost <= maxSteps):
  generated := helpers.CraneGeneration(lm, parser, prompt, maxSteps, minReasoningSteps, eosToken);
  Starts unconstrained (free-form text). When the model generates <<, switches to constrained
  (parser-enforced) until the parser completes (e.g., expression + >>). Then switches back to
  unconstrained. Repeats until EOS or maxSteps.
  `minReasoningSteps` (nat): minimum unconstrained tokens before << is allowed (use 5-20).
  `eosToken` is available as a method parameter.
  USE THIS for tasks where the model generates reasoning text interspersed with << expr >> windows.
  This is the correct strategy for GSM-Symbolic and CRANE-style math reasoning.

5) Speculative (cost <= maxSteps):
  generated := helpers.SpeculativeGeneration(lm, parser, prompt, maxSteps, N);
  N in [2..8]. Rejected speculations still cost tokens from the shared budget.

6) Completion of an existing valid prefix (cost == |generated| - |partial|):
  generated := helpers.CompletePrefix(lm, parser, prompt, partial, maxSteps);

7) Generate with reasonable length (cost == |generated| <= maxSteps):
  generated := helpers.GenerateWithReasonableLength(lm, parser, prompt, maxSteps, reasonableLength);

8) Generate until first complete (cost == |generated| <= maxSteps):
  generated := helpers.GenerateUntilFirstComplete(lm, parser, prompt, maxSteps);

9) Generate multiple candidates and select best (cost <= maxSteps):
  generated := helpers.GenerateAndSelectBest(lm, parser, prompt, maxSteps, numCandidates, preferShorter);
  Requires numCandidates <= maxSteps. Each candidate gets maxSteps/numCandidates budget.

## Utility Functions for Contracts (use in `ensures` clauses)

1) Extract content between delimiters:
  `helpers.ExtractContentBetweenDelimiters(helpers.PrefixToString(generated), "<<", ">>")`
  Returns the content found between the last occurrence of "<<" and ">>". Useful for asserting things about the reasoning trace.

2) Convert prefix to string:
  `helpers.PrefixToString(generated)`
  Converts a sequence of tokens into a single concatenated string. Useful for string-based contracts.

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

Example format (showing the required rationale + a simple call):
  // CSD_RATIONALE_BEGIN
  // <your reasoning here>
  // CSD_RATIONALE_END
  generated := helpers.<SomeHelper>(lm, parser, prompt, maxSteps, ...);
  cost := helpers.cost;

Example format (branching):
  // CSD_RATIONALE_BEGIN
  // <your reasoning here>
  // CSD_RATIONALE_END
  if maxSteps < 5 {{
    generated := helpers.<HelperA>(lm, parser, prompt, maxSteps);
  }} else {{
    generated := helpers.<HelperB>(lm, parser, prompt, maxSteps, 5);
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

**CRITICAL: If the error says "member X does not exist in class 'CSDHelpers'":**
- That method DOES NOT EXIST. Do NOT try to use it again, even with different arguments.
- You MUST use ONLY the methods listed in the system prompt under "Available VERIFIED strategy helpers".
- The ONLY methods on `helpers` are: PureConstrainedGeneration, TryUnconstrainedThenConstrained,
  HybridGeneration, CraneGeneration, SpeculativeGeneration, CompletePrefix,
  GenerateWithReasonableLength, GenerateUntilFirstComplete, GenerateAndSelectBest,
  PrefixToString, ExtractContentBetweenDelimiters.
- There is NO ContainsArithmeticExpression, ContainsAnyVar, ContainsExpression, or similar method.
- Do NOT write loops or conditionals that check output content. The parser handles validation automatically.
- Instead, use a COMPLETELY DIFFERENT approach: just call a verified helper directly.

Common fixes:
- Do NOT write `var generated: Prefix;` (duplicate/shadowing). Assign to the existing out-parameter `generated` instead.
- Avoid subtracting from `maxSteps` (e.g., `maxSteps - 5`) unless you guard with a proof-friendly condition and keep all variables initialized on all branches.
- **CRITICAL**: When using TryUnconstrainedThenConstrained with a constant N, you MUST guard it with `if maxSteps >= N` or use PureConstrainedGeneration when maxSteps is too small. Otherwise Dafny cannot prove the precondition `N <= maxSteps`.
  Example fix for "precondition could not be proved" error:
    WRONG: generated := helpers.TryUnconstrainedThenConstrained(lm, parser, prompt, maxSteps, 5);
    RIGHT: if maxSteps < 5 {{
             generated := helpers.PureConstrainedGeneration(lm, parser, prompt, maxSteps);
           }} else {{
             generated := helpers.TryUnconstrainedThenConstrained(lm, parser, prompt, maxSteps, 5);
           }}

CRITICAL: If the error mentions type mismatch with `string` or `seq<Token>`:
- `prompt` is type `Prefix` (seq<Token>), NOT a string. You CANNOT use `+` with strings.
- Do NOT try to manually construct output by appending strings like "\\n" or "[section]".
- Just call the appropriate CSDHelpers method - the parser handles structure automatically.

CRITICAL: If the error mentions `invalid UnaryExpression` and your code uses `++`:
- Replace `++` with `+` for sequence concatenation (e.g., `a + b`, not `a ++ b`).

If the same strategy keeps failing, switch to a COMPLETELY DIFFERENT verified strategy helper. Do not keep trying the same approach with minor tweaks.
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
  SpeculativeGeneration -> CraneGeneration -> TryUnconstrainedThenConstrained -> PureConstrainedGeneration
- For CRANE-style tasks (math/logic with << >> delimiters in free text), prefer CraneGeneration.
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


EVALUATION_FAILURE_REFINEMENT_PROMPT = """\
Your method body passed verification, compilation, and runtime testing, but performed poorly on actual dataset evaluation.

Previous attempt:
```dafny
{previous_strategy}
```

Evaluation results:
```
{evaluation_feedback}
```

The strategy runs correctly but produces outputs that don't meet quality thresholds. Consider:

1. **Format issues (format rate = 0%)**: The model is not generating any `<< >>` delimiters at all.
   - This almost certainly means the strategy is starting in constrained mode (e.g., HybridGeneration,
     PureConstrainedGeneration), which forces the grammar immediately and never generates free text.
   - The grammar starts with `s_expr` (an arithmetic expression), so constrained mode generates a
     short expression and terminates — it never reaches `<<`.
   - FIX: Switch to `CraneGeneration`. It starts UNCONSTRAINED (free text), watches for `<<`,
     then switches to constrained mode for the expression, then back to unconstrained.
   - CraneGeneration is the ONLY strategy that correctly handles CRANE-style `<< expr >>` generation
     where the model produces reasoning text alongside constrained expression windows.

2. **Accuracy issues**: If format rate > 0 but accuracy is low, the model's constrained expressions may be wrong.
   - Consider increasing `minReasoningSteps` in CraneGeneration to let the model reason more.
   - The expressions must contain variables (not just numbers). The grammar enforces this.

3. **Syntax issues**: If syntax rate is low but format rate is OK, the expressions don't match the grammar.
   - CraneGeneration with constrained mode inside << >> guarantees grammar-valid expressions.
   - PureConstrainedGeneration guarantees syntax but won't produce reasoning text.

4. **Strategy selection**:
   - GSM-Symbolic (math reasoning with `<< expr >>` windows): USE CraneGeneration
   - FOLIO (logic with `<< formula >>` windows): USE CraneGeneration
   - Simple structured output only: PureConstrainedGeneration

Rules:
- Output ONLY a corrected method body (no signature, no braces).
- The corrected body MUST include the required rationale block at the top.
- Try a DIFFERENT strategy or different parameters than the previous attempt.
- If you've tried multiple strategies without success, try adjusting parameters (e.g., increase maxSteps budget, change speculative window size).
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


def build_evaluation_failure_prompt(previous_strategy: str, evaluation_feedback: str) -> tuple[str, str]:
    user_prompt = EVALUATION_FAILURE_REFINEMENT_PROMPT.format(
        previous_strategy=previous_strategy,
        evaluation_feedback=evaluation_feedback,
    )
    return SYSTEM_PROMPT, user_prompt


