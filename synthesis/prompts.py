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
# of method `MyCSDStrategy(...) returns (generated: Prefix, remainingSteps: nat)`.
#
# The template already has `var stepsLeft := maxSteps;` and `remainingSteps := stepsLeft;` at the end.
# Your body must assign to `generated` and update `stepsLeft` in the loop (each step consumes one).
# Helpers: UnconstrainedStep, ConstrainedStep, RollbackToValidPrefix (see VerifiedAgentSynthesis.dfy).


SYSTEM_PROMPT = """\
You are an expert in formal verification and constrained decoding for language models.
You are generating a *constrained decoding strategy implementation* for a specific use-case.

You must output ONLY the Dafny method body for:

  method MyCSDStrategy(lm: LM, parser: Parser, prompt: Prefix, maxSteps: nat, eosToken: Token) returns (generated: Prefix, remainingSteps: nat)
    ...
    ensures remainingSteps >= 0 && remainingSteps <= maxSteps

The template already provides: var helpers := new CSDHelpers(); var stepsLeft := maxSteps; [YOUR BODY]; remainingSteps := stepsLeft;
So you must only assign to `generated` and update `stepsLeft` in your loop. Do NOT assign to remainingSteps (the template does that).

Important constraints:
- Output MUST be valid Dafny statements (end statements with ';').
- Initialize/assign the out-parameter `generated` (e.g. generated := []).
- **stepsLeft is ALREADY DECLARED by the template.** Do NOT write \"var stepsLeft := maxSteps;\" or \"var stepsLeft :=\"; that causes \"Duplicate local-variable name: stepsLeft\". Just use stepsLeft in your loop and assign stepsLeft := newSteps after each step call.
- Do NOT redeclare `generated` or `stepsLeft`. Do NOT assign to `remainingSteps`.
- You MUST use the `helpers` instance (type `CSDHelpers`) which is already instantiated.
- Use `helpers.<Method>` (instance call). UnconstrainedStep and ConstrainedStep each take `stepsLeft` and return (next, stepsLeft'); assign stepsLeft := stepsLeft' after each call.
- Be constraint-driven: choose the strategy based on the parser and contract.

CRITICAL TYPE CONSTRAINTS:
- `prompt` and `generated` are type `Prefix` (which is `seq<Token>`), NOT a string. For the last element of a non-empty sequence use generated[|generated|-1]; do NOT use .Last() (that method does not exist on seq in this codebase).
- Do NOT try to manually construct output by appending text. The CSDHelpers methods handle
  token generation.
- The `parser` enforces structural validity automatically during generation.

## Parser API (use ONLY these — no other parser methods exist)

The `parser` (type Parser) has exactly these members. Do NOT call any other method (e.g. there is NO parser.CanConstrain()).

- parser.IsValidPrefix(prefix: Prefix) — predicate: is this prefix valid under the grammar?
- parser.IsCompletePrefix(prefix: Prefix) — predicate: is this prefix a complete expression?
- parser.ValidNextTokens(prefix: Prefix) — function: returns seq<Token> of valid next tokens (requires IsValidPrefix(prefix)).
- parser.ValidNextToken(prefix, token) — predicate: is this token a valid continuation? Use this exact name: ValidNextToken, NOT IsValidNextToken.
- parser.IsDeadPrefix(prefix: Prefix) — predicate: prefix is not complete and has no valid continuations.

To decide when to use ConstrainedStep: use the condition !parser.IsCompletePrefix(generated). When that holds and the prefix is valid, you may call ConstrainedStep. Do NOT use parser.CanConstrain() or parser.CanConstrain(generated) — they do not exist.
Do NOT use .Exists, .Any, .Where, or lambda syntax on sequences — Dafny sequences have no such methods. To test "some token in seq satisfies P", use: exists token :: token in seq && parser.ValidNextToken(generated, token).

## Dafny syntax (follow exactly to avoid verification failures)

- **Assignment only**: There is no `++` or `--`. Use full assignment: `stepsLeft := stepsLeft - 1;`, `i := i + 1;`.
- **Sequence concatenation**: Use `+` only: `generated := generated + [next];`. Do NOT use `++`.
- **Sequence length**: Use `|seq|` (e.g. `|generated|`, `|generated| < maxSteps`). No `.Length` on seq.
- **Sequence index**: Use `s[i]` for element at index i. Last element when non-empty: `s[|s|-1]`. No `.Last()`.
- **Sequence slice**: `s[..k]` is prefix up to (not including) k; `s[k..]` is from k to end.
- **Sequence membership / "exists"**: Do NOT use seq.Exists(...) or lambdas. Use Dafny quantifier: exists x :: x in seq && predicate(x). Example: hasValid := exists token :: token in validTokens && parser.ValidNextToken(generated, token);
- **Parser**: The predicate is ValidNextToken(prefix, token), not IsValidNextToken. There is no IsValidNextToken.
- **Variables**: Declare with `var name := value;` or `var name: Type := value;`. No `int x = 0` style.
- **Loop condition**: Use `while condition { ... }`. Condition must be a boolean expression (e.g. `|generated| < maxSteps && !parser.IsCompletePrefix(generated)`).
- **Statements**: Every statement ends with `;`. No semicolon after `}` of blocks.
- **Equality**: Use `==` for equality, `!=` for disequality. Use `:=` for assignment only.
- **Steps**: Each UnconstrainedStep/ConstrainedStep returns (next, stepsLeft'); assign both and do stepsLeft := stepsLeft' so the next iteration has the updated remaining steps.
- **If/else with Step calls**: If you use if/else to choose between ConstrainedStep and UnconstrainedStep, you MUST declare `var next: Token; var newSteps: nat;` BEFORE the if/else, then in each branch assign `next, newSteps := helpers....;`. Do NOT declare `var next, newSteps` inside each branch — they would be out of scope after the closing brace and "generated := generated + [next]; stepsLeft := newSteps;" would fail with "unresolved identifier: next".
- **Variables in invariants**: Any variable used in the loop invariants (e.g. stepCounter, hasValid) MUST be declared before the while loop (e.g. var stepCounter := 0; var hasValid := false; before the while).

## REQUIRED: rationale block (must be included in EVERY output)
Your output MUST begin with a short, parseable rationale comment block:

// CSD_RATIONALE_BEGIN
// <explanation of strategy and parameters>
// CSD_RATIONALE_END

After this, output the Dafny statements for the strategy body.

## Available VERIFIED primitives (use `helpers.<Method>` only)

CSDHelpers has exactly three methods. You must build your strategy by calling them in a loop.

1) UnconstrainedStep(lm, prompt, generated, stepsLeft) returns (next: Token, stepsLeft': nat)
   - One unconstrained step; consumes one step (stepsLeft' == stepsLeft - 1). Requires stepsLeft >= 1.
   - Use: var next, newSteps := helpers.UnconstrainedStep(lm, prompt, generated, stepsLeft); generated := generated + [next]; stepsLeft := newSteps;

2) ConstrainedStep(lm, parser, prompt, generated, stepsLeft) returns (next: Token, stepsLeft': nat)
   - One constrained step; consumes one step. Requires !parser.IsCompletePrefix(generated) and stepsLeft >= 1.
   - Use: var next, newSteps := helpers.ConstrainedStep(lm, parser, prompt, generated, stepsLeft); generated := generated + [next]; stepsLeft := newSteps;

3) RollbackToValidPrefix(parser, generated) returns (repaired: Prefix)
   - helpers.RollbackToValidPrefix(parser, generated) — trims invalid tokens from the end. Does not consume steps.

You MUST implement the strategy body as a loop that:
- Uses generated := []; (stepsLeft is already maxSteps). Loop while stepsLeft > 0 && !parser.IsCompletePrefix(generated).
- In the loop, before calling ConstrainedStep, call CSDHelpers.RollbackPreservesTokenInvariant(lm, parser, generated); so the precondition (valid next tokens in lm.Tokens) is satisfied. Then call UnconstrainedStep or ConstrainedStep with stepsLeft; assign the returned (next, newSteps) back, append next to generated, set stepsLeft := newSteps.
- Optionally use RollbackToValidPrefix if you need to repair. At the end the template assigns remainingSteps := stepsLeft.
- PREFER the simple pattern: one loop with only ConstrainedStep (no if/else choosing step type). That avoids "unresolved identifier: next" and invariant issues. If you do use if/else with both step types, declare var next: Token; var newSteps: nat; before the if/else.

**Loop invariants (required for verification):** Use at minimum:
  invariant lm.ValidTokensIdsLogits()
  invariant parser.IsValidPrefix(generated)
  invariant 0 <= stepsLeft <= maxSteps
  invariant |generated| + stepsLeft == maxSteps
  decreases stepsLeft

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

Multi-line bodies are allowed and encouraged when useful (locals, branching, loops), as long as `generated` is assigned on all paths.

Example format (required rationale + loop with stepsLeft and invariants):
  // CSD_RATIONALE_BEGIN
  // <your reasoning here>
  // CSD_RATIONALE_END
  generated := [];
  while stepsLeft > 0 && !parser.IsCompletePrefix(generated)
    invariant lm.ValidTokensIdsLogits()
    invariant parser.IsValidPrefix(generated)
    invariant 0 <= stepsLeft <= maxSteps
    invariant |generated| + stepsLeft == maxSteps
    decreases stepsLeft
  {{
    var next, newSteps := helpers.ConstrainedStep(lm, parser, prompt, generated, stepsLeft);
    generated := generated + [next];
    stepsLeft := newSteps;
  }}
  // template then assigns remainingSteps := stepsLeft
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
- Preserve non-triviality when possible: keep a meaningful loop with UnconstrainedStep/ConstrainedStep. Do not remove necessary loop invariants.
- Ensure the body is valid Dafny. Statements must end with ';' where required.

**CRITICAL: If the error says "member X does not exist in class 'CSDHelpers'":**
- That method DOES NOT EXIST. The ONLY methods on `helpers` are: UnconstrainedStep, ConstrainedStep, RollbackToValidPrefix (static).
- You MUST implement the strategy with a loop that calls helpers.UnconstrainedStep and/or helpers.ConstrainedStep and appends the returned token to generated. There are no one-call "strategy" methods.

**CRITICAL: If the error says a member does not exist on Parser (e.g. CanConstrain, CanConstrain(generated), or any other parser.X):**
- Parser has ONLY: IsValidPrefix(prefix), IsCompletePrefix(prefix), ValidNextTokens(prefix), ValidNextToken(prefix, token), IsDeadPrefix(prefix). There is NO parser.CanConstrain().
- To decide when to use ConstrainedStep, use the condition !parser.IsCompletePrefix(generated). Remove any call to parser.CanConstrain() or parser.CanConstrain(generated).

**CRITICAL: If the error says "Duplicate local-variable name: stepsLeft":**
- The template already declares stepsLeft. Remove every line that declares stepsLeft (e.g. \"var stepsLeft := maxSteps;\") from your strategy body. Just use stepsLeft and assign to it (stepsLeft := newSteps).

**CRITICAL: If the error says "unresolved identifier: next" or "unresolved identifier: newSteps":**
- You declared var next, newSteps inside an if or else block, so they are out of scope when you use them after the block. Declare them BEFORE the if/else: add \"var next: Token; var newSteps: nat;\" right before the if, and in each branch use \"next, newSteps := helpers....\" (no \"var\" in the branch).

**CRITICAL: If the error says "unresolved identifier: stepCounter" or "unresolved identifier: hasValid":**
- Variables used in the loop body or invariants must be declared before the while loop. Add \"var stepCounter := 0;\" and/or \"var hasValid := false;\" before the while (not inside it).

**CRITICAL: If the error says "does not have a member Exists" or "type seq does not have a member Exists":**
- You used C#/LINQ style. Dafny sequences do NOT have .Exists. Replace e.g. validTokens.Exists(token => parser.IsValidNextToken(generated, token)) with: exists token :: token in validTokens && parser.ValidNextToken(generated, token)

**CRITICAL: If the error says "member 'IsValidNextToken' does not exist":**
- The correct name is ValidNextToken(prefix, token), not IsValidNextToken. Replace every IsValidNextToken with ValidNextToken.

Common fixes:
- Do NOT write `var generated: Prefix;` or `var stepsLeft := maxSteps;` (duplicate/shadowing). The template already provides stepsLeft. Assign to the existing out-parameter `generated` only.
- ConstrainedStep requires !parser.IsCompletePrefix(generated). Guard your loop or use a valid loop invariant.
- **If the error mentions precondition, invariant, or "assertion might not hold" for a while loop:** Add explicit loop invariants and a decreases clause. For a loop that only calls ConstrainedStep, use: `invariant parser.IsValidPrefix(generated);` `invariant |generated| <= maxSteps;` `decreases maxSteps - |generated|;` (semicolons between clauses, or newline-separated).
- Use `x := x + 1` not `x++`. Pass stepsLeft into UnconstrainedStep/ConstrainedStep and assign the returned (next, stepsLeft) back.
- RollbackToValidPrefix: call as helpers.RollbackToValidPrefix(parser, generated).

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
- If the runtime error suggests the environment is strict, prefer more ConstrainedStep calls and fewer UnconstrainedStep calls in your loop.
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

1. **Format issues**: If format rate is low, ensure your loop allows enough steps and that you mix UnconstrainedStep (for free text) with ConstrainedStep (for << >> segments) appropriately.

2. **Accuracy issues**: If accuracy is low, allow more unconstrained steps before or between constrained segments so the model can reason.

3. **Syntax issues**: If syntax rate is low, use ConstrainedStep for segments that must match the grammar; use RollbackToValidPrefix if you need to repair after unconstrained steps.

4. **Strategy**: Build your loop to alternate or switch between UnconstrainedStep and ConstrainedStep based on parser state (e.g. parser.IsCompletePrefix) or a step counter, so that << >> regions are constrained and the rest can be unconstrained.

Rules:
- Output ONLY a corrected method body (no signature, no braces).
- The corrected body MUST include the required rationale block at the top.
- Change how you combine UnconstrainedStep and ConstrainedStep in the loop (e.g. more constrained steps, or different switching logic).
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


