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

**Strategy must fit the task.** The goal is to auto-synthesize whatever CSD best fits the task — not always the same pattern. The only hard requirements are: the final output must be grammatically correct under the parser, and when the task uses delimiters, the constrained part must appear in << >> (or whatever delimiters the task specifies; they can be set at the start of the program). Possible strategies include but are not limited to: Crane-style (unconstrained then constrained segments); k unconstrained steps then RollbackToValidPrefix to repair; compare n unconstrained generations and pick the best valid one; mostly constrained; or mostly unconstrained with a single constrained segment. Delimiters are only used when the task requires a constrained segment; some tasks may not use them at all.

**When the task has both plain text and a delimited constrained part** (e.g. "plain text then << formula >>"): the strategy must handle both phases. (1) Use *unconstrained* decoding (UnconstrainedStep) for the plain-text part so the LM can generate free text. (2) Switch to *constrained* decoding (ConstrainedStep) only when inside the constrained region (e.g. after emitting the left delimiter). Guard access to the last token with |generated| > 0 (e.g. use |generated| > 0 && generated[|generated|-1] != LeftDelimiter) to avoid index-out-of-range when generated is empty. Do NOT use only ConstrainedStep from the start for such tasks — that forces every token to satisfy the grammar and prevents normal plain-text generation. Do NOT call CSDHelpers.UnconstrainedPreservesValidWhenPermissive unless the parser is permissive for the current prefix; many parsers (e.g. FOLIO) are not permissive outside the << >> segment, so omit that lemma to avoid precondition failures.

You must output ONLY the Dafny method body for:

  method MyCSDStrategy(lm: LM, parser: Parser, prompt: Prefix, maxSteps: nat, eosToken: Token) returns (generated: Prefix, remainingSteps: nat)
    ...
    ensures remainingSteps >= 0 && remainingSteps <= maxSteps

The template already provides: var delim := new Delimiter(LeftDelimiter, RightDelimiter); var helpers := new CSDHelpers(lm, parser, delim); var stepsLeft := maxSteps; [YOUR BODY]; remainingSteps := stepsLeft;
So you must only assign to `generated` and update `stepsLeft` in your loop. Do NOT assign to remainingSteps (the template does that).

Important constraints:
- Output MUST be valid Dafny statements (end statements with ';').
- Initialize/assign the out-parameter `generated` (e.g. generated := []).
- **stepsLeft is ALREADY DECLARED by the template.** Do NOT write \"var stepsLeft := maxSteps;\" or \"var stepsLeft :=\"; that causes \"Duplicate local-variable name: stepsLeft\". Just use stepsLeft in your loop and assign stepsLeft := newSteps after each step call.
- Do NOT redeclare `generated`, `stepsLeft`, `helpers`, or `delim`. Do NOT assign to `remainingSteps`.
- You MUST use the `helpers` instance (type `CSDHelpers`) which is already instantiated with lm, parser, and delimiter.
- Use `helpers.<Method>` (instance call). UnconstrainedStep and ConstrainedStep each take `(prompt, generated, stepsLeft)` — do NOT pass lm or parser, they are captured by the helpers instance. Each returns (next, stepsLeft'); assign stepsLeft := stepsLeft' after each call.
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

- **Token is a type synonym for string.** It has NO static members. To compare with delimiter tokens use the module constants `LeftDelimiter` and `RightDelimiter` (e.g. `next == LeftDelimiter` or `next == RightDelimiter`). Do NOT use Token.LeftDelimiter or Token.RightDelimiter — they do not exist.

To decide when to use ConstrainedStep: guard with `helpers.InsideDelimitedWindow(generated) && !parser.IsCompletePrefix(helpers.GetDelimitedContent(generated))`. Only call ConstrainedStep when both conditions are true. Do NOT use parser.CanConstrain() or parser.CanConstrain(generated) — they do not exist.
Do NOT use .Exists, .Any, .Where, or lambda syntax on sequences — Dafny sequences have no such methods. To test "some token in seq satisfies P", use: exists token :: token in seq && parser.ValidNextToken(generated, token).

## Dafny syntax (follow exactly to avoid verification failures)

- **Assignment only**: There is no `++` or `--`. Use full assignment: `stepsLeft := stepsLeft - 1;`, `i := i + 1;`.
- **Sequence concatenation**: Use `+` only: `generated := generated + [next];`. Do NOT use `++`.
- **Sequence length**: Use `|seq|` (e.g. `|generated|`, `|generated| < maxSteps`). No `.Length` on seq.
- **Sequence index**: Use `s[i]` for element at index i. Last element when non-empty: `s[|s|-1]`. No `.Last()`. When testing the last token (e.g. for delimiter), guard with |generated| > 0: use (|generated| > 0 && generated[|generated|-1] != LeftDelimiter) so empty generated does not cause index out of range.
- **Sequence slice**: `s[..k]` is prefix up to (not including) k; `s[k..]` is from k to end.
- **Sequence membership / "exists"**: Do NOT use seq.Exists(...) or lambdas. Use Dafny quantifier: exists x :: x in seq && predicate(x). Example: hasValid := exists token :: token in validTokens && parser.ValidNextToken(generated, token);
- **Parser**: The predicate is ValidNextToken(prefix, token), not IsValidNextToken. There is no IsValidNextToken.
- **Variables**: Declare with `var name := value;` or `var name: Type := value;`. No `int x = 0` style.
- **Loop condition**: Use `while condition {{ ... }}`. Condition must be a boolean expression (e.g. `|generated| < maxSteps && !parser.IsCompletePrefix(generated)`).
- **For-loops (if needed)**: Use Dafny syntax `for i := 0 to |seq| - 1 {{ ... }}`. Use `{{` after the range, not `do`. Do NOT use Python/Rust-style `for i in 0 .. |seq| - 1` or `for ... do`.
- **Statements**: Every statement ends with `;`. No semicolon after `}` of blocks.
- **Equality**: Use `==` for equality, `!=` for disequality. Use `:=` for assignment only.
- **Mixed && and ||**: Use parentheses to disambiguate (e.g. `A || (B && C)` not `A || B && C`).
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

1) helpers.UnconstrainedStep(prompt, generated, stepsLeft) returns (next: Token, stepsLeft': nat)
   - One unconstrained step; consumes one step (stepsLeft' == stepsLeft - 1). Requires stepsLeft >= 1.
   - Use: var next, newSteps := helpers.UnconstrainedStep(prompt, generated, stepsLeft); generated := generated + [next]; stepsLeft := newSteps;

2) helpers.ConstrainedStep(prompt, generated, stepsLeft) returns (next: Token, stepsLeft': nat)
   - One constrained step; consumes one step. Requires: InsideDelimitedWindow(generated), !parser.IsCompletePrefix(helpers.GetDelimitedContent(generated)), stepsLeft >= 1.
   - InsideDelimitedWindow(generated) means generated already contains LeftDelimiter ("<<") with no matching ">>" yet. YOU CANNOT call ConstrainedStep when generated == [] or before "<<" has been emitted — it will fail verification. Always use UnconstrainedStep first until LeftDelimiter appears in generated, then switch to ConstrainedStep.
   - Use: var next, newSteps := helpers.ConstrainedStep(prompt, generated, stepsLeft); generated := generated + [next]; stepsLeft := newSteps;

3) helpers.RollbackToValidPrefix(generated) returns (repaired: Prefix)
   - Trims invalid tokens from the end of generated. Does not consume steps.
   - Use: generated := helpers.RollbackToValidPrefix(generated);

IMPORTANT: Do NOT pass `lm` or `parser` to any helpers method — they are captured by the helpers instance.

You MUST implement the strategy body as a loop that:
- Uses generated := []; (stepsLeft is already maxSteps). Loop while stepsLeft > 0 && !parser.IsCompletePrefix(generated).
- In the loop, call the appropriate primitive based on your strategy logic. ConstrainedStep may ONLY be called when helpers.InsideDelimitedWindow(generated) is true AND !parser.IsCompletePrefix(helpers.GetDelimitedContent(generated)) is true — both are verifiable preconditions. For all other states use UnconstrainedStep (or RollbackToValidPrefix to repair). How you track state, structure branches, or count steps is your design choice.
- If you use if/else with both step types and share next/newSteps across branches, declare var next: Token; var newSteps: nat; before the if/else and assign (without var) in each branch.
- YOU CANNOT call ConstrainedStep when generated == [] — InsideDelimitedWindow([]) is always false. You must use UnconstrainedStep until "<<" has been emitted.

**Loop invariants (required for verification):** Use at minimum:
  invariant lm.ValidTokensIdsLogits()
  invariant 0 <= stepsLeft <= maxSteps
  invariant |generated| + stepsLeft == maxSteps
  decreases stepsLeft
NOTE: Do NOT include `invariant parser.IsValidPrefix(generated)` when using UnconstrainedStep — unconstrained generation does not guarantee a valid prefix.

Output format:
- Return ONLY the method body (no signature, no outer braces).
- Multi-line bodies are allowed and encouraged when useful (e.g., locals, branching, loops).
- Ensure `generated` is assigned along all paths.
"""


INITIAL_GENERATION_PROMPT = """\
Generate a CSD strategy implementation for this use-case:

Use-case description: {task_description}

Choose the strategy to fit the task. Options include: Crane-style (unconstrained then constrained); k unconstrained steps then RollbackToValidPrefix; or other mixes. The only requirements are final output grammatically correct and, when the task uses delimiters, constrained content in << >>. If the use-case requires plain text then a constrained segment (e.g. << formula >>), use UnconstrainedStep for the plain part and ConstrainedStep only for the constrained part; guard last-token checks with |generated| > 0. Do NOT call UnconstrainedPreservesValidWhenPermissive unless the parser is permissive (many task parsers are not).

Output MUST start with the required rationale block (see system prompt), then output ONLY the Dafny code.

Output ONLY the method body (no signature, no outer braces). Do NOT wrap output in markdown code fences (no ```dafny).
Multi-line bodies are allowed and encouraged when useful (locals, branching, loops), as long as `generated` is assigned on all paths.

Your output must start with the rationale block, then Dafny statements. Include a while loop with the standard invariants (lm.ValidTokensIdsLogits(), 0 <= stepsLeft <= maxSteps, |generated| + stepsLeft == maxSteps, decreases stepsLeft). Assign generated on all paths. The template handles remainingSteps.
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
- That method DOES NOT EXIST. The ONLY methods on `helpers` are: UnconstrainedStep, ConstrainedStep, RollbackToValidPrefix.
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

**CRITICAL: If the error says "member 'LeftDelimiter' does not exist in type synonym 'Token'" or "member 'RightDelimiter' does not exist":**
- Token is a type synonym for string; it has no members. Use the module constants LeftDelimiter and RightDelimiter instead. Replace every `Token.LeftDelimiter` with `LeftDelimiter` and every `Token.RightDelimiter` with `RightDelimiter` (no Token. prefix).

**CRITICAL: If the error says "the method returns 0 value but is assigned to 1 variable":**
- RollbackToValidPrefix returns a Prefix — assign it: `generated := helpers.RollbackToValidPrefix(generated);`. Do NOT call it as a statement without assignment.

**CRITICAL: If the error says "Ambiguous use of && and ||" or "Use parentheses to disambiguate":**
- When mixing `||` and `&&` in one condition, add parentheses so Dafny knows the grouping. Example: use `parser.IsCompletePrefix(generated) || (|generated| > 0 && generated[|generated|-1] != LeftDelimiter)` not `... || |generated| > 0 && generated[...] != LeftDelimiter`.

**CRITICAL: If the error says "missing semicolon at end of statement" or "lbrace expected" at a for-loop:**
- Dafny for-loops use braces, not \"do\". Write `for i := 0 to |prompt| - 1 {{` (with `{{`), not `for i := 0 to |prompt| - 1 do`. The body of an `if` must be in braces when it has multiple statements: `if condition {{ stmt1; stmt2; }}`.

**CRITICAL: If the error says "rbrace expected" or you wrote \"remainingSteps := stepsLeft;\":**
- The template already assigns remainingSteps. Do NOT write `remainingSteps := stepsLeft;` in your body.

**CRITICAL: If the error says "index out of range" at generated[|generated|-1]:**
- When generated is empty, |generated|-1 is invalid. Guard the condition: use (|generated| > 0 && generated[|generated|-1] != LeftDelimiter) instead of just generated[|generated|-1] != LeftDelimiter.

**CRITICAL: If the error says "precondition for this call could not be proved" and mentions "InsideDelimitedWindow" for a ConstrainedStep call:**
- ConstrainedStep requires InsideDelimitedWindow(generated), meaning "<<" must already be in generated with no matching ">>" yet. generated starts as [] which is never inside the window.
- You CANNOT call ConstrainedStep unconditionally in a loop. It must only be called when helpers.InsideDelimitedWindow(generated) is true AND !parser.IsCompletePrefix(helpers.GetDelimitedContent(generated)) is true. Use a different primitive (e.g. UnconstrainedStep) for all other cases. Redesign your strategy logic accordingly — the specific approach (how you structure the branches, what you track with extra variables, etc.) is up to you.

**CRITICAL: If the error says "precondition for this call could not be proved" for UnconstrainedPreservesValidWhenPermissive or mentions IsPermissive:**
- That lemma requires parser.IsPermissive(generated). Many parsers (e.g. FOLIO) are not permissive outside << >>. Remove the line CSDHelpers.UnconstrainedPreservesValidWhenPermissive(parser, generated, next); from your strategy.

**CRITICAL: If the error mentions "invariant could not be proved" for stepCounter <= maxSteps:**
- Remove the invariant stepCounter <= maxSteps (and stepCounter >= 0 && stepCounter <= maxSteps if present). stepCounter is not bounded by stepsLeft in all branches.

**CRITICAL: If the error mentions invalid syntax for a for-loop or "for i in 0 ..":**
- Dafny does not use `for i in 0 .. |prompt| - 1`. Use `for i := 0 to |prompt| - 1 {{ ... }}` instead (assign with `:=`, use `to`, not `in` or `..`).

Common fixes:
- Do NOT write `var generated: Prefix;` or `var stepsLeft := maxSteps;` (duplicate/shadowing). The template already provides stepsLeft. Assign to the existing out-parameter `generated` only.
- ConstrainedStep requires !parser.IsCompletePrefix(generated). Guard your loop or use a valid loop invariant.
- **If the error mentions precondition, invariant, or "assertion might not hold" for a while loop:** Add explicit loop invariants and a decreases clause. For a loop that only calls ConstrainedStep, use: `invariant parser.IsValidPrefix(generated);` `invariant |generated| <= maxSteps;` `decreases maxSteps - |generated|;` (semicolons between clauses, or newline-separated).
- Use `x := x + 1` not `x++`. Pass stepsLeft into UnconstrainedStep/ConstrainedStep and assign the returned (next, stepsLeft) back.
- RollbackToValidPrefix: call as generated := helpers.RollbackToValidPrefix(generated);.

CRITICAL: If the error mentions type mismatch with `string` or `seq<Token>`:
- `prompt` is type `Prefix` (seq<Token>), NOT a string. You CANNOT use `+` with strings.
- Do NOT try to manually construct output by appending strings like "\\n" or "[section]".
- Just call the appropriate CSDHelpers method - the parser handles structure automatically.

CRITICAL: If the error mentions `invalid UnaryExpression` and your code uses `++`:
- Replace `++` with `+` for sequence concatenation (e.g., `a + b`, not `a ++ b`).

**CRITICAL: If the task requires "plain text" or "unconstrained" first and then a constrained segment (e.g. << formula >>), but your strategy uses only ConstrainedStep:**
- You must use BOTH UnconstrainedStep and ConstrainedStep. Use UnconstrainedStep when the prefix is complete (parser.IsCompletePrefix(generated)) or when you are not yet in the constrained region; use ConstrainedStep when inside the constrained segment. Declare var next: Token; var newSteps: nat; before the if/else.

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
    # Use replace() so strategy/error text containing { } doesn't break format()
    user_prompt = VERIFICATION_ERROR_REFINEMENT_PROMPT.replace(
        "{previous_strategy}", previous_strategy
    ).replace("{error_message}", error_message)
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


