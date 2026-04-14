# Agents Guide

## Purpose
This repository studies autosynthesizing verified constrained decoding strategies (CSDs).

A CSD here is an algorithm that takes:
- an LM
- a parser / grammar

and guarantees that the answer-bearing part of the output is grammar-valid.

## Source Of Truth
The repository is Python-first.

Authoritative inputs:
- [VerifiedAgentSynthesis.py](/home/advayth2/projects/verified-agent-synthesis/VerifiedAgentSynthesis.py): verified helper library and contracts
- [GeneratedAgentTemplate.py](/home/advayth2/projects/verified-agent-synthesis/GeneratedAgentTemplate.py): synthesis template for new strategies

Dafny is a generated intermediate used for verification and compilation.
Do not treat hand-authored `.dfy` files as the primary authoring format.

## Core Architecture
The active template uses a split output architecture:
- `generated`: expressive free-form text outside the final constrained answer segment
- `answer`: constrained answer content only, without delimiters

The template always assembles the final output as:
- `generated + [LeftDelimiter] + answer + [RightDelimiter]`

This means the answer-bearing segment is always the final `<< ... >>` span, and that span must be grammar-valid.
A trivial unconstrained-only loop should not satisfy the template.

## Core Flow
The synthesis / verification / compilation loop is:
1. An LLM generates a Python body for `MyCSDStrategy`.
2. That body is injected into [GeneratedAgentTemplate.py](/home/advayth2/projects/verified-agent-synthesis/GeneratedAgentTemplate.py#L1).
3. The transpiler lowers Python sources to Dafny in a temporary workspace.
4. `dafny verify` checks the transpiled program.
5. `dafny build --target:py` compiles the transpiled program to Python.
6. The evaluation stack runs the compiled modules on dataset samples.

## Strategy Requirements
Generated strategies are Python method bodies, not Dafny method bodies.

Required conventions:
- rationale blocks use Python comments:
  - `# CSD_RATIONALE_BEGIN`
  - `# CSD_RATIONALE_END`
- loop invariants are written as Python comments immediately above `while` loops:
  - `# invariant ...`
  - `# decreases ...`
- invariant / decreases comments must use Dafny syntax
- executable statements must use Python syntax

Transpiler-supported Python constructs that are especially relevant for synthesis:
- `next_token is None` / `next_token is not None`
- `isinstance(token, str)` when the model emits defensive type checks
- `str(token)` identity-style coercions
- token predicates like `token.isalpha()` and `token.isdigit()`
- `list_var.append(token)` for local token buffers
- `break` inside `while` loops

These are lowered to Dafny-friendly forms by [transpiler/transpiler.py](/home/advayth2/projects/verified-agent-synthesis/transpiler/transpiler.py#L1238).
If a newly generated strategy fails because of another ordinary Python construct, prefer extending the transpiler plus adding a focused test before overfitting prompts around that one syntax error.

Helper usage conventions:
- use `helpers.ExpressiveStep(...)` for expressive free-form output in `generated`
- use `helpers.ConstrainedAnswerStep(...)` to build the constrained `answer` channel
- do not rely on `helpers.UnconstrainedStep(...)` for synthesized strategies
- do not rely on `helpers.ConstrainedStep(...)`, `helpers.InsideDelimitedWindow(...)`, or rollback-style repair as the main synthesis pattern

## Novelty Objective
This project is not trying to rediscover boring baseline CSDs.

Generated CSDs should be:
- expressive
- accurate
- novel
- verifier-compatible

Strategies that are explicitly not interesting enough:
- basic CRANE-style window switching
- a simple `while` loop with an `if constrained else unconstrained` shell
- generate unconstrained output and then rollback to a valid prefix

Generated strategies should instead explore richer control policies with meaningful local state.
The synthesis stack should prefer strategies with multiple interacting control signals and nontrivial evolution over time, while still guaranteeing that the final answer segment is grammar-constrained.

## GSM-Specific Guidance
For GSM-style tasks:
- the final `<< ... >>` segment is the answer-bearing segment used for grading
- that final segment should usually converge to a short arithmetic expression or equation
- the right-most numeric value in the final constrained segment is what the evaluator will typically extract as the answer
- free-form reasoning outside the final answer segment is allowed, but delimiter control should stay explicit

The GSM grammar in [grammars/gsm.lark](/home/advayth2/projects/verified-agent-synthesis/grammars/gsm.lark#L1) is now aimed at a single final answer expression/equation rather than many interleaved constrained windows.

## Important Files
- [GeneratedAgentTemplate.py](/home/advayth2/projects/verified-agent-synthesis/GeneratedAgentTemplate.py#L1): Python template with the synthesis hole
- [VerifiedAgentSynthesis.py](/home/advayth2/projects/verified-agent-synthesis/VerifiedAgentSynthesis.py#L1): verified helper library and contracts
- [transpiler/transpiler.py](/home/advayth2/projects/verified-agent-synthesis/transpiler/transpiler.py#L1): Python-to-Dafny lowering and verification entrypoint
- [synthesis/generator.py](/home/advayth2/projects/verified-agent-synthesis/synthesis/generator.py#L1): Qwen strategy generation and structural filtering
- [synthesis/prompts.py](/home/advayth2/projects/verified-agent-synthesis/synthesis/prompts.py#L1): prompting rules for Python strategy bodies
- [synthesis/feedback_loop.py](/home/advayth2/projects/verified-agent-synthesis/synthesis/feedback_loop.py#L1): generate → verify → compile → run → evaluate loop
- [synthesis/evaluator.py](/home/advayth2/projects/verified-agent-synthesis/synthesis/evaluator.py#L1): dataset-driven evaluation and syntax checks
- [dafny/dafny](/home/advayth2/projects/verified-agent-synthesis/dafny/dafny#L1): repo-local Dafny binary

## Verification Commands
Verify the helper library:

```bash
python transpiler/transpiler.py VerifiedAgentSynthesis.py
```

Print only the transpiled Dafny:

```bash
python transpiler/transpiler.py VerifiedAgentSynthesis.py --print-dafny
```

Verify the strategy template with dependencies auto-transpiled:

```bash
python transpiler/transpiler.py GeneratedAgentTemplate.py
```

## Editing Guidance
- Do not casually change [VerifiedAgentSynthesis.py](/home/advayth2/projects/verified-agent-synthesis/VerifiedAgentSynthesis.py#L1); it defines the contract surface for the whole synthesis stack.
- Keep the final answer-channel guarantee intact: the last `<< ... >>` segment must remain grammar-constrained.
- Prefer Python-first prompt repair and transpiler fixes over reintroducing Dafny-authored strategy templates.
- Remove or avoid stale Dafny-era synthesis assumptions when they conflict with the current Python-first architecture.
- When a failure report shows the same verification error across many attempts, first separate:
  - transpiler subset gaps such as unsupported Python syntax
  - from real proof failures such as missing invariants or unmet helper preconditions
- After extending the transpiler, add a direct unit test under [tests](/home/advayth2/projects/verified-agent-synthesis/tests) for the specific construct that failed.
- When improving synthesis quality, push on both:
  - search pressure toward novel strategies
  - proof ergonomics so novel strategies can still verify

## Sanity Checks
After changing the synthesis stack, run:

```bash
python -m py_compile synthesis/*.py transpiler/*.py GeneratedAgentTemplate.py VerifiedAgentSynthesis.py run_synthesis.py test_pipeline.py
python transpiler/transpiler.py VerifiedAgentSynthesis.py
python transpiler/transpiler.py GeneratedAgentTemplate.py
python test_pipeline.py
```
