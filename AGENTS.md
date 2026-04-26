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
- [VerifiedAgentSynthesis.py](/home/advayth2/projects/verified-agent-synthesis/generation/csd/VerifiedAgentSynthesis.py): verified helper library and contracts
- [GeneratedAgentTemplate.py](/home/advayth2/projects/verified-agent-synthesis/generation/csd/GeneratedAgentTemplate.py): synthesis template for new strategies

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
2. That body is injected into [GeneratedAgentTemplate.py](/home/advayth2/projects/verified-agent-synthesis/generation/csd/GeneratedAgentTemplate.py#L1).
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

These are lowered to Dafny-friendly forms by [verification/transpiler/transpiler.py](/home/advayth2/projects/verified-agent-synthesis/verification/transpiler/transpiler.py#L1238).
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

The GSM grammar in [utils/grammars/gsm.lark](/home/advayth2/projects/verified-agent-synthesis/utils/grammars/gsm.lark#L1) is now aimed at a single final answer expression/equation rather than many interleaved constrained windows.

## Important Files

| File | Stage | Role |
|------|-------|------|
| [generation/csd/GeneratedAgentTemplate.py](/home/advayth2/projects/verified-agent-synthesis/generation/csd/GeneratedAgentTemplate.py) | **Generation** | Template with dafny_spec contracts into which Qwen injects strategy bodies; defines `MyCSDStrategy` method signature and split output architecture |
| [generation/csd/VerifiedAgentSynthesis.py](/home/advayth2/projects/verified-agent-synthesis/generation/csd/VerifiedAgentSynthesis.py) | **Generation** | Contract library providing verified helpers (CSDHelpers, Delimiter, LM, Parser, dafny_spec decorator) used by template and all strategies |
| [generation/generator.py](/home/advayth2/projects/verified-agent-synthesis/generation/generator.py) | **Generation** | Qwen model loader and strategy generator; handles initial generation and error-based refinement prompts; extracts and validates strategy bodies |
| [generation/prompts.py](/home/advayth2/projects/verified-agent-synthesis/generation/prompts.py) | **Generation** | Prompt template system for Python-first strategy generation; builds initial task prompts and feedback repair prompts for verification/compilation/runtime/evaluation failures |
| [generation/rationale.py](/home/advayth2/projects/verified-agent-synthesis/generation/rationale.py) | **Generation** | Utilities for extracting rationale blocks from generated strategy bodies (handles both Python `#` and legacy Dafny `//` style markers) |
| [verification/transpiler/transpiler.py](/home/advayth2/projects/verified-agent-synthesis/verification/transpiler/transpiler.py) | **Verification** | Main Python-to-Dafny transpiler; parses AST, extracts dafny_spec annotations, lowers Python constructs to Dafny equivalents, generates workspace and dafny files |
| [verification/transpiler/support/result.py](/home/advayth2/projects/verified-agent-synthesis/verification/transpiler/support/result.py) | **Verification** | Rust-style Result/Ok/Err types for safe error handling in transpiler pipeline |
| [verification/transpiler/support/comment_stripping.py](/home/advayth2/projects/verified-agent-synthesis/verification/transpiler/support/comment_stripping.py) | **Verification** | AST utilities for removing comments/docstrings; used to clean Python before transpilation |
| [verification/transpiler/support/dynamic_type_resolution.py](/home/advayth2/projects/verified-agent-synthesis/verification/transpiler/support/dynamic_type_resolution.py) | **Verification** | Runtime type resolution for Dafny spec annotations (resolves string type paths to callables) |
| [verification/transpiler/support/mypy_type_checker.py](/home/advayth2/projects/verified-agent-synthesis/verification/transpiler/support/mypy_type_checker.py) | **Verification** | Mypy integration for type-checking Python sources before transpilation; builds and runs mypy stubs |
| [verification/verifier.py](/home/advayth2/projects/verified-agent-synthesis/verification/verifier.py) | **Verification** | Dafny verification wrapper; runs `dafny verify` on transpiled code, parses error output, generates error summaries for LLM refinement |
| [verification/compiler.py](/home/advayth2/projects/verified-agent-synthesis/verification/compiler.py) | **Verification** | Dafny to Python compiler wrapper; runs `dafny build --target:py`, manages output directories, parses compilation errors |
| [verification/dafny_runner.py](/home/advayth2/projects/verified-agent-synthesis/verification/dafny_runner.py) | **Verification** | Shared Dafny workspace setup; prepares temp directory with Python sources and transpiled Dafny, checks Dafny availability |
| [synthesis/feedback_loop.py](/home/advayth2/projects/verified-agent-synthesis/synthesis/feedback_loop.py) | **Synthesis** | Main orchestration loop: generate → verify → compile → run → evaluate → repair/refine until thresholds met; handles error-based repair and fallback strategies |
| [synthesis/runner.py](/home/advayth2/projects/verified-agent-synthesis/synthesis/runner.py) | **Synthesis** | Python runtime executor for compiled Dafny strategies; injects stubs (LM, Parser), captures output and errors, supports permissive and real parsing modes |
| [synthesis/presets.py](/home/advayth2/projects/verified-agent-synthesis/synthesis/presets.py) | **Synthesis** | Dataset and model preset definitions (gsm_symbolic, folio, pddl, sygus_slia) with default task descriptions and evaluation thresholds |
| [synthesis/cli/run_synthesis.py](/home/advayth2/projects/verified-agent-synthesis/synthesis/cli/run_synthesis.py) | **Synthesis** | Main CLI entrypoint for end-to-end synthesis with feedback loop; configurable iterations, model, dataset, thresholds |
| [synthesis/cli/generate_csd.py](/home/advayth2/projects/verified-agent-synthesis/synthesis/cli/generate_csd.py) | **Synthesis** | Preset-driven CSD generator wrapper; delegates to run_synthesis.py with dataset-specific defaults |
| [synthesis/cli/evaluate_existing_run.py](/home/advayth2/projects/verified-agent-synthesis/synthesis/cli/evaluate_existing_run.py) | **Synthesis** | Re-evaluator for existing synthesis runs; loads compiled modules and runs fresh evaluation without re-generating |
| [synthesis/cli/run_csd_with_grammar.py](/home/advayth2/projects/verified-agent-synthesis/synthesis/cli/run_csd_with_grammar.py) | **Synthesis** | Manual tester to run compiled CSD against arbitrary grammar without full evaluation |
| [evaluation/evaluator.py](/home/advayth2/projects/verified-agent-synthesis/evaluation/evaluator.py) | **Evaluation** | Base evaluator interface; computes accuracy, format-rate, syntax-rate, and sample failure cases for feedback to generator |
| [evaluation/common/environment.py](/home/advayth2/projects/verified-agent-synthesis/evaluation/common/environment.py) | **Evaluation** | Shared environment loader; loads compiled Dafny modules, initializes LM/parser/tokenizer, sets up grammar |
| [evaluation/common/parser_utils.py](/home/advayth2/projects/verified-agent-synthesis/evaluation/common/parser_utils.py) | **Evaluation** | Factory for Lark-based Dafny-compatible parsers from grammar files; implements VerifiedDecoderAgent.Parser interface |
| [evaluation/common/generation.py](/home/advayth2/projects/verified-agent-synthesis/evaluation/common/generation.py) | **Evaluation** | Shared CSD execution methods; dafny_seq_to_str converter, run_crane_csd executor, Dafny Seq/Python list bridging |
| [evaluation/common/run_artifacts.py](/home/advayth2/projects/verified-agent-synthesis/evaluation/common/run_artifacts.py) | **Evaluation** | Run directory utilities; resolve_run_dir handles "latest" indirection, find_compiled_module_dir locates GeneratedCSD.py |
| [evaluation/common/cli_utils.py](/home/advayth2/projects/verified-agent-synthesis/evaluation/common/cli_utils.py) | **Evaluation** | CLI helpers for evaluator commands; shared argument parsing and output formatting |
| [evaluation/common/model_utils.py](/home/advayth2/projects/verified-agent-synthesis/evaluation/common/model_utils.py) | **Evaluation** | LM loading and tokenizer utilities; handles HuggingFace model loading with quantization options |
| [evaluation/gsm_symbolic/dataset.py](/home/advayth2/projects/verified-agent-synthesis/evaluation/gsm_symbolic/dataset.py) | **Evaluation** | GSM-Symbolic dataset loader from HuggingFace (apple/GSM-Symbolic) with config variants (main, p1, p2) |
| [evaluation/gsm_symbolic/grammar.py](/home/advayth2/projects/verified-agent-synthesis/evaluation/gsm_symbolic/grammar.py) | **Evaluation** | Dynamic grammar builder for GSM-Symbolic; constructs variable-aware arithmetic expression grammar from problem context |
| [evaluation/gsm_symbolic/metrics.py](/home/advayth2/projects/verified-agent-synthesis/evaluation/gsm_symbolic/metrics.py) | **Evaluation** | GSM-Symbolic metrics; extracts numeric answers from constrained segments, compares to gold |
| [evaluation/gsm_symbolic/generation.py](/home/advayth2/projects/verified-agent-synthesis/evaluation/gsm_symbolic/generation.py) | **Evaluation** | GSM-specific CSD execution wrapper around common generation methods |
| [evaluation/folio/dataset.py](/home/advayth2/projects/verified-agent-synthesis/evaluation/folio/dataset.py) | **Evaluation** | FOLIO dataset loader from HuggingFace (yale-nlp/FOLIO); first-order logic reasoning task |
| [evaluation/folio/grammar.py](/home/advayth2/projects/verified-agent-synthesis/evaluation/folio/grammar.py) | **Evaluation** | FOLIO grammar for first-order logic; defines valid FOL formula syntax |
| [evaluation/folio/metrics.py](/home/advayth2/projects/verified-agent-synthesis/evaluation/folio/metrics.py) | **Evaluation** | FOLIO metrics; parses FOL conclusions, sends to Prover9, compares symbolic results |
| [evaluation/folio/generation.py](/home/advayth2/projects/verified-agent-synthesis/evaluation/folio/generation.py) | **Evaluation** | FOLIO-specific CSD execution and conclusion extraction |
| [evaluation/folio/fol_utils.py](/home/advayth2/projects/verified-agent-synthesis/evaluation/folio/fol_utils.py) | **Evaluation** | FOL formula utilities; segmentation, extraction, validation |
| [evaluation/folio/environment.py](/home/advayth2/projects/verified-agent-synthesis/evaluation/folio/environment.py) | **Evaluation** | FOLIO-specific environment setup |
| [evaluation/pddl/dataset.py](/home/advayth2/projects/verified-agent-synthesis/evaluation/pddl/dataset.py) | **Evaluation** | PDDL dataset loader; sequential planning task dataset |
| [evaluation/sygus_slia/dataset.py](/home/advayth2/projects/verified-agent-synthesis/evaluation/sygus_slia/dataset.py) | **Evaluation** | SyGuS-SLIA dataset loader; string manipulation synthesis task |
| [utils/parsers/lark_parser.py](/home/advayth2/projects/verified-agent-synthesis/utils/parsers/lark_parser.py) | **Utils** | Grammar-agnostic Lark-based parser for any grammar file; implements IsValidPrefix, IsCompletePrefix, ValidNextTokens |
| [utils/symbolic_solvers/fol_solver/fol_parser.py](/home/advayth2/projects/verified-agent-synthesis/utils/symbolic_solvers/fol_solver/fol_parser.py) | **Utils** | FOL formula parser using NLTK CFG; parses text FOL to tree, handles quantifiers and operators |
| [utils/symbolic_solvers/fol_solver/prover9_solver.py](/home/advayth2/projects/verified-agent-synthesis/utils/symbolic_solvers/fol_solver/prover9_solver.py) | **Utils** | Prover9 solver integration; runs Prover9 with FOL premises/conclusion, captures proof results |
| [run_synthesis.py](/home/advayth2/projects/verified-agent-synthesis/run_synthesis.py) | **Root** | Thin wrapper delegating to synthesis/cli/run_synthesis.py (compatibility entrypoint) |

## Module Map

### generation/
Responsible for converting natural language task descriptions into Python strategy code via Qwen-based generation and refinement.

**Flow**: LLM prompt + template → generate strategy body → extract and validate → inject into GeneratedAgentTemplate.py

**Key modules**:
- `generator.py`: Loads Qwen, generates bodies, applies structural validation (helper methods, no for-loops, contains ConstrainedAnswerStep)
- `prompts.py`: Task-specific prompts and feedback repair prompts (handles verification/compilation/runtime/evaluation errors)
- `rationale.py`: Extracts embedded rationale blocks from bodies for logging
- `csd/GeneratedAgentTemplate.py`: The synthesis hole (QWEN_INSERT_STRATEGY_BEGIN/END markers)
- `csd/VerifiedAgentSynthesis.py`: Verified contract library (the dafny_spec'd types, helpers, and predicates)

**Gotchas**:
- Model must write ONLY the method body, not the function signature or imports
- Rationale blocks are optional but help with reasoning trace
- Loop invariants must use Dafny syntax even in Python comments
- The body must contain at least one ConstrainedAnswerStep to satisfy the split-output contract

### verification/
Transpiles Python to Dafny, runs verification, and compiles back to Python.

**Flow**: Generated Python → transpiler (AST walk + dafny_spec extraction) → Dafny workspace → dafny verify → dafny build → compiled Python modules

**Key modules**:
- `transpiler/transpiler.py`: Main AST-to-Dafny lowering; extracts dafny_spec annotations, handles Python-specific constructs (is/is not, isinstance, token methods)
- `transpiler/support/`: Helper utilities for type resolution, comment stripping, mypy type-checking
- `verifier.py`: Runs dafny verify, parses structured error output for LLM feedback
- `compiler.py`: Runs dafny build --target:py, manages output directories
- `dafny_runner.py`: Shared workspace setup (copies VerifiedAgentSynthesis.py, transpiles, prepares dafny_workspace/)

**Gotchas**:
- Transpiler is strict about Python subset (no for-loops, lambdas, nested functions)
- Type annotations must be resolvable (uses mypy for static checking pre-transpile)
- Dafny errors often mention line/column in transpiled .dfy, not original Python
- Verification can timeout on large strategies; no artificial limit but Dafny internal timeout ~100s
- Compiled modules land in a subdirectory named after a hash; run_artifacts.py helps locate them

### evaluation/
Evaluates compiled CSDs on benchmark datasets and reports metrics for feedback.

**Flow**: Load compiled modules → initialize LM/parser/tokenizer → run CSD on examples → extract answers → score against gold → report accuracy/format-rate/syntax-rate

**Key modules**:
- `evaluator.py`: Base Evaluator class with meet_threshold() and feedback_summary() methods
- `common/environment.py`: Shared loader for compiled modules and environment setup
- `common/parser_utils.py`: Factory for Lark-based parsers matching VerifiedDecoderAgent.Parser interface
- `common/generation.py`: CSD executor (run_crane_csd) that bridges Dafny Seq and Python lists
- `common/run_artifacts.py`: Utilities for locating run directories and compiled modules
- Dataset-specific modules (gsm_symbolic/, folio/, pddl/, sygus_slia/): dataset loaders, grammar builders, metrics extractors

**Dataset specifics**:
- **gsm_symbolic**: Dynamic grammar from problem variables; extracts rightmost number from final <<...>>
- **folio**: FOL grammar; segments output on '$', sends conclusion to Prover9, compares proof result to gold label
- **pddl** and **sygus_slia**: Planned/extensible (stubs present)

**Gotchas**:
- Compiled modules use Dafny.Seq which is NOT iterable; must use index-based access or dafny_seq_to_str()
- Parser modes: "permissive" (all tokens valid) vs "real" (Lark grammar validation); synthesis uses permissive
- Grammar files (.lark) are per-dataset; gsm.lark is variable-aware, folio.lark is FOL formula syntax
- Evaluation sample size is separate from synthesis iteration counts; preset thresholds vary by dataset

### synthesis/
End-to-end orchestration and CLI entrypoints for the entire loop.

**Flow**: User CLI → parse args → initialize preset/dataset/model → loop: generate → verify → compile → run → evaluate → repair/refine

**Key modules**:
- `feedback_loop.py`: Main orchestration; implements generate→verify→compile→run→evaluate loop with error-based repair
- `runner.py`: Executes compiled strategies in Python (injects LM/Parser stubs, catches errors)
- `presets.py`: Dataset/model preset definitions (task description, thresholds, sample sizes)
- `cli/run_synthesis.py`: Main entry point for end-to-end synthesis
- `cli/generate_csd.py`: Wrapper for preset-driven generation
- `cli/evaluate_existing_run.py`: Re-evaluate existing synthesis output
- `cli/run_csd_with_grammar.py`: Manual tester for compiled CSD + grammar

**Gotchas**:
- Error repair strategies in feedback_loop.py are heuristic-based (regex replacements); if repair fails, loop moves to next iteration
- Runtime errors from stubs are caught and reported; strategy may still be evaluated if error is recoverable
- Preset thresholds vary; gsm_symbolic typically needs min_accuracy 0.3+, folio needs 0.5+
- Max iterations defaults to 5-10; iteration budget is the main loop termination criterion

### utils/
Stage-agnostic utilities for grammar parsing and symbolic solving.

**Modules**:
- `parsers/lark_parser.py`: Generic Lark-based parser for any .lark grammar; implements prefix validation
- `symbolic_solvers/fol_solver/`: FOL parsing (NLTK CFG) and Prover9 integration for FOLIO evaluation
- `grammars/`: Grammar files (json.lark, python.lark, sql.lark, gsm.lark, folio.lark, etc.)

**Gotchas**:
- Lark parser uses LALR for speed; some grammars may need tweaks for LALR compatibility
- Prover9 solver is external binary; must be installed; timeouts are configurable per call

## Verification Commands
Verify the helper library:

```bash
python verification/transpiler/transpiler.py generation/csd/VerifiedAgentSynthesis.py
```

Print only the transpiled Dafny:

```bash
python verification/transpiler/transpiler.py generation/csd/VerifiedAgentSynthesis.py --print-dafny
```

Verify the strategy template with dependencies auto-transpiled:

```bash
python verification/transpiler/transpiler.py generation/csd/GeneratedAgentTemplate.py
```

## Editing Guidance
- Do not casually change [VerifiedAgentSynthesis.py](/home/advayth2/projects/verified-agent-synthesis/generation/csd/VerifiedAgentSynthesis.py); it defines the contract surface for the whole synthesis stack.
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
python -m py_compile generation/*.py verification/*.py verification/transpiler/*.py verification/transpiler/support/*.py evaluation/common/*.py synthesis/*.py synthesis/cli/*.py generation/csd/*.py run_synthesis.py tests/synthesis/test_pipeline.py
python verification/transpiler/transpiler.py generation/csd/VerifiedAgentSynthesis.py
python verification/transpiler/transpiler.py generation/csd/GeneratedAgentTemplate.py
python -m pytest tests/synthesis/test_pipeline.py
```
