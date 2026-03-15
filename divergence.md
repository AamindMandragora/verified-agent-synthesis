# Branch divergence: `advayth/attempted_library_fixes` vs `master`

This document summarizes the differences between this branch and `master` in human-readable form.

---

## 1. Dafny verified core (`dafny/VerifiedAgentSynthesis.dfy`)

- **Cost replaced by step count:** The verified strategy no longer uses a `cost` field on `CSDHelpers`. Instead, `UnconstrainedStep` and `ConstrainedStep` take a `stepsLeft: nat` and return `stepsLeft' == stepsLeft - 1`. Termination is expressed as consuming one step per decoding step.
- **New parser predicate:** Parsers can implement `IsPermissive(prefix)`. It is intended to be true when every token is valid (e.g. free-form intro/outro in FOLIO). The default implementation returns `false` (strict grammar).
- **Delimiter constants:** `LeftDelimiter` and `RightDelimiter` are defined as `"<<"` and `">>"` in the Dafny module for reference; actual FOLIO evaluation uses `$` and `%` in the Python parser wrapper.

---

## 2. Shared evaluation layer (`evaluations/common/`)

- **New `environment.py`:** Centralizes run-dir resolution (including the `latest` shortcut via `latest_run.txt`), loading of compiled CSD modules (`_dafny`, `VerifiedDecoderAgent`, `GeneratedCSD`), and `setup_dafny_environment`. FOLIO and GSM symbolic both use this instead of duplicating logic.
- **New `generation.py`:** Shared helpers for running the compiled CSD strategy: building the full prefix (prompt + generated), calling the Dafny strategy, and converting the result back to text. Defines `dafny_seq_to_str` for robust Dafny-Seq-to-string conversion (used by the parser and generation). Handles edge cases where `len(csd_output)` is wrong by iterating and using index/len fallbacks. Adds `[GEN]` logging (output length, repr, RETURN).
- `**parser_utils.py`:**
  - `create_lark_dafny_parser`: Uses shared `dafny_seq_to_str` from `generation`; `IsCompletePrefix` treats empty/None prefix as incomplete; new `IsPermissive` on the parser class (returns `False`).
  - **New `create_folio_wrapper_parser`:** A wrapper parser for FOLIO that treats the output as: free-form text, then `$`, then FOL formula (Prover9 grammar), then `%`, then optional text. Single-character delimiters `$` and `%` are used (not `<<` / `>>`). The wrapper delegates prefix validity and valid-next-tokens to the underlying FOL parser for the segment between `$` and `%`; intro before `$` and text after `%` are permissive. Valid-next filtering:
    - Inside the formula section: whitespace-only tokens (newline, tab, etc.) are filtered out so the model cannot fill the formula with blank lines.
    - When the formula is complete, `%` is included in the valid set so the model can close and stop (avoids overgeneration).
    - When the filtered valid set would be empty but the formula is complete, only `%` is offered (avoids picking a garbage token like `!`).
- `**model_utils.py`:**
  - **FOL keyword tokens:** `FOL_KEYWORD_TOKENS` lists `{forall}`, `{exists}`, `{and}`, `{or}`, `{not}`, `{implies}`, `{iff}`, `{xor}`. When `add_fol_keyword_tokens=True`, the tokenizer gets these added and the model’s embedding layer is resized; their IDs are appended to the token list passed to Dafny so that formulas can be extended in one token (e.g. `Pred(x) {and}`).
  - **First-token masking:** The LM forbids as the first generated token: delimiter-like strings (`<<`, `<`,  `<<`, `<<`, `$`), empty string, EOS, and PAD so the model outputs real text before the formula and does not immediately stop (fixing empty output).
  - **Non-finite logits:** Logits set to `-inf` (e.g. for forbidden tokens) are clamped to a large negative value (e.g. `-1e30`) before converting to Dafny `BigRational`, which cannot represent ±inf (fixes OverflowError in the verified path).

---

## 3. FOLIO evaluation (`evaluations/folio/`)

- **Environment:** `environment.py` is reduced to a thin wrapper that calls `evaluations.common.environment` (`resolve_run_dir`, `load_compiled_modules`, `setup_dafny_environment`, `verify_critical_tokens`). FOLIO uses `start_rule="start"` for the FOL grammar and passes `add_fol_keyword_tokens=True` when creating the LM.
- **Generation:** Uses the shared `generation.run_crane_csd`-style flow and the FOLIO wrapper parser (`create_folio_wrapper_parser`) with `$` / `%` and the tokenizer’s token list. No separate CRANE-specific generation loop in this branch; the single CSD strategy runs over the full output.
- **Dataset / fol_utils:** `dataset.py` and `fol_utils.py` updated for the new pipeline; `fol_utils` comments reference `$` / `%` as delimiters.
- **CLI:** `cli.py` updated to use the shared environment and generation entrypoints.

---

## 4. GSM symbolic evaluation (`evaluations/gsm_symbolic/`)

- **Environment and generation:** Similarly refactored to use `evaluations.common.environment` and the shared generation helpers so GSM and FOLIO share the same loading and execution path for the compiled CSD.
- **CLI:** Adjusted for the common setup.

---

## 5. Synthesis and evaluator (`synthesis/`)

- **Evaluator:**
  - FOLIO prompt and extraction use `$` and `%` (regex/formats updated from `<<` / `>>`).
  - When creating the LM for FOLIO, `add_fol_keyword_tokens=True` is passed so that FOL keywords are single tokens.
  - New helpers for debugging: `get_detailed_samples()` and `print_outputs_vs_expected()` to show expected vs actual, FOLIO segments sent to Prover9, and conclusion/answer.
- **Generator / feedback_loop / prompts / runner / verifier / compiler / dafny_runner:** Updated as needed for the new environment and strategy interface (e.g. stepsLeft instead of cost, and the shared run flow).

---

## 6. Dafny externs and generated code

- `**dafny_externs/extern_functions.py`:** Small additions for the new parser/LM behavior (e.g. supporting the wrapper and token lists).
- `**dafny/GeneratedCSD.dfy`:** Minor updates to align with the new strategy signature (stepsLeft, etc.).

---

## 7. Symbolic solvers and FOL

- **FOL solver / Prover9:** `symbolic_solvers/fol_solver/` (e.g. `fol_parser.py`, `fol_prover9_parser.py`, `Formula.py`, `prover9_solver.py`) may have small changes for compatibility with the FOL segment extracted between `$` and `%` and for robustness.
- **CSP / Pyke / Z3:** `symbolic_solvers/csp_solver`, `pyke_solver`, `z3_solver` have minor changes (imports, init, or tests) to work with the rest of the refactor.

---

## 8. Removed or replaced files

- `**comprehensive_eval.py`:** Removed (large script; evaluation is done via the synthesis evaluator and scripts such as `eval_csd_with_synthesis_prompt.py`).
- `**check_tokens.py`:** Removed; its role is covered by the shared environment and FOL token handling; `.gitignore` now ignores `check_tokens.py`.
- `**eval_output.log`:** Removed from the repo (log artifact).

---

## 9. New files and scripts

- `**scripts/eval_csd_with_synthesis_prompt.py`:** Evaluates a compiled CSD run using the synthesis evaluator (current prompt, first-token masking, optional sample size and 4-bit loading).
- `**tests/test_fol_grammar.py`:** Grammar tests for FOL: valid prefix and complete, valid next tokens (including after `Alkane(mixture)` and inside predicates e.g. `P1(be`), tokenizer behavior with and without FOL keyword tokens, and a check that constant continuation (e.g. toward `beijing`) is allowed.

---

## 10. Documentation and config

- **README.md:** Wording updated to describe strategies as loops over `UnconstrainedStep` / `ConstrainedStep` (and optional `RollbackToValidPrefix`) instead of a single `CraneGeneration`; references to cost and hybrid strategy updated to stepsLeft and the current pipeline.
- **.gitignore:** Expanded: `.pytest_cache/`, `__pycache__/`, `*.pyc`, `.DS_Store`, `*.log`, `test_output.txt`, `temp_test_strategy.dfy`, `check_tokens.py`; `dafny-lang/` and `dafny/obj/`* retained.

---

## 11. Other diffs (brief)

- **docs/papers:** Many renames (e.g. `IterGen paper` → `IterGen`, `Syncode paper` → `Syncode`); content is largely unchanged.
- **datasets:** `datasets/ml-gsm-symbolic` appears as a submodule or path change.
- **outputs:** New run directories and artifacts under `outputs/generated-csd/runs/`; `latest_run.txt` and `latest` symlink may point to different runs. These are not considered part of the “library” divergence.
- **requirements.txt:** Possible dependency tweaks for the evaluation environment.

---

## Summary table


| Area             | Master (baseline)                    | This branch                                   |
| ---------------- | ------------------------------------ | --------------------------------------------- |
| Step accounting  | `CSDHelpers.cost` incremented        | `stepsLeft` passed in/out of steps            |
| FOLIO delimiters | `<<` / `>>` (or similar)             | `$` / `%`                                     |
| FOL connectives  | Multi-token (e.g. `{`, `and`, `}`)   | Single tokens via `add_fol_keyword_tokens`    |
| Empty output     | Possible (EOS/delimiter first token) | First-token forbid set + EOS/PAD masked       |
| Logits to Dafny  | Raw (could be ±inf)                  | Clamped to finite value                       |
| Parser for FOLIO | Likely single grammar                | Wrapper: intro / `$` FOL `%` / outro          |
| Formula complete | Valid next could omit `%`            | `%` always in valid set when formula complete |
| Env / generation | Per-dataset (folio, gsm_symbolic)    | Shared `evaluations.common`                   |
| Big eval script  | `comprehensive_eval.py`              | Removed; use synthesis evaluator + scripts    |
| Grammar tests    | —                                    | `tests/test_fol_grammar.py`                   |


This file was generated to capture the main divergences between `advayth/attempted_library_fixes` and `master` for code review and merge planning.