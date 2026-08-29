# Evaluate Stage

The evaluate stage executes compiled strategies on benchmark tasks and returns structured metrics used by the synthesis feedback loop.

## Responsibilities

- Build runtime environment around compiled Dafny output.
- Load benchmark datasets and grammar resources.
- Run constrained decoding evaluation for each sample.
- Compute benchmark metrics and synthesis gate checks.
- Emit rich diagnostics used for strategy refinement.

## Main Components

- `evaluator.py`
  - Core sample evaluation loop and orchestration.
  - Delegates benchmark-specific prompt/answer/parser/scoring logic to `benchmarks/*/eval_logic.py`.
- `feedback_loop.py`
  - Generate/verify/compile/evaluate orchestration with iterative refinement.
- `runner.py`
  - Runtime helper paths used by local smoke/runtime routines.
- `parser_utils.py`
  - Compatibility wrapper re-exporting canonical parser utilities.
- `benchmarks/`
  - Dataset-specific modules (GSM-Symbolic, SQL Spider, SMILES).
  - `benchmarks/registry.py` selects the benchmark logic module.
  - `benchmarks/*/eval_logic.py` contains benchmark-specific evaluation behavior for easier unit testing.
- `grammars/`
  - Lark grammar definitions used by constrained decoding.
- `syncode/`
  - Vendored Syncode dependency for DFA mask store + parser internals.

## Runtime Constraints

- The parser path depends on Syncode DFA-mask caching for practical performance.
- Evaluation backends currently support runtime modes that provide token-level control (`huggingface`, `vllm`).
- Runtime LM wrappers support `AppendTaskGuidance`: the first non-empty CSD
  guidance block is rebuilt into the registered structured/chat prompt before
  the assistant-generation cue, later calls are ignored, and accepted guidance
  is surfaced in evaluation feedback. Missing prompt state fails closed with a
  descriptive error; guidance is never appended as an unregistered suffix.
- Spider prompt rendering is shared by evaluator and fixed IterGen paths; Qwen3.5
  disables thinking in its chat template, while Qwen2.5 receives the raw prompt.
- Output artifacts from this stage are saved under per-run `results/` folders in `outputs/generated/`.
- Evaluation refinement prompts include a compact attempt outcome ledger once
  multiple evaluated attempts exist. The ledger lists the best result, recent
  evaluated branches, rationale-claim summaries, measured deltas, and all
  observed failure-location counts for each listed attempt.
- Evaluation feedback lists every detected failure-mode bucket. Verbatim
  rollout examples remain capped separately because they carry full prompts and
  full model outputs; the aggregate mode summary should not top-k truncate.
- Baseline snapshots are JSON files in `outputs/baselines/` with:
  - `accuracy`, `syntax_rate`
  - `metrics` (counts, optional sums/means for `generation_seconds` / `num_tokens` / `constrained_work`, optional `run_wall_time_seconds` or evaluator totals)
  - `answers[]` with `question`, `generated_answer`, and optional `generation_seconds` / `num_tokens` / `constrained_work` per row
- Reevaluation exports keep `answers[]` backward-compatible and add a dedicated
  `reevaluation_sample_evidence[]` row for each evaluated example. Each row
  carries the contract outcome fields, prompt contract, safe helper/provenance
  metadata, and `generation_token_evidence` with raw IDs, raw decoded text,
  terminal IDs removed at the declared stop boundary, and the scored decoded
  text. Spider strategy fields (`strategy_output_relation`, `strategy_mutation`,
  and `strategy_removed_sampled_token_ids`) are copied from the same published
  generation evidence, including late generation-error samples.
- Fixed-strategy GSM baselines use the local CRANE GSM source rows so
  `unconstrained`, `gcd`, `crane`, `itergen`, and `cars` are compared on the
  same questions.
- The GCD adapter uses Syncode DFA-mask decoding but keeps GSM-Symbolic generation scoped to expression bodies: it starts after `<<`, wraps the generated body for scoring, caps expression length, finalizes the longest parseable expression prefix, and restricts GSM variables to numeric placeholders observed in the evaluation sample.
- GSM syntax checks use a numeric-only grammar when examples do not expose numeric symbolic variables; arbitrary identifiers such as `reasoning` must not pass syntax on instantiated GSM rows.
- The legacy CARS adapter runs through the same benchmark registry as the other fixed strategies and raises on failed runs so incomplete artifacts are not mistaken for valid zero-score baselines; it uses the same GSM grammar tightening as the GCD adapter (allowed variables or numeric-only, inferred from the evaluation batch) and expression-only prompts for gsm_symbolic, Spider, and SMILES like IterGen/GCD.
- The legacy IterGen adapter supports Transformers 5, where the private `_get_logits_warper` method was removed and beam counts may default to `None`. It restores the Transformers 4 greedy beam defaults, creates a config-aware cache only for models with linear-attention layers, and treats that config-allocated cache as empty until its sequence length is positive so the complete prompt reaches the first forward pass. It keeps an identity processor for greedy runs and uses Transformers 5's own processor list for approved SMILES sampling at temperature 0.7.
- Qwen3.5 IterGen cache readiness must use `get_seq_length()`, not cache object
  truthiness; a newly allocated 24/32-layer cache is truthy while still empty.
- IterGen recurrence penalty 0.3 is sign-aware: positive repeated-token logits
  are multiplied by 0.3 and negative logits are divided by 0.3.
- For Spider, the IterGen adapter follows the checked-in upstream protocol:
  incremental column/table generation, schema validation with bounded
  backtracking, 20 search iterations, and recurrence penalty 0.3 under greedy
  decoding. Qwen3.5 prompts are rendered through the model chat template with
  thinking disabled; other models and datasets keep their existing prompt
  surface.
- SMILES GCD, IterGen, and CRANE sample at temperature 0.7 so a failed first
  molecule does not force the same malformed molecule for the full trial. GSM
  and Spider remain greedy. SMILES GCD and IterGen honor the requested token
  budget. SMILES CRANE permits neutral reasoning before `<<`, constrains the
  final molecule inside `<< >>`, stops at `>>`, and scores only the inner span.
  Its prompt contains no molecule examples, chemistry hints, or preferred
  structures. Delimiter-based GSM and Spider behavior is unchanged.
- Concurrent CRANE-backed runs select result JSONL files only from the requested model, strategy mode, grammar, and chain-of-thought directory, then require the exact requested row count before exporting a baseline artifact.
- Generic baseline exports may contain empty generated strings, but campaign
  evidence rejects an exact 0/0 batch when every answer is blank or when all
  rows repeat one malformed answer. Evidence records nonblank and distinct
  output counts so degenerate runs cannot silently set thresholds.
- Campaign evidence rescoring uses trial-wide distinct-valid accuracy for
  SMILES: duplicate RDKit-valid, in-class, non-exemplar molecules count once.
- Legacy rows that do not report a syntax boolean are treated as syntax-invalid unless the adapter can annotate them with benchmark parser checks before export.
- CRANE-backed GSM rows do not carry `variable_types`; the exporter infers numeric symbolic identifiers from `gold_answer` before applying the GSM syntax parser.
