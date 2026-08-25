# SQL Spider Benchmark

This module evaluates synthesized CSD strategies on text-to-SQL tasks using the Spider benchmark.

## Responsibilities

- Load Spider examples and schema context.
- Build schema-aware prompting context.
- Run constrained decoding for SQL generation.
- Score predictions with execution-based matching.

## Key Files

- `dataset.py`: dataset loading and schema/context utilities.
- `grammar.py`: SQL grammar helpers.
- `generation.py`: generation wrappers integrated with evaluator.
- `environment.py`: runtime setup for compiled strategy execution.
- `executor.py`: execution-accuracy scoring against SQLite databases.
- `metrics.py`: aggregate metrics and reporting helpers.

## Prompt and output contract

`SpiderPromptParts` is the shared immutable prompt value used by evaluator, CSD, and fixed IterGen paths. It preserves generated-only completion text, places CSD guidance before the final SQL cue, and renders Qwen3.5 as one user turn with `apply_chat_template(..., add_generation_prompt=True, enable_thinking=False)`. Qwen2.5 model names use the composed raw prompt. A Qwen3.5 renderer error is a harness error, not an accuracy sample. The legacy `SPIDER_TOKEN0_CONSTRAINED=0` switch remains available for the visible-delimiter path.

Token-0 Spider scoring accepts one parser-valid bare SQL statement only. Labels, prose wrappers, delimiters, multiple statements, and trailing code are rejected; markers inside parser-supported SQL strings and line comments remain valid. Each Spider row records `output_contract_valid` and `output_rejection_reason` consistently. Generated-token evidence records only the committed ordered generation across constrained and unconstrained chunks: speculative retries, rollback-discarded IDs, and tokens after the first unconstrained marker are excluded. It records raw IDs and decoded text, removes only terminal IDs supplied by the generation adapter's exact stop set, and fails visibly if the committed decode differs from the scored output. SQL extraction and execution comparison share the evaluator's per-example timer.

## Runtime Notes

- Spider evaluation is execution-grounded: generated SQL is executed and compared against gold-query behavior.
- The benchmark includes vendored evaluator dependencies under `syncode` support paths and benchmark utilities.

When a real CSD helper retries through `MaskToken` or `PenalizeTriedTokenAt`, finalization reconciles the recorded history to the actual scored prefix and preserves only declared terminal stop IDs. Callback-free strategy rollback is aligned at the next real generation entry over the concatenated current prefix, preserving exact sampled-ID occurrences without re-tokenizing.
The trace wrapper preserves the public static descriptors for `CSDHelpers.RollbackToCompletePrefix` and `CSDHelpers.RollbackToValidPrefix`, records only safe before/after lengths for those aliases, and never treats unit-only regeneration results as full-prefix rollback state.
Nested rollback helpers keep their returned prefixes private until the outermost rollback completes; only a later top-level LM helper or finalization consumes that state. The private depth and pending prefix are cleared for every run and are never serialized.
The explicit returned strategy sequence is scored as returned and recorded separately as `strategy_output_text`, `strategy_output_relation`, and `strategy_mutation`; sampled raw-ID evidence is never relabeled as strategy-authored text. The relation is `mixed` whenever sampled IDs were removed or the returned sequence contains authored content; when origin is not proven, the conservative relation is also `mixed`. `sampled_output` requires an independent origin marker, never decoded-text equality alone. An unreconcilable mismatch raises `SpiderEvidenceContractError` and aborts the Spider harness rather than creating an accuracy row.
Public helper traces are metadata-only: snapshot events expose size only, while generation, rollout, and rollback events expose counts, flags, lengths, and status. Unknown helper results use `result redacted`; `BackwardToSymbol` is length-only and does not publish a local prefix, while the outer `CraneGeneration` event owns its full returned prefix for pending alignment.
Parser validation uses a temporary lexical view only for the live parser: outer CR/LF becomes inline whitespace and supported `--` comment bodies are omitted, while string-literal contents and the original returned/scored SQL candidate remain unchanged.
