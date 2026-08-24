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
