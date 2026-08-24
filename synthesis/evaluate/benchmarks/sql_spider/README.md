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

`SpiderPromptParts` is the shared prompt value used by evaluator and fixed IterGen paths. It preserves the raw user text, places CSD guidance before the final SQL cue, and renders Qwen3.5 with `apply_chat_template(..., enable_thinking=False)`. Qwen2.5 model names use the raw prompt. The legacy `SPIDER_TOKEN0_CONSTRAINED=0` switch remains available for the visible-delimiter path.

Token-0 Spider scoring accepts one bare SQL statement only. Labels, prose wrappers, delimiters, multiple statements, and trailing code are rejected; markers inside parser-supported SQL strings and line comments remain valid. Each Spider row records `output_contract_valid` and `output_rejection_reason` consistently. Generated-token evidence records the generated IDs and decoded text before and after removing only a terminal suffix declared by the tokenizer as special.

## Runtime Notes

- Spider evaluation is execution-grounded: generated SQL is executed and compared against gold-query behavior.
- The benchmark includes vendored evaluator dependencies under `syncode` support paths and benchmark utilities.
