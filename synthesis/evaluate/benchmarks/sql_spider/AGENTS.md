# AGENTS.md — `synthesis/evaluate/benchmarks/sql_spider/`

## Scope

**Spider** text-to-SQL evaluation: schema formatting, gold execution comparison, and SQL constrained decoding.

## Rules

- Respect path overrides (`SPIDER_*` env vars) documented in root **`README.md`**; keep DB and table metadata loading robust.
- Execution accuracy and SQL syntax validity belong here or in delegated helpers, not in **`evaluator.py`** as one-off branches.
- Keep `SpiderPromptParts` in `prompts.py` as the source of truth for token-0
  evaluator and fixed-IterGen delivery. Qwen3.5 uses one user turn with
  `enable_thinking=False`; Qwen2.5 receives the composed raw prompt.
- Register Spider prompt state before CSD guidance arrives. The first non-empty
  guidance block is rebuilt before the final `SQL:` cue; missing state fails
  closed with a descriptive error, and guidance is never appended after the
  answer cue.
- Token-0 Spider scoring accepts only one parser-valid bare SQL statement and
  records a stable rejection reason plus generated-token boundary evidence.
  Preserve `SPIDER_TOKEN0_CONSTRAINED=0` as the explicit legacy visible-span
  mode.

## See also

- **`README.md`** in this folder.
