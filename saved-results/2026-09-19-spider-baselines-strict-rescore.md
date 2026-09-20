# Spider baselines: strict-scorer rescore (2026-09-19)

What this is: the saved Spider baseline generations from
`~/modal-baseline-rerun-20260914/runs/baseline-rerun-20260918-focal/lanes/spider-*-q35-*-sNN/output.json`
rescored with the strict output-contract scorer on branch `span-contract-fix`
(worktree `span-contract-fix-20260917`), instead of the scorer that produced
the stored numbers. No GPU used, no files under `~/csd-generation` touched,
nothing in the lanes directory touched.

Each cell = 4 shards x 75 examples = 300 examples (the held-out side of
`spider_dev_proportional_300x300_seed334.json`). Verified: each shard's index
list matches `test_indices[shard*75:(shard+1)*75]` of that split file exactly,
and every answer's stored `question` text matches the example at that
position, so shard-to-example alignment is confirmed, not assumed.

## Results

| Method | Model | n | Stored acc | Stored syntax | Strict acc | Strict syntax |
|---|---|---|---|---|---|---|
| unconstrained | 2B | 300 | 22.0% | 64.7% | 17.0% | 38.7% |
| unconstrained | 4B | 300 | 52.7% | 99.0% | 0.0% | 0.3% |
| crane | 2B | 300 | 20.7% | 61.0% | 15.0% | 37.3% |
| crane | 4B | 300 | 52.0% | 99.3% | 0.0% | 0.0% |
| gcd | 2B | 300 | 15.3% | 53.0% | 15.3% | 51.0% |
| gcd | 4B | 300 | 28.3% | 100.0% | 28.3% | 96.3% |
| cars | 2B | 300 | 19.0% | 100.0% | 19.0% | 99.3% |
| cars | 4B | 300 | 42.3% | 100.0% | 42.3% | 99.7% |
| itergen | 2B | — | — | — | — | — |
| itergen | 4B | — | — | — | — | — |

"Stored" = mean of the four shards' saved `accuracy`/`syntax_rate` fields
(equal-weighted since each shard is 75 examples). "Strict" = recomputed over
all 300 examples with `eval_logic.extract_actual` / `is_correct` from the
worktree.

**itergen is excluded, not zero.** Spider itergen lanes are incomplete:
- 2B: only shard s01 has `output.json`. s00 and s02 have only `stdout.log`
  (still running or crashed, no result yet); s03's directory doesn't exist at
  all.
- 4B: no shard has `output.json` — all four lane directories don't exist yet
  under this run.

Since a full 300-example cell needs all 4 shards, both itergen cells were
skipped entirely rather than scored on a partial/misaligned subset.

## Why unconstrained-4B and crane-4B collapse to ~0%

Verified, not guessed: pulled the raw `generated_answer` strings for these
lanes. Every example is of the form:

```
<think>\n\n</think>\n\nSQL: SELECT COUNT(*) FROM singer
```

The strict output contract in `output_contract.py` treats a `SQL:` (or
`question:`, `db_id:`, etc.) label anywhere outside a quoted string as a
"prompt/wrapper" artifact and rejects the whole output —
`_LABEL_RE = r"(?i)\b(?:sql|db_id|db_info|question|assistant|user|system)\s*:"`.
The stored scorer apparently pulled the SQL out from after that label; the
strict scorer requires the model's answer to be bare SQL with no label at
all, so it rejects on `output_rejection_reason == "prompt_or_wrapper"` for
essentially every example (298/300 unconstrained-4B, 300/300 crane-4B). This
is a real behavior difference between the two scorers, not a scoring bug.

2B unconstrained/crane don't collapse the same way because those two models
frequently exhaust their token budget while still inside `<think>...</think>`
(never emit an answer), which the strict scorer counts as `"empty"` rather
than `"prompt_or_wrapper"` — a different failure mode, same ballpark score
drop.

## Top 3 strict rejection reasons per cell (count out of 300)

| Method | Model | #1 | #2 | #3 |
|---|---|---|---|---|
| unconstrained | 2B | empty (110) | prompt_or_wrapper (67) | invalid_or_non_bare_sql (7) |
| unconstrained | 4B | prompt_or_wrapper (298) | invalid_or_non_bare_sql (1) | — |
| crane | 2B | empty (118) | prompt_or_wrapper (70) | — |
| crane | 4B | prompt_or_wrapper (300) | — | — |
| gcd | 2B | empty (139) | invalid_or_non_bare_sql (8) | — |
| gcd | 4B | invalid_or_non_bare_sql (11) | — | — |
| cars | 2B | invalid_or_non_bare_sql (2) | — | — |
| cars | 4B | invalid_or_non_bare_sql (1) | — | — |

(A rejection reason is only recorded when the strict scorer actually rejects;
cells with fewer than 3 distinct reasons have fewer than 3 rows above.)

## Reproduce

```
ssh focal 'bash -l -s' <<'EOF'
conda activate csd
cd ~/csd-generation-worktrees/span-contract-fix-20260917
python scripts/analysis/spider_rescore_strict.py
EOF
```

Writes `scripts/analysis/spider_rescore_strict_results.json` (not committed)
next to the script (also not committed — per task instructions). Script:
`~/csd-generation-worktrees/span-contract-fix-20260917/scripts/analysis/spider_rescore_strict.py`.

## What I verified vs. assumed

Verified:
- Shard-to-split-index alignment (exact match against the parent split file).
- Question-text match between every stored answer and its paired example
  (script asserts this; a mismatch would have aborted that cell with
  `status: alignment_failed` — none did).
- `generated_answer` is already the post-`completion_for_scoring` text (read
  `run_legacy_fixed_strategy.py`), so no further prompt-stripping was needed
  before handing it to `extract_actual`.
- The `SQL:` label collapse for 4B unconstrained/crane, by reading raw
  `generated_answer` strings and the `_LABEL_RE` regex in
  `output_contract.py`.
- All 32 non-itergen lanes (4 methods x 2 models x 4 shards) have
  `output.json`; itergen has none for Spider.

Assumed / not independently re-verified:
- That `stdout.log`-only itergen lanes (2B s00/s02) are still running or
  crashed rather than something else — I only checked directory contents
  (no `output.json`/`result.json` yet), and did not open `stdout.log` or
  touch these lanes further, since they may be actively written by the job
  queue.
