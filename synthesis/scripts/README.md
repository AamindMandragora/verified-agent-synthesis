# `synthesis/scripts/`

Optional **maintenance and ablation** scripts that drive `python -m synthesis.run_synthesis` (or inspect `outputs/generated/`) from the repository root.

They are not imported by the core package at runtime; run them explicitly with `PYTHONPATH` set to the repo root when documented in each script's docstring.

## Contents

- **`ablation_beam_bandit.py`** — Grid search over refinement beam size and helper-selection policy.
- **`reevaluate_compiled_csd.py`** — Re-run evaluation on an already-compiled GeneratedCSD.py.
- **`collect_paper_results.py`** — Collect baseline and synthesis results into paper-ready LaTeX table fragments. Reads `outputs/baselines/` and `outputs/generated/`, emits main results + ablation tables. Use **`--paper-main-table`** / **`--paper-bold-best`** to print Table~1 rows for `paper/experiments.tex`. Pass **`--git-tracked-only`** to include only metrics whose source `outputs/**/*.json` paths are tracked by git (cells without such JSON emit `\todo{--}`).

Scripts are self-contained CLIs. See each file's module docstring for arguments and examples.

## Reevaluation evidence

The reevaluate_compiled_csd script keeps the historical minimal answers list
for compatibility and adds a separate reevaluation_sample_evidence list. Each
row records its evaluated/source index, correctness and denominator flags,
Spider contract result and rejection reason, timeout/error status, terminal
token removal and full generation-token evidence, strategy mutation
provenance, safe helper-trace tags, failure location, and the safe prompt
delivery contract. Prompt records contain only renderer mode, booleans, and
lengths; they never contain prompt or schema bodies. The reevaluation
provenance records the requested sample offset and the exact source-index
order returned by Evaluator, so smoke and pilot slices cannot claim a
different list. Sharded reevaluation applies the same contract: it merges
answers and evidence in shard order, requires each shard to return only a prefix
of its planned canonical slice, assigns global evaluated indices, combines only
the source indices actually returned before an early stop, canonicalizes the split
provenance, keeps immutable model/run identity consistent across shards, records
planned sample size separately from evaluated count, and fails closed on
answer/evidence/source misalignment. Generic shard outputs with no child
provenance retain answers, evidence, and split data but omit the provenance
block; mixed provenance presence is rejected.

## See also

- **`AGENTS.md`** in this folder for agent constraints when adding or editing scripts.
