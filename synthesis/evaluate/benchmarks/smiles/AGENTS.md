# AGENTS.md — `synthesis/evaluate/benchmarks/smiles/`

## Scope

**SMILES** class-constrained molecular string benchmark: prompts, class-specific grammars, RDKit-backed checks.

## Rules

- Evaluation uses **one class per run** for grammar binding (see CLI `--smiles-classes`); keep **`eval_logic.py`** and **`dataset.py`** consistent with that contract.
- Grammar text and assets under **`data/`** must stay in sync; update **`data/AGENTS.md`** when adding classes or files.
- Class membership must use `metrics.target_class_membership`: the original CARS
  SMARTS patterns from commit `f46607f983d89e4ca4f40527cd231ab04adc6ed7`, not
  substring matching. Both metaDecode and fixed baselines use this shared scorer.
  Missing RDKit must raise an error rather than silently change the scoring rule.
  Keep syntax, duplicate handling, prompt-exemplar exclusion, and collection
  denominators separate from class membership; do not change them implicitly.

## See also

- **`README.md`** and **`data/README.md`**.
