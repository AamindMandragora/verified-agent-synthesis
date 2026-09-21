# SMILES Benchmark

This module evaluates synthesized CSD strategies on constrained molecular-string generation tasks.

## Responsibilities

- Load class-specific SMILES tasks and prompts.
- Apply class-dependent grammar constraints.
- Evaluate generated candidates for syntax validity and class membership.
- Report paper-aligned metrics plus anti-degeneracy diagnostics.

## Key Files

- `dataset.py`: class/task loading and prompt assembly.
- `generation.py`: generation wrappers for evaluator integration.
- `environment.py`: runtime setup for compiled strategy execution.
- `metrics.py`: RDKit-based validity and membership metrics.
- `data/`: per-class assets (grammars, exemplars, reference sets).

## Constraint mode

Strategies decide their own constraint behaviour. The prompt appends `<< >>` delimiter instructions — CRANE can reason before emitting `<<SMILES>>`, while GCD constrains from token 1. The evaluator prefers extracting from `<< >>` when present and falls back to `clean_smiles_output` on the raw text otherwise.

## Evaluation Behavior

SMILES scoring treats syntax validity and class-membership quality as separate signals.
The synthesis loop can use these diagnostics to discourage degenerate strategies that exploit formatting without producing meaningful molecules.

Class membership uses the original CARS chemical-substructure patterns from
`experiments/smiles/evaluate_molecules.py:eval_membership`, pinned at CARS commit
`f46607f983d89e4ca4f40527cd231ab04adc6ed7`: `C=CC(=O)O*` for acrylates,
`CO`, `OC`, or `N` for chain extenders, and `[*]N=C=O` for isocyanates.
These are RDKit SMARTS queries, not text substrings. The shared
`metrics.target_class_membership` serves both metaDecode evaluation and the
fixed baseline adapters. Missing RDKit raises an error; an unknown class raises
`ValueError`; an invalid molecule has false membership.

This matches the original class predicate, not every step of the original CARS
evaluation program. Syntax checks, duplicate handling, prompt-exemplar exclusion,
and collection denominators remain separate and unchanged by this scorer update.
