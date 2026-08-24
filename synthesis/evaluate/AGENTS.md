# AGENTS.md — `synthesis/evaluate/`

## Scope

**Evaluation orchestration**, runtime wiring, feedback loop hooks, and shared evaluation utilities—not benchmark-specific business logic.

## Rules

- Follow root **`AGENTS.md`**: benchmark-specific behavior belongs under **`benchmarks/<name>/`** (especially **`eval_logic.py`**).
- Preserve **DFA-mask / Syncode** integration assumptions documented in **`README.md`**; do not regress per-step validity to full-vocabulary parsing.
- **`evaluator.py`** should delegate; avoid growing monolithic if/else by benchmark.
- Keep CSD-authored prompt guidance capture generic: `AppendTaskGuidance`
  belongs in shared runtime/evaluation plumbing, not benchmark-specific scoring.
- Preserve registered prompt state across the evaluator, CSD runner, and fixed
  IterGen adapter. For Spider token-0 runs, rebuild the shared structured prompt
  before its final answer cue; do not force generic `completion_mode` or append
  guidance as an unregistered suffix. Missing state must raise a descriptive
  error.
- Attempt outcome ledgers must remain empirical: metrics, measured deltas,
  rationale-claim summaries, and observed failure-location counts. Include all
  small failure-location buckets rather than top-k truncating them.
- Aggregate failure-mode summaries in **`evaluator.py`** should list all
  detected mode buckets. Keep any cap on verbatim rollout examples separate
  from the aggregate counts.
- **`run_legacy_fixed_strategy.main`** calls **`_ensure_repo_cache_env`** so subprocess CRANE runs inherit **`HF_HOME`**, **`HF_CACHE`**, **`TRANSFORMERS_CACHE`**, **`SYNCODE_CACHE`**, and **`ITER_SYNCODE_CACHE`** under the repository **`cache/`** unless **`CSD_CACHE_ROOT`** (or those variables) are already set; vendored **`syncode/syncode/common.py`** and legacy forks walk up to the same root when imports happen outside that entrypoint.
- Edits inside gitignored **`legacy/{CRANE,itergen,cars}`** require tracked patches under **`environment/legacy_patches/`** per **`environment/legacy/AGENTS.md`** (prefer fixing **`run_legacy_fixed_strategy.py`** when that suffices).
- Keep the tracked IterGen compatibility adapter faithful to Transformers:
  greedy runs use an identity logits processor, while approved SMILES sampling
  at temperature 0.7 uses Transformers 5's own `_get_logits_processor` output.
  Do not hand-build or approximate the sampling processors.
- Config-allocated Qwen3.5 caches are truthy before they contain tokens. Check
  `get_seq_length()` before switching IterGen to latest-token-only input so the
  first forward pass receives the complete prompt.
- Keep CRANE result discovery scoped to the requested model, grammar mode, grammar, and chain-of-thought setting; never select the newest result across the whole dataset when baseline jobs run concurrently.
- Spider IterGen must use the checked-in upstream iterative column/table search,
  schema backtracking, 20-iteration limit, and greedy recurrence penalty 0.3;
  a single default-unit `forward()` is not the Spider protocol.
- Spider IterGen must render Qwen3.5 prompts through the model chat template
  with `enable_thinking=False`; other model and dataset prompt surfaces remain
  unchanged.
- Spider token-0 output validation belongs in the Spider benchmark module: accept
  one full-string parser-valid bare SQL statement, reject outer labels/wrappers
  and extra statements with stable reasons, and retain complete ordered
  raw/generated token evidence while removing only terminal IDs supplied by
  the generation adapter's exact stop set.
- SMILES CRANE samples at temperature 0.7, permits neutral reasoning before
  `<<`, constrains only the final SMILES inside `<< >>`, stops at `>>`, and
  scores only that inner span. Keep molecule examples, chemistry hints, and
  preferred structures out of the prompt.
- GCD SMILES evaluation samples at temperature 0.7 to avoid repeating one
  malformed output across the whole trial; GSM and Spider GCD evaluation stays
  greedy.
- GCD and IterGen SMILES adapters must honor the requested generation-token
  budget rather than silently capping a 400-token campaign at 256.

## See also

- **`README.md`** in this folder for component list and artifact paths.
- **`grammars/AGENTS.md`**, **`benchmarks/AGENTS.md`**, **`syncode/AGENTS.md`**.

## Temporary A-only win (2026-08-10)

Slice-B confirmation is disabled under `if False:` in `feedback_loop.py` (empty B on GSM/Spider train). Restore by flipping that branch. Search `TEMPORARY A-ONLY`.
