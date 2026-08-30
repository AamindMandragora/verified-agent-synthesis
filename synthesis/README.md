# Synthesis Package

Each first-party subdirectory under `synthesis/` includes **`AGENTS.md`** (automation/agent constraints) alongside **`README.md`** (human-oriented overview) where applicable. Vendored Syncode under `evaluate/syncode/` uses a single **`AGENTS.md`** at that root; do not blanket nested vendor trees.

The `synthesis/` package is the operational center of the repository.
It provides an end-to-end loop that produces candidate CSD strategies, proves correctness properties in Dafny, compiles those strategies to executable Python modules, evaluates them on target benchmarks, and feeds failures back into the next generation attempt.

## Top-Level Modules

- `run_synthesis.py`
  - Main CLI entry point for iterative synthesis: task/dataset, author + eval
    model, sample size and per-example limits, eval seed, SMILES knobs, and a
    few generation-only escape hatches (initial-strategy-file for pure
    re-evals, reasoning budget, max tokens).
  - Everything else — eval backend (vLLM), temperature (0.7), vLLM memory/
    context sizing, split file and side (always the canonical train split),
    delimiter requirement per dataset, helper-mask/bandit/beam settings,
    Claude transport — is a constant in `run_constants.py` or a `CSD_*`
    environment variable (2026-07-18 bucket-1 audit; see
    `planning/ws2-ws3-landed-audit.md`).
  - **Env overrides used by the cold queue:**
    - `CSD_VLLM_GPU_MEMORY_UTILIZATION` — per-job vLLM memory fraction
      (falls back to `VLLM_GPU_MEMORY_UTILIZATION` when unset).
    - `CSD_CONSTRAINED_TEMPERATURE` — constrained-span sampling temperature
      (read in `evaluate/benchmarks/common/model_utils.py`; SMILES cold
      jobs must set `0.7` or unique-valid collapses under argmax).
  - **GSM-Symbolic:** the only supported data source is local CRANE-style
    JSONs (vendored `legacy/CRANE/src/gsm_symbolic`). HuggingFace loading has
    been removed.
  - Generation backends: `openai` (default), `codex` (pinned Pi provider layer
    with ChatGPT/Codex OAuth and fixed `gpt-5.6-sol`), `claude` (Claude Code Max),
    `claude-bedrock`, `anthropic`, plus local HuggingFace/vLLM for targeted
    experiments (a large reasoning author is enforced for real synthesis).
  - **BYOD (bring your own credentials):** API keys are never CLI flags —
    `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, AWS credentials, etc. load from
    the environment / `.env`. The old `--generation-api-key` /
    `--generation-api-base-url` flags were removed.
  - Held-out ("test" side) re-evaluations no longer go through
    `run_synthesis`; use `synthesis/scripts/reevaluate_compiled_csd.py`.
- `run_constants.py`
  - The hard-coded run settings listed above, including
    `SPLIT_FILE_BY_DATASET` (canonical split manifest per dataset).
- `project_defaults.py`
  - Centralized defaults for local paths (Dafny binary, CRANE/Spider resources, etc.).

## Stage Subpackages

- `generate/`
  - Prompt construction and strategy body generation/refinement.
- `verify/`
  - Dafny verification and Dafny-to-Python compilation wrappers.
  - `verify/reference/` holds verified example strategies (CRANE / IterGen / CARS-style); see `verify/reference/README.md`.
- `evaluate/`
  - Runtime environment setup, benchmark evaluation, parser integration, and feedback-loop orchestration.
  - Captures CSD-authored `AppendTaskGuidance` prompt guidance in evaluation
    feedback so refinement can compare guidance choices against metrics.
  - The Spider token-0 prompt/output contract is shared across benchmark
    prompting, runtime guidance rebuilding, evaluator records, and fixed
    IterGen delivery; the legacy visible-span mode remains explicit.

## Canonical Stage Flow

The package is intentionally structured to mirror the pipeline order:

1. `generate`
2. `verify`
3. `evaluate`

Compile is implemented under `verify` because compilation is only valid after verification and shares Dafny-specific lifecycle code.

## Artifact Ownership

- Successful and failed run artifacts are written under `outputs/generated/`.
- Baseline experiment artifacts are written under `outputs/baselines/`.
- `synthesis/` itself contains implementation, not experiment outputs.

## Path Overrides

`run_synthesis.py` does not take `--output-dir`, `--baseline-output-dir`, or
`--grammars-dir` flags — those paths are settled constants in
`run_constants.py` (`OUTPUT_DIR`, `GRAMMARS_DIR`). Remaining overrides:

- `CSD_OUTPUT_DIR` — recovery-resume-only override (a resumed run keeps
  writing under its original directory); not a general output-location knob.
- `CSD_GRAMMARS_DIR` — honored by the evaluator when `GRAMMARS_DIR` is unset
  (otherwise falls back to the built-in `synthesis/evaluate/grammars/`).
- `run_all_tests.py` (the separate matrix runner) has its own
  `--generated-output-dir` / `CSD_OUTPUT_DIR` and `--baseline-output-dir` /
  `CSD_BASELINE_OUTPUT_DIR` flags.
- `--dafny-path` or `DAFNY_PATH`
- `DAFNY_EXTRA_PATH` (colon-separated PATH entries for Dafny subprocesses)
- `VERIFIED_AGENT_SYNTHESIS_DFY` or `DAFNY_PROOFS_DIR` (override proof include source)
- `CSD_SYNCODE_DIR` (vendored Syncode root)
- `SPIDER_DATA_DIR`, `SPIDER_DB_DIR`, `SPIDER_TABLES_JSON`
- `SPIDER_EVAL_DIR` / `SPIDER_EVAL_PY` (Spider evaluator location)
- `SMILES_DATA_DIR`, `SMILES_GRAMMAR_DIR`
- `CSD_JSON_GRAMMAR_PATH` (JSON grammar for smoke-test runner)
- `OPENAI_API_KEY` / `OPENAI_GENERATION_MODEL` for OpenAI (default `--generation-backend openai` in `run_synthesis`, model `gpt-5.4` unless overridden).
- `AWS_BEARER_TOKEN_BEDROCK` and Bedrock model ids when using `--generation-backend bedrock`.

## Design Philosophy

- Keep benchmark-specific behavior in benchmark packages, not in generic pipeline code.
- Keep parser/runtime performance guarantees explicit (Syncode DFA masks, caching, parser factories).
- Keep each synthesis attempt auditable through saved reports and run-local artifacts.
