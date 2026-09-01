# AGENTS.md

## Scope

These instructions apply to work in this repository.

## Project Goal

Synthesize Dafny CSD strategies that verify, compile, and evaluate successfully, with the objective of outperforming the CRANE baseline on the same evaluation setup.

## Critical Prompting Rule

Do not include strategy guidance in synthesis prompts or task descriptions.
Neutral API reference is tool/contract content, not strategy guidance.

Allowed prompt content:
- Task objective.
- Tool signatures, neutral API reference, and formal contracts
  (preconditions, postconditions, types, ranges, mechanics, cost/state effects,
  proof obligations).
- Neutral documentation for CSD-authored evaluator prompt guidance, including
  `AppendTaskGuidance` placement and first-call-wins semantics.
- Verified method-body examples as contract/format examples, not benchmark
  answers or task-specific strategy prescriptions.
- Empirical refinement context from the current synthesis run, including
  measured failures, search memory, helper usage, and evaluation history,
  without prescriptive strategy advice.

Disallowed prompt content:
- Recommendations about which tool to use.
- Preferred or forbidden strategy patterns.
- Baseline-comparison hints that imply structure.
- Benchmark-specific answer hints, dataset shortcuts, or evaluation leaks.
- Procedural "NOTE" hints about when or why to apply a tool.

## Key Paths

- Non-hidden project directories are intentionally limited to `synthesis/`, `environment/`, `cache/`, and `outputs/`.
- Fixed GSM-Symbolic / Spider eval subsets: `environment/benchmark_splits/` (proportional easy/medium/hard[/extra]; regenerate via `python -m synthesis.evaluate.benchmarks.write_fixed_benchmark_splits`).
- Legacy baseline codebases (CRANE / IterGen / CARS): clone with `bash environment/clone_legacy_csds.sh` into gitignored `legacy/*`; tracked pointer `legacy/README.md`; harness-vs-upstream notes `environment/legacy/DIFFERENCES.md`; **any edit under `legacy/{CRANE,itergen,cars}` must be captured as patches under `environment/legacy_patches/`** (see `environment/legacy/AGENTS.md`).
- Dafny binary: set `DAFNY_PATH` when needed; otherwise the runner uses repo-local `dafny/dafny` only if present, then falls back to `dafny` on `PATH` or `~/.dotnet/tools/dafny`.
- OpenAI API key: `synthesis/.env`
- Hugging Face checkpoints and SynCode mask/parser pickles: repository `cache/` (set `CSD_CACHE_ROOT` to relocate; legacy CRANE/IterGen/GCD paths resolve here when unset).

## Core Files

- `synthesis/verify/library/GeneratedCSD.dfy`
- `synthesis/verify/library/VerifiedAgentSynthesis.dfy` (member index: `synthesis/verify/library/README.md`)
- `synthesis/generate/generator.py`
- `synthesis/verify/verifier.py`
- `synthesis/verify/compiler.py`
- `synthesis/evaluate/feedback_loop.py`
- `synthesis/evaluate/evaluator.py`
- `synthesis/run_synthesis.py`

## Pipeline Run Modes

Use `python -m synthesis.run_synthesis` from the repo root. Prefer `CUDA_VISIBLE_DEVICES=2,3` unless intentionally using another allocation. By default, **generation** uses **OpenAI** (`OPENAI_API_KEY` and `OPENAI_GENERATION_MODEL` / `--generation-model`); **evaluation** still defaults to local vLLM with Qwen unless you pass other flags. Matrix model ablations must use direct hosted APIs and must not route through Bedrock.

For the paper Tables 5--8 `gpt5.6-sol` profile, the `codex` backend uses the
pinned Pi provider layer with ChatGPT/Codex OAuth. It sends the campaign system
instructions and one user message directly to `gpt-5.6-sol`, with no Pi agent
session, tools, prior conversation, or Codex CLI instructions. It must not use
`OPENAI_API_KEY` or `codex exec`.
The Tables 5--8 controller launches 8 one-GPU GSM-Symbolic runs and maps the
shared default Opus control into 11 paper cells. Its terminal evidence must keep
synthesis, held-out, and total wall times plus phase timestamps and any
available per-attempt evaluation times.

- Quick smoke run (fast sanity check, low sample count):
  `CUDA_VISIBLE_DEVICES=2,3 python -m synthesis.run_synthesis --task "Solve math word problems with constrained symbolic expressions." --dataset gsm_symbolic --min-accuracy 0.0 --min-syntax-rate 0.0 --max-iterations 1 --eval-sample-size 1 --eval-max-steps 256 --output-name smoke_gsm`
- Standard GSM-Symbolic synthesis run:
  `CUDA_VISIBLE_DEVICES=2,3 python -m synthesis.run_synthesis --task "Solve math word problems with constrained symbolic expressions." --dataset gsm_symbolic --min-accuracy 0.4 --min-syntax-rate 1.0 --max-iterations 5 --eval-sample-size 20 --eval-max-steps 900 --output-name gsm_main`
- Spider synthesis run (SQL), usually with higher per-step token budget:
  `CUDA_VISIBLE_DEVICES=2,3 python -m synthesis.run_synthesis --task "Generate executable SQL queries from natural language questions." --dataset spider --min-accuracy 0.6 --min-syntax-rate 0.95 --max-iterations 5 --eval-sample-size 20 --eval-max-steps 900 --eval-step-token-budget 8 --output-name spider_main`
- Spider run with explicit split manifest:
  `CUDA_VISIBLE_DEVICES=2,3 python -m synthesis.run_synthesis --task "Generate executable SQL queries from natural language questions." --dataset spider --spider-split-file <path/to/split.json> --spider-split-name train --min-accuracy 0.6 --min-syntax-rate 0.95 --max-iterations 5 --eval-sample-size 20 --output-name spider_split`
- SMILES synthesis run (all classes or class subset):
  `CUDA_VISIBLE_DEVICES=2,3 python -m synthesis.run_synthesis --task "Generate valid molecules in the requested class." --dataset smiles --smiles-classes acrylates,chain_extenders,isocyanates --smiles-samples-per-class 10 --min-accuracy 0.5 --min-syntax-rate 1.0 --max-iterations 5 --output-name smiles_main`
- Local generation with vLLM (override default hosted generation):
  `CUDA_VISIBLE_DEVICES=2,3 python -m synthesis.run_synthesis --task "Solve math word problems with constrained symbolic expressions." --dataset gsm_symbolic --generation-backend vllm --generation-model Qwen/Qwen2.5-Coder-7B-Instruct --eval-backend vllm --min-accuracy 0.4 --min-syntax-rate 1.0 --output-name vllm_run`
- Hosted generation defaults to **OpenAI** (`OPENAI_API_KEY`, model `gpt-5.4` or `OPENAI_GENERATION_MODEL`). **`gpt5.5`** in `run_all_tests.py` uses OpenAI with synthesis author reasoning effort `xhigh` by default (`CSD_OPENAI_REASONING_EFFORT` / `OPENAI_GENERATION_REASONING_EFFORT` override). **`opus4.7`** uses the **Anthropic** backend (`ANTHROPIC_API_KEY`, optional `ANTHROPIC_OPUS_MODEL`) with adaptive thinking, `xhigh` effort, and summarized thinking by default. **`gemini`** uses the direct **Gemini** API (`GEMINI_API_KEY`, optional `GEMINI_GENERATION_MODEL`, default `gemini-3-pro-preview`) with `CSD_GEMINI_THINKING_LEVEL=high` by default. The Claude Code backend uses the isolated Claude Code Max login and the fixed model `claude-opus-5`. Bedrock and Bedrock-backed **`gemini-pro`** profiles are rejected by the matrix runner.
- Full matrix runs cover `gsm, spider`, corresponding to GSM and SQL/Spider. The CARS molecular/SMILES benchmark remains available for explicit manual runs, but it is not part of the default matrix. Run matrix jobs with `--reuse-baselines` when cached baseline JSONs are acceptable; matrix Metadecode launches should keep `--accuracy-win-margin 0.03`, `--max-tokens 32768`, `--restart-after-stuck-iters 0`, `--helper-selection-policy bandit`, `--refinement-beam-size 2`, `--eval-max-seconds-per-example 90`, and `--eval-min-examples-before-threshold-stop 15` unless a specific ablation intentionally changes one of those knobs. The accuracy target should be a real margin over the best matching legacy CSD baseline; syntax remains a thresholded floor, not a paper win margin.
- Full repository test sweep:
  `python run_all_tests.py`
- `run_all_tests.py` activates `/apps/conda/advayth2/envs/advayth2` by default and verifies RDKit import before starting the matrix. Partners using a different prefix should `export VAS_CONDA_ENV=/path/to/env`; `VAS_RDKIT_CONDA_ENV` remains as a legacy alias. The launcher prepends `CONDA_PREFIX/lib` to `LD_LIBRARY_PATH` so SciPy/transformers wheels resolve `libstdc++` correctly; Syncode needs **`mxeval`** with bundled **`data/`** — run **`bash environment/install_mxeval_into_env.sh`** once per env (see **`environment/README.md`**).

## Evaluation Expectations

- Always compare synthesized strategy performance against a CRANE baseline on the same model/split/sample settings before claiming success.
- Maintain high syntax/format validity while improving accuracy.
- Fixed-strategy GSM baselines should use the local CRANE GSM rows across `unconstrained`, `gcd`, `crane`, `itergen`, and `cars` so comparisons are row-aligned.
- For fixed-strategy baseline JSONs, do not infer valid syntax from missing legacy metadata. Annotate rows with benchmark parser checks or treat missing syntax booleans as invalid.
- Full-campaign evidence must reject exact 0/0 batches that are all blank or
  repeat one malformed output, record nonblank/distinct-output diagnostics,
  and recompute SMILES accuracy from distinct valid molecules across the trial.
- Selective baseline repairs must write to a new campaign output root and
  overlay only explicitly named cells when rebuilding evidence; never replace
  the original campaign artifacts in place.
- Fixed-strategy SMILES GCD, IterGen, and CRANE use sampling at temperature
  0.7; GSM and Spider stay greedy. SMILES CRANE may reason neutrally before
  `<<`, constrains only the final molecule inside `<< >>`, and scores only the
  inner span. Do not add molecule examples or chemistry strategy guidance.
- Spider IterGen recurrence penalty 0.3 must lower both positive and negative
  repeated-token logits; never multiply a negative logit by 0.3.
- Qwen3.5 IterGen must check cached sequence length before latest-token-only
  input; config-allocated empty cache layers are truthy but contain no prompt.
- For CRANE-backed GSM rows that lack `variable_types`, infer numeric symbolic identifiers from `gold_answer` before syntax checking.
- Keep the GCD GSM-Symbolic adapter scoped to constrained expression bodies after `<<`; wrap those bodies for scoring, finalize the longest parseable expression prefix, and restrict identifiers to numeric placeholders from the evaluation sample.
- For instantiated GSM rows without symbolic numeric variables, use numeric-only syntax checks; do not let arbitrary identifiers satisfy GSM expression syntax.
- Keep benchmark-specific evaluation behavior in `synthesis/evaluate/benchmarks/*/eval_logic.py` and keep `synthesis/evaluate/evaluator.py` focused on orchestration/delegation.

## Performance Constraint

When touching parser validity logic, preserve DFA-mask-based validity checks (Syncode `DFAMaskStore`) for `ValidNextTokens` behavior. Do not introduce brute-force O(vocab) Lark parsing for per-step validity.

## Operational Defaults

- Prefer GPUs `2,3` for local runs unless intentionally using another allocation.
- For the approved focal five-strategy baseline collection, use
  `scripts/run_focal_collection_pool.py --campaign full-baseline-20260803`.
  Shared-GPU scheduling must add measured memory from other users to this
  controller's outstanding reservations until each child has allocated; using
  only the larger of those values can over-pack a GPU during model startup.
- Build the matching cold queue only with
  `scripts/runtime/build_full_baseline_cold_manifest.py`. It must bind all five
  raw baseline hashes, derive thresholds from integer counts, cap syntax at
  90%, and label the approved 95% perfect-baseline accuracy exception.
- Keep changes minimal and localized. Occasionally scan through the repo and cut fat, because bloat is the enemy of progress.
- Do not remove or alter formal contracts in Dafny files unless required by the task.
- Do not create, edit, delete, or commit files under `paper/` unless the user explicitly requests changes there (the paper tree is out of scope for routine agent work).
- When asked to modify files under `paper/`, do not run paper compilation checks; after making the requested edits, always automatically proceed with `git add`, `git commit`, and `git push` within the subdirectory to update Overleaf.
- When using adaptive helper masking or beam refinement, keep selection rules empirical/contract-based (measured metrics, verifier checks), not heuristic strategy advice in prompts.
- For bandit-style helper selection, keep exploration/exploitation policy in pipeline code/CLI knobs (e.g., UCB parameters), not in synthesis prompt prose.
- For live experiment monitoring, prefer `scripts/experiment_dashboard.py` over ad hoc repeated log polling; keep it dependency-free so it can run on focal with the standard Python environment.
- After the fixed queue reaches Spider-14B, post-14B metaDecode reruns must go through `scripts/results_finalization/validate_pre_rebar.py`, `scripts/results_finalization/scan_metadecode_rebar.py`, and `.context/run_post14b_rebar_queue.sh`. The scanner and runner must both require the validator's complete matrix-bound snapshot. Require complete active+archived row coverage, exact-count targets, a 90% syntax cap, and a fresh approval bound to the snapshot, frozen matrix, scanner audit, candidate manifest, exact cells, and synthesis-launcher hashes.
- A post-14B cell gets at most one new cold 40-iteration synthesis cycle. Never remove its claim directory or relaunch it after `started`; preserve interrupted/failed artifacts, require direct held-out evaluation for a win, and keep author credentials out of that evaluation.
- Always update the `README.md` and `AGENTS.md` local to the folder you made changes in, and the global `README.md` and `AGENTS.md` for large changes. Occasionally scan the repo at the end of a request to ensure they are up-to-date.
- Under `synthesis/`, update the nearest subdirectory **`README.md`** / **`AGENTS.md`** when behavior or conventions change (see `synthesis/README.md` for the layout); do not add documentation inside vendored `synthesis/evaluate/syncode/syncode/` except via the root `evaluate/syncode/AGENTS.md` policy unless upgrading the vendor drop.
- **Legacy upstream trees:** never leave manual edits only under gitignored **`legacy/CRANE`**, **`legacy/itergen`**, or **`legacy/cars`**. Add matching unified patches under **`environment/legacy_patches/<name>/`**, refresh **`environment/legacy/DIFFERENCES.md`** when behavior changes, and verify with **`bash environment/clone_legacy_csds.sh`** (see **`environment/legacy/AGENTS.md`**).
