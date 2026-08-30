# Research Notes

## 2026-05-09 — Current State Assessment

### Project Summary

**metaDecode** is a system that synthesizes task-specific constrained decoding strategies (CSDs) as formally verified Dafny programs, compiles them to Python, and evaluates them against expert-designed baselines (CRANE, IterGen, CARS, GCD/SynCode) on three benchmarks: GSM-Symbolic (math), Spider (text-to-SQL), and SMILES (molecular generation). The thesis is that automated synthesis + formal verification can discover strategies that outperform hand-crafted ones per-task.

Target venue: ACL (paper uses `acl.sty`, review mode).

### What Exists and Works

1. **Full synthesis pipeline** — generate (GPT-5.4 default) → verify (Dafny) → compile (Dafny→Python) → evaluate (vLLM/Qwen) → refine. End-to-end functional.
2. **Verified reference strategies** — GCD, CRANE, IterGen, CARS formalized in Dafny against a shared primitive library (`VerifiedAgentSynthesis.dfy`). All verify cleanly.
3. **Three benchmark integrations** — GSM-Symbolic, Spider, SMILES with dataset loaders, grammars, and scoring logic.
4. **Legacy baseline runners** — external codebases for CRANE, IterGen, CARS wired into the evaluation harness for apples-to-apples comparison.
5. **Experiment orchestration** — `run_all_tests.py` covers the full strategy × model × benchmark matrix (4 models × 6 strategies × 3 benchmarks) plus ablations on step budget, synthesis iterations, synthesizer model, per-step token budget, beam refinement × helper selection policy, and adaptive helper masking.
6. **Ablation infrastructure** — beam/bandit sweep (`outputs/ablations/beam_bandit_20260509_034827.json`) shows the helper-selection-policy and beam-size ablation grid ran successfully on GSM-Symbolic with Qwen-7B. Results: beam=2 + utility policy achieved 80% accuracy (4/5), the best configuration in that sweep.
7. **Paper skeleton** — full LaTeX structure: abstract, intro, related work, problem formulation, approach, experiments, conclusion, appendix (Dafny spec, prompts, strategies). The experiments section has table shells for all planned results.
8. **DFA-mask constrained decoding** — Syncode vendor drop with DFAMaskStore for efficient per-step validity.

### What's Missing (Critical Path to Publication)

#### A. Experimental Results (the #1 blocker)

The experiment tables in `paper/experiments.tex` are entirely `\todo{--}`. No final numbers have been committed for the main results table. Specifically needed:

| Experiment | Status | What to Run |
|---|---|---|
| Main matrix (4 models × 6 strategies × 3 benchmarks) | **NOT DONE** | `run_all_tests.py --skip-ablations --eval-sample-size <N>` with publication sample size |
| Step-budget ablation (n=256,512,1024) | **NOT DONE** | Ablation A in `run_all_tests.py` |
| Synthesis-iterations ablation (K=3,5,10) | **NOT DONE** | Ablation B in `run_all_tests.py` |
| Synthesizer-model ablation (GPT-5.6 Sol, Opus 5, Gemini 3.7 Flash) | **NOT DONE** | Table 5 campaign; requires verified routes for all three |
| Per-step token budget (b=1,2,4) | **NOT DONE** | Ablation D in `run_all_tests.py` |
| Beam refinement × helper selection (B=1,2,4 × utility,bandit) | **NOT DONE** | Ablation E in `run_all_tests.py` |
| Adaptive helper masking (on/off) | **NOT DONE** | Ablation F in `run_all_tests.py` |
| Qualitative strategy examples | **NOT DONE** | Manual curation from successful synthesis runs |

**Models:** Qwen2.5-Coder-1.5B-Instruct, Qwen2.5-Coder-7B-Instruct, Qwen2.5-Coder-14B-Instruct, Llama-3.1-8B-Instruct (cross-family generalization).

**Estimated compute:** Full matrix is ~72 baseline evals + 12 metaDecode synthesis runs (main) plus ~82 ablation cells. With Qwen-14B needing tensor-parallel=2, expect 3–5 GPU-days on the 4×A100 node depending on sample size.

**Sample size decision:** Paper says `\todo{XX} examples per benchmark`. Need to commit to a number (50–100 is typical for Spider/GSM in CSD papers).

#### B. Paper Writing

- `\improve` macro is undefined (`\todo{+xx%}`) — need headline improvement number.
- Main results paragraph is a `\todo{Write analysis...}`.
- All three ablation analysis paragraphs are `\todo`.
- Qualitative analysis paragraph is `\todo`.
- Conclusion likely needs updating once numbers are in.
- `checklist.tex` (reproducibility checklist) probably needs a pass.

#### C. Reproducibility & Artifacts

- `outputs/baselines/` is empty (only a README). Baseline JSON artifacts haven't been exported yet.
- No committed final results anywhere in the repo — only dev/debug runs with sample_size=5.
- Need a clear artifact bundle (strategies, compiled code, evaluation logs) for review.

### What's In Good Shape

- Pipeline code is mature and well-documented (READMEs everywhere, AGENTS.md policies).
- The Dafny formalization is complete for all baseline strategies + the generated template.
- The experiment runner can produce all needed data without code changes.
- Paper framing/narrative (abstract, intro, approach) appears solid.
- Three synthesizer backends (GPT-5.6 Sol, Opus 5, Gemini 3.7 Flash) are wired into the Table 5 campaign.

### Preliminary Signal

From the dev ablation (beam/bandit sweep, n=5, GSM-Symbolic, Qwen-7B, 2 synthesis iterations):
- Best config (beam=2, utility): **80% accuracy, 80% syntax rate**
- Baseline CRANE on GSM-Symbolic (from git history, earlier runs): appears to be in the 40–60% range on small samples.
- This is encouraging but not publishable without proper sample sizes and all baselines on the same split.

---

## Action Plan: Path to Submission

### Phase 1: Lock Experimental Design (1 day)

- [ ] Decide final evaluation sample size per benchmark (recommend: GSM=100, Spider=50, SMILES=30 per class × 3 classes = 90).
- [ ] Decide on synthesis iteration count for final metaDecode strategies (K=10 per paper draft).
- [ ] Confirm all API keys available (OpenAI, Anthropic, Google) for synthesizer ablation.
- [ ] Verify that `run_all_tests.py --dry-run` produces the correct command matrix.

### Phase 2: Run Full Experiments (3–5 days compute)

- [ ] Run main baseline matrix: all 5 baselines × **4 models** × 3 benchmarks (72 baseline evals).
- [ ] Run metaDecode synthesis: 3 benchmarks × 4 models (12 synthesis runs at K=10).
- [ ] Run step-budget ablation (A): 5 strategies × 3 budgets × 2 benchmarks = 30 cells.
- [ ] Run synthesis-iteration ablation (B): K=3,5,10 × 2 benchmarks = 6 synthesis runs.
- [ ] Run synthesizer-model ablation (C): 3 models × 2 benchmarks = 6 synthesis runs.
- [ ] Run per-step token budget ablation (D): 5 strategies × 3 token budgets × 2 benchmarks = 30 cells.
- [ ] Run beam × helper-selection policy ablation (E): 3 beams × 2 policies × GSM = 6 synthesis runs.
- [ ] Run adaptive helper masking ablation (F): on/off × 2 benchmarks = 4 synthesis runs.
- [ ] Export all baseline artifacts to `outputs/baselines/`.

### Phase 3: Fill Paper (2–3 days writing)

- [ ] Populate Table 1 (main results — now 4 model blocks).
- [ ] Populate Table 2 (step-budget ablation).
- [ ] Populate Table 3 (synthesis-iterations ablation).
- [ ] Populate Table 4 (synthesizer-model ablation).
- [ ] Populate Table 5 (per-step token budget ablation).
- [ ] Populate Table 6 (beam refinement × helper selection).
- [ ] Populate Table 7 (adaptive helper masking).
- [ ] Write analysis paragraphs for each table.
- [ ] Compute `\improve` headline number.
- [ ] Curate 2–3 qualitative strategy examples for appendix.
- [ ] Update conclusion with concrete findings.
- [ ] Pass over checklist.tex.
- [ ] Camera-ready formatting pass (switch from `review` to `final`).

### Phase 4: Polish & Submit (1–2 days)

- [ ] Internal review among co-authors.
- [ ] Proofread, fix notation consistency.
- [ ] Prepare supplementary materials (code, artifacts).
- [ ] Submit.

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| metaDecode doesn't beat CRANE on all benchmarks | Weakens main claim | Paper already frames per-task adaptation; even matching CRANE on its home turf while winning on SMILES is a valid story |
| 14B model runs OOM or are too slow | Missing table cells | Use tensor_parallel=2, enforce_eager, reduce max_model_len to 3072 (already configured) |
| Synthesizer API rate limits | Delays synthesis runs | Stagger runs; only need ~15 synthesis calls total for ablation |
| Dafny verification flakiness on novel strategies | Failed synthesis iterations | K=10 gives multiple attempts; feedback loop already handles verification failures |
| Spider evaluation requires external DB setup | Missing one benchmark | SPIDER_DATA_DIR/SPIDER_DB_DIR env vars already supported; ensure data is mounted |

---

## Key Decisions Still Needed

1. **Evaluation sample size** — Paper credibility depends on this. CSD papers typically use full dev sets (GSM-Symbolic has 100 dev problems, Spider has 1034 dev, SMILES varies).
2. **Which split for Spider** — train vs. dev? Dev is standard for comparison but larger.
3. **Whether to include GCD as a separate baseline or fold into SynCode** — currently both are listed; paper treats GCD as the SynCode strategy.
4. **Step-token-budget default** — `--eval-step-token-budget` defaults vary; need consistency across all runs (main table uses b=1).

## Decisions Made (2026-05-09)

- **Models:** Added Llama-3.1-8B-Instruct for cross-family generalization. Total: 4 eval models.
- **Baselines:** 5 baselines (Unconstrained, GCD, CRANE, IterGen, CARS) are sufficient — cover all CSD paradigms.
- **Benchmarks:** GSM-Symbolic, Spider, SMILES are sufficient — three distinct constraint regimes, each baseline has a home advantage.
- **Additional ablations:** Per-step token budget, beam × helper policy, adaptive masking. These justify metaDecode-specific design choices.
- **Bandit helper set size (top-k):** Not ablated. Second-order hyperparameter; beam × policy ablation covers the relevant design axis.

---

## Notes on the Codebase

- The synthesis pipeline is single-threaded per run but `run_all_tests.py` can launch parallel jobs.
- Generated strategies from dev runs are in `outputs/generated/` — 44 run directories as of today, all from 2026-05-08/09 debugging.
- No paper/ TeX files reference any figure files yet — may need result plots (accuracy vs. budget curves, convergence plots).
- The `legacy/` directory contains full external codebases (CRANE, IterGen, CARS) which are used for baseline evaluation but not for verification.
