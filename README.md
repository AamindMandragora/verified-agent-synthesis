# Verified Agent Synthesis

This project synthesizes constrained decoding strategies (CSDs) in a Python-first workflow:

1. **Generation**: an LLM writes a Python strategy body against a fixed CSD template.
2. **Verification**: that Python is transpiled to Dafny and verified there.
3. **Evaluation**: the verified compiled strategy is measured on real benchmark datasets.
4. **Synthesis loop**: generation, verification, compilation, runtime checks, and evaluation are repeated until thresholds are met or the budget is exhausted.

The source of truth is the Python code under `generation/`, not handwritten Dafny source files.

## Pipeline

```text
Python prompt + template
        |
        v
generation/
  LLM writes a strategy body into generation/csd/GeneratedAgentTemplate.py
        |
        v
verification/
  Python -> Dafny transpilation
  dafny verify
  dafny build --target:py
        |
        v
evaluation/
  run the compiled strategy on dataset examples
  compute accuracy / format / syntax metrics
        |
        v
synthesis/
  orchestration, repair loop, CLI entrypoints, shell presets
```

## Main Folders

```text
verified-agent-synthesis/
├── generation/        # Python-authored strategy generation assets
│   ├── generator.py
│   ├── prompts.py
│   ├── rationale.py
│   └── csd/
│       ├── VerifiedAgentSynthesis.py
│       ├── VerifiedAgentSynthesis.md
│       ├── GeneratedAgentTemplate.py
│       └── GeneratedAgentTemplate.md
│
├── verification/      # Python -> Dafny verification and compilation
│   ├── verifier.py
│   ├── compiler.py
│   ├── dafny_runner.py
│   └── transpiler/
│
├── evaluation/        # Dataset evaluation packages and shared eval helpers
│   ├── evaluator.py
│   ├── common/
│   ├── gsm_symbolic/
│   ├── folio/
│   ├── pddl/
│   └── sygus_slia/
│
├── utils/             # Stage-agnostic assets and helpers
│   ├── grammars/
│   ├── parsers/
│   └── symbolic_solvers/
│
├── synthesis/         # End-to-end orchestration and runnable entrypoints
│   ├── feedback_loop.py
│   ├── runner.py
│   ├── presets.py
│   ├── cli/
│   └── shell/
│
├── run_synthesis.py   # Thin compatibility wrapper for the main synthesis CLI
├── run_eval.sh        # Small convenience wrapper for GSM evaluation
├── outputs/           # Runtime artifacts only
├── tests/
├── docs/              # Excluded from code reorganization
└── dafny/             # Bundled Dafny toolchain files
```

## Core Ideas

### Generation is Python-first

The generator does **not** ask the model to write Dafny directly. It asks for a Python method body that fits the template in [generation/csd/GeneratedAgentTemplate.py](/home/advayth2/projects/verified-agent-synthesis/generation/csd/GeneratedAgentTemplate.py).

The shared contract library lives in [generation/csd/VerifiedAgentSynthesis.py](/home/advayth2/projects/verified-agent-synthesis/generation/csd/VerifiedAgentSynthesis.py).

### Verification happens through transpilation

The verification stage lowers Python CSD code through [verification/transpiler/transpiler.py](/home/advayth2/projects/verified-agent-synthesis/verification/transpiler/transpiler.py), then runs Dafny verification and Dafny-to-Python compilation through [verification/verifier.py](/home/advayth2/projects/verified-agent-synthesis/verification/verifier.py) and [verification/compiler.py](/home/advayth2/projects/verified-agent-synthesis/verification/compiler.py).

### Evaluation is dataset-driven

The compiled strategy is then evaluated against benchmark tasks:

- `gsm_symbolic`
- `folio`
- `pddl`
- `sygus_slia`

Shared evaluation helpers live under [evaluation/common](/home/advayth2/projects/verified-agent-synthesis/evaluation/common).

## Main Commands

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the end-to-end synthesis loop:

```bash
python run_synthesis.py \
  --task "Generate a GSM-Symbolic strategy" \
  --dataset gsm_symbolic \
  --min-accuracy 0.3 \
  --min-format-rate 0.5 \
  --min-syntax-rate 0.5
```

Use a preset-driven wrapper:

```bash
python synthesis/cli/generate_csd.py gsm_symbolic --model-preset qwen7b
```

Use shell shortcuts for specific dataset/model pairs:

```bash
bash synthesis/shell/gsm_symbolic_qwen3b.sh
bash synthesis/shell/gsm_symbolic_qwen7b.sh
bash synthesis/shell/folio_qwen3b.sh
bash synthesis/shell/pddl_qwen7b.sh
```

Evaluate a saved run directly:

```bash
python -m evaluation.gsm_symbolic.cli \
  --run-dir outputs/latest \
  --limit 10
```

Re-evaluate an existing synthesized run with the current evaluator:

```bash
python synthesis/cli/evaluate_existing_run.py \
  --run-dir outputs/latest \
  --dataset folio
```

Run a compiled strategy against a grammar manually:

```bash
python synthesis/cli/run_csd_with_grammar.py \
  --run-dir outputs/latest \
  --grammar utils/grammars/json.lark
```

## Outputs

Runtime artifacts are written under:

```text
outputs/
├── latest_run.txt
└── YYYYMMDD_HHMMSS_HASH/
```

Each run directory typically contains:

- the generated Python strategy
- the transpiled Dafny source
- the compiled Python module
- success or failure reports

## Notes

- `generation/csd/` is the core authoring surface for the CSD contract library and template.
- `docs/` is documentation/paper material and intentionally separate from the code pipeline.
