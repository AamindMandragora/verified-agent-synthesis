# CSD Generation: Constrained Decoding Strategy Synthesis Pipeline

A synthesis pipeline for generating **Constrained Decoding Strategies (CSD)** using LLMs (Qwen) with formal verification via Dafny. The pipeline automatically generates, verifies, compiles, and tests constrained decoding strategies that guarantee valid output from language models according to specified grammars (JSON, Math, FOL, etc.).

## Overview

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  1. Generate    │────▶│   2. Verify     │────▶│   3. Compile    │────▶│    4. Test      │
│  (Qwen LLM)     │     │   (Dafny)       │     │  (Dafny → Py)   │     │   (Runtime)     │
└─────────────────┘     └─────────────────┘     └─────────────────┘     └─────────────────┘
        │                       │                       │                       │
        │                       │                       │                       │
        ▼                       ▼                       ▼                       ▼
   Dafny Strategy         Proof Checked           Python Module          Validated Output
        │                       │                       │                       │
        └───────────────────────┴───────────────────────┴───────────────────────┘
                                        │
                                        ▼
                              Feedback Loop (on failure)
```

## Quick Start

### Prerequisites

1. **Python 3.10+** with dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. **Dafny 4.x** (for verification and compilation):
   ```bash
   # macOS
   brew install dafny

   # Ubuntu/Debian
   # Download from: https://github.com/dafny-lang/dafny/releases

   # Verify installation
   dafny --version
   ```

3. **GPU (recommended)** for running Qwen models efficiently, or use a smaller model for CPU.

### Basic Usage

```bash
# Generate a CSD strategy for JSON output
python run_synthesis.py --task "Generate a strategy for JSON output"

# With custom max iterations
python run_synthesis.py --task "Generate a CRANE-style strategy" --max-iterations 10

# Use a smaller model for faster testing (CPU-friendly)
python run_synthesis.py --task "Generate a simple retry strategy" \
    --model Qwen/Qwen2.5-Coder-3B-Instruct

# Specify output name
python run_synthesis.py --task "Create a hybrid JSON strategy" --output-name my_strategy
```

---

## Main Entry Point: `run_synthesis.py`

This is the CLI entry point for the synthesis pipeline.

### Command Line Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--task` | `-t` | (required) | Task description for strategy generation |
| `--max-iterations` | `-n` | 5 | Maximum refinement iterations |
| `--model` | `-m` | `Qwen/Qwen2.5-Coder-7B-Instruct` | HuggingFace model name |
| `--output-name` | `-o` | `generated_csd` | Name for the output module |
| `--output-dir` | | `outputs/generated-csd/` | Base output directory |
| `--dafny-path` | | `dafny` | Path to Dafny executable |
| `--temperature` | | 0.7 | Sampling temperature for Qwen |
| `--max-tokens` | | 256 | Maximum tokens to generate per attempt |
| `--device` | | `auto` | Device for inference: `cuda`, `mps`, `cpu`, `auto` |
| `--verify-only` | | | Only verify existing `GeneratedCSD.dfy` |
| `--compile-only` | | | Verify and compile without generating |
| `--no-save-reports` | | | Don't save failure/success reports |

### Examples

```bash
# Full synthesis with JSON task
python run_synthesis.py --task "Generate a constrained decoding strategy that ensures valid JSON output using hybrid generation with occasional unconstrained steps"

# Verify an existing Dafny file
python run_synthesis.py --verify-only

# Verify and compile an existing file
python run_synthesis.py --compile-only --output-name my_strategy
```

---

## Project Structure

```
focal-lab/
├── run_synthesis.py          # Main CLI entry point for CSD strategy synthesis
├── run_eval.sh               # Example evaluation runner script
├── requirements.txt          # Python dependencies
│
├── synthesis/                # Core synthesis pipeline
│   ├── generator.py          # Qwen-based strategy generation (Dafny code)
│   ├── verifier.py           # Dafny verification wrapper (proof checking)
│   ├── compiler.py           # Dafny → Python compilation (verified code to runtime)
│   ├── runner.py             # Runtime testing of compiled strategies
│   ├── feedback_loop.py      # Main orchestration with iterative refinement
│   ├── prompts.py            # LLM prompt templates for strategy generation
│   └── rationale.py          # Strategy rationale extraction from LLM output
│
├── evaluations/              # Evaluation framework (modular design)
│   ├── __init__.py           # Package exports
│   ├── common/               # Shared utilities across evaluations
│   │   ├── __init__.py
│   │   ├── model_utils.py    # HuggingFace model loading with Dafny interface
│   │   ├── parser_utils.py   # Lark grammar parser creation utilities
│   │   └── token_selection.py # Token vocabulary selection for constrained decoding
│   │
│   ├── gsm_symbolic/         # GSM-Symbolic math reasoning evaluation
│   │   ├── __init__.py       # Package exports
│   │   ├── dataset.py        # Dataset loading from HuggingFace
│   │   ├── prompts.py        # CRANE-style prompt formatting
│   │   ├── answer_extraction.py # Answer extraction and evaluation
│   │   ├── grammar.py        # Dynamic grammar construction
│   │   ├── generation.py     # Generation methods (CRANE-CSD)
│   │   ├── environment.py    # Dafny environment setup
│   │   ├── metrics.py        # Evaluation metrics
│   │   └── cli.py            # Command-line interface
│   │
│   └── folio/                # FOLIO first-order logic reasoning evaluation
│       ├── __init__.py       # Package exports
│       ├── dataset.py        # Dataset loading from HuggingFace (yale-nlp/FOLIO)
│       ├── prompts.py        # CRANE-style FOL prompt formatting
│       ├── answer_extraction.py # Answer extraction (True/False/Uncertain)
│       ├── grammar.py        # Dynamic FOL grammar construction
│       ├── generation.py     # CSD generation for FOL expressions
│       ├── environment.py    # Dafny environment setup
│       ├── metrics.py        # Evaluation metrics with per-label breakdown
│       └── cli.py            # Command-line interface
│
├── scripts/                  # CLI entry points and utilities
│   ├── comprehensive_eval.py # Full benchmark evaluation across models
│   ├── run_csd_with_grammar.py # Run CSD strategies with custom grammars
│   ├── generate_gsm_csd.sh   # Helper script for GSM CSD generation
│   └── run_gsm_vanilla.sh    # Run GSM evaluation without CSD (baseline)
│
├── dafny/                    # Dafny source files
│   ├── GeneratedCSD.dfy      # Template for generated strategies (injection point)
│   └── VerifiedAgentSynthesis.dfy  # Core Dafny verification module (LM/parser specs)
│
├── dafny_externs/            # Python implementations of Dafny {:extern} functions
│   └── extern_functions.py   # LM, Parser, and decoding primitives for strategy execution
│
├── parsers/                  # Grammar and parsing utilities
│   ├── __init__.py           # Package exports
│   └── lark_parser.py        # Generic Lark-based grammar parser (character-level)
│
├── grammars/                 # Lark grammar files for various formats
│   ├── json.lark             # JSON syntax (ECMA-404 compliant)
│   ├── json_charwise.lark    # Character-level JSON grammar
│   ├── math.lark             # Mathematical expressions
│   ├── gsm.lark              # Math expressions for GSM-Symbolic (arithmetic operations)
│   ├── gsm_math.lark         # Extended math grammar for GSM calculations
│   ├── gsm_vars_only.lark    # Variable-only grammar for GSM
│   ├── folio.lark            # First-order logic expressions for FOLIO (Prover9-style)
│   └── folio_charwise.lark   # Character-level FOL grammar
│
├── outputs/                  # Generated outputs
│   └── generated-csd/
│       ├── latest_run.txt    # Pointer to most recent run
│       └── runs/             # Individual run directories
│           └── YYYYMMDD_HHMMSS_HASH/
│               ├── generated_csd.dfy
│               ├── success_report.json (or failure_report.json)
│               └── generated_csd/      # Compiled Python module
│
└── docs/                     # Research paper and documentation
    └── papers/               # Academic paper LaTeX source
```

---

## Evaluations Package (`evaluations/`)

The `evaluations/` package provides a modular, well-organized structure for benchmark evaluations. Each evaluation task is organized as a sub-package with clear separation of concerns.

### Common Utilities (`evaluations/common/`)

Shared utilities used across all evaluations:

```python
from evaluations.common import (
    # Model loading
    create_huggingface_lm,      # Create Dafny-compatible LM wrapper
    get_model_input_device,     # Get correct device for multi-GPU models
    get_max_input_length,       # Get safe max input length
    
    # Parser creation
    create_lark_dafny_parser,   # Create Dafny-compatible grammar parser
    
    # Token selection
    select_math_token_ids,      # Build math-optimized token vocabulary
)
```

### GSM-Symbolic Evaluation (`evaluations/gsm_symbolic/`)

Complete evaluation for CRANE-style CSD on grade school math problems:

```python
from evaluations.gsm_symbolic import (
    # Dataset
    load_gsm_symbolic,           # Load dataset from HuggingFace
    
    # Prompts
    make_gsm_prompt,             # Format CRANE-style prompts
    make_chatml_instruction,     # Wrap for Qwen models
    symbolize_question,          # Replace numbers with variables
    extract_numbers_with_context, # Extract numbers with surrounding context
    extract_variables,           # Extract variable definitions
    CRANE_FEW_SHOT_EXAMPLES,     # Few-shot examples for prompting
    
    # Answer extraction
    extract_answer,              # Extract and evaluate answers
    extract_gold_answer,         # Extract ground truth
    extract_symbolic_expression, # Extract symbolic math expressions
    evaluate_symbolic_expression, # Evaluate symbolic expressions
    is_symbolic_valid,           # Check if symbolic expression is valid
    extract_constrained_segments, # Extract CSD-constrained segments
    validate_math_segment,       # Validate math segment syntax
    
    # Grammar
    build_dynamic_grammar,       # Build grammar for specific variables
    extract_variables_from_mapping, # Get variables from mapping
    
    # Generation
    run_crane_csd,               # CRANE with CSD strategy
    dafny_seq_to_str,            # Convert Dafny sequence to string
    
    # Environment
    setup_dafny_environment,     # Load Dafny modules
    load_compiled_modules,       # Load compiled CSD modules
    verify_critical_tokens,      # Verify tokenizer has critical tokens
    
    # Metrics
    GSMMetrics,                  # Track evaluation metrics
)
```

### FOLIO Evaluation (`evaluations/folio/`)

Complete evaluation for CRANE-style CSD on first-order logic reasoning:

```python
from evaluations.folio import (
    # Dataset
    FOLIOExample,                # Data class for FOLIO examples
    load_folio,                  # Load dataset from HuggingFace
    load_folio_from_json,        # Load from local JSON file
    create_synthetic_folio_examples, # Create synthetic examples for testing
    normalize_label,             # Normalize label strings
    
    # Prompts
    make_folio_prompt,           # Format CRANE-style FOL prompts
    make_folio_prompt_no_cot,    # Without chain-of-thought
    FOLIO_FEW_SHOT_EXAMPLES,     # Few-shot examples
    FOL_GRAMMAR_DESCRIPTION,     # Grammar description for prompts
    CONSTRAINT_START,            # Constraint delimiter start
    CONSTRAINT_END,              # Constraint delimiter end
    
    # Grammar
    build_dynamic_grammar,       # Build grammar for specific predicates
    build_grammar_from_context,  # Build grammar from context text
    extract_predicates_from_generation,  # Extract predicates from text
    extract_constants_from_generation,   # Extract constants from text
    load_base_grammar,           # Load base FOL grammar
    
    # Generation
    run_crane_csd,               # CRANE with CSD strategy
    run_unconstrained,           # Baseline without CSD
    
    # Answer extraction
    extract_answer,              # Extract True/False/Uncertain
    extract_fol_sections,        # Parse FOL structure sections
    extract_fol_expressions,     # Extract FOL expressions
    is_valid_fol_structure,      # Check if FOL structure is valid
    check_answer_correctness,    # Check if answer matches ground truth
    
    # Environment
    setup_dafny_environment,     # Load Dafny modules
    load_compiled_modules,       # Load compiled CSD modules
    verify_critical_tokens,      # Verify tokenizer has critical tokens
    
    # Metrics
    FOLIOMetrics,                # Track evaluation metrics
)
```

---

## Key Components

### 1. Strategy Generator (`synthesis/generator.py`)

The `StrategyGenerator` class uses Qwen to generate Dafny CSD strategy code.

```python
from synthesis.generator import StrategyGenerator

# Initialize with custom model
generator = StrategyGenerator(
    model_name="Qwen/Qwen2.5-Coder-7B-Instruct",
    device="cuda",           # or "mps", "cpu", "auto"
    max_new_tokens=512,
    temperature=0.7
)

# Generate initial strategy
strategy = generator.generate_initial("Generate a JSON validation strategy")

# Refine after errors
refined = generator.refine_after_verification_error(strategy, error_message)
refined = generator.refine_after_runtime_error(strategy, traceback)
refined = generator.refine_after_compilation_error(strategy, error_message)

# Inject into Dafny template
full_dafny_code = generator.inject_strategy(strategy)
```

### 2. Dafny Verifier (`synthesis/verifier.py`)

The `DafnyVerifier` runs formal verification on generated Dafny code.

```python
from synthesis.verifier import DafnyVerifier

verifier = DafnyVerifier(
    dafny_path="dafny",
    timeout=60
)

# Verify code string
result = verifier.verify(dafny_code)
if result.success:
    print("Verification passed!")
else:
    print(result.get_error_summary())

# Verify file directly
result = verifier.verify_file(Path("my_strategy.dfy"))
```

### 3. Dafny Compiler (`synthesis/compiler.py`)

The `DafnyCompiler` compiles verified Dafny code to Python.

```python
from synthesis.compiler import DafnyCompiler

compiler = DafnyCompiler(
    dafny_path="dafny",
    output_dir=Path("outputs/"),
    timeout=120
)

result = compiler.compile(dafny_code, output_name="my_csd")
if result.success:
    print(f"Compiled to: {result.output_dir}")
    print(f"Main module: {result.main_module_path}")
```

### 4. Synthesis Pipeline (`synthesis/feedback_loop.py`)

The `SynthesisPipeline` orchestrates the full generate → verify → compile → test loop.

```python
from synthesis.feedback_loop import SynthesisPipeline, SynthesisExhaustionError

pipeline = SynthesisPipeline(
    max_iterations=5,
    save_reports=True
)

try:
    result = pipeline.synthesize(
        task_description="Generate a JSON strategy",
        output_name="json_csd"
    )
    print(f"Success! Strategy: {result.strategy_code}")
    print(f"Compiled to: {result.compiled_module_path}")
except SynthesisExhaustionError as e:
    print(e.get_failure_summary())
```

---

## Evaluation Scripts

### Comprehensive Evaluation (`scripts/comprehensive_eval.py`)

**Purpose**: Run full benchmark evaluation across multiple models and datasets.

**Features**:
- Tests multiple CSDs per model (3 by default)
- Evaluates on both GSM-Symbolic and FOLIO
- Runs baseline (unconstrained) evaluation for comparison
- Collects statistics and identifies best performing CSDs

**Models tested**:
- Qwen/Qwen2.5-1.5B-Instruct
- Qwen/Qwen2.5-Coder-7B-Instruct
- meta-llama/Llama-3.1-8B-Instruct
- deepseek-ai/DeepSeek-R1-Distill-Llama-8B
- deepseek-ai/DeepSeek-R1-Distill-Qwen-14B

**Usage**:
```bash
# Full evaluation
python scripts/comprehensive_eval.py --output results.json

# Single model evaluation
python scripts/comprehensive_eval.py --model "Qwen/Qwen2.5-Coder-7B-Instruct" --output results.json

# Skip CSD synthesis (use existing)
python scripts/comprehensive_eval.py --skip-synthesis --output results.json

# Skip baseline evaluation
python scripts/comprehensive_eval.py --skip-baseline --output results.json
```

### GSM-Symbolic Math Reasoning (`evaluations/gsm_symbolic/`)

**Purpose**: Evaluates CRANE-style CSD on GSM-Symbolic dataset (grade school math word problems).

**Architecture**: The evaluation is organized as a modular package:

| Module | Description |
|--------|-------------|
| `dataset.py` | Dataset loading from HuggingFace |
| `prompts.py` | CRANE-style prompt formatting, variable extraction |
| `answer_extraction.py` | Answer extraction with multiple fallback strategies |
| `grammar.py` | Dynamic grammar construction for specific variables |
| `generation.py` | CSD generation method (CRANE-CSD) |
| `environment.py` | Dafny environment setup and module loading |
| `metrics.py` | Evaluation metrics (accuracy, format validity, etc.) |
| `cli.py` | Command-line interface |

**Key Features**:
- **CRANE-style dynamic switching**: Unconstrained reasoning until `<<` delimiter detected, then CSD-constrained math generation, resume unconstrained until `####`
- **Robust answer extraction**: Multiple fallback strategies (from `####`, from `<< >>` calculations, from last numbers)
- **Smart delimiter detection**: Handles tokenization issues (split delimiters, cooldown to prevent re-detection)
- **Repetition detection**: Early stopping when model gets stuck in loops

**Metrics**:
- Answer accuracy (exact numeric match)
- Valid format rate (outputs contain `#### <number>`)
- Syntax validity (math expressions in `<< >>` pass grammar validation)

**Usage**:
```bash
# Using the module directly (recommended)
python -m evaluations.gsm_symbolic.cli \
  --run-dir outputs/generated-csd/runs/20260110_180926_52ce55 \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --device cuda \
  --limit 10 \
  --max-steps 1024 \
  --vocab-size 2000 \
  --grammar grammars/gsm.lark \
  --debug-delimiters
```

**Python API**:
```python
from evaluations.gsm_symbolic import (
    load_gsm_symbolic,
    symbolize_question,
    make_gsm_prompt,
    extract_answer,
    GSMMetrics,
    setup_dafny_environment,
)

# Load dataset
ds = load_gsm_symbolic(config="main", limit=10, random_sample=True)

# Process a question
question = ds[0]["question"]
symbolic_question, variable_mapping = symbolize_question(question)
prompt = make_gsm_prompt(question, symbolic_question=symbolic_question)

# Extract answer from generated text
pred_answer, valid_format, symbolic_expr = extract_answer(
    generated_text,
    variable_mapping=variable_mapping,
)
```

### FOLIO First-Order Logic Reasoning (`evaluations/folio/`)

**Purpose**: Evaluates CRANE-style CSD on FOLIO dataset (first-order logic natural language inference).

**Architecture**: The evaluation is organized as a modular package:

| Module | Description |
|--------|-------------|
| `dataset.py` | Dataset loading from HuggingFace (yale-nlp/FOLIO) |
| `prompts.py` | CRANE-style FOL prompt formatting with Prover9 syntax |
| `answer_extraction.py` | Answer extraction for True/False/Uncertain labels |
| `grammar.py` | Dynamic grammar construction for per-question predicates/constants |
| `generation.py` | CSD generation method (CRANE-CSD for FOL) |
| `environment.py` | Dafny environment setup and module loading |
| `metrics.py` | Evaluation metrics with per-label accuracy breakdown |
| `cli.py` | Command-line interface |

**Key Features**:
- **CRANE-style dynamic switching**: Unconstrained reasoning until `<<` delimiter detected, then CSD-constrained FOL generation, resume unconstrained until `Answer:`
- **FOL Operators**: Supports `{and}`, `{or}`, `{xor}`, `{not}`, `{implies}`, `{iff}`, `{forall}`, `{exists}`
- **Prover9-style syntax**: Predicates like `Predicate(arg1, arg2)`, quantifiers like `{forall} x: P(x)`
- **Three-class classification**: True, False, Uncertain
- **Dynamic grammar**: Predicates and constants extracted from context to constrain generation

**Metrics**:
- Overall accuracy (three-class: True/False/Uncertain)
- Per-label accuracy breakdown
- Valid format rate (outputs contain `Answer:` with valid label)
- Syntax validity (FOL expressions in `<< >>` pass grammar validation)

**Usage**:
```bash
# Using the module directly (recommended)
python -m evaluations.folio.cli \
  --run-dir outputs/generated-csd/runs/YOUR_RUN \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --device cuda \
  --limit 50

# Run without constrained decoding (baseline)
python -m evaluations.folio.cli \
  --run-dir outputs/generated-csd/runs/YOUR_RUN \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --unconstrained \
  --limit 50

# Use synthetic examples (for testing without dataset access)
python -m evaluations.folio.cli \
  --run-dir outputs/generated-csd/runs/YOUR_RUN \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --synthetic \
  --limit 5

# Load from local JSON file
python -m evaluations.folio.cli \
  --run-dir outputs/generated-csd/runs/YOUR_RUN \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --json-path data/folio_validation.json
```

**Python API**:
```python
from evaluations.folio import (
    load_folio,
    make_folio_prompt,
    extract_answer,
    FOLIOMetrics,
    setup_dafny_environment,
    build_dynamic_grammar,
)

# Load dataset (requires HuggingFace authentication for gated dataset)
ds = load_folio(split="validation", limit=10)

# Process an example
example = ds[0]
prompt = make_folio_prompt(example.premises, example.conclusion)

# Build dynamic grammar with extracted predicates
predicates = ["Person", "Student", "Attends"]
constants = ["John", "Mary"]
grammar = build_dynamic_grammar(predicates, constants)

# Extract answer from generated text
pred_answer, confidence = extract_answer(generated_text)
# pred_answer is one of: "True", "False", "Uncertain"
```

**Dataset Note**: The FOLIO dataset (`yale-nlp/FOLIO`) is gated on HuggingFace and requires authentication:
```bash
huggingface-cli login
```
Alternatively, use `--synthetic` for testing or `--json-path` for local data.

---

## Running with Custom Grammars

Use `scripts/run_csd_with_grammar.py` to test compiled strategies with specific grammars:

```bash
# With a .lark grammar file
python scripts/run_csd_with_grammar.py \
    --run-dir outputs/generated-csd/runs/20260105_204255_8b7116 \
    --grammar grammars/json.lark

# With a built-in format
python scripts/run_csd_with_grammar.py \
    --run-dir outputs/generated-csd/runs/20260105_204255_8b7116 \
    --format json

# With HuggingFace tokenizer vocabulary
python scripts/run_csd_with_grammar.py \
    --run-dir outputs/generated-csd/runs/20260105_204255_8b7116 \
    --format json \
    --tokenizer Qwen/Qwen2.5-Coder-7B-Instruct

# Custom options
python scripts/run_csd_with_grammar.py \
    --run-dir outputs/generated-csd/runs/XXXXX \
    --grammar grammars/math.lark \
    --max-steps 100 \
    --vocab-size 1000 \
    --seed 42
```

### Built-in Formats

- `json` - JSON according to ECMA-404
- `math` - Mathematical expressions

---

## Parsers Module

The `parsers/` module provides grammar-based validation using Lark:

### Lark Grammar Parser

```python
from parsers import LarkGrammarParser, create_parser_for_format

# Use built-in format
parser = create_parser_for_format("json")

# Use custom grammar file
parser = LarkGrammarParser.from_grammar_file("my_grammar.lark")

# Check validity
is_valid = parser.is_valid_prefix('{"key": ')  # True
is_complete = parser.is_complete('{"key": "value"}')  # True
```

### Interactive Parser

```python
from parsers import InteractiveLarkParser

# Create parser for interactive/incremental validation
parser = InteractiveLarkParser.from_grammar_file("grammars/json.lark")

# Check prefixes incrementally
parser.is_valid_prefix("{")     # True
parser.is_valid_prefix('{"x"')  # True
parser.is_complete('{"x": 1}')  # True
```

### Available Grammar Creators

```python
from parsers import (
    create_json_lark_grammar,    # Create JSON grammar string
    create_python_lark_grammar,  # Create Python grammar string
    create_sql_lark_grammar,     # Create SQL grammar string
)
```

---

## Output Structure

Each synthesis run creates a unique directory:

```
outputs/generated-csd/runs/20260105_204255_8b7116/
├── generated_csd.dfy           # The Dafny source
├── success_report.json         # Metadata and rationale
└── generated_csd/              # Compiled Python module
    ├── __main__.py
    ├── _dafny/                 # Dafny runtime
    ├── GeneratedCSD.py         # Main generated module
    ├── VerifiedDecoderAgent.py # Verification support
    └── ...
```

### Success Report Format

```json
{
  "strategy_code": "// CSD_RATIONALE_BEGIN\n// Hybrid approach...\n// CSD_RATIONALE_END\ngenerated := CSDHelpers.HybridGeneration(...);",
  "tool_choice_rationale": "Hybrid approach balances creativity and validity...",
  "dafny_file": "/path/to/generated_csd.dfy",
  "compiled_dir": "/path/to/generated_csd/",
  "total_attempts": 1,
  "timestamp": "2026-01-05T20:43:18.641197"
}
```

### Failure Report Format

```json
{
  "task_description": "Generate a strategy...",
  "total_attempts": 5,
  "timestamp": "...",
  "attempts": [
    {
      "attempt_number": 1,
      "strategy_code": "...",
      "failed_at": "verification",
      "error_summary": "..."
    }
  ],
  "failure_patterns": {
    "verification_failures": 3,
    "compilation_failures": 1,
    "runtime_failures": 1
  }
}
```

---

## Available CSD Strategies

The Dafny template supports these helper functions for building strategies:

| Function | Description |
|----------|-------------|
| `CSDHelpers.PureConstrainedGeneration(lm, parser, prompt, maxSteps)` | Fully constrained generation - always valid |
| `CSDHelpers.UnconstrainedGeneration(lm, prompt, maxSteps)` | No constraints - may produce invalid output |
| `CSDHelpers.HybridGeneration(lm, parser, prompt, maxSteps, interval)` | Alternates constrained/unconstrained every `interval` steps |
| `CSDHelpers.TryUnconstrainedThenConstrained(lm, parser, prompt, maxSteps, n)` | Try `n` unconstrained steps, fall back to constrained |
| `CSDHelpers.ConstrainedStep(lm, parser, prompt, generated)` | Single constrained step |
| `CSDHelpers.UnconstrainedStep(lm, prompt, generated)` | Single unconstrained step |

---

## Dafny Template

The generated strategies are injected into `dafny/GeneratedCSD.dfy`:

```dafny
method MyCSDStrategy(lm: LM, parser: Parser, prompt: Prefix, maxSteps: nat) 
  returns (generated: Prefix)
  modifies lm.Logits
  requires lm.ValidTokensIdsLogits()
  requires parser.IsValidPrefix([])
  ensures lm.ValidTokensIdsLogits()
  ensures |generated| <= maxSteps
  ensures parser.IsValidPrefix(generated)
  ensures |generated| == maxSteps || parser.IsCompletePrefix(generated)
{
  // QWEN_INSERT_STRATEGY_HERE
}
```

The postconditions guarantee:
1. Token/logit consistency is maintained
2. Output length doesn't exceed `maxSteps`
3. Output is always a valid prefix according to the grammar
4. Output is complete if it stopped before `maxSteps`

---

## CRANE-Style CSD for GSM-Symbolic

The GSM-Symbolic evaluation implements a **CRANE-style** (Constrained Reasoning with Adaptive Natural Expression) approach:

### Architecture

1. **Unconstrained Reasoning Phase**: Model generates natural language reasoning freely
2. **Delimiter Detection**: When `<<` is detected, switch to constrained mode
3. **Constrained Math Generation**: Use CSD strategy to generate valid math expressions within `<< >>`
4. **Validation**: Validate math expression against `grammars/gsm.lark` or `grammars/gsm_math.lark`
5. **Resume Unconstrained**: After `>>`, return to unconstrained until `####` or EOS

### Key Implementation Features

- **Smart Delimiter Detection**: 
  - Checks last 20 tokens for `<<` (handles multi-token delimiters)
  - 25-step cooldown prevents re-detecting same delimiter
  - Only detects when in unconstrained mode
  
- **Robust Answer Extraction**:
  - Primary: Extract number after `####`
  - Fallback 1: Extract from last `<< ... = X>>` calculation
  - Fallback 2: Extract number before `####` (e.g., "20 ####")
  - Fallback 3: Extract last reasonable number in text
  
- **Repetition Detection**: 
  - Monitors last 100 tokens for repeating 40-character chunks
  - Breaks early if same pattern appears 4+ times
  
- **Number Completion**: 
  - Waits for complete numbers after `####` (handles multi-token numbers)
  - Checks for whitespace/punctuation after number to confirm completion

### Expected Performance

With Qwen2.5-Coder-7B-Instruct on GSM-Symbolic:
- **Answer Accuracy**: ~70% (model reasoning limitations)
- **Valid Format**: 100% (CSD ensures correct structure)
- **Syntax Validity**: 100% (all `<< >>` expressions are valid)

For higher accuracy, use larger models (13B, 70B) or math-specialized models (Qwen-Math, DeepSeekMath).

---

## Troubleshooting

### Dafny not found
```bash
# Check if dafny is in PATH
which dafny

# Or specify path explicitly
python run_synthesis.py --task "..." --dafny-path /path/to/dafny
```

### GPU memory issues
```bash
# Use a smaller model
python run_synthesis.py --task "..." --model Qwen/Qwen2.5-Coder-3B-Instruct

# Or force CPU
python run_synthesis.py --task "..." --device cpu
```

### Verification timeout
```bash
# Increase iterations (gives more chances to find a valid strategy)
python run_synthesis.py --task "..." --max-iterations 10
```

### Import errors with compiled modules
```python
# Add the output directory to Python path
import sys
sys.path.insert(0, "outputs/generated-csd/runs/XXXXX/generated_csd")
import GeneratedCSD
```

### GSM-Symbolic evaluation issues

**Low answer accuracy despite 100% format validity**:
- This is expected for 7B models - the CSD system is working correctly
- The errors are model reasoning mistakes, not CSD failures
- Try larger models or math-specialized models for better accuracy

**Model not generating `####` delimiter**:
- Check prompt includes explicit example with `####`
- Ensure `--max-steps` is high enough (1024 recommended)
- Model may need more explicit instruction in prompt

**Multiple `<<` detections for same calculation**:
- Cooldown should prevent this (25 steps)
- If still happening, check delimiter detection logic in generation module

---

## License

See repository for license details.
