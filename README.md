# CSD Generation: Constrained Decoding Strategy Synthesis Pipeline

A synthesis pipeline for generating **Constrained Decoding Strategies (CSD)** using LLMs (Qwen) with formal verification via Dafny. The pipeline automatically generates, verifies, compiles, and tests constrained decoding strategies that guarantee valid output from language models according to specified grammars (JSON, Math, etc.).

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
csd-generation/
├── run_synthesis.py          # Main CLI entry point for CSD strategy synthesis
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
│   └── gsm_symbolic/         # GSM-Symbolic math reasoning evaluation
│       ├── __init__.py       # Package exports
│       ├── dataset.py        # Dataset loading from HuggingFace
│       ├── prompts.py        # CRANE-style prompt formatting
│       ├── answer_extraction.py # Answer extraction and evaluation
│       ├── grammar.py        # Dynamic grammar construction
│       ├── generation.py     # Generation methods (standard, CRANE, CSD)
│       ├── environment.py    # Dafny environment setup
│       ├── metrics.py        # Evaluation metrics
│       └── cli.py            # Command-line interface
│
├── scripts/                  # CLI entry points and utilities
│   ├── run_csd_with_grammar.py    # Run CSD strategies with custom grammars
│   ├── validate_json_csd.py       # Validate JSON-oriented strategies
│   └── generate_gsm_csd.sh        # Helper script for GSM CSD generation
│
├── dafny/                    # Dafny source files
│   ├── GeneratedCSD.dfy      # Template for generated strategies (injection point)
│   └── VerifiedAgentSynthesis.dfy  # Core Dafny verification module (LM/parser specs)
│
├── dafny_externs/            # Python implementations of Dafny {:extern} functions
│   └── extern_functions.py   # LM, Parser, and decoding primitives for strategy execution
│
├── parsers/                  # Grammar and parsing utilities
│   ├── json_prefix.py        # Hand-optimized JSON prefix validator (fast incremental)
│   ├── lark_parser.py        # Generic Lark-based grammar parser (character-level)
│   ├── model_token_parser.py # Token-level parsing for LMs (vocabulary-aware)
│   └── schema_to_grammar.py  # JSON Schema → Lark grammar converter (for dynamic schemas)
│
├── grammars/                 # Lark grammar files for various formats
│   ├── json.lark             # JSON syntax (ECMA-404 compliant)
│   ├── json_charwise.lark    # Character-level JSON grammar
│   ├── math.lark             # Mathematical expressions
│   ├── gsm.lark              # Math expressions for GSM-Symbolic (arithmetic operations)
│   └── gsm_math.lark         # Extended math grammar for GSM calculations
│
├── tests/                    # Unit tests
│   └── test_json_prefix.py   # Tests for JSON prefix validation
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
    
    # Answer extraction
    extract_answer,              # Extract and evaluate answers
    extract_gold_answer,         # Extract ground truth
    
    # Generation
    run_crane_csd,               # CRANE with CSD strategy (default)
    
    # Environment
    setup_dafny_environment,     # Load Dafny modules
    
    # Metrics
    GSMMetrics,                  # Track evaluation metrics
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

# Or using the backward-compatible script
python scripts/evaluate_gsm_symbolic.py \
  --run-dir outputs/generated-csd/runs/20260110_180926_52ce55 \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --device cuda \
  --limit 10
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

**Implementation Details**:
- **Prompt Engineering**: Explicit step-by-step examples showing `<< >>` usage and `####` format
- **Delimiter Cooldown**: 25-step cooldown (longer than 20-token window) prevents re-detecting same `<<`
- **Number Extraction**: Waits for complete numbers after `####` (handles multi-token numbers like "20" = "2" + "0")
- **Grammar Validation**: Uses `grammars/gsm.lark` or `grammars/gsm_math.lark` for math expression validation

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

## JSON Validation Script

Use `scripts/validate_json_csd.py` to validate JSON-oriented strategies:

```bash
# Run full validation
python scripts/validate_json_csd.py --run-dir outputs/generated-csd/runs/XXXXX

# Test only the prefix validator
python scripts/validate_json_csd.py --test-only

# Output as JSON
python scripts/validate_json_csd.py --json
```

---

## Parsers Module

The `parsers/` module provides grammar-based validation:

### Lark Grammar Parser (Recommended)

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

### JSON Prefix Validator (Optimized)

```python
from parsers import is_valid_json_prefix, is_complete_json, JsonPrefixValidator

# Quick checks
is_valid_json_prefix('{"key": "value"')  # True - valid prefix
is_complete_json('{"key": "value"}')      # True - complete JSON

# Incremental validation
validator = JsonPrefixValidator()
validator.feed("{")    # True
validator.feed('"')    # True
validator.feed("key")  # True
validator.is_complete() # False
```

### Schema to Grammar Converter (`parsers/schema_to_grammar.py`)

**Purpose**: Converts JSON Schema to Lark grammar for character-level parsing.

**Use Case**: Dynamic schema validation when schemas are not known at compile time.

```python
from parsers.schema_to_grammar import schema_to_lark_grammar

schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "number"}
    }
}

grammar = schema_to_lark_grammar(schema)
parser = LarkGrammarParser.from_grammar_string(grammar)
```

---

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_json_prefix.py -v

# Run with timeout
pytest tests/ -v --timeout=30
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
- If still happening, check delimiter detection logic in `evaluate_gsm_symbolic.py`

---

## License

See repository for license details.
