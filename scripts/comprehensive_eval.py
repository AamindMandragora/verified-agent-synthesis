#!/usr/bin/env python3
"""
Comprehensive CSD Evaluation Script

Testing Strategy:
- 3 CSDs generated per model (for both gsm and folio tasks)
- Each dataset (GSM-Symbolic, FOLIO) tested 3 times per CSD
- Total: 9 evaluation runs per dataset per model
- Collects average stats per CSD and identifies best CSD
- Also runs BASELINE (unconstrained) evaluation for comparison

Models tested:
1. Qwen/Qwen2.5-1.5B-Instruct
2. Qwen/Qwen2.5-Coder-7B-Instruct
3. meta-llama/Llama-3.1-8B-Instruct
4. deepseek-ai/DeepSeek-R1-Distill-Llama-8B
5. deepseek-ai/DeepSeek-R1-Distill-Qwen-14B

Usage:
    python scripts/comprehensive_eval.py --output results.json
    python scripts/comprehensive_eval.py --model "Qwen/Qwen2.5-Coder-7B-Instruct" --output results.json
    python scripts/comprehensive_eval.py --skip-synthesis --output results.json  # Use existing CSDs
    python scripts/comprehensive_eval.py --skip-baseline --output results.json  # Skip baseline evaluation
"""

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import statistics
import secrets

# Project root
PROJECT_ROOT = Path(__file__).parent.parent

# Models to test (excluding Qwen2.5-Math-7B which is fine-tuned on LaTeX)
MODELS = [
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-Coder-7B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
]

# Task descriptions for CSD generation
GSM_TASK_DESC = """Generate short symbolic mathematical expressions for GSM-Symbolic reasoning. \
The parser enforces a strict arithmetic expression grammar with VARIABLES and numeric constants. \
CRITICAL RULES: \
1. Every expression MUST contain at least one variable - pure numeric expressions like '2 * 2' are INVALID. \
   Variables may appear as letter+digits (n1, x2), split letter/digit tokens (n 1, x 2), or letter-only (n, x, foo). \
2. Variables represent problem values; numeric constants (12, 100) are for unit conversions and percentages. \
3. The constrained windows are SHORT (typically 5-20 tokens for expressions like 'n1 + n2 * 12'). \
4. The grammar is RECURSIVE but depth is BOUNDED in runtime; prefer balanced parentheses and compact expressions. \
5. The grammar includes the closing delimiter '>>' - expressions must be complete and compact. \
6. Avoid trivial outputs (e.g., just a single variable) unless the problem truly requires it; include necessary operators."""

FOLIO_TASK_DESC = """Generate first-order logic (FOL) expressions for logical reasoning. \
The parser enforces a strict FOL grammar with predicates, constants, variables, and logical connectives. \
CRITICAL RULES: \
1. Predicates are CamelCase (e.g., IsStudent, LivesIn) and constants are lowercase (e.g., john, mary). \
2. Variables are single lowercase letters (x, y, z). \
3. Logical connectives: ∧ (and), ∨ (or), → (implies), ¬ (not), ∀ (forall), ∃ (exists). \
4. The constrained windows contain FOL formulas inside << >> delimiters. \
5. Expressions must be well-formed with balanced parentheses. \
6. The grammar is strict - prefer simpler, well-typed formulas over complex nested ones."""

# Evaluation configuration
NUM_CSDS = 3  # Number of CSDs to generate per model
NUM_EVAL_RUNS = 3  # Number of evaluation runs per CSD per dataset
EVAL_LIMIT = 100  # Number of examples to evaluate per run


@dataclass
class EvalRunResult:
    """Result of a single evaluation run."""
    run_id: int
    accuracy: float
    format_rate: float  # For GSM: valid format rate, For FOLIO: structure validity
    syntax_rate: float  # Syntax validity of constrained segments
    avg_tokens: float
    avg_time: float
    total_time: float
    num_examples: int


@dataclass
class CSDEvalResult:
    """Aggregated results for a single CSD across multiple runs."""
    csd_id: str
    csd_run_dir: str
    runs: List[EvalRunResult] = field(default_factory=list)
    
    @property
    def avg_accuracy(self) -> float:
        if not self.runs:
            return 0.0
        return statistics.mean(r.accuracy for r in self.runs)
    
    @property
    def std_accuracy(self) -> float:
        if len(self.runs) < 2:
            return 0.0
        return statistics.stdev(r.accuracy for r in self.runs)
    
    @property
    def avg_format_rate(self) -> float:
        if not self.runs:
            return 0.0
        return statistics.mean(r.format_rate for r in self.runs)
    
    @property
    def avg_syntax_rate(self) -> float:
        if not self.runs:
            return 0.0
        return statistics.mean(r.syntax_rate for r in self.runs)
    
    @property
    def avg_time(self) -> float:
        if not self.runs:
            return 0.0
        return statistics.mean(r.avg_time for r in self.runs)
    
    def to_dict(self) -> dict:
        return {
            "csd_id": self.csd_id,
            "csd_run_dir": self.csd_run_dir,
            "num_runs": len(self.runs),
            "avg_accuracy": self.avg_accuracy,
            "std_accuracy": self.std_accuracy,
            "avg_format_rate": self.avg_format_rate,
            "avg_syntax_rate": self.avg_syntax_rate,
            "avg_time_per_example": self.avg_time,
            "runs": [asdict(r) for r in self.runs],
        }


@dataclass
class BaselineResults:
    """Results for baseline (unconstrained) evaluation."""
    runs: List[EvalRunResult] = field(default_factory=list)
    
    @property
    def avg_accuracy(self) -> float:
        if not self.runs:
            return 0.0
        return statistics.mean(r.accuracy for r in self.runs)
    
    @property
    def std_accuracy(self) -> float:
        if len(self.runs) < 2:
            return 0.0
        return statistics.stdev(r.accuracy for r in self.runs)
    
    @property
    def avg_format_rate(self) -> float:
        if not self.runs:
            return 0.0
        return statistics.mean(r.format_rate for r in self.runs)
    
    @property
    def avg_time(self) -> float:
        if not self.runs:
            return 0.0
        return statistics.mean(r.avg_time for r in self.runs)
    
    def to_dict(self) -> dict:
        return {
            "num_runs": len(self.runs),
            "avg_accuracy": self.avg_accuracy,
            "std_accuracy": self.std_accuracy,
            "avg_format_rate": self.avg_format_rate,
            "avg_time_per_example": self.avg_time,
            "runs": [asdict(r) for r in self.runs],
        }


@dataclass
class DatasetResults:
    """Results for a dataset (GSM or FOLIO) across all CSDs."""
    dataset: str
    csd_results: List[CSDEvalResult] = field(default_factory=list)
    baseline_results: Optional[BaselineResults] = None
    
    @property
    def best_csd(self) -> Optional[CSDEvalResult]:
        if not self.csd_results:
            return None
        return max(self.csd_results, key=lambda c: c.avg_accuracy)
    
    @property
    def overall_avg_accuracy(self) -> float:
        if not self.csd_results:
            return 0.0
        return statistics.mean(c.avg_accuracy for c in self.csd_results)
    
    @property
    def overall_std_accuracy(self) -> float:
        if len(self.csd_results) < 2:
            return 0.0
        return statistics.stdev(c.avg_accuracy for c in self.csd_results)
    
    @property
    def improvement_over_baseline(self) -> Optional[float]:
        """Calculate accuracy improvement of best CSD over baseline."""
        if not self.baseline_results or not self.best_csd:
            return None
        return self.best_csd.avg_accuracy - self.baseline_results.avg_accuracy
    
    @property
    def avg_improvement_over_baseline(self) -> Optional[float]:
        """Calculate average accuracy improvement of all CSDs over baseline."""
        if not self.baseline_results or not self.csd_results:
            return None
        return self.overall_avg_accuracy - self.baseline_results.avg_accuracy
    
    def to_dict(self) -> dict:
        best = self.best_csd
        result = {
            "dataset": self.dataset,
            "num_csds": len(self.csd_results),
            "overall_avg_accuracy": self.overall_avg_accuracy,
            "overall_std_accuracy": self.overall_std_accuracy,
            "best_csd": {
                "csd_id": best.csd_id if best else None,
                "accuracy": best.avg_accuracy if best else None,
                "std": best.std_accuracy if best else None,
            },
            "csd_results": [c.to_dict() for c in self.csd_results],
        }
        
        # Add baseline comparison if available
        if self.baseline_results:
            result["baseline"] = self.baseline_results.to_dict()
            result["comparison"] = {
                "baseline_accuracy": self.baseline_results.avg_accuracy,
                "best_csd_accuracy": best.avg_accuracy if best else None,
                "improvement_best_csd": self.improvement_over_baseline,
                "improvement_avg_csd": self.avg_improvement_over_baseline,
                "improvement_best_csd_pct": (
                    (self.improvement_over_baseline / self.baseline_results.avg_accuracy * 100)
                    if self.baseline_results.avg_accuracy > 0 and self.improvement_over_baseline is not None
                    else None
                ),
            }
        
        return result


@dataclass
class ModelResults:
    """Complete results for a model."""
    model_name: str
    gsm_results: Optional[DatasetResults] = None
    folio_results: Optional[DatasetResults] = None
    
    def to_dict(self) -> dict:
        return {
            "model": self.model_name,
            "gsm": self.gsm_results.to_dict() if self.gsm_results else None,
            "folio": self.folio_results.to_dict() if self.folio_results else None,
        }


def run_command(cmd: List[str], env: Optional[Dict] = None, timeout: int = 3600) -> Tuple[int, str, str]:
    """Run a command and return (returncode, stdout, stderr)."""
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=full_env,
            cwd=str(PROJECT_ROOT),
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Command timed out"
    except Exception as e:
        return -1, "", str(e)


def generate_csd(
    task: str,
    output_name: str,
    model: str,
    dataset: str,
    temperature: float = 0.7,
    max_iterations: int = 10,
) -> Optional[str]:
    """
    Generate a CSD using the synthesis pipeline.
    
    Returns the run directory path if successful, None otherwise.
    """
    cmd = [
        sys.executable, "run_synthesis.py",
        "--task", task,
        "--dataset", dataset,
        "--output-name", output_name,
        "--model", model,
        "--temperature", str(temperature),
        "--max-iterations", str(max_iterations),
        "--device", "auto",
    ]

    if dataset == "gsm_symbolic":
        cmd.extend([
            "--min-accuracy", "0.3",
            "--min-format-rate", "0.5",
            "--min-syntax-rate", "0.5",
            "--eval-sample-size", "10",
            "--eval-max-steps", "2048",
        ])
    elif dataset == "folio":
        cmd.extend([
            "--min-accuracy", "0.5",
            "--min-format-rate", "0.8",
            "--min-syntax-rate", "0.8",
            "--eval-sample-size", "10",
        ])
    else:
        raise ValueError(f"Unsupported dataset for comprehensive_eval synthesis: {dataset}")
    
    print(f"  Running: python run_synthesis.py --task '...' --output-name {output_name}")
    returncode, stdout, stderr = run_command(cmd, timeout=1800)  # 30 min timeout
    
    if returncode != 0:
        print(f"  ✗ CSD generation failed")
        print(f"  stderr: {stderr[:500]}")
        return None
    
    # Extract run directory from output
    # Look for "Run directory: ..." or read from latest_run.txt
    latest_run_file = PROJECT_ROOT / "outputs" / "generated-csd" / "latest_run.txt"
    if latest_run_file.exists():
        run_dir = latest_run_file.read_text().strip()
        print(f"  ✓ CSD generated: {run_dir}")
        return run_dir
    
    print(f"  ✗ Could not find run directory")
    return None


def run_gsm_evaluation(
    run_dir: str,
    model: str,
    limit: int = EVAL_LIMIT,
    random_sample: bool = True,
    unconstrained: bool = False,
) -> Optional[EvalRunResult]:
    """
    Run GSM-Symbolic evaluation and parse results.
    
    Args:
        run_dir: Path to CSD run directory (ignored if unconstrained=True)
        model: Model name
        limit: Number of examples
        random_sample: Whether to randomly sample
        unconstrained: If True, run baseline without CSD constraints
    
    Returns EvalRunResult if successful, None otherwise.
    """
    cmd = [
        sys.executable, "-m", "evaluations.gsm_symbolic.cli",
        "--run-dir", run_dir,
        "--model", model,
        "--device", "cuda",
        "--limit", str(limit),
        "--max-steps", "1024",
        "--vocab-size", "3000",
    ]
    if random_sample:
        cmd.append("--random-sample")
    if unconstrained:
        cmd.append("--unconstrained")
    
    mode_str = "baseline (unconstrained)" if unconstrained else "CSD"
    print(f"    Running GSM {mode_str} evaluation (limit={limit})...")
    start_time = time.time()
    returncode, stdout, stderr = run_command(cmd, timeout=7200)  # 2 hour timeout
    total_time = time.time() - start_time
    
    if returncode != 0:
        print(f"    ✗ GSM evaluation failed")
        print(f"    stderr: {stderr[:500]}")
        return None
    
    # Parse results from stdout
    # Look for lines like:
    # Examples: N
    # Answer Accuracy: X.X%
    # Valid Format Rate: X.X%
    # Syntax Validity: X.X%
    # Avg Tokens: X.X
    # Avg Time: X.Xs
    
    result = parse_gsm_output(stdout, total_time)
    if result:
        print(f"    ✓ Accuracy: {result.accuracy:.1f}% | Format: {result.format_rate:.1f}% | Time: {result.avg_time:.2f}s/ex")
    return result


def parse_gsm_output(stdout: str, total_time: float) -> Optional[EvalRunResult]:
    """Parse GSM evaluation output to extract metrics."""
    lines = stdout.strip().split('\n')
    
    num_examples = 0
    accuracy = 0.0
    format_rate = 0.0
    syntax_rate = 0.0
    avg_tokens = 0.0
    avg_time = 0.0
    
    for line in lines:
        line = line.strip()
        if line.startswith("Examples:"):
            try:
                num_examples = int(line.split(":")[1].strip())
            except:
                pass
        elif line.startswith("Answer Accuracy:"):
            try:
                accuracy = float(line.split(":")[1].strip().rstrip('%'))
            except:
                pass
        elif line.startswith("Valid Format Rate:"):
            try:
                format_rate = float(line.split(":")[1].strip().rstrip('%'))
            except:
                pass
        elif line.startswith("Syntax Validity:"):
            try:
                # Format: "Syntax Validity: X.X% (N/M segments)"
                syntax_rate = float(line.split(":")[1].split('%')[0].strip())
            except:
                pass
        elif line.startswith("Avg Tokens:"):
            try:
                avg_tokens = float(line.split(":")[1].strip())
            except:
                pass
        elif line.startswith("Avg Time:"):
            try:
                avg_time = float(line.split(":")[1].strip().rstrip('s'))
            except:
                pass
    
    if num_examples == 0:
        return None
    
    return EvalRunResult(
        run_id=0,  # Will be set by caller
        accuracy=accuracy,
        format_rate=format_rate,
        syntax_rate=syntax_rate,
        avg_tokens=avg_tokens,
        avg_time=avg_time,
        total_time=total_time,
        num_examples=num_examples,
    )


def run_folio_evaluation(
    run_dir: str,
    model: str,
    limit: int = EVAL_LIMIT,
    unconstrained: bool = False,
) -> Optional[EvalRunResult]:
    """
    Run FOLIO evaluation and parse results.
    
    Args:
        run_dir: Path to CSD run directory (required even for unconstrained for env setup)
        model: Model name
        limit: Number of examples
        unconstrained: If True, run baseline without CSD constraints
    
    Returns EvalRunResult if successful, None otherwise.
    """
    cmd = [
        sys.executable, "-m", "evaluations.folio.cli",
        "--run-dir", run_dir,
        "--model", model,
        "--device", "cuda",
        "--limit", str(limit),
        "--max-steps", "1500",
        "--vocab-size", "3000",
    ]
    if unconstrained:
        cmd.append("--unconstrained")
    
    mode_str = "baseline (unconstrained)" if unconstrained else "CSD"
    print(f"    Running FOLIO {mode_str} evaluation (limit={limit})...")
    start_time = time.time()
    returncode, stdout, stderr = run_command(cmd, timeout=7200)  # 2 hour timeout
    total_time = time.time() - start_time
    
    if returncode != 0:
        print(f"    ✗ FOLIO evaluation failed")
        print(f"    stderr: {stderr[:500]}")
        return None
    
    # Parse results from stdout
    result = parse_folio_output(stdout, total_time)
    if result:
        print(f"    ✓ Accuracy: {result.accuracy:.1f}% | Structure: {result.format_rate:.1f}% | Time: {result.avg_time:.2f}s/ex")
    return result


def parse_folio_output(stdout: str, total_time: float) -> Optional[EvalRunResult]:
    """Parse FOLIO evaluation output to extract metrics."""
    lines = stdout.strip().split('\n')
    
    num_examples = 0
    accuracy = 0.0
    structure_rate = 0.0
    syntax_rate = 0.0
    avg_tokens = 0.0
    avg_time = 0.0
    
    for line in lines:
        line = line.strip()
        # Look for "Overall Accuracy: X.X% (N/M)"
        if "Overall Accuracy:" in line:
            try:
                parts = line.split(":")
                acc_part = parts[1].strip()
                accuracy = float(acc_part.split('%')[0].strip())
                # Extract total from (N/M)
                if '(' in acc_part and '/' in acc_part:
                    total_part = acc_part.split('(')[1].split('/')[1].split(')')[0]
                    num_examples = int(total_part)
            except:
                pass
        elif "Structure Validity:" in line:
            try:
                structure_rate = float(line.split(":")[1].split('%')[0].strip())
            except:
                pass
        elif "FOL Syntax Validity:" in line:
            try:
                syntax_rate = float(line.split(":")[1].split('%')[0].strip())
            except:
                pass
        elif "Avg tokens/example:" in line:
            try:
                avg_tokens = float(line.split(":")[1].strip())
            except:
                pass
        elif "Avg time/example:" in line:
            try:
                avg_time = float(line.split(":")[1].strip().rstrip('s'))
            except:
                pass
    
    if num_examples == 0:
        # Try to find it elsewhere
        for line in lines:
            if line.strip().startswith("Total:"):
                try:
                    num_examples = int(line.split(":")[1].strip())
                except:
                    pass
    
    if num_examples == 0:
        return None
    
    return EvalRunResult(
        run_id=0,  # Will be set by caller
        accuracy=accuracy,
        format_rate=structure_rate,
        syntax_rate=syntax_rate,
        avg_tokens=avg_tokens,
        avg_time=avg_time,
        total_time=total_time,
        num_examples=num_examples,
    )


def evaluate_model(
    model: str,
    gsm_csd_run_dirs: List[str],
    folio_csd_run_dirs: List[str],
    eval_limit: int = EVAL_LIMIT,
    num_eval_runs: int = NUM_EVAL_RUNS,
    run_baseline: bool = True,
) -> ModelResults:
    """
    Evaluate a model using pre-generated CSDs.
    
    Args:
        model: Model name/path
        gsm_csd_run_dirs: List of GSM-specific CSD run directories
        folio_csd_run_dirs: List of FOLIO-specific CSD run directories
        eval_limit: Number of examples per evaluation run
        num_eval_runs: Number of evaluation runs per CSD per dataset
        run_baseline: Whether to run baseline (unconstrained) evaluation for comparison
    
    Returns:
        ModelResults with all evaluation data
    """
    print(f"\n{'='*60}")
    print(f"Evaluating model: {model}")
    print(f"GSM CSDs: {len(gsm_csd_run_dirs)}, FOLIO CSDs: {len(folio_csd_run_dirs)}")
    print(f"Runs per CSD: {num_eval_runs}")
    print(f"Baseline evaluation: {'Yes' if run_baseline else 'No'}")
    print(f"{'='*60}")
    
    model_results = ModelResults(model_name=model)
    
    # Need at least one CSD run dir for environment setup (even for baseline)
    reference_run_dir = gsm_csd_run_dirs[0] if gsm_csd_run_dirs else (folio_csd_run_dirs[0] if folio_csd_run_dirs else None)
    
    # ==================== GSM Evaluation ====================
    print("\n--- GSM-Symbolic Evaluation ---")
    gsm_results = DatasetResults(dataset="gsm_symbolic")
    
    # Run baseline first (if enabled)
    if run_baseline and reference_run_dir:
        print("\n  [BASELINE] Running unconstrained baseline...")
        baseline_results = BaselineResults()
        
        for run_idx in range(num_eval_runs):
            print(f"  Baseline Run {run_idx + 1}/{num_eval_runs}:")
            result = run_gsm_evaluation(
                run_dir=reference_run_dir,
                model=model,
                limit=eval_limit,
                random_sample=True,
                unconstrained=True,
            )
            if result:
                result.run_id = run_idx + 1
                baseline_results.runs.append(result)
        
        if baseline_results.runs:
            print(f"  → Baseline avg accuracy: {baseline_results.avg_accuracy:.1f}% ± {baseline_results.std_accuracy:.1f}%")
            gsm_results.baseline_results = baseline_results
    
    # Run CSD evaluation (using GSM-specific CSDs)
    for csd_idx, run_dir in enumerate(gsm_csd_run_dirs):
        csd_id = f"gsm_csd_{csd_idx + 1}"
        print(f"\n  [CSD {csd_idx + 1}/{len(gsm_csd_run_dirs)}]: {Path(run_dir).name}")
        
        csd_result = CSDEvalResult(csd_id=csd_id, csd_run_dir=run_dir)
        
        for run_idx in range(num_eval_runs):
            print(f"  Run {run_idx + 1}/{num_eval_runs}:")
            result = run_gsm_evaluation(
                run_dir=run_dir,
                model=model,
                limit=eval_limit,
                random_sample=True,  # Random sample for variance
                unconstrained=False,
            )
            if result:
                result.run_id = run_idx + 1
                csd_result.runs.append(result)
        
        if csd_result.runs:
            print(f"  → CSD avg accuracy: {csd_result.avg_accuracy:.1f}% ± {csd_result.std_accuracy:.1f}%")
            gsm_results.csd_results.append(csd_result)
    
    # Print GSM comparison summary
    if gsm_results.baseline_results and gsm_results.best_csd:
        improvement = gsm_results.improvement_over_baseline
        print(f"\n  📊 GSM Summary: Baseline={gsm_results.baseline_results.avg_accuracy:.1f}% | "
              f"Best CSD={gsm_results.best_csd.avg_accuracy:.1f}% | "
              f"Improvement={improvement:+.1f}%")
    
    model_results.gsm_results = gsm_results
    
    # ==================== FOLIO Evaluation ====================
    print("\n--- FOLIO Evaluation ---")
    folio_results = DatasetResults(dataset="folio")
    
    # Use FOLIO CSD as reference for baseline if available
    folio_reference_dir = folio_csd_run_dirs[0] if folio_csd_run_dirs else reference_run_dir
    
    # Run baseline first (if enabled)
    if run_baseline and folio_reference_dir:
        print("\n  [BASELINE] Running unconstrained baseline...")
        baseline_results = BaselineResults()
        
        for run_idx in range(num_eval_runs):
            print(f"  Baseline Run {run_idx + 1}/{num_eval_runs}:")
            result = run_folio_evaluation(
                run_dir=folio_reference_dir,
                model=model,
                limit=eval_limit,
                unconstrained=True,
            )
            if result:
                result.run_id = run_idx + 1
                baseline_results.runs.append(result)
        
        if baseline_results.runs:
            print(f"  → Baseline avg accuracy: {baseline_results.avg_accuracy:.1f}% ± {baseline_results.std_accuracy:.1f}%")
            folio_results.baseline_results = baseline_results
    
    # Run CSD evaluation (using FOLIO-specific CSDs)
    for csd_idx, run_dir in enumerate(folio_csd_run_dirs):
        csd_id = f"folio_csd_{csd_idx + 1}"
        print(f"\n  [CSD {csd_idx + 1}/{len(folio_csd_run_dirs)}]: {Path(run_dir).name}")
        
        csd_result = CSDEvalResult(csd_id=csd_id, csd_run_dir=run_dir)
        
        for run_idx in range(num_eval_runs):
            print(f"  Run {run_idx + 1}/{num_eval_runs}:")
            result = run_folio_evaluation(
                run_dir=run_dir,
                model=model,
                limit=eval_limit,
                unconstrained=False,
            )
            if result:
                result.run_id = run_idx + 1
                csd_result.runs.append(result)
        
        if csd_result.runs:
            print(f"  → CSD avg accuracy: {csd_result.avg_accuracy:.1f}% ± {csd_result.std_accuracy:.1f}%")
            folio_results.csd_results.append(csd_result)
    
    # Print FOLIO comparison summary
    if folio_results.baseline_results and folio_results.best_csd:
        improvement = folio_results.improvement_over_baseline
        print(f"\n  📊 FOLIO Summary: Baseline={folio_results.baseline_results.avg_accuracy:.1f}% | "
              f"Best CSD={folio_results.best_csd.avg_accuracy:.1f}% | "
              f"Improvement={improvement:+.1f}%")
    
    model_results.folio_results = folio_results
    
    return model_results


def generate_csds_for_model(
    model: str,
    num_csds: int = NUM_CSDS,
    task_type: str = "gsm",  # "gsm" or "folio" or "both"
) -> List[str]:
    """
    Generate multiple CSDs for a model.
    
    Args:
        model: Model name/path
        num_csds: Number of CSDs to generate
        task_type: Type of task ("gsm", "folio", or "both")
    
    Returns:
        List of run directory paths
    """
    print(f"\n{'='*60}")
    print(f"Generating CSDs for: {model}")
    print(f"Number of CSDs: {num_csds}")
    print(f"{'='*60}")
    
    run_dirs = []
    
    for i in range(num_csds):
        print(f"\nGenerating CSD {i + 1}/{num_csds}...")
        
        # Use GSM task description (works for both GSM and FOLIO)
        # The CSD is grammar-agnostic - it's the evaluation that uses different grammars
        task_desc = GSM_TASK_DESC if task_type == "gsm" else FOLIO_TASK_DESC
        output_name = f"{task_type}_csd_{i + 1}"
        
        # Add slight temperature variation for diversity
        temp = 0.7 + (i * 0.1)  # 0.7, 0.8, 0.9
        
        run_dir = generate_csd(
            task=task_desc,
            output_name=output_name,
            model=model,
            dataset="gsm_symbolic" if task_type == "gsm" else "folio",
            temperature=temp,
            max_iterations=10,
        )
        
        if run_dir:
            run_dirs.append(run_dir)
        else:
            print(f"  ⚠ Failed to generate CSD {i + 1}")
    
    return run_dirs


def print_summary(all_results: List[ModelResults]):
    """Print a summary of all results."""
    print("\n")
    print("=" * 80)
    print("COMPREHENSIVE EVALUATION SUMMARY")
    print("=" * 80)
    
    for model_result in all_results:
        print(f"\n{'='*60}")
        print(f"Model: {model_result.model_name}")
        print(f"{'='*60}")
        
        if model_result.gsm_results:
            gsm = model_result.gsm_results
            print(f"\n  GSM-Symbolic:")
            
            # Baseline results
            if gsm.baseline_results:
                print(f"    Baseline (Unconstrained): {gsm.baseline_results.avg_accuracy:.1f}% ± {gsm.baseline_results.std_accuracy:.1f}%")
            
            # CSD results
            print(f"    CSD Overall Accuracy: {gsm.overall_avg_accuracy:.1f}% ± {gsm.overall_std_accuracy:.1f}%")
            if gsm.best_csd:
                print(f"    Best CSD: {gsm.best_csd.csd_id} ({gsm.best_csd.avg_accuracy:.1f}% ± {gsm.best_csd.std_accuracy:.1f}%)")
            
            # Improvement
            if gsm.improvement_over_baseline is not None:
                improvement = gsm.improvement_over_baseline
                improvement_pct = (improvement / gsm.baseline_results.avg_accuracy * 100) if gsm.baseline_results.avg_accuracy > 0 else 0
                print(f"    🚀 Improvement (Best CSD vs Baseline): {improvement:+.1f}% ({improvement_pct:+.1f}% relative)")
        
        if model_result.folio_results:
            folio = model_result.folio_results
            print(f"\n  FOLIO:")
            
            # Baseline results
            if folio.baseline_results:
                print(f"    Baseline (Unconstrained): {folio.baseline_results.avg_accuracy:.1f}% ± {folio.baseline_results.std_accuracy:.1f}%")
            
            # CSD results
            print(f"    CSD Overall Accuracy: {folio.overall_avg_accuracy:.1f}% ± {folio.overall_std_accuracy:.1f}%")
            if folio.best_csd:
                print(f"    Best CSD: {folio.best_csd.csd_id} ({folio.best_csd.avg_accuracy:.1f}% ± {folio.best_csd.std_accuracy:.1f}%)")
            
            # Improvement
            if folio.improvement_over_baseline is not None:
                improvement = folio.improvement_over_baseline
                improvement_pct = (improvement / folio.baseline_results.avg_accuracy * 100) if folio.baseline_results.avg_accuracy > 0 else 0
                print(f"    🚀 Improvement (Best CSD vs Baseline): {improvement:+.1f}% ({improvement_pct:+.1f}% relative)")


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive CSD Evaluation Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="Specific model to test (default: all models)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="comprehensive_results.json",
        help="Output JSON file for results"
    )
    parser.add_argument(
        "--num-csds",
        type=int,
        default=NUM_CSDS,
        help=f"Number of CSDs to generate per model (default: {NUM_CSDS})"
    )
    parser.add_argument(
        "--num-eval-runs",
        type=int,
        default=NUM_EVAL_RUNS,
        help=f"Number of evaluation runs per CSD (default: {NUM_EVAL_RUNS})"
    )
    parser.add_argument(
        "--eval-limit",
        type=int,
        default=EVAL_LIMIT,
        help=f"Number of examples per evaluation (default: {EVAL_LIMIT})"
    )
    parser.add_argument(
        "--skip-synthesis",
        action="store_true",
        help="Skip CSD generation, use existing CSDs from --csd-dirs"
    )
    parser.add_argument(
        "--csd-dirs",
        type=str,
        nargs="+",
        default=None,
        help="Existing CSD run directories to use for BOTH datasets (with --skip-synthesis)"
    )
    parser.add_argument(
        "--gsm-csd-dirs",
        type=str,
        nargs="+",
        default=None,
        help="Existing GSM-specific CSD run directories (with --skip-synthesis)"
    )
    parser.add_argument(
        "--folio-csd-dirs",
        type=str,
        nargs="+",
        default=None,
        help="Existing FOLIO-specific CSD run directories (with --skip-synthesis)"
    )
    parser.add_argument(
        "--gsm-only",
        action="store_true",
        help="Only run GSM evaluation"
    )
    parser.add_argument(
        "--folio-only",
        action="store_true",
        help="Only run FOLIO evaluation"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for evaluation (default: cuda)"
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip baseline (unconstrained) evaluation"
    )
    
    args = parser.parse_args()
    
    # Determine which models to test
    models_to_test = [args.model] if args.model else MODELS
    
    run_baseline = not args.skip_baseline
    
    print("=" * 80)
    print("COMPREHENSIVE CSD EVALUATION")
    print("=" * 80)
    print(f"Models: {len(models_to_test)}")
    print(f"CSDs per dataset per model: {args.num_csds} (3 GSM + 3 FOLIO = 6 total)")
    print(f"Eval runs per CSD: {args.num_eval_runs}")
    print(f"Examples per eval: {args.eval_limit}")
    print(f"Baseline evaluation: {'Yes' if run_baseline else 'No (skipped)'}")
    baseline_runs = args.num_eval_runs * 2 if run_baseline else 0  # GSM + FOLIO baseline
    csd_runs = args.num_csds * args.num_eval_runs * 2  # GSM CSDs + FOLIO CSDs
    print(f"Total evaluations per model: {baseline_runs + csd_runs} ({baseline_runs} baseline + {csd_runs} CSD)")
    print(f"Output file: {args.output}")
    print("=" * 80)
    
    all_results = []
    start_time = datetime.now()
    
    for model in models_to_test:
        print(f"\n\n{'#'*80}")
        print(f"# Processing: {model}")
        print(f"{'#'*80}")
        
        # Generate or use existing CSDs
        if args.skip_synthesis:
            # Check for dataset-specific CSD dirs first, then fall back to shared --csd-dirs
            gsm_csd_run_dirs = args.gsm_csd_dirs or args.csd_dirs
            folio_csd_run_dirs = args.folio_csd_dirs or args.csd_dirs
            
            if not gsm_csd_run_dirs and not folio_csd_run_dirs:
                print("Error: --skip-synthesis requires --csd-dirs, --gsm-csd-dirs, or --folio-csd-dirs")
                sys.exit(1)
            
            gsm_csd_run_dirs = gsm_csd_run_dirs or []
            folio_csd_run_dirs = folio_csd_run_dirs or []
        else:
            # Generate GSM-specific CSDs
            print(f"\n  Generating {args.num_csds} GSM-specific CSDs...")
            gsm_csd_run_dirs = generate_csds_for_model(
                model=model,
                num_csds=args.num_csds,
                task_type="gsm",
            )
            
            # Generate FOLIO-specific CSDs
            print(f"\n  Generating {args.num_csds} FOLIO-specific CSDs...")
            folio_csd_run_dirs = generate_csds_for_model(
                model=model,
                num_csds=args.num_csds,
                task_type="folio",
            )
            
            if not gsm_csd_run_dirs and not folio_csd_run_dirs:
                print(f"⚠ No CSDs generated for {model}, skipping evaluation")
                continue
        
        # Evaluate model
        model_results = evaluate_model(
            model=model,
            gsm_csd_run_dirs=gsm_csd_run_dirs,
            folio_csd_run_dirs=folio_csd_run_dirs,
            eval_limit=args.eval_limit,
            num_eval_runs=args.num_eval_runs,
            run_baseline=run_baseline,
        )
        
        all_results.append(model_results)
        
        # Save intermediate results
        output_data = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "num_csds": args.num_csds,
                "num_eval_runs": args.num_eval_runs,
                "eval_limit": args.eval_limit,
                "run_baseline": run_baseline,
            },
            "results": [r.to_dict() for r in all_results],
        }
        
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\n✓ Intermediate results saved to {args.output}")
    
    # Print final summary
    print_summary(all_results)
    
    # Save final results
    end_time = datetime.now()
    output_data = {
        "timestamp": end_time.isoformat(),
        "duration_seconds": (end_time - start_time).total_seconds(),
        "config": {
            "num_csds": args.num_csds,
            "num_eval_runs": args.num_eval_runs,
            "eval_limit": args.eval_limit,
            "run_baseline": run_baseline,
            "models": models_to_test,
        },
        "results": [r.to_dict() for r in all_results],
    }
    
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n✓ Final results saved to {args.output}")
    print(f"Total duration: {(end_time - start_time).total_seconds() / 3600:.1f} hours")


if __name__ == "__main__":
    main()
