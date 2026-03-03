"""
Evaluation module for synthesis feedback loop.

Provides quick evaluation of synthesized CSD strategies on dataset samples
to enable feedback-driven refinement based on actual performance metrics.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class EvaluationResult:
    """
    Result of evaluating a CSD strategy on a dataset sample.

    Contains metrics and sample outputs for feedback to the generator.
    """
    success: bool
    accuracy: float  # 0.0 to 1.0
    format_rate: float  # 0.0 to 1.0
    syntax_rate: float  # 0.0 to 1.0
    num_examples: int
    num_correct: int
    total_time_seconds: float

    # Sample outputs for feedback (question, expected, actual, is_correct)
    sample_outputs: List[Dict[str, Any]] = field(default_factory=list)

    # Error information if evaluation failed
    error: Optional[str] = None

    def meets_threshold(
        self,
        min_accuracy: float = 0.0,
        min_format_rate: float = 0.0,
        min_syntax_rate: float = 0.0,
    ) -> bool:
        """Check if ALL individual examples meet the specified thresholds."""
        if not self.sample_outputs:
            return False
        for sample in self.sample_outputs:
            ex_accuracy = 1.0 if sample.get("is_correct", False) else 0.0
            ex_format = 1.0 if sample.get("is_valid_format", False) else 0.0
            ex_syntax = sample.get("syntax_rate", 0.0)
            if ex_accuracy < min_accuracy or ex_format < min_format_rate or ex_syntax < min_syntax_rate:
                return False
        return True

    def get_feedback_summary(self) -> str:
        """Generate a summary for feedback to the generator."""
        lines = [
            f"Evaluation Results ({self.num_examples} examples):",
            f"  Accuracy: {self.accuracy:.1%} ({self.num_correct}/{self.num_examples})",
            f"  Format Rate: {self.format_rate:.1%}",
            f"  Syntax Rate: {self.syntax_rate:.1%}",
            f"  Total Time: {self.total_time_seconds:.2f}s",
        ]

        if self.sample_outputs:
            lines.append("\nSample Failures:")
            failures = [s for s in self.sample_outputs if not s.get("is_correct", False)]
            for i, sample in enumerate(failures[:3]):  # Show up to 3 failures
                lines.append(f"\n  Example {i+1}:")
                lines.append(f"    Question: {sample.get('question', 'N/A')[:100]}...")
                lines.append(f"    Expected: {sample.get('expected', 'N/A')}")
                lines.append(f"    Got: {sample.get('actual', 'N/A')}")
                if sample.get("error"):
                    lines.append(f"    Error: {sample.get('error')}")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "success": self.success,
            "accuracy": self.accuracy,
            "format_rate": self.format_rate,
            "syntax_rate": self.syntax_rate,
            "num_examples": self.num_examples,
            "num_correct": self.num_correct,
            "total_time_seconds": self.total_time_seconds,
            "error": self.error,
            "sample_outputs": self.sample_outputs,
        }


class Evaluator:
    """
    Evaluates synthesized CSD strategies on dataset samples.

    Supports both GSM-Symbolic and FOLIO datasets with their respective
    evaluation metrics and syntax validation.
    """

    def __init__(
        self,
        dataset_name: str = "gsm_symbolic",
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        device: str = "cuda",
        sample_size: int = 10,
        max_steps: int = 150,
    ):
        """
        Initialize the evaluator.

        Args:
            dataset_name: Dataset to evaluate on ("gsm_symbolic" or "folio")
            model_name: HuggingFace model for generation
            device: Device to run on ("cuda", "mps", "cpu")
            sample_size: Number of examples to evaluate on
            max_steps: Maximum generation steps per example
        """
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.device = device
        self.sample_size = sample_size
        self.max_steps = max_steps

        # Lazy-loaded components
        self._dataset = None
        self._env = None
        self._grammar_file = None

    def _get_grammar_file(self) -> Path:
        """Get the grammar file path for the dataset."""
        if self._grammar_file is None:
            grammars_dir = Path(__file__).parent.parent / "grammars"
            if self.dataset_name == "gsm_symbolic":
                self._grammar_file = grammars_dir / "gsm.lark"
            elif self.dataset_name == "folio":
                self._grammar_file = grammars_dir / "folio.lark"
            else:
                raise ValueError(f"Unknown dataset: {self.dataset_name}")
        return self._grammar_file

    def _load_dataset_sample(self) -> list:
        """Load a sample of the dataset for evaluation."""
        if self._dataset is not None:
            return self._dataset

        if self.dataset_name == "gsm_symbolic":
            from evaluations.gsm_symbolic.dataset import load_gsm_symbolic
            ds = load_gsm_symbolic(
                config="main",
                split="test",
                limit=self.sample_size,
                random_sample=True,
            )
            self._dataset = list(ds)
        elif self.dataset_name == "folio":
            from evaluations.folio.dataset import load_folio
            ds = load_folio(
                split="validation",
                limit=self.sample_size,
                random_sample=True,
            )
            self._dataset = list(ds)
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")

        return self._dataset

    def _setup_environment(self, compiled_module_path: Path) -> Dict[str, Any]:
        """
        Set up the Dafny environment for evaluation.

        Args:
            compiled_module_path: Path to the compiled CSD module

        Returns:
            Environment dict with loaded modules
        """
        run_dir = compiled_module_path.parent
        if run_dir.name == "generated_csd":
            run_dir = run_dir.parent

        if self.dataset_name == "gsm_symbolic":
            from evaluations.gsm_symbolic.environment import setup_dafny_environment
        else:
            from evaluations.folio.environment import setup_dafny_environment

        return setup_dafny_environment(
            run_dir=run_dir,
            model_name=self.model_name,
            device=self.device,
            grammar_file=self._get_grammar_file(),
        )

    def _extract_constrained_content(self, output: str) -> List[str]:
        """Extract content within << >> delimiters."""
        return re.findall(r"<<\s*([^<>]+?)\s*>>", output)

    def _parse_variable_assignments(self, text: str) -> dict:
        """Parse variable assignments from text like 'a = 5', 'n1 = 72.5', etc."""
        assignments = {}
        pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*([-+]?\d+(?:\.\d+)?)\b'
        for match in re.finditer(pattern, text):
            var_name = match.group(1)
            try:
                assignments[var_name] = float(match.group(2))
            except ValueError:
                pass
        return assignments

    def _safe_eval_arithmetic(self, expr: str) -> Optional[float]:
        """Safely evaluate a numeric arithmetic expression using AST (no eval())."""
        import ast
        import operator as op

        ops = {
            ast.Add: op.add, ast.Sub: op.sub,
            ast.Mult: op.mul, ast.Div: op.truediv,
            ast.FloorDiv: op.floordiv, ast.Mod: op.mod,
            ast.USub: op.neg, ast.UAdd: op.pos,
        }

        def _eval(node):
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                return float(node.value)
            elif isinstance(node, ast.Num):  # Python 3.7 compat
                return float(node.n)
            elif isinstance(node, ast.BinOp) and type(node.op) in ops:
                return ops[type(node.op)](_eval(node.left), _eval(node.right))
            elif isinstance(node, ast.UnaryOp) and type(node.op) in ops:
                return ops[type(node.op)](_eval(node.operand))
            elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                  and node.func.id == 'int' and len(node.args) == 1):
                return float(int(_eval(node.args[0])))
            else:
                raise ValueError(f"Unsupported node: {type(node)}")

        try:
            tree = ast.parse(expr.strip(), mode='eval')
            return _eval(tree.body)
        except Exception:
            return None

    def _evaluate_symbolic_expression(self, expr: str, var_values: dict) -> Optional[float]:
        """Substitute variable values into a symbolic expression and evaluate."""
        substituted = expr
        # Substitute longest names first to avoid partial replacement (n10 before n1)
        for var in sorted(var_values.keys(), key=len, reverse=True):
            substituted = re.sub(r'\b' + re.escape(var) + r'\b',
                                 str(var_values[var]), substituted)
        # If alphabetic chars remain, some variables were unresolved
        if re.search(r'[a-zA-Z_]', substituted):
            return None
        return self._safe_eval_arithmetic(substituted)

    def _extract_answer_gsm(self, output: str) -> Optional[str]:
        """Extract numeric answer from GSM-Symbolic output within << >> delimiters."""
        matches = self._extract_constrained_content(output)
        if not matches:
            return None

        last_match = matches[-1].strip()

        # Case 1: expression contains "=" — take the part after "=" (e.g. "a + b = 8")
        if "=" in last_match:
            answer_part = last_match.split("=")[-1].strip()
            num_match = re.search(r"[-+]?\d*\.?\d+", answer_part)
            if num_match:
                return num_match.group()

        # Case 2: purely numeric expression — evaluate directly (e.g. "5 + 3")
        if not re.search(r'[a-zA-Z_]', last_match):
            result = self._safe_eval_arithmetic(last_match)
            if result is not None:
                val = int(result) if result == int(result) else result
                return str(val)

        # Case 3: symbolic expression — parse variable assignments from surrounding text
        # and substitute in (e.g. "a + b" with "a = 5, b = 3" defined earlier)
        var_values = self._parse_variable_assignments(output)
        if var_values:
            result = self._evaluate_symbolic_expression(last_match, var_values)
            if result is not None:
                val = int(result) if result == int(result) else result
                return str(val)

        return None

    def _fol_keyword_to_unicode(self, text: str) -> str:
        """Convert {keyword} FOL syntax from grammar to Unicode symbols for Prover9."""
        replacements = {
            "{forall}": "∀",
            "{exists}": "∃",
            "{and}": "∧",
            "{or}": "∨",
            "{xor}": "⊕",
            "{not}": "¬",
            "{implies}": "→",
            "{iff}": "↔",
        }
        for keyword, symbol in replacements.items():
            text = text.replace(keyword, symbol)
        return text

    def _extract_answer_folio(self, output: str, example: Optional[Any] = None) -> Optional[str]:
        """
        Extract FOL answer from FOLIO output using Prover9 solver.

        Extracts FOL formulas from constrained << >> segments, converts
        {keyword} syntax to Unicode, builds a Prover9 logic program, and
        runs the solver to determine True/False/Unknown.

        Falls back to keyword matching if the solver fails.
        """
        segments = self._extract_constrained_content(output)
        if not segments:
            return self._extract_answer_folio_fallback(output)

        # Convert {keyword} syntax to Unicode FOL symbols
        fol_segments = [self._fol_keyword_to_unicode(s.strip()) for s in segments]

        # Heuristic: all segments except the last are premises, last is conclusion
        if len(fol_segments) >= 2:
            premises = fol_segments[:-1]
            conclusion = fol_segments[-1]
        elif len(fol_segments) == 1:
            # Single segment — treat as conclusion, no FOL premises
            premises = []
            conclusion = fol_segments[0]
        else:
            return self._extract_answer_folio_fallback(output)

        # Build Prover9 logic program
        logic_lines = ["Premises:"]
        for p in premises:
            logic_lines.append(f"{p} ::: premise")
        logic_lines.append("Conclusion:")
        logic_lines.append(f"{conclusion} ::: conclusion")
        logic_program = "\n".join(logic_lines)

        try:
            from symbolic_solvers.fol_solver.prover9_solver import FOL_Prover9_Program

            program = FOL_Prover9_Program(logic_program, dataset_name="FOLIO")
            if not program.flag:
                return self._extract_answer_folio_fallback(output)

            answer, error = program.execute_program()
            if answer in ("True", "False", "Unknown"):
                return answer
        except Exception:
            pass

        return self._extract_answer_folio_fallback(output)

    def _extract_answer_folio_fallback(self, output: str) -> Optional[str]:
        """Fallback: extract FOL answer via keyword matching."""
        output_lower = output.lower()
        if "true" in output_lower:
            return "True"
        elif "false" in output_lower:
            return "False"
        elif "unknown" in output_lower or "uncertain" in output_lower:
            return "Unknown"
        return None

    def _answers_match(self, actual: Optional[str], expected: str) -> bool:
        """Check if actual and expected answers match, normalizing Uncertain/Unknown."""
        if actual is None:
            return False
        a = str(actual).strip().lower()
        e = str(expected).strip().lower()
        # Normalize "uncertain" and "unknown" to be equivalent
        if a in ("uncertain", "unknown"):
            a = "unknown"
        if e in ("uncertain", "unknown"):
            e = "unknown"
        return a == e

    def _get_expected_answer(self, example: dict) -> str:
        """Get the expected answer from a dataset example."""
        if self.dataset_name == "gsm_symbolic":
            answer_str = example.get("answer", "")
            match = re.search(r"####\s*([-+]?\d*\.?\d+)", answer_str)
            if match:
                return match.group(1)
            return answer_str
        else:
            return example.get("label", "Unknown")

    def _format_prompt(self, example: dict) -> str:
        """Format a dataset example as a prompt."""
        if self.dataset_name == "gsm_symbolic":
            question = example.get("question", "")
            return (
                "Solve the following math problem step by step. "
                "Assign a single-letter variable to each quantity and state its numeric value. "
                "Write each computation step as a SHORT expression inside << >> delimiters — "
                "one step per << >> window, closing >> before starting the next step.\n\n"
                "Example:\n"
                "Q: A store sells pens for $3 each and notebooks for $8 each. "
                "Bob buys 4 pens and 2 notebooks. How much does he spend?\n"
                "A: Let p = 3, n = 8, a = 4, b = 2.\n"
                "Pen total = <<a * p>>\n"
                "Notebook total = <<b * n>>\n"
                "Total spent = <<a * p + b * n>>\n"
                "The answer is a * p + b * n.\n\n"
                f"Q: {question}\nA:"
            )
        else:
            premises = example.get("premises", [])
            conclusion = example.get("conclusion", "")
            premises_str = "\n".join(f"- {p}" for p in premises)
            return f"Given the following premises:\n{premises_str}\n\nDetermine if the following conclusion is True, False, or Unknown:\n{conclusion}\n\nAnswer:"

    def _check_format_validity(self, output: str) -> bool:
        """Check if the output has valid format with << >> delimiters."""
        return "<<" in output and ">>" in output

    def _check_syntax_validity(self, output: str) -> Tuple[bool, List[Tuple[str, bool]]]:
        """
        Check if constrained segments have valid syntax.

        Returns:
            Tuple of (all_valid, list of (segment, is_valid) tuples)
        """
        from lark import Lark
        from lark.exceptions import LarkError

        segments: List[Tuple[str, bool]] = []
        matches = self._extract_constrained_content(output)

        if not matches:
            return True, []

        grammar_text = self._get_grammar_file().read_text()
        try:
            parser = Lark(grammar_text, start="start", parser="lalr")
            for match in matches:
                try:
                    parser.parse(match.strip())
                    segments.append((match, True))
                except LarkError:
                    segments.append((match, False))
        except Exception:
            return True, [(m, True) for m in matches]

        all_valid = all(is_valid for _, is_valid in segments) if segments else True
        return all_valid, segments

    def evaluate_sample(
        self,
        compiled_module_path: Path,
        sample_size: Optional[int] = None,
    ) -> EvaluationResult:
        """
        Evaluate the compiled CSD on a sample of the dataset.

        Args:
            compiled_module_path: Path to the compiled GeneratedCSD.py module
            sample_size: Number of examples to evaluate (overrides init value)

        Returns:
            EvaluationResult with metrics and sample outputs
        """
        if sample_size is not None:
            self.sample_size = sample_size

        # Always re-sample so each iteration gets a fresh random example
        self._dataset = None

        start_time = time.time()
        sample_outputs: List[Dict[str, Any]] = []

        try:
            dataset = self._load_dataset_sample()
            env = self._setup_environment(compiled_module_path)

            if self.dataset_name == "gsm_symbolic":
                from evaluations.gsm_symbolic.generation import run_crane_csd
            else:
                from evaluations.folio.generation import run_crane_csd

            num_correct = 0
            num_valid_format = 0
            num_valid_syntax = 0
            total_segments = 0

            for i, example in enumerate(dataset):
                prompt = self._format_prompt(example)
                expected = self._get_expected_answer(example)

                try:
                    output_text, token_count, gen_time, _ = run_crane_csd(
                        env=env,
                        prompt_text=prompt,
                        max_steps=self.max_steps,
                        grammar_file=self._get_grammar_file(),
                    )

                    if self.dataset_name == "gsm_symbolic":
                        actual = self._extract_answer_gsm(output_text)
                    else:
                        actual = self._extract_answer_folio(output_text, example=example)

                    is_correct = self._answers_match(actual, expected)
                    if is_correct:
                        num_correct += 1

                    is_valid_format = self._check_format_validity(output_text)
                    if is_valid_format:
                        num_valid_format += 1

                    all_valid_syntax, segments = self._check_syntax_validity(output_text)
                    total_segments += len(segments)
                    example_valid_segs = sum(1 for _, v in segments if v)
                    num_valid_syntax += example_valid_segs
                    example_syntax_rate = example_valid_segs / len(segments) if segments else 0.0

                    sample_outputs.append({
                        "question": example.get("question", str(example.get("premises", "")))[:200],
                        "expected": expected,
                        "actual": actual or output_text[:100],
                        "full_output": output_text,
                        "is_correct": is_correct,
                        "is_valid_format": is_valid_format,
                        "is_syntax_valid": all_valid_syntax,
                        "syntax_rate": example_syntax_rate,
                        "token_count": token_count,
                        "time_seconds": gen_time,
                    })

                except Exception as e:
                    sample_outputs.append({
                        "question": example.get("question", str(example.get("premises", "")))[:200],
                        "expected": expected,
                        "actual": None,
                        "is_correct": False,
                        "is_valid_format": False,
                        "is_syntax_valid": False,
                        "syntax_rate": 0.0,
                        "error": str(e),
                    })

            total_time = time.time() - start_time
            num_examples = len(dataset)

            return EvaluationResult(
                success=True,
                accuracy=num_correct / max(1, num_examples),
                format_rate=num_valid_format / max(1, num_examples),
                syntax_rate=num_valid_syntax / total_segments if total_segments > 0 else 0.0,
                num_examples=num_examples,
                num_correct=num_correct,
                total_time_seconds=total_time,
                sample_outputs=sample_outputs,
            )

        except Exception as e:
            return EvaluationResult(
                success=False,
                accuracy=0.0,
                format_rate=0.0,
                syntax_rate=0.0,
                num_examples=0,
                num_correct=0,
                total_time_seconds=time.time() - start_time,
                error=str(e),
                sample_outputs=sample_outputs,
            )
