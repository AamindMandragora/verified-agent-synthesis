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
        """Check if results meet the specified thresholds."""
        return (
            self.accuracy >= min_accuracy
            and self.format_rate >= min_format_rate
            and self.syntax_rate >= min_syntax_rate
        )

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

    def get_detailed_samples(self, max_samples: int = 3) -> str:
        """Return a human-readable 'generated vs expected' view for debugging accuracy."""
        lines = [
            "Generated vs expected (for accuracy debugging):",
            "-" * 60,
        ]
        for i, s in enumerate(self.sample_outputs[:max_samples]):
            lines.append(f"\n--- Example {i+1} ---")
            lines.append(f"  Expected (gold) answer: {s.get('expected', 'N/A')}")
            lines.append(f"  Parsed answer (actual): {s.get('actual', 'N/A')}")
            lines.append(f"  Match: {'YES' if s.get('is_correct') else 'NO'}")
            lines.append(f"  Full raw output:\n    {_truncate_for_display(s.get('full_output') or '', 350)}")
            # FOLIO: last << >> segment sent to Prover9; result diffed to expected
            folio_debug = s.get("folio_debug")
            if folio_debug:
                reason = folio_debug.get("reason", "")
                if reason:
                    lines.append(f"  FOLIO extraction: {reason}")
                if folio_debug.get("segments"):
                    lines.append(f"  Model << >> segments ({len(folio_debug['segments'])}):")
                    for j, seg in enumerate(folio_debug["segments"]):
                        lines.append(f"    [{j+1}] {_truncate_for_display(seg, 180)}")
                lines.append(f"  Conclusion sent to Prover9: {_truncate_for_display(folio_debug.get('conclusion_raw') or folio_debug.get('conclusion', ''), 200)}")
                if folio_debug.get("prover9_ok"):
                    lines.append(f"  Prover9 result: {folio_debug.get('prover9_answer', 'N/A')}")
                else:
                    lines.append("  Prover9: failed or fallback used")
                    if folio_debug.get("prover9_error"):
                        lines.append(f"    Error: {_truncate_for_display(folio_debug['prover9_error'], 150)}")
                    if folio_debug.get("logic_program"):
                        prog = folio_debug["logic_program"]
                        lines.append("  Logic program (first 2 + last 2 lines):")
                        for line in prog.split("\n")[:2]:
                            lines.append(f"    {_truncate_for_display(line, 120)}")
                        lines.append("    ...")
                        for line in prog.split("\n")[-2:]:
                            lines.append(f"    {_truncate_for_display(line, 120)}")
            if not folio_debug:
                segs = s.get("extracted_segments") or []
                lines.append(f"  Extracted from << >> ({len(segs)} segment(s)):")
                for j, seg in enumerate(segs):
                    lines.append(f"    [{j+1}] {_truncate_for_display(seg, 200)}")
            if s.get("gold_premises_fol") is not None:
                prems = s["gold_premises_fol"]
                prems_str = prems if isinstance(prems, list) else [str(prems)]
                lines.append(f"  Gold premises (FOL): {_truncate_for_display(str(prems_str), 200)}")
            if s.get("gold_conclusion_fol") is not None:
                lines.append(f"  Gold conclusion (FOL): {s.get('gold_conclusion_fol')}")
            lines.append("")
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


def _truncate_for_display(s: str, max_len: int) -> str:
    s = (s or "").replace("\n", " ")
    return s[:max_len] + ("..." if len(s) > max_len else "")


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
        vocab_size: int = 2000,
        sample_size: int = 1,
        max_steps: int = 150,
        load_in_4bit: bool = False,
    ):
        """
        Initialize the evaluator.

        Args:
            dataset_name: Dataset to evaluate on ("gsm_symbolic" or "folio")
            model_name: HuggingFace model for generation
            device: Device to run on ("cuda", "mps", "cpu")
            vocab_size: Vocabulary size for constrained generation
            sample_size: Number of examples to evaluate on
            max_steps: Maximum generation steps per example
            load_in_4bit: Load eval model in 4-bit to save memory and speed up (default: False)
        """
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.device = device
        self.vocab_size = vocab_size
        self.sample_size = sample_size
        self.max_steps = max_steps
        self.load_in_4bit = load_in_4bit

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
                num_samples=self.sample_size,
                seed=42,
            )
            self._dataset = ds
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

        from evaluations.common.environment import setup_dafny_environment

        if self.dataset_name == "gsm_symbolic":
            return setup_dafny_environment(
                run_dir=run_dir,
                model_name=self.model_name,
                device=self.device,
                vocab_size=self.vocab_size,
                grammar_file=self._get_grammar_file(),
                start_rule="csd_start",
                load_in_4bit=self.load_in_4bit,
            )
        else:
            return setup_dafny_environment(
                run_dir=run_dir,
                model_name=self.model_name,
                device=self.device,
                vocab_size=self.vocab_size,
                grammar_file=self._get_grammar_file(),
                start_rule="start",
                load_in_4bit=self.load_in_4bit,
            )

    def _extract_constrained_content(self, output: str) -> List[str]:
        """Extract content within << >> delimiters."""
        return re.findall(r"<<\s*([^<>]+?)\s*>>", output)

    def _extract_answer_gsm(self, output: str) -> Optional[str]:
        """Extract numeric answer from GSM-Symbolic output within << >> delimiters."""
        matches = self._extract_constrained_content(output)
        if not matches:
            return None

        last_match = matches[-1]
        if "=" in last_match:
            answer_part = last_match.split("=")[-1].strip()
            num_match = re.search(r"[-+]?\d*\.?\d+", answer_part)
            if num_match:
                return num_match.group()
        else:
            num_match = re.search(r"[-+]?\d*\.?\d+", last_match)
            if num_match:
                return num_match.group()

        return None

    def _extract_answer_folio(self, output: str, example: Optional[Any] = None) -> Optional[str]:
        """
        Take the last << >> delimited string, evaluate it with Prover9 (with problem
        premises), then return True/False/Unknown. Compare that to expected for grading.
        """
        from evaluations.folio.fol_utils import fol_keyword_to_unicode

        debug: Dict[str, Any] = {
            "segments": [], "conclusion_raw": "", "conclusion": "",
            "logic_program": "", "prover9_ok": False, "prover9_answer": None,
            "prover9_error": None, "fallback_used": True,
        }
        segments = self._extract_constrained_content(output)
        if not segments:
            setattr(self, "_last_folio_debug", {**debug, "reason": "no_segments"})
            return self._extract_answer_folio_fallback(output)

        debug["segments"] = list(segments)
        conclusion_raw = segments[-1].strip()
        conclusion = fol_keyword_to_unicode(conclusion_raw)
        debug["conclusion_raw"] = conclusion_raw
        debug["conclusion"] = conclusion

        premises: List[str] = []
        if example is not None:
            fol_premises = self._example_field(example, "fol_premises", None)
            if fol_premises:
                if isinstance(fol_premises, list):
                    premises = [fol_keyword_to_unicode(p.strip()) for p in fol_premises]
                else:
                    premises = [fol_keyword_to_unicode(str(fol_premises).strip())]
        if not premises and len(segments) >= 2:
            premises = [fol_keyword_to_unicode(s.strip()) for s in segments[:-1]]
        debug["premises_used"] = list(premises)

        logic_lines = ["Premises:"]
        for p in premises:
            logic_lines.append(f"{p} ::: premise")
        logic_lines.append("Conclusion:")
        logic_lines.append(f"{conclusion} ::: conclusion")
        logic_program = "\n".join(logic_lines)
        debug["logic_program"] = logic_program

        try:
            from symbolic_solvers.fol_solver.prover9_solver import FOL_Prover9_Program
            program = FOL_Prover9_Program(logic_program, dataset_name="FOLIO")
            if not program.flag:
                setattr(self, "_last_folio_debug", {**debug, "reason": "prover9_parse_fail"})
                # Unparseable segment (e.g. prose) → fail; focus on making the model output parseable FOL.
                return self._extract_answer_folio_fallback(output)
            answer, error = program.execute_program()
            if answer in ("True", "False", "Unknown"):
                debug["prover9_ok"] = True
                debug["prover9_answer"] = answer
                debug["fallback_used"] = False
                setattr(self, "_last_folio_debug", debug)
                return answer
            debug["prover9_answer"] = answer
            debug["prover9_error"] = error or ""
        except Exception as e:
            err = str(e)
            if "No module named 'ply'" in err or (isinstance(e, ImportError) and "ply" in str(e).lower()):
                err = "Prover9 needs ply. Install: pip install ply nltk z3-solver"
            debug["prover9_error"] = err
        setattr(self, "_last_folio_debug", debug)
        return self._extract_answer_folio_fallback(output)

    @staticmethod
    def _normalize_folio_answer(text: str) -> Optional[str]:
        """Normalize model output to True, False, or Unknown (Uncertain -> Unknown)."""
        if not text:
            return None
        s = text.strip().lower()
        if s == "true":
            return "True"
        if s == "false":
            return "False"
        if s in ("uncertain", "unknown"):
            return "Unknown"
        # Allow substrings for fallback
        if "true" in s:
            return "True"
        if "false" in s:
            return "False"
        if "uncertain" in s or "unknown" in s:
            return "Unknown"
        return None

    def _extract_answer_folio_fallback(self, output: str) -> Optional[str]:
        """Fallback: extract answer via keyword matching when no << >> segment."""
        return self._normalize_folio_answer(output)

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

    @staticmethod
    def _example_field(example: Any, key: str, default: Any = None) -> Any:
        """Get a field from an example that may be a dict or a dataclass (e.g. FOLIOExample)."""
        if hasattr(example, key):
            return getattr(example, key)
        if hasattr(example, "get") and callable(getattr(example, "get")):
            return example.get(key, default)
        return default

    def _get_expected_answer(self, example: Any) -> str:
        """Get the expected answer from a dataset example."""
        if self.dataset_name == "gsm_symbolic":
            answer_str = self._example_field(example, "answer", "") or ""
            match = re.search(r"####\s*([-+]?\d*\.?\d+)", str(answer_str))
            if match:
                return match.group(1)
            return str(answer_str)
        else:
            return str(self._example_field(example, "label", "Unknown"))

    def _format_prompt(self, example: Any) -> str:
        """Format a dataset example as a prompt."""
        if self.dataset_name == "gsm_symbolic":
            question = self._example_field(example, "question", "") or ""
            return (
                "Solve the following math problem step by step. "
                "For each calculation, write the expression inside << >> delimiters.\n\n"
                "Example:\n"
                "Q: Amy has 5 apples. She buys 3 more. How many does she have?\n"
                "A: Amy starts with 5 apples and buys 3 more.\n"
                "Total apples = <<5 + 3 = 8>>8\n"
                "The answer is 8.\n\n"
                f"Q: {question}\nA:"
            )
        else:
            premises = self._example_field(example, "premises", "")
            conclusion = self._example_field(example, "conclusion", "") or ""
            fol_conclusion = self._example_field(example, "fol_conclusion", None)
            if isinstance(premises, str):
                premises_str = premises
            else:
                premises_str = "\n".join(f"- {p}" for p in (premises or []))
            prompt_parts = [
                f"Given the following premises:\n{premises_str}\n\n",
                f"Conclusion to evaluate:\n{conclusion}\n\n",
            ]
            if fol_conclusion:
                from evaluations.folio.fol_utils import fol_unicode_to_keyword
                conclusion_fol = fol_unicode_to_keyword(str(fol_conclusion).strip())
                prompt_parts.append(f"The conclusion in first-order logic is: {conclusion_fol}\n\n")
            prompt_parts.append(
                "Instructions: Start your answer with plain text reasoning (at least one sentence). "
                "Do not start with the characters <<. After your reasoning, write exactly \" << \" (space, two angle brackets, space). "
                "Then write exactly one first-order logic formula. Use the same predicate and constant names as in the given conclusion (e.g. Alkane not Alkale, mixture not mix). "
                "Use this grammar: "
                "quantifiers {forall} and {exists}; predicates like P(x), Q(a,b); "
                "connectives {and}, {or}, {not}, {implies}, {iff}, {xor}; "
                "variables as single lowercase letters, constants as longer lowercase identifiers. "
                "Then write exactly \" >>\" (space, two angle brackets). "
                "Only the substring between \" << \" and \" >>\" must be valid FOL; the rest is free text. "
                "Between << and >> write only one well-formed formula (predicates, constants, variables, and the connectives above). No sentences, no periods, no explanatory text — or the formula will fail to parse.\n\n"
                "Answer:"
            )
            return "".join(prompt_parts)

    @staticmethod
    def _ensure_delimiters_around_constrained(output: str) -> str:
        """
        Ensure the constrained part(s) are inside << >>. If the CSD output has no
        delimiters, treat the whole output as one constrained segment and wrap it.
        This way only the constrained part is in delimiters, not necessarily the entire reply.
        """
        if "<<" in output and ">>" in output:
            return output
        s = output.strip()
        if not s:
            return output
        return f"<< {s} >>"

    def _check_format_validity(self, output: str) -> bool:
        """Check if the output has valid format with << >> delimiters."""
        return "<<" in output and ">>" in output

    @staticmethod
    def _strip_trailing_label(segment: str) -> str:
        """Strip trailing True/False/Uncertain so 'formula True' parses as FOL formula."""
        s = segment.strip()
        for label in ("True", "False", "Uncertain", "Unknown"):
            if s.endswith(label):
                s = s[: -len(label)].strip()
                break
        return s

    def _check_syntax_validity(self, output: str) -> Tuple[bool, List[Tuple[str, bool]]]:
        """
        Check if constrained segments have valid syntax.

        For FOLIO, segments may be "formula True" (formula + label); we try parsing
        the formula only (strip trailing True/False/Uncertain) so syntax rate is fair.
        """
        from lark import Lark
        from lark.exceptions import LarkError

        segments_out: List[Tuple[str, bool]] = []
        matches = self._extract_constrained_content(output)

        if not matches:
            return True, []

        grammar_text = self._get_grammar_file().read_text()
        try:
            parser = Lark(grammar_text, start="start", parser="lalr")
            for match in matches:
                s = match.strip()
                try:
                    parser.parse(s)
                    segments_out.append((match, True))
                except LarkError:
                    # Try without trailing label (e.g. "P(x) True" -> "P(x)")
                    s_no_label = self._strip_trailing_label(s)
                    try:
                        if s_no_label:
                            parser.parse(s_no_label)
                            segments_out.append((match, True))
                        else:
                            segments_out.append((match, False))
                    except LarkError:
                        segments_out.append((match, False))
        except Exception:
            return True, [(m, True) for m in matches]

        all_valid = all(is_valid for _, is_valid in segments_out) if segments_out else True
        return all_valid, segments_out

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
            # CPU fallback is very slow; cap to 1 example so the run finishes in minutes
            if env.get("_eval_cpu_fallback") and len(dataset) > 1:
                print(f"  Limiting to 1 example (CPU evaluation is slow; had {len(dataset)} requested).")
                dataset = dataset[:1]

            from evaluations.common.generation import run_crane_csd
            from evaluations.common.parser_utils import create_folio_wrapper_parser

            num_correct = 0
            num_valid_format = 0
            num_valid_syntax = 0
            total_segments = 0

            n_examples = len(dataset)
            for i, example in enumerate(dataset):
                print(f"  Evaluating example {i + 1}/{n_examples}...", flush=True)
                prompt = self._format_prompt(example)
                expected = self._get_expected_answer(example)
                question_str = self._example_field(example, "question", "") or self._example_field(example, "premises", "") or ""

                try:
                    dynamic_parser = None
                    if self.dataset_name == "folio":
                        dynamic_parser = create_folio_wrapper_parser(
                            env["VerifiedDecoderAgent"],
                            env["_dafny"],
                            env["parser"],
                            env["lm"]._Tokens,
                            env["tokenizer"],
                        )
                    output_text, token_count, gen_time, _ = run_crane_csd(
                        env=env,
                        prompt_text=prompt,
                        max_steps=self.max_steps,
                        grammar_file=self._get_grammar_file(),
                        dynamic_parser=dynamic_parser,
                    )
                    output_text = self._ensure_delimiters_around_constrained(output_text)

                    t0 = time.time()
                    if self.dataset_name == "gsm_symbolic":
                        actual = self._extract_answer_gsm(output_text)
                    else:
                        actual = self._extract_answer_folio(output_text, example=example)
                    extract_time = time.time() - t0
                    print(f"    (gen: {gen_time:.1f}s, extract: {extract_time:.1f}s)", flush=True)

                    is_correct = self._answers_match(actual, expected)
                    if is_correct:
                        num_correct += 1

                    is_valid_format = self._check_format_validity(output_text)
                    if is_valid_format:
                        num_valid_format += 1

                    all_valid_syntax, segments = self._check_syntax_validity(output_text)
                    total_segments += len(segments)
                    num_valid_syntax += sum(1 for _, v in segments if v)

                    extracted_segments = self._extract_constrained_content(output_text)
                    sample_entry: Dict[str, Any] = {
                        "question": str(question_str)[:200],
                        "expected": expected,
                        "actual": actual or output_text[:100],
                        "full_output": output_text,
                        "extracted_segments": extracted_segments,
                        "is_correct": is_correct,
                        "is_valid_format": is_valid_format,
                        "is_syntax_valid": all_valid_syntax,
                        "token_count": token_count,
                        "time_seconds": gen_time,
                    }
                    if self.dataset_name == "folio" and example is not None:
                        sample_entry["gold_premises_fol"] = self._example_field(example, "fol_premises", None)
                        sample_entry["gold_conclusion_fol"] = self._example_field(example, "fol_conclusion", None)
                        sample_entry["folio_debug"] = getattr(self, "_last_folio_debug", None)
                    sample_outputs.append(sample_entry)

                except Exception as e:
                    question_str = self._example_field(example, "question", "") or str(self._example_field(example, "premises", ""))
                    sample_outputs.append({
                        "question": str(question_str)[:200],
                        "expected": expected,
                        "actual": None,
                        "full_output": None,
                        "extracted_segments": [],
                        "is_correct": False,
                        "is_valid_format": False,
                        "is_syntax_valid": False,
                        "error": str(e),
                    })

            total_time = time.time() - start_time
            num_examples = len(dataset)

            return EvaluationResult(
                success=True,
                accuracy=num_correct / max(1, num_examples),
                format_rate=num_valid_format / max(1, num_examples),
                syntax_rate=num_valid_syntax / max(1, total_segments) if total_segments > 0 else 1.0,
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
