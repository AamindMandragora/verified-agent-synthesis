"""
Synthesis-stage orchestration for the end-to-end CSD pipeline.

This package coordinates Python generation, verification through transpilation,
runtime execution, and dataset evaluation.
"""

from evaluation import EvaluationResult, Evaluator
from generation import StrategyGenerator
from verification import CompilationResult, DafnyCompiler, DafnyVerifier, VerificationResult
from .runner import StrategyRunner, RuntimeResult
from .feedback_loop import SynthesisPipeline, SynthesisExhaustionError
from .presets import DATASET_PRESETS, MODEL_PRESETS, SynthesisPreset, get_synthesis_preset, resolve_model_name

__all__ = [
    "StrategyGenerator",
    "DafnyVerifier",
    "VerificationResult",
    "DafnyCompiler",
    "CompilationResult",
    "Evaluator",
    "EvaluationResult",
    "StrategyRunner",
    "RuntimeResult",
    "SynthesisPipeline",
    "SynthesisExhaustionError",
    "SynthesisPreset",
    "DATASET_PRESETS",
    "MODEL_PRESETS",
    "get_synthesis_preset",
    "resolve_model_name",
]
