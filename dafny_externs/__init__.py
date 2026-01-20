"""dafny_externs: Python implementations of Dafny extern functions.

This package provides Python implementations of the {:extern} functions defined
in VerifiedAgentSynthesis.dfy, enabling execution of Dafny-compiled strategies.
"""

from .extern_functions import (
    LM,
    Parser,
    AllowedNext,
    ChooseToken,
    ApplyRepair,
    CheckSemantic,
    CompletePrefixConstrained,
    ApplySeqOp,
    CheckOutput,
    RunAttempt,
    RunStrategy,
    Run,
    ConstrainedDecode,
)

__all__ = [
    "LM",
    "Parser",
    "AllowedNext",
    "ChooseToken",
    "ApplyRepair",
    "CheckSemantic",
    "CompletePrefixConstrained",
    "ApplySeqOp",
    "CheckOutput",
    "RunAttempt",
    "RunStrategy",
    "Run",
    "ConstrainedDecode",
]

