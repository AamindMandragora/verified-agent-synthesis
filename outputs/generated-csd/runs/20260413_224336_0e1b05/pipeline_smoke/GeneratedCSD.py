import sys
from typing import Callable, Any, TypeVar, NamedTuple
from math import floor
from itertools import count

import module_ as module_
import _dafny as _dafny
import System_ as System_
import VerifiedDecoderAgent as VerifiedDecoderAgent

# Module: GeneratedCSD

class default__:
    def  __init__(self):
        pass

    @staticmethod
    def MyCSDStrategy(lm, parser, prompt, maxSteps, eosToken):
        generated: _dafny.Seq = _dafny.Seq({})
        remainingSteps: int = int(0)
        d_0_delim_: VerifiedDecoderAgent.Delimiter
        nw0_ = VerifiedDecoderAgent.Delimiter()
        nw0_.ctor__(VerifiedDecoderAgent.default__.LeftDelimiter, VerifiedDecoderAgent.default__.RightDelimiter)
        d_0_delim_ = nw0_
        d_1_helpers_: VerifiedDecoderAgent.CSDHelpers
        nw1_ = VerifiedDecoderAgent.CSDHelpers()
        nw1_.ctor__(lm, parser, d_0_delim_)
        d_1_helpers_ = nw1_
        generated = _dafny.SeqWithoutIsStrInference([])
        d_2_stepsLeft_: int
        d_2_stepsLeft_ = maxSteps
        remainingSteps = d_2_stepsLeft_
        return generated, remainingSteps

