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
    def MyCSDStrategy(lm, parser, prompt, maxSteps):
        generated: _dafny.Seq = _dafny.Seq({})
        d_0_success_: bool = False
        d_0_success_ = False
        if (maxSteps) >= (3):
            out0_: _dafny.Seq
            out0_ = VerifiedDecoderAgent.CSDHelpers.TryUnconstrainedThenConstrained(lm, parser, prompt, maxSteps, 3)
            generated = out0_
            d_0_success_ = (parser).IsValidPrefix(generated)
        if not(d_0_success_):
            out1_: _dafny.Seq
            out1_ = VerifiedDecoderAgent.CSDHelpers.PureConstrainedGeneration(lm, parser, prompt, maxSteps)
            generated = out1_
        return generated

