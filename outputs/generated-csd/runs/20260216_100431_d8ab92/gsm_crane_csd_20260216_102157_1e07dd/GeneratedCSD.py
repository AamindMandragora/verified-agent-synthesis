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
        cost: int = int(0)
        d_0_helpers_: VerifiedDecoderAgent.CSDHelpers
        nw0_ = VerifiedDecoderAgent.CSDHelpers()
        nw0_.ctor__()
        d_0_helpers_ = nw0_
        if (maxSteps) >= (10):
            out0_: _dafny.Seq
            out0_ = (d_0_helpers_).TryUnconstrainedThenConstrained(lm, parser, prompt, maxSteps, 5)
            generated = out0_
        elif True:
            out1_: _dafny.Seq
            out1_ = (d_0_helpers_).PureConstrainedGeneration(lm, parser, prompt, maxSteps)
            generated = out1_
        cost = d_0_helpers_.cost
        cost = d_0_helpers_.cost
        return generated, cost

