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
        d_0_helpers_: VerifiedDecoderAgent.CSDHelpers
        nw0_ = VerifiedDecoderAgent.CSDHelpers()
        nw0_.ctor__()
        d_0_helpers_ = nw0_
        d_1_stepsLeft_: int
        d_1_stepsLeft_ = maxSteps
        generated = _dafny.SeqWithoutIsStrInference([])
        while ((d_1_stepsLeft_) > (0)) and (not((parser).IsCompletePrefix(generated))):
            d_2_next_: _dafny.Seq = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
            d_3_newSteps_: int = int(0)
            if (parser).IsCompletePrefix(generated):
                out0_: _dafny.Seq
                out1_: int
                out0_, out1_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated, d_1_stepsLeft_)
                d_2_next_ = out0_
                d_3_newSteps_ = out1_
            elif True:
                out2_: _dafny.Seq
                out3_: int
                out2_, out3_ = (d_0_helpers_).ConstrainedStep(lm, parser, prompt, generated, d_1_stepsLeft_)
                d_2_next_ = out2_
                d_3_newSteps_ = out3_
            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_2_next_]))
            d_1_stepsLeft_ = d_3_newSteps_
        remainingSteps = d_1_stepsLeft_
        return generated, remainingSteps

