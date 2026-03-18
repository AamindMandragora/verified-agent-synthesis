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
        d_2_stepsLeft_: int
        d_2_stepsLeft_ = maxSteps
        generated = _dafny.SeqWithoutIsStrInference([])
        d_3_next_: _dafny.Seq = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
        d_4_newSteps_: int = int(0)
        while ((d_2_stepsLeft_) > (0)) and (not((parser).IsCompletePrefix(generated))):
            if ((d_1_helpers_).InsideDelimitedWindow(generated)) and (not((parser).IsCompletePrefix((d_1_helpers_).GetDelimitedContent(generated)))):
                out0_: _dafny.Seq
                out1_: int
                out0_, out1_ = (d_1_helpers_).ConstrainedStep(prompt, generated, d_2_stepsLeft_)
                d_3_next_ = out0_
                d_4_newSteps_ = out1_
            elif True:
                out2_: _dafny.Seq
                out3_: int
                out2_, out3_ = (d_1_helpers_).UnconstrainedStep(prompt, generated, d_2_stepsLeft_)
                d_3_next_ = out2_
                d_4_newSteps_ = out3_
            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_3_next_]))
            d_2_stepsLeft_ = d_4_newSteps_
        remainingSteps = d_2_stepsLeft_
        return generated, remainingSteps

