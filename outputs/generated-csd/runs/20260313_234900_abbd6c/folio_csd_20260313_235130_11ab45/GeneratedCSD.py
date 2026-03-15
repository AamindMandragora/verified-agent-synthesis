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
            d_2_validTokens_: _dafny.Seq
            d_2_validTokens_ = (parser).ValidNextTokens(generated)
            if (not((parser).IsPermissive(generated))) or ((d_1_stepsLeft_) == (0)):
                d_3_next_: _dafny.Seq
                d_4_newSteps_: int
                out0_: _dafny.Seq
                out1_: int
                out0_, out1_ = (d_0_helpers_).ConstrainedStep(lm, parser, prompt, generated, d_1_stepsLeft_)
                d_3_next_ = out0_
                d_4_newSteps_ = out1_
                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_3_next_]))
                d_1_stepsLeft_ = d_4_newSteps_
            elif True:
                d_5_next_: _dafny.Seq
                d_6_newSteps_: int
                out2_: _dafny.Seq
                out3_: int
                out2_, out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated, d_1_stepsLeft_)
                d_5_next_ = out2_
                d_6_newSteps_ = out3_
                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_5_next_]))
                d_1_stepsLeft_ = d_6_newSteps_
        remainingSteps = d_1_stepsLeft_
        remainingSteps = d_1_stepsLeft_
        return generated, remainingSteps

