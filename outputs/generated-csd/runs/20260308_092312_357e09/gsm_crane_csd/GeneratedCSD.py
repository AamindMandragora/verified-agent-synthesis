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
        d_2_hasValid_: bool
        d_2_hasValid_ = False
        with _dafny.label("0"):
            while ((d_1_stepsLeft_) > (0)) and (not((parser).IsCompletePrefix(generated))):
                with _dafny.c_label("0"):
                    if d_2_hasValid_:
                        raise _dafny.Continue("0")
                    if not((parser).IsCompletePrefix(generated)):
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
                        d_2_hasValid_ = True
                    pass
            pass
        remainingSteps = d_1_stepsLeft_
        remainingSteps = d_1_stepsLeft_
        return generated, remainingSteps

