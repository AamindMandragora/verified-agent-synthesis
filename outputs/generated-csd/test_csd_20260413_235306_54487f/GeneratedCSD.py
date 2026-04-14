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
        d_2_answer_: _dafny.Seq
        d_2_answer_ = _dafny.SeqWithoutIsStrInference([])
        d_3_stepsLeft_: int
        d_3_stepsLeft_ = (maxSteps) - (2)
        d_4_phase_: int
        d_4_phase_ = 0
        d_5_preamble__tokens_: int
        d_5_preamble__tokens_ = 0
        d_6_exploration__budget_: int
        d_6_exploration__budget_ = 1
        d_7_answer__tokens_: int
        d_7_answer__tokens_ = 0
        while ((d_3_stepsLeft_) > (0)) and (not((parser).IsCompletePrefix(d_2_answer_))):
            d_8_next__token_: _dafny.Seq
            d_8_next__token_ = eosToken
            d_9_new__steps_: int
            d_9_new__steps_ = d_3_stepsLeft_
            d_10_spend__freeform_: bool
            d_10_spend__freeform_ = ((((d_4_phase_) < (2)) and ((d_6_exploration__budget_) > (0))) and ((d_5_preamble__tokens_) < (1))) and ((d_3_stepsLeft_) > (1))
            if d_10_spend__freeform_:
                out0_: _dafny.Seq
                out1_: int
                out0_, out1_ = (d_1_helpers_).ExpressiveStep(prompt, generated, d_3_stepsLeft_)
                d_8_next__token_ = out0_
                d_9_new__steps_ = out1_
                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_8_next__token_]))
                d_3_stepsLeft_ = d_9_new__steps_
                d_5_preamble__tokens_ = (d_5_preamble__tokens_) + (1)
                d_6_exploration__budget_ = (d_6_exploration__budget_) - (1)
                if (d_5_preamble__tokens_) >= (1):
                    d_4_phase_ = 1
                if ((d_5_preamble__tokens_) >= (1)) or ((d_3_stepsLeft_) <= (1)):
                    d_4_phase_ = 2
            elif True:
                out2_: _dafny.Seq
                out3_: int
                out2_, out3_ = (d_1_helpers_).ConstrainedAnswerStep(prompt, generated, d_2_answer_, d_3_stepsLeft_)
                d_8_next__token_ = out2_
                d_9_new__steps_ = out3_
                d_2_answer_ = (d_2_answer_) + (_dafny.SeqWithoutIsStrInference([d_8_next__token_]))
                d_3_stepsLeft_ = d_9_new__steps_
                d_7_answer__tokens_ = (d_7_answer__tokens_) + (1)
                if (d_7_answer__tokens_) >= (1):
                    d_4_phase_ = 3
        generated = (((generated) + (_dafny.SeqWithoutIsStrInference([VerifiedDecoderAgent.default__.LeftDelimiter]))) + (d_2_answer_)) + (_dafny.SeqWithoutIsStrInference([VerifiedDecoderAgent.default__.RightDelimiter]))
        remainingSteps = d_3_stepsLeft_
        return generated, remainingSteps

