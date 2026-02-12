import sys
from typing import Callable, Any, TypeVar, NamedTuple
from math import floor
from itertools import count

import module_ as module_
import _dafny as _dafny
import System_ as System_

# Module: VerifiedDecoderAgent


class LM:
    def  __init__(self):
        self.Logits: _dafny.Array = _dafny.Array(None, 0)
        self._Tokens: _dafny.Seq = _dafny.Seq({})
        self._Ids: _dafny.Seq = _dafny.Seq({})
        pass

    def __dafnystr__(self) -> str:
        return "VerifiedDecoderAgent.LM"
    def ValidTokensIdsLogits(self):
        def lambda0_(forall_var_0_):
            d_0_i_: int = forall_var_0_
            return not (((0) <= (d_0_i_)) and ((d_0_i_) < (len((self).Ids)))) or (((d_0_i_) == (((self).Ids)[d_0_i_])) and ((d_0_i_) in ((self).Ids)))

        def lambda1_(forall_var_1_):
            def lambda2_(forall_var_2_):
                d_2_j_: int = forall_var_2_
                return not (((((0) <= (d_1_i_)) and ((d_1_i_) < (len((self).Tokens)))) and (((0) <= (d_2_j_)) and ((d_2_j_) < (len((self).Tokens))))) and ((d_1_i_) != (d_2_j_))) or ((((self).Tokens)[d_1_i_]) != (((self).Tokens)[d_2_j_]))

            d_1_i_: int = forall_var_1_
            return _dafny.quantifier(_dafny.IntegerRange(0, len((self).Tokens)), True, lambda2_)

        def lambda3_(forall_var_3_):
            def lambda4_(exists_var_0_):
                d_4_i_: int = exists_var_0_
                return (((0) <= (d_4_i_)) and ((d_4_i_) < (len((self).Ids)))) and ((((self).Tokens)[d_4_i_]) == (d_3_token_))

            d_3_token_: _dafny.Seq = forall_var_3_
            return not ((d_3_token_) in ((self).Tokens)) or (_dafny.quantifier(_dafny.IntegerRange(0, len((self).Ids)), False, lambda4_))

        def lambda5_(forall_var_4_):
            d_5_i_: int = forall_var_4_
            return not (((0) <= (d_5_i_)) and ((d_5_i_) < ((self.Logits).length(0)))) or ((((self.Logits)[d_5_i_]) <= (_dafny.BigRational('1e9'))) and (((self.Logits)[d_5_i_]) >= (_dafny.BigRational('-1e9'))))

        return (((((((len((self).Tokens)) == (len((self).Ids))) and ((len((self).Ids)) == ((self.Logits).length(0)))) and (((len((self).Ids)) > (0)) and ((((self).Ids)[0]) == (0)))) and (_dafny.quantifier(_dafny.IntegerRange(0, len((self).Ids)), True, lambda0_))) and (_dafny.quantifier(_dafny.IntegerRange(0, len((self).Tokens)), True, lambda1_))) and (_dafny.quantifier(((self).Tokens).UniqueElements, True, lambda3_))) and (_dafny.quantifier(_dafny.IntegerRange(0, (self.Logits).length(0)), True, lambda5_))

    def IdToToken(self, id_):
        return ((self).Tokens)[id_]

    def TokenToId(self, token):
        return (self).TokenToIdRecursive(token, 0)

    def TokenToIdRecursive(self, token, offset):
        _this = self
        while True:
            with _dafny.label():
                if (((_this).Tokens)[offset]) == (token):
                    return offset
                elif True:
                    in0_ = _this
                    in1_ = token
                    in2_ = (offset) + (1)
                    _this = in0_
                    
                    token = in1_
                    offset = in2_
                    raise _dafny.TailCall()
                break

    def IdToLogit(self, id_):
        return (self.Logits)[id_]

    def TokenToLogit(self, token):
        return (self).IdToLogit((self).TokenToId(token))

    def TokensToLogits(self, tokens):
        d_0___accumulator_ = _dafny.SeqWithoutIsStrInference([])
        _this = self
        while True:
            with _dafny.label():
                if (len(tokens)) == (1):
                    return (d_0___accumulator_) + (_dafny.SeqWithoutIsStrInference([(_this).TokenToLogit((tokens)[0])]))
                elif True:
                    d_0___accumulator_ = (d_0___accumulator_) + (_dafny.SeqWithoutIsStrInference([(_this).TokenToLogit((tokens)[0])]))
                    in0_ = _this
                    in1_ = _dafny.SeqWithoutIsStrInference((tokens)[1::])
                    _this = in0_
                    
                    tokens = in1_
                    raise _dafny.TailCall()
                break

    def IdsToLogits(self, ids):
        d_0___accumulator_ = _dafny.SeqWithoutIsStrInference([])
        _this = self
        while True:
            with _dafny.label():
                if (len(ids)) == (1):
                    return (d_0___accumulator_) + (_dafny.SeqWithoutIsStrInference([(_this).IdToLogit((ids)[0])]))
                elif True:
                    d_0___accumulator_ = (d_0___accumulator_) + (_dafny.SeqWithoutIsStrInference([(_this).IdToLogit((ids)[0])]))
                    in0_ = _this
                    in1_ = _dafny.SeqWithoutIsStrInference((ids)[1::])
                    _this = in0_
                    
                    ids = in1_
                    raise _dafny.TailCall()
                break

    def MaskToken(self, token):
        d_0_id_: int
        d_0_id_ = (self).TokenToId(token)
        arr0_ = self.Logits
        arr0_[(d_0_id_)] = _dafny.BigRational('-1e9')

    def MaskTokens(self, tokens):
        d_0_N_: int
        d_0_N_ = len(tokens)
        d_1_i_: int
        d_1_i_ = 0
        while (d_1_i_) < (d_0_N_):
            (self).MaskToken((tokens)[d_1_i_])
            d_1_i_ = (d_1_i_) + (1)

    def MaskTokensExcept(self, tokens):
        d_0_toMask_: _dafny.Seq
        d_0_toMask_ = _dafny.SeqWithoutIsStrInference([])
        d_1_N_: int
        d_1_N_ = len((self).Tokens)
        d_2_i_: int
        d_2_i_ = 0
        while (d_2_i_) < (d_1_N_):
            if not((((self).Tokens)[d_2_i_]) in (tokens)):
                d_0_toMask_ = (d_0_toMask_) + (_dafny.SeqWithoutIsStrInference([((self).Tokens)[d_2_i_]]))
            d_2_i_ = (d_2_i_) + (1)
        if (len(d_0_toMask_)) > (0):
            (self).MaskTokens(d_0_toMask_)

    def IsMasked(self, token):
        return ((self.Logits)[(self).TokenToId(token)]) == (_dafny.BigRational('-1e9'))

    def HasUnmaskedToken(self):
        def lambda0_(exists_var_0_):
            d_0_t_: _dafny.Seq = exists_var_0_
            return ((d_0_t_) in ((self).Tokens)) and (not((self).IsMasked(d_0_t_)))

        return _dafny.quantifier(((self).Tokens).UniqueElements, False, lambda0_)

    @property
    def Tokens(self):
        return self._Tokens
    @property
    def Ids(self):
        return self._Ids

class Parser:
    def  __init__(self):
        pass

    def __dafnystr__(self) -> str:
        return "VerifiedDecoderAgent.Parser"
    def IsDeadPrefix(self, prefix):
        return (not((self).IsCompletePrefix(prefix))) and ((len((self).ValidNextTokens(prefix))) == (0))

    def ValidNextToken(self, prefix, token):
        return (token) in ((self).ValidNextTokens(prefix))


class CSDHelpers:
    def  __init__(self):
        self.cost: int = int(0)
        pass

    def __dafnystr__(self) -> str:
        return "VerifiedDecoderAgent.CSDHelpers"
    def ctor__(self):
        (self).cost = 0

    def UnconstrainedStep(self, lm, prompt, generated):
        next_: _dafny.Seq = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
        (lm).GenerateLogits((prompt) + (generated))
        out0_: _dafny.Seq
        out0_ = (lm).ChooseNextToken()
        next_ = out0_
        (self).cost = (self.cost) + (1)
        return next_

    def ConstrainedStep(self, lm, parser, prompt, generated):
        next_: _dafny.Seq = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
        (lm).GenerateLogits((prompt) + (generated))
        (lm).MaskTokensExcept((parser).ValidNextTokens(generated))
        out0_: _dafny.Seq
        out0_ = (lm).ChooseNextToken()
        next_ = out0_
        (self).cost = (self.cost) + (1)
        return next_

    def UnconstrainedGeneration(self, lm, prompt, maxSteps):
        generated: _dafny.Seq = _dafny.Seq({})
        generated = _dafny.SeqWithoutIsStrInference([])
        d_0_steps_: int
        d_0_steps_ = 0
        while (d_0_steps_) < (maxSteps):
            d_1_next_: _dafny.Seq
            out0_: _dafny.Seq
            out0_ = (self).UnconstrainedStep(lm, prompt, generated)
            d_1_next_ = out0_
            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_1_next_]))
            d_0_steps_ = (d_0_steps_) + (1)
        return generated

    def ConstrainedGeneration(self, lm, parser, prompt, maxSteps):
        generated: _dafny.Seq = _dafny.Seq({})
        generated = _dafny.SeqWithoutIsStrInference([])
        d_0_steps_: int
        d_0_steps_ = 0
        while ((d_0_steps_) < (maxSteps)) and (not((parser).IsCompletePrefix(generated))):
            d_1_next_: _dafny.Seq
            out0_: _dafny.Seq
            out0_ = (self).ConstrainedStep(lm, parser, prompt, generated)
            d_1_next_ = out0_
            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_1_next_]))
            d_0_steps_ = (d_0_steps_) + (1)
        return generated

    @staticmethod
    def RollbackToValidPrefix(parser, generated):
        repaired: _dafny.Seq = _dafny.Seq({})
        repaired = generated
        while (not((parser).IsValidPrefix(repaired))) or ((parser).IsDeadPrefix(repaired)):
            repaired = _dafny.SeqWithoutIsStrInference((repaired)[:(len(repaired)) - (1):])
        return repaired

    def TryUnconstrainedThenConstrained(self, lm, parser, prompt, maxSteps, unconstrainedSteps):
        generated: _dafny.Seq = _dafny.Seq({})
        d_0_unconstrained_: _dafny.Seq
        out0_: _dafny.Seq
        out0_ = (self).UnconstrainedGeneration(lm, prompt, maxSteps)
        d_0_unconstrained_ = out0_
        d_1_validPrefix_: _dafny.Seq
        out1_: _dafny.Seq
        out1_ = CSDHelpers.RollbackToValidPrefix(parser, d_0_unconstrained_)
        d_1_validPrefix_ = out1_
        if (parser).IsCompletePrefix(d_1_validPrefix_):
            generated = d_1_validPrefix_
        elif True:
            d_2_remainingSteps_: int
            d_2_remainingSteps_ = (maxSteps) - (len(d_1_validPrefix_))
            generated = d_1_validPrefix_
            d_3_steps_: int
            d_3_steps_ = len(d_1_validPrefix_)
            while ((d_3_steps_) < (maxSteps)) and (not((parser).IsCompletePrefix(generated))):
                d_4_next_: _dafny.Seq
                out2_: _dafny.Seq
                out2_ = (self).ConstrainedStep(lm, parser, prompt, generated)
                d_4_next_ = out2_
                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_4_next_]))
                d_3_steps_ = (d_3_steps_) + (1)
        return generated

    def CompletePrefix(self, lm, parser, prompt, partial, maxSteps):
        generated: _dafny.Seq = _dafny.Seq({})
        generated = partial
        d_0_steps_: int
        d_0_steps_ = len(partial)
        while ((d_0_steps_) < (maxSteps)) and (not((parser).IsCompletePrefix(generated))):
            d_1_next_: _dafny.Seq
            out0_: _dafny.Seq
            out0_ = (self).ConstrainedStep(lm, parser, prompt, generated)
            d_1_next_ = out0_
            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_1_next_]))
            d_0_steps_ = (d_0_steps_) + (1)
        return generated

    def UnconstrainedWithCompletion(self, lm, parser, prompt, maxSteps):
        generated: _dafny.Seq = _dafny.Seq({})
        d_0_unconstrained_: _dafny.Seq
        out0_: _dafny.Seq
        out0_ = (self).UnconstrainedGeneration(lm, prompt, maxSteps)
        d_0_unconstrained_ = out0_
        d_1_validPrefix_: _dafny.Seq
        out1_: _dafny.Seq
        out1_ = CSDHelpers.RollbackToValidPrefix(parser, d_0_unconstrained_)
        d_1_validPrefix_ = out1_
        if (parser).IsCompletePrefix(d_1_validPrefix_):
            generated = d_1_validPrefix_
        elif True:
            d_2_remainingSteps_: int
            d_2_remainingSteps_ = (maxSteps) - (len(d_1_validPrefix_))
            generated = d_1_validPrefix_
            d_3_steps_: int
            d_3_steps_ = len(d_1_validPrefix_)
            while ((d_3_steps_) < (maxSteps)) and (not((parser).IsCompletePrefix(generated))):
                d_4_next_: _dafny.Seq
                out2_: _dafny.Seq
                out2_ = (self).ConstrainedStep(lm, parser, prompt, generated)
                d_4_next_ = out2_
                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_4_next_]))
                d_3_steps_ = (d_3_steps_) + (1)
        return generated

    def HybridGeneration(self, lm, parser, prompt, maxSteps):
        generated: _dafny.Seq = _dafny.Seq({})
        generated = _dafny.SeqWithoutIsStrInference([])
        d_0_totalSteps_: int
        d_0_totalSteps_ = 0
        d_1_insideHybrid_: bool
        d_1_insideHybrid_ = False
        while (((d_0_totalSteps_) < (maxSteps)) and (not((parser).IsCompletePrefix(generated)))) and ((len(generated)) < (maxSteps)):
            if not(d_1_insideHybrid_):
                d_2_next_: _dafny.Seq
                out0_: _dafny.Seq
                out0_ = (self).ConstrainedStep(lm, parser, prompt, generated)
                d_2_next_ = out0_
                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_2_next_]))
                d_0_totalSteps_ = (d_0_totalSteps_) + (1)
                if (d_2_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                    d_1_insideHybrid_ = True
            elif True:
                d_3_next_: _dafny.Seq
                out1_: _dafny.Seq
                out1_ = (self).UnconstrainedStep(lm, prompt, generated)
                d_3_next_ = out1_
                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_3_next_]))
                d_0_totalSteps_ = (d_0_totalSteps_) + (1)
                if (d_3_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))):
                    if (parser).IsValidPrefix(generated):
                        d_1_insideHybrid_ = False
                    elif True:
                        out2_: _dafny.Seq
                        out2_ = CSDHelpers.RollbackToValidPrefix(parser, generated)
                        generated = out2_
                        d_1_insideHybrid_ = False
        if d_1_insideHybrid_:
            out3_: _dafny.Seq
            out3_ = CSDHelpers.RollbackToValidPrefix(parser, generated)
            generated = out3_
            d_1_insideHybrid_ = False
        d_4_stepsBeforeFinal_: int
        d_4_stepsBeforeFinal_ = d_0_totalSteps_
        d_5_lengthBeforeFinal_: int
        d_5_lengthBeforeFinal_ = len(generated)
        while ((len(generated)) < (maxSteps)) and (not((parser).IsCompletePrefix(generated))):
            d_6_next_: _dafny.Seq
            out4_: _dafny.Seq
            out4_ = (self).ConstrainedStep(lm, parser, prompt, generated)
            d_6_next_ = out4_
            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_6_next_]))
            d_0_totalSteps_ = (d_0_totalSteps_) + (1)
        return generated

    def SpeculativeGeneration(self, lm, parser, prompt, maxSteps, speculativeWindow):
        generated: _dafny.Seq = _dafny.Seq({})
        generated = _dafny.SeqWithoutIsStrInference([])
        d_0_steps_: int
        d_0_steps_ = 0
        while ((d_0_steps_) < (maxSteps)) and (not((parser).IsCompletePrefix(generated))):
            d_1_speculateSteps_: int
            if ((d_0_steps_) + (speculativeWindow)) <= (maxSteps):
                d_1_speculateSteps_ = speculativeWindow
            elif True:
                d_1_speculateSteps_ = (maxSteps) - (d_0_steps_)
            d_2_speculated_: _dafny.Seq
            out0_: _dafny.Seq
            out0_ = (self).UnconstrainedGeneration(lm, (prompt) + (generated), d_1_speculateSteps_)
            d_2_speculated_ = out0_
            d_3_validCount_: int
            d_3_validCount_ = 0
            d_4_tempPrefix_: _dafny.Seq
            d_4_tempPrefix_ = generated
            while ((d_3_validCount_) < (len(d_2_speculated_))) and ((parser).IsValidPrefix((d_4_tempPrefix_) + (_dafny.SeqWithoutIsStrInference([(d_2_speculated_)[d_3_validCount_]])))):
                d_4_tempPrefix_ = (d_4_tempPrefix_) + (_dafny.SeqWithoutIsStrInference([(d_2_speculated_)[d_3_validCount_]]))
                d_3_validCount_ = (d_3_validCount_) + (1)
            if (d_3_validCount_) > (0):
                generated = d_4_tempPrefix_
                d_0_steps_ = (d_0_steps_) + (d_3_validCount_)
            elif True:
                d_5_next_: _dafny.Seq
                out1_: _dafny.Seq
                out1_ = (self).ConstrainedStep(lm, parser, prompt, generated)
                d_5_next_ = out1_
                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_5_next_]))
                d_0_steps_ = (d_0_steps_) + (1)
        return generated

    def PureConstrainedGeneration(self, lm, parser, prompt, maxSteps):
        generated: _dafny.Seq = _dafny.Seq({})
        out0_: _dafny.Seq
        out0_ = (self).ConstrainedGeneration(lm, parser, prompt, maxSteps)
        generated = out0_
        return generated

    def GenerateWithReasonableLength(self, lm, parser, prompt, maxSteps, reasonableLength):
        generated: _dafny.Seq = _dafny.Seq({})
        generated = _dafny.SeqWithoutIsStrInference([])
        d_0_steps_: int
        d_0_steps_ = 0
        with _dafny.label("0"):
            while ((d_0_steps_) < (maxSteps)) and (not((parser).IsCompletePrefix(generated))):
                with _dafny.c_label("0"):
                    d_1_next_: _dafny.Seq
                    out0_: _dafny.Seq
                    out0_ = (self).ConstrainedStep(lm, parser, prompt, generated)
                    d_1_next_ = out0_
                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_1_next_]))
                    d_0_steps_ = (d_0_steps_) + (1)
                    if ((parser).IsCompletePrefix(generated)) and ((len(generated)) <= (reasonableLength)):
                        raise _dafny.Break("0")
                    pass
            pass
        return generated

    def GenerateUntilFirstComplete(self, lm, parser, prompt, maxSteps):
        generated: _dafny.Seq = _dafny.Seq({})
        out0_: _dafny.Seq
        out0_ = (self).ConstrainedGeneration(lm, parser, prompt, maxSteps)
        generated = out0_
        return generated

    @staticmethod
    def SelectBestCandidate(candidates, parser, preferShorter):
        best: _dafny.Seq = _dafny.Seq({})
        d_0_result_: _dafny.Seq
        d_0_result_ = (candidates)[0]
        d_1_bestScore_: int
        if (preferShorter) and ((parser).IsCompletePrefix(d_0_result_)):
            d_1_bestScore_ = (0) - (len(d_0_result_))
        elif True:
            d_1_bestScore_ = -1000
        d_2_i_: int
        d_2_i_ = 1
        while (d_2_i_) < (len(candidates)):
            d_3_candidate_: _dafny.Seq
            d_3_candidate_ = (candidates)[d_2_i_]
            d_4_score_: int
            if (preferShorter) and ((parser).IsCompletePrefix(d_3_candidate_)):
                d_4_score_ = (0) - (len(d_3_candidate_))
            elif True:
                d_4_score_ = -1000
            if ((d_4_score_) > (d_1_bestScore_)) or (((((d_4_score_) == (d_1_bestScore_)) and (preferShorter)) and ((parser).IsCompletePrefix(d_3_candidate_))) and ((len(d_3_candidate_)) < (len(d_0_result_)))):
                d_0_result_ = d_3_candidate_
                d_1_bestScore_ = d_4_score_
            d_2_i_ = (d_2_i_) + (1)
        best = d_0_result_
        return best

    def GenerateAndSelectBest(self, lm, parser, prompt, maxSteps, numCandidates, preferShorter):
        generated: _dafny.Seq = _dafny.Seq({})
        d_0_firstCandidate_: _dafny.Seq = _dafny.Seq({})
        out0_: _dafny.Seq
        out0_ = (self).ConstrainedGeneration(lm, parser, prompt, maxSteps)
        d_0_firstCandidate_ = out0_
        d_1_candidates_: _dafny.Seq
        d_1_candidates_ = _dafny.SeqWithoutIsStrInference([d_0_firstCandidate_])
        d_2_i_: int
        d_2_i_ = 1
        while (d_2_i_) < (numCandidates):
            d_3_candidate_: _dafny.Seq = _dafny.Seq({})
            out1_: _dafny.Seq
            out1_ = (self).ConstrainedGeneration(lm, parser, prompt, maxSteps)
            d_3_candidate_ = out1_
            d_1_candidates_ = (d_1_candidates_) + (_dafny.SeqWithoutIsStrInference([d_3_candidate_]))
            d_2_i_ = (d_2_i_) + (1)
        d_4_best_: _dafny.Seq = _dafny.Seq({})
        out2_: _dafny.Seq
        out2_ = CSDHelpers.SelectBestCandidate(d_1_candidates_, parser, preferShorter)
        d_4_best_ = out2_
        generated = d_4_best_
        return generated

    def GenerateReasoningAndAnswer(self, lm, parser, prompt, maxSteps):
        generated: _dafny.Seq = _dafny.Seq({})
        reasoning: _dafny.Seq = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
        out0_: _dafny.Seq
        out0_ = (self).HybridGeneration(lm, parser, prompt, maxSteps)
        generated = out0_
        reasoning = CSDHelpers.ExtractContentBetweenDelimiters(CSDHelpers.PrefixToString(generated), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>")))
        return generated, reasoning

    @staticmethod
    def PrefixToString(p):
        d_0___accumulator_ = _dafny.SeqWithoutIsStrInference([])
        while True:
            with _dafny.label():
                if (len(p)) == (0):
                    return (d_0___accumulator_) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "")))
                elif True:
                    d_0___accumulator_ = (d_0___accumulator_) + ((p)[0])
                    in0_ = _dafny.SeqWithoutIsStrInference((p)[1::])
                    p = in0_
                    raise _dafny.TailCall()
                break

    @staticmethod
    def ExtractContentBetweenDelimiters(input_, startDelim, endDelim):
        return CSDHelpers.ExtractContentExtern(input_, startDelim, endDelim)

    def CraneGeneration(self, lm, parser, prompt, maxSteps):
        generated: _dafny.Seq = _dafny.Seq({})
        generated = _dafny.SeqWithoutIsStrInference([])
        d_0_steps_: int
        d_0_steps_ = 0
        d_1_insideConstrained_: bool
        d_1_insideConstrained_ = False
        d_2_currentConstrained_: _dafny.Seq
        d_2_currentConstrained_ = _dafny.SeqWithoutIsStrInference([])
        while (d_0_steps_) < (maxSteps):
            if not(d_1_insideConstrained_):
                d_3_next_: _dafny.Seq
                out0_: _dafny.Seq
                out0_ = (self).UnconstrainedStep(lm, prompt, generated)
                d_3_next_ = out0_
                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_3_next_]))
                d_0_steps_ = (d_0_steps_) + (1)
                if (d_3_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                    d_1_insideConstrained_ = True
                    d_2_currentConstrained_ = _dafny.SeqWithoutIsStrInference([])
            elif True:
                if (parser).IsCompletePrefix(d_2_currentConstrained_):
                    d_1_insideConstrained_ = False
                elif True:
                    d_4_next_: _dafny.Seq
                    out1_: _dafny.Seq
                    out1_ = (self).ConstrainedStep(lm, parser, (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(d_2_currentConstrained_)):])), d_2_currentConstrained_)
                    d_4_next_ = out1_
                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_4_next_]))
                    d_2_currentConstrained_ = (d_2_currentConstrained_) + (_dafny.SeqWithoutIsStrInference([d_4_next_]))
                    d_0_steps_ = (d_0_steps_) + (1)
                    if (d_4_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))):
                        d_1_insideConstrained_ = False
        return generated

