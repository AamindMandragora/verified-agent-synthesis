import sys
from typing import Callable, Any, TypeVar, NamedTuple
from math import floor
from itertools import count

import module_ as module_
import _dafny as _dafny
import System_ as System_

# Module: VerifiedDecoderAgent

class default__:
    def  __init__(self):
        pass

    @staticmethod
    def Contains(s, sub):
        def lambda0_(exists_var_0_):
            def lambda1_(exists_var_1_):
                d_1_j_: int = exists_var_1_
                return ((((0) <= (d_0_i_)) and ((d_0_i_) <= (d_1_j_))) and ((d_1_j_) <= (len(s)))) and ((_dafny.SeqWithoutIsStrInference((s)[d_0_i_:d_1_j_:])) == (sub))

            d_0_i_: int = exists_var_0_
            return _dafny.quantifier(_dafny.IntegerRange(d_0_i_, (len(s)) + (1)), False, lambda1_)

        return _dafny.quantifier(_dafny.IntegerRange(0, ((len(s)) + (1)) + (1)), False, lambda0_)

    @staticmethod
    def PrefixContains(p, t):
        def lambda0_(exists_var_0_):
            d_0_i_: int = exists_var_0_
            return (((0) <= (d_0_i_)) and ((d_0_i_) < (len(p)))) and (((p)[d_0_i_]) == (t))

        return _dafny.quantifier(_dafny.IntegerRange(0, len(p)), False, lambda0_)

    @_dafny.classproperty
    def LeftDelimiter(instance):
        return _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " <<"))
    @_dafny.classproperty
    def RightDelimiter(instance):
        return _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " >>"))

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


class Delimiter:
    def  __init__(self):
        self._Left: _dafny.Seq = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
        self._Right: _dafny.Seq = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
        pass

    def __dafnystr__(self) -> str:
        return "VerifiedDecoderAgent.Delimiter"
    def ctor__(self, left, right):
        (self)._Left = left
        (self)._Right = right

    def LastLeftDelimiterIndex(self, prefix):
        if (len(prefix)) == (0):
            return 0
        elif ((prefix)[(len(prefix)) - (1)]) == ((self).Left):
            return (len(prefix)) - (1)
        elif True:
            d_0_lastInRest_ = (self).LastLeftDelimiterIndex(_dafny.SeqWithoutIsStrInference((prefix)[:(len(prefix)) - (1):]))
            if (d_0_lastInRest_) < ((len(prefix)) - (1)):
                return d_0_lastInRest_
            elif True:
                return len(prefix)

    def FirstRightDelimiterIndex(self, content):
        d_0___accumulator_ = 0
        _this = self
        while True:
            with _dafny.label():
                if (len(content)) == (0):
                    return (0) + (d_0___accumulator_)
                elif ((content)[0]) == ((_this).Right):
                    return (0) + (d_0___accumulator_)
                elif True:
                    d_0___accumulator_ = (d_0___accumulator_) + (1)
                    in0_ = _this
                    in1_ = _dafny.SeqWithoutIsStrInference((content)[1::])
                    _this = in0_
                    
                    content = in1_
                    raise _dafny.TailCall()
                break

    def GetDelimitedContent(self, prefix):
        d_0_start_ = ((self).LastLeftDelimiterIndex(prefix)) + (1)
        if (d_0_start_) > (len(prefix)):
            return _dafny.SeqWithoutIsStrInference([])
        elif True:
            d_1_afterLeft_ = _dafny.SeqWithoutIsStrInference((prefix)[d_0_start_:len(prefix):])
            d_2_endIdx_ = (self).FirstRightDelimiterIndex(d_1_afterLeft_)
            return _dafny.SeqWithoutIsStrInference((d_1_afterLeft_)[:d_2_endIdx_:])

    def InsideDelimitedWindow(self, prefix):
        d_0_start_ = ((self).LastLeftDelimiterIndex(prefix)) + (1)
        return ((d_0_start_) <= (len(prefix))) and (((self).FirstRightDelimiterIndex(_dafny.SeqWithoutIsStrInference((prefix)[d_0_start_:len(prefix):]))) == (len(_dafny.SeqWithoutIsStrInference((prefix)[d_0_start_:len(prefix):]))))

    @property
    def Left(self):
        return self._Left
    @property
    def Right(self):
        return self._Right

class CSDHelpers:
    def  __init__(self):
        self._lm: LM = None
        self._parser: Parser = None
        self._delimiter: Delimiter = None
        pass

    def __dafnystr__(self) -> str:
        return "VerifiedDecoderAgent.CSDHelpers"
    def ctor__(self, lm, parser, delimiter):
        (self)._lm = lm
        (self)._parser = parser
        (self)._delimiter = delimiter

    def DelimitersInLM(self):
        return ((((self).lm).ValidTokensIdsLogits()) and ((((self).delimiter).Left) in (((self).lm).Tokens))) and ((((self).delimiter).Right) in (((self).lm).Tokens))

    def LeftDelimiter(self):
        return ((self).delimiter).Left

    def RightDelimiter(self):
        return ((self).delimiter).Right

    def GetDelimitedContent(self, prefix):
        return ((self).delimiter).GetDelimitedContent(prefix)

    def InsideDelimitedWindow(self, prefix):
        return ((self).delimiter).InsideDelimitedWindow(prefix)

    def ConstrainedWindowValid(self, prefix):
        return (not(((self).delimiter).InsideDelimitedWindow(prefix))) or (((self).parser).IsValidPrefix(((self).delimiter).GetDelimitedContent(prefix)))

    def UnconstrainedStep(self, prompt, generated, stepsLeft):
        next_: _dafny.Seq = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
        stepsLeft_k: int = int(0)
        ((self).lm).GenerateLogits((prompt) + (generated))
        out0_: _dafny.Seq
        out0_ = ((self).lm).ChooseNextToken()
        next_ = out0_
        stepsLeft_k = (stepsLeft) - (1)
        return next_, stepsLeft_k

    def ConstrainedStep(self, prompt, generated, stepsLeft):
        next_: _dafny.Seq = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
        stepsLeft_k: int = int(0)
        d_0_content_: _dafny.Seq
        d_0_content_ = (self).GetDelimitedContent(generated)
        ((self).lm).GenerateLogits((prompt) + (generated))
        ((self).lm).MaskTokensExcept(((self).parser).ValidNextTokens(d_0_content_))
        ((self).lm).MaskToken((self).LeftDelimiter())
        out0_: _dafny.Seq
        out0_ = ((self).lm).ChooseNextToken()
        next_ = out0_
        stepsLeft_k = (stepsLeft) - (1)
        return next_, stepsLeft_k

    def RollbackToValidPrefix(self, generated):
        repaired: _dafny.Seq = _dafny.Seq({})
        repaired = generated
        while ((len(repaired)) > (0)) and ((not(((self).parser).IsValidPrefix(repaired))) or (((self).parser).IsDeadPrefix(repaired))):
            repaired = _dafny.SeqWithoutIsStrInference((repaired)[:(len(repaired)) - (1):])
        return repaired

    @property
    def lm(self):
        return self._lm
    @property
    def parser(self):
        return self._parser
    @property
    def delimiter(self):
        return self._delimiter
