module VerifiedDecoderAgent {
  type Token = string
  type Prefix = seq<Token>
  type Id = nat
  type Logit = real

  class LM {
    const Tokens: seq<Token>
    const Ids: seq<Id>
    var Logits: array<Logit>

    predicate ValidTokensIdsLogits()
      reads this
      reads this.Logits
    {
      ((|Tokens| == |Ids|) && (|Ids| == Logits.Length) && (|Ids| > 0 && Ids[0] == 0)) &&
      (forall i :: 0 <= i < |Ids| ==> (i == Ids[i]) && (i in Ids)) && 
      (forall i, j :: 0 <= i < |Tokens| && 0 <= j < |Tokens| && i != j ==> Tokens[i] != Tokens[j]) &&
      (forall token: Token :: token in Tokens ==> (exists i :: 0 <= i < |Ids| && Tokens[i] == token)) &&
      (forall i :: (0 <= i < Logits.Length) ==> (-1000000000.0 <= Logits[i] && Logits[i] <= 1000000000.0))
    }

    constructor {:extern} {:axiom} ()
      ensures ValidTokensIdsLogits()

    function IdToToken(id: Id) : (token: Token)
      reads this
      reads this.Logits
      requires ValidTokensIdsLogits()
      requires id in Ids
      ensures token in Tokens
      ensures Tokens[id] == token
      ensures id == TokenToId(token)
      ensures ValidTokensIdsLogits()
    {
      Tokens[id]
    }

    function TokenToId(token: Token) : (id: Id)
      reads this
      reads this.Logits
      requires ValidTokensIdsLogits()
      requires token in Tokens
      ensures id in Ids
      ensures Tokens[id] == token
      ensures TokenToId(Tokens[id]) == id
      ensures ValidTokensIdsLogits()
    {
      TokenToIdRecursive(token, 0)
    }

    function TokenToIdRecursive(token: Token, offset: nat) : (id: Id)
      reads this
      reads this.Logits
      requires ValidTokensIdsLogits()
      requires token in Tokens
      requires 0 <= offset < |Tokens|
      requires (Tokens[offset] == token) || (token in Tokens[offset + 1..])
      ensures id in Ids
      ensures 0 <= TokenToIdRecursive(token, offset) < |Ids|
      ensures Tokens[id] == token
      ensures ValidTokensIdsLogits()
      decreases |Tokens| - offset
    {
      if Tokens[offset] == token then offset
      else TokenToIdRecursive(token, offset + 1)
    }

    function IdToLogit(id: Id) : (logit: Logit)
      reads this
      reads this.Logits
      requires ValidTokensIdsLogits()
      requires id in Ids
      ensures logit in Logits[0..Logits.Length]
      ensures ValidTokensIdsLogits()
    {
      Logits[id]
    }

    function TokenToLogit(token: Token): (logit: Logit)
      reads this
      reads this.Logits
      requires ValidTokensIdsLogits()
      requires token in Tokens
      ensures ValidTokensIdsLogits()
    {
      IdToLogit(TokenToId(token))
    }

    function TokensToLogits(tokens: seq<Token>): (logits: seq<Logit>)
      reads this
      reads this.Logits
      requires ValidTokensIdsLogits()
      requires |tokens| > 0
      requires forall token: Token :: token in tokens ==> token in Tokens
      ensures ValidTokensIdsLogits()
    {
      if (|tokens| == 1) then [TokenToLogit(tokens[0])]
      else [TokenToLogit(tokens[0])] + TokensToLogits(tokens[1..])
    }

    function IdsToLogits(ids: seq<Id>): (logits: seq<Logit>)
      reads this
      reads this.Logits
      requires ValidTokensIdsLogits()
      requires |ids| > 0
      requires forall id: Id :: id in ids ==> id in Ids
      ensures ValidTokensIdsLogits()
    {
      if (|ids| == 1) then [IdToLogit(ids[0])]
      else [IdToLogit(ids[0])] + IdsToLogits(ids[1..])
    }

    method MaskToken(token: Token)
      modifies this.Logits
      requires ValidTokensIdsLogits()
      requires token in Tokens
      ensures ValidTokensIdsLogits()
      ensures Tokens[TokenToId(token)] == token
      ensures IsMasked(token)
      ensures forall t: Token :: t in Tokens && t != token ==> Logits[TokenToId(t)] == old(Logits[TokenToId(t)])
    {
      var id := TokenToId(token);
      Logits[id] := -1000000000.0;
    }

    method MaskTokens(tokens: seq<Token>)
      modifies this.Logits
      requires ValidTokensIdsLogits()
      requires |tokens| > 0
      requires forall token :: token in tokens ==> token in Tokens
      ensures ValidTokensIdsLogits()
      ensures forall t :: t in tokens ==> IsMasked(t)
      ensures forall t :: t in Tokens && !(t in tokens) ==> Logits[TokenToId(t)] == old(Logits[TokenToId(t)])
    {
      var N := |tokens|;
      var i := 0;
      while i < N
        invariant 0 <= i <= N
        invariant ValidTokensIdsLogits()
        invariant forall j :: 0 <= j < i ==> IsMasked(tokens[j])
        invariant forall t :: t in Tokens && !(t in tokens[..i]) ==> Logits[TokenToId(t)] == old(Logits[TokenToId(t)])
        decreases N - i
      {
        MaskToken(tokens[i]);
        i := i + 1;
      }
    }

    method MaskTokensExcept(tokens: seq<Token>)
      modifies this.Logits
      requires ValidTokensIdsLogits()
      requires |tokens| > 0
      requires forall token :: token in tokens ==> token in Tokens
      ensures ValidTokensIdsLogits()
      ensures forall t :: t in Tokens && !(t in tokens) ==> IsMasked(t)
      ensures forall t :: t in tokens ==> Logits[TokenToId(t)] == old(Logits[TokenToId(t)])
    {
      var toMask: seq<Token> := [];
      var N := |Tokens|;
      var i := 0;

      while i < N
        invariant 0 <= i <= N
        invariant ValidTokensIdsLogits()
        invariant forall j :: 0 <= j < i && !(Tokens[j] in tokens) ==> Tokens[j] in toMask
        invariant forall j :: 0 <= j < i && Tokens[j] in tokens ==> !(Tokens[j] in toMask)
        invariant forall t: Token :: t in toMask ==> t !in tokens && t in Tokens
        decreases N - i
      {
        if !(Tokens[i] in tokens) {
          toMask := toMask + [Tokens[i]];
        }
        i := i + 1;
      }

      if |toMask|> 0 {
        MaskTokens(toMask);
      }
    }

    predicate IsMasked(token: Token)
      reads this
      reads this.Logits
      requires ValidTokensIdsLogits()
      requires token in Tokens
      ensures ValidTokensIdsLogits()
    {
      Logits[TokenToId(token)] == -1000000000.0
    }

    predicate HasUnmaskedToken()
      reads this
      reads this.Logits
      requires ValidTokensIdsLogits()
      ensures ValidTokensIdsLogits()
    {
      exists t: Token :: t in Tokens && !IsMasked(t)
    }

    method {:extern} {:axiom} GenerateLogits(input: Prefix)
      modifies this.Logits
      requires ValidTokensIdsLogits()
      ensures ValidTokensIdsLogits()

    method {:extern} {:axiom} AppendTaskGuidance(guidance: string)
      requires ValidTokensIdsLogits()
      ensures ValidTokensIdsLogits()

    // Persistently down-weight `token` as a next-token at position `prefix`, so a
    // later regeneration at this position (e.g. after a grounding rollback) picks
    // a DIFFERENT token instead of looping. Host-state only: does NOT change the
    // current logits, so the Logits-array contract is preserved. (Faithful analog
    // of IterGen's recurrence_penalty on a backtracked trace position.)
    method {:extern} {:axiom} PenalizeTriedTokenAt(prefix: Prefix, token: Token)
      requires ValidTokensIdsLogits()
      ensures ValidTokensIdsLogits()

    method {:extern} {:axiom} ResetOracleTrie()
      requires ValidTokensIdsLogits()
      ensures ValidTokensIdsLogits()

    method {:extern} {:axiom} CarsAdvanceTrieAndAdjustScores(parser: Parser, prefix: Prefix, constrainFirst: bool) returns (ok: bool)
      modifies this.Logits
      requires ValidTokensIdsLogits()
      requires parser.IsValidPrefix(prefix) || |prefix| == 0
      ensures ValidTokensIdsLogits()

    method {:extern} {:axiom} RejectLastInTrie()
      requires ValidTokensIdsLogits()
      ensures ValidTokensIdsLogits()

    method {:extern} {:axiom} ApplyTraceRecurrence(factor: real)
      modifies this.Logits
      requires ValidTokensIdsLogits()
      requires factor > 0.0 && factor <= 1.0
      ensures ValidTokensIdsLogits()

    method {:extern} {:axiom} ChooseNextToken() returns (token: Token)
      requires ValidTokensIdsLogits()
      ensures token in Tokens
      ensures !IsMasked(token)
      ensures ValidTokensIdsLogits()

    method {:extern} {:axiom} ChooseNextTokenUnconstrained() returns (token: Token)
      ensures token in Tokens
      ensures ValidTokensIdsLogits()

    method {:extern} {:axiom} GenerateUnconstrainedChunk(
      input: Prefix, maxNewTokens: nat, openSpanToken: Token, eosToken: Token
    ) returns (chunk: Prefix, stoppedOnOpenSpan: bool, stoppedOnEos: bool, stepsUsed: nat)
      modifies this.Logits
      requires ValidTokensIdsLogits()
      requires openSpanToken in Tokens
      requires eosToken in Tokens
      ensures ValidTokensIdsLogits()
      ensures |chunk| <= stepsUsed <= maxNewTokens
      ensures forall i :: 0 <= i < |chunk| ==> chunk[i] in Tokens && chunk[i] != eosToken
      ensures !(stoppedOnOpenSpan && stoppedOnEos)
      ensures stoppedOnOpenSpan ==> |chunk| > 0 && chunk[|chunk| - 1] == openSpanToken
      ensures stoppedOnEos ==> stepsUsed == |chunk| + 1
      ensures !stoppedOnEos ==> stepsUsed == |chunk|
      ensures maxNewTokens > 0 ==> stepsUsed > 0

    method {:extern} {:axiom} MaskValidNextAndEos(parser: Parser, prefix: Prefix, eosToken: Token)
      modifies this.Logits
      requires ValidTokensIdsLogits()
      requires parser.IsValidPrefix(prefix)
      requires eosToken in Tokens
      ensures ValidTokensIdsLogits()
      ensures forall t :: t in Tokens && !parser.ValidNextToken(prefix, t) && t != eosToken ==> IsMasked(t)
      // Stopping (picking eosToken) is only left selectable when the prefix is
      // already a complete query, or when it is a genuine dead end (no legal
      // continuation exists). Otherwise eosToken itself gets masked, same as
      // any other grammar-invalid token. This is what actually closes the
      // "stop before writing anything" hole: earlier this method force-allowed
      // eosToken unconditionally, so a strategy could halt on its very first
      // token before appending anything to the answer.
      ensures !(parser.IsCompletePrefix(prefix) || parser.ValidNextTokenCount(prefix) == 0) ==> IsMasked(eosToken)

    method {:extern} {:axiom} BoostValidNextAndEos(parser: Parser, prefix: Prefix, amount: real, eosToken: Token)
      modifies this.Logits
      requires ValidTokensIdsLogits()
      requires parser.IsValidPrefix(prefix)
      requires eosToken in Tokens
      requires amount >= 0.0 && amount <= 100000000.0
      ensures ValidTokensIdsLogits()

    // Grounding predicate. Returns whether every identifier-like token in `text`
    // appears in the support set the host derives from the task input (the
    // prompt); returns true when the prompt contains no recognizable support set.
    // Pure with respect to the Dafny heap (reads nothing): the support set lives
    // in host state, not in any Dafny field. Implemented in the host language.
    predicate {:extern} {:axiom} SpanGrounded(text: string)

    // Prompt-visible duplicate predicate. Returns whether normalized `text`
    // appears as a candidate span in the current prompt/instruction context.
    // Intended for no-gold duplicate/exemplar checks in tasks such as SMILES:
    // the host may inspect prompt-visible examples and rolling prompt suffixes,
    // but never gold labels, scorer state, or evaluator results.
    predicate {:extern} {:axiom} SpanAppearsInPrompt(text: string)

    // Prompt-derived resemblance score in [0,1]: how structurally similar the
    // normalized `text` is to the example spans shown in the current prompt.
    // The host may inspect only prompt-visible examples and compute similarity with
    // generic tooling; it never reads gold labels, scorer state, held-out data, or
    // evaluator results. Higher means more similar to the shown examples.
    function {:extern} {:axiom} SpanResemblanceToPromptExamples(text: string): real

    // Locate the FIRST identifier-like token in `unitTokens` whose rendered text
    // is out-of-schema for the current example. The membership signal is identical
    // to SpanGrounded (same prompt-derived support set, same identifier filtering);
    // the addition is `idx`, the index of that token WITHIN `unitTokens`, so a
    // rollback can penalize THAT token instead of the unit's first token. `found`
    // is false (and `idx` meaningless) when every identifier is in the support set
    // or no support set was parsed. Pure with respect to the Dafny heap (reads only
    // host state); implemented in the host language. `unitTokens` is rendered by
    // concatenation, exactly as RenderPrefix renders it.
    method {:extern} {:axiom} FirstUngroundedIdentifierTokenIdx(unitTokens: Prefix)
        returns (found: bool, idx: nat)
      ensures found ==> idx < |unitTokens|
  }

  class Parser {
    predicate {:extern} {:axiom} IsValidPrefix(prefix: Prefix)
      ensures forall k: nat :: 0 <= k < |prefix| - 1 ==> IsValidPrefix(prefix[k..])

    predicate {:extern} {:axiom} IsCompletePrefix(prefix: Prefix)
      ensures IsValidPrefix(prefix)

    function {:extern} {:axiom} ValidNextTokenCount(prefix: Prefix): nat
      requires IsValidPrefix(prefix)
      ensures ValidNextTokenCount(prefix) == |ValidNextTokens(prefix)|

    predicate IsDeadPrefix(prefix: Prefix)
    {
      !IsCompletePrefix(prefix) && ValidNextTokenCount(prefix) == 0
    }

    predicate {:extern} {:axiom} ValidNextToken(prefix: Prefix, token: Token)
      requires IsValidPrefix(prefix)
      ensures ValidNextToken(prefix, token) <==> token in ValidNextTokens(prefix)

    function {:extern} {:axiom} ValidNextTokens(prefix: Prefix): seq<Token>
      requires IsValidPrefix(prefix)
      ensures forall t :: t in ValidNextTokens(prefix) ==> IsValidPrefix(prefix + [t])
      ensures (IsCompletePrefix(prefix) || |ValidNextTokens(prefix)| > 0)

    method {:extern} {:axiom} ParseG(input: string) returns (isSuccess: bool)

    // Number of schema-bearing grammar symbols (table_ref / column_ref) that have
    // COMPLETED within `prefix`, read from the parser's SymbolPosMap side-record
    // (IterGen's mechanism, ported into the vendored incremental parser). The
    // record is a pure side-effect of parsing: it never changes which tokens are
    // accepted, so reading this count cannot alter decode for any caller. Used as
    // a mid-query unit boundary — when the count rises, one more table/column name
    // just finished, so the host can ground-check it without waiting for the whole
    // query to parse. Implemented in the host language.
    function {:extern} {:axiom} CompletedSchemaSymbolCount(prefix: Prefix): nat
      requires IsValidPrefix(prefix)

    // IterGen-style: count of completed `unit` symbols whose end char > afterChar
    // on the structured parse text. Empty unit "" sums table_ref+column_ref.
    function {:extern} {:axiom} CompletedSymbolCount(prefix: Prefix, unit: string, afterChar: nat): nat
      requires IsValidPrefix(prefix)

    function {:extern} {:axiom} StructuredCharLength(prefix: Prefix): nat
      requires IsValidPrefix(prefix)

    function {:extern} {:axiom} SymbolStartTokenIndex(prefix: Prefix, unit: string, which: nat): nat
      requires IsValidPrefix(prefix)
      ensures 0 <= SymbolStartTokenIndex(prefix, unit, which) <= |prefix|

    function {:extern} {:axiom} SymbolEndTokenIndex(prefix: Prefix, unit: string, which: nat): nat
      requires IsValidPrefix(prefix)
      ensures 0 <= SymbolEndTokenIndex(prefix, unit, which) <= |prefix|

    function {:extern} {:axiom} RenderSymbol(prefix: Prefix, unit: string, which: nat): string
      requires IsValidPrefix(prefix)
  }

  function Contains(s: string, sub: string): bool
  {
    exists i, j :: 0 <= i <= j <= |s| && s[i..j] == sub
  }

  // Flatten a token prefix back to the plain string it renders to. Used to test
  // a span's closing delimiter by its rendered SURFACE TEXT rather than by exact
  // final-token identity, so a span that closed as a split '>'+'>' or a
  // space-prefixed ' >>' token is still recognized as already-closed.
  function RenderPrefix(p: Prefix): string
  {
    if |p| == 0 then ""
    else p[0] + RenderPrefix(p[1..])
  }

  predicate RenderedEndsWith(p: Prefix, suf: string)
  {
    var s := RenderPrefix(p);
    |s| >= |suf| && s[|s| - |suf|..] == suf
  }

  // ---------------------------------------------------------------------------
  // Span contract: what the OUTPUT token sequence is allowed to look like.
  //
  // Walking the output left to right, we are either in free text or inside a
  // `<<` span. Free-text tokens carry no delimiter text. A span closes either
  // because its own content ends in `>>` (GSM: the grammar includes the closer)
  // or because an explicit `>>` token follows complete content (SQL, SMILES).
  // A closed span must be a complete parse. Anything else is Bad.
  // ---------------------------------------------------------------------------
  datatype SpanState = Bad | Outside | Inside(content: Prefix)

  predicate DelimFree(t: Token)
  {
    !Contains(t, "<<") && !Contains(t, ">>")
  }

  ghost function SpanStep(parser: Parser, st: SpanState, t: Token): SpanState
  {
    match st
    case Bad => Bad
    case Outside => if t == "<<" then Inside([]) else if DelimFree(t) then Outside else Bad
    case Inside(c) =>
      if t == ">>" && parser.IsCompletePrefix(c) && !RenderedEndsWith(c, ">>") then Outside
      else if !parser.IsValidPrefix(c + [t]) then Bad
      else if RenderedEndsWith(c + [t], ">>") then Outside
      else Inside(c + [t])
  }

  ghost function SpanStateOf(parser: Parser, g: Prefix): SpanState
    decreases |g|
  {
    if |g| == 0 then Outside
    else SpanStep(parser, SpanStateOf(parser, g[..|g| - 1]), g[|g| - 1])
  }

  // The decoder's tracked state agrees with the output.
  //   outside: the whole output walks to Outside (every span so far closed and complete);
  //   inside : output == history + ["<<"] + current, history walks to Outside, current is a valid prefix.
  ghost predicate Tied(parser: Parser, g: Prefix, inside: bool, cur: Prefix)
  {
    if inside then
      |cur| + 1 <= |g| &&
      g[|g| - |cur|..] == cur &&
      g[|g| - |cur| - 1] == "<<" &&
      SpanStateOf(parser, g[..|g| - |cur| - 1]) == Outside &&
      parser.IsValidPrefix(cur)
    else
      cur == [] && SpanStateOf(parser, g) == Outside
  }

  // Assumptions about the grammar side, made true by the Python parser wrapper:
  // a valid prefix stays valid when shortened, and once content ends in `>>`
  // it is complete and nothing can follow it.
  lemma {:axiom} ValidPrefixesAreClosed(parser: Parser, p: Prefix, k: nat)
    requires parser.IsValidPrefix(p) && k <= |p|
    ensures parser.IsValidPrefix(p[..k])

  lemma {:axiom} CloserIsTerminal(parser: Parser, p: Prefix)
    requires parser.IsValidPrefix(p) && RenderedEndsWith(p, ">>")
    ensures parser.IsCompletePrefix(p)
    ensures forall t: Token :: !parser.IsValidPrefix(p + [t])

  lemma SpanStateAppend(parser: Parser, g: Prefix, t: Token)
    ensures SpanStateOf(parser, g + [t]) == SpanStep(parser, SpanStateOf(parser, g), t)
  {
    assert (g + [t])[..|g + [t]| - 1] == g;
  }

  // Walking `hist + ["<<"] + cur` for valid `cur`: still inside with content `cur`,
  // unless `cur` itself ends in `>>`, in which case the span has closed.
  lemma InsideRun(parser: Parser, hist: Prefix, cur: Prefix)
    requires SpanStateOf(parser, hist) == Outside
    requires parser.IsValidPrefix(cur)
    ensures SpanStateOf(parser, hist + ["<<"] + cur) ==
            (if RenderedEndsWith(cur, ">>") then Outside else Inside(cur))
    decreases |cur|
  {
    if |cur| == 0 {
      SpanStateAppend(parser, hist, "<<");
      assert hist + ["<<"] + cur == hist + ["<<"];
      assert RenderPrefix(cur) == "";
    } else {
      var init := cur[..|cur| - 1];
      var t := cur[|cur| - 1];
      ValidPrefixesAreClosed(parser, cur, |cur| - 1);
      InsideRun(parser, hist, init);
      assert cur == init + [t];
      if RenderedEndsWith(init, ">>") {
        CloserIsTerminal(parser, init);
        assert false;
      }
      assert hist + ["<<"] + cur == (hist + ["<<"] + init) + [t];
      SpanStateAppend(parser, hist + ["<<"] + init, t);
      if t == ">>" && parser.IsCompletePrefix(init) {
        // explicit-closer reading and content reading agree: both give Outside
        assert RenderedEndsWith(cur, ">>") by { RenderAppend(init, t); }
      }
    }
  }

  ghost predicate AllDelimFree(ts: Prefix) { forall i :: 0 <= i < |ts| ==> DelimFree(ts[i]) }

  // Free text appended outside a span leaves us outside a span.
  lemma FreeRun(parser: Parser, g: Prefix, ts: Prefix)
    requires SpanStateOf(parser, g) == Outside
    requires AllDelimFree(ts)
    ensures SpanStateOf(parser, g + ts) == Outside
    decreases |ts|
  {
    if |ts| == 0 {
      assert g + ts == g;
    } else {
      var init := ts[..|ts| - 1];
      FreeRun(parser, g, init);
      assert g + ts == (g + init) + [ts[|ts| - 1]];
      var t := ts[|ts| - 1];
      assert DelimFree(t);
      if t == "<<" { assert t[0..|t|] == "<<"; assert Contains(t, "<<"); }
      SpanStateAppend(parser, g + init, ts[|ts| - 1]);
    }
  }

  // Closing an open span whose content is complete puts us back outside.
  lemma CloseTied(parser: Parser, g: Prefix, cur: Prefix)
    requires Tied(parser, g, true, cur)
    requires parser.IsCompletePrefix(cur)
    ensures RenderedEndsWith(cur, ">>") ==> Tied(parser, g, false, [])
    ensures !RenderedEndsWith(cur, ">>") ==> Tied(parser, g + [">>"], false, [])
  {
    var hist := g[..|g| - |cur| - 1];
    InsideRun(parser, hist, cur);
    assert g == hist + ["<<"] + cur;
    if !RenderedEndsWith(cur, ">>") {
      SpanStateAppend(parser, g, ">>");
    }
  }

  lemma AppendTied(parser: Parser, g: Prefix, cur: Prefix, t: Token)
    requires Tied(parser, g, true, cur)
    requires parser.IsValidPrefix(cur + [t])
    ensures Tied(parser, g + [t], true, cur + [t])
  {
    var g2 := g + [t];
    var c2 := cur + [t];
    assert g2[|g2| - |c2|..] == c2;
    assert g2[..|g2| - |c2| - 1] == g[..|g| - |cur| - 1];
  }

  lemma OpenTied(parser: Parser, g: Prefix)
    requires Tied(parser, g, false, [])
    requires parser.IsValidPrefix([])
    ensures Tied(parser, g + ["<<"], true, [])
  {
    var g2 := g + ["<<"];
    assert g2[..|g2| - 1] == g;
    assert g2[|g2|..] == [];
  }

  // Replacing the open span's content (rollback, regenerate) keeps the tie.
  lemma ReplaceTied(parser: Parser, g: Prefix, cur: Prefix, newCur: Prefix)
    requires Tied(parser, g, true, cur)
    requires parser.IsValidPrefix(newCur)
    ensures Tied(parser, g[..|g| - |cur|] + newCur, true, newCur)
  {
    var stable := g[..|g| - |cur|];
    var g2 := stable + newCur;
    assert g2[|g2| - |newCur|..] == newCur;
    assert g2[..|g2| - |newCur| - 1] == g[..|g| - |cur| - 1];
    assert g2[|g2| - |newCur| - 1] == g[|g| - |cur| - 1];
  }

  lemma RenderAppend(p: Prefix, t: Token)
    ensures RenderPrefix(p + [t]) == RenderPrefix(p) + t
    decreases |p|
  {
    if |p| == 0 {
      assert p + [t] == [t];
      assert RenderPrefix([t]) == t + RenderPrefix([]);
    } else {
      assert (p + [t])[0] == p[0];
      assert (p + [t])[1..] == p[1..] + [t];
      RenderAppend(p[1..], t);
    }
  }

  class CSDHelpers {
    var cost: int

    constructor()
      ensures cost == 0
    {
      cost := 0;
    }

    method AppendTaskGuidance(lm: LM, guidance: string)
      requires lm.ValidTokensIdsLogits()
      ensures lm.ValidTokensIdsLogits()
      ensures cost == old(cost)
    {
      lm.AppendTaskGuidance(guidance);
    }

    method PrefixAppearsInPrompt(lm: LM, prefix: Prefix) returns (appears: bool)
      ensures appears == lm.SpanAppearsInPrompt(RenderPrefix(prefix))
      ensures cost == old(cost)
    {
      appears := lm.SpanAppearsInPrompt(RenderPrefix(prefix));
    }

    method PrefixResemblesPromptExamples(lm: LM, prefix: Prefix) returns (score: real)
      ensures score == lm.SpanResemblanceToPromptExamples(RenderPrefix(prefix))
      ensures cost == old(cost)
    {
      score := lm.SpanResemblanceToPromptExamples(RenderPrefix(prefix));
    }

    method UnconstrainedStep(lm: LM, prompt: Prefix, generated: Prefix) returns (next: Token)
      modifies lm.Logits, this
      requires lm.ValidTokensIdsLogits()
      ensures lm.ValidTokensIdsLogits()
      ensures cost == old(cost) + 1
      // Runtime rule (delimiter_hygiene.py): outside a span the sampler bans every
      // token carrying delimiter text except the exact opener.
      ensures forall p: Parser {:trigger SpanStateOf(p, generated)} ::
                SpanStateOf(p, generated) == Outside ==> next == "<<" || DelimFree(next)
      ensures forall p: Parser {:trigger Tied(p, generated + [next], false, [])} ::
                Tied(p, generated, false, []) && next != "<<" ==> Tied(p, generated + [next], false, [])
      ensures forall p: Parser {:trigger Tied(p, generated + [next], true, [])} ::
                Tied(p, generated, false, []) && next == "<<" && p.IsValidPrefix([]) ==> Tied(p, generated + [next], true, [])
    {
      lm.GenerateLogits(prompt + generated);
      next := lm.ChooseNextTokenUnconstrained();
      cost := cost + 1;
      assume {:axiom} forall p: Parser {:trigger SpanStateOf(p, generated)} ::
                SpanStateOf(p, generated) == Outside ==> next == "<<" || DelimFree(next);
      forall p: Parser | Tied(p, generated, false, []) && next != "<<"
        ensures Tied(p, generated + [next], false, [])
      { FreeRun(p, generated, [next]); }
      forall p: Parser | Tied(p, generated, false, []) && next == "<<" && p.IsValidPrefix([])
        ensures Tied(p, generated + [next], true, [])
      { OpenTied(p, generated); }
    }

    method UnconstrainedChunk(
      lm: LM, prompt: Prefix, generated: Prefix, maxChunkTokens: nat, openSpanToken: Token, eosToken: Token
    ) returns (generatedOut: Prefix, stoppedOnOpenSpan: bool, stoppedOnEos: bool, stepsUsed: nat)
      modifies lm.Logits, this
      requires lm.ValidTokensIdsLogits()
      requires openSpanToken in lm.Tokens
      requires eosToken in lm.Tokens
      ensures lm.ValidTokensIdsLogits()
      ensures |generated| <= |generatedOut|
      ensures generatedOut[..|generated|] == generated
      ensures |generatedOut| <= |generated| + stepsUsed
      ensures stepsUsed <= maxChunkTokens
      ensures cost == old(cost) + stepsUsed
      ensures !(stoppedOnOpenSpan && stoppedOnEos)
      ensures stoppedOnOpenSpan ==> |generatedOut| > |generated| && generatedOut[|generatedOut| - 1] == openSpanToken
      ensures stoppedOnEos ==> |generatedOut| + 1 == |generated| + stepsUsed
      ensures !stoppedOnEos ==> |generatedOut| == |generated| + stepsUsed
      ensures maxChunkTokens > 0 ==> stepsUsed > 0
    {
      var chunk: Prefix;
      chunk, stoppedOnOpenSpan, stoppedOnEos, stepsUsed := lm.GenerateUnconstrainedChunk(
        prompt + generated, maxChunkTokens, openSpanToken, eosToken
      );
      generatedOut := generated + chunk;
      cost := cost + stepsUsed;
    }

    // Generates one symbol worth of tokens via a multi-token LM call,
    // then accepts the longest parser-valid prefix of the emitted chunk.
    method ConstrainedSymbol(
      lm: LM, parser: Parser, constrainedPrompt: Prefix, currentConstrained: Prefix,
      maxSymbolTokens: nat, eosToken: Token
    ) returns (currentOut: Prefix, hitEos: bool, stepsUsed: nat)
      modifies lm.Logits, this
      requires lm.ValidTokensIdsLogits()
      requires parser.IsValidPrefix(currentConstrained)
      requires "<<" in lm.Tokens
      requires eosToken in lm.Tokens
      requires maxSymbolTokens > 0
      ensures lm.ValidTokensIdsLogits()
      ensures parser.IsValidPrefix(currentOut)
      ensures |currentOut| >= |currentConstrained|
      ensures |currentOut| <= |currentConstrained| + stepsUsed
      ensures stepsUsed <= maxSymbolTokens
      ensures stepsUsed > 0
      ensures cost == old(cost) + stepsUsed
    {
      var chunk: Prefix;
      var stoppedOnOpen: bool;
      var stoppedOnEos: bool;
      chunk, stoppedOnOpen, stoppedOnEos, stepsUsed := lm.GenerateUnconstrainedChunk(
        constrainedPrompt + currentConstrained, maxSymbolTokens, "<<", eosToken
      );
      cost := cost + stepsUsed;
      hitEos := stoppedOnEos;
      currentOut := currentConstrained;
      var i := 0;
      while i < |chunk|
        invariant 0 <= i <= |chunk|
        invariant parser.IsValidPrefix(currentOut)
        invariant |currentOut| >= |currentConstrained|
        invariant |currentOut| <= |currentConstrained| + i
        decreases |chunk| - i
      {
        var tok := chunk[i];
        var extended := currentOut + [tok];
        if parser.IsValidPrefix(extended) && !parser.IsDeadPrefix(extended) {
          currentOut := extended;
        } else {
          break;
        }
        i := i + 1;
      }
    }

    method ConstrainedSymbolInGenerated(
      lm: LM, parser: Parser, constrainedPrompt: Prefix, generated: Prefix,
      currentConstrained: Prefix, maxSymbolTokens: nat, eosToken: Token
    ) returns (generatedOut: Prefix, currentOut: Prefix, hitEos: bool, stepsUsed: nat)
      modifies lm.Logits, this
      requires lm.ValidTokensIdsLogits()
      requires parser.IsValidPrefix(currentConstrained)
      requires |currentConstrained| <= |generated|
      requires "<<" in lm.Tokens
      requires eosToken in lm.Tokens
      requires maxSymbolTokens > 0
      ensures lm.ValidTokensIdsLogits()
      ensures parser.IsValidPrefix(currentOut)
      ensures |currentOut| >= |currentConstrained|
      ensures |currentOut| <= |currentConstrained| + stepsUsed
      ensures |generatedOut| <= |generated| + stepsUsed
      ensures |currentOut| <= |generatedOut|
      ensures generatedOut[|generatedOut| - |currentOut|..] == currentOut
      ensures stepsUsed <= maxSymbolTokens
      ensures stepsUsed > 0
      ensures cost == old(cost) + stepsUsed
    {
      var stablePrefix := generated[..|generated| - |currentConstrained|];
      currentOut, hitEos, stepsUsed := ConstrainedSymbol(
        lm, parser, constrainedPrompt, currentConstrained, maxSymbolTokens, eosToken
      );
      generatedOut := stablePrefix + currentOut;
      assert |stablePrefix| == |generated| - |currentConstrained|;
      assert |generatedOut| == |stablePrefix| + |currentOut|;
      assert |generatedOut| <= |generated| + stepsUsed;
      assert |currentOut| <= |generatedOut|;
      assert generatedOut[|generatedOut| - |currentOut|..] == currentOut;
    }

    method OpenConstrainedSpan(lm: LM, generated: Prefix) returns (generatedOut: Prefix, insideOut: bool, currentOut: Prefix)
      modifies this
      requires lm.ValidTokensIdsLogits()
      requires "<<" in lm.Tokens
      ensures lm.ValidTokensIdsLogits()
      ensures generatedOut == generated + ["<<"]
      ensures insideOut
      ensures currentOut == []
      ensures cost == old(cost) + 1
      ensures forall p: Parser {:trigger Tied(p, generatedOut, true, [])} ::
                Tied(p, generated, false, []) && p.IsValidPrefix([]) ==> Tied(p, generatedOut, true, [])
    {
      forall p: Parser | Tied(p, generated, false, []) && p.IsValidPrefix([])
        ensures Tied(p, generated + ["<<"], true, [])
      { OpenTied(p, generated); }
      generatedOut := generated + ["<<"];
      insideOut := true;
      currentOut := [];
      cost := cost + 1;
    }


    method EnterObservedConstrainedSpan(lm: LM, generated: Prefix) returns (generatedOut: Prefix, insideOut: bool, currentOut: Prefix)
      modifies this
      requires lm.ValidTokensIdsLogits()
      ensures lm.ValidTokensIdsLogits()
      ensures generatedOut == generated
      ensures insideOut
      ensures currentOut == []
      ensures cost == old(cost)
    {
      generatedOut := generated;
      insideOut := true;
      currentOut := [];
    }

    method AppendConstrainedToken(
      lm: LM, parser: Parser, generated: Prefix, currentConstrained: Prefix, next: Token
    ) returns (generatedOut: Prefix, insideOut: bool, currentOut: Prefix)
      modifies this
      requires lm.ValidTokensIdsLogits()
      requires parser.IsValidPrefix(currentConstrained)
      requires next in lm.Tokens
      requires parser.IsValidPrefix(currentConstrained + [next])
      ensures lm.ValidTokensIdsLogits()
      ensures generatedOut == generated + [next]
      ensures insideOut
      ensures currentOut == currentConstrained + [next]
      ensures parser.IsValidPrefix(currentOut)
      ensures cost == old(cost)
      ensures Tied(parser, generated, true, currentConstrained) ==> Tied(parser, generatedOut, true, currentOut)
    {
      if Tied(parser, generated, true, currentConstrained) { AppendTied(parser, generated, currentConstrained, next); }
      generatedOut := generated + [next];
      insideOut := true;
      currentOut := currentConstrained + [next];
    }

    method CloseConstrainedSpan(
      lm: LM, parser: Parser, generated: Prefix, currentConstrained: Prefix
    ) returns (generatedOut: Prefix, insideOut: bool, currentOut: Prefix)
      modifies this
      requires lm.ValidTokensIdsLogits()
      requires parser.IsCompletePrefix(currentConstrained)
      requires ">>" in lm.Tokens
      ensures lm.ValidTokensIdsLogits()
      ensures RenderedEndsWith(currentConstrained, ">>") ==>
              generatedOut == generated
      ensures !RenderedEndsWith(currentConstrained, ">>") ==>
              generatedOut == generated + [">>"]
      ensures !insideOut
      ensures currentOut == []
      ensures cost == old(cost) + 1
      ensures Tied(parser, generated, true, currentConstrained) ==> Tied(parser, generatedOut, false, [])
    {
      if Tied(parser, generated, true, currentConstrained) { CloseTied(parser, generated, currentConstrained); }
      if RenderedEndsWith(currentConstrained, ">>") {
        generatedOut := generated;
      } else {
        generatedOut := generated + [">>"];
      }
      insideOut := false;
      currentOut := [];
      cost := cost + 1;
    }


    method CarsTrieStep(
      lm: LM, parser: Parser, prompt: Prefix, cur: Prefix, eosToken: Token, constrainFirst: bool
    ) returns (next: Token, ok: bool)
      modifies lm.Logits, this
      requires lm.ValidTokensIdsLogits()
      requires parser.IsValidPrefix(cur)
      requires eosToken in lm.Tokens
      ensures lm.ValidTokensIdsLogits()
      ensures next in lm.Tokens
      ensures cost == old(cost) + 1
    {
      lm.GenerateLogits(prompt + cur);
      ok := lm.CarsAdvanceTrieAndAdjustScores(parser, cur, constrainFirst);
      if !ok {
        next := eosToken;
        cost := cost + 1;
        return;
      }
      next := lm.ChooseNextTokenUnconstrained();
      cost := cost + 1;
    }

    method RejectLastInTrieHelper(lm: LM)
      requires lm.ValidTokensIdsLogits()
      ensures lm.ValidTokensIdsLogits()
      ensures cost == old(cost)
    {
      lm.RejectLastInTrie();
    }

    method ApplyTraceRecurrenceHelper(lm: LM, factor: real)
      modifies lm.Logits, this
      requires lm.ValidTokensIdsLogits()
      requires factor > 0.0 && factor <= 1.0
      ensures lm.ValidTokensIdsLogits()
      ensures cost == old(cost)
    {
      lm.ApplyTraceRecurrence(factor);
    }

    // IterGen forward(unit, num): opportunistic steps until `num` new completions
    // of `unit` (empty unit = SQL table_ref+column_ref aggregate), then crop to
    // the finished unit end (cursor semantics without KV rewind).
    method ForwardUntilSymbol(
      lm: LM, parser: Parser, prompt: Prefix, cur: Prefix, eosToken: Token,
      unit: string, num: nat, budget: nat
    ) returns (out: Prefix)
      modifies lm.Logits, this
      requires lm.ValidTokensIdsLogits()
      requires parser.IsValidPrefix(cur)
      requires eosToken in lm.Tokens
      ensures lm.ValidTokensIdsLogits()
      ensures parser.IsValidPrefix(out)
      ensures |out| <= |cur| + budget
      ensures old(cost) <= cost <= old(cost) + budget
    {
      out := cur;
      if num == 0 || budget == 0 {
        return;
      }
      var baseline := parser.StructuredCharLength(cur);
      var steps := 0;
      while steps < budget
        invariant lm.ValidTokensIdsLogits()
        invariant parser.IsValidPrefix(out)
        invariant 0 <= steps <= budget
        invariant |out| <= |cur| + steps
        invariant cost == old(cost) + steps
        decreases budget - steps
      {
        if parser.CompletedSymbolCount(out, unit, baseline) >= num {
          break;
        }
        var next, wasConstrained := ConfidenceGatedStep(lm, parser, prompt, out, eosToken);
        steps := steps + 1;
        if next == eosToken {
          break;
        }
        out := out + [next];
        // GSM CRANE spans close with ">>". CompletedSymbolCount(unit="start") stays
        // 0 on the GSM symbol map (it tracks var/VARIABLE, not Lark start), so without
        // this break ForwardUntilSymbol burns the remaining budget under a degenerate
        // mask and emits token-id-0 bangs — never returning to unconstrained CoT.
        // Domains that never emit ">>" (SQL/SMILES) are unchanged.
        if Contains(next, ">>") || RenderedEndsWith(out, ">>") {
          break;
        }
      }
      // Crop to end of the last newly completed unit when possible (IterGen cursor).
      if parser.CompletedSymbolCount(out, unit, baseline) >= num {
        var total := parser.CompletedSymbolCount(out, unit, 0);
        if total > 0 {
          var endIdx := parser.SymbolEndTokenIndex(out, unit, total - 1);
          if endIdx <= |out| {
            ValidPrefixesAreClosed(parser, out, endIdx);
            out := out[..endIdx];
          }
        }
      }
    }

    method BackwardToSymbol(
      parser: Parser, cur: Prefix, unit: string, num: nat
    ) returns (truncated: Prefix)
      requires parser.IsValidPrefix(cur)
      ensures parser.IsValidPrefix(truncated)
      ensures |truncated| <= |cur|
      ensures cost == old(cost)
    {
      if |cur| == 0 || num == 0 {
        truncated := cur;
        return;
      }
      var total := parser.CompletedSymbolCount(cur, unit, 0);
      if total < num {
        truncated := cur;
        return;
      }
      var which := total - num;
      var idx := parser.SymbolStartTokenIndex(cur, unit, which);
      ValidPrefixesAreClosed(parser, cur, idx);
      truncated := cur[..idx];
    }

    method ViewLastSymbol(parser: Parser, cur: Prefix, unit: string) returns (text: string)
      requires parser.IsValidPrefix(cur)
      ensures cost == old(cost)
    {
      var total := parser.CompletedSymbolCount(cur, unit, 0);
      if total == 0 {
        text := "";
        return;
      }
      text := parser.RenderSymbol(cur, unit, total - 1);
    }

    method IsAllowedVarText(groups: seq<seq<Token>>, text: string) returns (ok: bool)
      ensures cost == old(cost)
    {
      ok := false;
      if text == "" {
        return;
      }
      var flat := FlattenTokenGroups(groups);
      var i := 0;
      while i < |flat|
        invariant 0 <= i <= |flat|
        decreases |flat| - i
      {
        if flat[i] == text {
          ok := true;
          return;
        }
        i := i + 1;
      }
    }

    method ConstrainedStep(lm: LM, parser: Parser, prompt: Prefix, generated: Prefix, eosToken: Token) returns (next: Token)
      modifies lm.Logits, this
      requires lm.ValidTokensIdsLogits()
      requires parser.IsValidPrefix(generated)
      requires eosToken in lm.Tokens
      ensures lm.ValidTokensIdsLogits()
      ensures next in lm.Tokens
      ensures (next == eosToken) || (parser.ValidNextToken(generated, next))
      ensures (next != eosToken) ==> (forall t: Token :: t in parser.ValidNextTokens(generated + [next]) ==> t in lm.Tokens)
      ensures next == eosToken ==> (parser.IsCompletePrefix(generated) || parser.ValidNextTokenCount(generated) == 0)
      ensures cost == old(cost) + 1
    {
      lm.GenerateLogits(prompt + generated);
      RollbackPreservesTokenInvariant(lm, parser, generated);
      lm.MaskValidNextAndEos(parser, generated, eosToken);
      next := lm.ChooseNextToken();
      if next != eosToken {
        assert !lm.IsMasked(next);
        assert parser.ValidNextToken(generated, next);
        assert parser.IsValidPrefix(generated + [next]);
        ConstrainedStepNextValid(lm, parser, generated, next);
      }
      cost := cost + 1;
    }

    // Dead-end-avoiding constrained step. Plain constrained decoding masks the
    // grammar-invalid next tokens and commits whatever the model samples. But the
    // runtime grammar mask over-approximates (a whitespace-prefixed token can slip
    // through), so the sampled token can land in a prefix that is technically valid
    // yet has ZERO valid continuations -- a "dead" prefix the weak model can never
    // finish. This step uses the runtime-accurate IsDeadPrefix oracle: if the chosen
    // token would create a dead (or invalid) prefix, it masks just that token and
    // RE-SAMPLES from the same logits, up to maxRetries times, instead of committing
    // it. success=false means no non-dead continuation was found within the budget,
    // so the caller can fall back to rollback. This is one-step lookahead: it rules
    // out committing an immediately-dead token, not one whose continuations all die
    // several steps later.
    method DeadEndAvoidingStep(
      lm: LM, parser: Parser, prompt: Prefix, generated: Prefix, eosToken: Token, maxRetries: nat
    ) returns (next: Token, success: bool)
      modifies lm.Logits, this
      requires lm.ValidTokensIdsLogits()
      requires parser.IsValidPrefix(generated)
      requires eosToken in lm.Tokens
      ensures lm.ValidTokensIdsLogits()
      ensures next in lm.Tokens
      ensures success ==>
        (next == eosToken) ||
        (parser.IsValidPrefix(generated + [next]) && !parser.IsDeadPrefix(generated + [next]))
      ensures (success && next != eosToken) ==> parser.ValidNextToken(generated, next)
      ensures (success && next != eosToken) ==>
        (forall t: Token :: t in parser.ValidNextTokens(generated + [next]) ==> t in lm.Tokens)
      ensures cost == old(cost) + 1
    {
      lm.GenerateLogits(prompt + generated);
      RollbackPreservesTokenInvariant(lm, parser, generated);
      lm.MaskValidNextAndEos(parser, generated, eosToken);
      next := lm.ChooseNextToken();
      var tries := 0;
      while next != eosToken
            && (!parser.IsValidPrefix(generated + [next]) || parser.IsDeadPrefix(generated + [next]))
            && tries < maxRetries
        invariant lm.ValidTokensIdsLogits()
        invariant next in lm.Tokens
        invariant !lm.IsMasked(next)
        invariant forall t: Token ::
          t in lm.Tokens && !parser.ValidNextToken(generated, t) && t != eosToken ==> lm.IsMasked(t)
        invariant cost == old(cost)
        decreases maxRetries - tries
      {
        lm.MaskToken(next);
        next := lm.ChooseNextToken();
        tries := tries + 1;
      }
      if next != eosToken {
        // An unmasked, non-eos token cannot be one MaskValidNextAndEos masked,
        // so it is a genuine valid next token.
        assert !parser.ValidNextToken(generated, next) ==> lm.IsMasked(next);
        assert parser.ValidNextToken(generated, next);
        assert parser.IsValidPrefix(generated + [next]);
        RollbackPreservesTokenInvariant(lm, parser, generated);
        ConstrainedStepNextValid(lm, parser, generated, next);
      }
      success := next == eosToken ||
        (parser.IsValidPrefix(generated + [next]) && !parser.IsDeadPrefix(generated + [next]));
      cost := cost + 1;
    }

    method GroupHasValidMember(parser: Parser, prefix: Prefix, group: seq<Token>) returns (anyValid: bool)
      requires parser.IsValidPrefix(prefix)
      ensures anyValid <==> (exists t :: t in group && parser.ValidNextToken(prefix, t))
    {
      anyValid := false;
      var i := 0;
      while i < |group|
        invariant 0 <= i <= |group|
        invariant anyValid <==> (exists j :: 0 <= j < i && parser.ValidNextToken(prefix, group[j]))
        decreases |group| - i
      {
        if parser.ValidNextToken(prefix, group[i]) {
          anyValid := true;
        }
        i := i + 1;
      }
    }

    method BoostValidGroups(lm: LM, parser: Parser, prefix: Prefix, groups: seq<seq<Token>>, amount: real)
      modifies lm.Logits, this
      requires lm.ValidTokensIdsLogits()
      requires parser.IsValidPrefix(prefix)
      requires amount >= 0.0 && amount <= 100000000.0
      ensures lm.ValidTokensIdsLogits()
      ensures cost == old(cost)
    {
      var i := 0;
      while i < |groups|
        invariant 0 <= i <= |groups|
        invariant lm.ValidTokensIdsLogits()
        invariant cost == old(cost)
        decreases |groups| - i
      {
        var anyValid := GroupHasValidMember(parser, prefix, groups[i]);
        if anyValid {
          SafeBoostTokenLogits(lm, groups[i], amount);
        }
        i := i + 1;
      }
    }

    method GroupBoostedConstrainedStep(
      lm: LM, parser: Parser, prompt: Prefix, constrainedPrefix: Prefix,
      groups: seq<seq<Token>>, boostAmount: real, eosToken: Token
    ) returns (next: Token)
      modifies lm.Logits, this
      requires lm.ValidTokensIdsLogits()
      requires parser.IsValidPrefix(constrainedPrefix)
      requires boostAmount >= 0.0 && boostAmount <= 100000000.0
      requires eosToken in lm.Tokens
      ensures lm.ValidTokensIdsLogits()
      ensures next in lm.Tokens
      ensures (next == eosToken) || (parser.ValidNextToken(constrainedPrefix, next))
      ensures (next != eosToken) ==> parser.IsValidPrefix(constrainedPrefix + [next])
      ensures (next != eosToken) ==> (forall t: Token :: t in parser.ValidNextTokens(constrainedPrefix + [next]) ==> t in lm.Tokens)
      ensures next == eosToken ==> (parser.IsCompletePrefix(constrainedPrefix) || parser.ValidNextTokenCount(constrainedPrefix) == 0)
      ensures cost == old(cost) + 1
    {
      lm.GenerateLogits(prompt + constrainedPrefix);
      if |groups| > 0 {
        BoostValidGroups(lm, parser, constrainedPrefix, groups, boostAmount);
      }
      RollbackPreservesTokenInvariant(lm, parser, constrainedPrefix);
      lm.MaskValidNextAndEos(parser, constrainedPrefix, eosToken);
      next := lm.ChooseNextToken();
      if next != eosToken {
        assert !lm.IsMasked(next);
        assert parser.ValidNextToken(constrainedPrefix, next);
        assert parser.IsValidPrefix(constrainedPrefix + [next]);
        ConstrainedStepNextValid(lm, parser, constrainedPrefix, next);
      }
      cost := cost + 1;
    }

    method AdaptiveConstrainedStep(
      lm: LM, parser: Parser, prompt: Prefix, constrainedPrefix: Prefix,
      groups: seq<seq<Token>>, boostAmount: real, narrowThreshold: nat, eosToken: Token
    ) returns (next: Token)
      modifies lm.Logits, this
      requires lm.ValidTokensIdsLogits()
      requires parser.IsValidPrefix(constrainedPrefix)
      requires boostAmount >= 0.0 && boostAmount <= 100000000.0
      requires eosToken in lm.Tokens
      ensures lm.ValidTokensIdsLogits()
      ensures next in lm.Tokens
      ensures (next == eosToken) || (parser.ValidNextToken(constrainedPrefix, next))
      ensures (next != eosToken) ==> parser.IsValidPrefix(constrainedPrefix + [next])
      ensures (next != eosToken) ==> (forall t: Token :: t in parser.ValidNextTokens(constrainedPrefix + [next]) ==> t in lm.Tokens)
      ensures next == eosToken ==> (parser.IsCompletePrefix(constrainedPrefix) || parser.ValidNextTokenCount(constrainedPrefix) == 0)
      ensures cost == old(cost) + 1
    {
      lm.GenerateLogits(prompt + constrainedPrefix);
      if |groups| > 0 {
        var validCount := parser.ValidNextTokenCount(constrainedPrefix);
        if validCount <= narrowThreshold {
          BoostValidGroups(lm, parser, constrainedPrefix, groups, boostAmount);
        }
      }
      RollbackPreservesTokenInvariant(lm, parser, constrainedPrefix);
      lm.MaskValidNextAndEos(parser, constrainedPrefix, eosToken);
      next := lm.ChooseNextToken();
      if next != eosToken {
        assert !lm.IsMasked(next);
        assert parser.ValidNextToken(constrainedPrefix, next);
        assert parser.IsValidPrefix(constrainedPrefix + [next]);
        ConstrainedStepNextValid(lm, parser, constrainedPrefix, next);
      }
      cost := cost + 1;
    }

    method AdaptiveConstrainedStepWithPenalties(
      lm: LM, parser: Parser, prompt: Prefix, constrainedPrefix: Prefix,
      boostGroups: seq<seq<Token>>, boostAmount: real,
      penaltyTokens: seq<Token>, penaltyAmount: real,
      narrowThreshold: nat, eosToken: Token
    ) returns (next: Token)
      modifies lm.Logits, this
      requires lm.ValidTokensIdsLogits()
      requires parser.IsValidPrefix(constrainedPrefix)
      requires boostAmount >= 0.0 && boostAmount <= 100000000.0
      requires penaltyAmount >= 0.0 && penaltyAmount <= 100000000.0
      requires eosToken in lm.Tokens
      ensures lm.ValidTokensIdsLogits()
      ensures next in lm.Tokens
      ensures (next == eosToken) || (parser.ValidNextToken(constrainedPrefix, next))
      ensures (next != eosToken) ==> parser.IsValidPrefix(constrainedPrefix + [next])
      ensures (next != eosToken) ==> (forall t: Token :: t in parser.ValidNextTokens(constrainedPrefix + [next]) ==> t in lm.Tokens)
      ensures next == eosToken ==> (parser.IsCompletePrefix(constrainedPrefix) || parser.ValidNextTokenCount(constrainedPrefix) == 0)
      ensures cost == old(cost) + 1
    {
      lm.GenerateLogits(prompt + constrainedPrefix);
      if |boostGroups| > 0 {
        var validCount := parser.ValidNextTokenCount(constrainedPrefix);
        if validCount <= narrowThreshold {
          BoostValidGroups(lm, parser, constrainedPrefix, boostGroups, boostAmount);
        }
      }
      SafePenalizeTokenLogits(lm, penaltyTokens, penaltyAmount);
      RollbackPreservesTokenInvariant(lm, parser, constrainedPrefix);
      lm.MaskValidNextAndEos(parser, constrainedPrefix, eosToken);
      next := lm.ChooseNextToken();
      if next != eosToken {
        assert !lm.IsMasked(next);
        assert parser.ValidNextToken(constrainedPrefix, next);
        assert parser.IsValidPrefix(constrainedPrefix + [next]);
        ConstrainedStepNextValid(lm, parser, constrainedPrefix, next);
      }
      cost := cost + 1;
    }

    method PenalizedConstrainedStep(
      lm: LM, parser: Parser, prompt: Prefix, constrainedPrefix: Prefix,
      tokensToPenalize: seq<Token>, penaltyAmount: real, eosToken: Token
    ) returns (next: Token)
      modifies lm.Logits, this
      requires lm.ValidTokensIdsLogits()
      requires parser.IsValidPrefix(constrainedPrefix)
      requires forall t :: t in tokensToPenalize ==> t in lm.Tokens
      requires penaltyAmount >= 0.0 && penaltyAmount <= 100000000.0
      requires eosToken in lm.Tokens
      ensures lm.ValidTokensIdsLogits()
      ensures next in lm.Tokens
      ensures (next == eosToken) || (parser.ValidNextToken(constrainedPrefix, next))
      ensures (next != eosToken) ==> (forall t: Token :: t in parser.ValidNextTokens(constrainedPrefix + [next]) ==> t in lm.Tokens)
      ensures next == eosToken ==> (parser.IsCompletePrefix(constrainedPrefix) || parser.ValidNextTokenCount(constrainedPrefix) == 0)
      ensures cost == old(cost) + 1
    {
      lm.GenerateLogits(prompt + constrainedPrefix);
      PenalizeTokenLogits(lm, tokensToPenalize, penaltyAmount);
      RollbackPreservesTokenInvariant(lm, parser, constrainedPrefix);
      lm.MaskValidNextAndEos(parser, constrainedPrefix, eosToken);
      next := lm.ChooseNextToken();
      if next != eosToken {
        assert !lm.IsMasked(next);
        assert parser.ValidNextToken(constrainedPrefix, next);
        assert parser.IsValidPrefix(constrainedPrefix + [next]);
        ConstrainedStepNextValid(lm, parser, constrainedPrefix, next);
      }
      cost := cost + 1;
    }

    method BoostedConstrainedStep(
      lm: LM, parser: Parser, prompt: Prefix, constrainedPrefix: Prefix,
      tokensToBoost: seq<Token>, boostAmount: real, eosToken: Token
    ) returns (next: Token)
      modifies lm.Logits, this
      requires lm.ValidTokensIdsLogits()
      requires parser.IsValidPrefix(constrainedPrefix)
      requires forall t :: t in tokensToBoost ==> t in lm.Tokens
      requires boostAmount >= 0.0 && boostAmount <= 100000000.0
      requires eosToken in lm.Tokens
      ensures lm.ValidTokensIdsLogits()
      ensures next in lm.Tokens
      ensures (next == eosToken) || (parser.ValidNextToken(constrainedPrefix, next))
      ensures (next != eosToken) ==> (forall t: Token :: t in parser.ValidNextTokens(constrainedPrefix + [next]) ==> t in lm.Tokens)
      ensures next == eosToken ==> (parser.IsCompletePrefix(constrainedPrefix) || parser.ValidNextTokenCount(constrainedPrefix) == 0)
      ensures cost == old(cost) + 1
    {
      lm.GenerateLogits(prompt + constrainedPrefix);
      BoostTokenLogits(lm, tokensToBoost, boostAmount);
      RollbackPreservesTokenInvariant(lm, parser, constrainedPrefix);
      lm.MaskValidNextAndEos(parser, constrainedPrefix, eosToken);
      next := lm.ChooseNextToken();
      if next != eosToken {
        assert !lm.IsMasked(next);
        assert parser.ValidNextToken(constrainedPrefix, next);
        assert parser.IsValidPrefix(constrainedPrefix + [next]);
        ConstrainedStepNextValid(lm, parser, constrainedPrefix, next);
      }
      cost := cost + 1;
    }

    method SafeBoostedConstrainedStep(
      lm: LM, parser: Parser, prompt: Prefix, constrainedPrefix: Prefix,
      tokensToBoost: seq<Token>, boostAmount: real, eosToken: Token
    ) returns (next: Token)
      modifies lm.Logits, this
      requires lm.ValidTokensIdsLogits()
      requires parser.IsValidPrefix(constrainedPrefix)
      requires boostAmount >= 0.0 && boostAmount <= 100000000.0
      requires eosToken in lm.Tokens
      ensures lm.ValidTokensIdsLogits()
      ensures next in lm.Tokens
      ensures (next == eosToken) || (parser.ValidNextToken(constrainedPrefix, next))
      ensures (next != eosToken) ==> (forall t: Token :: t in parser.ValidNextTokens(constrainedPrefix + [next]) ==> t in lm.Tokens)
      ensures next == eosToken ==> (parser.IsCompletePrefix(constrainedPrefix) || parser.ValidNextTokenCount(constrainedPrefix) == 0)
      ensures cost == old(cost) + 1
    {
      lm.GenerateLogits(prompt + constrainedPrefix);
      SafeBoostTokenLogits(lm, tokensToBoost, boostAmount);
      RollbackPreservesTokenInvariant(lm, parser, constrainedPrefix);
      lm.MaskValidNextAndEos(parser, constrainedPrefix, eosToken);
      next := lm.ChooseNextToken();
      if next != eosToken {
        assert !lm.IsMasked(next);
        assert parser.ValidNextToken(constrainedPrefix, next);
        assert parser.IsValidPrefix(constrainedPrefix + [next]);
        ConstrainedStepNextValid(lm, parser, constrainedPrefix, next);
      }
      cost := cost + 1;
    }

    method SafePenalizedConstrainedStep(
      lm: LM, parser: Parser, prompt: Prefix, constrainedPrefix: Prefix,
      tokensToPenalize: seq<Token>, penaltyAmount: real, eosToken: Token
    ) returns (next: Token)
      modifies lm.Logits, this
      requires lm.ValidTokensIdsLogits()
      requires parser.IsValidPrefix(constrainedPrefix)
      requires penaltyAmount >= 0.0 && penaltyAmount <= 100000000.0
      requires eosToken in lm.Tokens
      ensures lm.ValidTokensIdsLogits()
      ensures next in lm.Tokens
      ensures (next == eosToken) || (parser.ValidNextToken(constrainedPrefix, next))
      ensures (next != eosToken) ==> (forall t: Token :: t in parser.ValidNextTokens(constrainedPrefix + [next]) ==> t in lm.Tokens)
      ensures next == eosToken ==> (parser.IsCompletePrefix(constrainedPrefix) || parser.ValidNextTokenCount(constrainedPrefix) == 0)
      ensures cost == old(cost) + 1
    {
      lm.GenerateLogits(prompt + constrainedPrefix);
      SafePenalizeTokenLogits(lm, tokensToPenalize, penaltyAmount);
      RollbackPreservesTokenInvariant(lm, parser, constrainedPrefix);
      lm.MaskValidNextAndEos(parser, constrainedPrefix, eosToken);
      next := lm.ChooseNextToken();
      if next != eosToken {
        assert !lm.IsMasked(next);
        assert parser.ValidNextToken(constrainedPrefix, next);
        assert parser.IsValidPrefix(constrainedPrefix + [next]);
        ConstrainedStepNextValid(lm, parser, constrainedPrefix, next);
      }
      cost := cost + 1;
    }

    // Performs unconstrained decoding until we run out of steps.
    method UnconstrainedGeneration(lm: LM, prompt: Prefix, maxSteps: nat) returns (generated: Prefix)
      modifies lm.Logits, this
      requires lm.ValidTokensIdsLogits()
      ensures lm.ValidTokensIdsLogits()
      ensures |generated| == maxSteps
      ensures cost == old(cost) + |generated|
    {
      generated := [];
      var steps := 0;
      while steps < maxSteps
        invariant 0 <= steps <= maxSteps
        invariant lm.ValidTokensIdsLogits()
        invariant steps == |generated|
        invariant cost == old(cost) + steps
        decreases maxSteps - steps
      {
        var next := UnconstrainedStep(lm, prompt, generated);
        generated := generated + [next];
        steps := steps + 1;
      }
    }

    // A lemma that lets us say if the LM can generate all next valid tokens, then if we append one of those to the end, the LM can still generate all next valid tokens for the new prefix.
    static lemma {:axiom} ConstrainedStepNextValid(lm: LM, parser: Parser, generated: Prefix, next: Token)
      requires lm.ValidTokensIdsLogits()
      requires parser.IsValidPrefix(generated)
      requires forall t: Token :: t in parser.ValidNextTokens(generated) ==> t in lm.Tokens
      requires parser.IsValidPrefix(generated + [next])
      ensures forall t: Token :: t in parser.ValidNextTokens(generated + [next]) ==> t in lm.Tokens

    // Performs constrained decoding until we run out of steps or the generated string is complete in the grammar.
    method ConstrainedGeneration(lm: LM, parser: Parser, prompt: Prefix, maxSteps: nat, eosToken: Token) returns (generated: Prefix, terminatedByEos: bool)
      modifies lm.Logits, this
      requires lm.ValidTokensIdsLogits()
      requires parser.IsValidPrefix([])
      requires eosToken in lm.Tokens
      ensures lm.ValidTokensIdsLogits()
      ensures |generated| <= maxSteps
      ensures parser.IsValidPrefix(generated)
      ensures terminatedByEos ==> (cost == old(cost) + |generated| + 1)
      ensures !terminatedByEos ==> (cost == old(cost) + |generated|)
    {
      generated := [];
      var steps := 0;
      terminatedByEos := false;
      while steps < maxSteps && !parser.IsCompletePrefix(generated)
        invariant 0 <= steps <= maxSteps
        invariant lm.ValidTokensIdsLogits()
        invariant steps == |generated|
        invariant parser.IsValidPrefix(generated)
        invariant cost == old(cost) + steps
        invariant !terminatedByEos
        decreases maxSteps - steps
      {
        var next := ConstrainedStep(lm, parser, prompt, generated, eosToken);
        if next == eosToken {
          steps := steps + 1;
          terminatedByEos := true;
          break;
        }
        generated := generated + [next];
        steps := steps + 1;
      }
    }


    // Returns the tokens in `prefix` that appear immediately after `keyword`.
    // Use: extract which tokens the model has emitted after a specific keyword
    // (e.g., table names after "FROM", variable names after "LET", etc.) so
    // a strategy can maintain its own semantic context across loop iterations.
    static method ExtractAfterKeyword(prefix: Prefix, keyword: Token) returns (following: seq<Token>)
      ensures forall t :: t in following ==> t in prefix
    {
      following := [];
      var i := 0;
      while i < |prefix|
        invariant 0 <= i <= |prefix|
        invariant forall t :: t in following ==> t in prefix
        decreases |prefix| - i
      {
        if prefix[i] == keyword && i + 1 < |prefix| {
          following := following + [prefix[i + 1]];
        }
        i := i + 1;
      }
    }

    static method IntersectTokenSets(a: seq<Token>, b: seq<Token>) returns (result: seq<Token>)
      ensures forall t :: t in result ==> t in a && t in b
      ensures |result| <= |a|
    {
      result := [];
      var i := 0;
      while i < |a|
        invariant 0 <= i <= |a|
        invariant |result| <= i
        invariant forall t :: t in result ==> t in a && t in b
        decreases |a| - i
      {
        if a[i] in b {
          result := result + [a[i]];
        }
        i := i + 1;
      }
    }

    static method SubtractTokenSets(a: seq<Token>, b: seq<Token>) returns (result: seq<Token>)
      ensures forall t :: t in result ==> t in a && t !in b
      ensures |result| <= |a|
    {
      result := [];
      var i := 0;
      while i < |a|
        invariant 0 <= i <= |a|
        invariant |result| <= i
        invariant forall t :: t in result ==> t in a && t !in b
        decreases |a| - i
      {
        if a[i] !in b {
          result := result + [a[i]];
        }
        i := i + 1;
      }
    }

    static method RollbackToValidPrefix(parser: Parser, generated: Prefix) returns (repaired: Prefix)
      requires parser.IsValidPrefix([])
      ensures parser.IsValidPrefix(repaired)
      ensures |repaired| <= |generated|
    {
      repaired := generated;

      while !parser.IsValidPrefix(repaired) || parser.IsDeadPrefix(repaired)
        invariant |repaired| <= |generated|
        invariant parser.IsValidPrefix(repaired) || |repaired| > 0
        decreases |repaired|
      {
        repaired := repaired[..|repaired|-1];
      }
    }

    method RollbackConstrainedSpan(
      parser: Parser, stablePrefix: Prefix, generated: Prefix, currentConstrained: Prefix
    ) returns (generatedOut: Prefix, currentOut: Prefix)
      requires parser.IsValidPrefix([])
      requires generated == stablePrefix + currentConstrained
      ensures parser.IsValidPrefix(currentOut)
      ensures |currentOut| <= |currentConstrained|
      ensures generatedOut == stablePrefix + currentOut
    {
      currentOut := RollbackToValidPrefix(parser, currentConstrained);
      generatedOut := stablePrefix + currentOut;
    }

    method RollbackConstrainedSuffix(
      parser: Parser, generated: Prefix, currentConstrained: Prefix
    ) returns (generatedOut: Prefix, currentOut: Prefix)
      requires parser.IsValidPrefix([])
      requires |currentConstrained| <= |generated|
      ensures parser.IsValidPrefix(currentOut)
      ensures |currentOut| <= |currentConstrained|
      ensures generatedOut == generated[..|generated| - |currentConstrained|] + currentOut
      ensures |currentOut| <= |generatedOut|
      ensures generatedOut[|generatedOut| - |currentOut|..] == currentOut
      ensures |generatedOut| <= |generated|
    {
      var stablePrefix := generated[..|generated| - |currentConstrained|];
      currentOut := RollbackToValidPrefix(parser, currentConstrained);
      generatedOut := stablePrefix + currentOut;
      assert |stablePrefix| == |generated| - |currentConstrained|;
      assert |generatedOut| == |stablePrefix| + |currentOut|;
      assert |generatedOut| <= |generated|;
      assert |currentOut| <= |generatedOut|;
      assert generatedOut[|generatedOut| - |currentOut|..] == currentOut;
    }

    static method RollbackToCompletePrefix(parser: Parser, generated: Prefix) returns (repaired: Prefix)
      requires parser.IsValidPrefix([])
      ensures parser.IsCompletePrefix(repaired) || repaired == []
      ensures parser.IsValidPrefix(repaired)
      ensures |repaired| <= |generated|
    {
      repaired := generated;

      while repaired != [] && !parser.IsCompletePrefix(repaired)
        invariant |repaired| <= |generated|
        decreases |repaired|
      {
        repaired := repaired[..|repaired|-1];
      }
    }

    method RollbackConstrainedToComplete(
      parser: Parser, generated: Prefix, currentConstrained: Prefix
    ) returns (generatedOut: Prefix, currentOut: Prefix)
      requires parser.IsValidPrefix([])
      requires |currentConstrained| <= |generated|
      ensures parser.IsCompletePrefix(currentOut) || currentOut == []
      ensures parser.IsValidPrefix(currentOut)
      ensures |currentOut| <= |currentConstrained|
      ensures generatedOut == generated[..|generated| - |currentConstrained|] + currentOut
      ensures |currentOut| <= |generatedOut|
      ensures generatedOut[|generatedOut| - |currentOut|..] == currentOut
      ensures |generatedOut| <= |generated|
    {
      var stablePrefix := generated[..|generated| - |currentConstrained|];
      currentOut := RollbackToCompletePrefix(parser, currentConstrained);
      generatedOut := stablePrefix + currentOut;
      assert |stablePrefix| == |generated| - |currentConstrained|;
      assert |generatedOut| == |stablePrefix| + |currentOut|;
      assert |generatedOut| <= |generated|;
      assert |currentOut| <= |generatedOut|;
      assert generatedOut[|generatedOut| - |currentOut|..] == currentOut;
    }

    // Rollback that actually lets the model RE-GENERATE. Plain RollbackToValidPrefix
    // only amputates trailing tokens down to a valid, non-dead prefix and stops --
    // the span is lost. This rolls back the same way, then re-generates forward FROM
    // that point using DeadEndAvoidingStep, which re-detects the dead branch and steers
    // around it. So the model gets a fresh, dead-end-aware attempt at the span instead
    // of just losing it.
    method RollbackAndRegenerate(
      lm: LM, parser: Parser, prompt: Prefix, generated: Prefix,
      eosToken: Token, maxSteps: nat, maxRetries: nat
    ) returns (regenerated: Prefix)
      modifies lm.Logits, this
      requires lm.ValidTokensIdsLogits()
      requires parser.IsValidPrefix([])
      requires eosToken in lm.Tokens
      ensures lm.ValidTokensIdsLogits()
      ensures parser.IsValidPrefix(regenerated)
      ensures |regenerated| <= |generated| + maxSteps
      ensures cost <= old(cost) + maxSteps
      ensures cost >= old(cost)
    {
      var repaired := RollbackToValidPrefix(parser, generated);
      regenerated := repaired;
      var steps := 0;
      while steps < maxSteps && !parser.IsCompletePrefix(regenerated)
        invariant lm.ValidTokensIdsLogits()
        invariant parser.IsValidPrefix(regenerated)
        invariant 0 <= steps <= maxSteps
        invariant |regenerated| <= |generated| + steps
        invariant cost == old(cost) + steps
        decreases maxSteps - steps
      {
        var next, ok := DeadEndAvoidingStep(lm, parser, prompt, regenerated, eosToken, maxRetries);
        steps := steps + 1;
        if !ok || next == eosToken {
          break;
        }
        regenerated := regenerated + [next];
      }
    }

    // Roll the constrained span back to its last point that already parses as a
    // complete expression, then KEEP GENERATING forward from there (dead-end-aware),
    // tracking the longest complete point reached. Returns that best complete point.
    // Generation is capped at (maxSteps - closeReserve), so at least closeReserve
    // steps remain afterwards for the caller to emit the closing delimiter. The
    // returned span is always either empty or a complete, valid prefix -- so the
    // caller can always close it.
    method RollbackAndContinue(
      lm: LM, parser: Parser, prompt: Prefix, generated: Prefix,
      currentConstrained: Prefix, eosToken: Token, maxSteps: nat, closeReserve: nat, maxRetries: nat
    ) returns (generatedOut: Prefix, currentOut: Prefix)
      modifies lm.Logits, this
      requires lm.ValidTokensIdsLogits()
      requires parser.IsValidPrefix([])
      requires eosToken in lm.Tokens
      requires |currentConstrained| <= |generated|
      requires closeReserve <= maxSteps
      ensures lm.ValidTokensIdsLogits()
      ensures parser.IsCompletePrefix(currentOut) || currentOut == []
      ensures parser.IsValidPrefix(currentOut)
      ensures generatedOut == generated[..|generated| - |currentConstrained|] + currentOut
      ensures cost <= old(cost) + (maxSteps - closeReserve)
      ensures cost >= old(cost)
    {
      var stablePrefix := generated[..|generated| - |currentConstrained|];
      var budget := maxSteps - closeReserve;
      var bestComplete := RollbackToCompletePrefix(parser, currentConstrained);
      var running := bestComplete;
      var steps := 0;
      while steps < budget
        invariant lm.ValidTokensIdsLogits()
        invariant parser.IsValidPrefix(running)
        invariant parser.IsCompletePrefix(bestComplete) || bestComplete == []
        invariant parser.IsValidPrefix(bestComplete)
        invariant 0 <= steps <= budget
        invariant cost == old(cost) + steps
        decreases budget - steps
      {
        var next, ok := DeadEndAvoidingStep(lm, parser, prompt + stablePrefix, running, eosToken, maxRetries);
        steps := steps + 1;
        if !ok || next == eosToken {
          break;
        }
        running := running + [next];
        if parser.IsCompletePrefix(running) {
          bestComplete := running;
        }
      }
      currentOut := bestComplete;
      generatedOut := stablePrefix + currentOut;
    }

    static method FlattenTokenGroups(groups: seq<seq<Token>>) returns (flat: seq<Token>)
      ensures forall t :: t in flat ==> exists g :: g in groups && t in g
    {
      flat := [];
      var i := 0;
      while i < |groups|
        invariant 0 <= i <= |groups|
        invariant forall t :: t in flat ==> exists g :: g in groups && t in g
        decreases |groups| - i
      {
        flat := flat + groups[i];
        i := i + 1;
      }
    }

    static method GroupContaining(groups: seq<seq<Token>>, tok: Token) returns (idx: int)
      ensures -1 <= idx < |groups|
      ensures idx >= 0 ==> tok in groups[idx]
    {
      idx := -1;
      var i := 0;
      while i < |groups|
        invariant 0 <= i <= |groups|
        invariant -1 <= idx < |groups|
        invariant idx >= 0 ==> tok in groups[idx]
        decreases |groups| - i
      {
        if tok in groups[i] {
          idx := i;
          return;
        }
        i := i + 1;
      }
    }

    method LastTokenBefore(s: Prefix, sep: Token) returns (tok: Token, found: bool)
      ensures found ==> exists i :: 1 <= i < |s| && s[i] == sep && tok == s[i-1]
    {
      var idx: int := |s|;
      while idx > 0 && s[idx-1] != sep
        invariant 0 <= idx <= |s|
        invariant forall j :: idx <= j < |s| ==> s[j] != sep
        decreases idx
      {
        idx := idx - 1;
      }
      if idx >= 2 {
        found := true;
        tok := s[idx - 2];
      } else {
        found := false;
        tok := "";
      }
    }

    static lemma {:axiom} RollbackPreservesTokenInvariant(lm: LM, parser: Parser, prefix: Prefix)
      requires lm.ValidTokensIdsLogits()
      requires parser.IsValidPrefix(prefix)
      ensures forall t: Token :: t in parser.ValidNextTokens(prefix) ==> t in lm.Tokens

    static function PrefixToString(p: Prefix): string
    {
      if |p| == 0 then ""
      else p[0] + PrefixToString(p[1..])
    }

    static function ExtractContentBetweenDelimiters(input: string, startDelim: string, endDelim: string): (content: string)
      ensures content != "" ==> exists pre, post :: input == pre + startDelim + content + endDelim + post
    {
      ExtractContentExtern(input, startDelim, endDelim)
    }

    static function {:extern} {:axiom} ExtractContentExtern(input: string, startDelim: string, endDelim: string): (content: string)
      ensures content != "" ==> exists pre, post :: input == pre + startDelim + content + endDelim + post
    method BoostTokenLogits(lm: LM, tokens: seq<Token>, amount: real)
      modifies lm.Logits
      requires lm.ValidTokensIdsLogits()
      requires forall t :: t in tokens ==> t in lm.Tokens
      requires amount >= 0.0 && amount <= 100000000.0
      ensures lm.ValidTokensIdsLogits()
      ensures forall t :: t in lm.Tokens && !(t in tokens) ==>
        lm.Logits[lm.TokenToId(t)] == old(lm.Logits[lm.TokenToId(t)])
    {
      var i := 0;
      while i < |tokens|
        invariant 0 <= i <= |tokens|
        invariant lm.ValidTokensIdsLogits()
        invariant forall t :: t in lm.Tokens && !(t in tokens[..i]) ==>
          lm.Logits[lm.TokenToId(t)] == old(lm.Logits[lm.TokenToId(t)])
        decreases |tokens| - i
      {
        var id := lm.TokenToId(tokens[i]);
        var newVal := lm.Logits[id] + amount;
        if newVal > 1000000000.0 { newVal := 1000000000.0; }
        lm.Logits[id] := newVal;
        i := i + 1;
      }
    }

    method SafeBoostTokenLogits(lm: LM, tokens: seq<Token>, amount: real)
      modifies lm.Logits
      requires lm.ValidTokensIdsLogits()
      requires amount >= 0.0 && amount <= 100000000.0
      ensures lm.ValidTokensIdsLogits()
    {
      var validTokens := IntersectTokenSets(lm.Tokens, tokens);
      BoostTokenLogits(lm, validTokens, amount);
    }

    method PenalizeTokenLogits(lm: LM, tokens: seq<Token>, amount: real)
      modifies lm.Logits
      requires lm.ValidTokensIdsLogits()
      requires forall t :: t in tokens ==> t in lm.Tokens
      requires amount >= 0.0 && amount <= 100000000.0
      ensures lm.ValidTokensIdsLogits()
      ensures forall t :: t in lm.Tokens && !(t in tokens) ==>
        lm.Logits[lm.TokenToId(t)] == old(lm.Logits[lm.TokenToId(t)])
    {
      var i := 0;
      while i < |tokens|
        invariant 0 <= i <= |tokens|
        invariant lm.ValidTokensIdsLogits()
        invariant forall t :: t in lm.Tokens && !(t in tokens[..i]) ==>
          lm.Logits[lm.TokenToId(t)] == old(lm.Logits[lm.TokenToId(t)])
        decreases |tokens| - i
      {
        var id := lm.TokenToId(tokens[i]);
        var newVal := lm.Logits[id] - amount;
        if newVal < -1000000000.0 { newVal := -1000000000.0; }
        lm.Logits[id] := newVal;
        i := i + 1;
      }
    }

    method SafePenalizeTokenLogits(lm: LM, tokens: seq<Token>, amount: real)
      modifies lm.Logits
      requires lm.ValidTokensIdsLogits()
      requires amount >= 0.0 && amount <= 100000000.0
      ensures lm.ValidTokensIdsLogits()
    {
      var validTokens := IntersectTokenSets(lm.Tokens, tokens);
      PenalizeTokenLogits(lm, validTokens, amount);
    }

    method MaskTokensInPrefix(lm: LM, prefix: Prefix)
      modifies lm.Logits
      requires lm.ValidTokensIdsLogits()
      ensures lm.ValidTokensIdsLogits()
      ensures forall t :: t in prefix && t in lm.Tokens ==> lm.IsMasked(t)
      ensures forall t :: t in lm.Tokens && !(t in prefix) ==>
        lm.Logits[lm.TokenToId(t)] == old(lm.Logits[lm.TokenToId(t)])
      ensures cost == old(cost)
    {
      var i := 0;
      while i < |prefix|
        invariant 0 <= i <= |prefix|
        invariant lm.ValidTokensIdsLogits()
        invariant forall t :: t in prefix[..i] && t in lm.Tokens ==> lm.IsMasked(t)
        invariant forall t :: t in lm.Tokens && !(t in prefix[..i]) ==>
          lm.Logits[lm.TokenToId(t)] == old(lm.Logits[lm.TokenToId(t)])
        invariant cost == old(cost)
        decreases |prefix| - i
      {
        if prefix[i] in lm.Tokens {
          lm.MaskToken(prefix[i]);
        }
        i := i + 1;
      }
    }

    method GetHighestLogitToken(lm: LM) returns (token: Token)
      requires lm.ValidTokensIdsLogits()
      requires |lm.Tokens| > 0
      ensures lm.ValidTokensIdsLogits()
      ensures token in lm.Tokens
    {
      var bestIdx := 0;
      var i := 1;
      while i < |lm.Tokens|
        invariant 1 <= i <= |lm.Tokens|
        invariant 0 <= bestIdx < |lm.Tokens|
        invariant lm.ValidTokensIdsLogits()
        decreases |lm.Tokens| - i
      {
        if lm.Logits[i] > lm.Logits[bestIdx] {
          bestIdx := i;
        }
        i := i + 1;
      }
      token := lm.IdToToken(bestIdx);
    }

    method GetLogitGap(lm: LM) returns (gap: real)
      requires lm.ValidTokensIdsLogits()
      requires |lm.Tokens| >= 2
      ensures lm.ValidTokensIdsLogits()
      ensures gap >= 0.0
    {
      var top1: real := -1000000001.0;
      var top2: real := -1000000001.0;
      var i := 0;
      while i < |lm.Tokens|
        invariant 0 <= i <= |lm.Tokens|
        invariant lm.ValidTokensIdsLogits()
        invariant top2 <= top1
        decreases |lm.Tokens| - i
      {
        if lm.Logits[i] > -1000000000.0 {
          var L := lm.Logits[i];
          if L > top1 {
            top2 := top1;
            top1 := L;
          } else if L > top2 {
            top2 := L;
          }
        }
        i := i + 1;
      }
      if top2 < -1000000000.0 {
        gap := 0.0;
      } else {
        gap := top1 - top2;
      }
    }

    static lemma {:axiom} UnchosenIndexExists(n: nat, chosenIdx: seq<int>, k: nat, picked: nat)
      requires picked < k <= n
      requires forall u :: u in chosenIdx ==> 0 <= u < n
      requires |chosenIdx| == picked
      requires forall i, j :: 0 <= i < j < |chosenIdx| ==> chosenIdx[i] != chosenIdx[j]
      ensures exists u :: 0 <= u < n && !(u in chosenIdx)

    static lemma {:axiom} DistinctChosenSeq(lm: LM, idx: seq<int>)
      requires lm.ValidTokensIdsLogits()
      requires forall u :: u in idx ==> 0 <= u < |lm.Tokens|
      requires forall i, j :: 0 <= i < j < |idx| ==> idx[i] != idx[j]
      ensures lm.ValidTokensIdsLogits()

    method GetTopKTokens(lm: LM, k: nat) returns (tokens: seq<Token>)
      requires lm.ValidTokensIdsLogits()
      requires 1 <= k <= |lm.Tokens|
      ensures lm.ValidTokensIdsLogits()
      ensures |tokens| <= k
      ensures forall t :: t in tokens ==> t in lm.Tokens
      ensures forall i, j :: 0 <= i < j < |tokens| ==> tokens[i] != tokens[j]
      ensures cost == old(cost)
    {
      tokens := [];
      var picked := 0;
      while picked < k
        invariant 0 <= picked <= k
        invariant |tokens| == picked
        invariant lm.ValidTokensIdsLogits()
        invariant forall t :: t in tokens ==> t in lm.Tokens
        invariant forall i, j :: 0 <= i < j < |tokens| ==> tokens[i] != tokens[j]
        invariant cost == old(cost)
        decreases k - picked
      {
        // Pick the highest-logit index whose decoded STRING is not already
        // chosen. Dedup is by string, not by index: at runtime two vocab ids
        // can decode to the same string, so distinct indices need not be
        // distinct strings. If no unused string remains, stop early so the
        // result holds genuinely distinct tokens (|tokens| < k is allowed).
        var bestIdx: int := -1;
        var j := 0;
        while j < |lm.Tokens|
          invariant 0 <= j <= |lm.Tokens|
          invariant lm.ValidTokensIdsLogits()
          invariant bestIdx == -1 || (0 <= bestIdx < |lm.Tokens| && !(lm.Tokens[bestIdx] in tokens))
          decreases |lm.Tokens| - j
        {
          if !(lm.Tokens[j] in tokens) {
            if bestIdx == -1 || lm.Logits[j] > lm.Logits[bestIdx] {
              bestIdx := j;
            }
          }
          j := j + 1;
        }
        if bestIdx == -1 {
          break;
        }
        tokens := tokens + [lm.Tokens[bestIdx]];
        picked := picked + 1;
      }
    }

    method DeadEndDetection(parser: Parser, prefix: Prefix, minValidCount: nat) returns (isNarrow: bool)
      requires parser.IsValidPrefix(prefix)
      ensures isNarrow <==> parser.ValidNextTokenCount(prefix) < minValidCount
    {
      var validCount := parser.ValidNextTokenCount(prefix);
      isNarrow := validCount < minValidCount;
    }

    method SoftConstrainedStep(
      lm: LM, parser: Parser, prompt: Prefix, constrainedPrefix: Prefix,
      boostAmount: real, eosToken: Token
    ) returns (next: Token, isValid: bool)
      modifies lm.Logits, this
      requires lm.ValidTokensIdsLogits()
      requires parser.IsValidPrefix(constrainedPrefix)
      requires boostAmount >= 0.0 && boostAmount <= 100000000.0
      requires eosToken in lm.Tokens
      ensures lm.ValidTokensIdsLogits()
      ensures next in lm.Tokens
      ensures isValid <==> (next == eosToken || parser.IsValidPrefix(constrainedPrefix + [next]))
      ensures isValid && next != eosToken ==> (forall t: Token :: t in parser.ValidNextTokens(constrainedPrefix + [next]) ==> t in lm.Tokens)
      ensures next == eosToken ==> (parser.IsCompletePrefix(constrainedPrefix) || parser.ValidNextTokenCount(constrainedPrefix) == 0)
      ensures cost == old(cost) + 1
    {
      lm.GenerateLogits(prompt + constrainedPrefix);
      RollbackPreservesTokenInvariant(lm, parser, constrainedPrefix);
      lm.BoostValidNextAndEos(parser, constrainedPrefix, boostAmount, eosToken);
      var softNext := lm.ChooseNextTokenUnconstrained();
      // The soft draw is unconstrained, so it can land on eosToken even when
      // stopping is not yet legal (prefix incomplete and not a dead end). In
      // that case, reject the premature stop and fall back to a masked,
      // grammar-constrained draw -- same fallback MaskValidNextAndEos already
      // uses to keep eosToken itself masked until stopping is legal.
      if softNext == eosToken && !(parser.IsCompletePrefix(constrainedPrefix) || parser.ValidNextTokenCount(constrainedPrefix) == 0) {
        lm.MaskValidNextAndEos(parser, constrainedPrefix, eosToken);
        next := lm.ChooseNextToken();
        if next != eosToken {
          assert !lm.IsMasked(next);
          assert parser.ValidNextToken(constrainedPrefix, next);
          assert parser.IsValidPrefix(constrainedPrefix + [next]);
          ConstrainedStepNextValid(lm, parser, constrainedPrefix, next);
        }
      } else {
        next := softNext;
        if next != eosToken && parser.IsValidPrefix(constrainedPrefix + [next]) {
          ConstrainedStepNextValid(lm, parser, constrainedPrefix, next);
        }
      }
      cost := cost + 1;
      isValid := next == eosToken || parser.IsValidPrefix(constrainedPrefix + [next]);
    }

    method SafeSoftConstrainedStep(
      lm: LM, parser: Parser, prompt: Prefix, constrainedPrefix: Prefix,
      boostAmount: real, eosToken: Token
    ) returns (next: Token, usedFallback: bool)
      modifies lm.Logits, this
      requires lm.ValidTokensIdsLogits()
      requires parser.IsValidPrefix(constrainedPrefix)
      requires boostAmount >= 0.0 && boostAmount <= 100000000.0
      requires eosToken in lm.Tokens
      ensures lm.ValidTokensIdsLogits()
      ensures next in lm.Tokens
      ensures (next == eosToken) || parser.IsValidPrefix(constrainedPrefix + [next])
      ensures (next != eosToken) ==> (forall t: Token :: t in parser.ValidNextTokens(constrainedPrefix + [next]) ==> t in lm.Tokens)
      ensures next == eosToken ==> (parser.IsCompletePrefix(constrainedPrefix) || parser.ValidNextTokenCount(constrainedPrefix) == 0)
      ensures cost == old(cost) + 1
    {
      lm.GenerateLogits(prompt + constrainedPrefix);
      RollbackPreservesTokenInvariant(lm, parser, constrainedPrefix);
      lm.BoostValidNextAndEos(parser, constrainedPrefix, boostAmount, eosToken);
      var softNext := lm.ChooseNextTokenUnconstrained();
      // Only accept an eosToken drawn from the unconstrained sampler when
      // stopping is actually legal (prefix complete, or a dead end). A
      // premature eosToken -- or any grammar-invalid non-eos token -- falls
      // back to the masked, grammar-constrained draw below.
      var stopAllowed := parser.IsCompletePrefix(constrainedPrefix) || parser.ValidNextTokenCount(constrainedPrefix) == 0;
      if softNext == eosToken && stopAllowed {
        next := softNext;
        usedFallback := false;
      } else if softNext != eosToken && parser.IsValidPrefix(constrainedPrefix + [softNext]) {
        next := softNext;
        usedFallback := false;
        ConstrainedStepNextValid(lm, parser, constrainedPrefix, next);
      } else {
        lm.MaskValidNextAndEos(parser, constrainedPrefix, eosToken);
        next := lm.ChooseNextToken();
        usedFallback := true;
        if next != eosToken {
          assert !lm.IsMasked(next);
          assert parser.ValidNextToken(constrainedPrefix, next);
          assert parser.IsValidPrefix(constrainedPrefix + [next]);
          ConstrainedStepNextValid(lm, parser, constrainedPrefix, next);
        }
      }
      cost := cost + 1;
    }

    method ConfidenceGatedStep(
      lm: LM, parser: Parser, prompt: Prefix, constrainedPrefix: Prefix, eosToken: Token
    ) returns (next: Token, wasConstrained: bool)
      modifies lm.Logits, this
      requires lm.ValidTokensIdsLogits()
      requires parser.IsValidPrefix(constrainedPrefix)
      requires eosToken in lm.Tokens
      ensures lm.ValidTokensIdsLogits()
      ensures next in lm.Tokens
      ensures (next == eosToken) || parser.IsValidPrefix(constrainedPrefix + [next])
      ensures (next != eosToken) ==> (forall t: Token :: t in parser.ValidNextTokens(constrainedPrefix + [next]) ==> t in lm.Tokens)
      ensures next == eosToken ==> (parser.IsCompletePrefix(constrainedPrefix) || parser.ValidNextTokenCount(constrainedPrefix) == 0)
      ensures cost == old(cost) + 1
    {
      lm.GenerateLogits(prompt + constrainedPrefix);
      var topToken := GetHighestLogitToken(lm);
      // The top-logit token is only accepted as an eosToken stop when
      // stopping is actually legal. Otherwise -- even if the model's top
      // pick is eosToken -- fall through to the grammar-constrained branch
      // below, same as an ordinary grammar-invalid top pick.
      var stopAllowed := parser.IsCompletePrefix(constrainedPrefix) || parser.ValidNextTokenCount(constrainedPrefix) == 0;
      if topToken == eosToken && stopAllowed {
        next := topToken;
        wasConstrained := false;
      } else if topToken != eosToken && parser.IsValidPrefix(constrainedPrefix + [topToken]) {
        next := topToken;
        wasConstrained := false;
        RollbackPreservesTokenInvariant(lm, parser, constrainedPrefix);
        ConstrainedStepNextValid(lm, parser, constrainedPrefix, next);
      } else {
        RollbackPreservesTokenInvariant(lm, parser, constrainedPrefix);
        lm.MaskValidNextAndEos(parser, constrainedPrefix, eosToken);
        next := lm.ChooseNextToken();
        wasConstrained := true;
        if next != eosToken {
          assert !lm.IsMasked(next);
          assert parser.ValidNextToken(constrainedPrefix, next);
          assert parser.IsValidPrefix(constrainedPrefix + [next]);
          RollbackPreservesTokenInvariant(lm, parser, constrainedPrefix + [next]);
        }
      }
      cost := cost + 1;
    }

    static function CountSubstring(s: string, sub: string): nat
      requires |sub| > 0
      decreases |s|
    {
      if |s| < |sub| then 0
      else if s[..|sub|] == sub then 1 + CountSubstring(s[|sub|..], sub)
      else CountSubstring(s[1..], sub)
    }

    static function OccurrencesInRange(prefix: Prefix, target: Token, hi: nat): nat
      requires hi <= |prefix|
    {
      if hi == 0 then 0
      else OccurrencesInRange(prefix, target, hi - 1) + (if prefix[hi - 1] == target then 1 else 0)
    }

    static method CountTokenOccurrences(prefix: Prefix, target: Token) returns (count: nat)
      ensures count == OccurrencesInRange(prefix, target, |prefix|)
    {
      count := 0;
      var i := 0;
      while i < |prefix|
        invariant 0 <= i <= |prefix|
        invariant count == OccurrencesInRange(prefix, target, i)
        decreases |prefix| - i
      {
        if prefix[i] == target {
          count := count + 1;
        }
        i := i + 1;
      }
    }

    static method TokensSinceLastOccurrence(prefix: Prefix, target: Token) returns (dist: nat)
      ensures dist <= |prefix|
      ensures dist == |prefix| ==> forall i :: 0 <= i < |prefix| ==> prefix[i] != target
      ensures dist < |prefix| ==> prefix[|prefix| - 1 - dist] == target
    {
      dist := 0;
      while dist < |prefix| && prefix[|prefix| - 1 - dist] != target
        invariant 0 <= dist <= |prefix|
        invariant forall j :: |prefix| - dist <= j < |prefix| ==> prefix[j] != target
        decreases |prefix| - dist
      {
        dist := dist + 1;
      }
    }

    method GetTokenLogit(lm: LM, token: Token) returns (logit: real)
      requires lm.ValidTokensIdsLogits()
      requires token in lm.Tokens
      ensures lm.ValidTokensIdsLogits()
      ensures logit == lm.Logits[lm.TokenToId(token)]
    {
      logit := lm.Logits[lm.TokenToId(token)];
    }

    method ScaleAllLogits(lm: LM, scalar: real)
      modifies lm.Logits
      requires lm.ValidTokensIdsLogits()
      requires scalar > 0.0 && scalar <= 100000000.0
      ensures lm.ValidTokensIdsLogits()
      ensures forall t :: t in lm.Tokens && !(t in lm.Tokens) ==>
        lm.Logits[lm.TokenToId(t)] == old(lm.Logits[lm.TokenToId(t)])
    {
      var i := 0;
      while i < |lm.Tokens|
        invariant 0 <= i <= |lm.Tokens|
        invariant lm.ValidTokensIdsLogits()
        invariant forall t :: t in lm.Tokens && !(t in lm.Tokens[..i]) ==>
          lm.Logits[lm.TokenToId(t)] == old(lm.Logits[lm.TokenToId(t)])
        decreases |lm.Tokens| - i
      {
        var id := lm.TokenToId(lm.Tokens[i]);
        var newVal := lm.Logits[id] * scalar;
        if newVal > 1000000000.0 { newVal := 1000000000.0; }
        if newVal < -1000000000.0 { newVal := -1000000000.0; }
        lm.Logits[id] := newVal;
        i := i + 1;
      }
    }

    method ValidTokenCount(parser: Parser, prefix: Prefix) returns (count: nat)
      requires parser.IsValidPrefix(prefix)
      ensures count == parser.ValidNextTokenCount(prefix)
    {
      count := parser.ValidNextTokenCount(prefix);
    }

    method TopValidCandidates(
      lm: LM, parser: Parser, prompt: Prefix, prefix: Prefix, maxCandidates: nat, eosToken: Token
    ) returns (candidates: seq<Token>)
      modifies lm.Logits, this
      requires lm.ValidTokensIdsLogits()
      requires parser.IsValidPrefix(prefix)
      requires maxCandidates > 0
      requires eosToken in lm.Tokens
      ensures lm.ValidTokensIdsLogits()
      ensures 0 < |candidates| <= maxCandidates
      ensures forall t :: t in candidates ==> t in lm.Tokens
      ensures forall t :: t in candidates ==> t == eosToken || t in parser.ValidNextTokens(prefix)
      ensures forall t :: t in candidates && t == eosToken ==> (parser.IsCompletePrefix(prefix) || parser.ValidNextTokenCount(prefix) == 0)
      ensures forall i, j :: 0 <= i < j < |candidates| ==> candidates[i] != candidates[j]
      ensures cost == old(cost) + 1
    {
      var baseCost := cost;
      lm.GenerateLogits(prompt + prefix);
      RollbackPreservesTokenInvariant(lm, parser, prefix);
      // eosToken is only offered as a candidate when stopping is actually
      // legal (prefix complete, or a dead end with no valid continuation).
      // Mask everything else out first, so every token that survives the
      // mask already carries both guarantees this method needs to make
      // about eosToken -- straight from MaskValidNextAndEos's own contract,
      // with no separate assumption about the grammar's relationship to
      // eosToken required.
      var stopAllowed := parser.IsCompletePrefix(prefix) || parser.ValidNextTokenCount(prefix) == 0;
      lm.MaskValidNextAndEos(parser, prefix, eosToken);
      // ChooseNextToken is trusted to always hand back an unmasked token, so
      // this seeds the pool with one guaranteed-legal candidate up front --
      // that is what keeps the pool from ever coming back empty below.
      var seed := lm.ChooseNextToken();
      if seed != eosToken {
        assert parser.ValidNextToken(prefix, seed);
        assert seed in parser.ValidNextTokens(prefix);
      } else {
        assert stopAllowed;
      }
      var pool: seq<Token> := [seed];
      var i := 0;
      var N := |lm.Tokens|;

      while i < N && |pool| < maxCandidates
        invariant lm.ValidTokensIdsLogits()
        invariant 0 <= i <= N
        invariant 0 < |pool| <= maxCandidates
        invariant forall t :: t in pool ==> t in lm.Tokens
        invariant forall t :: t in pool ==> t == eosToken || t in parser.ValidNextTokens(prefix)
        invariant forall t :: t in pool && t == eosToken ==> stopAllowed
        invariant forall j, k :: 0 <= j < k < |pool| ==> pool[j] != pool[k]
        invariant cost == baseCost
        decreases N - i
      {
        var tok := lm.Tokens[i];
        if !lm.IsMasked(tok) && !(tok in pool) {
          if tok != eosToken {
            assert parser.ValidNextToken(prefix, tok);
            assert tok in parser.ValidNextTokens(prefix);
          } else {
            assert stopAllowed;
          }
          pool := pool + [tok];
        }
        i := i + 1;
      }
      // The loop above always admits at least one token, so the old
      // "if empty, fall back to EOS" branch below was unreachable. Falling
      // back to EOS is exactly the bug this file was changed to stop, so the
      // branch is gone rather than left sitting there waiting to be reached.
      assert |pool| > 0;

      var target := if maxCandidates < |pool| then maxCandidates else |pool|;
      var chosen: seq<Token> := [];

      while |chosen| < target
        invariant lm.ValidTokensIdsLogits()
        invariant 0 < target <= |pool|
        invariant 0 <= |chosen| <= target
        invariant forall t :: t in chosen ==> t in pool
        invariant forall t :: t in chosen ==> t in lm.Tokens
        invariant forall t :: t in chosen ==> t == eosToken || t in parser.ValidNextTokens(prefix)
        invariant forall t :: t in chosen && t == eosToken ==> stopAllowed
        invariant forall i, j :: 0 <= i < j < |chosen| ==> chosen[i] != chosen[j]
        invariant forall i, j :: 0 <= i < j < |pool| ==> pool[i] != pool[j]
        invariant cost == baseCost
        decreases target - |chosen|
      {
        var bestTok := pool[0];
        var bestLogit := -1000000000.0;
        var found := false;
        var j := 0;

        while j < |pool|
          invariant lm.ValidTokensIdsLogits()
          invariant 0 <= j <= |pool|
          invariant found ==> bestTok in pool
          invariant found ==> !(bestTok in chosen)
          invariant found ==> bestLogit == lm.Logits[lm.TokenToId(bestTok)]
          invariant found ==> forall k :: 0 <= k < j && !(pool[k] in chosen) ==> lm.Logits[lm.TokenToId(pool[k])] <= bestLogit
          invariant !found ==> forall k :: 0 <= k < j ==> pool[k] in chosen
          invariant cost == baseCost
          decreases |pool| - j
        {
          var tok := pool[j];
          if !(tok in chosen) {
            var tokLogit := lm.Logits[lm.TokenToId(tok)];
            if !found || tokLogit > bestLogit {
              bestTok := tok;
              bestLogit := tokLogit;
              found := true;
            }
          }
          j := j + 1;
        }

        if found {
          chosen := chosen + [bestTok];
        } else {
          break;
        }
      }

      if |chosen| == 0 {
        candidates := [pool[0]];
      } else {
        candidates := chosen;
      }
      cost := cost + 1;
    }

    method IsTokenValidNext(parser: Parser, prefix: Prefix, token: Token) returns (isValid: bool)
      requires parser.IsValidPrefix(prefix)
      ensures isValid <==> parser.ValidNextToken(prefix, token)
    {
      isValid := parser.ValidNextToken(prefix, token);
    }

    method RepetitionPenaltyStep(
      lm: LM, parser: Parser, prompt: Prefix, prefix: Prefix,
      generated: Prefix, penaltyAmount: real, eosToken: Token
    ) returns (next: Token)
      modifies lm.Logits, this
      requires lm.ValidTokensIdsLogits()
      requires parser.IsValidPrefix(prefix)
      requires forall t :: t in generated ==> t in lm.Tokens
      requires penaltyAmount >= 0.0 && penaltyAmount <= 100000000.0
      requires eosToken in lm.Tokens
      ensures lm.ValidTokensIdsLogits()
      ensures next in lm.Tokens
      ensures (next == eosToken) || (parser.ValidNextToken(prefix, next))
      ensures (next != eosToken) ==> (forall t: Token :: t in parser.ValidNextTokens(prefix + [next]) ==> t in lm.Tokens)
      ensures next == eosToken ==> (parser.IsCompletePrefix(prefix) || parser.ValidNextTokenCount(prefix) == 0)
      ensures cost == old(cost) + 1
    {
      lm.GenerateLogits(prompt + prefix);
      PenalizeTokenLogits(lm, generated, penaltyAmount);
      RollbackPreservesTokenInvariant(lm, parser, prefix);
      lm.MaskValidNextAndEos(parser, prefix, eosToken);
      next := lm.ChooseNextToken();
      if next != eosToken {
        assert !lm.IsMasked(next);
        assert parser.ValidNextToken(prefix, next);
        assert parser.IsValidPrefix(prefix + [next]);
        ConstrainedStepNextValid(lm, parser, prefix, next);
      }
      cost := cost + 1;
    }

    method SafeRepetitionPenaltyStep(
      lm: LM, parser: Parser, prompt: Prefix, prefix: Prefix,
      generated: Prefix, penaltyAmount: real, eosToken: Token
    ) returns (next: Token)
      modifies lm.Logits, this
      requires lm.ValidTokensIdsLogits()
      requires parser.IsValidPrefix(prefix)
      requires penaltyAmount >= 0.0 && penaltyAmount <= 100000000.0
      requires eosToken in lm.Tokens
      ensures lm.ValidTokensIdsLogits()
      ensures next in lm.Tokens
      ensures (next == eosToken) || (parser.ValidNextToken(prefix, next))
      ensures (next != eosToken) ==> (forall t: Token :: t in parser.ValidNextTokens(prefix + [next]) ==> t in lm.Tokens)
      ensures next == eosToken ==> (parser.IsCompletePrefix(prefix) || parser.ValidNextTokenCount(prefix) == 0)
      ensures cost == old(cost) + 1
    {
      lm.GenerateLogits(prompt + prefix);
      SafePenalizeTokenLogits(lm, generated, penaltyAmount);
      RollbackPreservesTokenInvariant(lm, parser, prefix);
      lm.MaskValidNextAndEos(parser, prefix, eosToken);
      next := lm.ChooseNextToken();
      if next != eosToken {
        assert !lm.IsMasked(next);
        assert parser.ValidNextToken(prefix, next);
        assert parser.IsValidPrefix(prefix + [next]);
        ConstrainedStepNextValid(lm, parser, prefix, next);
      }
      cost := cost + 1;
    }

    method TemperatureConstrainedStep(
      lm: LM, parser: Parser, prompt: Prefix, prefix: Prefix, temperature: real, eosToken: Token
    ) returns (next: Token)
      modifies lm.Logits, this
      requires lm.ValidTokensIdsLogits()
      requires parser.IsValidPrefix(prefix)
      requires temperature >= 0.00000001 && temperature <= 100000000.0
      requires eosToken in lm.Tokens
      ensures lm.ValidTokensIdsLogits()
      ensures next in lm.Tokens
      ensures (next == eosToken) || (parser.ValidNextToken(prefix, next))
      ensures (next != eosToken) ==> (forall t: Token :: t in parser.ValidNextTokens(prefix + [next]) ==> t in lm.Tokens)
      ensures next == eosToken ==> (parser.IsCompletePrefix(prefix) || parser.ValidNextTokenCount(prefix) == 0)
      ensures cost == old(cost) + 1
    {
      lm.GenerateLogits(prompt + prefix);
      var scalar := 1.0 / temperature;
      if scalar > 100000000.0 { scalar := 100000000.0; }
      ScaleAllLogits(lm, scalar);
      RollbackPreservesTokenInvariant(lm, parser, prefix);
      lm.MaskValidNextAndEos(parser, prefix, eosToken);
      next := lm.ChooseNextToken();
      if next != eosToken {
        assert !lm.IsMasked(next);
        assert parser.ValidNextToken(prefix, next);
        assert parser.IsValidPrefix(prefix + [next]);
        ConstrainedStepNextValid(lm, parser, prefix, next);
      }
      cost := cost + 1;
    }

    method SafeTemperatureConstrainedStep(
      lm: LM, parser: Parser, prompt: Prefix, prefix: Prefix, temperature: real, eosToken: Token
    ) returns (next: Token)
      modifies lm.Logits, this
      requires lm.ValidTokensIdsLogits()
      requires parser.IsValidPrefix(prefix)
      requires eosToken in lm.Tokens
      ensures lm.ValidTokensIdsLogits()
      ensures next in lm.Tokens
      ensures (next == eosToken) || (parser.ValidNextToken(prefix, next))
      ensures (next != eosToken) ==> (forall t: Token :: t in parser.ValidNextTokens(prefix + [next]) ==> t in lm.Tokens)
      ensures next == eosToken ==> (parser.IsCompletePrefix(prefix) || parser.ValidNextTokenCount(prefix) == 0)
      ensures cost == old(cost) + 1
    {
      lm.GenerateLogits(prompt + prefix);
      var safeTemperature := temperature;
      if safeTemperature < 0.00000001 {
        safeTemperature := 0.00000001;
      }
      if safeTemperature > 100000000.0 {
        safeTemperature := 100000000.0;
      }
      var scalar := 1.0 / safeTemperature;
      if scalar > 100000000.0 {
        scalar := 100000000.0;
      }
      if scalar <= 0.0 {
        scalar := 1.0;
      }
      ScaleAllLogits(lm, scalar);
      RollbackPreservesTokenInvariant(lm, parser, prefix);
      lm.MaskValidNextAndEos(parser, prefix, eosToken);
      next := lm.ChooseNextToken();
      if next != eosToken {
        assert !lm.IsMasked(next);
        assert parser.ValidNextToken(prefix, next);
        assert parser.IsValidPrefix(prefix + [next]);
        ConstrainedStepNextValid(lm, parser, prefix, next);
      }
      cost := cost + 1;
    }

    method SaveLogitsSnapshot(lm: LM) returns (snapshot: seq<Logit>)
      requires lm.ValidTokensIdsLogits()
      ensures lm.ValidTokensIdsLogits()
      ensures |snapshot| == lm.Logits.Length
      ensures forall i :: 0 <= i < lm.Logits.Length ==> snapshot[i] == lm.Logits[i]
      ensures forall i :: 0 <= i < |snapshot| ==>
        -1000000000.0 <= snapshot[i] && snapshot[i] <= 1000000000.0
      ensures cost == old(cost)
    {
      snapshot := lm.Logits[0..lm.Logits.Length];
    }

    method RestoreLogitsSnapshot(lm: LM, snapshot: seq<Logit>)
      modifies lm.Logits, this
      requires lm.ValidTokensIdsLogits()
      requires |snapshot| == lm.Logits.Length
      requires forall i :: 0 <= i < |snapshot| ==> -1000000000.0 <= snapshot[i] <= 1000000000.0
      ensures lm.ValidTokensIdsLogits()
      ensures forall i :: 0 <= i < lm.Logits.Length ==> lm.Logits[i] == snapshot[i]
      ensures cost == old(cost)
    {
      var i := 0;
      while i < lm.Logits.Length
        invariant 0 <= i <= lm.Logits.Length
        invariant lm.ValidTokensIdsLogits()
        invariant forall j :: 0 <= j < i ==> lm.Logits[j] == snapshot[j]
        invariant forall j :: i <= j < lm.Logits.Length ==> lm.Logits[j] == old(lm.Logits[j])
        invariant cost == old(cost)
        decreases lm.Logits.Length - i
      {
        lm.Logits[i] := snapshot[i];
        i := i + 1;
      }
    }

    method RolloutConstrainedWithPenalties(
      lm: LM, parser: Parser, prompt: Prefix, startPrefix: Prefix,
      totalBudget: nat, penalties: seq<Token>, penaltyAmount: real, eosToken: Token
    ) returns (generatedOut: Prefix, stepsUsed: nat, terminatedByEos: bool)
      modifies lm.Logits, this
      requires lm.ValidTokensIdsLogits()
      requires parser.IsValidPrefix(startPrefix)
      requires eosToken in lm.Tokens
      requires penaltyAmount >= 0.0 && penaltyAmount <= 100000000.0
      ensures lm.ValidTokensIdsLogits()
      ensures parser.IsValidPrefix(generatedOut)
      ensures |startPrefix| <= |generatedOut| <= |startPrefix| + totalBudget
      ensures |generatedOut| <= |startPrefix| + stepsUsed
      ensures stepsUsed <= totalBudget
      ensures cost == old(cost) + stepsUsed
    {
      generatedOut := startPrefix;
      stepsUsed := 0;
      terminatedByEos := false;
      while stepsUsed < totalBudget && !parser.IsCompletePrefix(generatedOut)
        invariant 0 <= stepsUsed <= totalBudget
        invariant lm.ValidTokensIdsLogits()
        invariant parser.IsValidPrefix(generatedOut)
        invariant |startPrefix| <= |generatedOut| <= |startPrefix| + stepsUsed
        invariant !terminatedByEos
        invariant cost == old(cost) + stepsUsed
        decreases totalBudget - stepsUsed
      {
        var next := SafePenalizedConstrainedStep(
          lm, parser, prompt, generatedOut, penalties, penaltyAmount, eosToken
        );
        if next == eosToken {
          terminatedByEos := true;
          stepsUsed := stepsUsed + 1;
          break;
        }
        generatedOut := generatedOut + [next];
        stepsUsed := stepsUsed + 1;
      }
    }

    method SpeculativeConstrainedRollout(
      lm: LM, parser: Parser, prompt: Prefix, constrainedPrefix: Prefix,
      numTokens: nat, eosToken: Token
    ) returns (candidateTokens: Prefix, candidatePrefix: Prefix,
               hitComplete: bool, hitEos: bool, stepsUsed: nat)
      modifies lm.Logits, this
      requires lm.ValidTokensIdsLogits()
      requires parser.IsValidPrefix(constrainedPrefix)
      requires eosToken in lm.Tokens
      requires numTokens >= 1
      ensures lm.ValidTokensIdsLogits()
      ensures parser.IsValidPrefix(candidatePrefix)
      ensures candidatePrefix == constrainedPrefix + candidateTokens
      ensures |candidateTokens| <= numTokens
      ensures stepsUsed <= numTokens
      ensures cost == old(cost) + stepsUsed
    {
      var snap := SaveLogitsSnapshot(lm);
      candidateTokens := [];
      var cur := constrainedPrefix;
      stepsUsed := 0;
      hitEos := false;

      while stepsUsed < numTokens && !parser.IsCompletePrefix(cur) && !hitEos
        invariant lm.ValidTokensIdsLogits()
        invariant parser.IsValidPrefix(cur)
        invariant |constrainedPrefix| <= |cur|
        invariant cur[..|constrainedPrefix|] == constrainedPrefix
        invariant candidateTokens == cur[|constrainedPrefix|..]
        invariant |candidateTokens| + |constrainedPrefix| == |cur|
        invariant |candidateTokens| <= stepsUsed <= numTokens
        invariant hitEos ==> |candidateTokens| + 1 <= stepsUsed
        invariant !hitEos ==> |candidateTokens| == stepsUsed
        invariant cost == old(cost) + stepsUsed
        decreases numTokens - stepsUsed, if hitEos || parser.IsCompletePrefix(cur) then 0 else 1
      {
        var next := ConstrainedStep(lm, parser, prompt, cur, eosToken);
        stepsUsed := stepsUsed + 1;
        if next == eosToken {
          hitEos := true;
        } else {
          cur := cur + [next];
          candidateTokens := candidateTokens + [next];
        }
      }

      RestoreLogitsSnapshot(lm, snap);
      candidatePrefix := cur;
      hitComplete := parser.IsCompletePrefix(cur);
    }

    // CRANE GSM adaptive: unconstrained until "<<", then
    // forward(start)/view/valid_vars/backward loop (max_iter=80, backwards_limit=20).
    method CraneGeneration(
      lm: LM,
      parser: Parser,
      prompt: Prefix,
      maxSteps: nat,
      minReasoningSteps: nat,
      validTokenGroups: seq<seq<Token>>,
      eosToken: Token
    ) returns (generated: Prefix)
      modifies lm.Logits, this
      requires lm.ValidTokensIdsLogits()
      requires "<<" in lm.Tokens && ">>" in lm.Tokens
      requires eosToken in lm.Tokens
      requires parser.IsValidPrefix([])
      ensures lm.ValidTokensIdsLogits()
      ensures |generated| <= maxSteps
      ensures cost <= old(cost) + maxSteps
    {
      generated := [];
      var startCost := cost;
      var insideConstrained := false;
      var currentConstrained: Prefix := [];
      var unitIters := 0;
      var numBackwards := 0;
      var maxIter := 80;
      var backwardsLimit := 20;

      while cost < startCost + maxSteps
        invariant startCost <= cost <= startCost + maxSteps
        invariant |generated| <= maxSteps
        invariant |currentConstrained| <= |generated|
        invariant lm.ValidTokensIdsLogits()
        invariant !insideConstrained ==> currentConstrained == []
        invariant insideConstrained ==> parser.IsValidPrefix(currentConstrained)
        invariant 0 <= unitIters <= maxIter + 1
        invariant 0 <= numBackwards
        decreases startCost + maxSteps - cost, (if insideConstrained then 1 else 0), (maxIter + 1 - unitIters)
      {
        if !insideConstrained {
          var next := UnconstrainedStep(lm, prompt, generated);
          if next == eosToken {
            break;
          }
          if |generated| >= maxSteps {
            break;
          }
          generated := generated + [next];
          // CRANE: `start_symbol in unconstrained_gen`. Check last token and the
          // last-two-token render so split '<'+'<' opens without O(n^2) full re-render.
          var openHit := Contains(next, "<<");
          if !openHit && |generated| >= 2 {
            openHit := Contains(RenderPrefix(generated[|generated| - 2..]), "<<");
          }
          if openHit {
            insideConstrained := true;
            currentConstrained := [];
            unitIters := 0;
            numBackwards := 0;
          }
        } else {
          if RenderedEndsWith(generated, ">>") {
            insideConstrained := false;
            currentConstrained := [];
          } else if unitIters >= maxIter {
            // CRANE parity: do not flip to unconstrained mid-<<...>>; stop.
            break;
          } else {
            var spanStart := |generated| - |currentConstrained|;
            var constrainedPrompt := prompt + generated[..spanStart];
            var budgetLeft := startCost + maxSteps - cost;
            var beforeCost := cost;
            // CRANE gsm_symbolic_constraints: forward(num=1) with default_unit=start
            // (one full <<expr>>), not per-var. Waiting on "var" never completes during
            // NUMBER → unbounded digit sink (ex0 zeros).
            currentConstrained := ForwardUntilSymbol(
              lm, parser, constrainedPrompt, currentConstrained, eosToken,
              "start", 1, budgetLeft
            );
            unitIters := unitIters + 1;
            if |currentConstrained| > maxSteps {
              ValidPrefixesAreClosed(parser, currentConstrained, maxSteps);
              currentConstrained := currentConstrained[..maxSteps];
            }
            generated := generated[..spanStart] + currentConstrained;
            if |generated| > maxSteps {
              generated := generated[..maxSteps];
              if |generated| >= spanStart {
                ValidPrefixesAreClosed(parser, currentConstrained, maxSteps - spanStart);
                assert generated[spanStart..] == currentConstrained[..maxSteps - spanStart];
                currentConstrained := generated[spanStart..];
              } else {
                currentConstrained := [];
              }
            }
            if RenderedEndsWith(generated, ">>") {
              insideConstrained := false;
              currentConstrained := [];
            } else if cost == beforeCost {
              // No progress: stop rather than unconstrained mid-span.
              break;
            } else {
              var lastVar := ViewLastSymbol(parser, currentConstrained, "var");
              var allowed := IsAllowedVarText(validTokenGroups, lastVar);
              if lastVar != "" && !allowed {
                if numBackwards < backwardsLimit {
                  currentConstrained := BackwardToSymbol(parser, currentConstrained, "var", 1);
                  generated := generated[..spanStart] + currentConstrained;
                  numBackwards := numBackwards + 1;
                } else {
                  numBackwards := 0;
                }
              }
            }
          }
        }
      }
    }

    // One self-discharging decode step. Advances generation by at most one token
    // and charges exactly one unit of cost on EVERY control path, so a caller's
    // single `while steps < maxSteps` loop that calls only ManagedStep and then
    // sets `cost := helpers.cost` discharges the strategy-level length, cost and
    // progress postconditions by construction (loop runs >=1 iteration when
    // maxSteps>0, and cost==steps). `done` is true when the step hit EOS or closed
    // the span (the caller may stop). Outside a span: one UnconstrainedStep; "<<"
    // opens a constrained span. Inside a span: close if the parser reports a
    // complete prefix, else one AdaptiveConstrainedStep and append it. Composes
    // only already-verified CSDHelpers primitives; adds no new decode behavior.
    // Body is exactly one iteration of GenerateWithManagedSpan's loop.
    method ManagedStep(
      lm: LM,
      parser: Parser,
      prompt: Prefix,
      generated: Prefix,
      insideConstrained: bool,
      currentConstrained: Prefix,
      validTokenGroups: seq<seq<Token>>,
      boostAmount: real,
      narrowThreshold: nat,
      eosToken: Token
    ) returns (
      generatedOut: Prefix,
      insideOut: bool,
      currentOut: Prefix,
      done: bool
    )
      modifies lm.Logits, this
      requires lm.ValidTokensIdsLogits()
      requires parser.IsValidPrefix([])
      requires !insideConstrained ==> currentConstrained == []
      requires insideConstrained ==> parser.IsValidPrefix(currentConstrained)
      requires insideConstrained ==> |currentConstrained| <= |generated|
      requires "<<" in lm.Tokens && ">>" in lm.Tokens
      requires boostAmount >= 0.0 && boostAmount <= 100000000.0
      requires eosToken in lm.Tokens
      ensures lm.ValidTokensIdsLogits()
      ensures cost == old(cost) + 1
      ensures |generatedOut| <= |generated| + 1
      ensures !insideOut ==> currentOut == []
      ensures insideOut ==> parser.IsValidPrefix(currentOut)
      ensures insideOut ==> |currentOut| <= |generatedOut|
    {
      generatedOut := generated;
      insideOut := insideConstrained;
      currentOut := currentConstrained;
      done := false;
      if !insideConstrained {
        var next := UnconstrainedStep(lm, prompt, generated);
        if next == eosToken {
          done := true;
          return;
        }
        generatedOut := generated + [next];
        // Match the opener the same way the closer is matched: on the rendered
        // text, not on an exact token. Qwen tokenizes the opener as ' <<' (with
        // a leading space) in prose, so `next == "<<"` almost never fires.
        var openHit := Contains(next, "<<");
        if !openHit && |generatedOut| >= 2 {
          openHit := Contains(RenderPrefix(generatedOut[|generatedOut| - 2..]), "<<");
        }
        if openHit {
          insideOut := true;
          currentOut := [];
        }
      } else {
        var cg, ci, cc, closed := CloseSpanIfComplete(lm, parser, generated, currentConstrained);
        if closed {
          generatedOut := cg;
          insideOut := ci;
          currentOut := cc;
          done := true;
          return;
        } else {
          var constrainedPrompt := prompt + generated[..|generated| - |currentConstrained|];
          var next := AdaptiveConstrainedStep(
            lm, parser, constrainedPrompt, currentConstrained,
            validTokenGroups, boostAmount, narrowThreshold, eosToken
          );
          if next == eosToken {
            done := true;
            return;
          } else {
            var appendedGenerated, appendedInside, appendedCurrent := AppendConstrainedToken(
              lm, parser, generated, currentConstrained, next
            );
            generatedOut := appendedGenerated;
            insideOut := appendedInside;
            currentOut := appendedCurrent;
          }
        }
      }
    }

    // Higher-order span-managed generation. Runs a full free-then-constrained
    // decode loop and discharges the strategy-level length, cost, progress and
    // parser-validity postconditions internally, so a caller need not write a
    // loop or its proof. Outside a span: UnconstrainedStep until "<<" is observed.
    // Inside a span: close if the parser reports a complete prefix, else take one
    // AdaptiveConstrainedStep and append it. Composes only already-verified
    // CSDHelpers primitives; adds no new decode behavior.
    method GenerateWithManagedSpan(
      lm: LM,
      parser: Parser,
      prompt: Prefix,
      generatedPrefix: Prefix,
      insideConstrained: bool,
      currentConstrained: Prefix,
      maxSteps: nat,
      validTokenGroups: seq<seq<Token>>,
      boostAmount: real,
      narrowThreshold: nat,
      eosToken: Token
    ) returns (
      generated: Prefix,
      insideConstrainedOut: bool,
      currentConstrainedOut: Prefix
    )
      modifies lm.Logits, this
      requires lm.ValidTokensIdsLogits()
      requires parser.IsValidPrefix([])
      requires !insideConstrained ==> currentConstrained == []
      requires insideConstrained ==> parser.IsValidPrefix(currentConstrained)
      requires insideConstrained ==> |currentConstrained| <= |generatedPrefix|
      requires "<<" in lm.Tokens && ">>" in lm.Tokens
      requires boostAmount >= 0.0 && boostAmount <= 100000000.0
      requires eosToken in lm.Tokens
      ensures lm.ValidTokensIdsLogits()
      ensures |generated| <= |generatedPrefix| + maxSteps
      ensures !insideConstrainedOut ==> currentConstrainedOut == []
      ensures insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)
      ensures cost <= old(cost) + maxSteps
      ensures maxSteps == 0 || cost > old(cost) || generated != generatedPrefix ||
              insideConstrainedOut != insideConstrained ||
              currentConstrainedOut != currentConstrained
    {
      generated := generatedPrefix;
      insideConstrainedOut := insideConstrained;
      currentConstrainedOut := currentConstrained;

      var steps: nat := 0;
      while steps < maxSteps
        invariant 0 <= steps <= maxSteps
        invariant lm.ValidTokensIdsLogits()
        invariant !insideConstrainedOut ==> currentConstrainedOut == []
        invariant insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)
        invariant insideConstrainedOut ==> |currentConstrainedOut| <= |generated|
        invariant |generated| <= |generatedPrefix| + steps
        invariant cost == old(cost) + steps
        decreases maxSteps - steps
      {
        if !insideConstrainedOut {
          var next := UnconstrainedStep(lm, prompt, generated);
          steps := steps + 1;
          if next == eosToken {
            break;
          }
          generated := generated + [next];
          // Rendered-text opener match; see the note at the sibling site above.
          var openHit := Contains(next, "<<");
          if !openHit && |generated| >= 2 {
            openHit := Contains(RenderPrefix(generated[|generated| - 2..]), "<<");
          }
          if openHit {
            insideConstrainedOut := true;
            currentConstrainedOut := [];
          }
        } else {
          var cg, ci, cc, closed := CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut);
          steps := steps + 1;
          if closed {
            generated := cg;
            insideConstrainedOut := ci;
            currentConstrainedOut := cc;
            break;
          } else {
            var constrainedPrompt := prompt + generated[..|generated| - |currentConstrainedOut|];
            var next := AdaptiveConstrainedStep(
              lm, parser, constrainedPrompt, currentConstrainedOut,
              validTokenGroups, boostAmount, narrowThreshold, eosToken
            );
            if next == eosToken {
              break;
            } else {
              var appendedGenerated, appendedInside, appendedCurrent := AppendConstrainedToken(
                lm, parser, generated, currentConstrainedOut, next
              );
              generated := appendedGenerated;
              insideConstrainedOut := appendedInside;
              currentConstrainedOut := appendedCurrent;
            }
          }
        }
      }
    }

    // Like GenerateWithManagedSpan, but the unconstrained PREAMBLE is hard-capped at
    // `prefixBudget` steps: after that many unconstrained tokens (or once "<<" is
    // observed) the span is force-opened, so the constrained phase is guaranteed
    // budget to reach ">>".  A single unified step counter advances by exactly 1 per
    // step, so this discharges the length (|generated| <= |generatedPrefix| + maxSteps),
    // cost (cost <= old(cost) + maxSteps), and progress postconditions by construction
    // — a strategy needs only one call plus `cost := helpers.cost`, with no hand-rolled
    // budget bookkeeping.  Modeled on GenerateWithManagedSpan (identical invariants); the
    // only added branch force-opens via the proven OpenConstrainedSpan.
    method GenerateWithPrefixAndManagedSpan(
      lm: LM,
      parser: Parser,
      prompt: Prefix,
      generatedPrefix: Prefix,
      insideConstrained: bool,
      currentConstrained: Prefix,
      maxSteps: nat,
      prefixBudget: nat,
      validTokenGroups: seq<seq<Token>>,
      boostAmount: real,
      narrowThreshold: nat,
      eosToken: Token
    ) returns (
      generated: Prefix,
      insideConstrainedOut: bool,
      currentConstrainedOut: Prefix
    )
      modifies lm.Logits, this
      requires lm.ValidTokensIdsLogits()
      requires parser.IsValidPrefix([])
      requires !insideConstrained ==> currentConstrained == []
      requires insideConstrained ==> parser.IsValidPrefix(currentConstrained)
      requires insideConstrained ==> |currentConstrained| <= |generatedPrefix|
      requires "<<" in lm.Tokens && ">>" in lm.Tokens
      requires boostAmount >= 0.0 && boostAmount <= 100000000.0
      requires prefixBudget <= maxSteps
      requires eosToken in lm.Tokens
      ensures lm.ValidTokensIdsLogits()
      ensures |generated| <= |generatedPrefix| + maxSteps
      ensures !insideConstrainedOut ==> currentConstrainedOut == []
      ensures insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)
      ensures cost <= old(cost) + maxSteps
      ensures maxSteps == 0 || cost > old(cost) || generated != generatedPrefix ||
              insideConstrainedOut != insideConstrained ||
              currentConstrainedOut != currentConstrained
    {
      generated := generatedPrefix;
      insideConstrainedOut := insideConstrained;
      currentConstrainedOut := currentConstrained;

      var steps: nat := 0;
      while steps < maxSteps
        invariant 0 <= steps <= maxSteps
        invariant lm.ValidTokensIdsLogits()
        invariant !insideConstrainedOut ==> currentConstrainedOut == []
        invariant insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)
        invariant insideConstrainedOut ==> |currentConstrainedOut| <= |generated|
        invariant |generated| <= |generatedPrefix| + steps
        invariant cost == old(cost) + steps
        decreases maxSteps - steps
      {
        if !insideConstrainedOut {
          if steps < prefixBudget {
            var next := UnconstrainedStep(lm, prompt, generated);
            steps := steps + 1;
            if next == eosToken {
              break;
            }
            generated := generated + [next];
            // Rendered-text opener match; see the note at the sibling site above.
            var openHit := Contains(next, "<<");
            if !openHit && |generated| >= 2 {
              openHit := Contains(RenderPrefix(generated[|generated| - 2..]), "<<");
            }
            if openHit {
              insideConstrainedOut := true;
              currentConstrainedOut := [];
            }
          } else {
            var go, io, co := OpenConstrainedSpan(lm, generated);
            steps := steps + 1;
            generated := go;
            insideConstrainedOut := io;
            currentConstrainedOut := co;
          }
        } else {
          var cg, ci, cc, closed := CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut);
          steps := steps + 1;
          if closed {
            generated := cg;
            insideConstrainedOut := ci;
            currentConstrainedOut := cc;
            break;
          } else {
            var constrainedPrompt := prompt + generated[..|generated| - |currentConstrainedOut|];
            var next := AdaptiveConstrainedStep(
              lm, parser, constrainedPrompt, currentConstrainedOut,
              validTokenGroups, boostAmount, narrowThreshold, eosToken
            );
            if next == eosToken {
              break;
            } else {
              var appendedGenerated, appendedInside, appendedCurrent := AppendConstrainedToken(
                lm, parser, generated, currentConstrainedOut, next
              );
              generated := appendedGenerated;
              insideConstrainedOut := appendedInside;
              currentConstrainedOut := appendedCurrent;
            }
          }
        }
      }
    }

    // Thin wrapper: closes the current constrained span if the parser reports the
    // accumulated tokens form a complete (accepting) parse, otherwise leaves the
    // span open unchanged.  All heavy lifting delegates to the proven
    // CloseConstrainedSpan method; this wrapper adds no new logic.
    method CloseSpanIfComplete(
      lm: LM, parser: Parser, generated: Prefix, currentConstrained: Prefix
    ) returns (generatedOut: Prefix, insideOut: bool, currentOut: Prefix, closed: bool)
      modifies this
      requires lm.ValidTokensIdsLogits()
      requires parser.IsValidPrefix(currentConstrained)
      requires ">>" in lm.Tokens
      ensures lm.ValidTokensIdsLogits()
      ensures |generatedOut| <= |generated| + 1
      ensures parser.IsCompletePrefix(currentConstrained) ==>
              (!insideOut && currentOut == [] && cost == old(cost) + 1 && closed)
      ensures !parser.IsCompletePrefix(currentConstrained) ==>
              (generatedOut == generated && insideOut == true &&
               currentOut == currentConstrained && cost == old(cost) && !closed)
    {
      if parser.IsCompletePrefix(currentConstrained) {
        generatedOut, insideOut, currentOut := CloseConstrainedSpan(lm, parser, generated, currentConstrained);
        closed := true;
      } else {
        generatedOut := generated;
        insideOut := true;
        currentOut := currentConstrained;
        closed := false;
      }
    }

    // IterGen-style unit-level iterative improvement.
    //
    // Generates constrained tokens one at a time. Each time the parser
    // transitions to a COMPLETE prefix (a "unit boundary"), it checks whether
    // the rendered text of that unit is an element of `allowedUnits`. If
    // `allowedUnits` is empty the check is disabled (all units pass). On a
    // check failure:
    //   * if rollbackBudgetLeft > 0 AND retryCount < maxRetries, the span is
    //     rolled back to the last complete point before this unit, the first
    //     token after that point is penalized in the logits (recurrence penalty),
    //     and generation continues — mirroring IterGen's backward(unit) + retry.
    //   * otherwise the current (possibly bad) result is accepted and generation
    //     continues, preserving termination.
    //
    // Fairness: allowedUnits is expected to be populated from the schema text
    // already visible in the prompt (the same information IterGen's db_info
    // string provides), not from DB execution.
    //
    // Cost accounting: bounded by a single flat `budget` of total steps (NOT a
    // product of two free parameters), mirroring CloseSpanWithinBudget /
    // RegenerateUnitOnGroundingFailure. A caller passes its whole remaining step
    // budget and the per-call bound composes directly against the strategy
    // template's `cost <= maxSteps`, with no nonlinear-arithmetic / division lemma.
    // maxRetries / maxRollbackBudget stay as behavioral knobs (they gate the
    // rollback branch; they no longer size the bound).
    method RegenerateUnitOnCheckFailure(
      lm: LM, parser: Parser, prompt: Prefix, currentConstrained: Prefix,
      eosToken: Token,
      budget: nat,
      maxRetries: nat,
      maxRollbackBudget: nat,
      allowedUnits: seq<string>
    ) returns (resultConstrained: Prefix)
      modifies lm.Logits, this
      requires lm.ValidTokensIdsLogits()
      requires parser.IsValidPrefix(currentConstrained)
      requires eosToken in lm.Tokens
      ensures lm.ValidTokensIdsLogits()
      ensures parser.IsValidPrefix(resultConstrained)
      ensures cost <= old(cost) + budget
      ensures cost >= old(cost)
      // Length bound (mirrors RegenerateUnitOnGroundingFailure): each loop step
      // appends at most one net token (rollbacks only shrink the span), so the
      // produced span grows by at most the total step budget. Without this a
      // caller cannot prove the strategy template's
      // `|generated| <= |generatedPrefix| + maxSteps` postcondition.
      ensures |resultConstrained| <= |currentConstrained| + budget
    {
      resultConstrained := currentConstrained;
      // The last prefix for which IsCompletePrefix held (or the entry point).
      var checkpointConstrained := currentConstrained;
      var retryCount := 0;
      var rollbackBudgetUsed := 0;
      var steps := 0;
      var totalBound := budget;

      while steps < totalBound
        invariant lm.ValidTokensIdsLogits()
        invariant parser.IsValidPrefix(resultConstrained)
        invariant parser.IsValidPrefix(checkpointConstrained)
        invariant |checkpointConstrained| <= |resultConstrained|
        invariant |resultConstrained| <= |currentConstrained| + steps
        invariant 0 <= steps <= totalBound
        invariant cost == old(cost) + steps
        decreases totalBound - steps
      {
        var next := ConstrainedStep(lm, parser, prompt, resultConstrained, eosToken);
        steps := steps + 1;
        if next == eosToken {
          break;
        }
        var extended := resultConstrained + [next];
        resultConstrained := extended;

        // Unit boundary: IsCompletePrefix just became true.
        if parser.IsCompletePrefix(resultConstrained) {
          // Render the unit (from checkpointConstrained to here) and check it.
          var unitText := RenderPrefix(resultConstrained[|checkpointConstrained|..]);
          var passes := |allowedUnits| == 0 || unitText in allowedUnits;
          if passes {
            // Accept the unit: advance the checkpoint.
            checkpointConstrained := resultConstrained;
            retryCount := 0;
          } else if rollbackBudgetUsed < maxRollbackBudget && retryCount < maxRetries {
            // Roll back to the checkpoint and penalize the continuation.
            retryCount := retryCount + 1;
            rollbackBudgetUsed := rollbackBudgetUsed + 1;
            resultConstrained := checkpointConstrained;
            // Penalize the rejected first-token-past-checkpoint in current logits
            // so the model is steered away from the same continuation on retry.
            lm.GenerateLogits(prompt + resultConstrained);
            lm.MaskValidNextAndEos(parser, resultConstrained, eosToken);
            if next in lm.Tokens {
              lm.MaskToken(next);
            }
          } else {
            // No budget or retries left: accept the unit anyway to preserve termination.
            checkpointConstrained := resultConstrained;
            retryCount := 0;
          }
        }
      }
    }

    // Like RegenerateUnitOnCheckFailure, but the per-unit acceptance test is the
    // grounding predicate lm.SpanGrounded(renderedUnit) instead of membership in
    // a caller-supplied allowed set. It checks the identifier-like tokens WITHIN
    // each completed unit against the support set the host derives from the
    // prompt, rather than matching the whole rendered unit string. The rollback /
    // penalize / regenerate loop and all bounds are identical.
    //
    // Fairness: the grounding support set is taken only from prompt text (the
    // same information visible in the prompt to any baseline), not from DB
    // execution or gold labels.
    //
    // Cost accounting: bounded by a single flat `budget` of total steps (NOT a
    // product of two free parameters), mirroring CloseSpanWithinBudget. A caller
    // passes its whole remaining step budget and the per-call bound composes
    // directly against the strategy template's `cost <= maxSteps` postcondition,
    // with no nonlinear-arithmetic or division-sizing proof obligation.
    method RegenerateUnitOnGroundingFailure(
      lm: LM, parser: Parser, prompt: Prefix, currentConstrained: Prefix,
      eosToken: Token,
      budget: nat,
      maxRetries: nat,
      maxRollbackBudget: nat
    ) returns (resultConstrained: Prefix)
      modifies lm.Logits, this
      requires lm.ValidTokensIdsLogits()
      requires parser.IsValidPrefix(currentConstrained)
      requires eosToken in lm.Tokens
      ensures lm.ValidTokensIdsLogits()
      ensures parser.IsValidPrefix(resultConstrained)
      ensures cost <= old(cost) + budget
      ensures cost >= old(cost)
      // Length bound: each loop step appends at most one net token (rollbacks
      // only shrink the span), so the produced span grows by at most `budget`.
      // Without this a caller cannot prove the strategy template's
      // `|generated| <= |generatedPrefix| + maxSteps` postcondition.
      ensures |resultConstrained| <= |currentConstrained| + budget
    {
      resultConstrained := currentConstrained;
      // The last prefix at which a schema symbol (table/column) completed grounded
      // (or the entry point). Rollbacks return here, so retries are CHEAP — they
      // replay only the current symbol, not the whole query.
      var checkpointConstrained := currentConstrained;
      // Count of schema symbols completed at the checkpoint. The unit boundary
      // fires when the live count rises above this — i.e. one more table/column
      // name just finished mid-query (IterGen's SymbolPosMap boundary), instead of
      // waiting for the whole query to parse (IsCompletePrefix).
      var prevCount := parser.CompletedSchemaSymbolCount(currentConstrained);
      var retryCount := 0;
      var rollbackBudgetUsed := 0;
      var steps := 0;
      var totalBound := budget;

      while steps < totalBound
        invariant lm.ValidTokensIdsLogits()
        invariant parser.IsValidPrefix(resultConstrained)
        invariant parser.IsValidPrefix(checkpointConstrained)
        invariant |checkpointConstrained| <= |resultConstrained|
        invariant 0 <= steps <= totalBound
        invariant cost == old(cost) + steps
        invariant |resultConstrained| <= |currentConstrained| + steps
        decreases totalBound - steps
      {
        var next := ConstrainedStep(lm, parser, prompt, resultConstrained, eosToken);
        steps := steps + 1;
        if next == eosToken {
          break;
        }
        var extended := resultConstrained + [next];
        resultConstrained := extended;

        // Unit boundary: a table_ref/column_ref symbol just COMPLETED mid-query
        // (the schema-symbol count rose). Ground-check from the last grounded
        // checkpoint to here, exactly when IterGen checks membership.
        var newCount := parser.CompletedSchemaSymbolCount(resultConstrained);
        if newCount > prevCount {
          // Ground-check the unit (from checkpointConstrained to here) AND locate
          // the first out-of-schema identifier's token in one call. found=false
          // means every identifier is grounded (same signal as SpanGrounded).
          var unit := resultConstrained[|checkpointConstrained|..];
          var found, idx := lm.FirstUngroundedIdentifierTokenIdx(unit);
          if !found {
            // Accept the unit: advance the checkpoint and the symbol count.
            checkpointConstrained := resultConstrained;
            prevCount := newCount;
            retryCount := 0;
          } else if rollbackBudgetUsed < maxRollbackBudget && retryCount < maxRetries {
            // Roll the generation cursor back to the checkpoint and PERSISTENTLY
            // penalize the OUT-OF-SCHEMA identifier's token at its OWN position
            // (not the unit's first token). Under greedy decode the replay from
            // the checkpoint is deterministic, so it re-reaches that exact prefix;
            // PenalizeTriedTokenAt is re-applied there every regen, steering the
            // identifier away from the out-of-schema name. (A one-shot MaskToken
            // would be wiped by the next GenerateLogits and loop forever.)
            // `idx < |unit|` (extern postcondition) ==> badPos < |resultConstrained|.
            retryCount := retryCount + 1;
            rollbackBudgetUsed := rollbackBudgetUsed + 1;
            var badPos := |checkpointConstrained| + idx;
            var badToken := resultConstrained[badPos];
            var penalizePrefix := resultConstrained[..badPos];
            resultConstrained := checkpointConstrained;
            lm.GenerateLogits(prompt + resultConstrained);
            lm.MaskValidNextAndEos(parser, resultConstrained, eosToken);
            lm.PenalizeTriedTokenAt(prompt + penalizePrefix, badToken);
          } else {
            // No budget or retries left: accept the unit anyway to preserve termination.
            checkpointConstrained := resultConstrained;
            prevCount := newCount;
            retryCount := 0;
          }
        }
      }
    }

    // Advance an open constrained span toward a completable state and emit the
    // closing delimiter, all within `budget` steps. Generates forward
    // (dead-end-aware) tracking the longest prefix that parses as complete, then
    // emits ">>" at that longest complete point, reserving one step for the close.
    // When no completable state is reachable within the budget the span is left
    // open (the grammar forbids closing an incomplete prefix). Composes only the
    // already-verified DeadEndAvoidingStep and CloseConstrainedSpan primitives.
    method CloseSpanWithinBudget(
      lm: LM, parser: Parser, prompt: Prefix, generated: Prefix,
      currentConstrained: Prefix, eosToken: Token, budget: nat
    ) returns (generatedOut: Prefix, insideOut: bool, currentOut: Prefix)
      modifies lm.Logits, this
      requires lm.ValidTokensIdsLogits()
      requires parser.IsValidPrefix(currentConstrained)
      requires |currentConstrained| <= |generated|
      requires eosToken in lm.Tokens
      requires ">>" in lm.Tokens
      ensures lm.ValidTokensIdsLogits()
      ensures !insideOut ==> currentOut == []
      ensures insideOut ==> parser.IsValidPrefix(currentOut)
      ensures insideOut ==> |currentOut| <= |generatedOut|
      ensures insideOut ==> generatedOut[|generatedOut| - |currentOut|..] == currentOut
      ensures |generatedOut| <= |generated| + budget
      ensures cost <= old(cost) + budget
      ensures cost >= old(cost)
    {
      var stablePrefix := generated[..|generated| - |currentConstrained|];
      var running := currentConstrained;
      var bestComplete: Prefix := [];
      var haveComplete := false;
      if parser.IsCompletePrefix(currentConstrained) {
        bestComplete := currentConstrained;
        haveComplete := true;
      }
      var steps := 0;

      while steps + 1 < budget
        invariant lm.ValidTokensIdsLogits()
        invariant parser.IsValidPrefix(running)
        invariant |running| <= |currentConstrained| + steps
        invariant haveComplete ==>
          (parser.IsCompletePrefix(bestComplete) && |bestComplete| <= |running|)
        invariant !haveComplete ==> bestComplete == []
        invariant 0 <= steps <= budget
        invariant cost == old(cost) + steps
        decreases budget - steps
      {
        var next, ok := DeadEndAvoidingStep(lm, parser, prompt + stablePrefix, running, eosToken, 8);
        steps := steps + 1;
        if !ok || next == eosToken {
          break;
        }
        running := running + [next];
        if parser.IsCompletePrefix(running) {
          bestComplete := running;
          haveComplete := true;
        }
      }

      if steps < budget && haveComplete {
        var gc, ci, cc := CloseConstrainedSpan(lm, parser, stablePrefix + bestComplete, bestComplete);
        generatedOut := gc;
        insideOut := ci;
        currentOut := cc;
      } else {
        generatedOut := stablePrefix + running;
        insideOut := true;
        currentOut := running;
        assert |generatedOut| - |currentOut| == |stablePrefix|;
        assert generatedOut[|generatedOut| - |currentOut|..] == currentOut;
      }
    }
  }
}
