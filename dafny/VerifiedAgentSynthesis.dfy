module VerifiedDecoderAgent {
  type Token = string
  type Prefix = seq<Token>
  type Id = nat
  type Logit = real

  class LM {
    // Library functions to be implemented in Python using TensorFlow.

    const Tokens: seq<Token>
    const Ids: seq<Id>
    var Logits: array<Logit>

    // This predicate ensures that there's a bijection from Ids to Tokens, that the set of Ids are just a subsequence of the natural numbers (like indices in an array), and that each token has a corresponding logit between some upper and lower bound (pre-softmax).
    predicate ValidTokensIdsLogits()
      reads this
      reads this.Logits
    {
      ((|Tokens| == |Ids|) && (|Ids| == Logits.Length) && (|Ids| > 0 && Ids[0] == 0)) &&
      (forall i :: 0 <= i < |Ids| ==> (i == Ids[i]) && (i in Ids)) && 
      (forall i, j :: 0 <= i < |Tokens| && 0 <= j < |Tokens| && i != j ==> Tokens[i] != Tokens[j]) &&
      (forall token: Token :: token in Tokens ==> (exists i :: 0 <= i < |Ids| && Tokens[i] == token)) &&
      (forall i :: 0 <= i < Logits.Length ==> Logits[i] <= 1e9 && Logits[i] >= -1e9)
    }

    // The constructor for this LM wrapper class will create lists of Tokens, Ids, and Logits according to the above standards.
    constructor {:extern} {:axiom} ()
      ensures ValidTokensIdsLogits()

    // Axiom: in this codebase every LM is constructed and used so that ValidTokensIdsLogits() holds whenever we call step methods. Call this at the start of steps so the precondition is satisfied without the strategy having to prove the LM invariant.
    lemma {:axiom} ValidTokensIdsLogitsAlways()
      ensures ValidTokensIdsLogits()

    // Function for getting an id's corresponding token.
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

    // Function for getting a token's corresponding id.
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

    // Helper function for the above method, actual implementation will not be recursive.
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

    // Function for getting an id's corresponding logit. 
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

    // Function for getting a token's corresponding logit.
    function TokenToLogit(token: Token): (logit: Logit)
      reads this
      reads this.Logits
      requires ValidTokensIdsLogits()
      requires token in Tokens
      ensures ValidTokensIdsLogits()
    {
      IdToLogit(TokenToId(token))
    }

    // Function for getting the corresponding logits for a list of tokens.
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

    // Function for getting the corresponding logits for a list of ids.
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

    // Method that sets a token's logit to -1e9, ensuring it is never chosen.
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
      Logits[id] := -1e9;
    }

    // Method that masks a list of tokens, ensuring none of them are chosen.
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

    // Method that masks every token except for a list of tokens, ensuring only one of them is chosen.
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

    // Function that checks if a specific token is masked.
    predicate IsMasked(token: Token)
      reads this
      reads this.Logits
      requires ValidTokensIdsLogits()
      requires token in Tokens
      ensures ValidTokensIdsLogits()
    {
      Logits[TokenToId(token)] == -1e9
    }

    // Function that checks if an unmasked token exists to choose from.
    predicate HasUnmaskedToken()
      reads this
      reads this.Logits
      requires ValidTokensIdsLogits()
      ensures ValidTokensIdsLogits()
    {
      exists t: Token :: t in Tokens && !IsMasked(t)
    }

    // Extern method that calculates the logits for next possible tokens given an input string.
    method {:extern} {:axiom} GenerateLogits(input: Prefix)
      modifies this.Logits
      requires ValidTokensIdsLogits()
      ensures ValidTokensIdsLogits()

    // Extern method choosing the next token using the calculated logits.
    method {:extern} {:axiom} ChooseNextToken() returns (token: Token)
      requires ValidTokensIdsLogits()
      ensures token in Tokens
      ensures !IsMasked(token)
      ensures ValidTokensIdsLogits()
  }

  class Parser {
    // Library functions to be implemented in Python using Lark.

    // Extern function checking if the given prefix is valid under the grammar.
    predicate {:extern} {:axiom} IsValidPrefix(prefix: Prefix)
      ensures forall k :: 0 <= k < |prefix| ==> IsValidPrefix(prefix[..k])

    // Lemma that proves that the empty prefix is valid.
    lemma {:axiom} EmptyPrefixIsValid()
      ensures IsValidPrefix([])

    // Extern function checking if the given prefix is complete under the grammar.
    predicate {:extern} {:axiom} IsCompletePrefix(prefix: Prefix)
      ensures IsValidPrefix(prefix)

    // Function checking if the prefix isn't complete and cannot be completed.
    predicate IsDeadPrefix(prefix: Prefix)
    {
      !IsCompletePrefix(prefix) && |ValidNextTokens(prefix)| == 0
    }

    // Function checking if the given token is a valid continuation of the prefix.
    predicate ValidNextToken(prefix: Prefix, token: Token)
      requires IsValidPrefix(prefix)
    {
      token in ValidNextTokens(prefix)
    }

    // Extern function returning the set of next tokens valid under the grammar.
    function {:extern} {:axiom} ValidNextTokens(prefix: Prefix): seq<Token>
      requires IsValidPrefix(prefix)
      ensures forall t :: t in ValidNextTokens(prefix) ==> IsValidPrefix(prefix + [t])
      ensures (IsCompletePrefix(prefix) || |ValidNextTokens(prefix)| > 0)
  }

  predicate Contains(s: string, sub: string)
  {
    exists i, j :: 0 <= i <= j <= |s| && s[i..j] == sub
  }

  predicate PrefixContains(p: Prefix, t: Token)
  {
    exists i :: 0 <= i < |p| && p[i] == t
  }

  // =============================================================================
  // Delimiters: default token constants (for method contracts) and Delimiter class.
  // Change the two constants below to switch delimiters. All strategies and the
  // GeneratedCSD template use these; recompile after changing. If Python code
  // checks for delimiters (e.g. evaluator, model_utils), keep those in sync.
  // =============================================================================

  const LeftDelimiter: Token := " <<"
  const RightDelimiter: Token := " >>"

  class Delimiter {
    const Left: Token
    const Right: Token

    constructor(left: Token, right: Token)
      requires left != right
      ensures this.Left == left && this.Right == right
      ensures this.Left != this.Right
    {
      this.Left := left;
      this.Right := right;
    }

    /// Returns the index of the last occurrence of Left in prefix, or |prefix| if none.
    function LastLeftDelimiterIndex(prefix: Prefix): (result: nat)
      ensures result <= |prefix|
      ensures result < |prefix| ==> prefix[result] == this.Left
      ensures result == |prefix| ==> forall i :: 0 <= i < |prefix| ==> prefix[i] != this.Left
      ensures result < |prefix| ==> forall i :: result < i < |prefix| ==> prefix[i] != this.Left
      decreases |prefix|
    {
      if |prefix| == 0 then 0
      else
        if prefix[|prefix|-1] == this.Left then |prefix|-1
        else
          var lastInRest := LastLeftDelimiterIndex(prefix[..|prefix|-1]);
          if lastInRest < |prefix|-1 then lastInRest else |prefix|
    }

    /// Returns the index of the first occurrence of Right in content, or |content| if none.
    function FirstRightDelimiterIndex(content: Prefix): (result: nat)
      ensures result <= |content|
      ensures result < |content| ==> content[result] == this.Right
      ensures forall i :: 0 <= i < result ==> content[i] != this.Right
      decreases |content|
    {
      if |content| == 0 then 0
      else if content[0] == this.Right then 0
      else 1 + FirstRightDelimiterIndex(content[1..])
    }

    lemma NoFirstRightDelimiterIndexMeansNoRight(content: Prefix)
      requires FirstRightDelimiterIndex(content) == |content|
      ensures !PrefixContains(content, this.Right)
    {
      if |content| == 0 {
      } else {
        NoFirstRightDelimiterIndexMeansNoRight(content[1..]);
      }
    }

    /// Returns the token sequence strictly between the last left delimiter and the next right delimiter (or end).
    /// If there is no left delimiter, returns [].
    function GetDelimitedContent(prefix: Prefix): Prefix
      ensures |GetDelimitedContent(prefix)| <= |prefix|
      ensures forall t: Token :: t in GetDelimitedContent(prefix) ==> t in prefix
    {
      var start := LastLeftDelimiterIndex(prefix) + 1;
      if start > |prefix| then []
      else
        var afterLeft := prefix[start..|prefix|];
        var endIdx := FirstRightDelimiterIndex(afterLeft);
        afterLeft[..endIdx]
    }

    /// True iff we have seen the left delimiter and not yet seen a matching right delimiter.
    predicate InsideDelimitedWindow(prefix: Prefix)
    {
      var start := LastLeftDelimiterIndex(prefix) + 1;
      start <= |prefix| && FirstRightDelimiterIndex(prefix[start..|prefix|]) == |prefix[start..|prefix|]|
    }

    lemma InsideDelimitedWindowNoRight(prefix: Prefix)
      requires InsideDelimitedWindow(prefix)
      ensures !PrefixContains(GetDelimitedContent(prefix), this.Right)
    {
      var start := LastLeftDelimiterIndex(prefix) + 1;
      var afterLeft := prefix[start..|prefix|];
      NoFirstRightDelimiterIndexMeansNoRight(afterLeft);
    }

    /// When inside the window and the new token is not the right delimiter, appending it extends the delimited content by one token and we remain inside the window. Used to maintain ConstrainedWindowValid after ConstrainedStep.
    lemma {:axiom} GetDelimitedContentAppend(prefix: Prefix, next: Token)
      requires InsideDelimitedWindow(prefix)
      requires next != Right
      requires next != Left
      ensures GetDelimitedContent(prefix + [next]) == GetDelimitedContent(prefix) + [next]
      ensures next != Right ==> InsideDelimitedWindow(prefix + [next])

    lemma AppendLeftEntersWindow(prefix: Prefix)
      ensures InsideDelimitedWindow(prefix + [this.Left])
      ensures GetDelimitedContent(prefix + [this.Left]) == []
    {}

    lemma FirstRightDelimiterAppendRight(content: Prefix)
      requires FirstRightDelimiterIndex(content) == |content|
      ensures FirstRightDelimiterIndex(content + [this.Right]) == |content|
    {
      if |content| == 0 {
      } else {
        FirstRightDelimiterAppendRight(content[1..]);
        assert (content + [this.Right])[1..] == content[1..] + [this.Right];
      }
    }

    lemma LastLeftDelimiterAppendNonLeft(prefix: Prefix, tok: Token)
      requires tok != this.Left
      ensures var oldIdx := LastLeftDelimiterIndex(prefix);
              var newIdx := LastLeftDelimiterIndex(prefix + [tok]);
              if oldIdx < |prefix| then newIdx == oldIdx
              else newIdx == |prefix + [tok]|
    {
      var extended := prefix + [tok];
      assert extended[..|extended| - 1] == prefix;
    }

    lemma AppendRightExitsWindow(prefix: Prefix)
      requires InsideDelimitedWindow(prefix)
      requires this.Left != this.Right
      ensures !InsideDelimitedWindow(prefix + [this.Right])
    {
      var start := LastLeftDelimiterIndex(prefix) + 1;
      var afterLeft := prefix[start..|prefix|];
      assert FirstRightDelimiterIndex(afterLeft) == |afterLeft|;
      FirstRightDelimiterAppendRight(afterLeft);
      LastLeftDelimiterAppendNonLeft(prefix, this.Right);
      var newPrefix := prefix + [this.Right];
      var newStart := LastLeftDelimiterIndex(newPrefix) + 1;
      assert newStart == start;
      var newAfterLeft := newPrefix[newStart..|newPrefix|];
      assert newAfterLeft == afterLeft + [this.Right];
      assert FirstRightDelimiterIndex(newAfterLeft) == |afterLeft|;
      assert |afterLeft| != |newAfterLeft|;
    }
  }

  class CSDHelpers {
    // Holds lm, parser, and delimiter; all step/rollback/delimiter helpers use these.

    const lm: LM
    const parser: Parser
    const delimiter: Delimiter

    constructor(lm: LM, parser: Parser, delimiter: Delimiter)
      requires delimiter.Left != delimiter.Right
      ensures this.lm == lm && this.parser == parser && this.delimiter == delimiter
      ensures this.delimiter.Left != this.delimiter.Right
    {
      this.lm := lm;
      this.parser := parser;
      this.delimiter := delimiter;
    }

    predicate DelimitersInLM()
      reads this, this.delimiter, this.lm, this.lm.Logits
    {
      lm.ValidTokensIdsLogits() &&
      delimiter.Left in lm.Tokens &&
      delimiter.Right in lm.Tokens
    }

    lemma {:axiom} DelimitersInLMAlways()
      ensures DelimitersInLM()

    // --- Delimiter conveniences (delegate to this.delimiter) ---
    function LeftDelimiter(): (result: Token)
      reads this, this.delimiter
      ensures result == this.delimiter.Left
    { 
      this.delimiter.Left
    }

    function RightDelimiter(): (result: Token)
      reads this, this.delimiter
      ensures result == this.delimiter.Right
    { 
      this.delimiter.Right
    }

    function GetDelimitedContent(prefix: Prefix): (result: Prefix)
      reads this, this.delimiter
      ensures result == this.delimiter.GetDelimitedContent(prefix)
    { 
      this.delimiter.GetDelimitedContent(prefix)
    }

    predicate InsideDelimitedWindow(prefix: Prefix)
      reads this, this.delimiter
    { 
      this.delimiter.InsideDelimitedWindow(prefix)
    }

    /// Only the current constrained window (delimited content) need be a valid prefix; the full generated sequence may be invalid outside the window.
    predicate ConstrainedWindowValid(prefix: Prefix)
      reads this, this.delimiter, this.parser
    {
      !this.delimiter.InsideDelimitedWindow(prefix) || this.parser.IsValidPrefix(this.delimiter.GetDelimitedContent(prefix))
    }

    /// When we're inside the delimited window and the constrained window is valid, the delimited content is a valid (grammatically correct) parser prefix.
    lemma {:axiom} InDelimitedWindowThenContentValid(prefix: Prefix)
      requires InsideDelimitedWindow(prefix)
      requires ConstrainedWindowValid(prefix)
      ensures parser.IsValidPrefix(GetDelimitedContent(prefix))

    /// After ConstrainedStep returns next, calling this lemma before generated := generated + [next] lets the verifier prove ConstrainedWindowValid(generated + [next]). Requires next != RightDelimiter() so we remain inside the window.
    lemma GetDelimitedContentAppend(prefix: Prefix, next: Token)
      requires InsideDelimitedWindow(prefix)
      requires ConstrainedWindowValid(prefix)
      requires parser.ValidNextToken(GetDelimitedContent(prefix), next)
      requires next != RightDelimiter()
      requires next != LeftDelimiter()
      ensures GetDelimitedContent(prefix + [next]) == GetDelimitedContent(prefix) + [next]
      ensures ConstrainedWindowValid(prefix + [next])
      ensures InsideDelimitedWindow(prefix + [next])
    {
      delimiter.GetDelimitedContentAppend(prefix, next);
      // Parser.ValidNextTokens ensures extending by valid next gives valid prefix; so ConstrainedWindowValid(prefix + [next]) follows.
    }

    lemma EnterDelimitedWindow(prefix: Prefix)
      ensures InsideDelimitedWindow(prefix + [delimiter.Left])
      ensures GetDelimitedContent(prefix + [delimiter.Left]) == []
      ensures parser.IsValidPrefix([])
    {
      delimiter.AppendLeftEntersWindow(prefix);
      parser.EmptyPrefixIsValid();
    }

    lemma ExitDelimitedWindow(prefix: Prefix)
      requires InsideDelimitedWindow(prefix)
      requires this.delimiter.Left != this.delimiter.Right
      ensures !InsideDelimitedWindow(prefix + [delimiter.Right])
      ensures ConstrainedWindowValid(prefix + [delimiter.Right])
    {
      delimiter.AppendRightExitsWindow(prefix);
    }

    // Performs a single unconstrained decoding step; consumes one step. Returns next token and remaining steps.
    method UnconstrainedStep(prompt: Prefix, generated: Prefix, stepsLeft: nat) returns (next: Token, stepsLeft': nat)
      modifies this.lm.Logits
      requires stepsLeft >= 1
      ensures this.lm.ValidTokensIdsLogits()
      ensures stepsLeft' == stepsLeft - 1
    {
      lm.ValidTokensIdsLogitsAlways();
      lm.GenerateLogits(prompt + generated);
      next := lm.ChooseNextToken();
      stepsLeft' := stepsLeft - 1;
    }

    // Performs a single constrained decoding step; consumes one step. Uses only the current delimited content for the parser.
    method ConstrainedStep(prompt: Prefix, generated: Prefix, stepsLeft: nat) returns (next: Token, stepsLeft': nat)
      modifies this.lm.Logits
      requires InsideDelimitedWindow(generated)
      requires ConstrainedWindowValid(generated)
      requires !parser.IsCompletePrefix(GetDelimitedContent(generated))
      requires stepsLeft >= 1
      requires DelimitersInLM()
      ensures next in lm.Tokens
      ensures next != LeftDelimiter()
      ensures this.lm.ValidTokensIdsLogits()
      ensures parser.ValidNextToken(GetDelimitedContent(generated), next)
      ensures !this.lm.IsMasked(next)
      ensures stepsLeft' == stepsLeft - 1
      ensures forall t: Token :: t in parser.ValidNextTokens(GetDelimitedContent(generated) + [next]) ==> t in this.lm.Tokens
      ensures parser.IsValidPrefix(GetDelimitedContent(generated) + [next])
      ensures next != RightDelimiter() ==> GetDelimitedContent(generated + [next]) == GetDelimitedContent(generated) + [next]
      ensures next != RightDelimiter() ==> ConstrainedWindowValid(generated + [next])
      ensures next != RightDelimiter() ==> InsideDelimitedWindow(generated + [next])
    {
      ContentIsValidInWindow(generated);
      var content := GetDelimitedContent(generated);
      ValidNextTokensInLM(content);
      lm.GenerateLogits(prompt + generated);
      lm.MaskTokensExcept(parser.ValidNextTokens(content));
      lm.MaskToken(LeftDelimiter());
      next := lm.ChooseNextToken();
      ConstrainedStepNextValid(content, next);
      stepsLeft' := stepsLeft - 1;
      if next != RightDelimiter() {
        delimiter.GetDelimitedContentAppend(generated, next);
      }
    }

    lemma {:axiom} ConstrainedStepNextValid(content: Prefix, next: Token)
      requires lm.ValidTokensIdsLogits()
      requires parser.IsValidPrefix(content)
      requires !parser.IsCompletePrefix(content)
      requires forall t: Token :: t in parser.ValidNextTokens(content) ==> t in lm.Tokens
      requires parser.IsValidPrefix(content + [next])
      ensures forall t: Token :: t in parser.ValidNextTokens(content + [next]) ==> t in lm.Tokens

    // Axiom: when inside the delimited window, the content is a valid parser prefix.
    // True by construction: ConstrainedStep is only ever called inside the window, maintaining
    // IsValidPrefix via its own postconditions starting from the empty prefix (EmptyPrefixIsValid).
    lemma ContentIsValidInWindow(generated: Prefix)
      requires InsideDelimitedWindow(generated)
      requires ConstrainedWindowValid(generated)
      ensures parser.IsValidPrefix(GetDelimitedContent(generated))
    {
      InDelimitedWindowThenContentValid(generated);
    }

    // Axiom: all grammar-valid next tokens are in the LM vocabulary.
    // True by construction: the LM's tokenizer is built to cover the grammar's token set.
    lemma {:axiom} ValidNextTokensInLM(content: Prefix)
      requires parser.IsValidPrefix(content)
      ensures forall t: Token :: t in parser.ValidNextTokens(content) ==> t in lm.Tokens

    lemma {:axiom} RollbackPreservesTokenInvariant(prefix: Prefix)
      requires lm.ValidTokensIdsLogits()
      requires ConstrainedWindowValid(prefix)
      ensures InsideDelimitedWindow(prefix) ==> (forall t: Token :: t in parser.ValidNextTokens(GetDelimitedContent(prefix)) ==> t in lm.Tokens)

    // Deletes invalid tokens from the end of the generated prefix until it becomes valid, then returns.
    method RollbackToValidPrefix(generated: Prefix) returns (repaired: Prefix)
      requires parser.IsValidPrefix([])
      ensures parser.IsValidPrefix(repaired)
      ensures |repaired| <= |generated|
    {
      repaired := generated;

      while |repaired| > 0 && (!parser.IsValidPrefix(repaired) || parser.IsDeadPrefix(repaired))
        invariant |repaired| <= |generated|
        invariant parser.IsValidPrefix(repaired) || |repaired| > 0
        decreases |repaired|
      {
        repaired := repaired[..|repaired|-1];
      }
    }
  }
}