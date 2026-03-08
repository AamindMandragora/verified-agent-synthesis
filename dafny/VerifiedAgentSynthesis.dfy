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

    // Extern method choosing the next token from the FULL vocabulary.
    method {:extern} {:axiom} ChooseNextTokenUnconstrained() returns (token: Token)
      ensures ValidTokensIdsLogits()
  }

  class Parser {
    // Library functions to be implemented in Python using Lark.

    // Extern function checking if the given prefix is valid under the grammar.
    predicate {:extern} {:axiom} IsValidPrefix(prefix: Prefix)
      ensures forall k: nat :: 0 <= k < |prefix| - 1 ==> IsValidPrefix(prefix[k..])

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

  function Contains(s: string, sub: string): bool
  {
    exists i, j :: 0 <= i <= j <= |s| && s[i..j] == sub
  }

  class CSDHelpers {
    // Library functions that QWEN must directly use to synthesize the constrained decoding agent.

    var cost: int

    constructor()
      ensures cost == 0
    {
      cost := 0;
    }

    // Performs a single unconstrained decoding step and returns the next token.
    method UnconstrainedStep(lm: LM, prompt: Prefix, generated: Prefix) returns (next: Token)
      modifies lm.Logits, this
      requires lm.ValidTokensIdsLogits()
      ensures lm.ValidTokensIdsLogits()
      ensures cost == old(cost) + 1
    {
      lm.GenerateLogits(prompt + generated);
      next := lm.ChooseNextTokenUnconstrained();
      cost := cost + 1;
    }

    // Performs a single constrained decoding step and returns the next token.
    method ConstrainedStep(lm: LM, parser: Parser, prompt: Prefix, generated: Prefix) returns (next: Token)
      modifies lm.Logits, this
      requires lm.ValidTokensIdsLogits()
      requires parser.IsValidPrefix(generated)
      requires !parser.IsCompletePrefix(generated)
      requires forall t: Token :: t in parser.ValidNextTokens(generated) ==> t in lm.Tokens
      ensures lm.ValidTokensIdsLogits()
      ensures forall t: Token :: t in lm.Tokens ==> (lm.IsMasked(t) || parser.ValidNextToken(generated, t))
      ensures parser.ValidNextToken(generated, next)
      ensures !lm.IsMasked(next)
      ensures forall t: Token :: t in parser.ValidNextTokens(generated + [next]) ==> t in lm.Tokens
      ensures cost == old(cost) + 1
    {
      lm.GenerateLogits(prompt + generated);
      lm.MaskTokensExcept(parser.ValidNextTokens(generated));
      next := lm.ChooseNextToken();
      ConstrainedStepNextValid(lm, parser, generated, next);
      cost := cost + 1;
    }

    // A lemma that lets us say if the LM can generate all next valid tokens, then if we append one of those to the end, the LM can still generate all next valid tokens for the new prefix.
    static lemma {:axiom} ConstrainedStepNextValid(lm: LM, parser: Parser, generated: Prefix, next: Token)
      requires lm.ValidTokensIdsLogits()
      requires parser.IsValidPrefix(generated)
      requires !parser.IsCompletePrefix(generated)
      requires forall t: Token :: t in parser.ValidNextTokens(generated) ==> t in lm.Tokens
      requires parser.IsValidPrefix(generated + [next])
      ensures forall t: Token :: t in parser.ValidNextTokens(generated + [next]) ==> t in lm.Tokens

    // Deletes invalid tokens from the end of the generated prefix until it becomes valid, then returns.
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

    // Lemma: After rollback, valid next tokens are still in LM vocabulary
    static lemma {:axiom} RollbackPreservesTokenInvariant(lm: LM, parser: Parser, prefix: Prefix)
      requires lm.ValidTokensIdsLogits()
      requires parser.IsValidPrefix(prefix)
      ensures forall t: Token :: t in parser.ValidNextTokens(prefix) ==> t in lm.Tokens
  }
}