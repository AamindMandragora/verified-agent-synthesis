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
      requires !parser.IsCompletePrefix(generated)
      requires forall t: Token :: t in parser.ValidNextTokens(generated) ==> t in lm.Tokens
      requires parser.IsValidPrefix(generated + [next])
      ensures forall t: Token :: t in parser.ValidNextTokens(generated + [next]) ==> t in lm.Tokens

    // Performs constrained decoding until we run out of steps or the generated string is complete in the grammar.
    method ConstrainedGeneration(lm: LM, parser: Parser, prompt: Prefix, maxSteps: nat) returns (generated: Prefix)
      modifies lm.Logits, this
      requires lm.ValidTokensIdsLogits()
      requires parser.IsValidPrefix([])
      requires forall t: Token :: t in parser.ValidNextTokens([]) ==> t in lm.Tokens
      ensures lm.ValidTokensIdsLogits()
      ensures |generated| <= maxSteps
      ensures parser.IsValidPrefix(generated)
      ensures |generated| == maxSteps || parser.IsCompletePrefix(generated)
      ensures cost == old(cost) + |generated|
    {
      generated := [];
      var steps := 0;
      while steps < maxSteps && !parser.IsCompletePrefix(generated)
        invariant 0 <= steps <= maxSteps
        invariant lm.ValidTokensIdsLogits()
        invariant steps == |generated|
        invariant parser.IsValidPrefix(generated)
        invariant forall t: Token :: t in parser.ValidNextTokens(generated) ==> t in lm.Tokens
        invariant cost == old(cost) + steps
        decreases maxSteps - steps
      {
        var next := ConstrainedStep(lm, parser, prompt, generated);
        generated := generated + [next];
        steps := steps + 1;
      }
    }

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

    // =========================================================================
    // COMPOSABLE STRATEGY HELPERS
    // These can be combined to create more interesting strategies.
    // =========================================================================

    // Strategy 1: Try unconstrained first, then fall back to constrained.
    // Generates unconstrainedSteps tokens freely, validates, and if invalid
    // or incomplete, switches to constrained for remaining steps.
    method TryUnconstrainedThenConstrained(
      lm: LM, 
      parser: Parser, 
      prompt: Prefix, 
      maxSteps: nat,
      unconstrainedSteps: nat
    ) returns (generated: Prefix)
      modifies lm.Logits, this
      requires lm.ValidTokensIdsLogits()
      requires parser.IsValidPrefix([])
      requires forall t: Token :: t in parser.ValidNextTokens([]) ==> t in lm.Tokens
      requires unconstrainedSteps <= maxSteps
      ensures lm.ValidTokensIdsLogits()
      ensures |generated| <= maxSteps
      ensures parser.IsValidPrefix(generated)
      ensures cost >= old(cost) + |generated|
    {
      // Phase 1: Unconstrained generation
      var unconstrained := UnconstrainedGeneration(lm, prompt, maxSteps);
      assert cost == old(cost) + maxSteps;
      
      // Phase 2: Rollback to valid prefix
      var validPrefix := RollbackToValidPrefix(parser, unconstrained);
      
      // Phase 3: If complete, we're done
      if parser.IsCompletePrefix(validPrefix) {
        generated := validPrefix;
      } else {
        // Phase 4: Complete with constrained decoding
        var remainingSteps := maxSteps - |validPrefix|;
        generated := validPrefix;
        var steps := |validPrefix|;
        
        // Need to establish the invariant for valid next tokens
        RollbackPreservesTokenInvariant(lm, parser, validPrefix);
        
        while steps < maxSteps && !parser.IsCompletePrefix(generated)
          invariant |validPrefix| <= steps <= maxSteps
          invariant lm.ValidTokensIdsLogits()
          invariant steps == |generated|
          invariant parser.IsValidPrefix(generated)
          invariant forall t: Token :: t in parser.ValidNextTokens(generated) ==> t in lm.Tokens
          invariant cost >= old(cost) + steps
          decreases maxSteps - steps
        {
          var next := ConstrainedStep(lm, parser, prompt, generated);
          generated := generated + [next];
          steps := steps + 1;
        }
      }
    }

    // Lemma: After rollback, valid next tokens are still in LM vocabulary
    static lemma {:axiom} RollbackPreservesTokenInvariant(lm: LM, parser: Parser, prefix: Prefix)
      requires lm.ValidTokensIdsLogits()
      requires parser.IsValidPrefix(prefix)
      ensures forall t: Token :: t in parser.ValidNextTokens(prefix) ==> t in lm.Tokens

    // Helper: Complete an existing VALID prefix using constrained steps.
    // This is useful for multi-stage strategies that first construct a partial valid prefix
    // (e.g., via rollback/validation/speculation) and then want a simple verified completion step.
    method CompletePrefix(
      lm: LM,
      parser: Parser,
      prompt: Prefix,
      partial: Prefix,
      maxSteps: nat
    ) returns (generated: Prefix)
      modifies lm.Logits, this
      requires lm.ValidTokensIdsLogits()
      requires parser.IsValidPrefix(partial)
      requires |partial| <= maxSteps
      ensures lm.ValidTokensIdsLogits()
      ensures |generated| <= maxSteps
      ensures |generated| >= |partial|
      ensures generated[..|partial|] == partial
      ensures parser.IsValidPrefix(generated)
      ensures |generated| == maxSteps || parser.IsCompletePrefix(generated)
      ensures cost == old(cost) + (|generated| - |partial|)
    {
      generated := partial;
      var steps := |partial|;

      // Establish the "valid next tokens are in LM vocabulary" invariant for the starting prefix.
      RollbackPreservesTokenInvariant(lm, parser, partial);

      while steps < maxSteps && !parser.IsCompletePrefix(generated)
        invariant |partial| <= steps <= maxSteps
        invariant lm.ValidTokensIdsLogits()
        invariant steps == |generated|
        invariant parser.IsValidPrefix(generated)
        invariant generated[..|partial|] == partial
        invariant forall t: Token :: t in parser.ValidNextTokens(generated) ==> t in lm.Tokens
        invariant cost == old(cost) + (steps - |partial|)
        decreases maxSteps - steps
      {
        var next := ConstrainedStep(lm, parser, prompt, generated);
        generated := generated + [next];
        steps := steps + 1;
      }
    }

    // Strategy 2: Unconstrained with rollback and completion.
    // Generates fully unconstrained, rolls back to valid, completes with constrained.
    method UnconstrainedWithCompletion(
      lm: LM,
      parser: Parser,
      prompt: Prefix,
      maxSteps: nat
    ) returns (generated: Prefix)
      modifies lm.Logits, this
      requires lm.ValidTokensIdsLogits()
      requires parser.IsValidPrefix([])
      requires forall t: Token :: t in parser.ValidNextTokens([]) ==> t in lm.Tokens
      ensures lm.ValidTokensIdsLogits()
      ensures |generated| <= maxSteps
      ensures parser.IsValidPrefix(generated)
      ensures |generated| == maxSteps || parser.IsCompletePrefix(generated)
      ensures cost >= old(cost) + maxSteps
    {
      // Generate unconstrained
      var unconstrained := UnconstrainedGeneration(lm, prompt, maxSteps);
      
      // Rollback to valid
      var validPrefix := RollbackToValidPrefix(parser, unconstrained);
      
      // If already complete, done
      if parser.IsCompletePrefix(validPrefix) {
        generated := validPrefix;
      } else {
        // Complete with constrained
        var remainingSteps := maxSteps - |validPrefix|;
        generated := validPrefix;
        var steps := |validPrefix|;
        
        RollbackPreservesTokenInvariant(lm, parser, validPrefix);
        
        while steps < maxSteps && !parser.IsCompletePrefix(generated)
          invariant |validPrefix| <= steps <= maxSteps
          invariant lm.ValidTokensIdsLogits()
          invariant steps == |generated|
          invariant parser.IsValidPrefix(generated)
          invariant forall t: Token :: t in parser.ValidNextTokens(generated) ==> t in lm.Tokens
          invariant cost >= old(cost) + maxSteps
          decreases maxSteps - steps
        {
          var next := ConstrainedStep(lm, parser, prompt, generated);
          generated := generated + [next];
          steps := steps + 1;
        }
      }
    }

    // Strategy 5: Hybrid generation - switch between unconstrained and constrained using << >>.
    // This is the implementation requested by the user to handle CRANE-style windowing.
    method HybridGeneration(
      lm: LM,
      parser: Parser,
      prompt: Prefix,
      maxSteps: nat
    ) returns (generated: Prefix)
      modifies lm.Logits, this
      requires lm.ValidTokensIdsLogits()
      requires parser.IsValidPrefix([])
      requires forall t: Token :: t in parser.ValidNextTokens([]) ==> t in lm.Tokens
      requires "<<" in lm.Tokens && ">>" in lm.Tokens
      ensures lm.ValidTokensIdsLogits()
      ensures |generated| <= maxSteps
      ensures parser.IsValidPrefix(generated)
      ensures |generated| == maxSteps || parser.IsCompletePrefix(generated)
      ensures old(cost) + |generated| <= cost <= old(cost) + 2 * maxSteps
    {
      generated := [];
      var totalSteps := 0;
      var insideHybrid := false;
      
      while totalSteps < maxSteps && !parser.IsCompletePrefix(generated) && |generated| < maxSteps
        invariant 0 <= totalSteps <= maxSteps
        invariant |generated| <= totalSteps
        invariant |generated| <= maxSteps
        invariant lm.ValidTokensIdsLogits()
        invariant !insideHybrid ==> parser.IsValidPrefix(generated)
        invariant !insideHybrid ==> forall t: Token :: t in parser.ValidNextTokens(generated) ==> t in lm.Tokens
        invariant old(cost) + |generated| <= cost <= old(cost) + totalSteps
        decreases maxSteps - totalSteps
      {
        if !insideHybrid {
          // Constrained mode
          var next := ConstrainedStep(lm, parser, prompt, generated);
          generated := generated + [next];
          totalSteps := totalSteps + 1;
          
          if next == "<<" {
            insideHybrid := true;
          }
        } else {
          // Unconstrained hybrid mode
          var next := UnconstrainedStep(lm, prompt, generated);
          generated := generated + [next];
          totalSteps := totalSteps + 1;
          
          if next == ">>" {
            // Check if adding ">>" preserved validity or if we need to roll back
            if parser.IsValidPrefix(generated) {
              insideHybrid := false;
              RollbackPreservesTokenInvariant(lm, parser, generated);
            } else {
              // Roll back the entire hybrid section if it's invalid
              generated := RollbackToValidPrefix(parser, generated);
              insideHybrid := false;
              // Restore token invariant
              RollbackPreservesTokenInvariant(lm, parser, generated);
            }
          }
        }
      }
      
      // If we ended while still in a hybrid block, roll back to last valid
      if insideHybrid {
        generated := RollbackToValidPrefix(parser, generated);
        insideHybrid := false;
        RollbackPreservesTokenInvariant(lm, parser, generated);
      }

      var stepsBeforeFinal := totalSteps;
      var lengthBeforeFinal := |generated|;

      // Final constrained phase to ensure the contract (complete or maxSteps) is met
      while |generated| < maxSteps && !parser.IsCompletePrefix(generated)
        invariant |generated| <= maxSteps
        invariant stepsBeforeFinal <= totalSteps <= stepsBeforeFinal + (|generated| - lengthBeforeFinal)
        invariant totalSteps <= 2 * maxSteps
        invariant lm.ValidTokensIdsLogits()
        invariant parser.IsValidPrefix(generated)
        invariant forall t: Token :: t in parser.ValidNextTokens(generated) ==> t in lm.Tokens
        invariant old(cost) + |generated| <= cost <= old(cost) + totalSteps
        decreases maxSteps - |generated|
      {
        var next := ConstrainedStep(lm, parser, prompt, generated);
        generated := generated + [next];
        totalSteps := totalSteps + 1;
      }
    }

    // Strategy 4: Speculative decoding - generate K tokens unconstrained,
    // validate, keep valid prefix, repeat.
    method SpeculativeGeneration(
      lm: LM,
      parser: Parser,
      prompt: Prefix,
      maxSteps: nat,
      speculativeWindow: nat  // How many tokens to speculate at once
    ) returns (generated: Prefix)
      modifies lm.Logits, this
      requires lm.ValidTokensIdsLogits()
      requires parser.IsValidPrefix([])
      requires forall t: Token :: t in parser.ValidNextTokens([]) ==> t in lm.Tokens
      requires speculativeWindow > 0
      ensures lm.ValidTokensIdsLogits()
      ensures |generated| <= maxSteps
      ensures parser.IsValidPrefix(generated)
      ensures |generated| == maxSteps || parser.IsCompletePrefix(generated)
      ensures cost >= old(cost)
    {
      generated := [];
      var steps := 0;
      
      while steps < maxSteps && !parser.IsCompletePrefix(generated)
        invariant 0 <= steps <= maxSteps
        invariant lm.ValidTokensIdsLogits()
        invariant steps == |generated|
        invariant parser.IsValidPrefix(generated)
        invariant forall t: Token :: t in parser.ValidNextTokens(generated) ==> t in lm.Tokens
        invariant cost >= old(cost)
        decreases maxSteps - steps
      {
        // Speculate: generate up to speculativeWindow tokens unconstrained
        var speculateSteps := if steps + speculativeWindow <= maxSteps 
                              then speculativeWindow 
                              else maxSteps - steps;
        var speculated := UnconstrainedGeneration(lm, prompt + generated, speculateSteps);
        
        // Validate each token one by one, keep valid prefix
        var validCount := 0;
        var tempPrefix := generated;
        
        while validCount < |speculated| && parser.IsValidPrefix(tempPrefix + [speculated[validCount]])
          invariant 0 <= validCount <= |speculated|
          invariant parser.IsValidPrefix(tempPrefix)
          invariant tempPrefix == generated + speculated[..validCount]
          decreases |speculated| - validCount
        {
          tempPrefix := tempPrefix + [speculated[validCount]];
          validCount := validCount + 1;
        }
        
        // If we accepted at least one speculated token
        if validCount > 0 {
          generated := tempPrefix;
          steps := steps + validCount;
          RollbackPreservesTokenInvariant(lm, parser, generated);
        } else {
          // No speculated tokens valid, fall back to single constrained step
          var next := ConstrainedStep(lm, parser, prompt, generated);
          generated := generated + [next];
          steps := steps + 1;
        }
      }
    }

    // Strategy 5: Pure constrained (alias for clarity in prompts)
    method PureConstrainedGeneration(
      lm: LM,
      parser: Parser,
      prompt: Prefix,
      maxSteps: nat
    ) returns (generated: Prefix)
      modifies lm.Logits, this
      requires lm.ValidTokensIdsLogits()
      requires parser.IsValidPrefix([])
      requires forall t: Token :: t in parser.ValidNextTokens([]) ==> t in lm.Tokens
      ensures lm.ValidTokensIdsLogits()
      ensures |generated| <= maxSteps
      ensures parser.IsValidPrefix(generated)
      ensures |generated| == maxSteps || parser.IsCompletePrefix(generated)
      ensures cost == old(cost) + |generated|
    {
      generated := ConstrainedGeneration(lm, parser, prompt, maxSteps);
    }

    // =========================================================================
    // NEW: COMPACTNESS-AWARE STRATEGIES FOR GSM-SYMBOLIC
    // =========================================================================

    // Strategy 6: Generate with reasonable length constraint
    // Stops early if expression is complete AND within reasonable length
    method GenerateWithReasonableLength(
      lm: LM,
      parser: Parser,
      prompt: Prefix,
      maxSteps: nat,
      reasonableLength: nat
    ) returns (generated: Prefix)
      modifies lm.Logits, this
      requires lm.ValidTokensIdsLogits()
      requires parser.IsValidPrefix([])
      requires forall t: Token :: t in parser.ValidNextTokens([]) ==> t in lm.Tokens
      requires reasonableLength > 0
      ensures lm.ValidTokensIdsLogits()
      ensures |generated| <= maxSteps
      ensures parser.IsValidPrefix(generated)
      ensures |generated| == maxSteps || parser.IsCompletePrefix(generated)
      // If complete and within reasonable length, we stopped early
      ensures parser.IsCompletePrefix(generated) && |generated| <= reasonableLength ==>
        |generated| <= reasonableLength
      ensures cost == old(cost) + |generated|
    {
      generated := [];
      var steps := 0;
      
      while steps < maxSteps && !parser.IsCompletePrefix(generated)
        invariant 0 <= steps <= maxSteps
        invariant lm.ValidTokensIdsLogits()
        invariant steps == |generated|
        invariant parser.IsValidPrefix(generated)
        invariant forall t: Token :: t in parser.ValidNextTokens(generated) ==> t in lm.Tokens
        invariant cost == old(cost) + steps
        decreases maxSteps - steps
      {
        var next := ConstrainedStep(lm, parser, prompt, generated);
        generated := generated + [next];
        steps := steps + 1;
        
        // Stop early if complete and within reasonable length
        if parser.IsCompletePrefix(generated) && |generated| <= reasonableLength {
          break;
        }
      }
    }

    // Strategy 7: Generate until first complete (explicit early stopping)
    // This is similar to ConstrainedGeneration but makes the early stop explicit
    method GenerateUntilFirstComplete(
      lm: LM,
      parser: Parser,
      prompt: Prefix,
      maxSteps: nat
    ) returns (generated: Prefix)
      modifies lm.Logits, this
      requires lm.ValidTokensIdsLogits()
      requires parser.IsValidPrefix([])
      requires forall t: Token :: t in parser.ValidNextTokens([]) ==> t in lm.Tokens
      ensures lm.ValidTokensIdsLogits()
      ensures |generated| <= maxSteps
      ensures parser.IsValidPrefix(generated)
      ensures |generated| == maxSteps || parser.IsCompletePrefix(generated)
      ensures cost == old(cost) + |generated|
    {
      // This is essentially the same as ConstrainedGeneration - it already stops at first complete
      generated := ConstrainedGeneration(lm, parser, prompt, maxSteps);
    }

    // Helper: Select best candidate from a sequence based on a scoring function
    // Note: This is a method that doesn't modify LM state (pure computation)
    static method SelectBestCandidate(
      candidates: seq<Prefix>,
      parser: Parser,
      preferShorter: bool
    ) returns (best: Prefix)
      requires |candidates| > 0
      requires forall c: Prefix :: c in candidates ==> parser.IsValidPrefix(c)
      ensures best in candidates
    {
      var result := candidates[0];
      var bestScore := if preferShorter && parser.IsCompletePrefix(result) then -|result| else -1000;
      
      var i := 1;
      while i < |candidates|
        invariant 0 <= i <= |candidates|
        invariant result in candidates
        decreases |candidates| - i
      {
        var candidate := candidates[i];
        var score := if preferShorter && parser.IsCompletePrefix(candidate) then -|candidate| else -1000;
        
        if score > bestScore || (score == bestScore && preferShorter && parser.IsCompletePrefix(candidate) && |candidate| < |result|) {
          result := candidate;
          bestScore := score;
        }
        i := i + 1;
      }
      best := result;
    }

    // Strategy 8: Generate multiple candidates and select the best one
    // Note: This generates candidates sequentially, which may not be ideal for diversity
    // but is necessary given the LM state modification constraints
    method GenerateAndSelectBest(
      lm: LM,
      parser: Parser,
      prompt: Prefix,
      maxSteps: nat,
      numCandidates: nat,
      preferShorter: bool
    ) returns (generated: Prefix)
      modifies lm.Logits, this
      requires lm.ValidTokensIdsLogits()
      requires parser.IsValidPrefix([])
      requires forall t: Token :: t in parser.ValidNextTokens([]) ==> t in lm.Tokens
      requires numCandidates > 0
      ensures lm.ValidTokensIdsLogits()
      ensures |generated| <= maxSteps
      ensures parser.IsValidPrefix(generated)
      ensures |generated| == maxSteps || parser.IsCompletePrefix(generated)
      ensures cost >= old(cost)
    {
      // Generate first candidate
      var firstCandidate: Prefix;
      firstCandidate := ConstrainedGeneration(lm, parser, prompt, maxSteps);
      var candidates: seq<Prefix> := [firstCandidate];
      
      // Generate additional candidates (they may be similar due to deterministic generation)
      // In practice, this would use temperature sampling, but that requires LM interface changes
      var i := 1;
      while i < numCandidates
        invariant 1 <= i <= numCandidates
        invariant |candidates| == i
        invariant lm.ValidTokensIdsLogits()
        invariant forall c: Prefix :: c in candidates ==>
          (|c| <= maxSteps && parser.IsValidPrefix(c) && 
           (parser.IsCompletePrefix(c) || |c| == maxSteps))
        invariant cost >= old(cost)
        decreases numCandidates - i
      {
        // Generate another candidate
        var candidate: Prefix;
        candidate := ConstrainedGeneration(lm, parser, prompt, maxSteps);
        candidates := candidates + [candidate];
        i := i + 1;
      }
      
      // Select best candidate
      var best: Prefix;
      best := SelectBestCandidate(candidates, parser, preferShorter);
      generated := best;
    }

    // Strategy 9: Generate reasoning inside delimiters and then extract it.
    // Useful for Chain-of-Thought reasoning where the model uses <<reasoning>> format.
    method GenerateReasoningAndAnswer(
      lm: LM,
      parser: Parser,
      prompt: Prefix,
      maxSteps: nat
    ) returns (generated: Prefix, reasoning: string)
      modifies lm.Logits, this
      requires lm.ValidTokensIdsLogits()
      requires parser.IsValidPrefix([])
      requires forall t: Token :: t in parser.ValidNextTokens([]) ==> t in lm.Tokens
      requires "<<" in lm.Tokens && ">>" in lm.Tokens
      ensures lm.ValidTokensIdsLogits()
      ensures parser.IsValidPrefix(generated)
       ensures reasoning != "" ==> exists pre, post :: PrefixToString(generated) == pre + "<<" + reasoning + ">>" + post
       ensures cost >= old(cost) + |generated|
     {
      // Use hybrid generation to allow unconstrained reasoning inside brackets
      generated := HybridGeneration(lm, parser, prompt, maxSteps); 
      
      // Extract the reasoning part
      reasoning := ExtractContentBetweenDelimiters(PrefixToString(generated), "<<", ">>");
    }

    // Helper: Convert Prefix (seq<Token>) to a single string
    static function PrefixToString(p: Prefix): string
    {
      if |p| == 0 then ""
      else p[0] + PrefixToString(p[1..])
    }

    // New library function for extracting content between delimiters
    // Defined as a function to allow reasoning in specifications
    static function ExtractContentBetweenDelimiters(input: string, startDelim: string, endDelim: string): (content: string)
      ensures content != "" ==> exists pre, post :: input == pre + startDelim + content + endDelim + post
    {
      ExtractContentExtern(input, startDelim, endDelim)
    }

    static function {:extern} {:axiom} ExtractContentExtern(input: string, startDelim: string, endDelim: string): (content: string)
      ensures content != "" ==> exists pre, post :: input == pre + startDelim + content + endDelim + post
    // Strategy 7: CRANE-style generation (Reasoning-Math-Reasoning).
    // Starts unconstrained. When "<<" is seen, switches to constrained.
    // When ">>" is seen, switches back to unconstrained.
    method CraneGeneration(
      lm: LM,
      parser: Parser,
      prompt: Prefix,
      maxSteps: nat,
      minReasoningSteps: nat,
      eosToken: Token
    ) returns (generated: Prefix)
      modifies lm.Logits, this
      requires lm.ValidTokensIdsLogits()
      requires "<<" in lm.Tokens && ">>" in lm.Tokens
      requires parser.IsValidPrefix([])
      requires forall t: Token :: t in parser.ValidNextTokens([]) ==> t in lm.Tokens
      ensures lm.ValidTokensIdsLogits()
      ensures |generated| <= maxSteps
      ensures cost >= old(cost) + |generated|
    {
      generated := [];
      var steps := 0;
      var insideConstrained := false;
      var currentConstrained: Prefix := [];

      while steps < maxSteps
        invariant 0 <= steps <= maxSteps
        invariant steps == |generated|
        invariant |currentConstrained| <= |generated|
        invariant lm.ValidTokensIdsLogits()
        invariant insideConstrained ==> parser.IsValidPrefix(currentConstrained)
        invariant insideConstrained ==> forall t: Token :: t in parser.ValidNextTokens(currentConstrained) ==> t in lm.Tokens
        invariant cost >= old(cost) + steps
        decreases maxSteps - steps, (if insideConstrained then 1 else 0)
      {
        if !insideConstrained {
          // If we haven't reached minReasoningSteps, mask "<<"
          if steps < minReasoningSteps {
            lm.GenerateLogits(prompt + generated);
            var next := lm.ChooseNextTokenUnconstrained();
            if next == eosToken {
              break;
            }
            if Contains(next, "<<") {
              lm.MaskToken("<<");
              next := lm.ChooseNextToken();
            }
            generated := generated + [next];
            steps := steps + 1;
            cost := cost + 1;
          } else {
            var next := UnconstrainedStep(lm, prompt, generated);
            if next == eosToken {
              break;
            }
            generated := generated + [next];
            steps := steps + 1;
            if Contains(next, "<<") {
              insideConstrained := true;
              currentConstrained := [];
              RollbackPreservesTokenInvariant(lm, parser, []);
            }
          }
        } else {
          if parser.IsCompletePrefix(currentConstrained) {
            insideConstrained := false;
            // No steps added here, just state change
          } else {
            var next := ConstrainedStep(lm, parser, prompt + generated[..|generated|-|currentConstrained|], currentConstrained);
            generated := generated + [next];
            currentConstrained := currentConstrained + [next];
            steps := steps + 1;
            
            if Contains(next, ">>") {
              insideConstrained := false;
            }
          }
        }
      }
    }
  }
}