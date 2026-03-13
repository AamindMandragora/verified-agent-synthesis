include "VerifiedAgentSynthesis.dfy"

module GeneratedCSD {
  import opened VerifiedDecoderAgent

  method MyCSDStrategy(lm: LM, parser: Parser, prompt: Prefix, maxSteps: nat, eosToken: Token) returns (generated: Prefix, remainingSteps: nat)
    modifies lm.Logits
    requires lm.ValidTokensIdsLogits()
    requires parser.IsValidPrefix([])
    requires forall t: Token :: t in parser.ValidNextTokens([]) ==> t in lm.Tokens
    requires "<<" in lm.Tokens && ">>" in lm.Tokens
    ensures lm.ValidTokensIdsLogits()
    ensures |generated| <= maxSteps
    ensures remainingSteps >= 0 && remainingSteps <= maxSteps
  {
    var helpers := new CSDHelpers();
    var stepsLeft := maxSteps;
    // CSD_RATIONALE_BEGIN
// This strategy generates first-order logic (FOL) formulas using the given parser
// and logical model. It uses constrained decoding to ensure that the generated
// expressions are valid FOL statements according to the specified rules.
// The parser enforces strict FOL grammar with quantifiers, predicates, and logical
// connectives.
// Critical rules:
// 1. Use {forall} and {exists} for quantifiers, followed by a single lowercase variable and a formula.
// 2. Predicates start with uppercase (e.g., Dog(x), Likes(john, mary)) with lowercase arguments.
// 3. Single lowercase letters (x, y, z) are variables; multi-character lowercase identifiers are constants.
// 4. Logical connectives: {and}, {or}, {not}, {implies}, {iff}, {xor}.
// 5. The constrained windows contain complete FOL statements — one per premise and one for the conclusion.
// 6. Parentheses are used for grouping; operator precedence is: iff < implies < xor < or < and < not.
// The strategy starts with an empty generated sequence and iteratively appends tokens
// to it until a complete FOL statement is formed. It uses ConstrainedStep to ensure
// that the generated sequence remains a valid prefix under the grammar.
// CSD_RATIONALE_END

generated := [];
while stepsLeft > 0 && !parser.IsCompletePrefix(generated)
  invariant lm.ValidTokensIdsLogits()
  invariant parser.IsValidPrefix(generated)
  invariant 0 <= stepsLeft <= maxSteps
  invariant |generated| + stepsLeft == maxSteps
  decreases stepsLeft
{
  CSDHelpers.RollbackPreservesTokenInvariant(lm, parser, generated);
  var next, newSteps := helpers.ConstrainedStep(lm, parser, prompt, generated, stepsLeft);
  generated := generated + [next];
  stepsLeft := newSteps;
}
    remainingSteps := stepsLeft;
  }
}
