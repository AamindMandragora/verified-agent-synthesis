"""
Python implementations of Dafny {:extern} functions.

These implementations provide the runtime behavior for the {:extern} functions
defined in VerifiedAgentSynthesis.dfy. They are used when executing compiled
Dafny strategies in Python.

Note: These are simplified implementations suitable for testing and demonstration.
Production use would require integration with actual LLM APIs and grammar parsers.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence
import random


# =============================================================================
# Type Aliases (matching Dafny types)
# =============================================================================

Token = str
Prefix = list[Token]
Id = int
Logit = float


# =============================================================================
# Language Model (LM) Class
# =============================================================================

class LM:
    """
    Language Model stub for constrained decoding.
    
    This is a simplified implementation that can be extended to wrap
    actual LLM APIs (HuggingFace, OpenAI, etc.).
    """
    
    def __init__(
        self,
        tokens: Optional[list[Token]] = None,
        vocab_size: int = 1000
    ):
        """
        Initialize the LM.
        
        Args:
            tokens: List of tokens in vocabulary. If None, generates dummy vocab.
            vocab_size: Size of vocabulary if generating dummy tokens.
        """
        if tokens is None:
            # Generate a basic vocabulary
            self.Tokens = self._generate_default_vocab(vocab_size)
        else:
            self.Tokens = tokens
        
        self.Ids = list(range(len(self.Tokens)))
        self.Logits = [0.0] * len(self.Tokens)
        
        # Token to ID mapping for fast lookup
        self._token_to_id = {t: i for i, t in enumerate(self.Tokens)}
    
    def _generate_default_vocab(self, size: int) -> list[Token]:
        """Generate a default vocabulary for testing."""
        vocab = []
        
        # Add some common tokens
        common = [
            "<EOS>", "<PAD>", " ", "\n", "\t",
            "(", ")", "[", "]", "{", "}",
            "+", "-", "*", "/", "=", "<", ">",
            "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
            "<<", ">>",  # CRANE delimiters
        ]
        vocab.extend(common)
        
        # Add alphabet
        for c in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ":
            vocab.append(c)
        
        # Fill remaining with numbered tokens
        while len(vocab) < size:
            vocab.append(f"<T{len(vocab)}>")
        
        return vocab[:size]
    
    def ValidTokensIdsLogits(self) -> bool:
        """Check validity invariant."""
        return (
            len(self.Tokens) == len(self.Ids) == len(self.Logits) and
            len(self.Ids) > 0 and
            self.Ids[0] == 0
        )
    
    def IdToToken(self, id: Id) -> Token:
        """Convert token ID to token string."""
        return self.Tokens[id]
    
    def TokenToId(self, token: Token) -> Id:
        """Convert token string to ID."""
        return self._token_to_id[token]
    
    def IdToLogit(self, id: Id) -> Logit:
        """Get logit for a token ID."""
        return self.Logits[id]
    
    def TokenToLogit(self, token: Token) -> Logit:
        """Get logit for a token."""
        return self.Logits[self.TokenToId(token)]
    
    def TokensToLogits(self, tokens: list[Token]) -> list[Logit]:
        """Get logits for multiple tokens."""
        return [self.TokenToLogit(t) for t in tokens]
    
    def IdsToLogits(self, ids: list[Id]) -> list[Logit]:
        """Get logits for multiple IDs."""
        return [self.IdToLogit(i) for i in ids]
    
    def MaskToken(self, token: Token) -> None:
        """Mask a token by setting its logit to 0."""
        id = self.TokenToId(token)
        self.Logits[id] = 0.0
    
    def MaskTokens(self, tokens: list[Token]) -> None:
        """Mask multiple tokens."""
        for token in tokens:
            self.MaskToken(token)
    
    def GenerateLogits(self, input: Prefix) -> None:
        """
        Generate logits for next token prediction.
        
        This is a stub - in production, this would call the actual LLM.
        For testing, generates random logits.
        """
        # Simple stub: generate random logits
        self.Logits = [random.gauss(0, 1) for _ in self.Tokens]
    
    def ChooseNextToken(self, input: Prefix) -> Token:
        """
        Choose the next token based on current logits.
        
        Uses greedy decoding (argmax).
        """
        self.GenerateLogits(input)
        max_idx = max(range(len(self.Logits)), key=lambda i: self.Logits[i])
        return self.Tokens[max_idx]


# =============================================================================
# Parser Class
# =============================================================================

class Parser:
    """
    Grammar parser for validating token sequences.
    
    This is a stub implementation. In production, would integrate with
    Lark or another parser library.
    """
    
    def __init__(
        self,
        grammar: Optional[str] = None,
        valid_tokens: Optional[set[Token]] = None
    ):
        """
        Initialize the parser.
        
        Args:
            grammar: Lark grammar string (for future integration)
            valid_tokens: Set of tokens that are always valid (for testing)
        """
        self.grammar = grammar
        self._valid_tokens = valid_tokens or set()
        self._always_valid = True  # For testing: accept everything
    
    def IsValidPrefix(self, prefix: Prefix) -> bool:
        """
        Check if a prefix is valid under the grammar.
        
        Stub: returns True for testing purposes.
        """
        if self._always_valid:
            return True
        return all(t in self._valid_tokens for t in prefix)
    
    def IsCompletePrefix(self, prefix: Prefix) -> bool:
        """
        Check if a prefix is a complete valid output.
        
        Stub: considers it complete if it ends with EOS.
        """
        if not prefix:
            return False
        return prefix[-1] == "<EOS>"
    
    def ValidNextTokens(self, prefix: Prefix) -> list[Token]:
        """
        Get tokens that can validly follow the prefix.
        
        Stub: returns all valid tokens.
        """
        return list(self._valid_tokens) if self._valid_tokens else []


# =============================================================================
# Token Constraint Types (matching Dafny datatypes)
# =============================================================================

@dataclass
class GrammarMask:
    """Only grammar-valid tokens."""
    pass

@dataclass
class Lookahead:
    """Avoid dead ends within depth steps."""
    depth: int

@dataclass
class LengthBound:
    """Enforce length constraints."""
    min: int
    max: int

@dataclass
class BanTokens:
    """Blacklist specific tokens."""
    banned: set[Token]

@dataclass
class AllowOnlyTokens:
    """Whitelist specific tokens."""
    allowed: set[Token]

@dataclass
class Intersect:
    """Must pass both constraints."""
    a: Any  # TokenConstraint
    b: Any  # TokenConstraint

@dataclass
class Union:
    """Can pass either constraint."""
    a: Any  # TokenConstraint
    b: Any  # TokenConstraint

@dataclass
class NoConstraint:
    """Allow all tokens."""
    pass


TokenConstraint = GrammarMask | Lookahead | LengthBound | BanTokens | AllowOnlyTokens | Intersect | Union | NoConstraint


# =============================================================================
# Repair Rules (matching Dafny datatypes)
# =============================================================================

@dataclass
class BracketBalance:
    """Fix mismatched brackets."""
    pass

@dataclass
class QuoteFix:
    """Fix unclosed quotes."""
    pass

@dataclass
class WhitespaceNormalize:
    """Normalize whitespace."""
    pass

@dataclass
class ComposedRepair:
    """Apply multiple repairs."""
    a: Any  # RepairRules
    b: Any  # RepairRules

@dataclass
class NoRepair:
    """No repair."""
    pass


RepairRules = BracketBalance | QuoteFix | WhitespaceNormalize | ComposedRepair | NoRepair


# =============================================================================
# Sequence Operations (matching Dafny datatypes)
# =============================================================================

@dataclass
class Repair:
    """Apply deterministic fixes."""
    rules: RepairRules

@dataclass
class PrefixCompleteOp:
    """Complete a valid prefix under constraint."""
    constraint: TokenConstraint

@dataclass
class ValidateOp:
    """Check semantic validity."""
    pred: Any  # SemanticPredicate

@dataclass
class Identity:
    """No-op, return as-is."""
    pass


SeqOperation = Repair | PrefixCompleteOp | ValidateOp | Identity


# =============================================================================
# Check Predicates (matching Dafny datatypes)
# =============================================================================

@dataclass
class ParseOk:
    """Output parses under grammar."""
    pass

@dataclass
class SemanticOk:
    """Custom semantic check."""
    pred: Any  # SemanticPredicate

@dataclass
class Both:
    """Must pass both checks."""
    a: Any  # CheckPredicate
    b: Any  # CheckPredicate

@dataclass
class Either:
    """Must pass at least one."""
    a: Any  # CheckPredicate
    b: Any  # CheckPredicate


CheckPredicate = ParseOk | SemanticOk | Both | Either


# =============================================================================
# Attempts (matching Dafny datatypes)
# =============================================================================

@dataclass
class Unconstrained:
    """Free LLM generation."""
    pass

@dataclass
class ConstrainedAttempt:
    """Constrained generation."""
    constraint: TokenConstraint

@dataclass
class WithRepair:
    """Attempt + repair on output."""
    base: Any  # Attempt
    rules: RepairRules

@dataclass
class WithSeqOp:
    """Attempt + sequence operation."""
    base: Any  # Attempt
    op: SeqOperation


Attempt = Unconstrained | ConstrainedAttempt | WithRepair | WithSeqOp


# =============================================================================
# Strategies (matching Dafny datatypes)
# =============================================================================

@dataclass
class Window:
    """CRANE-style windowing."""
    startDelim: Token
    endDelim: Token
    inside: TokenConstraint
    outside: TokenConstraint

@dataclass
class TryK:
    """Retry k times, then fallback."""
    k: int
    attempt: Attempt
    check: CheckPredicate
    fallback: Any  # Strategy

@dataclass
class Cascade:
    """Try strategies in order."""
    strategies: list[Any]  # list[Strategy]
    check: CheckPredicate

@dataclass
class BestOfN:
    """Generate n, pick first valid."""
    n: int
    base: Any  # Strategy
    check: CheckPredicate

@dataclass
class Constrained:
    """Terminal constrained decode."""
    constraint: TokenConstraint

@dataclass
class Free:
    """Terminal free generation."""
    pass


Strategy = Window | TryK | Cascade | BestOfN | Constrained | Free


# =============================================================================
# CSDHelpers Externs
# =============================================================================

class CSDHelpers:
    """
    Extern implementations for CSDHelpers static methods.
    """
    
    @staticmethod
    def ExtractContentExtern(input_str: str, start_delim: str, end_delim: str) -> str:
        """
        Extract the last occurrence of content between start and end delimiters.
        """
        import re
        # Escape delimiters for regex
        start = re.escape(start_delim)
        end = re.escape(end_delim)
        # Find all occurrences
        pattern = f"{start}(.*?){end}"
        matches = re.findall(pattern, input_str, re.DOTALL)
        if matches:
            return matches[-1]
        return ""

def AllowedNext(
    c: TokenConstraint,
    parser: Parser,
    prefix: Prefix,
    allTokens: set[Token]
) -> set[Token]:
    """
    Compute the set of allowed next tokens given a constraint.
    
    Args:
        c: Token constraint to apply
        parser: Grammar parser
        prefix: Current prefix
        allTokens: Universe of all tokens
        
    Returns:
        Set of allowed tokens
    """
    if isinstance(c, GrammarMask):
        # Return tokens that continue a valid parse
        valid = parser.ValidNextTokens(prefix)
        return set(valid) & allTokens if valid else allTokens
    
    elif isinstance(c, Lookahead):
        # Stub: same as GrammarMask for now
        valid = parser.ValidNextTokens(prefix)
        return set(valid) & allTokens if valid else allTokens
    
    elif isinstance(c, LengthBound):
        # Length constraints don't filter tokens directly
        return allTokens
    
    elif isinstance(c, BanTokens):
        return allTokens - c.banned
    
    elif isinstance(c, AllowOnlyTokens):
        return allTokens & c.allowed
    
    elif isinstance(c, Intersect):
        left = AllowedNext(c.a, parser, prefix, allTokens)
        right = AllowedNext(c.b, parser, prefix, allTokens)
        return left & right
    
    elif isinstance(c, Union):
        left = AllowedNext(c.a, parser, prefix, allTokens)
        right = AllowedNext(c.b, parser, prefix, allTokens)
        return left | right
    
    elif isinstance(c, NoConstraint):
        return allTokens
    
    else:
        return allTokens


def ChooseToken(
    lm: LM,
    c: TokenConstraint,
    parser: Parser,
    prefix: Prefix,
    allTokens: set[Token]
) -> Token:
    """
    Choose the highest-logit token from the allowed set.
    """
    allowed = AllowedNext(c, parser, prefix, allTokens)
    
    if not allowed:
        # Fallback: return first token
        return lm.Tokens[0]
    
    # Find max logit among allowed tokens
    best_token = None
    best_logit = float('-inf')
    
    for token in allowed:
        if token in lm._token_to_id:
            logit = lm.TokenToLogit(token)
            if logit > best_logit:
                best_logit = logit
                best_token = token
    
    return best_token if best_token else list(allowed)[0]


def ApplyRepair(rules: RepairRules, output: Prefix) -> Prefix:
    """
    Apply repair rules to an output.
    """
    if isinstance(rules, BracketBalance):
        # Simple bracket balancing
        result = list(output)
        open_count = sum(1 for t in result if t in "([{")
        close_count = sum(1 for t in result if t in ")]}")
        
        # Add missing closing brackets
        while open_count > close_count:
            result.append(")")
            close_count += 1
        
        return result
    
    elif isinstance(rules, QuoteFix):
        # Simple quote fixing
        result = list(output)
        quote_count = sum(1 for t in result if t == '"')
        if quote_count % 2 == 1:
            result.append('"')
        return result
    
    elif isinstance(rules, WhitespaceNormalize):
        # Remove consecutive whitespace
        result = []
        prev_ws = False
        for t in output:
            is_ws = t in " \t\n"
            if not (is_ws and prev_ws):
                result.append(t)
            prev_ws = is_ws
        return result
    
    elif isinstance(rules, ComposedRepair):
        intermediate = ApplyRepair(rules.a, output)
        return ApplyRepair(rules.b, intermediate)
    
    elif isinstance(rules, NoRepair):
        return output
    
    else:
        return output


def CheckSemantic(pred: Any, output: Prefix) -> bool:
    """
    Evaluate a semantic predicate on an output.
    
    Stub: always returns True for testing.
    """
    return True


def CompletePrefixConstrained(
    lm: LM,
    parser: Parser,
    prefix: Prefix,
    constraint: TokenConstraint,
    allTokens: set[Token],
    maxSteps: int
) -> Prefix:
    """
    Complete a valid prefix using constrained greedy decoding.
    """
    result = list(prefix)
    
    for _ in range(maxSteps):
        # Check if complete
        if parser.IsCompletePrefix(result):
            break
        
        # Generate logits for next token
        lm.GenerateLogits(result)
        
        # Choose best allowed token
        token = ChooseToken(lm, constraint, parser, result, allTokens)
        result.append(token)
    
    return result


def ApplySeqOp(
    lm: LM,
    op: SeqOperation,
    parser: Parser,
    output: Prefix,
    allTokens: set[Token],
    maxSteps: int
) -> Prefix:
    """
    Apply a sequence operation.
    """
    if isinstance(op, Identity):
        return output
    
    elif isinstance(op, Repair):
        return ApplyRepair(op.rules, output)
    
    elif isinstance(op, PrefixCompleteOp):
        if parser.IsValidPrefix(output):
            return CompletePrefixConstrained(
                lm, parser, output, op.constraint, allTokens, maxSteps
            )
        return output
    
    elif isinstance(op, ValidateOp):
        # Just return the output (validation is a check, not a transform)
        return output
    
    else:
        return output


def CheckOutput(check: CheckPredicate, parser: Parser, output: Prefix) -> bool:
    """
    Evaluate a check predicate on an output.
    """
    if isinstance(check, ParseOk):
        return parser.IsValidPrefix(output)
    
    elif isinstance(check, SemanticOk):
        return CheckSemantic(check.pred, output)
    
    elif isinstance(check, Both):
        return CheckOutput(check.a, parser, output) and CheckOutput(check.b, parser, output)
    
    elif isinstance(check, Either):
        return CheckOutput(check.a, parser, output) or CheckOutput(check.b, parser, output)
    
    else:
        return True


def RunAttempt(
    lm: LM,
    attempt: Attempt,
    parser: Parser,
    prompt: Prefix,
    allTokens: set[Token],
    maxSteps: int
) -> Prefix:
    """
    Execute a single generation attempt.
    """
    if isinstance(attempt, Unconstrained):
        # Free generation
        result = list(prompt)
        for _ in range(maxSteps):
            token = lm.ChooseNextToken(result)
            result.append(token)
            if token == "<EOS>":
                break
        return result
    
    elif isinstance(attempt, ConstrainedAttempt):
        return CompletePrefixConstrained(
            lm, parser, prompt, attempt.constraint, allTokens, maxSteps
        )
    
    elif isinstance(attempt, WithRepair):
        intermediate = RunAttempt(lm, attempt.base, parser, prompt, allTokens, maxSteps)
        return ApplyRepair(attempt.rules, intermediate)
    
    elif isinstance(attempt, WithSeqOp):
        intermediate = RunAttempt(lm, attempt.base, parser, prompt, allTokens, maxSteps)
        return ApplySeqOp(lm, attempt.op, parser, intermediate, allTokens, maxSteps)
    
    else:
        return prompt


def RunStrategy(
    lm: LM,
    strategy: Strategy,
    parser: Parser,
    prompt: Prefix,
    allTokens: set[Token],
    maxSteps: int
) -> Prefix:
    """
    Execute a strategy.
    """
    if isinstance(strategy, Constrained):
        return CompletePrefixConstrained(
            lm, parser, prompt, strategy.constraint, allTokens, maxSteps
        )
    
    elif isinstance(strategy, Free):
        result = list(prompt)
        for _ in range(maxSteps):
            token = lm.ChooseNextToken(result)
            result.append(token)
            if token == "<EOS>":
                break
        return result
    
    elif isinstance(strategy, Window):
        # CRANE-style windowing
        result = list(prompt)
        inside_window = False
        
        for _ in range(maxSteps):
            lm.GenerateLogits(result)
            
            # Check for delimiter transitions
            if not inside_window:
                # Outside: use outside constraint, watch for start delimiter
                constraint = strategy.outside
                token = ChooseToken(lm, constraint, parser, result, allTokens)
                result.append(token)
                if token == strategy.startDelim:
                    inside_window = True
            else:
                # Inside: use inside constraint, watch for end delimiter
                constraint = strategy.inside
                token = ChooseToken(lm, constraint, parser, result, allTokens)
                result.append(token)
                if token == strategy.endDelim:
                    inside_window = False
            
            if token == "<EOS>":
                break
        
        return result
    
    elif isinstance(strategy, TryK):
        for _ in range(strategy.k):
            output = RunAttempt(lm, strategy.attempt, parser, prompt, allTokens, maxSteps)
            if CheckOutput(strategy.check, parser, output):
                return output
        
        # Fallback
        return RunStrategy(lm, strategy.fallback, parser, prompt, allTokens, maxSteps)
    
    elif isinstance(strategy, Cascade):
        for s in strategy.strategies:
            output = RunStrategy(lm, s, parser, prompt, allTokens, maxSteps)
            if CheckOutput(strategy.check, parser, output):
                return output
        
        # Return last output even if it didn't pass
        if strategy.strategies:
            return RunStrategy(lm, strategy.strategies[-1], parser, prompt, allTokens, maxSteps)
        return prompt
    
    elif isinstance(strategy, BestOfN):
        outputs = []
        for _ in range(strategy.n):
            output = RunStrategy(lm, strategy.base, parser, prompt, allTokens, maxSteps)
            outputs.append(output)
            if CheckOutput(strategy.check, parser, output):
                return output
        
        # Return first output if none passed
        return outputs[0] if outputs else prompt
    
    else:
        return prompt


def Run(
    lm: LM,
    strategy: Strategy,
    parser: Parser,
    prompt: Prefix,
    allTokens: set[Token],
    maxSteps: int
) -> Prefix:
    """
    Main entry point for executing a CSD strategy.
    
    This wraps RunStrategy and is the function that verified code should call.
    """
    return RunStrategy(lm, strategy, parser, prompt, allTokens, maxSteps)


def ConstrainedDecode(
    lm: LM,
    parser: Parser,
    prefix: Prefix,
    maxSteps: int
) -> Prefix:
    """
    Perform constrained decoding with GrammarMask.
    """
    allTokens = set(lm.Tokens)
    return CompletePrefixConstrained(
        lm, parser, prefix, GrammarMask(), allTokens, maxSteps
    )


# =============================================================================
# Convenience Functions (matching Dafny convenience constructors)
# =============================================================================

def CRANEStyle(startDelim: Token, endDelim: Token) -> Strategy:
    """CRANE-style: unconstrained reasoning with constrained answer windows."""
    return Window(startDelim, endDelim, GrammarMask(), NoConstraint())


def RetryThenConstrained(k: int) -> Strategy:
    """Retry K times unconstrained, then fall back to constrained."""
    return TryK(k, Unconstrained(), ParseOk(), Constrained(GrammarMask()))


def BestOfNWithRepair(n: int, repairRules: RepairRules) -> Strategy:
    """Best-of-N with repair, falling back to constrained."""
    return BestOfN(
        n,
        TryK(1, WithRepair(Unconstrained(), repairRules), ParseOk(), Constrained(GrammarMask())),
        ParseOk()
    )


# =============================================================================
# Strategy Validity Check
# =============================================================================

def GuaranteesValidOutput(strategy: Strategy) -> bool:
    """Check if a strategy guarantees valid output."""
    if isinstance(strategy, Constrained):
        return True
    
    elif isinstance(strategy, Window):
        return True
    
    elif isinstance(strategy, TryK):
        return GuaranteesValidOutput(strategy.fallback)
    
    elif isinstance(strategy, Cascade):
        if strategy.strategies:
            return GuaranteesValidOutput(strategy.strategies[-1])
        return False
    
    elif isinstance(strategy, BestOfN):
        return GuaranteesValidOutput(strategy.base)
    
    elif isinstance(strategy, Free):
        return False
    
    else:
        return False

