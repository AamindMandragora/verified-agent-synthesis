"""
Model-token-based JSON Parser for constrained decoding.

This module provides a Parser implementation that operates on model tokens
(specifically Qwen/Qwen2.5-Coder-7B-Instruct tokens) and validates JSON
structure using the streaming JSON prefix validator.

The parser can:
- Check if a token sequence decodes to a valid JSON prefix
- Check if a token sequence decodes to complete JSON
- Filter vocabulary to only tokens that continue valid JSON
"""

from typing import Optional, Sequence
from functools import lru_cache
import logging

from .json_prefix import is_valid_json_prefix, is_complete_json, JsonPrefixValidator

logger = logging.getLogger(__name__)


class ModelTokenJsonParser:
    """
    JSON Parser that operates on model tokens.
    
    Uses a HuggingFace tokenizer to decode token sequences, then validates
    the decoded text against JSON grammar using the streaming validator.
    
    Implements the Parser interface expected by the CSD runtime:
    - IsValidPrefix(prefix_tokens) -> bool
    - IsCompletePrefix(prefix_tokens) -> bool  
    - ValidNextTokens(prefix_tokens) -> list[str]
    """
    
    def __init__(
        self,
        tokenizer=None,
        tokenizer_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct",
        cache_valid_tokens: bool = True,
        precompute_single_tokens: bool = True
    ):
        """
        Initialize the parser.
        
        Args:
            tokenizer: Pre-loaded HuggingFace tokenizer (optional)
            tokenizer_name: Name of tokenizer to load if not provided
            cache_valid_tokens: Whether to cache valid token computations
            precompute_single_tokens: Whether to precompute which single tokens are valid JSON starts
        """
        self._tokenizer = tokenizer
        self._tokenizer_name = tokenizer_name
        self._cache_valid_tokens = cache_valid_tokens
        
        # Lazy load tokenizer
        self._tokenizer_loaded = tokenizer is not None
        
        # Cache for valid next tokens given a prefix hash
        self._valid_tokens_cache: dict[str, list[str]] = {}
        
        # Precomputed set of tokens that are valid JSON starters/continuations
        self._single_token_valid: Optional[dict[str, bool]] = None
        self._precompute_single = precompute_single_tokens
    
    @property
    def tokenizer(self):
        """Lazy-load the tokenizer."""
        if not self._tokenizer_loaded:
            self._load_tokenizer()
        return self._tokenizer
    
    def _load_tokenizer(self):
        """Load the HuggingFace tokenizer."""
        try:
            from transformers import AutoTokenizer
            logger.info(f"Loading tokenizer: {self._tokenizer_name}")
            self._tokenizer = AutoTokenizer.from_pretrained(
                self._tokenizer_name,
                trust_remote_code=True
            )
            self._tokenizer_loaded = True
            logger.info(f"Loaded tokenizer with vocab size: {len(self._tokenizer)}")
        except ImportError:
            raise ImportError(
                "transformers library required. Install with: pip install transformers"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load tokenizer {self._tokenizer_name}: {e}")
    
    def _precompute_single_token_validity(self):
        """Precompute which single tokens are valid JSON starts."""
        if self._single_token_valid is not None:
            return
        
        logger.info("Precomputing single-token JSON validity...")
        self._single_token_valid = {}
        
        vocab = self.get_vocabulary()
        for token in vocab:
            # Check if this token alone is a valid JSON prefix
            self._single_token_valid[token] = is_valid_json_prefix(token)
        
        valid_count = sum(1 for v in self._single_token_valid.values() if v)
        logger.info(f"Precomputed validity: {valid_count}/{len(vocab)} tokens are valid JSON starters")
    
    def get_vocabulary(self) -> list[str]:
        """Get all tokens in the vocabulary as strings."""
        vocab_size = len(self.tokenizer)
        tokens = []
        for i in range(vocab_size):
            try:
                token = self.tokenizer.decode([i])
                tokens.append(token)
            except Exception:
                # Some token IDs may be invalid
                tokens.append("")
        return tokens
    
    def decode_tokens(self, token_ids: Sequence[int]) -> str:
        """Decode token IDs to a string."""
        if not token_ids:
            return ""
        return self.tokenizer.decode(list(token_ids), skip_special_tokens=False)
    
    def decode_token_strings(self, tokens: Sequence[str]) -> str:
        """
        Decode token strings to a single string.
        
        Note: This simply concatenates tokens. For accurate decoding from
        token IDs, use decode_tokens() instead.
        """
        return "".join(tokens)
    
    def encode_text(self, text: str) -> list[int]:
        """Encode text to token IDs."""
        return self.tokenizer.encode(text, add_special_tokens=False)
    
    def IsValidPrefix(self, prefix: Sequence[str]) -> bool:
        """
        Check if the token sequence decodes to a valid JSON prefix.
        
        Args:
            prefix: Sequence of token strings
            
        Returns:
            True if the decoded text is a valid JSON prefix
        """
        if not prefix:
            return True  # Empty prefix is valid
        
        text = self.decode_token_strings(prefix)
        return is_valid_json_prefix(text)
    
    def IsCompletePrefix(self, prefix: Sequence[str]) -> bool:
        """
        Check if the token sequence decodes to complete JSON.
        
        Args:
            prefix: Sequence of token strings
            
        Returns:
            True if the decoded text is complete, valid JSON
        """
        if not prefix:
            return False  # Empty is not complete
        
        text = self.decode_token_strings(prefix)
        return is_complete_json(text)
    
    def ValidNextTokens(self, prefix: Sequence[str]) -> list[str]:
        """
        Get all vocabulary tokens that can validly follow the prefix.
        
        This filters the entire vocabulary to only tokens that, when appended
        to the prefix, still form a valid JSON prefix.
        
        Args:
            prefix: Current sequence of token strings
            
        Returns:
            List of valid next tokens
        """
        # Get current decoded text
        current_text = self.decode_token_strings(prefix) if prefix else ""
        
        # Check cache
        cache_key = current_text if self._cache_valid_tokens else None
        if cache_key and cache_key in self._valid_tokens_cache:
            return self._valid_tokens_cache[cache_key]
        
        # If current prefix is already invalid, no tokens are valid
        if current_text and not is_valid_json_prefix(current_text):
            return []
        
        # Filter vocabulary
        valid_tokens = []
        vocab = self.get_vocabulary()
        
        for token in vocab:
            if not token:  # Skip empty tokens
                continue
            
            # Check if appending this token keeps the prefix valid
            extended = current_text + token
            if is_valid_json_prefix(extended):
                valid_tokens.append(token)
        
        # Cache result
        if cache_key:
            self._valid_tokens_cache[cache_key] = valid_tokens
        
        return valid_tokens
    
    def ValidNextTokenIds(self, prefix_ids: Sequence[int]) -> list[int]:
        """
        Get all vocabulary token IDs that can validly follow the prefix.
        
        This is a more efficient version that works with token IDs directly.
        
        Args:
            prefix_ids: Current sequence of token IDs
            
        Returns:
            List of valid next token IDs
        """
        # Decode current prefix
        current_text = self.decode_tokens(prefix_ids) if prefix_ids else ""
        
        # If current prefix is invalid, no tokens are valid
        if current_text and not is_valid_json_prefix(current_text):
            return []
        
        # Filter vocabulary by ID
        valid_ids = []
        vocab_size = len(self.tokenizer)
        
        for token_id in range(vocab_size):
            try:
                token_text = self.tokenizer.decode([token_id])
                if not token_text:
                    continue
                
                extended = current_text + token_text
                if is_valid_json_prefix(extended):
                    valid_ids.append(token_id)
            except Exception:
                continue
        
        return valid_ids
    
    def get_completion_tokens(self, prefix: Sequence[str]) -> list[str]:
        """
        Get tokens that would complete the current JSON structure.
        
        Useful for finishing incomplete JSON.
        
        Args:
            prefix: Current sequence of token strings
            
        Returns:
            List of tokens that would make the prefix complete
        """
        current_text = self.decode_token_strings(prefix) if prefix else ""
        
        if is_complete_json(current_text):
            return []  # Already complete
        
        if not is_valid_json_prefix(current_text):
            return []  # Invalid prefix, can't complete
        
        completion_tokens = []
        vocab = self.get_vocabulary()
        
        for token in vocab:
            if not token:
                continue
            
            extended = current_text + token
            if is_complete_json(extended):
                completion_tokens.append(token)
        
        return completion_tokens
    
    def clear_cache(self):
        """Clear the valid tokens cache."""
        self._valid_tokens_cache.clear()


class CachedModelTokenJsonParser(ModelTokenJsonParser):
    """
    Optimized JSON parser with aggressive caching and batch operations.
    
    Uses LRU caching and batch validation for better performance when
    filtering large vocabularies repeatedly.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._vocab_cache: Optional[list[str]] = None
        self._vocab_ids_cache: Optional[list[int]] = None
    
    @lru_cache(maxsize=10000)
    def _is_valid_cached(self, text: str) -> bool:
        """Cached version of is_valid_json_prefix."""
        return is_valid_json_prefix(text)
    
    @lru_cache(maxsize=10000)  
    def _is_complete_cached(self, text: str) -> bool:
        """Cached version of is_complete_json."""
        return is_complete_json(text)
    
    def IsValidPrefix(self, prefix: Sequence[str]) -> bool:
        text = self.decode_token_strings(prefix) if prefix else ""
        return self._is_valid_cached(text)
    
    def IsCompletePrefix(self, prefix: Sequence[str]) -> bool:
        text = self.decode_token_strings(prefix) if prefix else ""
        return self._is_complete_cached(text)
    
    def get_vocabulary(self) -> list[str]:
        """Cached vocabulary access."""
        if self._vocab_cache is None:
            self._vocab_cache = super().get_vocabulary()
        return self._vocab_cache
    
    def get_vocabulary_ids(self) -> list[int]:
        """Get all valid token IDs."""
        if self._vocab_ids_cache is None:
            self._vocab_ids_cache = list(range(len(self.tokenizer)))
        return self._vocab_ids_cache
    
    def clear_cache(self):
        """Clear all caches."""
        super().clear_cache()
        self._is_valid_cached.cache_clear()
        self._is_complete_cached.cache_clear()


def create_json_parser(
    tokenizer_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct",
    use_cache: bool = True
) -> ModelTokenJsonParser:
    """
    Factory function to create a JSON parser.
    
    Args:
        tokenizer_name: HuggingFace tokenizer name
        use_cache: Whether to use the cached version
        
    Returns:
        ModelTokenJsonParser instance
    """
    cls = CachedModelTokenJsonParser if use_cache else ModelTokenJsonParser
    return cls(tokenizer_name=tokenizer_name)

