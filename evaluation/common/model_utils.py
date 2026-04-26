"""
Model loading and management utilities for CSD evaluation.

Provides optimized model loading for CUDA/CPU with proper device handling
for multi-GPU setups using accelerate's device_map="auto".
"""

from __future__ import annotations

import math
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Set precision before any torch operations to avoid TensorFloat32 warning
torch.set_float32_matmul_precision('high')

# FOL (FOLIO) grammar keywords as single tokens so ValidNextTokens can extend formulas (e.g. Pred(x) + "{and}")
FOL_KEYWORD_TOKENS = [
    "{forall}", "{exists}", "{and}", "{or}", "{not}", "{implies}", "{iff}", "{xor}",
]

# GSM delimiter strings to add explicitly to the vocabulary.
# The Qwen tokenizer produces ' <<' (id 1115, with leading space) for << in context,
# and '>>' (id 2452) for >>. We ensure both ' <<' and ' >>' are in vocab, plus '>>'
# as a fallback in case ' >>' is not a single BPE token.
GSM_DELIMITER_STRINGS = [" <<", " >>", ">>"]


def _hf_offline_enabled() -> bool:
    """True when HuggingFace model/tokenizer loaders should stay offline."""
    return any(os.environ.get(name, "").strip() in {"1", "true", "True"} for name in (
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
    ))


def _is_hf_connection_error(exc: Exception) -> bool:
    """Best-effort detection for DNS/network lookup failures in HF loading."""
    text = str(exc).lower()
    return any(marker in text for marker in (
        "failed to resolve",
        "name or service not known",
        "temporary failure in name resolution",
        "connection error",
        "maxretryerror",
        "httpsconnectionpool",
        "offline mode",
    ))


def _load_tokenizer(model_name: str, **kwargs):
    """Load a tokenizer, falling back to local-files-only on network failure."""
    load_kwargs = dict(kwargs)
    if _hf_offline_enabled():
        load_kwargs["local_files_only"] = True
    try:
        return AutoTokenizer.from_pretrained(model_name, **load_kwargs)
    except Exception as exc:
        if load_kwargs.get("local_files_only") or not _is_hf_connection_error(exc):
            raise
        print("  HuggingFace network lookup failed; retrying tokenizer load from local cache only.")
        load_kwargs["local_files_only"] = True
        return AutoTokenizer.from_pretrained(model_name, **load_kwargs)


def _load_causal_lm(**kwargs):
    """Load a causal LM, falling back to local-files-only on network failure."""
    load_kwargs = dict(kwargs)
    if _hf_offline_enabled():
        load_kwargs["local_files_only"] = True
    try:
        return AutoModelForCausalLM.from_pretrained(**load_kwargs)
    except Exception as exc:
        if load_kwargs.get("local_files_only") or not _is_hf_connection_error(exc):
            raise
        print("  HuggingFace network lookup failed; retrying model load from local cache only.")
        load_kwargs["local_files_only"] = True
        return AutoModelForCausalLM.from_pretrained(**load_kwargs)


def _decode_single_token(tokenizer, token_id: int) -> str:
    """Decode one token ID without cleanup so token strings stay stable."""
    return tokenizer.decode([token_id], clean_up_tokenization_spaces=False)


def _dedupe_token_ids_by_decoded_string(tokenizer, token_ids: list[int]) -> tuple[list[int], int]:
    """
    Keep only the first HF token ID for each decoded token string.

    The verified LM invariant requires logical tokens to be unique strings, but
    HuggingFace vocabularies often contain multiple IDs that decode to the same
    surface form. Collapsing duplicates here preserves a stable logical vocab
    while still letting us use the original HF IDs for logits lookup.
    """
    unique_ids: list[int] = []
    seen_tokens: set[str] = set()
    dropped = 0
    for token_id in token_ids:
        token = _decode_single_token(tokenizer, token_id)
        if token in seen_tokens:
            dropped += 1
            continue
        seen_tokens.add(token)
        unique_ids.append(token_id)
    return unique_ids, dropped


def _valid_tokens_ids_logits_py(tokens: list[str], ids: list[int], logits: list[float]) -> bool:
    """Python-native equivalent of LM.ValidTokensIdsLogits with linear-time uniqueness."""
    return (
        len(tokens) == len(ids) == len(logits)
        and len(ids) > 0
        and ids[0] == 0
        and all(i == ids[i] for i in range(len(ids)))
        and len(set(tokens)) == len(tokens)
        and all(-1e9 <= logit <= 1e9 for logit in logits)
    )


def get_model_input_device(model) -> torch.device:
    """
    Find the device where the model's embedding layer resides.
    
    For models with hf_device_map (multi-GPU via accelerate), this determines
    where input tensors should be placed.
    
    Args:
        model: A HuggingFace model instance
        
    Returns:
        The torch.device for input tensors
    """
    # For models with hf_device_map (multi-GPU via accelerate)
    if hasattr(model, 'hf_device_map') and model.hf_device_map:
        # Look for embedding layer in device map
        for key, device in model.hf_device_map.items():
            if 'embed' in key.lower():
                return torch.device(f"cuda:{device}" if isinstance(device, int) else device)
        # Fallback to first device in map
        first_device = next(iter(model.hf_device_map.values()))
        return torch.device(f"cuda:{first_device}" if isinstance(first_device, int) else first_device)
    
    # For single-device models
    return next(model.parameters()).device


def get_max_input_length(model, tokenizer) -> int:
    """
    Choose a safe max input length for the model/tokenizer.
    
    Args:
        model: A HuggingFace model instance
        tokenizer: A HuggingFace tokenizer instance
        
    Returns:
        Maximum input length in tokens
    """
    max_len = None
    if hasattr(model, "config") and getattr(model.config, "max_position_embeddings", None):
        max_len = int(model.config.max_position_embeddings)
    tok_max = getattr(tokenizer, "model_max_length", None)
    if tok_max and tok_max < 1_000_000:
        max_len = min(max_len, int(tok_max)) if max_len else int(tok_max)
    return max_len or 4096


def create_huggingface_lm(
    model_name: str,
    device: str,
    vocab_size: int,
    VerifiedDecoderAgent,
    _dafny,
    token_ids=None,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    add_fol_keyword_tokens: bool = False,
    add_gsm_delimiter_tokens: bool = False,
):
    """
    Create a HuggingFace LM wrapped with a Dafny-compatible interface.

    Args:
        model_name: HuggingFace model identifier
        device: Device to use ("cuda", "cpu", etc.)
        vocab_size: Size of constrained vocabulary
        VerifiedDecoderAgent: Imported Dafny module for LM interface
        _dafny: Dafny runtime module
        token_ids: Optional list of token IDs for constrained vocabulary
        load_in_4bit: Whether to load in 4-bit quantization
        load_in_8bit: Whether to load in 8-bit quantization

    Returns:
        A Dafny-compatible LM wrapper
    """
    prec_str = "FP16"
    if load_in_4bit: prec_str = "4-bit"
    elif load_in_8bit: prec_str = "8-bit"
    
    print(f"Loading model: {model_name} on {device}... ({prec_str})")
    tokenizer = _load_tokenizer(model_name, trust_remote_code=True)

    # Add FOL keywords as single tokens so "{and}", "{or}" etc. are one token (ValidNextTokens can then extend formulas)
    if add_fol_keyword_tokens:
        added = tokenizer.add_tokens(FOL_KEYWORD_TOKENS, special_tokens=False)
        if added:
            print(f"  Added {added} FOL keyword token(s) for single-token formula extension.")

    # Always use device_map="auto" for CUDA to leverage all available GPUs
    if device.startswith("cuda"):
        kwargs = {
            "pretrained_model_name_or_path": model_name,
            "trust_remote_code": True,
            "device_map": "auto",
        }
        
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        elif load_in_8bit:
            kwargs["load_in_8bit"] = True
        else:
            kwargs["torch_dtype"] = torch.float16
            
        model = _load_causal_lm(**kwargs)
        input_device = get_model_input_device(model)
        num_gpus = torch.cuda.device_count()
        print(f"Model loaded across {num_gpus} GPU(s), inputs go to {input_device}")
    else:
        # CPU fallback
        model = _load_causal_lm(
            pretrained_model_name_or_path=model_name,
            trust_remote_code=True,
            torch_dtype=torch.float32,
        )
        input_device = torch.device("cpu")

    model.eval()

    # Resize embeddings if we added FOL tokens
    if add_fol_keyword_tokens and len(tokenizer) != model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(len(tokenizer))

    if token_ids is None:
        token_ids = list(range(vocab_size))
        if add_fol_keyword_tokens:
            # Include new FOL keyword IDs so they appear in _Tokens and can be chosen by ValidNextTokens
            for t in FOL_KEYWORD_TOKENS:
                tid = tokenizer.convert_tokens_to_ids(t)
                if tid != tokenizer.unk_token_id and tid not in token_ids:
                    token_ids.append(tid)
        if add_gsm_delimiter_tokens:
            # Add GSM delimiter tokens: ' <<' (id≈1115) is likely already in default 2000-token vocab,
            # but ' >>' and '>>' (id≈2452) are typically outside. Add whichever encode as single tokens.
            existing_ids = set(token_ids)
            for s in GSM_DELIMITER_STRINGS:
                ids = tokenizer.encode(s, add_special_tokens=False)
                if len(ids) == 1:
                    tid = ids[0]
                    if tid not in existing_ids:
                        token_ids.append(tid)
                        existing_ids.add(tid)
                        print(f"  Added GSM delimiter token {repr(s)} (id={tid}) to vocabulary.")

    token_ids, dropped_duplicates = _dedupe_token_ids_by_decoded_string(tokenizer, token_ids)
    if dropped_duplicates:
        print(f"  Deduplicated {dropped_duplicates} decoded token string(s) from vocabulary.")

    tokens_dafny = _dafny.SeqWithoutIsStrInference(
        [_dafny.Seq(_decode_single_token(tokenizer, tid)) for tid in token_ids]
    )

    class HuggingFaceLM(VerifiedDecoderAgent.LM):
        """Wrapper that bridges HuggingFace models to the Dafny LM interface."""
        
        # Token IDs that must not be chosen as the first output token (so we get plain text before the delimiter)
        _FORBID_FIRST_STRINGS = frozenset({"<<", "<", " <<", " << ", " >>", ">>", "$"})

        def __init__(self, hf_model, hf_tokenizer, tokens, tids, dev):
            super().__init__()
            self.model = hf_model
            self.tokenizer = hf_tokenizer
            self._Tokens = tokens
            self._token_ids = tids
            self._input_device = dev
            self._max_input_len = get_max_input_length(hf_model, hf_tokenizer)
            self.instruction_text = ""
            self.Logits = _dafny.Array(None, len(tids))
            for i in range(len(tids)):
                self.Logits[i] = _dafny.BigRational(0)
            # Store full logits for unconstrained generation
            self._full_logits = None
            # Cache token IDs forbidden as first output token (delimiter + EOS so we get real text, not empty)
            vocab_size = hf_tokenizer.vocab_size if hasattr(hf_tokenizer, "vocab_size") else len(hf_tokenizer)
            self._forbid_first_token_ids = set()
            for vid in range(min(vocab_size, 200000)):  # cap for speed
                try:
                    s = _decode_single_token(hf_tokenizer, vid)
                    if s in HuggingFaceLM._FORBID_FIRST_STRINGS:
                        self._forbid_first_token_ids.add(vid)
                    elif not s:  # forbid tokens that decode to empty string
                        self._forbid_first_token_ids.add(vid)
                except Exception:
                    pass
            # Forbid EOS/pad as first token so the model cannot immediately stop (which produced empty output)
            eos_id = getattr(hf_tokenizer, "eos_token_id", None)
            if eos_id is not None:
                self._forbid_first_token_ids.add(int(eos_id))
            pad_id = getattr(hf_tokenizer, "pad_token_id", None)
            if pad_id is not None and pad_id != eos_id:
                self._forbid_first_token_ids.add(int(pad_id))

        def _to_str(self, obj):
            """Convert a Dafny object (potentially a Seq of chars) to a Python string."""
            if isinstance(obj, str):
                return obj
            try:
                # Dafny Seqs of chars can be converted by joining their elements
                return "".join(obj[i] for i in range(len(obj)))
            except:
                return str(obj)

        def GenerateLogits(self, input_prefix):
            """Compute logits for the next token given a prefix."""
            import os
            debug = os.environ.get('CSD_MASK_DEBUG', '').lower() in ('1', 'true', 'yes')

            # Correctly handle Dafny sequences which might contain char sequences
            prefix_parts = []
            for i in range(len(input_prefix)):
                prefix_parts.append(self._to_str(input_prefix[i]))
            prefix_text = "".join(prefix_parts)
            
            full_prompt = self.instruction_text + prefix_text
            
            if debug and len(input_prefix) <= 5:
                print(f"    [GENERATE DEBUG] Step {len(input_prefix)} prompt tail:\n...{full_prompt[-200:]}")
                if len(input_prefix) == 0:
                    print(f"    [GENERATE DEBUG] Full initial prompt length: {len(full_prompt)}")

            inputs = self.tokenizer(
                full_prompt,
                return_tensors="pt",
                add_special_tokens=False,
            )
            if inputs["input_ids"].shape[-1] > self._max_input_len:
                inputs["input_ids"] = inputs["input_ids"][:, -self._max_input_len:]
                if "attention_mask" in inputs:
                    inputs["attention_mask"] = inputs["attention_mask"][:, -self._max_input_len:]
            inputs = inputs.to(self._input_device)

            with torch.no_grad():
                # Get logits and immediately move to CPU to avoid cross-device issues
                output = self.model(**inputs)
                logits = output.logits[0, -1, :].float().cpu()
            
            # Forbid delimiter + EOS as first output token so model outputs plain text before the formula
            self._first_token_choice = len(input_prefix) == 0
            if self._first_token_choice and getattr(self, "_forbid_first_token_ids", None):
                for vid in self._forbid_first_token_ids:
                    if vid < logits.shape[0]:
                        logits[vid] = float("-inf")

            # Store full logits for unconstrained generation
            self._full_logits = logits

            # BigRational cannot represent ±inf; clamp so forbidden tokens stay worst
            _LOGIT_FORBIDDEN = -1e9
            for i, tid in enumerate(self._token_ids):
                v = float(logits[tid].item())
                if not math.isfinite(v):
                    v = _LOGIT_FORBIDDEN
                self.Logits[i] = _dafny.BigRational(v)

        def ChooseNextToken(self):
            """Return the token with the highest logit score (constrained to vocab)."""
            import os
            debug = os.environ.get('CSD_MASK_DEBUG', '').lower() in ('1', 'true', 'yes')

            best_idx, best_val = 0, float(self.Logits[0])
            for i in range(1, self.Logits.length(0)):
                val = float(self.Logits[i])
                if val > best_val:
                    best_val, best_idx = val, i

            chosen_token = self._Tokens[best_idx]
            if debug:
                # Convert Dafny Seq to string for display
                try:
                    token_str = ''.join(chosen_token[i] for i in range(len(chosen_token)))
                except:
                    token_str = str(chosen_token)
                print(f"    [CHOOSE DEBUG] Best idx={best_idx}, logit={best_val:.2f}, token={repr(token_str)}")

            return chosen_token
        
        def ChooseNextTokenUnconstrained(self):
            """Return the token with the highest logit score from FULL vocabulary."""
            import os
            debug = os.environ.get('CSD_MASK_DEBUG', '').lower() in ('1', 'true', 'yes')

            if self._full_logits is None:
                raise RuntimeError("Must call GenerateLogits before ChooseNextTokenUnconstrained")
            logits = self._full_logits.clone()
            best_idx = int(logits.argmax().item())
            token_text = _decode_single_token(self.tokenizer, best_idx)
            # If this was the first token and we got empty/EOS (mask failed or tokenizer quirk), take next-best
            if getattr(self, "_first_token_choice", False):
                self._first_token_choice = False
                eos_str = (
                    _decode_single_token(self.tokenizer, self.tokenizer.eos_token_id)
                    if getattr(self.tokenizer, "eos_token_id", None) is not None
                    else ""
                )
                for _ in range(50):
                    if token_text and token_text.strip() and token_text != eos_str:
                        break
                    logits[best_idx] = float("-inf")
                    best_idx = int(logits.argmax().item())
                    token_text = _decode_single_token(self.tokenizer, best_idx)
            
            if debug:
                print(f"    [UNCONSTRAINED DEBUG] chosen_token={repr(token_text)}")

            return _dafny.Seq(token_text)
        
        def MaskTokensExcept(self, valid_tokens, debug=False):
            """Mask all tokens except those in valid_tokens.
            
            This implementation follows the Dafny specification strictly.
            When this is the first token (_first_token_choice), we also exclude EOS and
            empty-decoding tokens so strategies that use ConstrainedStep first still get real text.
            """
            import os
            debug = debug or os.environ.get('CSD_MASK_DEBUG', '').lower() in ('1', 'true', 'yes')

            # Helper to convert Dafny Seq to string
            def seq_to_str(seq):
                try:
                    return ''.join(seq)
                except TypeError:
                    try:
                        return ''.join(seq[i] for i in range(len(seq)))
                    except:
                        return str(seq)

            # Get the set of valid token strings
            valid_set = set()
            for i in range(len(valid_tokens)):
                valid_set.add(seq_to_str(valid_tokens[i]))

            # First token: exclude EOS and delimiter so we don't produce blank output (some strategies use ConstrainedStep first)
            if getattr(self, "_first_token_choice", False):
                self._first_token_choice = False
                eos_str = self.tokenizer.decode([self.tokenizer.eos_token_id]) if getattr(self.tokenizer, "eos_token_id", None) is not None else ""
                forbid = {"", eos_str} | set(HuggingFaceLM._FORBID_FIRST_STRINGS)
                reduced = valid_set - forbid
                if reduced:
                    valid_set = reduced

            # Mask everything not in the valid set
            masked_val = _dafny.BigRational(-1000000000, 1) # -1e9
            
            masked_count = 0
            for i in range(self.Logits.length(0)):
                token_str = seq_to_str(self._Tokens[i])
                if token_str not in valid_set:
                    self.Logits[i] = masked_val
                    masked_count += 1
            
            if debug:
                print(f"    [MASK DEBUG] Masked {masked_count} tokens, {len(valid_set)} remain valid.")

    return HuggingFaceLM(model, tokenizer, tokens_dafny, token_ids, input_device)


def create_huggingface_lm_native(
    model_name: str,
    device: str,
    vocab_size: int,
    VerifiedAgentSynthesis,
    token_ids=None,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    add_gsm_delimiter_tokens: bool = False,
):
    """
    Create a HuggingFace LM wrapped with a Python-native (non-Dafny) interface.

    Uses plain Python lists for Tokens/Ids/Logits so the strategy code runs
    directly without the Dafny runtime.
    """
    prec_str = "FP16"
    if load_in_4bit:
        prec_str = "4-bit"
    elif load_in_8bit:
        prec_str = "8-bit"

    print(f"Loading model (native): {model_name} on {device}... ({prec_str})")
    tokenizer = _load_tokenizer(model_name, trust_remote_code=True)

    if device.startswith("cuda"):
        kwargs = {
            "pretrained_model_name_or_path": model_name,
            "trust_remote_code": True,
            "device_map": "auto",
        }
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        elif load_in_8bit:
            kwargs["load_in_8bit"] = True
        else:
            kwargs["torch_dtype"] = torch.float16
        model = _load_causal_lm(**kwargs)
        input_device = get_model_input_device(model)
    else:
        model = _load_causal_lm(
            pretrained_model_name_or_path=model_name,
            trust_remote_code=True,
            torch_dtype=torch.float32,
        )
        input_device = torch.device("cpu")

    model.eval()

    if token_ids is None:
        token_ids = list(range(vocab_size))
        if add_gsm_delimiter_tokens:
            existing_ids = set(token_ids)
            for s in GSM_DELIMITER_STRINGS:
                ids = tokenizer.encode(s, add_special_tokens=False)
                if len(ids) == 1:
                    tid = ids[0]
                    if tid not in existing_ids:
                        token_ids.append(tid)
                        existing_ids.add(tid)
                        print(f"  Added GSM delimiter token {repr(s)} (id={tid}) to vocabulary.")

    token_ids, dropped_duplicates = _dedupe_token_ids_by_decoded_string(tokenizer, token_ids)
    if dropped_duplicates:
        print(f"  Deduplicated {dropped_duplicates} decoded token string(s) from vocabulary.")

    tokens = [_decode_single_token(tokenizer, tid) for tid in token_ids]
    max_input_len = get_max_input_length(model, tokenizer)

    class HuggingFaceLMNative(VerifiedAgentSynthesis.LM):
        """HuggingFace model wrapped with Python-native (non-Dafny) LM interface."""

        _FORBID_FIRST_STRINGS = frozenset({"<<", "<", " <<", " << ", " >>", ">>", "$"})

        def __init__(self):
            # Bypass parent __init__ (sets 2-token dummy vocab); set full vocab directly
            self.Tokens = tokens
            self.Ids = list(range(len(tokens)))
            self.Logits = [0.0] * len(tokens)
            self._Tokens = self.Tokens  # alias for evaluator compat
            self._token_ids = token_ids
            self._input_device = input_device
            self._max_input_len = max_input_len
            self.tokenizer = tokenizer
            self.instruction_text = ""
            self._full_logits = None
            self._first_token_choice = False
            vocab_sz = tokenizer.vocab_size if hasattr(tokenizer, "vocab_size") else len(tokenizer)
            self._forbid_first_token_ids: set = set()
            for vid in range(min(vocab_sz, 200000)):
                try:
                    s = _decode_single_token(tokenizer, vid)
                    if s in HuggingFaceLMNative._FORBID_FIRST_STRINGS or not s:
                        self._forbid_first_token_ids.add(vid)
                except Exception:
                    pass
            eos_id = getattr(tokenizer, "eos_token_id", None)
            if eos_id is not None:
                self._forbid_first_token_ids.add(int(eos_id))
            pad_id = getattr(tokenizer, "pad_token_id", None)
            if pad_id is not None and pad_id != eos_id:
                self._forbid_first_token_ids.add(int(pad_id))

        def ValidTokensIdsLogits(self) -> bool:
            return _valid_tokens_ids_logits_py(self.Tokens, self.Ids, self.Logits)

        def ValidTokensIdsLogitsAlways(self) -> None:
            assert self.ValidTokensIdsLogits()

        def GenerateLogits(self, input_prefix: list) -> None:
            prefix_text = "".join(str(t) for t in input_prefix)
            full_prompt = self.instruction_text + prefix_text
            inputs = tokenizer(full_prompt, return_tensors="pt", add_special_tokens=False)
            if inputs["input_ids"].shape[-1] > self._max_input_len:
                inputs["input_ids"] = inputs["input_ids"][:, -self._max_input_len:]
                if "attention_mask" in inputs:
                    inputs["attention_mask"] = inputs["attention_mask"][:, -self._max_input_len:]
            inputs = inputs.to(self._input_device)

            self._first_token_choice = len(input_prefix) == 0
            with torch.no_grad():
                output = model(**inputs)
                logits = output.logits[0, -1, :].float().cpu()

            if self._first_token_choice and self._forbid_first_token_ids:
                for vid in self._forbid_first_token_ids:
                    if vid < logits.shape[0]:
                        logits[vid] = float("-inf")

            self._full_logits = logits
            _LOGIT_FORBIDDEN = -1e9
            for i, tid in enumerate(self._token_ids):
                v = float(logits[tid].item())
                if not math.isfinite(v):
                    v = _LOGIT_FORBIDDEN
                self.Logits[i] = v

        def ChooseNextToken(self) -> str:
            best_i = max(range(len(self.Logits)), key=lambda i: self.Logits[i])
            return self.Tokens[best_i]

        def ChooseNextTokenUnconstrained(self) -> str:
            if self._full_logits is None:
                raise RuntimeError("Call GenerateLogits before ChooseNextTokenUnconstrained")
            logits = self._full_logits.clone()
            best_idx = int(logits.argmax().item())
            return _decode_single_token(tokenizer, best_idx)

        def MaskTokensExcept(self, valid_tokens: list) -> None:
            valid_set = set(str(t) for t in valid_tokens)
            if self._first_token_choice:
                self._first_token_choice = False
                eos_str = (
                    _decode_single_token(tokenizer, tokenizer.eos_token_id)
                    if getattr(tokenizer, "eos_token_id", None) is not None
                    else ""
                )
                forbid = {"", eos_str} | set(HuggingFaceLMNative._FORBID_FIRST_STRINGS)
                reduced = valid_set - forbid
                if reduced:
                    valid_set = reduced
            for i in range(len(self.Logits)):
                if self.Tokens[i] not in valid_set:
                    self.Logits[i] = -1e9

    return HuggingFaceLMNative()
