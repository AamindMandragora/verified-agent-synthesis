"""
Model loading and management utilities for CSD evaluation.

Provides optimized model loading for CUDA/CPU with proper device handling
for multi-GPU setups using accelerate's device_map="auto".

Masking and token selection use torch tensor operations (masked_fill, argmax)
for O(1) performance independent of vocabulary size, following the approach
used by syncode's DFAMaskStore.
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Set precision before any torch operations to avoid TensorFloat32 warning
torch.set_float32_matmul_precision('high')


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
    VerifiedDecoderAgent,
    _dafny,
    token_ids=None,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
):
    """
    Create a HuggingFace LM wrapped with a Dafny-compatible interface.

    Uses the full tokenizer vocabulary. All logit manipulation (masking,
    token selection) is done via torch tensor ops for O(1) performance.

    Args:
        model_name: HuggingFace model identifier
        device: Device to use ("cuda", "cpu", etc.)
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
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

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

        model = AutoModelForCausalLM.from_pretrained(**kwargs)
        input_device = get_model_input_device(model)
        num_gpus = torch.cuda.device_count()
        print(f"Model loaded across {num_gpus} GPU(s), inputs go to {input_device}")
    else:
        # CPU fallback
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float32,
        )
        input_device = torch.device("cpu")

    model.eval()

    if token_ids is None:
        token_ids = list(range(len(tokenizer)))

    tokens_dafny = _dafny.SeqWithoutIsStrInference(
        [_dafny.Seq(tokenizer.decode([tid])) for tid in token_ids]
    )

    class HuggingFaceLM(VerifiedDecoderAgent.LM):
        """Wrapper that bridges HuggingFace models to the Dafny LM interface.

        Keeps a torch tensor (_logits_tensor) as the source of truth for
        logit values.  All extern methods (GenerateLogits, MaskTokensExcept,
        ChooseNextToken, ChooseNextTokenUnconstrained) and the non-extern
        helpers that read individual logits (IdToLogit, MaskToken, IsMasked)
        are overridden to operate on this tensor, giving O(1) masked_fill
        and argmax instead of O(vocab) Python loops.
        """

        def __init__(self, hf_model, hf_tokenizer, tokens, tids, dev):
            super().__init__()
            self.model = hf_model
            self.tokenizer = hf_tokenizer
            self._Tokens = tokens
            self._token_ids = tids
            self._input_device = dev
            self._max_input_len = get_max_input_length(hf_model, hf_tokenizer)
            self.instruction_text = ""

            n = len(tids)

            # Dafny Array kept at correct length for ValidTokensIdsLogits
            self.Logits = _dafny.Array(None, n)
            for i in range(n):
                self.Logits[i] = _dafny.BigRational(0)

            # Primary logit storage — all hot-path ops use this tensor
            self._logits_tensor = torch.zeros(n, dtype=torch.float32)

            # Precomputed gather index for extracting constrained logits
            self._token_ids_tensor = torch.tensor(tids, dtype=torch.long)

            # Store full logits for unconstrained generation
            self._full_logits = None

            # Precompute token string -> indices mapping for fast mask building
            self._token_str_to_indices = {}
            for i in range(n):
                token_str = self._to_str(tokens[i])
                self._token_str_to_indices.setdefault(token_str, []).append(i)

        # ── helpers ──────────────────────────────────────────────

        def _to_str(self, obj):
            """Convert a Dafny object (potentially a Seq of chars) to a Python string."""
            if isinstance(obj, str):
                return obj
            try:
                return "".join(obj[i] for i in range(len(obj)))
            except:
                return str(obj)

        # ── overridden non-extern methods (read/write self.Logits) ──

        def IdToLogit(self, id_):
            """Override: read from tensor instead of Dafny array."""
            return _dafny.BigRational(self._logits_tensor[id_].item())

        def MaskToken(self, token):
            """Override: write to tensor instead of Dafny array."""
            d_0_id_ = (self).TokenToId(token)
            self._logits_tensor[d_0_id_] = -1e9

        def IsMasked(self, token):
            """Override: read from tensor instead of Dafny array."""
            return self._logits_tensor[(self).TokenToId(token)].item() == -1e9

        # ── extern methods ───────────────────────────────────────

        def GenerateLogits(self, input_prefix):
            """Extern: compute logits for the next token given a prefix.

            Stores logits in _logits_tensor via a single torch.gather,
            replacing the O(n) BigRational construction loop.
            """
            import os
            debug = os.environ.get('CSD_MASK_DEBUG', '').lower() in ('1', 'true', 'yes')

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
                output = self.model(**inputs)
                logits = output.logits[0, -1, :].float().cpu()

            # Store full logits for unconstrained generation
            self._full_logits = logits

            # Single tensor gather — O(1) instead of O(n) BigRational loop
            self._logits_tensor = logits[self._token_ids_tensor]

        def ChooseNextToken(self):
            """Extern: return the token with the highest logit (constrained).

            Uses torch.argmax — O(1) instead of O(n) Python loop.
            """
            import os
            debug = os.environ.get('CSD_MASK_DEBUG', '').lower() in ('1', 'true', 'yes')

            best_idx = int(self._logits_tensor.argmax().item())
            chosen_token = self._Tokens[best_idx]

            if debug:
                try:
                    token_str = ''.join(chosen_token[i] for i in range(len(chosen_token)))
                except:
                    token_str = str(chosen_token)
                print(f"    [CHOOSE DEBUG] Best idx={best_idx}, logit={self._logits_tensor[best_idx].item():.2f}, token={repr(token_str)}")

            return chosen_token

        def ChooseNextTokenUnconstrained(self):
            """Extern: return the token with the highest logit from FULL vocabulary."""
            import os
            debug = os.environ.get('CSD_MASK_DEBUG', '').lower() in ('1', 'true', 'yes')

            if self._full_logits is None:
                raise RuntimeError("Must call GenerateLogits before ChooseNextTokenUnconstrained")
            best_idx = int(self._full_logits.argmax().item())
            token_text = self.tokenizer.decode([best_idx])

            if debug:
                print(f"    [UNCONSTRAINED DEBUG] chosen_token={repr(token_text)}")

            return _dafny.Seq(token_text)

        def MaskTokensExcept(self, valid_tokens, debug=False):
            """Extern: mask all tokens except those in valid_tokens.

            Builds a boolean accept mask from valid_tokens (O(|valid_tokens|)),
            then applies it with torch.masked_fill_ (single tensor op, O(1)),
            following syncode's SyncodeLogitsProcessor approach.

            Note: uses -1e9 (not -inf) to satisfy the Dafny invariant that
            all logits remain in [-1e9, 1e9].
            """
            import os
            debug = debug or os.environ.get('CSD_MASK_DEBUG', '').lower() in ('1', 'true', 'yes')

            # Build boolean accept mask — O(|valid_tokens|)
            accept_mask = torch.zeros(len(self._token_ids), dtype=torch.bool)
            for i in range(len(valid_tokens)):
                token_str = self._to_str(valid_tokens[i])
                indices = self._token_str_to_indices.get(token_str)
                if indices is not None:
                    for idx in indices:
                        accept_mask[idx] = True

            if torch.sum(accept_mask) == 0:
                # No valid tokens — skip masking to avoid killing all logits
                # (matches syncode's SyncodeLogitsProcessor fallback)
                if debug:
                    print("    [MASK DEBUG] WARNING: No acceptable tokens found, skipping masking.")
                return

            # Pad mask if logits tensor is longer (some models pad beyond vocab)
            if len(self._logits_tensor) > len(accept_mask):
                accept_mask = torch.cat((accept_mask, torch.zeros(len(self._logits_tensor) - len(accept_mask), dtype=torch.bool)))

            # Move mask to same device as logits, then apply
            self._logits_tensor.masked_fill_(~accept_mask.to(self._logits_tensor.device), -1e9)

            if debug:
                masked_count = int((~accept_mask).sum().item())
                print(f"    [MASK DEBUG] Masked {masked_count} tokens, {int(accept_mask.sum().item())} remain valid.")

    return HuggingFaceLM(model, tokenizer, tokens_dafny, token_ids, input_device)
