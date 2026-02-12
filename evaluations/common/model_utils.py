"""
Model loading and management utilities for CSD evaluation.

Provides optimized model loading for CUDA/CPU with proper device handling
for multi-GPU setups using accelerate's device_map="auto".
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
    vocab_size: int,
    VerifiedDecoderAgent,
    _dafny,
    token_ids=None
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

    Returns:
        A Dafny-compatible LM wrapper
    """
    print(f"Loading model: {model_name} on {device}... (FP16)")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Always use device_map="auto" for CUDA to leverage all available GPUs
    if device.startswith("cuda"):
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto",
        )
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
        from evaluations.common.token_selection import select_math_token_ids
        token_ids = select_math_token_ids(tokenizer, vocab_size)

    tokens_dafny = _dafny.SeqWithoutIsStrInference(
        [_dafny.Seq(tokenizer.decode([tid])) for tid in token_ids]
    )

    class HuggingFaceLM(VerifiedDecoderAgent.LM):
        """Wrapper that bridges HuggingFace models to the Dafny LM interface."""
        
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
            
            # Store full logits for unconstrained generation
            self._full_logits = logits

            for i, tid in enumerate(self._token_ids):
                self.Logits[i] = _dafny.BigRational(float(logits[tid].item()))

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
            best_idx = int(self._full_logits.argmax().item())
            token_text = self.tokenizer.decode([best_idx])
            
            if debug:
                print(f"    [UNCONSTRAINED DEBUG] chosen_token={repr(token_text)}")

            return _dafny.Seq(token_text)
        
        def MaskTokensExcept(self, valid_tokens, debug=False):
            """Mask all tokens except those in valid_tokens.
            
            This implementation follows the Dafny specification strictly.
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
