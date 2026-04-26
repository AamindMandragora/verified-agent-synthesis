"""
GSM-Symbolic dataset loading utilities.

Handles loading the apple/GSM-Symbolic dataset from HuggingFace with various
configurations (main, p1, p2) and supports limiting/sampling for efficient evaluation.
"""

from __future__ import annotations

import os
from typing import Optional


def _datasets_offline_enabled() -> bool:
    """True when dataset loading should stay offline / cache-only."""
    return any(os.environ.get(name, "").strip() in {"1", "true", "True"} for name in (
        "HF_DATASETS_OFFLINE",
        "HF_HUB_OFFLINE",
    ))


def _is_hf_connection_error(exc: Exception) -> bool:
    """Best-effort detection for HF dataset connectivity failures."""
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


def load_gsm_symbolic(
    config: str = "main",
    split: str = "test",
    limit: Optional[int] = None,
    random_sample: bool = False
):
    """
    Load GSM-Symbolic dataset from HuggingFace.

    Args:
        config: Dataset configuration - "main", "p1", or "p2"
                - "main": Standard difficulty
                - "p1": Perturbation level 1
                - "p2": Perturbation level 2
        split: Dataset split (usually "test")
        limit: Optional limit on number of examples to load (for efficiency)
        random_sample: If True and limit is set, randomly sample examples 
                       instead of taking first N

    Returns:
        HuggingFace dataset object (limited if limit is specified)
        
    Raises:
        RuntimeError: If the datasets library is not installed
        ValueError: If config is not one of the valid options
    """
    try:
        from datasets import DownloadConfig, load_dataset
    except ImportError as e:
        raise RuntimeError(
            "Missing dependency `datasets`. Install with: pip install datasets"
        ) from e

    valid_configs = ["main", "p1", "p2"]
    if config not in valid_configs:
        raise ValueError(f"Config must be one of {valid_configs}, got: {config}")

    sample_str = " (random sample)" if random_sample and limit else ""
    limit_str = f" (limit={limit})" if limit else ""
    print(f"Loading GSM-Symbolic dataset (config={config}, split={split}{limit_str}{sample_str})...")

    def _load_split(split_name: str, *, local_only: bool) -> any:
        kwargs = {
            "path": "apple/GSM-Symbolic",
            "name": config,
            "split": split_name,
        }
        if local_only:
            kwargs["download_config"] = DownloadConfig(local_files_only=True)
        return load_dataset(**kwargs)

    offline_only = _datasets_offline_enabled()

    def _try_sequence(local_only: bool):
        try:
            return _load_split(split, local_only=local_only)
        except Exception:
            print(f"Failed to load split '{split}', trying 'test'...")
            try:
                return _load_split("test", local_only=local_only)
            except Exception:
                print(f"Failed to load 'test', trying 'train'...")
                return _load_split("train", local_only=local_only)

    try:
        ds = _try_sequence(local_only=offline_only)
    except Exception as e:
        if offline_only or not _is_hf_connection_error(e):
            raise
        print("  HuggingFace dataset lookup failed; retrying from local cache only.")
        ds = _try_sequence(local_only=True)

    # Apply limit if specified
    if limit is not None and limit > 0:
        if random_sample:
            # Random sample for diverse evaluation
            import random
            random.seed(42)  # Fixed seed for reproducibility
            indices = random.sample(range(len(ds)), min(limit, len(ds)))
            ds = ds.select(indices)
        else:
            # Sequential (first N examples)
            ds = ds.select(range(min(limit, len(ds))))

    print(f"Loaded {len(ds)} examples")
    return ds
