"""
GSM-Symbolic dataset loading utilities.

Handles loading the apple/GSM-Symbolic dataset from HuggingFace with various
configurations (main, p1, p2) and supports limiting/sampling for efficient evaluation.
"""

from __future__ import annotations

from typing import Optional


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
        from datasets import load_dataset
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
    
    try:
        ds = load_dataset("apple/GSM-Symbolic", name=config, split=split)
    except Exception as e:
        # Some datasets only have certain splits
        print(f"Failed to load split '{split}', trying 'test'...")
        try:
            ds = load_dataset("apple/GSM-Symbolic", name=config, split="test")
        except Exception:
            print(f"Failed to load 'test', trying 'train'...")
            ds = load_dataset("apple/GSM-Symbolic", name=config, split="train")

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
