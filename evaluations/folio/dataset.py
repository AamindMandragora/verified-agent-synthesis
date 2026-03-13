"""
Dataset loading utilities for FOLIO evaluation.

FOLIO is a first-order logic reasoning dataset from Yale NLP.
Dataset: yale-nlp/FOLIO
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass


@dataclass
class FOLIOExample:
    """A single FOLIO example."""
    id: str
    premises: str
    conclusion: str
    label: str  # 'True', 'False', or 'Uncertain'
    # Optional fields that may be present in annotated versions
    fol_premises: Optional[List[str]] = None
    fol_conclusion: Optional[str] = None
    
    @property
    def problem(self) -> str:
        """Return the premises as the problem description."""
        return self.premises
    
    @property
    def question(self) -> str:
        """Format the question asking about the conclusion."""
        return f"Based on the above information, is the following statement true, false, or uncertain? {self.conclusion}"


def load_folio(
    split: str = "validation",
    num_samples: Optional[int] = None,
    seed: Optional[int] = None,
) -> List[FOLIOExample]:
    """
    Load the FOLIO dataset from HuggingFace.
    
    Args:
        split: Dataset split to load ('train', 'validation', 'test')
        num_samples: Number of samples to load (None for all)
        seed: Random seed for sampling (if num_samples is specified)
    
    Returns:
        List of FOLIOExample objects
    
    Note:
        The FOLIO dataset (yale-nlp/FOLIO) requires authentication.
        You may need to run `huggingface-cli login` first.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Please install datasets: pip install datasets")
    
    # Load the dataset
    # Note: FOLIO dataset may require authentication
    try:
        dataset = load_dataset("yale-nlp/FOLIO", split=split)
    except Exception as e:
        # Try alternative loading methods or provide helpful error
        raise RuntimeError(
            f"Failed to load FOLIO dataset: {e}\n"
            "The FOLIO dataset (yale-nlp/FOLIO) is gated and requires:\n"
            "1. Accept the dataset terms at https://huggingface.co/datasets/yale-nlp/FOLIO (log in, then click to agree)\n"
            "2. A token with read access to *gated* repos (not 'namespace only') — create at https://huggingface.co/settings/tokens\n"
            "3. Login: `huggingface-cli login` or set HF_TOKEN\n"
            "Alternatively, use load_folio_from_json(path_to_local_folio.json) with a local FOLIO JSON file."
        )
    
    # Sample if requested
    if num_samples is not None and num_samples < len(dataset):
        if seed is not None:
            dataset = dataset.shuffle(seed=seed)
        dataset = dataset.select(range(num_samples))
    
    # Convert to FOLIOExample objects
    examples = []
    for idx, item in enumerate(dataset):
        example = FOLIOExample(
            id=str(item.get("id", idx)),
            premises=item["premises"],
            conclusion=item["conclusion"],
            label=item["label"],
            fol_premises=item.get("premises-FOL"),
            fol_conclusion=item.get("conclusion-FOL"),
        )
        examples.append(example)
    
    return examples


def load_folio_from_json(
    json_path: str,
    num_samples: Optional[int] = None,
    seed: Optional[int] = None,
) -> List[FOLIOExample]:
    """
    Load FOLIO dataset from a local JSON file.
    
    This is an alternative to HuggingFace loading for offline use.
    
    Args:
        json_path: Path to the JSON file
        num_samples: Number of samples to load (None for all)
        seed: Random seed for sampling (if num_samples is specified)
    
    Returns:
        List of FOLIOExample objects
    """
    import json
    import random
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Handle both list and dict formats
    if isinstance(data, dict):
        data = list(data.values())
    
    # Sample if requested
    if num_samples is not None and num_samples < len(data):
        if seed is not None:
            random.seed(seed)
        data = random.sample(data, num_samples)
    
    # Convert to FOLIOExample objects
    examples = []
    for idx, item in enumerate(data):
        example = FOLIOExample(
            id=str(item.get("id", idx)),
            premises=item["premises"],
            conclusion=item["conclusion"],
            label=item["label"],
            fol_premises=item.get("premises-FOL"),
            fol_conclusion=item.get("conclusion-FOL"),
        )
        examples.append(example)
    
    return examples


def create_synthetic_folio_examples() -> List[FOLIOExample]:
    """
    Create synthetic FOLIO examples for testing.
    
    These examples are simpler versions of real FOLIO problems
    useful for debugging and testing the pipeline.
    """
    examples = [
        FOLIOExample(
            id="synthetic_1",
            premises="All cats are mammals. All mammals are animals. Felix is a cat.",
            conclusion="Felix is an animal.",
            label="True",
        ),
        FOLIOExample(
            id="synthetic_2", 
            premises="All birds can fly. Penguins are birds.",
            conclusion="Penguins can fly.",
            label="True",  # Note: This is logically true given the premises, even if factually false
        ),
        FOLIOExample(
            id="synthetic_3",
            premises="Some students are athletes. All athletes exercise regularly. John is a student.",
            conclusion="John exercises regularly.",
            label="Uncertain",
        ),
        FOLIOExample(
            id="synthetic_4",
            premises="No reptiles are mammals. All snakes are reptiles. All dogs are mammals.",
            conclusion="Some snakes are dogs.",
            label="False",
        ),
        FOLIOExample(
            id="synthetic_5",
            premises="Either it is raining or it is sunny. It is not sunny.",
            conclusion="It is raining.",
            label="True",
        ),
    ]
    return examples


# Label normalization utilities
LABEL_MAP = {
    "true": "True",
    "True": "True",
    "TRUE": "True",
    "false": "False", 
    "False": "False",
    "FALSE": "False",
    "uncertain": "Uncertain",
    "Uncertain": "Uncertain",
    "UNCERTAIN": "Uncertain",
    "unknown": "Uncertain",
    "Unknown": "Uncertain",
}


def normalize_label(label: str) -> str:
    """Normalize a label to standard format (True/False/Uncertain)."""
    label = label.strip()
    return LABEL_MAP.get(label, label)
