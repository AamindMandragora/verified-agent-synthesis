"""Static SMILES prompt/grammar tasks copied from the CARS benchmark."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

SMILES_CLASSES: tuple[str, ...] = ("acrylates", "chain_extenders", "isocyanates")
DATA_DIR = Path(
    os.environ.get("SMILES_DATA_DIR", str(Path(__file__).resolve().parent / "data"))
).expanduser()
GRAMMAR_DIR = Path(
    os.environ.get("SMILES_GRAMMAR_DIR", str(Path(__file__).resolve().parents[2] / "grammars"))
).expanduser()


def normalize_smiles_classes(
    classes: Sequence[str] | str | None,
    *,
    dedupe: bool = False,
    require_non_empty: bool = False,
) -> list[str]:
    """Normalize SMILES class names with validation against known classes."""
    if classes is None:
        selected = list(SMILES_CLASSES)
    elif isinstance(classes, str):
        raw = [part.strip() for part in classes.split(",")]
        selected = [part for part in raw if part]
    else:
        selected = [str(part).strip() for part in classes if str(part).strip()]

    unknown = sorted(set(selected) - set(SMILES_CLASSES))
    if unknown:
        raise ValueError(f"Unknown SMILES class(es): {unknown}. Expected one of {SMILES_CLASSES}.")

    if dedupe:
        deduped: list[str] = []
        seen: set[str] = set()
        for class_name in selected:
            if class_name in seen:
                continue
            deduped.append(class_name)
            seen.add(class_name)
        selected = deduped

    if require_non_empty and not selected:
        raise ValueError("At least one SMILES class is required.")

    return selected


def extract_prompt_exemplars(prompt: str, grammar_text: str) -> list[str]:
    """Every molecule this prompt puts in front of the model.

    A baseline that hands back a molecule the prompt already showed it is repeating the
    prompt, not answering, so the scorer has to know about all of them. Two sources. The
    labelled ``Molecule: X`` few-shot lines, and every other whitespace-separated word in
    the prompt that both the class grammar and RDKit accept as a molecule. The second
    source exists because the instruction line the harness appends carries a molecule
    inline, as "(example: CC(=O)OC=C)"; scanning words rather than matching that exact
    wording keeps the list correct if the wording ever moves.

    Scanning cannot forbid an answer the prompt never showed unless ordinary English in
    the prompt happens to parse as a molecule. Checked on all three shipped prompts on
    2026-09-20: the scan adds exactly ``CC(=O)OC=C`` and nothing else.
    """
    from synthesis.evaluate.benchmarks.smiles.metrics import grammar_valid, rdkit_valid

    exemplars: list[str] = []
    seen: set[str] = set()

    def _keep(value: str) -> None:
        if value and value not in seen:
            seen.add(value)
            exemplars.append(value)

    for line in prompt.splitlines():
        line = line.strip()
        if line.startswith("Molecule:"):
            _keep(line.split(":", 1)[1].strip())
    for word in prompt.split():
        for candidate in (word.strip(".,;:!?"), word.strip(".,;:!?()")):
            if candidate and grammar_valid(candidate, grammar_text) and rdkit_valid(candidate):
                _keep(candidate)
    return exemplars


@lru_cache(maxsize=None)
def get_smiles_task(class_name: str) -> Dict[str, Any]:
    class_name = class_name.strip()
    if class_name not in SMILES_CLASSES:
        raise ValueError(f"Unknown SMILES class: {class_name}")
    grammar_path = GRAMMAR_DIR / f"smiles_{class_name}.lark"
    prompt_path = DATA_DIR / f"{class_name}.txt"
    grammar_text = grammar_path.read_text()
    prompt = prompt_path.read_text()
    return {
        "class_name": class_name,
        "question": class_name,
        "prompt": prompt,
        "grammar_path": grammar_path,
        "grammar_text": grammar_text,
        "prompt_path": prompt_path,
        "prompt_exemplars": extract_prompt_exemplars(prompt, grammar_text),
    }


def load_smiles(
    classes: Sequence[str] | str | None = None,
    samples_per_class: int = 1,
    *,
    split_name: str,
    trial: int,
    first_draw_index: int = 0,
) -> List[Dict[str, Any]]:
    """Return ``samples_per_class`` draws per class from one side of the seed split.

    Each row is the class task plus which draw it is: ``draw_index`` (0-based,
    counted from the start of the split, per class), ``draw_seed`` (the sampler
    seed for that draw, see seed_split.py), ``smiles_split_name`` and
    ``smiles_trial`` (1, 2 or 3: each trial is its own block of seeds). There is no
    default split or trial: SMILES train and test are different seed ranges, and picking one
    silently is how a test-side number ends up scored against a train-side bar.
    ``first_draw_index`` skips the first draws of the split (a later shard or the
    confirmation slice), so it never replays draw 0.
    """
    from synthesis.evaluate.benchmarks.smiles.seed_split import smiles_draw_seed

    if samples_per_class < 1:
        raise ValueError("samples_per_class must be >= 1")
    rows: list[dict[str, Any]] = []
    for class_name in normalize_smiles_classes(classes):
        task = get_smiles_task(class_name)
        for draw_index in range(first_draw_index, first_draw_index + samples_per_class):
            row = dict(task)
            row["attempt_index"] = draw_index
            row["draw_index"] = draw_index
            row["draw_seed"] = smiles_draw_seed(split_name, trial, draw_index)
            row["smiles_split_name"] = split_name
            row["smiles_trial"] = trial
            rows.append(row)
    return rows
