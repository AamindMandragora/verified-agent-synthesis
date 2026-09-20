"""SMILES validity, uniqueness, and target-class membership metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from itertools import combinations
from typing import Any, Dict, Iterable, Optional, Sequence

from lark import Lark
from lark.exceptions import LarkError

try:  # Optional on focal; metrics report when it is absent.
    from rdkit import Chem, DataStructs  # type: ignore
    from rdkit.Chem import AllChem  # type: ignore
except Exception:  # pragma: no cover - depends on host environment
    Chem = None  # type: ignore
    DataStructs = None  # type: ignore
    AllChem = None  # type: ignore

EOS_MARKERS = ("<|im_end|>", "<|eot_id|>", "<|endoftext|>")
CLASS_MOTIFS: dict[str, tuple[str, ...]] = {
    "acrylates": (
        "C=CC(=O)O",
        "C=CC(O)=O",
        "C(=C)C(=O)O",
        "C(=C)C(O)=O",
        "OC(=O)C=C",
        "O=C(O)C=C",
        "OC(=O)C(=C)",
        "O=C(O)C(=C)",
    ),
    "chain_extenders": ("OC", "CO", "N"),
    "isocyanates": ("N=C=O", "O=C=N"),
}


def clean_smiles_output(output: str | None) -> str:
    if output is None:
        return ""
    text = str(output).strip()
    for marker in EOS_MARKERS:
        text = text.replace(marker, "")
    text = text.replace("<<", "").replace(">>", "")
    # Safety net: remove orphan angle-bracket control tokens that can leak from
    # delimiter generation into the extracted molecule string.
    text = text.replace("<", "").replace(">", "")
    text = text.split("\n\n", 1)[0].strip()
    if "Molecule:" in text:
        text = text.rsplit("Molecule:", 1)[-1].strip()
    return text.splitlines()[0].strip() if text else ""


@lru_cache(maxsize=None)
def _lark_parser(grammar_text: str) -> Lark:
    return Lark(grammar_text, start="start", parser="lalr")


def grammar_valid(smiles: str, grammar_text: str) -> bool:
    if not smiles:
        return False
    try:
        _lark_parser(grammar_text).parse(smiles)
        return True
    except LarkError:
        return False


def rdkit_available() -> bool:
    # Core availability for syntax validity checks (MolFromSmiles).
    return Chem is not None


def _rdkit_similarity_available() -> bool:
    # Extended availability for fingerprint/diversity metrics.
    return Chem is not None and AllChem is not None and DataStructs is not None


def rdkit_valid(smiles: str) -> bool | None:
    if Chem is None:
        return None
    if not smiles:
        return False
    try:
        return Chem.MolFromSmiles(smiles) is not None
    except Exception:
        return False


def target_class_membership(class_name: str, smiles: str) -> bool:
    motifs = CLASS_MOTIFS.get(class_name)
    if not motifs or not smiles:
        return False
    return any(motif in smiles for motif in motifs)


def is_prompt_exemplar(smiles: str, prompt_exemplars: Sequence[str]) -> bool:
    return smiles in set(prompt_exemplars)


def evaluate_smiles_output(
    class_name: str,
    output: str | None,
    grammar_text: str,
    prompt_exemplars: Sequence[str],
    require_rdkit: bool = True,
) -> Dict[str, Any]:
    smiles = clean_smiles_output(output)
    grammar_ok = grammar_valid(smiles, grammar_text)
    rdkit_ok = rdkit_valid(smiles)
    # RDKit validates chemistry when installed; when it is unavailable, fall
    # back to the project grammar so syntax rate remains measurable.
    if require_rdkit and rdkit_ok is not None:
        syntax_ok = bool(grammar_ok and rdkit_ok)
    else:
        syntax_ok = bool(grammar_ok)
    membership_ok = target_class_membership(class_name, smiles)
    exemplar = is_prompt_exemplar(smiles, prompt_exemplars)
    unique_valid_candidate = bool(smiles and syntax_ok and membership_ok and not exemplar)
    valid_class_membership = bool(syntax_ok and membership_ok)
    return {
        "smiles": smiles,
        "grammar_valid": grammar_ok,
        "rdkit_available": rdkit_available(),
        "rdkit_valid": rdkit_ok,
        "syntax_valid": syntax_ok,
        "class_membership": membership_ok,
        # Accuracy should be computed over syntactically valid molecules only.
        # This flag is the numerator for that conditional class-membership rate.
        "valid_class_membership": valid_class_membership,
        "accuracy_applicable": syntax_ok,
        "is_prompt_exemplar": exemplar,
        "unique_valid_candidate": unique_valid_candidate,
    }


def _morgan_fingerprint(smiles: str):
    if not _rdkit_similarity_available() or not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        generator = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
        return generator.GetFingerprint(mol)
    except Exception:
        return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)


def average_pairwise_tanimoto(smiles_values: Sequence[str]) -> Optional[float]:
    fps = [_morgan_fingerprint(s) for s in smiles_values]
    fps = [fp for fp in fps if fp is not None]
    if len(fps) < 2:
        return None
    distances = []
    for fp_a, fp_b in combinations(fps, 2):
        sim = DataStructs.TanimotoSimilarity(fp_a, fp_b)
        distances.append(1.0 - float(sim))
    if not distances:
        return None
    return sum(distances) / len(distances)


def retrosynthesis_score(smiles: str) -> Optional[float]:
    """
    Optional RetroStar hook.

    Returns None when RetroStar isn't installed/configured in this environment.
    """
    try:
        from retrostar.api import score_smiles  # type: ignore
    except Exception:
        return None
    try:
        value = score_smiles(smiles)
        return float(value) if value is not None else None
    except Exception:
        return None


def smiles_trial_metrics(
    samples: Sequence[Dict[str, Any]],
    target_unique_valid: int = 100,
    sample_cap: int = 1000,
) -> Dict[str, Any]:
    """
    Compute paper-aligned molecular generation metrics over one trial.
    """
    if not samples:
        return {
            "validity_rdkit": 0.0,
            "diversity_tanimoto": None,
            "retro_score": None,
            "membership": 0.0,
            "samples_to_target_unique_valid": sample_cap,
            "unique_valid_count": 0,
            "sample_count": 0,
            "retro_scored_count": 0,
            "retro_available": False,
        }

    considered = list(samples[:sample_cap])
    validity_values: list[bool] = []
    membership_values: list[bool] = []
    retro_values: list[float] = []

    unique_valid_ordered: list[str] = []
    unique_valid_seen: set[str] = set()
    samples_to_target = sample_cap

    for idx, sample in enumerate(considered, start=1):
        smiles_eval = sample.get("smiles_eval") or {}
        smiles = (smiles_eval.get("smiles") or "").strip()
        rdkit_ok = smiles_eval.get("rdkit_valid")
        validity_ok = bool(rdkit_ok is True)
        validity_values.append(validity_ok)
        membership_values.append(bool(smiles_eval.get("class_membership")))

        if smiles and bool(smiles_eval.get("unique_valid_candidate")) and smiles not in unique_valid_seen:
            unique_valid_seen.add(smiles)
            unique_valid_ordered.append(smiles)
            if len(unique_valid_ordered) >= target_unique_valid and samples_to_target == sample_cap:
                samples_to_target = idx

        retro = retrosynthesis_score(smiles) if smiles else None
        if retro is not None:
            retro_values.append(retro)

    validity = sum(1 for v in validity_values if v) / max(1, len(validity_values))
    membership = sum(1 for v in membership_values if v) / max(1, len(membership_values))
    diversity = average_pairwise_tanimoto(unique_valid_ordered)
    retro_mean = (sum(retro_values) / len(retro_values)) if retro_values else None

    return {
        "validity_rdkit": validity,
        "diversity_tanimoto": diversity,
        "retro_score": retro_mean,
        "membership": membership,
        "samples_to_target_unique_valid": samples_to_target,
        "unique_valid_count": len(unique_valid_ordered),
        "sample_count": len(considered),
        "retro_scored_count": len(retro_values),
        "retro_available": bool(retro_values),
    }


@dataclass
class SmilesMetrics:
    n: int = 0
    num_correct: int = 0
    num_contains_delimiters: int = 0
    num_syntax_valid: int = 0
    total_time: float = 0.0
    max_token_count: int = 0
    _recent_outputs: list[dict[str, Any]] = field(default_factory=list)

    def update(
        self,
        *,
        is_correct: bool,
        contains_delimiters: bool,
        token_count: int,
        time_seconds: float,
        constrained_segments: Sequence[tuple[str, bool]],
    ) -> None:
        syntax_valid = bool(constrained_segments) and all(is_valid for _, is_valid in constrained_segments)
        self.n += 1
        self.num_correct += int(is_correct)
        self.num_contains_delimiters += int(contains_delimiters)
        self.num_syntax_valid += int(syntax_valid)
        self.total_time += float(time_seconds)
        self.max_token_count = max(self.max_token_count, int(token_count))
        self._recent_outputs.append(
            {
                "is_correct": is_correct,
                "contains_delimiters": contains_delimiters,
                "token_count": token_count,
                "time_seconds": time_seconds,
                "syntax_valid": syntax_valid,
            }
        )

    @property
    def accuracy(self) -> float:
        return self.num_correct / max(1, self.n)

    @property
    def contains_delimiters_rate(self) -> float:
        return self.num_contains_delimiters / max(1, self.n)

    @property
    def syntax_rate(self) -> float:
        return self.num_syntax_valid / max(1, self.n)

    def summary(self) -> str:
        return "\n".join(
            [
                f"Examples: {self.n}",
                f"Accuracy: {self.accuracy:.1%} ({self.num_correct}/{self.n})",
                f"Contains delimiters: {self.contains_delimiters_rate:.1%}",
                f"Syntax rate: {self.syntax_rate:.1%}",
                f"Total time: {self.total_time:.2f}s",
                f"Slowest token count: {self.max_token_count}",
            ]
        )
