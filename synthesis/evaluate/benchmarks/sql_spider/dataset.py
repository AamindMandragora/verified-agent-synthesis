"""
Spider dataset loading utilities.

Loads text-to-SQL problems in the same shape itergen uses. Each row has:
    - db_id: database identifier (directory under the Spider databases root)
    - question: natural-language question
    - query: gold SQL (from local dev.json / dev_gold.sql)
    - db_info: schema string of the form
        "# t1 ( c1 , c2 , ... )\n# t2 ( ... )\n# t1.c1 = t2.c2\n..."
    - prompt: pre-formatted prompt body (db_id/db_info/question/SQL:)

Priority:
    1. HuggingFace `richardr1126/spider-context-validation` (matches itergen).
    2. Local fallback at $SPIDER_DATA_DIR (default: ~/spider_data/spider_data)
       using dev.json, dev_gold.sql, tables.json.
"""

from __future__ import annotations

import json
import os
import logging
import random
import sqlite3
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from synthesis.evaluate.benchmarks.split_utils import (
    proportional_allocations,
    split_manifest_metadata,
    stratified_sample_indices,
)
from synthesis.project_defaults import default_spider_data_dir

logger = logging.getLogger(__name__)

DEFAULT_SPIDER_DIR = Path(
    default_spider_data_dir()
)
PROJECT_ROOT = Path(__file__).parent.parent.parent

SPIDER_DIFFICULTIES: tuple[str, ...] = ("easy", "medium", "hard", "extra")


def _vendored_spider_eval_dir() -> Path:
    override = os.environ.get("SPIDER_EVAL_DIR")
    candidates = []
    if override:
        override_path = Path(override).expanduser()
        candidates.append(override_path.parent if override_path.name == "evaluation.py" else override_path)
    candidates.extend([
        PROJECT_ROOT / "syncode" / "syncode" / "utils" / "sql_spider_eval",
        default_spider_data_dir(),
        Path.home() / "CRANE" / "src" / "crane" / "iter_syncode" / "utils" / "sql_spider_eval",
        Path.home() / "itergen" / "itergen" / "syncode" / "syncode" / "utils" / "sql_spider_eval",
    ])
    for candidate in candidates:
        tables_json = candidate / "evaluation_examples" / "examples" / "tables.json"
        databases = candidate / "databases"
        if tables_json.exists() and databases.exists():
            return candidate
    return candidates[0]


def _build_db_info_from_tables_entry(entry: Dict[str, Any]) -> str:
    """Render a tables.json entry into the itergen-style db_info string."""
    table_names = entry.get("table_names_original") or entry.get("table_names") or []
    column_names = entry.get("column_names_original") or entry.get("column_names") or []
    foreign_keys = entry.get("foreign_keys") or []

    table_to_cols: Dict[int, List[str]] = {i: [] for i in range(len(table_names))}
    for col in column_names:
        try:
            tbl_idx, col_name = col
        except ValueError:
            continue
        if tbl_idx is None or tbl_idx < 0:
            continue
        if tbl_idx in table_to_cols and col_name != "*":
            table_to_cols[tbl_idx].append(str(col_name))

    lines: List[str] = []
    for tbl_idx, tbl_name in enumerate(table_names):
        cols = table_to_cols.get(tbl_idx, [])
        if not cols:
            continue
        lines.append(f"# {tbl_name} ( {' , '.join(cols)} )")

    for fk in foreign_keys:
        try:
            src_idx, dst_idx = fk
        except (TypeError, ValueError):
            continue
        try:
            src_t, src_c = column_names[src_idx]
            dst_t, dst_c = column_names[dst_idx]
        except (IndexError, ValueError):
            continue
        if src_t < 0 or dst_t < 0:
            continue
        lines.append(
            f"# {table_names[src_t]}.{src_c} = {table_names[dst_t]}.{dst_c}"
        )

    return "\n".join(lines) + ("\n" if lines else "")


@lru_cache(maxsize=None)
def _load_tables_map(tables_path: str) -> Dict[str, Dict[str, Any]]:
    with open(tables_path) as f:
        data = json.load(f)
    return {entry["db_id"]: entry for entry in data}


def _format_prompt_body(db_id: str, db_info: str, question: str) -> str:
    """Match syncode Dataset('spider')'s prompt shape."""
    return f"db_id: {db_id}\ndb_info: {db_info}\nquestion: {question}\nSQL:"


def _load_local_spider(spider_dir: Path) -> List[Dict[str, Any]]:
    dev_path = spider_dir / "dev.json"
    gold_path = spider_dir / "dev_gold.sql"
    tables_path = spider_dir / "tables.json"
    if not (dev_path.exists() and tables_path.exists()):
        raise FileNotFoundError(
            f"Spider local files not found under {spider_dir}: "
            "expected dev.json and tables.json"
        )

    dev = json.loads(dev_path.read_text())
    tables = _load_tables_map(str(tables_path))

    # Gold SQL file is parallel to dev.json: one line per dev example, "<sql>\t<db_id>"
    gold_lines: List[str] = []
    if gold_path.exists():
        gold_lines = [
            line.rstrip("\n")
            for line in gold_path.read_text().splitlines()
            if line.strip()
        ]

    rows: List[Dict[str, Any]] = []
    for i, ex in enumerate(dev):
        db_id = ex["db_id"]
        question = ex.get("question", "")
        query = ex.get("query", "")
        gold_db_id = db_id
        if i < len(gold_lines) and "\t" in gold_lines[i]:
            gold_sql, gold_db_id = gold_lines[i].split("\t", 1)
            if gold_sql.strip():
                query = gold_sql.strip()
            gold_db_id = gold_db_id.strip()

        entry = tables.get(db_id)
        db_info = _build_db_info_from_tables_entry(entry) if entry else ""

        rows.append(
            {
                "db_id": gold_db_id or db_id,
                "question": question,
                "query": query,
                "db_info": db_info,
                "prompt": _format_prompt_body(db_id, db_info, question),
            }
        )
    return rows


def _load_hf_spider() -> List[Dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise RuntimeError(
            "Missing dependency `datasets`. Install with: pip install datasets"
        ) from e

    ds = load_dataset("richardr1126/spider-context-validation", split="validation")
    rows: List[Dict[str, Any]] = []
    for ex in ds:
        db_id = ex.get("db_id", "")
        db_info = ex.get("db_info", "")
        question = ex.get("question", "")
        # HF split carries "response" as the gold SQL; fall back to legacy keys.
        query = ex.get("ground_truth") or ex.get("response") or ex.get("query") or ex.get("sql") or ""
        rows.append(
            {
                "db_id": db_id,
                "question": question,
                "query": query,
                "db_info": db_info,
                "prompt": _format_prompt_body(db_id, db_info, question),
            }
        )
    return rows


def load_spider(
    source: str = "auto",
    spider_dir: Optional[Path | str] = None,
    limit: Optional[int] = None,
    random_sample: bool = False,
    seed: Optional[int] = None,
    indices: Optional[Sequence[int]] = None,
) -> List[Dict[str, Any]]:
    """
    Load the Spider dev split.

    Args:
        source: "hf", "local", or "auto" (tries HF first, then local).
        spider_dir: override local Spider directory (defaults to SPIDER_DATA_DIR env
                    or ~/spider_data/spider_data).
        limit: maximum number of rows to return.
        random_sample: if True and limit is set, pick random indices.
        seed: RNG seed when random_sample is True.
        indices: Optional explicit row indices to select before applying limit.

    Returns:
        list of dicts (see module docstring).
    """
    spider_dir = Path(spider_dir) if spider_dir is not None else DEFAULT_SPIDER_DIR

    rows: List[Dict[str, Any]] = []
    errors: List[str] = []

    if source in ("hf", "auto"):
        try:
            rows = _load_hf_spider()
            print(f"Loaded {len(rows)} Spider examples from HuggingFace")
        except Exception as e:
            errors.append(f"HF load failed: {e}")
            if source == "hf":
                raise

    if not rows and source in ("local", "auto"):
        try:
            rows = _load_local_spider(spider_dir)
            print(f"Loaded {len(rows)} Spider examples from {spider_dir}")
        except Exception as e:
            errors.append(f"local load failed: {e}")
            if source == "local":
                raise

    if not rows:
        raise RuntimeError("Failed to load Spider: " + "; ".join(errors))

    if indices is not None:
        selected: List[Dict[str, Any]] = []
        for idx in indices:
            if idx < 0 or idx >= len(rows):
                raise IndexError(
                    f"Spider index {idx} is out of range for {len(rows)} loaded rows"
                )
            row = dict(rows[idx])
            row["spider_source_index"] = idx
            selected.append(row)
        rows = selected

    if limit is not None and limit > 0:
        if random_sample:
            rng = random.Random(seed) if seed is not None else random
            indices = rng.sample(range(len(rows)), min(limit, len(rows)))
            rows = [rows[i] for i in indices]
        else:
            rows = rows[: min(limit, len(rows))]

    return rows


def _load_spider_evaluator():
    import sys

    eval_dir = str(_vendored_spider_eval_dir())
    if eval_dir not in sys.path:
        sys.path.insert(0, eval_dir)
    from evaluation import Evaluator, get_sql
    from process_sql import Schema, get_schema

    return Evaluator(), get_sql, Schema, get_schema


def infer_spider_difficulty_labels(
    *,
    source: str = "auto",
    spider_dir: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """Infer official Spider easy/medium/hard/extra labels for loaded dev rows."""
    rows = load_spider(source=source, spider_dir=spider_dir)
    evaluator, get_sql, Schema, get_schema = _load_spider_evaluator()
    labels: Dict[int, str] = {}
    for idx, row in enumerate(rows):
        db_id = row.get("db_id", "")
        db_path = default_db_dir() / db_id / f"{db_id}.sqlite"
        try:
            schema = Schema(get_schema(str(db_path)))
            parsed = get_sql(schema, row.get("query") or "")
            labels[idx] = evaluator.eval_hardness(parsed)
        except Exception:
            labels[idx] = "unknown"

    composition = dict(sorted(Counter(labels.values()).items()))
    return {
        "difficulty_source": "spider_official_eval_hardness",
        "difficulty_order": list(SPIDER_DIFFICULTIES),
        "difficulty_by_index": {str(idx): label for idx, label in sorted(labels.items())},
        "difficulty_composition": composition,
        "total_examples": len(rows),
    }


def make_spider_proportional_train_test_split(
    *,
    source: str = "auto",
    spider_dir: Optional[Path | str] = None,
    train_size: int = 300,
    test_size: int = 300,
    seed: int = 334,
) -> Dict[str, Any]:
    """Create a disjoint train/test split with benchmark-proportional hardness mix.

    Defaults encode the canonical Spider split used for the IterGen comparison:
    300 train / 300 held-out at seed 334. These exact values reproduce the
    committed manifest IterGen was scored against
    (spider_dev_proportional_300x300_seed334).
    """
    if train_size < 0 or test_size < 0:
        raise ValueError("train_size and test_size must be non-negative")
    if train_size + test_size <= 0:
        raise ValueError("train_size + test_size must be positive")

    rows = load_spider(source=source, spider_dir=spider_dir)
    difficulty_info = infer_spider_difficulty_labels(source=source, spider_dir=spider_dir)
    labels = {
        int(idx): label
        for idx, label in difficulty_info["difficulty_by_index"].items()
        if label in SPIDER_DIFFICULTIES
    }
    unknown = len(rows) - len(labels)
    if unknown:
        raise ValueError(
            f"Could not assign Spider hardness to {unknown} of {len(rows)} examples"
        )

    selected = stratified_sample_indices(
        labels,
        split_sizes={"train": train_size, "test": test_size},
        difficulties=SPIDER_DIFFICULTIES,
        seed=seed,
    )
    population = {
        difficulty: len([idx for idx, label in labels.items() if label == difficulty])
        for difficulty in SPIDER_DIFFICULTIES
    }
    manifest = split_manifest_metadata(
        seed=seed,
        split_strategy="stratified_proportional",
        labels_by_index=labels,
        split_sizes={"train": train_size, "test": test_size},
        selected=selected,
        extra={
            "source": source,
            "spider_dir": str(spider_dir or DEFAULT_SPIDER_DIR),
            "train_size_requested": train_size,
            "test_size_requested": test_size,
            "train_allocations_requested": proportional_allocations(
                train_size, population, order=SPIDER_DIFFICULTIES
            ),
            "test_allocations_requested": proportional_allocations(
                test_size, population, order=SPIDER_DIFFICULTIES
            ),
            **difficulty_info,
        },
    )
    manifest["eval_indices"] = list(manifest["test_indices"])
    return manifest


def make_spider_train_test_split(
    total_examples: int,
    train_size: int = 300,
    test_size: int = 300,
    seed: int = 123,
) -> Dict[str, Any]:
    """Create a deterministic non-overlapping Spider train/test index split.

    Legacy plain-random ("random_non_overlapping") path, not used for the IterGen
    comparison. Sizes default to 300/300 for consistency with the proportional path;
    seed stays 123 because this draw differs from the seed-334 proportional split and
    does not reproduce IterGen's held-out 300.
    """
    if total_examples <= 0:
        raise ValueError("total_examples must be positive")
    if train_size <= 0 or test_size <= 0:
        raise ValueError("train_size and test_size must be positive")
    if train_size + test_size > total_examples:
        raise ValueError(
            f"Requested train_size + test_size = {train_size + test_size}, "
            f"but only {total_examples} Spider examples are available"
        )

    rng = random.Random(seed)
    shuffled = list(range(total_examples))
    rng.shuffle(shuffled)
    train_indices = sorted(shuffled[:train_size])
    test_indices = sorted(shuffled[train_size: train_size + test_size])
    return {
        "seed": seed,
        "split_strategy": "random_non_overlapping",
        "total_examples": total_examples,
        "train_indices": train_indices,
        "test_indices": test_indices,
        "train_size": len(train_indices),
        "test_size": len(test_indices),
    }


def _attach_split_previews(
    split: Dict[str, Any],
    rows: List[Dict[str, Any]],
    *,
    include_difficulty: bool = False,
    preview_limit: int = 10,
) -> None:
    """Attach train/test preview rows to a split manifest in place."""
    for split_name in ("train", "test"):
        previews = []
        for idx in split[f"{split_name}_indices"][:preview_limit]:
            row = rows[idx]
            preview = {
                "index": idx,
                "db_id": row.get("db_id", ""),
                "question": row.get("question", ""),
            }
            if include_difficulty:
                preview["difficulty"] = split["difficulty_by_index"].get(str(idx))
            previews.append(preview)
        split[f"{split_name}_preview"] = previews


def write_spider_proportional_train_test_split(
    output_path: Path | str,
    *,
    source: str = "auto",
    spider_dir: Optional[Path | str] = None,
    train_size: int = 300,
    test_size: int = 300,
    seed: int = 334,
    include_preview: bool = True,
) -> Dict[str, Any]:
    """Write a proportional stratified Spider train/test manifest.

    Defaults to the canonical 300 train / 300 held-out at seed 334 (the IterGen
    comparison split).
    """
    rows = load_spider(source=source, spider_dir=spider_dir)
    split = make_spider_proportional_train_test_split(
        source=source,
        spider_dir=spider_dir,
        train_size=train_size,
        test_size=test_size,
        seed=seed,
    )
    if include_preview:
        _attach_split_previews(split, rows, include_difficulty=True)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(split, indent=2) + "\n")
    return split


def write_spider_train_test_split(
    output_path: Path | str,
    *,
    source: str = "auto",
    spider_dir: Optional[Path | str] = None,
    train_size: int = 300,
    test_size: int = 300,
    seed: int = 334,
    include_preview: bool = True,
    proportional: bool = True,
) -> Dict[str, Any]:
    """Write a deterministic Spider split manifest for synthesis/test workflows.

    Defaults to the canonical 300 train / 300 held-out at seed 334 via the
    proportional path. When proportional=False, the legacy plain-random draw is used
    (seed 334 there does not reproduce IterGen's held-out 300).
    """
    if proportional:
        return write_spider_proportional_train_test_split(
            output_path,
            source=source,
            spider_dir=spider_dir,
            train_size=train_size,
            test_size=test_size,
            seed=seed,
            include_preview=include_preview,
        )

    rows = load_spider(source=source, spider_dir=spider_dir)
    split = make_spider_train_test_split(
        total_examples=len(rows),
        train_size=train_size,
        test_size=test_size,
        seed=seed,
    )
    if include_preview:
        _attach_split_previews(split, rows)

    split["eval_indices"] = list(split["test_indices"])
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(split, indent=2) + "\n")
    return split


def write_gold_file(examples: List[Dict[str, Any]], path: Path) -> None:
    """Write a Spider-format gold file (<sql>\\t<db_id>) for the given examples."""
    with open(path, "w") as f:
        for ex in examples:
            gold = (ex.get("query") or "").replace("\n", " ").strip()
            db_id = ex.get("db_id", "")
            f.write(f"{gold}\t{db_id}\n")


def default_db_dir() -> Path:
    """SQLite databases for Spider execution accuracy: the official release copy.

    IterGen's vendored `sql_spider_eval/databases` is built by create_databases.py
    from tables.json alone (no INSERTs), so every query returns an empty result
    and wrong SQL scores as correct. The earlier "~8pp" gap that made the
    vendored copy look preferable was that artifact. See
    saved-results/2026-09-25-spider-populated-db-rescore.md.
    """
    return require_populated_db_dir(DEFAULT_SPIDER_DIR / "database")


def require_populated_db_dir(db_dir: Path) -> Path:
    """Refuse a schema-only database copy (probe: concert_singer.concert has rows)."""
    probe = Path(db_dir) / "concert_singer" / "concert_singer.sqlite"
    con = sqlite3.connect(f"file:{probe}?mode=ro", uri=True)
    try:
        rows = con.execute("SELECT COUNT(*) FROM concert").fetchone()[0]
    finally:
        con.close()
    if rows == 0:
        raise RuntimeError(f"Spider databases at {db_dir} have no rows (schema-only copy)")
    logger.info("[spider-db] using populated databases at %s", db_dir)
    return Path(db_dir)


def default_tables_json() -> Path:
    """Default tables.json path for Spider."""
    env = os.environ.get("SPIDER_TABLES_JSON")
    if env:
        return Path(env)
    local = DEFAULT_SPIDER_DIR / "tables.json"
    if local.exists():
        return local
    return _vendored_spider_eval_dir() / "evaluation_examples" / "examples" / "tables.json"
