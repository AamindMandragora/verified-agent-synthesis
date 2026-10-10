"""
Execution-accuracy scoring for Spider predictions.

Thin wrapper over the vendored Spider evaluator at
syncode/syncode/utils/sql_spider_eval/evaluation.py, which exposes:

    evaluate(predict, gold, db_dir, etype, table, result_jsonl=None)
        -> (scores, error_types)

We write predictions and gold to temp files in the order of `examples`,
then call evaluate() and return its structured results.
"""

from __future__ import annotations

import os
import sqlite3
import sys
import time
import types
import tempfile
import importlib.util
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from synthesis.evaluate.benchmarks.sql_spider.dataset import (
    _vendored_spider_eval_dir,
    default_db_dir,
    default_tables_json,
    write_gold_file,
)


# The Spider authors' evaluation.py runs predicted SQL with no time limit. A
# model-written always-true join (e.g. `ON p.player_id = p.player_id` on wta_1,
# ~10.5e9 rows) then hangs the whole lane. Any query still running after this many
# seconds is interrupted and the row is scored wrong, the same rule the train-bar
# scorer uses and the Spider test-suite evaluator's per-query timeout follows.
SPIDER_QUERY_CAP_S = 60.0


def _capped_sqlite3() -> types.ModuleType:
    """sqlite3 stand-in whose connections interrupt work SPIDER_QUERY_CAP_S after connect."""
    shim = types.ModuleType("sqlite3_capped")
    shim.__dict__.update({k: v for k, v in vars(sqlite3).items() if not k.startswith("__")})

    def connect(*args, **kwargs):
        conn = sqlite3.connect(*args, **kwargs)
        deadline = time.monotonic() + SPIDER_QUERY_CAP_S
        conn.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10000)
        return conn

    shim.connect = connect
    return shim


def _ensure_syncode_import_path() -> None:
    """
    Make `syncode.*` imports resolvable from the repo's vendored syncode.

    Mirrors the working pattern used elsewhere in the repo (see parser_utils):
    only the repo root needs to be on sys.path. The editable-install finder
    (or, in its absence, this sys.path entry) resolves ``syncode`` to
    ``synthesis/evaluate/syncode/syncode``, which makes qualified imports like
    ``from syncode.parsers.grammars import Grammar`` work — and the vendored
    package's internal unqualified imports resolve relative to that.
    """
    repo_root = Path(__file__).parent.parent.parent           # csd-generation/
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


def _load_spider_evaluate():
    """Load the vendored Spider evaluator without relying on `syncode` package resolution."""
    repo_root = Path(__file__).parent.parent.parent
    eval_override = os.environ.get("SPIDER_EVAL_PY")
    eval_override_dir = os.environ.get("SPIDER_EVAL_DIR")
    candidates = [
        Path(eval_override).expanduser() if eval_override else None,
        (Path(eval_override_dir).expanduser() / "evaluation.py") if eval_override_dir else None,
        _vendored_spider_eval_dir() / "evaluation.py",
        repo_root / "syncode" / "syncode" / "utils" / "sql_spider_eval" / "evaluation.py",
        Path.home() / "CRANE" / "src" / "crane" / "iter_syncode" / "utils" / "sql_spider_eval" / "evaluation.py",
    ]
    eval_path = next((path for path in candidates if path is not None and path.exists()), None)
    if eval_path is None:
        candidate_text = "\n".join(f"  - {path}" for path in candidates)
        raise FileNotFoundError(f"Could not find Spider evaluator. Checked:\n{candidate_text}")

    eval_dir = str(eval_path.parent)
    if eval_dir not in sys.path:
        sys.path.insert(0, eval_dir)

    module_name = "_vas_sql_spider_eval"
    module = sys.modules.get(module_name)
    if module is None or getattr(module, "__file__", None) != str(eval_path):
        spec = importlib.util.spec_from_file_location(module_name, eval_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load Spider evaluator from {eval_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    module.sqlite3 = _capped_sqlite3()
    return module.evaluate


def _clean_sql(text: str) -> str:
    """Post-process a model completion into a single-line SQL string."""
    if text is None:
        return ""
    s = str(text).strip()
    # Drop anything after a blank line (matches syncode Dataset.post_process_answer)
    s = s.split("\n\n")[0]
    markers = (
        r"\bHuman\s*:",
        r"\bAssistant\s*:",
        r"\bUser\s*:",
        r"\bSystem\s*:",
        r"\bdb_id\s*:",
        r"\bdb_info\s*:",
        r"\bquestion\s*:",
        r"\bSQL\s*:",
    )
    # Strip any marker the model echoed at the very start (e.g. "SQL: SELECT ..."):
    # the trailing-cut logic below ignores position-0 markers, so without this they
    # stay glued to the front of the query and make it fail to execute.
    stripped = True
    while stripped:
        stripped = False
        for marker in markers:
            m = re.match(r"\s*" + marker, s, flags=re.IGNORECASE)
            if m:
                s = s[m.end():].lstrip()
                stripped = True
    cut_positions = [len(s)]
    for marker in markers:
        match = re.search(marker, s, flags=re.IGNORECASE)
        if match and match.start() > 0:
            cut_positions.append(match.start())
    repeated_select = re.search(r"\s+SelEct\s+", s)
    if repeated_select and repeated_select.start() > 0:
        cut_positions.append(repeated_select.start())
    semicolon = s.find(";")
    if semicolon > 0:
        cut_positions.append(semicolon)
    s = s[: min(cut_positions)]
    # Collapse newlines to spaces
    s = s.replace("\n", " ").replace("\r", " ")
    # Strip trailing semicolons and whitespace
    s = s.strip().rstrip(";").strip()
    return s


def execute_accuracy(
    predictions: List[str],
    examples: List[Dict[str, Any]],
    db_dir: Optional[Path] = None,
    tables_json: Optional[Path] = None,
    etype: str = "exec",
) -> Tuple[Dict[str, Any], Dict[str, int], List[Dict[str, Any]]]:
    """
    Score a batch of predictions against their examples' gold SQL.

    Args:
        predictions: list of model completions, parallel to `examples`.
        examples: list of Spider example dicts (from load_spider).
        db_dir: path to Spider's database directory (defaults to SPIDER_DB_DIR).
        tables_json: path to Spider's tables.json (defaults to SPIDER_TABLES_JSON).
        etype: "exec" (execution accuracy only), "match" (exact set match),
               or "all" (both).

    Returns:
        (scores, error_types, per_row)
          - scores: nested dict by hardness level -> {"exec": float, "count": int, ...}
          - error_types: counter of syntax-validity outcomes
          - per_row: list of per-example dicts with {pred, gold, db_id, exec, validity}
    """
    _ensure_syncode_import_path()

    if db_dir is None:
        db_dir = default_db_dir()
    if tables_json is None:
        tables_json = default_tables_json()

    if len(predictions) != len(examples):
        raise ValueError(
            f"predictions ({len(predictions)}) and examples ({len(examples)}) must match"
        )

    with tempfile.TemporaryDirectory(prefix="sql_eval_") as tmp:
        pred_path = Path(tmp) / "pred.sql"
        gold_path = Path(tmp) / "gold.sql"

        with open(pred_path, "w") as f:
            for pred in predictions:
                f.write(_clean_sql(pred) + "\n")
        write_gold_file(examples, gold_path)

        # result_jsonl gets in-place mutated to attach validity/error per row
        result_jsonl: List[Dict[str, Any]] = [
            {
                "task_id": i,
                "pred": _clean_sql(predictions[i]),
                "gold": (examples[i].get("query") or "").strip(),
                "db_id": examples[i].get("db_id", ""),
            }
            for i in range(len(examples))
        ]

        evaluate = _load_spider_evaluate()

        try:
            scores, error_types = evaluate(
                str(pred_path),
                str(gold_path),
                str(db_dir),
                etype,
                str(tables_json),
                result_jsonl=result_jsonl,
            )
        except Exception as batch_exc:
            print(
                "[spider-eval] Batch Spider evaluator failed; "
                f"falling back to per-row scoring: {batch_exc}",
                flush=True,
            )
            scores, error_types = _evaluate_rows_resilient(
                predictions=predictions,
                examples=examples,
                db_dir=db_dir,
                etype=etype,
                tables_json=tables_json,
                result_jsonl=result_jsonl,
            )

    return scores, dict(error_types), result_jsonl


def _merge_scores(target: Dict[str, Any], source: Dict[str, Any]) -> None:
    for level, metrics in source.items():
        if not isinstance(metrics, dict):
            continue
        bucket = target.setdefault(level, {})
        for metric, value in metrics.items():
            if isinstance(value, (int, float)):
                bucket[metric] = bucket.get(metric, 0) + value


def _evaluate_rows_resilient(
    *,
    predictions: List[str],
    examples: List[Dict[str, Any]],
    db_dir: Path,
    etype: str,
    tables_json: Path,
    result_jsonl: List[Dict[str, Any]],
) -> Tuple[Dict[str, Any], Dict[str, int]]:
    evaluate = _load_spider_evaluate()

    merged_scores: Dict[str, Any] = {"all": {"count": 0, "exec": 0.0}}
    merged_errors: Counter[str] = Counter()

    for i, (pred, example) in enumerate(zip(predictions, examples)):
        with tempfile.TemporaryDirectory(prefix="sql_eval_row_") as row_tmp:
            row_pred_path = Path(row_tmp) / "pred.sql"
            row_gold_path = Path(row_tmp) / "gold.sql"
            row_pred_path.write_text(_clean_sql(pred) + "\n")
            write_gold_file([example], row_gold_path)
            row_jsonl = [dict(result_jsonl[i])]

            try:
                row_scores, row_errors = evaluate(
                    str(row_pred_path),
                    str(row_gold_path),
                    str(db_dir),
                    etype,
                    str(tables_json),
                    result_jsonl=row_jsonl,
                )
                _merge_scores(merged_scores, row_scores)
                merged_errors.update(dict(row_errors))
                result_jsonl[i].update(row_jsonl[0])
            except Exception as row_exc:
                merged_scores.setdefault("all", {})
                merged_scores["all"]["count"] = merged_scores["all"].get("count", 0) + 1
                merged_scores["all"].setdefault("exec", 0.0)
                merged_errors["Execution Error"] += 1
                result_jsonl[i].update({
                    "exec": False,
                    "validity": "Execution Error",
                    "error": str(row_exc),
                })

    for metrics in merged_scores.values():
        if isinstance(metrics, dict) and metrics.get("count"):
            count = metrics["count"]
            for key, value in list(metrics.items()):
                if key == "count" or not isinstance(value, (int, float)):
                    continue
                metrics[key] = value / count

    return merged_scores, dict(merged_errors)


def prediction_matches_gold(
    prediction: str,
    example: Dict[str, Any],
    *,
    db_dir: Optional[Path] = None,
    tables_json: Optional[Path] = None,
) -> bool:
    """Score one prediction with the vendored Spider execution-accuracy evaluator."""
    _, _, per_row = execute_accuracy(
        [prediction],
        [example],
        db_dir=db_dir,
        tables_json=tables_json,
        etype="exec",
    )
    if not per_row:
        return False
    return bool(per_row[0].get("exec"))
