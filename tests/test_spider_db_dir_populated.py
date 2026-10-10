"""Spider execution accuracy must run against databases that contain rows.

IterGen's vendored `sql_spider_eval/databases` is built from tables.json only
(no INSERTs), so every query returns an empty result and wrong SQL scores as
correct. See saved-results/2026-09-25-spider-populated-db-rescore.md.
"""
import sqlite3
from pathlib import Path

import pytest

from synthesis.evaluate.benchmarks.sql_spider import dataset


def _concert_rows(db_dir: Path) -> int:
    con = sqlite3.connect(f"file:{db_dir / 'concert_singer' / 'concert_singer.sqlite'}?mode=ro", uri=True)
    try:
        return con.execute("SELECT COUNT(*) FROM concert").fetchone()[0]
    finally:
        con.close()


def test_default_db_dir_has_data(monkeypatch):
    monkeypatch.delenv("SPIDER_DB_DIR", raising=False)
    assert _concert_rows(dataset.default_db_dir()) > 0


def test_env_cannot_point_scoring_at_schema_only_dbs(monkeypatch):
    empty = dataset._vendored_spider_eval_dir() / "databases"
    assert _concert_rows(empty) == 0, "precondition: vendored copy is schema-only"
    monkeypatch.setenv("SPIDER_DB_DIR", str(empty))
    assert _concert_rows(dataset.default_db_dir()) > 0


def test_schema_only_dir_is_rejected(tmp_path, monkeypatch):
    empty = dataset._vendored_spider_eval_dir() / "databases"
    with pytest.raises(RuntimeError, match="no rows"):
        dataset.require_populated_db_dir(empty)
