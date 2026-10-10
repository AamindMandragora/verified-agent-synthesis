"""A predicted query that runs too long must be scored wrong, not hang the lane.

Repro (2026-09-26, spider-cars-q35-2b-t2-s00, train row 104, db wta_1): the model
wrote `JOIN rankings r ON p.player_id = p.player_id`, an always-true join, so
SQLite built 20,662 x 510,437 ~= 10.5e9 rows and the lane sat for 20+ minutes.
Each case runs in a subprocess with a wall-clock limit, because a stuck SQLite
call cannot be interrupted from Python.
"""
import json
import subprocess
import sys
import textwrap
import time

import pytest

from synthesis.evaluate.benchmarks.sql_spider.dataset import default_db_dir

RUNAWAY = (
    "SELECT p.first_name, r.ranking FROM players p JOIN rankings r ON p.player_id = p.player_id "
    "GROUP BY p.player_id, p.first_name, p.hand, p.birth_date, p.country_code ORDER BY r.ranking"
)
GOLD = ("select avg ( ranking ) , players.first_name from players join rankings "
        "on players.player_id = rankings.player_id group by players.first_name")


def _db_dir_or_skip():
    try:
        db = default_db_dir()
    except Exception as exc:  # schema-only copy
        pytest.skip(f"populated Spider DBs not available: {exc}")
    if not (db / "wta_1" / "wta_1.sqlite").exists():
        pytest.skip("wta_1 not available")
    return db


def _score_in_subprocess(pred, cap_s, wall_s):
    code = textwrap.dedent(f"""
        import json
        from synthesis.evaluate.benchmarks.sql_spider import executor
        executor.SPIDER_QUERY_CAP_S = {cap_s!r}
        ex = {{"db_id": "wta_1", "query": {GOLD!r}, "question": "q"}}
        _, _, rows = executor.execute_accuracy([{pred!r}], [ex], etype="exec")
        print(json.dumps({{"exec": bool(rows[0].get("exec"))}}))
    """)
    t0 = time.time()
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=wall_s)
    assert out.returncode == 0, out.stderr[-2000:]
    return json.loads(out.stdout.strip().splitlines()[-1])["exec"], time.time() - t0


def test_runaway_prediction_is_scored_wrong_within_the_cap():
    _db_dir_or_skip()
    try:
        ok, took = _score_in_subprocess(RUNAWAY, cap_s=2.0, wall_s=90)
    except subprocess.TimeoutExpired:
        pytest.fail("scoring a runaway query did not return within 90 s: no per-query cap")
    assert ok is False
    assert took < 60


def test_the_cap_is_60_seconds():
    from synthesis.evaluate.benchmarks.sql_spider import executor
    assert executor.SPIDER_QUERY_CAP_S == 60.0


def test_a_normal_correct_prediction_still_scores_right_under_the_cap():
    _db_dir_or_skip()
    ok, _ = _score_in_subprocess(GOLD, cap_s=60.0, wall_s=120)
    assert ok is True
