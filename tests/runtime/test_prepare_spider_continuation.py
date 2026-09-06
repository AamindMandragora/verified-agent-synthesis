import copy

import pytest

from scripts.runtime.prepare_spider_continuation import (
    build_continuation_plan,
    reconstruct_failure_ledger,
)


def _attempt(
    number,
    *,
    accuracy,
    syntax_rate,
    failed_at="evaluation",
    complete=True,
    success=True,
    error_summary="",
):
    planned = 4
    observed = planned if complete else planned - 1
    return {
        "attempt_number": number,
        "strategy_code": f"STRATEGY_{number}",
        "failed_at": failed_at,
        "succeeded": False,
        "error_summary": error_summary,
        "evaluation": {
            "success": success,
            "accuracy": accuracy,
            "syntax_rate": syntax_rate,
            "planned_num_examples": planned,
            "num_examples": observed,
            "sample_outputs": [
                {"index": index, "is_correct": index < round(accuracy * planned)}
                for index in range(observed)
            ],
            "early_stopped": not complete,
        },
    }


def test_plan_selects_complete_timeout_but_keeps_latest_attempt_offset():
    attempts = [
        _attempt(number, accuracy=0.2, syntax_rate=0.9)
        for number in range(1, 6)
    ]
    attempts.append(
        _attempt(
            6,
            accuracy=0.65,
            syntax_rate=0.99,
            failed_at="timeout",
            complete=True,
        )
    )
    attempts.append(_attempt(7, accuracy=0.5, syntax_rate=0.99))

    plan = build_continuation_plan(
        {"attempts": attempts, "total_attempts": 7},
        min_accuracy=0.66,
        min_syntax_rate=0.95,
        final_attempt_limit=43,
    )

    assert plan.initial_attempt_offset == 7
    assert plan.remaining_attempts == 36
    assert plan.incumbent_attempt_number == 6
    assert plan.incumbent_strategy == "STRATEGY_6"


def test_plan_rejects_partial_timeout_as_incumbent():
    attempts = [
        _attempt(1, accuracy=0.4, syntax_rate=0.95),
        _attempt(
            2,
            accuracy=0.9,
            syntax_rate=1.0,
            failed_at="timeout",
            complete=False,
        ),
    ]

    plan = build_continuation_plan(
        {"attempts": attempts, "total_attempts": 2},
        min_accuracy=0.66,
        min_syntax_rate=0.95,
        final_attempt_limit=43,
    )

    assert plan.incumbent_attempt_number == 1


def test_plan_rejects_partial_evaluation_as_incumbent():
    attempts = [
        _attempt(1, accuracy=0.4, syntax_rate=0.95),
        _attempt(
            2,
            accuracy=0.9,
            syntax_rate=1.0,
            failed_at="evaluation",
            complete=False,
        ),
    ]

    plan = build_continuation_plan(
        {"attempts": attempts, "total_attempts": 2},
        min_accuracy=0.66,
        min_syntax_rate=0.95,
        final_attempt_limit=43,
    )

    assert plan.incumbent_attempt_number == 1


def test_plan_rejects_unsuccessful_evaluation_with_stale_high_scores():
    attempts = [
        _attempt(1, accuracy=0.4, syntax_rate=0.95),
        _attempt(2, accuracy=0.99, syntax_rate=1.0, success=False),
    ]

    plan = build_continuation_plan(
        {"attempts": attempts, "total_attempts": 2},
        min_accuracy=0.66,
        min_syntax_rate=0.95,
        final_attempt_limit=43,
    )

    assert plan.incumbent_attempt_number == 1


def test_plan_rejects_timeout_with_corrupt_sample_records():
    attempts = [
        _attempt(1, accuracy=0.4, syntax_rate=0.95),
        _attempt(
            2,
            accuracy=0.9,
            syntax_rate=1.0,
            failed_at="timeout",
            complete=True,
        ),
    ]
    attempts[1]["evaluation"]["sample_outputs"] = [None] * 4

    plan = build_continuation_plan(
        {"attempts": attempts, "total_attempts": 2},
        min_accuracy=0.66,
        min_syntax_rate=0.95,
        final_attempt_limit=43,
    )

    assert plan.incumbent_attempt_number == 1


def test_ledger_replay_verifies_saved_summary_and_records_skipped_timeout():
    persistence = (
        "Cross-attempt mode persistence:\n"
        "  - mode_A: appeared in attempt(s) 4"
    )
    attempt4 = _attempt(
        4,
        accuracy=0.4,
        syntax_rate=0.95,
        error_summary=f"feedback before\n{persistence}\nfeedback after",
    )
    attempt5 = _attempt(
        5,
        accuracy=0.5,
        syntax_rate=0.95,
        failed_at="timeout",
        error_summary="timeout before feedback was rendered",
    )
    seed = {
        "version": 1,
        "ledger": {"next_id": 0, "modes": []},
        "included_attempts": [1, 2, 3],
    }

    def fake_renderer(samples, *, persistent_ledger, attempt_index, **kwargs):
        persistent_ledger["modes"].append(
            {"id": "mode_A", "attempts": [attempt_index]}
        )
        return persistence

    rebuilt, replayed, skipped = reconstruct_failure_ledger(
        seed,
        [attempt4, attempt5],
        seed_through_attempt=3,
        render_cluster_block=fake_renderer,
    )

    assert replayed == [4]
    assert rebuilt["included_attempts"] == [1, 2, 3, 4]
    assert rebuilt["excluded_attempts"] == [5]
    assert rebuilt["ledger"]["modes"] == [
        {"id": "mode_A", "attempts": [4]}
    ]
    assert skipped == [
        {
            "attempt_number": 5,
            "reason": "timeout_without_saved_persistence_summary",
            "error_summary_sha256": skipped[0]["error_summary_sha256"],
        }
    ]


@pytest.mark.parametrize("failed_at", ["verification", "compilation"])
def test_ledger_replay_excludes_finalized_pre_evaluation_failure(
    failed_at,
):
    seed = {
        "version": 1,
        "ledger": {"next_id": 0, "modes": []},
        "included_attempts": [],
        "excluded_attempts": [],
    }
    pre_evaluation_failure = {
        "attempt_number": 1,
        "strategy_code": "BROKEN",
        "failed_at": failed_at,
        "error_summary": f"Dafny {failed_at} failed",
        "verification": {"success": False},
        "evaluation": None,
    }

    rebuilt, replayed, skipped = reconstruct_failure_ledger(
        seed,
        [pre_evaluation_failure],
        seed_through_attempt=0,
        render_cluster_block=lambda *args, **kwargs: pytest.fail("no evaluation to replay"),
    )

    assert replayed == []
    assert rebuilt["included_attempts"] == []
    assert rebuilt["excluded_attempts"] == [1]
    assert skipped == [
        {
            "attempt_number": 1,
            "reason": f"{failed_at}_without_evaluation",
            "error_summary_sha256": skipped[0]["error_summary_sha256"],
        }
    ]


def test_ledger_replay_rejects_missing_summary_when_evaluation_exists():
    seed = {
        "version": 1,
        "ledger": {"next_id": 0, "modes": []},
        "included_attempts": [],
        "excluded_attempts": [],
    }

    with pytest.raises(ValueError, match="no saved persistence summary"):
        reconstruct_failure_ledger(
            seed,
            [_attempt(1, accuracy=0.4, syntax_rate=0.95)],
            seed_through_attempt=0,
            render_cluster_block=lambda *args, **kwargs: "",
        )


def test_ledger_replay_fails_closed_on_mismatched_saved_summary():
    attempt4 = _attempt(
        4,
        accuracy=0.4,
        syntax_rate=0.95,
        error_summary=(
            "Cross-attempt mode persistence:\n"
            "  - mode_A: appeared in attempt(s) 4"
        ),
    )
    seed = {
        "version": 1,
        "ledger": {"next_id": 0, "modes": []},
        "included_attempts": [1, 2, 3],
    }

    with pytest.raises(ValueError, match="does not match"):
        reconstruct_failure_ledger(
            copy.deepcopy(seed),
            [attempt4],
            seed_through_attempt=3,
            render_cluster_block=lambda *args, **kwargs: (
                "Cross-attempt mode persistence:\n"
                "  - mode_B: appeared in attempt(s) 4"
            ),
        )


@pytest.mark.parametrize(
    "seed",
    [
        {
            "version": 1,
            "ledger": {"next_id": 0, "modes": []},
            "included_attempts": [1, 1, 2],
            "excluded_attempts": [3],
        },
        {
            "version": 1,
            "ledger": {"next_id": 0, "modes": []},
            "included_attempts": [1],
            "excluded_attempts": [3],
        },
        {
            "version": 1,
            "ledger": {"next_id": 0, "modes": []},
            "included_attempts": [0, 1, 2],
            "excluded_attempts": [3],
        },
    ],
)
def test_ledger_seed_must_cover_boundary_once(seed):
    with pytest.raises(ValueError, match="cover each attempt"):
        reconstruct_failure_ledger(
            seed,
            [],
            seed_through_attempt=3,
            render_cluster_block=lambda *args, **kwargs: "",
        )
