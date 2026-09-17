import pytest

from scripts.runtime.win_bar import accuracy_bar


def test_bar_is_ten_points_above_the_baseline():
    bar = accuracy_bar(219, 300)  # 73.0%
    assert bar.min_accuracy == 249 / 300  # 83.0%
    assert bar.policy == "baseline_plus_10_points"


def test_ten_points_round_up_to_a_whole_example():
    assert accuracy_bar(5, 49).min_accuracy == 10 / 49  # 4.9 examples -> 5


def test_near_the_ceiling_the_bar_is_halfway_to_perfect():
    bar = accuracy_bar(285, 300)  # 95%: +10 points is impossible
    assert bar.min_accuracy == 293 / 300
    assert bar.policy == "halfway_to_perfect"


def test_exactly_at_the_switch_point():
    assert accuracy_bar(241, 300).policy == "baseline_plus_10_points"
    assert accuracy_bar(242, 300).policy == "halfway_to_perfect"


def test_perfect_baseline_must_be_matched():
    bar = accuracy_bar(300, 300)
    assert bar.min_accuracy == 1.0 and bar.policy == "perfect_baseline_must_match"


def test_bar_is_always_above_the_baseline_and_reachable():
    for total in (49, 100, 300):
        for correct in range(total):
            bar = accuracy_bar(correct, total).min_accuracy
            assert correct / total < bar <= 1.0


@pytest.mark.parametrize("correct,total", [(-1, 10), (11, 10), (0, 0)])
def test_bad_counts_are_refused(correct, total):
    with pytest.raises(ValueError):
        accuracy_bar(correct, total)
