import pytest

from cvrp_tabu_search.experiments.execution import _measure_times
from cvrp_tabu_search.search.shared import SearchHistoryEntry


@pytest.mark.parametrize(
    ("target", "expected_target_time"),
    [(18, 0.0), (14, 0.25), (13, 0.5), (11, None)],
    ids=["initial-hit", "equal-to-target", "below-target", "miss"],
)
def test_run_times_use_the_first_occurrence_of_the_best_cost_and_target(
    target: int, expected_target_time: float | None
) -> None:
    history = [
        SearchHistoryEntry(0, 18, 0.0, 18),
        SearchHistoryEntry(1, 14, 0.25, 14),
        SearchHistoryEntry(2, 12, 0.5, 12),
        SearchHistoryEntry(3, 12, 0.75, 16),
    ]

    time_to_best, time_to_target = _measure_times(
        history, final_cost=12, target_cost=target
    )

    assert time_to_best == 0.5
    assert time_to_target == expected_target_time
