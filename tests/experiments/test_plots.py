from pathlib import Path

import pytest

from cvrp_tabu_search.experiments.reports.csv_writer import (
    read_history_csv,
    write_history_csv,
)
from cvrp_tabu_search.experiments.reports.plots import (
    plot_convergence,
    plot_time_to_target,
    time_to_target_points,
)
from cvrp_tabu_search.search.shared import SearchHistoryEntry

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


@pytest.mark.parametrize(
    ("times", "expected"),
    [
        ([2.0, 1.0, None, 2.0], ([1.0, 2.0, 2.0], [0.125, 0.375, 0.625])),
        ([None, None], ([], [])),
    ],
    ids=["hits-and-a-miss", "all-missed"],
)
def test_time_to_target_points_follow_aiex_et_al(
    times: list[float | None], expected: tuple[list[float], list[float]]
) -> None:
    assert time_to_target_points(times) == expected


def test_history_round_trips_through_csv(tmp_path: Path) -> None:
    history = [
        SearchHistoryEntry(0, 30, 0.0, 30),
        SearchHistoryEntry(1, 25, 0.5, 25),
        SearchHistoryEntry(2, 25, 1.0, 27),
    ]
    path = tmp_path / "history.csv"

    write_history_csv(path, history)

    assert read_history_csv(path) == history[1:]


def test_plots_are_pngs(tmp_path: Path) -> None:
    entries = [SearchHistoryEntry(1, 25, 0.5, 25), SearchHistoryEntry(2, 25, 1.0, 27)]

    plot_convergence(20, entries, tmp_path / "convergence.png")
    plot_time_to_target(20, [0.5, None], tmp_path / "ttt.png")

    assert (tmp_path / "convergence.png").read_bytes().startswith(PNG_SIGNATURE)
    assert (tmp_path / "ttt.png").read_bytes().startswith(PNG_SIGNATURE)
