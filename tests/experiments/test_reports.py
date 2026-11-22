import csv
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock

import pytest

from cvrp_tabu_search.experiments import reports
from cvrp_tabu_search.experiments.reports import TargetMeasurement, directory, plots
from cvrp_tabu_search.experiments.runner import Run
from cvrp_tabu_search.instances import Instance
from cvrp_tabu_search.search.shared import (
    SearchHistoryEntry,
    SearchParameters,
    TabuTenure,
)


def _read_csv(path: Path) -> list[list[str]]:
    with path.open(encoding="utf-8", newline="") as report:
        return list(csv.reader(report))


def test_reports_use_the_first_cheapest_run_for_history_and_plots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    instance = Instance("example", 18, 1, 2, 10, [[0, 0], [2, 0]], [0, 1], 1)
    parameters = SearchParameters(
        max_iterations=20, stagnation_limit=5, tabu_tenure=TabuTenure(2, 4)
    )
    initial_entry = SearchHistoryEntry(0, 30, 0.0, 30)
    tied_best_history = [initial_entry, SearchHistoryEntry(1, 18, 0.375, 18)]
    runs = [
        Run(
            6, 24, [initial_entry, SearchHistoryEntry(1, 24, 0.25, 24)], 1.0, 0.25, None
        ),
        Run(7, 18, tied_best_history, 0.75, 0.375, 0.375),
        Run(
            8,
            18,
            [initial_entry, SearchHistoryEntry(1, 18, 0.125, 18)],
            0.5,
            0.125,
            0.125,
        ),
    ]
    draw_plots = Mock()
    monkeypatch.setattr(reports, "generate_plots", draw_plots)
    monkeypatch.chdir(tmp_path)

    reports.generate_reports(instance, parameters, runs, prefix="test_")

    [run_directory] = (tmp_path / "results").iterdir()
    assert _read_csv(run_directory / "history_test_example.csv") == [
        ["iteration", "best_cost", "elapsed_time_s", "current_cost"],
        ["1", "18", "0.375", "18"],
    ]
    run_rows = _read_csv(run_directory / "runs_test_example.csv")
    assert run_rows[0] == [
        "instance",
        "repetition",
        "seed",
        "tabu_tenure_min",
        "tabu_tenure_max",
        "max_iterations",
        "stagnation_limit",
        "optimal_cost",
        "best_cost",
        "time_to_best_s",
        "total_time_s",
        "time_to_target_s",
    ]
    assert run_rows[1:] == [
        ["example", "1", "6", "2", "4", "20", "5", "18", "24", "0.25", "1.0", ""],
        [
            "example",
            "2",
            "7",
            "2",
            "4",
            "20",
            "5",
            "18",
            "18",
            "0.375",
            "0.75",
            "0.375",
        ],
        ["example", "3", "8", "2", "4", "20", "5", "18", "18", "0.125", "0.5", "0.125"],
    ]
    draw_plots.assert_called_once_with(
        "example",
        18,
        tied_best_history,
        [(1.0, None), (0.75, 0.375), (0.5, 0.125)],
        prefix="test_",
        output_dir=run_directory.relative_to(tmp_path),
    )


def test_report_directories_are_distinct_when_created_in_the_same_second(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = Mock()
    clock.now.return_value = datetime(2026, 11, 22, 18, 34, 34)
    monkeypatch.setattr(directory, "datetime", clock)
    output_dir = tmp_path / "results"

    first_directory = directory.create_report_directory(output_dir)
    second_directory = directory.create_report_directory(output_dir)

    assert first_directory == output_dir / "2026-11-22_18-34-34"
    assert second_directory == output_dir / "2026-11-22_18-34-34-2"
    assert set(output_dir.iterdir()) == {first_directory, second_directory}


@pytest.mark.parametrize(
    ("measurements", "times", "percentages", "hits"),
    [
        (
            [
                TargetMeasurement(3, 2),
                TargetMeasurement(1.25, 1),
                TargetMeasurement(4, None),
                TargetMeasurement(3, 2),
            ],
            [0, 1, 2, 2, 4],
            [0, 25, 50, 75, 75],
            3,
        ),
        ([TargetMeasurement(2, None), TargetMeasurement(4, None)], [0, 4], [0, 0], 0),
    ],
    ids=["sorted-hits-with-a-tie-and-a-miss", "all-missed"],
)
def test_time_to_target_curve_counts_misses_and_extends_to_the_longest_run(
    measurements: list[TargetMeasurement],
    times: list[float],
    percentages: list[float],
    hits: int,
) -> None:
    assert plots._time_to_target_curve(measurements) == (times, percentages, hits)
