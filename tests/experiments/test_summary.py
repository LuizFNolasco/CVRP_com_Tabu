from collections.abc import Callable
from pathlib import Path

import pytest

from cvrp_tabu_search.experiments.summary import ResultsError, load_summaries

type RunsWriter = Callable[[Path, str, int, list[int], list[float]], None]


def test_summaries_compute_the_assignment_statistics(
    tmp_path: Path, write_runs: RunsWriter
) -> None:
    write_runs(
        tmp_path / "a" / "runs_tabu_A-n32-k5.csv",
        "A-n32-k5",
        784,
        [800, 784, 790, 784, 792],
        [10.0, 12.0, 11.0, 9.0, 13.0],
    )

    [summary] = load_summaries([tmp_path], prefix="tabu_")

    assert summary.group == "A"
    assert summary.customers == 31
    assert summary.best_cost == 784
    assert summary.mean_cost == 790.0
    assert summary.best_run_time_s == 12.0
    assert summary.mean_time_s == 11.0
    assert summary.gap_best == 0.0
    assert summary.gap_mean == pytest.approx(100 * 6 / 784)


def test_summaries_ignore_other_variants_and_sort_by_set_and_size(
    tmp_path: Path, write_runs: RunsWriter
) -> None:
    write_runs(tmp_path / "1" / "runs_tabu_B-n31-k5.csv", "B-n31-k5", 672, [700], [1])
    write_runs(
        tmp_path / "2" / "runs_tabu_A-n80-k10.csv", "A-n80-k10", 1763, [1800], [1]
    )
    write_runs(tmp_path / "3" / "runs_tabu_A-n33-k5.csv", "A-n33-k5", 661, [661], [1])
    write_runs(
        tmp_path / "4" / "runs_tabu_nophases_A-n33-k5.csv", "A-n33-k5", 661, [700], [1]
    )

    summaries = load_summaries([tmp_path], prefix="tabu_")

    assert [summary.instance for summary in summaries] == [
        "A-n33-k5",
        "A-n80-k10",
        "B-n31-k5",
    ]
    assert summaries[0].costs == (661,)


def test_duplicate_run_sets_are_rejected(
    tmp_path: Path, write_runs: RunsWriter
) -> None:
    for folder in ("first", "second"):
        write_runs(
            tmp_path / folder / "runs_tabu_A-n33-k5.csv", "A-n33-k5", 661, [661], [1]
        )

    with pytest.raises(ResultsError, match="A-n33-k5"):
        load_summaries([tmp_path], prefix="tabu_")
