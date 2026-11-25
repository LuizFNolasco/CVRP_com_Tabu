import csv
from collections.abc import Callable
from pathlib import Path

import matplotlib
import pytest

from cvrp_tabu_search.experiments.reports.csv_writer import RUN_HEADER

matplotlib.use("Agg")

type RunsWriter = Callable[[Path, str, int, list[int], list[float]], None]


def _write_runs(
    path: Path, instance: str, optimal: int, costs: list[int], times: list[float]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(RUN_HEADER)
        for repetition, (cost, time) in enumerate(zip(costs, times), start=1):
            writer.writerow(
                [
                    instance,
                    repetition,
                    repetition,
                    1,
                    15,
                    5000,
                    1000,
                    optimal,
                    cost,
                    time / 2,
                    time,
                    "",
                    "",
                ]
            )


@pytest.fixture
def write_runs() -> RunsWriter:
    """Write a synthetic runs_*.csv in the format of RUN_HEADER."""
    return _write_runs
