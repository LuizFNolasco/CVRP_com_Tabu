from __future__ import annotations

import csv
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING

from ...instances import Instance
from ...search.shared import SearchHistory, SearchParameters

if TYPE_CHECKING:
    from ..runner import Run

RUN_HEADER = (
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
)
HISTORY_HEADER = ("iteration", "best_cost", "elapsed_time_s", "current_cost")

type CsvValue = str | int | float | None


def write_runs_csv(
    path: Path, instance: Instance, parameters: SearchParameters, runs: Iterable[Run]
) -> None:
    write_csv(path, RUN_HEADER, _run_rows(instance, parameters, runs))


def write_history_csv(path: Path, history: SearchHistory) -> None:
    write_csv(path, HISTORY_HEADER, history[1:])


def _run_rows(
    instance: Instance, parameters: SearchParameters, runs: Iterable[Run]
) -> Iterator[list[CsvValue]]:
    for repetition, run in enumerate(runs, start=1):
        yield [
            instance.name,
            repetition,
            run.seed,
            *parameters.tabu_tenure,
            parameters.max_iterations,
            parameters.stagnation_limit,
            instance.optimal_cost,
            run.cost,
            run.time_to_best_s,
            run.total_time_s,
            run.time_to_target_s,
        ]


def write_csv(
    path: str | Path, header: Iterable[str], rows: Iterable[Iterable[CsvValue]]
) -> None:
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)
