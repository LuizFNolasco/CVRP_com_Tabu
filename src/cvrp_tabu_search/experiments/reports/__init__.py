from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from ...instances import Instance
from ...search.shared import SearchParameters
from .csv_writer import (
    HISTORY_HEADER,
    RUN_HEADER,
    CsvValue,
    write_csv,
    write_history_csv,
    write_runs_csv,
)
from .directory import DEFAULT_OUTPUT_DIR, create_report_directory
from .environment import write_environment
from .plots import TargetMeasurement, generate_plots
from .solution import write_solution_json

if TYPE_CHECKING:
    from ..execution import Run

__all__ = [
    "DEFAULT_OUTPUT_DIR",
    "HISTORY_HEADER",
    "RUN_HEADER",
    "CsvValue",
    "TargetMeasurement",
    "generate_plots",
    "generate_reports",
    "write_csv",
]


def generate_reports(
    instance: Instance,
    parameters: SearchParameters,
    runs: Sequence[Run],
    *,
    prefix: str = "",
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> None:
    report_directory = create_report_directory(output_dir)
    write_environment(report_directory)

    name = f"{prefix}{instance.name}"
    runs_path = report_directory / f"runs_{name}.csv"
    history_path = report_directory / f"history_{name}.csv"
    write_runs_csv(runs_path, instance, parameters, runs)

    best_run = min(runs, key=lambda run: run.cost)
    check = write_solution_json(
        report_directory / f"solution_{name}.json", instance, best_run.routes
    )
    if best_run.routes and not check.feasible:
        print(f"Warning: infeasible best solution: {'; '.join(check.problems)}")
    best_history = best_run.history
    write_history_csv(history_path, best_history)

    measurements = [
        TargetMeasurement(run.total_time_s, run.time_to_target_s) for run in runs
    ]
    generate_plots(
        instance.name,
        instance.optimal_cost,
        best_history,
        measurements,
        prefix=prefix,
        output_dir=report_directory,
    )
