from collections.abc import Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from sys import maxsize
from typing import NamedTuple

from ..instances import Instance
from ..search.shared import SearchParameters, TabuTenure
from .execution import run_once, tabu_search_for
from .reports import write_csv
from .reports.directory import create_report_directory
from .reports.environment import write_environment
from .reports.plots import plot_time_to_target


class TttRow(NamedTuple):
    instance: str
    seed: int
    target_cost: int
    tabu_tenure_min: int
    tabu_tenure_max: int
    time_limit_s: float
    reached_target: bool
    time_to_target_s: float | None
    total_time_s: float
    iterations: int
    best_cost: int


TTT_HEADER = TttRow._fields


class TttError(RuntimeError):
    pass


def run_time_to_target(
    instance: Instance,
    tenure: TabuTenure,
    *,
    runs: int,
    time_limit_s: float,
    output_dir: Path,
    jobs: int = 1,
    first_seed: int = 1,
) -> Path:
    if runs < 1 or jobs < 1:
        raise ValueError("Runs and jobs must be positive.")

    report_directory = create_report_directory(output_dir)
    write_environment(report_directory)
    seeds = range(first_seed, first_seed + runs)
    completed: list[TttRow] = []
    rows = _announce(_ttt_rows(instance, tenure, seeds, time_limit_s, jobs), completed)
    csv_path = report_directory / f"ttt_runs_{instance.name}.csv"
    write_csv(csv_path, TTT_HEADER, rows, flush_rows=True)

    completed.sort(key=lambda row: row.seed)
    plot_time_to_target(
        instance.optimal_cost,
        [row.time_to_target_s for row in completed],
        report_directory / f"ttt_{instance.name}.png",
    )
    return report_directory


def _announce(rows: Iterable[TttRow], completed: list[TttRow]) -> Iterator[TttRow]:
    for index, row in enumerate(rows, start=1):
        completed.append(row)
        if row.time_to_target_s is None:
            status = "target missed"
        else:
            status = f"target in {row.time_to_target_s:.3f} s"
        print(f"[{index}] seed {row.seed}: {status} | cost {row.best_cost}", flush=True)
        yield row


def _ttt_rows(
    instance: Instance,
    tenure: TabuTenure,
    seeds: Iterable[int],
    time_limit_s: float,
    jobs: int,
) -> Iterator[TttRow]:
    if jobs == 1:
        for seed in seeds:
            yield _run_trial(instance, tenure, time_limit_s, seed)
        return

    with ProcessPoolExecutor(max_workers=jobs) as executor:
        futures = [
            executor.submit(_run_trial, instance, tenure, time_limit_s, seed)
            for seed in seeds
        ]
        try:
            for future in as_completed(futures):
                yield future.result()
        finally:
            for future in futures:
                future.cancel()


def _run_trial(
    instance: Instance, tenure: TabuTenure, time_limit_s: float, seed: int
) -> TttRow:
    # Only the target and the time limit stop a time-to-target run.
    parameters = SearchParameters(
        max_iterations=maxsize,
        stagnation_limit=maxsize,
        tabu_tenure=tenure,
        time_limit_s=time_limit_s,
        stop_at_target=True,
    )
    run = run_once(instance, tabu_search_for(instance), parameters, seed=seed)
    if run is None:
        raise TttError(
            f"Could not build a feasible initial solution for {instance.name}."
        )

    return TttRow(
        instance=instance.name,
        seed=seed,
        target_cost=instance.optimal_cost,
        tabu_tenure_min=tenure.minimum,
        tabu_tenure_max=tenure.maximum,
        time_limit_s=time_limit_s,
        reached_target=run.time_to_target_s is not None,
        time_to_target_s=run.time_to_target_s,
        total_time_s=run.total_time_s,
        iterations=run.iterations,
        best_cost=run.cost,
    )
