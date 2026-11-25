from collections.abc import Iterator, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from itertools import islice, product
from pathlib import Path
from sys import maxsize
from typing import NamedTuple

from ..instances import Instance, load_default_instance, load_instance
from ..search.shared import SearchParameters, TabuTenure
from .execution import run_once, tabu_search_for
from .reports import write_csv
from .reports.directory import create_report_directory

CALIBRATION_TIME_LIMIT_S = 180.0
CALIBRATION_SEED = 1

type Trial = tuple[Instance, TabuTenure]


class CalibrationRow(NamedTuple):
    instance: str
    tabu_tenure_min: int
    tabu_tenure_max: int
    seed: int
    optimal_cost: int
    time_limit_s: float
    total_time_s: float
    jobs: int
    best_cost: int


CALIBRATION_HEADER = CalibrationRow._fields


class CalibrationError(RuntimeError):
    pass


def load_calibration_instances(benchmarks_dir: Path) -> tuple[Instance, ...]:
    return (
        load_default_instance(),
        load_instance(benchmarks_dir / "A-n80-k10.vrp", optimal_cost=1763),
        load_instance(benchmarks_dir / "B-n78-k10.vrp", optimal_cost=1221),
    )


def run_calibration(
    instances: Sequence[Instance],
    tenures: Sequence[TabuTenure],
    *,
    time_limit_s: float,
    output_dir: Path,
    jobs: int = 1,
) -> None:
    if jobs < 1:
        raise ValueError("Calibration requires at least one job.")

    report_path = create_report_directory(output_dir) / "calibration.csv"
    total_trials = len(instances) * len(tenures)
    active_jobs = min(jobs, max(1, total_trials))
    print(f"Calibration table: {report_path} | jobs: {active_jobs}", flush=True)
    trials = product(instances, tenures)
    rows = _calibration_rows(trials, time_limit_s, active_jobs, total_trials)
    write_csv(report_path, CALIBRATION_HEADER, rows, flush_rows=True)


def _calibration_rows(
    trials: Iterator[Trial], time_limit_s: float, jobs: int, total_trials: int
) -> Iterator[CalibrationRow]:
    if jobs == 1:
        results = (
            _run_trial(instance, tenure, time_limit_s, jobs)
            for instance, tenure in trials
        )
    else:
        results = _run_parallel_trials(trials, time_limit_s, jobs)

    for index, row in enumerate(results, start=1):
        print(
            f"[{index}/{total_trials}] {row.instance}: tabu tenure "
            f"[{row.tabu_tenure_min}, {row.tabu_tenure_max}] | cost {row.best_cost}",
            flush=True,
        )
        yield row


def _run_parallel_trials(
    trials: Iterator[Trial], time_limit_s: float, jobs: int
) -> Iterator[CalibrationRow]:
    with ProcessPoolExecutor(max_workers=jobs) as executor:

        def submit(trial: Trial) -> Future[CalibrationRow]:
            instance, tenure = trial
            return executor.submit(_run_trial, instance, tenure, time_limit_s, jobs)

        pending = {submit(trial) for trial in islice(trials, jobs)}
        try:
            while pending:
                completed, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in completed:
                    yield future.result()

                pending.update(
                    submit(trial) for trial in islice(trials, len(completed))
                )
        finally:
            for future in pending:
                future.cancel()


def _run_trial(
    instance: Instance, tenure: TabuTenure, time_limit_s: float, jobs: int
) -> CalibrationRow:
    # Keep both iteration limits out of reach so the time budget governs the trial.
    parameters = SearchParameters(
        max_iterations=maxsize,
        stagnation_limit=maxsize,
        tabu_tenure=tenure,
        time_limit_s=time_limit_s,
    )
    search = tabu_search_for(instance)
    run = run_once(instance, search, parameters, seed=CALIBRATION_SEED)
    if run is None:
        raise CalibrationError(
            f"Could not build a feasible initial solution for {instance.name}."
        )

    return CalibrationRow(
        instance=instance.name,
        tabu_tenure_min=tenure.minimum,
        tabu_tenure_max=tenure.maximum,
        seed=run.seed,
        optimal_cost=instance.optimal_cost,
        time_limit_s=time_limit_s,
        total_time_s=run.total_time_s,
        jobs=jobs,
        best_cost=run.cost,
    )
