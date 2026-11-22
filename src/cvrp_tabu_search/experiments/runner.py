import random
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from time import perf_counter

from ..instances import Instance
from ..routes import Coordinates, Vehicle, build_initial_solution
from ..search.baseline import tabu_search as baseline_search
from ..search.shared import SearchHistory, SearchParameters, SearchResult
from ..search.tabu import tabu_search
from .reports import DEFAULT_OUTPUT_DIR, generate_reports

REPETITIONS = 5

type SearchAlgorithm = Callable[
    [list[Vehicle], Coordinates, int, tuple[int, int], int, int, float], SearchResult
]
type SearchRunner = Callable[[list[Vehicle], float], SearchResult]


@dataclass(frozen=True)
class Run:
    seed: int
    cost: int
    history: SearchHistory
    total_time_s: float
    time_to_best_s: float | None
    time_to_target_s: float | None


def run_baseline_experiment(
    instance: Instance,
    parameters: SearchParameters,
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> bool:
    return run_experiment(instance, baseline_search, parameters, output_dir=output_dir)


def run_tabu_search_experiment(
    instance: Instance,
    parameters: SearchParameters,
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> bool:
    search = partial(tabu_search, demands=instance.demands)
    return run_experiment(
        instance, search, parameters, report_prefix="tabu_", output_dir=output_dir
    )


def run_experiment(
    instance: Instance,
    search: SearchAlgorithm,
    parameters: SearchParameters,
    *,
    report_prefix: str = "",
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> bool:
    def run_search(fleet: list[Vehicle], start_time: float) -> SearchResult:
        return search(
            fleet,
            instance.coordinates,
            instance.depot,
            parameters.tabu_tenure,
            parameters.max_iterations,
            parameters.stagnation_limit,
            start_time,
        )

    runs = _run_repetitions(instance, run_search)
    if runs is None:
        return False

    generate_reports(
        instance, parameters, runs, prefix=report_prefix, output_dir=output_dir
    )
    return True


def _run_repetitions(instance: Instance, search: SearchRunner) -> list[Run] | None:
    runs = []
    for repetition in range(1, REPETITIONS + 1):
        run = _run_repetition(instance, search, seed=repetition)
        if run is None:
            print("Could not build a feasible initial solution.")
            return None

        runs.append(run)
        print(f"Run {repetition}: cost {run.cost} | time {run.total_time_s:.3f} s")

    return runs


def _run_repetition(instance: Instance, search: SearchRunner, seed: int) -> Run | None:
    random.seed(seed)
    start_time = perf_counter()
    fleet = build_initial_solution(
        instance.vehicle_count,
        instance.dimension,
        instance.capacity,
        instance.demands,
        instance.depot,
    )
    if fleet is None:
        return None

    _, cost, history = search(fleet, start_time)
    total_time_s = perf_counter() - start_time
    time_to_best_s, time_to_target_s = _measure_times(
        history, cost, instance.optimal_cost
    )
    return Run(seed, cost, history, total_time_s, time_to_best_s, time_to_target_s)


def _measure_times(
    history: SearchHistory, final_cost: int, target_cost: int
) -> tuple[float | None, float | None]:
    time_to_best_s = None
    time_to_target_s = None
    for _, best_cost, elapsed_time, _ in history:
        if time_to_best_s is None and best_cost == final_cost:
            time_to_best_s = elapsed_time
        if time_to_target_s is None and best_cost <= target_cost:
            time_to_target_s = elapsed_time
        if time_to_best_s is not None and time_to_target_s is not None:
            break

    return time_to_best_s, time_to_target_s
