import random
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from time import perf_counter
from typing import Protocol

from ..instances import Instance
from ..routes import Coordinates, Route, Vehicle, build_initial_solution
from ..search.shared import SearchHistory, SearchParameters, SearchResult
from ..search.tabu import tabu_search


class SearchAlgorithm(Protocol):
    def __call__(
        self,
        fleet: list[Vehicle],
        coordinates: Coordinates,
        depot: int,
        tabu_tenure: tuple[int, int],
        max_iterations: int,
        stagnation_limit: int,
        start_time: float,
        /,
        *,
        time_limit_s: float | None = None,
        target_cost: int | None = None,
    ) -> SearchResult: ...


@dataclass(frozen=True)
class Run:
    seed: int
    cost: int
    history: SearchHistory
    total_time_s: float
    time_to_best_s: float | None
    time_to_target_s: float | None
    routes: list[Route] = field(default_factory=list)

    @property
    def iterations(self) -> int:
        # The first history entry is the initial solution.
        return max(len(self.history) - 1, 0)


def tabu_search_for(instance: Instance, *, use_phases: bool = True) -> SearchAlgorithm:
    return partial(tabu_search, demands=instance.demands, use_phases=use_phases)


def run_once(
    instance: Instance,
    search: SearchAlgorithm,
    parameters: SearchParameters,
    *,
    seed: int,
) -> Run | None:
    run_search = _configure_search(instance, search, parameters)
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

    routes, cost, history = run_search(fleet, start_time)
    total_time_s = perf_counter() - start_time
    time_to_best_s, time_to_target_s = _measure_times(
        history, cost, instance.optimal_cost
    )
    return Run(
        seed=seed,
        cost=cost,
        history=history,
        total_time_s=total_time_s,
        time_to_best_s=time_to_best_s,
        time_to_target_s=time_to_target_s,
        routes=routes,
    )


def _configure_search(
    instance: Instance, search: SearchAlgorithm, parameters: SearchParameters
) -> Callable[[list[Vehicle], float], SearchResult]:
    if parameters.time_limit_s is not None:
        search = partial(search, time_limit_s=parameters.time_limit_s)
    if parameters.stop_at_target:
        search = partial(search, target_cost=instance.optimal_cost)

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

    return run_search


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
