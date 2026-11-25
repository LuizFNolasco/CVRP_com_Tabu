import random
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from time import perf_counter
from typing import NamedTuple

from ..routes import (
    Coordinates,
    Route,
    Vehicle,
    copy_routes,
    reverse_segment,
    route_cost,
)


class TabuTenure(NamedTuple):
    minimum: int
    maximum: int


DEFAULT_TABU_TENURE = TabuTenure(minimum=3, maximum=7)


class SearchHistoryEntry(NamedTuple):
    iteration: int
    best_cost: int
    elapsed_time_s: float
    current_cost: int


type SearchHistory = list[SearchHistoryEntry]


class SearchResult(NamedTuple):
    best_routes: list[Route]
    best_cost: int
    history: SearchHistory


class ReversalKey(NamedTuple):
    lower_customer: int
    upper_customer: int


class EvaluatedReversal(NamedTuple):
    start: int
    end: int
    route: Route
    cost: int
    key: ReversalKey


@dataclass(frozen=True)
class SearchParameters:
    max_iterations: int
    stagnation_limit: int
    tabu_tenure: TabuTenure = DEFAULT_TABU_TENURE
    time_limit_s: float | None = None
    stop_at_target: bool = False


class SearchProgress:
    def __init__(
        self,
        fleet: Iterable[Vehicle],
        coordinates: Coordinates,
        depot: int,
        start_time: float,
    ) -> None:
        self.best_routes = copy_routes(fleet)
        self.best_cost = sum(
            route_cost(route, coordinates, depot) for route in self.best_routes
        )
        self.iterations_without_improvement = 0
        self.history: SearchHistory = []
        self._start_time = start_time
        self._record_cost(0, self.best_cost)

    def time_limit_reached(self, time_limit_s: float | None) -> bool:
        return time_limit_s is not None and self._elapsed_time_s() >= time_limit_s

    def target_reached(self, target_cost: int | None) -> bool:
        return target_cost is not None and self.best_cost <= target_cost

    def record_iteration(
        self, iteration: int, fleet: Iterable[Vehicle], current_cost: int
    ) -> None:
        if current_cost < self.best_cost:
            self.best_cost = current_cost
            self.best_routes = copy_routes(fleet)
            self.iterations_without_improvement = 0
        else:
            self.iterations_without_improvement += 1

        self._record_cost(iteration, current_cost)

    def result(self) -> SearchResult:
        return SearchResult(
            best_routes=self.best_routes,
            best_cost=self.best_cost,
            history=self.history,
        )

    def _record_cost(self, iteration: int, current_cost: int) -> None:
        self.history.append(
            SearchHistoryEntry(
                iteration=iteration,
                best_cost=self.best_cost,
                elapsed_time_s=self._elapsed_time_s(),
                current_cost=current_cost,
            )
        )

    def _elapsed_time_s(self) -> float:
        return perf_counter() - self._start_time


def evaluate_reversals(
    route: Route, coordinates: Coordinates, depot: int, cost_without_route: int
) -> Iterator[EvaluatedReversal]:
    for start in range(len(route) - 1):
        for end in range(start + 1, len(route)):
            updated_route = reverse_segment(route, start, end)
            cost = cost_without_route + route_cost(updated_route, coordinates, depot)
            first, last = route[start], route[end]
            key = ReversalKey(min(first, last), max(first, last))
            yield EvaluatedReversal(start, end, updated_route, cost, key)


def is_tabu[Key](
    tabu_list: Mapping[Key, int],
    key: Key,
    iteration: int,
    cost: int,
    best_cost: int,
) -> bool:
    blocked = iteration <= tabu_list.get(key, 0)
    return blocked and cost >= best_cost


def register_tabu_move[Key](
    tabu_list: dict[Key, int],
    keys: Iterable[Key],
    iteration: int,
    tenure: tuple[int, int],
) -> None:
    expiration = iteration + random.randint(*tenure)
    for key in keys:
        tabu_list[key] = expiration
