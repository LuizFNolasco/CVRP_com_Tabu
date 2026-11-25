from collections.abc import Mapping, Sequence
from typing import NamedTuple

from ..routes import Coordinates, Route, Vehicle, route_cost
from .shared import (
    ReversalKey,
    SearchProgress,
    SearchResult,
    evaluate_reversals,
    is_tabu,
    register_tabu_move,
)


class ReversalMove(NamedTuple):
    vehicle_index: int
    start: int
    end: int
    route: Route
    cost: int
    key: ReversalKey


def find_best_neighbor(
    fleet: Sequence[Vehicle],
    current_cost: int,
    coordinates: Coordinates,
    depot: int,
    tabu_list: Mapping[tuple[int, int], int],
    iteration: int,
    best_cost: int,
) -> ReversalMove | None:
    best_move = None
    for vehicle_index, vehicle in enumerate(fleet):
        route = vehicle.route
        if len(route) < 2:
            continue

        cost_without_route = current_cost - route_cost(route, coordinates, depot)
        reversals = evaluate_reversals(route, coordinates, depot, cost_without_route)
        for reversal in reversals:
            if is_tabu(tabu_list, reversal.key, iteration, reversal.cost, best_cost):
                continue

            if best_move is None or reversal.cost < best_move.cost:
                best_move = ReversalMove(
                    vehicle_index=vehicle_index,
                    start=reversal.start,
                    end=reversal.end,
                    route=reversal.route,
                    cost=reversal.cost,
                    key=reversal.key,
                )

    return best_move


def tabu_search(
    fleet: list[Vehicle],
    coordinates: Coordinates,
    depot: int,
    tabu_tenure: tuple[int, int],
    max_iterations: int,
    stagnation_limit: int,
    start_time: float,
    *,
    time_limit_s: float | None = None,
    target_cost: int | None = None,
) -> SearchResult:
    progress = SearchProgress(fleet, coordinates, depot, start_time)
    current_cost = progress.best_cost
    tabu_list: dict[tuple[int, int], int] = {}

    for iteration in range(1, max_iterations + 1):
        if progress.time_limit_reached(time_limit_s) or progress.target_reached(
            target_cost
        ):
            break

        move = find_best_neighbor(
            fleet,
            current_cost,
            coordinates,
            depot,
            tabu_list,
            iteration,
            progress.best_cost,
        )
        if move is None:
            break

        fleet[move.vehicle_index].route = move.route
        current_cost = move.cost
        register_tabu_move(tabu_list, (move.key,), iteration, tabu_tenure)
        progress.record_iteration(iteration, fleet, current_cost)
        if progress.iterations_without_improvement >= stagnation_limit:
            break

    return progress.result()
