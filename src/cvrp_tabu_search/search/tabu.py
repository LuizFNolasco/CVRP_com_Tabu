from collections.abc import Sequence
from functools import partial

from ..routes import Coordinates, Route, Vehicle
from .moves import Move, TabuKeyTuple
from .neighborhood import find_best_neighbor
from .shared import (
    SearchProgress,
    SearchResult,
    register_tabu_move,
)


def tabu_search(
    fleet: list[Vehicle],
    coordinates: Coordinates,
    depot: int,
    tabu_tenure: tuple[int, int],
    max_iterations: int,
    stagnation_limit: int,
    start_time: float,
    demands: Sequence[int],
    use_phases: bool = True,
    *,
    time_limit_s: float | None = None,
    target_cost: int | None = None,
) -> SearchResult:
    progress = SearchProgress(fleet, coordinates, depot, start_time)
    current_cost = progress.best_cost
    tabu_list: dict[TabuKeyTuple, int] = {}
    frequencies: dict[int, int] = {}
    intensification_steps = 0
    select_neighbor = partial(
        find_best_neighbor, coordinates=coordinates, depot=depot, demands=demands
    )

    for iteration in range(1, max_iterations + 1):
        if progress.time_limit_reached(time_limit_s) or progress.target_reached(
            target_cost
        ):
            break

        stagnation = progress.iterations_without_improvement
        if use_phases and stagnation == 10:
            _restore_routes(fleet, progress.best_routes, demands)
            current_cost = progress.best_cost
            intensification_steps = 3

        intensify = intensification_steps > 0
        diversify = use_phases and stagnation > 0 and stagnation % 20 == 0
        move = select_neighbor(
            fleet,
            current_cost,
            tabu_list=tabu_list,
            iteration=iteration,
            best_cost=progress.best_cost,
            frequencies=frequencies,
            intensify=intensify,
            diversify=diversify,
        )

        if move is None and (intensify or diversify):
            if progress.time_limit_reached(time_limit_s):
                break

            intensification_steps = 0
            move = select_neighbor(
                fleet,
                current_cost,
                tabu_list=tabu_list,
                iteration=iteration,
                best_cost=progress.best_cost,
                frequencies=frequencies,
            )

        if move is None:
            break

        _apply_move(fleet, move)
        current_cost = move.cost
        register_tabu_move(tabu_list, move.tabu_keys, iteration, tabu_tenure)
        for customer in move.customers:
            frequencies[customer] = frequencies.get(customer, 0) + 1

        if intensification_steps > 0:
            intensification_steps -= 1

        progress.record_iteration(iteration, fleet, current_cost)
        if progress.iterations_without_improvement >= stagnation_limit:
            break

    return progress.result()


def _apply_move(fleet: list[Vehicle], move: Move) -> None:
    for update in move.route_updates:
        vehicle = fleet[update.vehicle_index]
        vehicle.route = update.route
        vehicle.load = update.load


def _restore_routes(
    fleet: Sequence[Vehicle], routes: Sequence[Route], demands: Sequence[int]
) -> None:
    for vehicle, route in zip(fleet, routes, strict=True):
        vehicle.route = route.copy()
        vehicle.load = sum(demands[customer - 1] for customer in route)
