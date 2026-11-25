import random
from time import perf_counter

from cvrp_tabu_search.experiments.execution import run_once, tabu_search_for
from cvrp_tabu_search.instances import Instance, load_default_instance
from cvrp_tabu_search.routes import Vehicle, build_initial_solution
from cvrp_tabu_search.search.shared import SearchParameters, TabuTenure
from cvrp_tabu_search.search.tabu import tabu_search


def _initial_fleet() -> tuple[Instance, list[Vehicle]]:
    instance = load_default_instance()
    fleet = build_initial_solution(
        instance.vehicle_count,
        instance.dimension,
        instance.capacity,
        instance.demands,
        instance.depot,
    )
    assert fleet is not None
    return instance, fleet


def test_tabu_search_stops_as_soon_as_the_target_is_reached() -> None:
    instance, fleet = _initial_fleet()
    random.seed(1)

    result = tabu_search(
        fleet,
        instance.coordinates,
        instance.depot,
        (3, 7),
        1000,
        1000,
        perf_counter(),
        instance.demands,
        target_cost=2000,
    )

    assert result.best_cost <= 2000
    assert all(entry.best_cost > 2000 for entry in result.history[:-1])


def test_tabu_search_skips_the_scan_when_the_initial_solution_meets_the_target() -> (
    None
):
    instance, fleet = _initial_fleet()

    result = tabu_search(
        fleet,
        instance.coordinates,
        instance.depot,
        (3, 7),
        1000,
        1000,
        perf_counter(),
        instance.demands,
        target_cost=10_000,
    )

    assert len(result.history) == 1


def test_run_keeps_the_best_routes_and_counts_iterations() -> None:
    instance = load_default_instance()
    parameters = SearchParameters(5, 5, tabu_tenure=TabuTenure(3, 7))

    run = run_once(instance, tabu_search_for(instance), parameters, seed=1)

    assert run is not None
    assert run.iterations == 5
    customers = sorted(customer for route in run.routes for customer in route)
    assert customers == [
        node for node in range(1, instance.dimension + 1) if node != instance.depot
    ]


def test_stop_at_target_uses_the_known_optimum_of_the_instance() -> None:
    instance = load_default_instance()
    parameters = SearchParameters(
        max_iterations=10**6,
        stagnation_limit=10**6,
        tabu_tenure=TabuTenure(1, 11),
        time_limit_s=30.0,
        stop_at_target=True,
    )

    run = run_once(instance, tabu_search_for(instance), parameters, seed=1)

    assert run is not None
    assert run.cost == instance.optimal_cost == 784
    assert run.time_to_target_s is not None
