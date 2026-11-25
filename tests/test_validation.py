from cvrp_tabu_search.instances import Instance
from cvrp_tabu_search.validation import check_solution

# Depot at (0, 0); customers 2, 3 and 4 with demands 3, 2 and 4; capacity 6.
INSTANCE = Instance(
    "toy-n4-k2", 0, 2, 4, 6, [[0, 0], [3, 0], [0, 4], [3, 4]], [0, 3, 2, 4], 1
)


def test_a_feasible_solution_reports_its_cost() -> None:
    check = check_solution(INSTANCE, [[2], [3, 4]])

    # Route 1: 3 + 3. Route 2: 4 + 3 + 5.
    assert check == (18, ())
    assert check.feasible


def test_missing_repeated_unknown_and_overloaded_customers_are_reported() -> None:
    check = check_solution(INSTANCE, [[2, 4, 2], [9]])

    assert not check.feasible
    assert check.problems == (
        "customers not visited: [3]",
        "customers visited more than once: [2]",
        "invalid nodes in routes: [9]",
        "route 1 load 10 exceeds capacity 6",
    )
    assert check.cost == -1


def test_more_routes_than_vehicles_is_infeasible() -> None:
    check = check_solution(INSTANCE, [[2], [3], [4]])

    assert check.problems == ("3 routes exceed the fleet of 2",)
