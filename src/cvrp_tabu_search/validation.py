from collections import Counter
from collections.abc import Sequence
from typing import NamedTuple

from .instances import Instance
from .routes import Route, route_cost


class SolutionCheck(NamedTuple):
    cost: int
    problems: tuple[str, ...]

    @property
    def feasible(self) -> bool:
        return not self.problems


def check_solution(instance: Instance, routes: Sequence[Route]) -> SolutionCheck:
    """Check a CVRP solution against the instance data, independently of the search."""
    customers = {
        node for node in range(1, instance.dimension + 1) if node != instance.depot
    }
    visits = Counter(customer for route in routes for customer in route)
    missing = sorted(customers - visits.keys())
    repeated = sorted(customer for customer, count in visits.items() if count > 1)
    unknown = sorted(customer for customer in visits if customer not in customers)

    problems: list[str] = []
    if missing:
        problems.append(f"customers not visited: {missing}")
    if repeated:
        problems.append(f"customers visited more than once: {repeated}")
    if unknown:
        problems.append(f"invalid nodes in routes: {unknown}")

    used_routes = [route for route in routes if route]
    if len(used_routes) > instance.vehicle_count:
        problems.append(
            f"{len(used_routes)} routes exceed the fleet of {instance.vehicle_count}"
        )

    for index, route in enumerate(routes, start=1):
        load = sum(
            instance.demands[customer - 1]
            for customer in route
            if 1 <= customer <= instance.dimension
        )
        if load > instance.capacity:
            problems.append(
                f"route {index} load {load} exceeds capacity {instance.capacity}"
            )

    cost = -1
    if not unknown:
        cost = sum(
            route_cost(route, instance.coordinates, instance.depot)
            for route in used_routes
        )

    return SolutionCheck(cost, tuple(problems))
