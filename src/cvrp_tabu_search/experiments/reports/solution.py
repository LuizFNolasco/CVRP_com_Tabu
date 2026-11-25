import json
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt

from ...instances import Instance
from ...routes import Route
from ...validation import SolutionCheck, check_solution
from .plots import save_figure


def write_solution_json(
    path: Path, instance: Instance, routes: Sequence[Route]
) -> SolutionCheck:
    check = check_solution(instance, routes)
    content = {
        "instance": instance.name,
        "optimal_cost": instance.optimal_cost,
        "cost": check.cost,
        "feasible": check.feasible,
        "problems": list(check.problems),
        "routes": [list(route) for route in routes],
    }
    path.write_text(json.dumps(content, indent=2) + "\n", encoding="utf-8")
    return check


def plot_solution(instance: Instance, routes: Sequence[Route], path: Path) -> None:
    figure, axes = plt.subplots(figsize=(6, 6))
    colors = plt.get_cmap("tab10")
    used_routes = [route for route in routes if route]

    for index, route in enumerate(used_routes):
        nodes = (instance.depot, *route, instance.depot)
        xs = [instance.coordinates[node - 1][0] for node in nodes]
        ys = [instance.coordinates[node - 1][1] for node in nodes]
        load = sum(instance.demands[customer - 1] for customer in route)
        axes.plot(
            xs,
            ys,
            "-o",
            color=colors(index % 10),
            markersize=4,
            linewidth=1.2,
            label=f"Rota {index + 1} (carga {load}/{instance.capacity})",
        )
        for customer in route:
            x, y = instance.coordinates[customer - 1]
            axes.annotate(
                str(customer),
                (x, y),
                textcoords="offset points",
                xytext=(3, 3),
                fontsize=6,
            )

    depot_x, depot_y = instance.coordinates[instance.depot - 1]
    axes.plot(depot_x, depot_y, "ks", markersize=9, label="Depósito")
    axes.set_aspect("equal")
    axes.set_xticks([])
    axes.set_yticks([])
    axes.legend(fontsize=7, loc="best")
    save_figure(figure, path)
