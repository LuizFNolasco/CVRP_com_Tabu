import json
from pathlib import Path

from cvrp_tabu_search.experiments.reports.solution import (
    plot_solution,
    write_solution_json,
)
from cvrp_tabu_search.instances import Instance

INSTANCE = Instance(
    "toy-n4-k2", 0, 2, 4, 6, [[0, 0], [3, 0], [0, 4], [3, 4]], [0, 3, 2, 4], 1
)


def test_solution_report_records_routes_cost_and_feasibility(tmp_path: Path) -> None:
    path = tmp_path / "solution.json"

    check = write_solution_json(path, INSTANCE, [[2], [3, 4]])

    assert check.feasible
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "instance": "toy-n4-k2",
        "optimal_cost": 0,
        "cost": 18,
        "feasible": True,
        "problems": [],
        "routes": [[2], [3, 4]],
    }


def test_solution_plot_is_a_png(tmp_path: Path) -> None:
    path = tmp_path / "solution.png"

    plot_solution(INSTANCE, [[2], [3, 4]], path)

    assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
