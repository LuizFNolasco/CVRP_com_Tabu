import csv
import json
from pathlib import Path

from typer.testing import CliRunner

import cvrp_tabu_search
from cvrp_tabu_search.cli import app

DEFAULT_INSTANCE = Path(cvrp_tabu_search.__file__).parent / "data" / "A-n32-k5.vrp"


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def test_ttt_runs_stop_at_the_target_and_write_the_report(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app,
        [
            "ttt",
            "--instance",
            str(DEFAULT_INSTANCE),
            "--target-cost",
            "2000",
            "--runs",
            "3",
            "--time-limit",
            "30",
            "--output-dir",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0, result.output
    [report_directory] = tmp_path.iterdir()
    rows = _rows(report_directory / "ttt_runs_A-n32-k5.csv")
    assert sorted(int(row["seed"]) for row in rows) == [1, 2, 3]
    assert {row["reached_target"] for row in rows} == {"True"}
    assert all(int(row["best_cost"]) <= 2000 for row in rows)
    assert (report_directory / "ttt_A-n32-k5.png").is_file()
    assert (report_directory / "environment.json").is_file()


def test_ttt_records_runs_that_miss_the_target(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app,
        [
            "ttt",
            "--instance",
            str(DEFAULT_INSTANCE),
            "--target-cost",
            "1",
            "--runs",
            "2",
            "--time-limit",
            "0.2",
            "--output-dir",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0, result.output
    [report_directory] = tmp_path.iterdir()
    rows = _rows(report_directory / "ttt_runs_A-n32-k5.csv")
    assert {row["reached_target"] for row in rows} == {"False"}
    assert {row["time_to_target_s"] for row in rows} == {""}


def test_plot_solution_validates_and_draws_the_routes(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app,
        [
            "plot-solution",
            "--instance",
            str(DEFAULT_INSTANCE),
            "--target-cost",
            "2000",
            "--time-limit",
            "30",
            "--output-dir",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0, result.output
    [report_directory] = tmp_path.iterdir()
    solution = json.loads(
        (report_directory / "solution_A-n32-k5.json").read_text(encoding="utf-8")
    )
    assert solution["feasible"] is True
    assert solution["cost"] <= 2000
    assert (report_directory / "solution_A-n32-k5.png").is_file()
