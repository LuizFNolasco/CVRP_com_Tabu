import csv
from itertools import product
from pathlib import Path

import pytest
from typer.testing import CliRunner

from cvrp_tabu_search.cli import app

BENCHMARKS_DIR = Path(__file__).resolve().parents[2] / "benchmarks" / "selected"
INSTANCE_OPTIMA = {"A-n32-k5": "784", "A-n80-k10": "1763", "B-n78-k10": "1221"}


@pytest.mark.parametrize(
    ("level_options", "minimums", "maximums"),
    [
        ([], (1, 3, 6), (7, 11, 15)),
        (
            ["--tabu-tenure-min", "2", "4", "6", "--tabu-tenure-max", "8", "12", "16"],
            (2, 4, 6),
            (8, 12, 16),
        ),
    ],
    ids=["default-levels", "custom-levels"],
)
def test_calibration_writes_one_row_per_instance_and_combination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    level_options: list[str],
    minimums: tuple[int, ...],
    maximums: tuple[int, ...],
) -> None:
    monkeypatch.chdir(tmp_path)
    output_dir = tmp_path / "calibration reports"

    result = CliRunner().invoke(
        app,
        [
            "calibrate",
            "--benchmarks-dir",
            str(BENCHMARKS_DIR),
            "--output-dir",
            str(output_dir),
            "--time-limit",
            "0",
            *level_options,
        ],
    )

    assert result.exit_code == 0, result.output
    [report] = output_dir.glob("*/calibration.csv")
    assert str(report) in result.output
    assert "[27/27]" in result.output
    with report.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        assert reader.fieldnames is not None
        assert reader.fieldnames[-1] == "best_cost"

    configurations = [
        (row["instance"], int(row["tabu_tenure_min"]), int(row["tabu_tenure_max"]))
        for row in rows
    ]
    assert configurations == list(product(INSTANCE_OPTIMA, minimums, maximums))
    assert {row["seed"] for row in rows} == {"1"}
    assert {row["jobs"] for row in rows} == {"1"}
    assert {row["time_limit_s"] for row in rows} == {"0.0"}
    for row in rows:
        assert row["optimal_cost"] == INSTANCE_OPTIMA[row["instance"]]
        assert int(row["best_cost"]) > int(row["optimal_cost"])


@pytest.mark.parametrize(
    ("options", "error_message"),
    [
        (["--tabu-tenure-min", "2", "2", "4"], "three distinct values"),
        (["--tabu-tenure-max", "5", "8", "10"], "at least 6"),
        (["--jobs", "0"], "--jobs"),
    ],
    ids=["duplicate-level", "invalid-range", "zero-jobs"],
)
def test_calibration_rejects_invalid_options_before_creating_reports(
    tmp_path: Path, options: list[str], error_message: str
) -> None:
    result = CliRunner().invoke(
        app,
        [
            "calibrate",
            "--benchmarks-dir",
            str(BENCHMARKS_DIR),
            "--output-dir",
            str(tmp_path),
            *options,
        ],
    )

    assert result.exit_code == 2, result.output
    assert error_message in result.output
    assert list(tmp_path.iterdir()) == []
