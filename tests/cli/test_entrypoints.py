import csv
import os
import subprocess
import sys
from collections.abc import Sequence
from itertools import product
from pathlib import Path

import pytest


def _run_process(
    arguments: Sequence[str], working_directory: Path
) -> subprocess.CompletedProcess[str]:
    environment = os.environ | {
        "MPLBACKEND": "Agg",
        "MPLCONFIGDIR": str(working_directory.parent / "matplotlib"),
        "PYTHONUTF8": "1",
    }
    return subprocess.run(
        arguments,
        cwd=working_directory,
        env=environment,
        capture_output=True,
        encoding="utf-8",
        timeout=30,
        check=False,
    )


@pytest.mark.parametrize(
    "command",
    [
        [
            str(
                Path(sys.executable).with_name(
                    "cvrp.exe" if os.name == "nt" else "cvrp"
                )
            )
        ],
        [sys.executable, "-m", "cvrp_tabu_search"],
    ],
    ids=["console-script", "package-module"],
)
def test_entrypoint_exposes_the_cli(tmp_path: Path, command: list[str]) -> None:
    result = _run_process([*command, "--help"], tmp_path)

    assert result.returncode == 0, result.stderr
    assert "baseline" in result.stdout
    assert "tabu-search" in result.stdout
    assert "calibrate" in result.stdout
    assert list(tmp_path.iterdir()) == []


def test_package_module_runs_the_full_calibration_grid_in_parallel(
    tmp_path: Path,
) -> None:
    benchmarks_dir = Path(__file__).resolve().parents[2] / "benchmarks" / "selected"
    command = [
        sys.executable,
        "-m",
        "cvrp_tabu_search",
        "calibrate",
        "--benchmarks-dir",
        str(benchmarks_dir),
        "--time-limit",
        "0",
        "--jobs",
        "2",
    ]

    result = _run_process(command, tmp_path)

    assert result.returncode == 0, result.stderr
    assert "[27/27]" in result.stdout
    [report] = (tmp_path / "results").glob("*/calibration.csv")
    with report.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    instances = ("A-n32-k5", "A-n80-k10", "B-n78-k10")
    expected_configurations = set(product(instances, (1, 3, 6), (7, 11, 15)))
    actual_configurations = {
        (row["instance"], int(row["tabu_tenure_min"]), int(row["tabu_tenure_max"]))
        for row in rows
    }
    assert len(rows) == len(expected_configurations)
    assert actual_configurations == expected_configurations
    assert {row["jobs"] for row in rows} == {"2"}
    assert {row["seed"] for row in rows} == {"1"}
