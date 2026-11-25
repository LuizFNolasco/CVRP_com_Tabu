import csv
import json
import random
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from textwrap import dedent

import matplotlib
import pytest
from typer.testing import CliRunner

from cvrp_tabu_search.cli import app


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as report:
        return list(csv.DictReader(report))


def _single_timestamped_directory(output_dir: Path) -> Path:
    [report_directory] = output_dir.iterdir()
    assert report_directory.is_dir()
    datetime.strptime(report_directory.name, "%Y-%m-%d_%H-%M-%S")
    return report_directory


@pytest.fixture
def working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Path]:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.chdir(workspace)
    random_state = random.getstate()
    try:
        with matplotlib.rc_context():
            yield workspace
    finally:
        random.setstate(random_state)


def test_tabu_search_without_phases_uses_its_own_report_prefix(
    working_directory: Path,
) -> None:
    result = CliRunner().invoke(
        app, ["tabu-search", "--no-phases", "--max-iterations", "5"]
    )

    assert result.exit_code == 0, result.output
    report_directory = _single_timestamped_directory(working_directory / "results")
    assert (report_directory / "runs_tabu_nophases_A-n32-k5.csv").is_file()


@pytest.mark.parametrize(
    (
        "subcommand",
        "prefix",
        "expected_cost",
        "iteration_limit",
        "stagnation_limit",
    ),
    [
        ("baseline", "", 1470, 10, 10),
        ("tabu-search", "tabu_", 915, 60, 30),
    ],
    ids=["baseline", "advanced"],
)
def test_subcommand_generates_reports_in_the_default_directory(
    working_directory: Path,
    subcommand: str,
    prefix: str,
    expected_cost: int,
    iteration_limit: int,
    stagnation_limit: int,
) -> None:
    result = CliRunner().invoke(app, [subcommand])

    assert result.exit_code == 0, result.output
    repetitions = list(range(1, 6))
    output = result.output.splitlines()
    assert len(output) == len(repetitions)
    for repetition, line in zip(repetitions, output, strict=True):
        assert line.startswith(f"Run {repetition}: cost {expected_cost} | time ")

    suffix = f"{prefix}A-n32-k5"
    output_dir = working_directory / "results"
    report_directory = _single_timestamped_directory(output_dir)
    runs_path = report_directory / f"runs_{suffix}.csv"
    history_path = report_directory / f"history_{suffix}.csv"
    plots = [
        report_directory / f"{kind}_{suffix}.png" for kind in ("convergence", "ttt")
    ]
    assert list(working_directory.iterdir()) == [output_dir]
    environment_path = report_directory / "environment.json"
    assert set(report_directory.iterdir()) == {
        runs_path,
        history_path,
        environment_path,
        report_directory / f"solution_{suffix}.json",
        *plots,
    }
    for plot in plots:
        assert plot.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"), plot

    runs = _read_csv(runs_path)
    assert [int(run["seed"]) for run in runs] == repetitions
    assert {int(run["best_cost"]) for run in runs} == {expected_cost}
    assert {int(run["max_iterations"]) for run in runs} == {iteration_limit}
    assert {int(run["stagnation_limit"]) for run in runs} == {stagnation_limit}
    assert {int(run["tabu_tenure_min"]) for run in runs} == {3}
    assert {int(run["tabu_tenure_max"]) for run in runs} == {7}
    assert {run["time_limit_s"] for run in runs} == {""}

    solution = json.loads(
        (report_directory / f"solution_{suffix}.json").read_text(encoding="utf-8")
    )
    assert solution["feasible"] is True
    assert solution["cost"] == expected_cost

    history = _read_csv(history_path)
    assert [int(row["iteration"]) for row in history] == list(
        range(1, len(history) + 1)
    )
    assert int(history[-1]["best_cost"]) == expected_cost


@pytest.fixture
def rectangle_path(tmp_path: Path) -> Path:
    instance = tmp_path / "rectangle instance.vrp"
    instance.write_text(
        dedent(
            """
            NAME: rectangle-n4-k1
            DIMENSION: 4
            CAPACITY: 3
            NODE_COORD_SECTION
            1 0 0
            2 4 0
            3 0 3
            4 4 3
            DEMAND_SECTION
            1 0
            2 1
            3 1
            4 1
            DEPOT_SECTION
            1
            -1
            EOF
            """
        ),
        encoding="utf-8",
    )
    return instance


@pytest.mark.parametrize(
    ("subcommand", "prefix", "iterations", "expected_cost"),
    [
        ("baseline", "", 0, 18),
        ("tabu-search", "tabu_", 1, 14),
    ],
    ids=["baseline", "advanced"],
)
def test_cli_uses_custom_instance_target_and_search_limits(
    working_directory: Path,
    rectangle_path: Path,
    subcommand: str,
    prefix: str,
    iterations: int,
    expected_cost: int,
) -> None:
    arguments = [
        subcommand,
        "--instance",
        str(rectangle_path),
        "--target-cost",
        "14",
        "--max-iterations",
        str(iterations),
        "--stagnation-limit",
        "2",
        "--tabu-tenure-min",
        "8",
        "--tabu-tenure-max",
        "8",
    ]

    result = CliRunner().invoke(app, arguments)

    assert result.exit_code == 0, result.output
    report_directory = _single_timestamped_directory(working_directory / "results")
    runs = _read_csv(report_directory / f"runs_{prefix}{rectangle_path.stem}.csv")
    assert len(runs) == 5
    assert {run["optimal_cost"] for run in runs} == {"14"}
    assert {run["best_cost"] for run in runs} == {str(expected_cost)}
    assert {run["max_iterations"] for run in runs} == {str(iterations)}
    assert {run["stagnation_limit"] for run in runs} == {"2"}
    assert {run["tabu_tenure_min"] for run in runs} == {"8"}
    assert {run["tabu_tenure_max"] for run in runs} == {"8"}
    history = _read_csv(report_directory / f"history_{prefix}{rectangle_path.stem}.csv")
    assert len(history) == iterations
    assert (runs[0]["time_to_target_s"] != "") is (expected_cost == 14)


@pytest.mark.parametrize(
    ("subcommand", "prefix"), [("baseline", ""), ("tabu-search", "tabu_")]
)
def test_cli_records_zero_time_budget_and_keeps_the_initial_solution(
    working_directory: Path, rectangle_path: Path, subcommand: str, prefix: str
) -> None:
    arguments = [
        subcommand,
        "--instance",
        str(rectangle_path),
        "--target-cost",
        "14",
        "--time-limit",
        "0",
    ]

    result = CliRunner().invoke(app, arguments)

    assert result.exit_code == 0, result.output
    report_directory = _single_timestamped_directory(working_directory / "results")
    runs = _read_csv(report_directory / f"runs_{prefix}{rectangle_path.stem}.csv")
    assert len(runs) == 5
    assert {run["time_limit_s"] for run in runs} == {"0.0"}
    assert {run["best_cost"] for run in runs} == {"18"}
    history = _read_csv(report_directory / f"history_{prefix}{rectangle_path.stem}.csv")
    assert history == []


@pytest.mark.parametrize(
    ("subcommand", "prefix", "absolute_path"),
    [("baseline", "", False), ("tabu-search", "tabu_", True)],
    ids=["baseline-relative-path", "tabu-search-absolute-path"],
)
def test_subcommand_saves_all_reports_in_the_requested_directory(
    working_directory: Path, subcommand: str, prefix: str, absolute_path: bool
) -> None:
    relative_path = Path("custom results") / "experiment reports"
    output_dir = working_directory / relative_path
    requested_path = output_dir if absolute_path else relative_path

    result = CliRunner().invoke(
        app, [subcommand, "--max-iterations", "0", "--output-dir", str(requested_path)]
    )

    assert result.exit_code == 0, result.output
    suffix = f"{prefix}A-n32-k5"
    report_directory = _single_timestamped_directory(output_dir)
    assert {path.name for path in report_directory.iterdir()} == {
        f"runs_{suffix}.csv",
        f"history_{suffix}.csv",
        f"convergence_{suffix}.png",
        f"ttt_{suffix}.png",
        "environment.json",
        f"solution_{suffix}.json",
    }
    assert list(working_directory.iterdir()) == [output_dir.parent]


def test_subcommand_rejects_a_file_as_the_output_directory(
    working_directory: Path,
) -> None:
    existing_file = working_directory / "existing.txt"
    existing_file.write_text("existing content", encoding="utf-8")

    result = CliRunner().invoke(app, ["baseline", "--output-dir", existing_file.name])

    assert result.exit_code == 2
    assert "--output-dir" in result.output
    assert existing_file.read_text(encoding="utf-8") == "existing content"
    assert list(working_directory.iterdir()) == [existing_file]


def test_subcommand_help_lists_the_available_options() -> None:
    result = CliRunner().invoke(app, ["baseline", "--help"])

    assert result.exit_code == 0
    for option in (
        "--instance",
        "--target-cost",
        "--max-iterations",
        "--stagnation-limit",
        "--time-limit",
        "--tabu-tenure-min",
        "--tabu-tenure-max",
        "--output-dir",
    ):
        assert option in result.output
    assert "default: ." not in result.output
    assert "default: results" in result.output


def test_custom_instance_requires_a_reference_cost(
    rectangle_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(app, ["baseline", "--instance", str(rectangle_path)])

    assert result.exit_code == 2
    assert "--target-cost" in result.output
    assert list(tmp_path.iterdir()) == [rectangle_path]


@pytest.mark.parametrize(
    ("options", "error_option"),
    [
        (["--target-cost", "20"], "--target-cost"),
        (["--target-cost", "-1"], "--target-cost"),
        (["--max-iterations", "-1"], "--max-iterations"),
        (["--stagnation-limit", "0"], "--stagnation-limit"),
        (["--time-limit=-0.5"], "--time-limit"),
        (["--time-limit", "nan"], "--time-limit"),
        (["--time-limit", "inf"], "--time-limit"),
        (["--tabu-tenure-min", "-1"], "--tabu-tenure-min"),
        (["--tabu-tenure-max", "-1"], "--tabu-tenure-max"),
        (["--tabu-tenure-min", "8", "--tabu-tenure-max", "4"], "--tabu-tenure-max"),
        (["--instance", "missing.vrp", "--target-cost", "14"], "--instance"),
    ],
    ids=[
        "target-without-instance",
        "negative-target",
        "negative-limit",
        "zero-stagnation",
        "negative-time-limit",
        "nan-time-limit",
        "infinite-time-limit",
        "negative-tenure-minimum",
        "negative-tenure-maximum",
        "reversed-tenure-bounds",
        "missing-file",
    ],
)
def test_invalid_options_report_usage_errors(
    options: list[str],
    error_option: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(app, ["tabu-search", *options])

    assert result.exit_code == 2
    assert error_option in result.output
    assert list(tmp_path.iterdir()) == []


def test_cli_rejects_an_incomplete_instance_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    path = Path("incomplete.vrp")
    path.write_text(
        "NAME: incomplete-n2-k1\nDIMENSION: 2\nCAPACITY: 1\nEOF\n", encoding="utf-8"
    )

    result = CliRunner().invoke(
        app, ["baseline", "--instance", str(path), "--target-cost", "14"]
    )

    assert result.exit_code == 2
    assert "declared dimension" in result.output
    assert list(tmp_path.iterdir()) == [tmp_path / path]


def test_cli_returns_failure_when_the_initial_fleet_is_infeasible(
    rectangle_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    contents = rectangle_path.read_text(encoding="utf-8")
    rectangle_path.write_text(
        contents.replace("CAPACITY: 3", "CAPACITY: 1"), encoding="utf-8"
    )

    result = CliRunner().invoke(
        app, ["baseline", "--instance", str(rectangle_path), "--target-cost", "14"]
    )

    assert result.exit_code == 1
    assert "Could not build a feasible initial solution." in result.output
    assert list(tmp_path.iterdir()) == [rectangle_path]
