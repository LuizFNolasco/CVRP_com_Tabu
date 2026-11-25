import csv
from collections.abc import Callable
from pathlib import Path

from typer.testing import CliRunner

from cvrp_tabu_search.cli import app
from cvrp_tabu_search.experiments.reports.csv_writer import write_history_csv
from cvrp_tabu_search.experiments.summary import load_summaries
from cvrp_tabu_search.search.shared import SearchHistoryEntry

type RunsWriter = Callable[[Path, str, int, list[int], list[float]], None]

CALIBRATION_HEADER = [
    "instance",
    "tabu_tenure_min",
    "tabu_tenure_max",
    "seed",
    "optimal_cost",
    "time_limit_s",
    "total_time_s",
    "jobs",
    "best_cost",
]


def test_final_report_writes_the_assignment_table(
    tmp_path: Path, write_runs: RunsWriter
) -> None:
    results = tmp_path / "results"
    write_runs(
        results / "x" / "runs_tabu_A-n32-k5.csv",
        "A-n32-k5",
        784,
        [800, 784, 790, 784, 792],
        [10.0, 12.0, 11.0, 9.0, 13.0],
    )
    write_runs(
        results / "y" / "runs_tabu_B-n31-k5.csv",
        "B-n31-k5",
        672,
        [672, 672, 680, 676, 672],
        [5.0, 5.0, 5.0, 5.0, 5.0],
    )
    benchmarks = tmp_path / "benchmarks"
    benchmarks.mkdir()
    for name in ("B-n31-k5", "F-n45-k4"):
        (benchmarks / f"{name}.vrp").write_text("", encoding="utf-8")
    output = tmp_path / "article"

    result = CliRunner().invoke(
        app,
        [
            "report",
            "final",
            str(results),
            "--output-dir",
            str(output),
            "--benchmarks-dir",
            str(benchmarks),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Missing instances (1): F-n45-k4" in result.output
    table = (output / "final_table.tex").read_text(encoding="utf-8")
    assert r"A-n32-k5 & 784 & 784 & 790,0 & 12,0 & 11,0 & 0,00 & 0,77 \\" in table
    assert r"\textit{Média geral}" in table
    first = (output / "final_table_part1.tex").read_text(encoding="utf-8")
    second = (output / "final_table_part2.tex").read_text(encoding="utf-8")
    assert "A-n32-k5" in first and "Média geral" not in first
    assert "B-n31-k5" in second and r"\textit{Média geral}" in second
    with (output / "final_table.csv").open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert [row["instance"] for row in rows] == ["A-n32-k5", "B-n31-k5"]
    assert (output / "gap_by_size.png").is_file()


def test_convergence_report_redraws_a_saved_history(tmp_path: Path) -> None:
    history_path = tmp_path / "history_tabu_A-n32-k5.csv"
    write_history_csv(
        history_path,
        [SearchHistoryEntry(0, 30, 0.0, 30), SearchHistoryEntry(1, 25, 0.5, 25)],
    )
    output = tmp_path / "convergence.png"

    result = CliRunner().invoke(
        app,
        [
            "report",
            "convergence",
            str(history_path),
            "--optimal-cost",
            "20",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_calibration_report_builds_long_and_pivot_tables(tmp_path: Path) -> None:
    calibration = tmp_path / "calibration.csv"
    with calibration.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(CALIBRATION_HEADER)
        for minimum, maximum, small, large in [
            (1, 7, 784, 1833),
            (1, 15, 784, 1797),
            (3, 7, 784, 1846),
            (3, 15, 784, 1855),
        ]:
            writer.writerow(["A-n32-k5", minimum, maximum, 1, 784, 180, 180, 8, small])
            writer.writerow(
                ["A-n80-k10", minimum, maximum, 1, 1763, 180, 180, 8, large]
            )
    output = tmp_path / "article"

    result = CliRunner().invoke(
        app, ["report", "calibration", str(calibration), "--output-dir", str(output)]
    )

    assert result.exit_code == 0, result.output
    long_table = (output / "calibration_long.tex").read_text(encoding="utf-8")
    assert r"A-n80-k10 & 1 & 15 & \textbf{1797} & 1,93 \\" in long_table
    assert r"A-n32-k5 & 1 & 7 & 784 & 0,00 \\" in long_table
    pivot = (output / "calibration_pivot.tex").read_text(encoding="utf-8")
    assert r"1 & 15 & 784 (0,00) & \textbf{1797 (1,93)} \\" in pivot
    effects_path = output / "calibration_effects.csv"
    with effects_path.open(encoding="utf-8", newline="") as file:
        effects = {
            (row["parameter"], row["value"]): row for row in csv.DictReader(file)
        }
    assert effects[("tabu_tenure_min", "1")]["instances"] == "A-n80-k10"
    assert float(effects[("tabu_tenure_min", "1")]["mean_gap_pct"]) == round(
        100 * ((1833 + 1797) / 2 - 1763) / 1763, 3
    )


def test_ablation_report_compares_the_three_variants(
    tmp_path: Path, write_runs: RunsWriter
) -> None:
    results = tmp_path / "ablation"
    write_runs(results / "1" / "runs_A-n32-k5.csv", "A-n32-k5", 784, [1470], [0.1])
    write_runs(
        results / "2" / "runs_tabu_nophases_A-n32-k5.csv", "A-n32-k5", 784, [830], [1]
    )
    write_runs(results / "3" / "runs_tabu_A-n32-k5.csv", "A-n32-k5", 784, [800], [1.2])
    output = tmp_path / "article"

    result = CliRunner().invoke(
        app, ["report", "ablation", str(results), "--output-dir", str(output)]
    )

    assert result.exit_code == 0, result.output
    table = (output / "ablation_table.tex").read_text(encoding="utf-8")
    assert r"A-n32-k5 & 784 & 87,50 & 0,1 & 5,87 & 1,0 & 2,04 & 1,2 \\" in table
    assert load_summaries([results], prefix="")[0].instance == "A-n32-k5"
