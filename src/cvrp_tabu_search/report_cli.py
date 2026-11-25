from pathlib import Path
from typing import Annotated

import typer

from .experiments.reports.csv_writer import read_history_csv
from .experiments.reports.plots import plot_convergence, plot_gap_by_size
from .experiments.reports.tables import (
    write_ablation_table,
    write_calibration_tables,
    write_final_table,
)
from .experiments.summary import ResultsError, instance_sort_key, load_summaries

report_app = typer.Typer(
    help="Build the article tables and figures from saved CSV reports.",
    no_args_is_help=True,
)

ResultsDirectories = Annotated[
    list[Path],
    typer.Argument(
        exists=True,
        file_okay=False,
        help="Folders searched recursively for runs_*.csv files.",
    ),
]
ArticleDirectory = Annotated[
    Path,
    typer.Option(
        "--output-dir",
        file_okay=False,
        help="Folder for the generated .csv, .tex and .png files.",
    ),
]
_READ_ERRORS = (ResultsError, OSError, KeyError, ValueError)


@report_app.command("final")
def final_command(
    results_dirs: ResultsDirectories,
    output_dir: ArticleDirectory,
    benchmarks_dir: Annotated[
        Path,
        typer.Option(
            exists=True,
            file_okay=False,
            help="Folder whose .vrp files (plus A-n32-k5) should all have results.",
        ),
    ] = Path("benchmarks/general"),
    expected_runs: Annotated[
        int, typer.Option(min=1, help="Runs expected per instance.")
    ] = 5,
) -> None:
    """Summarize runs_tabu_*.csv into the assignment table and the gap figure."""
    try:
        summaries = load_summaries(results_dirs, prefix="tabu_")
    except _READ_ERRORS as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=1) from error
    if not summaries:
        typer.echo("No runs_tabu_*.csv files found.", err=True)
        raise typer.Exit(code=1)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_final_table(summaries, output_dir)
    plot_gap_by_size(summaries, output_dir / "gap_by_size.png")

    expected = {path.stem for path in benchmarks_dir.glob("*.vrp")} | {"A-n32-k5"}
    missing = sorted(expected - {s.instance for s in summaries}, key=instance_sort_key)
    if missing:
        typer.echo(f"Missing instances ({len(missing)}): {', '.join(missing)}")
    for summary in summaries:
        if len(summary.costs) != expected_runs:
            typer.echo(f"{summary.instance} has {len(summary.costs)} runs.")
    typer.echo(f"{len(summaries)} instances summarized in {output_dir}")


@report_app.command("convergence")
def convergence_command(
    history: Annotated[
        Path, typer.Argument(exists=True, dir_okay=False, help="A history_*.csv file.")
    ],
    optimal_cost: Annotated[int, typer.Option(min=0, help="Known optimal cost.")],
    output: Annotated[Path, typer.Option(dir_okay=False, help="PNG file to write.")],
) -> None:
    """Redraw the best-so-far convergence of a saved run history."""
    try:
        entries = read_history_csv(history)
    except _READ_ERRORS as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=1) from error
    output.parent.mkdir(parents=True, exist_ok=True)
    plot_convergence(optimal_cost, entries, output)
    typer.echo(f"Convergence plot: {output}")


@report_app.command("calibration")
def calibration_command(
    calibration_csv: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, help="A calibration.csv file."),
    ],
    output_dir: ArticleDirectory,
) -> None:
    """Build the calibration tables and the mean gap of each parameter level."""
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        write_calibration_tables(calibration_csv, output_dir)
    except _READ_ERRORS as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=1) from error
    typer.echo(f"Calibration tables: {output_dir}")


@report_app.command("ablation")
def ablation_command(
    results_dirs: ResultsDirectories, output_dir: ArticleDirectory
) -> None:
    """Compare baseline, tabu search without phases and the full tabu search."""
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        names = write_ablation_table(results_dirs, output_dir)
    except _READ_ERRORS as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=1) from error
    typer.echo(f"Ablation table with {len(names)} instances: {output_dir}")
