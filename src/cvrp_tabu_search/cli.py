from pathlib import Path
from typing import Annotated

import typer

from .experiments.reports import DEFAULT_OUTPUT_DIR
from .experiments.runner import run_baseline_experiment, run_tabu_search_experiment
from .instances import Instance, load_default_instance, load_instance
from .search.shared import SearchParameters

app = typer.Typer(help="Run the CVRP tabu search experiments.", no_args_is_help=True)

InstancePath = Annotated[
    Path | None,
    typer.Option(
        "--instance",
        exists=True,
        dir_okay=False,
        readable=True,
        help="Path to a CVRP instance file. Defaults to the bundled A-n32-k5 instance.",
    ),
]
TargetCost = Annotated[
    int | None,
    typer.Option(
        min=0, help="Known optimal cost for --instance, used as the reporting target."
    ),
]
IterationLimit = Annotated[
    int, typer.Option(min=0, help="Maximum number of iterations per run.")
]
StagnationLimit = Annotated[
    int,
    typer.Option(min=1, help="Stop after this many iterations without improvement."),
]
OutputDirectory = Annotated[
    Path,
    typer.Option(
        file_okay=False,
        writable=True,
        show_default=True,
        help="Parent directory for timestamped CSV and PNG report folders.",
    ),
]


@app.command("baseline")
def baseline_command(
    instance: InstancePath = None,
    target_cost: TargetCost = None,
    max_iterations: IterationLimit = 10,
    stagnation_limit: StagnationLimit = 10,
    output_dir: OutputDirectory = DEFAULT_OUTPUT_DIR,
) -> None:
    """Run the baseline with reversals within each route."""
    problem = _resolve_instance(instance, target_cost)
    parameters = SearchParameters(max_iterations, stagnation_limit)
    if not run_baseline_experiment(problem, parameters, output_dir=output_dir):
        raise typer.Exit(code=1)


@app.command("tabu-search")
def tabu_search_command(
    instance: InstancePath = None,
    target_cost: TargetCost = None,
    max_iterations: IterationLimit = 60,
    stagnation_limit: StagnationLimit = 30,
    output_dir: OutputDirectory = DEFAULT_OUTPUT_DIR,
) -> None:
    """Run the search with inter-route moves, intensification and diversification."""
    problem = _resolve_instance(instance, target_cost)
    parameters = SearchParameters(max_iterations, stagnation_limit)
    if not run_tabu_search_experiment(problem, parameters, output_dir=output_dir):
        raise typer.Exit(code=1)


def _resolve_instance(path: Path | None, target_cost: int | None) -> Instance:
    if path is None:
        if target_cost is not None:
            raise typer.BadParameter("Requires --instance.", param_hint="--target-cost")
        return load_default_instance()

    if target_cost is None:
        raise typer.BadParameter(
            "Required with --instance.", param_hint="--target-cost"
        )

    try:
        return load_instance(path, optimal_cost=target_cost)
    except (OSError, ValueError, IndexError) as error:
        raise typer.BadParameter(
            f"Could not read {path}: {error}", param_hint="--instance"
        ) from error
