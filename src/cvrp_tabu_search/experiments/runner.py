from pathlib import Path

from ..instances import Instance
from ..search.baseline import tabu_search as baseline_search
from ..search.shared import SearchParameters
from .execution import Run, SearchAlgorithm, run_once, tabu_search_for
from .reports import DEFAULT_OUTPUT_DIR, generate_reports

REPETITIONS = 5


def run_baseline_experiment(
    instance: Instance,
    parameters: SearchParameters,
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> bool:
    return run_experiment(instance, baseline_search, parameters, output_dir=output_dir)


def run_tabu_search_experiment(
    instance: Instance,
    parameters: SearchParameters,
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    use_phases: bool = True,
) -> bool:
    search = tabu_search_for(instance, use_phases=use_phases)
    prefix = "tabu_" if use_phases else "tabu_nophases_"
    return run_experiment(
        instance, search, parameters, report_prefix=prefix, output_dir=output_dir
    )


def run_experiment(
    instance: Instance,
    search: SearchAlgorithm,
    parameters: SearchParameters,
    *,
    report_prefix: str = "",
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> bool:
    runs = _run_repetitions(instance, search, parameters)
    if runs is None:
        return False

    generate_reports(
        instance, parameters, runs, prefix=report_prefix, output_dir=output_dir
    )
    return True


def _run_repetitions(
    instance: Instance, search: SearchAlgorithm, parameters: SearchParameters
) -> list[Run] | None:
    runs = []
    for repetition in range(1, REPETITIONS + 1):
        run = run_once(instance, search, parameters, seed=repetition)
        if run is None:
            print("Could not build a feasible initial solution.")
            return None

        runs.append(run)
        print(f"Run {repetition}: cost {run.cost} | time {run.total_time_s:.3f} s")

    return runs
