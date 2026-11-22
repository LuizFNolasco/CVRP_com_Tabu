from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from ...search.shared import SearchHistory
from .directory import DEFAULT_OUTPUT_DIR, create_report_directory


class TargetMeasurement(NamedTuple):
    total_time_s: float
    time_to_target_s: float | None


def generate_plots(
    instance_name: str,
    optimal_cost: int,
    best_history: SearchHistory,
    measurements: Sequence[TargetMeasurement],
    *,
    prefix: str = "",
    output_dir: Path | None = None,
) -> None:
    if output_dir is None:
        output_dir = create_report_directory(DEFAULT_OUTPUT_DIR)

    plt.style.use("default")
    name = f"{prefix}{instance_name}"
    convergence_path = output_dir / f"convergence_{name}.png"
    time_to_target_path = output_dir / f"ttt_{name}.png"
    _plot_convergence(instance_name, optimal_cost, best_history, convergence_path)
    _plot_time_to_target(instance_name, optimal_cost, measurements, time_to_target_path)


def _plot_convergence(
    instance_name: str, optimal_cost: int, history: SearchHistory, path: Path
) -> None:
    iterations: list[int] = []
    costs: list[int] = []

    # The first history entry is the initial solution.
    for iteration, _, _, current_cost in history[1:]:
        iterations.append(iteration)
        costs.append(current_cost)

    figure, axes = plt.subplots(figsize=(9, 5))
    axes.plot(iterations, costs, color="tab:blue", label="Iteration cost")
    axes.axhline(
        optimal_cost,
        color="tab:orange",
        linestyle="--",
        label=f"Optimum: {optimal_cost}",
    )
    axes.set_title(f"Tabu search convergence — {instance_name}")
    axes.set_xlabel("Iteration")
    axes.set_ylabel("Total cost")
    axes.legend()

    _save_figure(figure, path)


def _plot_time_to_target(
    instance_name: str,
    optimal_cost: int,
    measurements: Sequence[TargetMeasurement],
    path: Path,
) -> None:
    times, percentages, hits = _time_to_target_curve(measurements)
    total = len(measurements)

    figure, axes = plt.subplots(figsize=(9, 5))
    axes.step(
        times,
        percentages,
        where="post",
        color="tab:blue",
        label=f"Target {optimal_cost}: {hits} of {total}",
    )
    axes.set_title(f"Time to target — {instance_name}")
    axes.set_xlabel("Elapsed time (s)")
    axes.set_ylabel("Runs that reached the target (%)")
    axes.set_ylim(-4, 104)
    axes.set_yticks([0, 25, 50, 75, 100])
    axes.legend()

    _save_figure(figure, path)


def _save_figure(figure: Figure, path: Path) -> None:
    try:
        figure.tight_layout()
        figure.savefig(path, dpi=200)
    finally:
        plt.close(figure)


def _time_to_target_curve(
    measurements: Sequence[TargetMeasurement],
) -> tuple[list[float], list[float], int]:
    target_times: list[float] = []
    longest_run = 0.0
    for total_time, target_time in measurements:
        longest_run = max(longest_run, total_time)
        if target_time is not None:
            target_times.append(target_time)

    target_times.sort()
    total = len(measurements)
    times: list[float] = [0]
    percentages: list[float] = [0]
    for hits, time in enumerate(target_times, start=1):
        times.append(time)
        percentages.append(100 * hits / total)

    # Extend the curve to the end of the observation window.
    times.append(longest_run)
    percentages.append(percentages[-1])
    return times, percentages, len(target_times)
