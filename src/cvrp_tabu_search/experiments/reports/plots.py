from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from ...search.shared import SearchHistory, SearchHistoryEntry
from ..summary import InstanceSummary, gap
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
    # The first history entry is the initial solution.
    plot_convergence(
        optimal_cost, best_history[1:], output_dir / f"convergence_{name}.png"
    )
    times = [measurement.time_to_target_s for measurement in measurements]
    plot_time_to_target(optimal_cost, times, output_dir / f"ttt_{name}.png")


def plot_convergence(
    optimal_cost: int, entries: Sequence[SearchHistoryEntry], path: Path
) -> None:
    iterations = [entry.iteration for entry in entries]
    figure, axes = plt.subplots(figsize=(7, 4))
    axes.plot(
        iterations,
        [entry.current_cost for entry in entries],
        color="0.7",
        linewidth=0.8,
        label="Solução corrente",
    )
    axes.step(
        iterations,
        [entry.best_cost for entry in entries],
        where="post",
        color="tab:blue",
        linewidth=1.6,
        label="Melhor solução",
    )
    axes.axhline(
        optimal_cost,
        color="tab:orange",
        linestyle="--",
        label=f"Ótimo conhecido ({optimal_cost})",
    )
    if iterations:
        axes.set_xlim(left=1)
    axes.set_xlabel("Iteração")
    axes.set_ylabel("Custo total")
    axes.grid(alpha=0.3)
    axes.legend()
    save_figure(figure, path)


def time_to_target_points(
    times: Sequence[float | None],
) -> tuple[list[float], list[float]]:
    """Points (t_i, (i - 1/2) / n) of Aiex et al. (2007); misses count in n."""
    total = len(times)
    hits = sorted(time for time in times if time is not None)
    probabilities = [(index - 0.5) / total for index in range(1, len(hits) + 1)]
    return hits, probabilities


def plot_time_to_target(
    target_cost: int, times: Sequence[float | None], path: Path
) -> None:
    hits, probabilities = time_to_target_points(times)
    figure, axes = plt.subplots(figsize=(7, 4))
    axes.plot(
        hits,
        probabilities,
        marker="o",
        markersize=3,
        color="tab:blue",
        label=f"Alvo {target_cost}: {len(hits)} de {len(times)} execuções",
    )
    axes.set_xlabel("Tempo até o alvo (s)")
    axes.set_ylabel("Probabilidade acumulada")
    axes.set_xlim(left=0)
    axes.set_ylim(0, 1)
    axes.grid(alpha=0.3)
    axes.legend(loc="lower right")
    save_figure(figure, path)


def save_figure(figure: Figure, path: Path) -> None:
    try:
        figure.tight_layout()
        figure.savefig(path, dpi=200)
    finally:
        plt.close(figure)


_GROUP_STYLE = {
    "A": ("o", "tab:blue"),
    "B": ("s", "tab:orange"),
    "F": ("^", "tab:green"),
}


def plot_gap_by_size(summaries: Sequence[InstanceSummary], path: Path) -> None:
    figure, axes = plt.subplots(figsize=(7, 4))
    for group in sorted({summary.group for summary in summaries}):
        members = [summary for summary in summaries if summary.group == group]
        marker, color = _GROUP_STYLE.get(group, ("o", "tab:gray"))
        sizes = [summary.customers for summary in members]
        worst = [max(gap(cost, s.optimal_cost) for cost in s.costs) for s in members]
        axes.vlines(sizes, [s.gap_best for s in members], worst, color=color, alpha=0.4)
        axes.scatter(
            sizes,
            [s.gap_mean for s in members],
            marker=marker,
            color=color,
            label=f"Conjunto {group}",
            zorder=3,
        )
    axes.axhline(0, color="0.5", linewidth=0.8)
    axes.set_xlabel("Número de clientes")
    axes.set_ylabel("Gap em relação ao ótimo (%)")
    axes.grid(alpha=0.3)
    axes.legend()
    save_figure(figure, path)
