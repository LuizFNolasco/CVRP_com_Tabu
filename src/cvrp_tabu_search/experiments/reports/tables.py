import csv
from collections import defaultdict
from collections.abc import Iterable, Sequence
from itertools import groupby
from pathlib import Path
from statistics import fmean

from ..summary import (
    InstanceSummary,
    ResultsError,
    gap,
    instance_sort_key,
    load_summaries,
)
from .csv_writer import write_csv

FINAL_HEADER = (
    "instance",
    "group",
    "customers",
    "optimal_cost",
    "runs",
    "best_cost",
    "mean_cost",
    "best_run_time_s",
    "mean_time_s",
    "gap_best_pct",
    "gap_mean_pct",
)
ABLATION_VARIANTS = (
    ("baseline", ""),
    ("sem_fases", "tabu_nophases_"),
    ("completa", "tabu_"),
)


def decimal(value: float, places: int) -> str:
    return f"{value:.{places}f}".replace(".", ",")


def write_final_table(summaries: Sequence[InstanceSummary], output_dir: Path) -> None:
    rows = (
        [
            s.instance,
            s.group,
            s.customers,
            s.optimal_cost,
            len(s.costs),
            s.best_cost,
            round(s.mean_cost, 2),
            round(s.best_run_time_s, 3),
            round(s.mean_time_s, 3),
            round(s.gap_best, 3),
            round(s.gap_mean, 3),
        ]
        for s in summaries
    )
    write_csv(output_dir / "final_table.csv", FINAL_HEADER, rows)
    (output_dir / "final_table.tex").write_text(
        final_table_latex(summaries), encoding="utf-8"
    )
    # Two halves (first set | remaining sets) to typeset side by side.
    first = [s for s in summaries if s.group == summaries[0].group]
    rest = [s for s in summaries if s.group != summaries[0].group]
    (output_dir / "final_table_part1.tex").write_text(
        final_table_latex(first, overall=()), encoding="utf-8"
    )
    (output_dir / "final_table_part2.tex").write_text(
        final_table_latex(rest, overall=summaries), encoding="utf-8"
    )


def final_table_latex(
    summaries: Sequence[InstanceSummary],
    overall: Sequence[InstanceSummary] | None = None,
) -> str:
    """Build the assignment table; ``overall`` feeds the last line (empty: omit)."""
    if overall is None:
        overall = summaries
    lines = [
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        (
            r"Instância & $f_{\text{ótimo}}$ & \multicolumn{2}{c}{$f_{MH}$}"
            r" & \multicolumn{2}{c}{Tempo (s)} & $\text{gap}_{\text{Min}}$"
            r" & $\text{gap}_{\text{Média}}$ \\"
        ),
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
        r" & & Min$_5$ & Média$_5$ & Min$_5$ & Média$_5$ & (\%) & (\%) \\",
        r"\midrule",
    ]
    for index, (group, members) in enumerate(
        (group, list(members))
        for group, members in groupby(summaries, key=lambda s: s.group)
    ):
        if index:
            lines.append(r"\midrule")
        for s in members:
            lines.append(
                f"{s.instance} & {s.optimal_cost} & {s.best_cost}"
                f" & {decimal(s.mean_cost, 1)} & {decimal(s.best_run_time_s, 1)}"
                f" & {decimal(s.mean_time_s, 1)} & {decimal(s.gap_best, 2)}"
                f" & {decimal(s.gap_mean, 2)} \\\\"
            )
        lines.append(_average_line(f"Média {group}", members))

    if overall:
        lines += [r"\midrule", _average_line("Média geral", overall)]
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(lines)


def _average_line(label: str, members: Sequence[InstanceSummary]) -> str:
    mean_time = fmean(s.mean_time_s for s in members)
    gap_best = fmean(s.gap_best for s in members)
    gap_mean = fmean(s.gap_mean for s in members)
    return (
        rf"\textit{{{label}}} & & & & & {decimal(mean_time, 1)}"
        rf" & {decimal(gap_best, 2)} & {decimal(gap_mean, 2)} \\"
    )


def write_calibration_tables(csv_path: Path, output_dir: Path) -> None:
    with csv_path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ResultsError(f"{csv_path} has no calibration rows.")

    optimum = {row["instance"]: int(row["optimal_cost"]) for row in rows}
    instances = sorted(optimum, key=instance_sort_key)
    cost = {
        (
            row["instance"],
            int(row["tabu_tenure_min"]),
            int(row["tabu_tenure_max"]),
        ): int(row["best_cost"])
        for row in rows
    }
    combos = sorted({(minimum, maximum) for _, minimum, maximum in cost})
    best = {
        instance: min(cost[instance, *combo] for combo in combos)
        for instance in instances
    }
    varying = [
        instance
        for instance in instances
        if len({cost[instance, *combo] for combo in combos}) > 1
    ]

    def cell(instance: str, combo: tuple[int, int]) -> str:
        value = cost[instance, *combo]
        return (
            rf"\textbf{{{value}}}"
            if instance in varying and value == best[instance]
            else str(value)
        )

    long_lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Instância & $t_{\min}$ & $t_{\max}$ & $f_{MH}$ & gap (\%) \\",
        r"\midrule",
    ]
    for index, instance in enumerate(instances):
        if index:
            long_lines.append(r"\midrule")
        for combo in combos:
            value = cost[instance, *combo]
            long_lines.append(
                f"{instance} & {combo[0]} & {combo[1]} & {cell(instance, combo)}"
                f" & {decimal(gap(value, optimum[instance]), 2)} \\\\"
            )
    long_lines += [r"\bottomrule", r"\end{tabular}", ""]
    (output_dir / "calibration_long.tex").write_text(
        "\n".join(long_lines), encoding="utf-8"
    )

    header = " & ".join(instances)
    pivot_lines = [
        r"\begin{tabular}{rr" + "r" * len(instances) + "}",
        r"\toprule",
        rf"$t_{{\min}}$ & $t_{{\max}}$ & {header} \\",
        r"\midrule",
    ]
    for combo in combos:
        cells = []
        for instance in instances:
            value = cost[instance, *combo]
            text = f"{value} ({decimal(gap(value, optimum[instance]), 2)})"
            if instance in varying and value == best[instance]:
                text = rf"\textbf{{{text}}}"
            cells.append(text)
        pivot_lines.append(f"{combo[0]} & {combo[1]} & {' & '.join(cells)} \\\\")
    pivot_lines += [r"\bottomrule", r"\end{tabular}", ""]
    (output_dir / "calibration_pivot.tex").write_text(
        "\n".join(pivot_lines), encoding="utf-8"
    )

    effects: dict[tuple[str, int], list[float]] = defaultdict(list)
    for instance in varying:
        for minimum, maximum in combos:
            value = gap(cost[instance, minimum, maximum], optimum[instance])
            effects["tabu_tenure_min", minimum].append(value)
            effects["tabu_tenure_max", maximum].append(value)
    write_csv(
        output_dir / "calibration_effects.csv",
        ("parameter", "value", "mean_gap_pct", "instances"),
        (
            [parameter, value, round(fmean(gaps), 3), " ".join(varying)]
            for (parameter, value), gaps in sorted(effects.items())
        ),
    )


def write_ablation_table(results_dirs: Iterable[Path], output_dir: Path) -> list[str]:
    directories = list(results_dirs)
    by_variant = {
        label: {s.instance: s for s in load_summaries(directories, prefix)}
        for label, prefix in ABLATION_VARIANTS
    }
    common = set.intersection(*(set(variant) for variant in by_variant.values()))
    names = sorted(common, key=instance_sort_key)
    if not names:
        raise ResultsError("No instance has results for all three variants.")

    header = ["instance", "optimal_cost"]
    for label, _ in ABLATION_VARIANTS:
        header += [
            f"{label}_mean_cost",
            f"{label}_gap_mean_pct",
            f"{label}_mean_time_s",
        ]
    csv_rows = []
    lines = [
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        (
            r" & & \multicolumn{2}{c}{Só inversões} & \multicolumn{2}{c}{Sem int./div.}"
            r" & \multicolumn{2}{c}{Completa} \\"
        ),
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}",
        (
            r"Instância & $f_{\text{ótimo}}$ & gap (\%) & t (s) & gap (\%) & t (s)"
            r" & gap (\%) & t (s) \\"
        ),
        r"\midrule",
    ]
    for name in names:
        summaries = [by_variant[label][name] for label, _ in ABLATION_VARIANTS]
        row: list[str | int | float] = [name, summaries[0].optimal_cost]
        cells = []
        for s in summaries:
            row += [
                round(s.mean_cost, 2),
                round(s.gap_mean, 3),
                round(s.mean_time_s, 3),
            ]
            cells += [decimal(s.gap_mean, 2), decimal(s.mean_time_s, 1)]
        csv_rows.append(row)
        lines.append(f"{name} & {summaries[0].optimal_cost} & {' & '.join(cells)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    write_csv(output_dir / "ablation_table.csv", header, csv_rows)
    (output_dir / "ablation_table.tex").write_text("\n".join(lines), encoding="utf-8")
    return names
