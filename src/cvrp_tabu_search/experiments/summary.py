import csv
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean

_INSTANCE_NAME = re.compile(r"^(?P<group>[A-Z]+)-n(?P<nodes>\d+)-k(?P<vehicles>\d+)$")


class ResultsError(ValueError):
    pass


def gap(cost: float, optimal_cost: int) -> float:
    return 100 * (cost - optimal_cost) / optimal_cost


def instance_sort_key(name: str) -> tuple[str, int, int]:
    match = _INSTANCE_NAME.match(name)
    if match is None:
        raise ResultsError(f"Unexpected instance name: {name}")
    return match["group"], int(match["nodes"]), int(match["vehicles"])


@dataclass(frozen=True)
class InstanceSummary:
    instance: str
    optimal_cost: int
    costs: tuple[int, ...]
    times_s: tuple[float, ...]

    @property
    def group(self) -> str:
        return instance_sort_key(self.instance)[0]

    @property
    def customers(self) -> int:
        # The node count in the name includes the depot.
        return instance_sort_key(self.instance)[1] - 1

    @property
    def best_cost(self) -> int:
        return min(self.costs)

    @property
    def mean_cost(self) -> float:
        return fmean(self.costs)

    @property
    def best_run_time_s(self) -> float:
        return self.times_s[self.costs.index(self.best_cost)]

    @property
    def mean_time_s(self) -> float:
        return fmean(self.times_s)

    @property
    def gap_best(self) -> float:
        return gap(self.best_cost, self.optimal_cost)

    @property
    def gap_mean(self) -> float:
        return gap(self.mean_cost, self.optimal_cost)


def load_summaries(results_dirs: Iterable[Path], prefix: str) -> list[InstanceSummary]:
    """Collect runs_<prefix><instance>.csv files below the given folders."""
    found: dict[str, tuple[Path, list[dict[str, str]]]] = {}
    for results_dir in results_dirs:
        for path in sorted(results_dir.rglob(f"runs_{prefix}*.csv")):
            rows = _read_rows(path)
            if not rows:
                continue

            instance = rows[0]["instance"]
            # A shorter prefix also matches other variants, e.g. runs_tabu_*.csv.
            if path.stem != f"runs_{prefix}{instance}":
                continue

            if instance in found:
                raise ResultsError(
                    f"{instance} appears in {found[instance][0]} and {path}; "
                    "keep one run set per instance."
                )
            found[instance] = (path, rows)

    summaries = [
        InstanceSummary(
            instance=instance,
            optimal_cost=int(rows[0]["optimal_cost"]),
            costs=tuple(int(row["best_cost"]) for row in rows),
            times_s=tuple(float(row["total_time_s"]) for row in rows),
        )
        for instance, (_, rows) in found.items()
    ]
    return sorted(summaries, key=lambda summary: instance_sort_key(summary.instance))


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))
