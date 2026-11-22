import random
from itertools import count
from pathlib import Path
from unittest.mock import Mock

import pytest

from cvrp_tabu_search.experiments import runner
from cvrp_tabu_search.experiments.runner import Run
from cvrp_tabu_search.instances import Instance
from cvrp_tabu_search.search import baseline, shared
from cvrp_tabu_search.search.shared import (
    SearchHistoryEntry,
    SearchParameters,
    SearchResult,
    TabuTenure,
)


@pytest.fixture
def rectangle_instance() -> Instance:
    return Instance(
        name="rectangle",
        optimal_cost=14,
        vehicle_count=1,
        dimension=4,
        capacity=3,
        coordinates=[[0, 0], [4, 0], [0, 3], [4, 3]],
        demands=[0, 1, 1, 1],
        depot=1,
    )


def test_experiment_rebuilds_the_initial_fleet_before_every_run(
    rectangle_instance: Instance, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = count(start=100)
    monkeypatch.setattr(runner, "perf_counter", lambda: next(clock))
    monkeypatch.setattr(shared, "perf_counter", lambda: next(clock))
    publish = Mock()
    monkeypatch.setattr(runner, "generate_reports", publish)
    parameters = SearchParameters(
        max_iterations=1, stagnation_limit=5, tabu_tenure=TabuTenure(2, 2)
    )

    completed = runner.run_experiment(
        rectangle_instance, baseline.tabu_search, parameters, report_prefix="trial_"
    )

    assert completed
    publish.assert_called_once()
    instance, reported_parameters, runs = publish.call_args.args
    assert instance is rectangle_instance
    assert reported_parameters is parameters
    assert publish.call_args.kwargs == {
        "prefix": "trial_",
        "output_dir": Path("results"),
    }

    history = [SearchHistoryEntry(0, 18, 1.0, 18), SearchHistoryEntry(1, 14, 2.0, 14)]
    expected_runs = [
        Run(
            seed,
            cost=14,
            history=history,
            total_time_s=3,
            time_to_best_s=2,
            time_to_target_s=2,
        )
        for seed in range(1, 6)
    ]
    assert runs == expected_runs


def test_experiment_seeds_each_search_reproducibly(
    rectangle_instance: Instance, monkeypatch: pytest.MonkeyPatch
) -> None:
    generator = random.Random()
    monkeypatch.setattr(runner, "random", generator)
    monkeypatch.setattr(runner, "generate_reports", Mock())
    samples: list[float] = []

    def observe_sample(*_arguments: object) -> SearchResult:
        samples.append(generator.random())
        return SearchResult(
            best_routes=[[2, 3, 4]],
            best_cost=18,
            history=[SearchHistoryEntry(0, 18, 0.0, 18)],
        )

    parameters = SearchParameters(max_iterations=0, stagnation_limit=5)
    runner.run_experiment(rectangle_instance, observe_sample, parameters)

    assert samples == [random.Random(seed).random() for seed in range(1, 6)]


def test_experiment_skips_search_and_reports_when_the_initial_fleet_is_infeasible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    instance = Instance("infeasible", 4, 1, 2, 1, [[0, 0], [2, 0]], [0, 2], 1)
    algorithm = Mock()
    parameters = SearchParameters(max_iterations=10, stagnation_limit=5)
    monkeypatch.chdir(tmp_path)

    completed = runner.run_experiment(instance, algorithm, parameters)

    assert not completed
    algorithm.assert_not_called()
    assert list(tmp_path.iterdir()) == []
    assert "Could not build a feasible initial solution." in capsys.readouterr().out


@pytest.mark.parametrize(
    ("target", "expected_target_time"),
    [(18, 0.0), (14, 0.25), (13, 0.5), (11, None)],
    ids=["initial-hit", "equal-to-target", "below-target", "miss"],
)
def test_experiment_measures_the_first_relevant_iteration(
    target: int, expected_target_time: float | None
) -> None:
    history = [
        SearchHistoryEntry(0, 18, 0.0, 18),
        SearchHistoryEntry(1, 14, 0.25, 14),
        SearchHistoryEntry(2, 12, 0.5, 12),
        SearchHistoryEntry(3, 12, 0.75, 16),
    ]

    time_to_best, time_to_target = runner._measure_times(
        history, final_cost=12, target_cost=target
    )

    assert time_to_best == 0.5
    assert time_to_target == expected_target_time
