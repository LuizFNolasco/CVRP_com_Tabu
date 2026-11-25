from functools import partial
from unittest.mock import DEFAULT, Mock

import pytest

from cvrp_tabu_search.experiments.execution import SearchAlgorithm
from cvrp_tabu_search.routes import Vehicle, build_initial_solution
from cvrp_tabu_search.search import baseline, shared, tabu

RECTANGLE = [(0, 0), (4, 0), (0, 3), (4, 3)]
RECTANGLE_DEMANDS = [0, 1, 1, 1]
PLATEAU_COORDINATES = [(0, 0)] * 5
PLATEAU_DEMANDS = [0, 1, 2, 1, 2]


@pytest.fixture
def rectangle_fleet() -> list[Vehicle]:
    fleet = build_initial_solution(1, 4, 3, RECTANGLE_DEMANDS, depot=1)
    assert fleet is not None
    return fleet


@pytest.fixture
def plateau_fleet() -> list[Vehicle]:
    fleet = build_initial_solution(1, 5, 6, PLATEAU_DEMANDS, depot=1)
    assert fleet is not None
    return fleet


@pytest.mark.parametrize(
    ("iteration", "cost", "registered", "blocked"),
    [
        (3, 80, False, False),
        (3, 80, True, True),
        (4, 81, True, True),
        (5, 81, True, False),
        (4, 79, True, False),
    ],
    ids=[
        "unregistered",
        "equal-to-best",
        "expiration-inclusive",
        "expired",
        "aspiration",
    ],
)
def test_tabu_blocking_and_aspiration(
    iteration: int, cost: int, registered: bool, blocked: bool
) -> None:
    tabu_list = {"move": 4} if registered else {}

    assert shared.is_tabu(tabu_list, "move", iteration, cost, best_cost=80) is blocked


def test_register_move_shares_one_tenure_across_its_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample_tenure = Mock(return_value=4)
    monkeypatch.setattr(shared.random, "randint", sample_tenure)
    tabu_list = {"existing": 20, "first": 3}

    shared.register_tabu_move(
        tabu_list, ("first", "second"), iteration=7, tenure=(2, 6)
    )

    sample_tenure.assert_called_once_with(2, 6)
    assert tabu_list == {"existing": 20, "first": 11, "second": 11}


def test_progress_tracks_improvement_and_keeps_an_independent_best_route(
    rectangle_fleet: list[Vehicle], monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = iter([100.0, 100.25, 100.5, 100.75])
    monkeypatch.setattr(shared, "perf_counter", lambda: next(clock))
    progress = shared.SearchProgress(
        rectangle_fleet, RECTANGLE, depot=1, start_time=100.0
    )
    route = rectangle_fleet[0].route

    progress.record_iteration(1, rectangle_fleet, current_cost=18)
    assert progress.iterations_without_improvement == 1

    route[:] = [2, 4, 3]
    assert progress.best_routes == [[2, 3, 4]]
    progress.record_iteration(2, rectangle_fleet, current_cost=14)
    assert progress.iterations_without_improvement == 0

    route[:] = [4, 3, 2]
    progress.record_iteration(3, rectangle_fleet, current_cost=18)
    result = progress.result()

    assert result.best_routes == [[2, 4, 3]]
    assert result.best_cost == 14
    assert result.history[-1].elapsed_time_s == 0.75
    assert result.history == [
        (0, 18, 0.0, 18),
        (1, 18, 0.25, 18),
        (2, 14, 0.5, 14),
        (3, 14, 0.75, 18),
    ]


@pytest.mark.parametrize(
    ("best_cost", "expected_route", "expected_cost"),
    [(14, [3, 2, 4], 16), (15, [2, 4, 3], 14)],
    ids=["equal-cost-stays-tabu", "strict-improvement-overrides-tabu"],
)
def test_baseline_selects_the_best_admissible_reversal(
    rectangle_fleet: list[Vehicle],
    best_cost: int,
    expected_route: list[int],
    expected_cost: int,
) -> None:
    # The three reversals cost 16, 18 and 14. The last one is marked tabu.
    move = baseline.find_best_neighbor(
        rectangle_fleet, 18, RECTANGLE, 1, {(3, 4): 3}, iteration=3, best_cost=best_cost
    )

    assert move is not None
    assert (move.route, move.cost) == (expected_route, expected_cost)
    assert rectangle_fleet[0].route == [2, 3, 4]


def test_baseline_applies_an_improving_reversal(rectangle_fleet: list[Vehicle]) -> None:
    best_routes, best_cost, history = baseline.tabu_search(
        rectangle_fleet, RECTANGLE, 1, (2, 2), 1, 10, start_time=0.0
    )

    assert best_routes == [[2, 4, 3]]
    assert best_cost == 14
    assert rectangle_fleet[0].route == [2, 4, 3]
    assert rectangle_fleet[0].load == 3
    assert [(iteration, best, current) for iteration, best, _, current in history] == [
        (0, 18, 18),
        (1, 14, 14),
    ]


@pytest.mark.parametrize(
    "algorithm",
    [baseline.tabu_search, partial(tabu.tabu_search, demands=PLATEAU_DEMANDS)],
    ids=["baseline", "advanced"],
)
@pytest.mark.parametrize(
    ("iteration_limit", "stagnation_limit", "last_iteration"),
    [(2, 10, 2), (10, 2, 2), (0, 10, 0)],
    ids=["iteration-limit", "stagnation-limit", "zero-iterations"],
)
def test_search_stops_at_the_configured_limit(
    plateau_fleet: list[Vehicle],
    algorithm: SearchAlgorithm,
    iteration_limit: int,
    stagnation_limit: int,
    last_iteration: int,
) -> None:
    initial_route = plateau_fleet[0].route.copy()

    best_routes, best_cost, history = algorithm(
        plateau_fleet,
        PLATEAU_COORDINATES,
        1,
        (1, 1),
        iteration_limit,
        stagnation_limit,
        0.0,
    )

    assert best_routes == [initial_route]
    assert best_cost == 0
    assert [iteration for iteration, *_ in history] == list(range(last_iteration + 1))


@pytest.mark.parametrize(
    "algorithm",
    [baseline.tabu_search, partial(tabu.tabu_search, demands=RECTANGLE_DEMANDS)],
    ids=["baseline", "advanced"],
)
def test_search_keeps_the_best_completed_move_at_the_time_limit(
    rectangle_fleet: list[Vehicle],
    algorithm: SearchAlgorithm,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initial_time = 100.0
    clock = Mock(side_effect=[100.25, 100.25, 100.5, 100.5])
    monkeypatch.setattr(shared, "perf_counter", clock)

    result = algorithm(
        rectangle_fleet, RECTANGLE, 1, (2, 2), 10, 10, initial_time, time_limit_s=0.5
    )

    assert result.best_routes == [[2, 4, 3]]
    assert result.best_cost == 14
    assert result.history == [(0, 18, 0.25, 18), (1, 14, 0.5, 14)]


def test_baseline_stops_when_no_route_can_be_reversed() -> None:
    fleet = build_initial_solution(2, 3, 1, [0, 1, 1], depot=1)
    assert fleet is not None

    routes, cost, history = baseline.tabu_search(
        fleet, [(0, 0), (1, 0), (2, 0)], 1, (1, 1), 10, 10, start_time=0.0
    )

    assert routes == [[2], [3]]
    assert cost == 6
    assert [iteration for iteration, *_ in history] == [0]


@pytest.mark.parametrize(
    ("use_phases", "expected_special_calls", "expected_final_attempts"),
    [
        (
            True,
            [
                (11, "intensify"),
                (12, "intensify"),
                (13, "intensify"),
                (21, "diversify"),
            ],
            2,
        ),
        (False, [], 1),
    ],
    ids=["phases-enabled", "phases-disabled"],
)
def test_advanced_search_uses_phases_and_retries_unavailable_diversification(
    plateau_fleet: list[Vehicle],
    monkeypatch: pytest.MonkeyPatch,
    use_phases: bool,
    expected_special_calls: list[tuple[int, str]],
    expected_final_attempts: int,
) -> None:
    # A single route permits reversals throughout the plateau and has no inter-route move.
    select_neighbor = Mock(wraps=tabu.find_best_neighbor)
    monkeypatch.setattr(tabu, "find_best_neighbor", select_neighbor)

    _, _, history = tabu.tabu_search(
        plateau_fleet,
        PLATEAU_COORDINATES,
        1,
        (1, 1),
        21,
        30,
        start_time=0.0,
        demands=PLATEAU_DEMANDS,
        use_phases=use_phases,
    )

    calls = [call.kwargs for call in select_neighbor.call_args_list]
    special_calls = [
        (kwargs["iteration"], phase)
        for kwargs in calls
        for phase in ("intensify", "diversify")
        if kwargs.get(phase, False)
    ]
    final_attempts = [kwargs for kwargs in calls if kwargs["iteration"] == 21]
    assert special_calls == expected_special_calls
    assert len(final_attempts) == expected_final_attempts
    assert not final_attempts[-1].get("diversify", False)
    assert history[-1][0] == 21


def test_advanced_search_skips_the_fallback_scan_after_time_expires(
    plateau_fleet: list[Vehicle], monkeypatch: pytest.MonkeyPatch
) -> None:
    start_time = 100.0
    time_limit_s = 1.0
    clock = Mock(return_value=start_time)
    monkeypatch.setattr(shared, "perf_counter", clock)

    def finish_scan(*_arguments: object, **options: object) -> object:
        if options.get("intensify"):
            clock.return_value = start_time + time_limit_s
            return None
        return DEFAULT

    select_neighbor = Mock(wraps=tabu.find_best_neighbor, side_effect=finish_scan)
    monkeypatch.setattr(tabu, "find_best_neighbor", select_neighbor)

    result = tabu.tabu_search(
        plateau_fleet,
        PLATEAU_COORDINATES,
        1,
        (1, 1),
        max_iterations=20,
        stagnation_limit=20,
        start_time=start_time,
        demands=PLATEAU_DEMANDS,
        time_limit_s=time_limit_s,
    )

    assert select_neighbor.call_count == 11
    assert select_neighbor.call_args.kwargs["intensify"] is True
    assert result.history[-1].iteration == 10
