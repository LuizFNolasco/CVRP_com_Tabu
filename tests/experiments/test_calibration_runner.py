import csv
from pathlib import Path
from unittest.mock import Mock

import pytest

from cvrp_tabu_search.experiments import calibration
from cvrp_tabu_search.experiments.execution import Run, SearchAlgorithm
from cvrp_tabu_search.instances import Instance, load_default_instance
from cvrp_tabu_search.search.shared import SearchParameters, TabuTenure


def _read_rows(output_dir: Path) -> list[dict[str, str]]:
    [report] = output_dir.glob("*/calibration.csv")
    with report.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def test_calibration_preserves_completed_rows_when_a_later_trial_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    instance = load_default_instance()
    tenures = [TabuTenure(2, 7), TabuTenure(2, 9)]
    run = Run(
        seed=1,
        cost=840,
        history=[],
        total_time_s=180.05,
        time_to_best_s=150.0,
        time_to_target_s=None,
    )

    def run_trial_and_check_table(
        _instance: Instance,
        _search: SearchAlgorithm,
        parameters: SearchParameters,
        *,
        seed: int,
    ) -> Run | None:
        assert seed == run.seed
        if parameters.tabu_tenure == tenures[0]:
            return run

        [completed_row] = _read_rows(tmp_path)
        assert completed_row["best_cost"] == "840"
        return None

    run_trial = Mock(side_effect=run_trial_and_check_table)
    monkeypatch.setattr(calibration, "run_once", run_trial)

    with pytest.raises(calibration.CalibrationError, match=instance.name):
        calibration.run_calibration(
            [instance], tenures, time_limit_s=180.0, output_dir=tmp_path
        )

    trial_parameters = [call.args[2] for call in run_trial.call_args_list]
    assert [parameters.tabu_tenure for parameters in trial_parameters] == tenures
    assert {parameters.time_limit_s for parameters in trial_parameters} == {180.0}
    [row] = _read_rows(tmp_path)
    assert row["instance"] == instance.name
    assert row["tabu_tenure_max"] == "7"
    assert row["total_time_s"] == "180.05"
    assert row["best_cost"] == "840"
