import json
from pathlib import Path

from cvrp_tabu_search.experiments.reports.environment import (
    ENVIRONMENT_FILENAME,
    collect_environment,
    write_environment,
)


def test_environment_describes_python_machine_and_code_version() -> None:
    environment = collect_environment()

    assert set(environment) == {
        "recorded_at",
        "python_version",
        "python_implementation",
        "platform",
        "machine",
        "processor",
        "logical_cpus",
        "memory_bytes",
        "git_commit",
    }
    assert str(environment["python_version"]).startswith("3.")
    assert isinstance(environment["logical_cpus"], int)


def test_environment_is_written_as_json(tmp_path: Path) -> None:
    path = write_environment(tmp_path)

    assert path == tmp_path / ENVIRONMENT_FILENAME
    content = json.loads(path.read_text(encoding="utf-8"))
    assert content["python_version"].startswith("3.")
