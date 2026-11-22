from datetime import datetime
from pathlib import Path

DEFAULT_OUTPUT_DIR = Path("results")


def create_report_directory(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    report_directory = output_dir / timestamp
    sequence = 1

    while True:
        try:
            report_directory.mkdir()
        except FileExistsError:
            sequence += 1
            report_directory = output_dir / f"{timestamp}-{sequence}"
        else:
            return report_directory
