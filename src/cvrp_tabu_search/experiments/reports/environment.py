import json
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ENVIRONMENT_FILENAME = "environment.json"

type EnvironmentValue = str | int | None


def collect_environment() -> dict[str, EnvironmentValue]:
    return {
        "recorded_at": datetime.now().isoformat(timespec="seconds"),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": _processor_name(),
        "logical_cpus": os.cpu_count(),
        "memory_bytes": _memory_bytes(),
        "git_commit": _git_commit(),
    }


def write_environment(directory: Path) -> Path:
    path = directory / ENVIRONMENT_FILENAME
    content = json.dumps(collect_environment(), indent=2, ensure_ascii=False)
    path.write_text(content + "\n", encoding="utf-8")
    return path


def _processor_name() -> str:
    if sys.platform == "darwin":
        name = _command_output(["sysctl", "-n", "machdep.cpu.brand_string"])
        if name:
            return name
    elif sys.platform.startswith("linux"):
        for line in _read_lines(Path("/proc/cpuinfo")):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    elif sys.platform == "win32":
        name = _windows_processor_name()
        if name:
            return name

    return platform.processor() or "unknown"


def _memory_bytes() -> int | None:
    if sys.platform == "darwin":
        value = _command_output(["sysctl", "-n", "hw.memsize"])
        return int(value) if value and value.isdigit() else None

    if sys.platform.startswith("linux"):
        for line in _read_lines(Path("/proc/meminfo")):
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024
        return None

    if sys.platform == "win32":
        return _windows_memory_bytes()

    return None


def _windows_processor_name() -> str | None:
    if sys.platform != "win32":
        return None

    try:
        import winreg

        key_path = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path) as key:
            value, _ = winreg.QueryValueEx(key, "ProcessorNameString")
            return str(value).strip()
    except (ImportError, OSError):
        return None


def _windows_memory_bytes() -> int | None:
    if sys.platform != "win32":
        return None

    import ctypes

    class _MemoryStatus(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = _MemoryStatus()
    status.dwLength = ctypes.sizeof(_MemoryStatus)
    kernel32 = ctypes.windll.kernel32
    if not kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return None
    return int(status.ullTotalPhys)


def _git_commit() -> str | None:
    package_directory = Path(__file__).resolve().parent
    return _command_output(["git", "-C", str(package_directory), "rev-parse", "HEAD"])


def _command_output(command: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=5, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None

    output = completed.stdout.strip()
    return output if completed.returncode == 0 and output else None


def _read_lines(path: Path) -> list[str]:
    try:
        return path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
