from __future__ import annotations

import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
WORKSPACE_ROOT = PROJECT_ROOT / "workspace" / "projects"
LOGS_DIR = PROJECT_ROOT / "logs"


def _ensure_dirs() -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)


def _log_test_run(project_dir: str, summary: dict, returncode: int, detail: str = "") -> None:
    """Записывает результат прогона pytest в logs/test_runs.log."""
    _ensure_dirs()
    log_file = LOGS_DIR / "test_runs.log"
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(
            f"{datetime.now().isoformat(timespec='seconds')} | {project_dir} | "
            f"passed={summary.get('passed', 0)} failed={summary.get('failed', 0)} "
            f"collection_errors={summary.get('collection_errors', 0)} "
            f"returncode={returncode} | {detail}\n"
        )


def find_project_python(project_path: Path) -> str:
    if sys.platform == "win32":
        candidates = [
            project_path / "venv" / "Scripts" / "python.exe",
            project_path / ".venv" / "Scripts" / "python.exe",
        ]
    else:
        candidates = [
            project_path / "venv" / "bin" / "python",
            project_path / ".venv" / "bin" / "python",
        ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return sys.executable


def _parse_pytest_summary(output: str) -> dict:
    """Разбирает финальную строку pytest (например,
    '2 passed, 1 failed in 0.34s' или '3 passed in 0.12s') в количества
    passed/failed/collection_errors."""
    summary_line = ""
    for line in reversed(output.strip().splitlines()):
        if re.search(r"\bin [\d.]+s\b", line) or "no tests ran" in line.lower():
            summary_line = line
            break
    target = summary_line or output

    def _extract(pattern: str) -> int:
        m = re.search(pattern, target)
        return int(m.group(1)) if m else 0

    return {
        "passed": _extract(r"(\d+)\s+passed"),
        "failed": _extract(r"(\d+)\s+failed"),
        "collection_errors": _extract(r"(\d+)\s+error"),
        "summary_line": summary_line,
    }


def run_pytest(project_dir: str, timeout: float = 180.0) -> dict:
    """Запускает pytest для сгенерированного проекта и возвращает разбор результата.

    :param project_dir: имя папки внутри workspace/projects/ (например,
        "telegram_bot_napominaniy").
    :return: словарь с returncode, raw_output (хвост вывода) и разбором
        passed/failed/collection_errors/summary_line.
    """
    project_path = WORKSPACE_ROOT / project_dir
    if not project_path.exists():
        detail = f"Папка проекта не найдена: {project_path}"
        _log_test_run(project_dir, {}, -3, detail)
        return {
            "returncode": -3, "raw_output": detail,
            "passed": 0, "failed": 0, "collection_errors": 0, "summary_line": "",
        }

    python_exe = find_project_python(project_path)

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    cmd = [python_exe, "-m", "pytest", str(project_path), "-q", "--no-header"]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            timeout=timeout,
            cwd=str(project_path),
        )
        output = (result.stdout or "") + "\n" + (result.stderr or "")
        summary = _parse_pytest_summary(output)
        _log_test_run(project_dir, summary, result.returncode, summary["summary_line"])
        return {
            "returncode": result.returncode,
            "raw_output": output.strip()[-4000:],
            **summary,
        }
    except subprocess.TimeoutExpired:
        detail = f"Pytest не успел выполниться за {timeout} секунд."
        _log_test_run(project_dir, {}, -1, detail)
        return {
            "returncode": -1, "raw_output": detail,
            "passed": 0, "failed": 0, "collection_errors": 0, "summary_line": "",
            "timeout": True,
        }
    except FileNotFoundError as exc:
        detail = f"pytest не найден ({exc}). Установи его: pip install pytest"
        _log_test_run(project_dir, {}, -2, detail)
        return {
            "returncode": -2, "raw_output": detail,
            "passed": 0, "failed": 0, "collection_errors": 0, "summary_line": "",
        }
