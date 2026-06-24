from __future__ import annotations

from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
WORKSPACE_ROOT = PROJECT_ROOT / "workspace" / "projects"
LOGS_DIR = PROJECT_ROOT / "logs"


class PathEscapeError(Exception):
    """Попытка выйти за пределы разрешённой директории workspace/projects."""


def _ensure_dirs() -> None:
    WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)


def _log_operation(op: str, path: Path, status: str, detail: str = "") -> None:
    """Записывает результат файловой операции в logs/fs_operations.log."""
    _ensure_dirs()
    log_file = LOGS_DIR / "fs_operations.log"
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(
            f"{datetime.now().isoformat(timespec='seconds')} | "
            f"{op} | {status} | {path} | {detail}\n"
        )


def _resolve_safe(relative_path: str) -> Path:
    """Резолвит относительный путь и гарантирует, что итоговый путь
    остаётся внутри WORKSPACE_ROOT."""
    _ensure_dirs()
    if Path(relative_path).is_absolute():
        raise PathEscapeError(
            f"Путь должен быть относительным, получен абсолютный: '{relative_path}'"
        )

    candidate = (WORKSPACE_ROOT / relative_path).resolve()
    workspace_resolved = WORKSPACE_ROOT.resolve()

    try:
        candidate.relative_to(workspace_resolved)
    except ValueError:
        raise PathEscapeError(
            f"Путь '{relative_path}' выходит за пределы {workspace_resolved}"
        )

    return candidate


def create_directory(relative_path: str) -> Path:
    """Создаёт директорию внутри workspace/projects (с родительскими)."""
    try:
        target = _resolve_safe(relative_path)
        target.mkdir(parents=True, exist_ok=True)
        _log_operation("create_directory", target, "ok")
        return target
    except PathEscapeError as exc:
        _log_operation("create_directory", Path(relative_path), "blocked", str(exc))
        raise
    except Exception as exc:
        _log_operation("create_directory", Path(relative_path), "error", str(exc))
        raise


def write_file(relative_path: str, content: str) -> Path:
    """Записывает файл внутри workspace/projects, создавая родительские
    директории при необходимости."""
    try:
        target = _resolve_safe(relative_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        _log_operation("write_file", target, "ok", f"{len(content)} символов")
        return target
    except PathEscapeError as exc:
        _log_operation("write_file", Path(relative_path), "blocked", str(exc))
        raise
    except Exception as exc:
        _log_operation("write_file", Path(relative_path), "error", str(exc))
        raise
