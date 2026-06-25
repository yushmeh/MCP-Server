"""
tester.py — агент-тестировщик.

Принимает информацию о только что сгенерированном проекте (папка и
список файлов от агента-разработчика), для каждого Python-файла
генерирует юнит-тесты через Ollama, записывает их в <project>/tests/ и
запускает pytest через core/mcp_tools.py, возвращая оркестратору
статус выполнения (сколько тестов прошло/упало).

Протокол общения с ядром:
вход (stdin):  {"type": "test_request", "project_dir": "...",
                 "files_written": [...], "spec": {...} | не указано}
выход (stdout): {"type": "agent_response", "agent": "tester",
                  "project_dir": "...", "tests_written": [...],
                  "passed": N, "failed": M, "errors": [...], "raw_output": "..."}
"""

import json
import re
import sys
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.fs_tools import write_file, PathEscapeError  # noqa: E402
from core.mcp_tools import run_pytest  # noqa: E402

PROMPT_PATH = PROJECT_ROOT / "workspace" / "prompts" / "tester.md"
CONFIG_PATH = PROJECT_ROOT / "workspace" / "configs" / "config.json"
MEMORY_PATH = PROJECT_ROOT / "workspace" / "memory.json"
WORKSPACE_PROJECTS = PROJECT_ROOT / "workspace" / "projects"


def load_system_prompt() -> str:
    if PROMPT_PATH.exists():
        return PROMPT_PATH.read_text(encoding="utf-8")
    return "Ты — агент-тестировщик. Возвращай только код тестов pytest, без пояснений."


def load_config() -> dict:
    if CONFIG_PATH.exists() and CONFIG_PATH.stat().st_size > 0:
        try:
            return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


def load_spec_from_memory() -> dict | None:
    if MEMORY_PATH.exists() and MEMORY_PATH.stat().st_size > 0:
        try:
            memory = json.loads(MEMORY_PATH.read_text(encoding="utf-8"))
            return memory.get("last_tech_spec")
        except json.JSONDecodeError:
            return None
    return None


def load_last_project_from_memory() -> dict | None:
    if MEMORY_PATH.exists() and MEMORY_PATH.stat().st_size > 0:
        try:
            memory = json.loads(MEMORY_PATH.read_text(encoding="utf-8"))
            return memory.get("last_generated_project")
        except json.JSONDecodeError:
            return None
    return None


def clean_code_response(raw_text: str) -> str:
    """Убирает markdown-обёртку и вступительные фразы, оставляя только код
    (та же логика, что и в agents/developer.py)."""
    text = raw_text.strip()

    fence_match = re.search(r"```[a-zA-Z]*\n(.*?)```", text, re.DOTALL)
    if fence_match:
        return fence_match.group(1).strip()

    lines = text.splitlines()
    code_start_pattern = re.compile(r"^\s*(#|import|from|def|class|@)")
    if lines and not code_start_pattern.match(lines[0]):
        intro_markers = ("вот", "конечно", "держи", "пожалуйста", "here is", "sure", "below is")
        if any(lines[0].lower().startswith(m) for m in intro_markers):
            lines = lines[1:]
    return "\n".join(lines).strip()


def call_ollama(
    system_prompt: str, user_prompt: str, host: str, model: str, timeout: float = 280.0
) -> str:
    url = f"{host.rstrip('/')}/api/generate"
    payload = {
        "model": model,
        "prompt": user_prompt,
        "system": system_prompt,
        "stream": False,
    }
    response = requests.post(url, json=payload, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    return data.get("response", "")


def build_test_prompt(spec: dict, module_name: str, source_code: str) -> str:
    return (
        f"Проект: {spec.get('title', '')}\n"
        f"Описание: {spec.get('description', '')}\n\n"
        f"Имя модуля для тестирования (уже импортирован как `{module_name}`): {module_name}\n"
        f"Исходный код модуля:\n{source_code}"
    )


def generate_tests_for_project(
    project_dir: str, files_written: list[str], spec: dict, host: str, model: str, system_prompt: str
) -> dict:
    """Для каждого Python-файла проекта генерирует тестовый файл в
    <project_dir>/tests/test_<module>.py и затем запускает pytest."""
    project_path = WORKSPACE_PROJECTS / project_dir

    py_files = [
        f for f in files_written
        if f.endswith(".py") and "/tests/" not in f and not f.split("/")[-1].startswith("test_")
    ]

    tests_written: list[str] = []
    errors: list[str] = []

    for file_rel in py_files:
        # file_rel выглядит как "<project_dir>/main.py" — берём имя файла
        # относительно самого проекта.
        try:
            source_rel = str(Path(file_rel).relative_to(project_dir))
        except ValueError:
            source_rel = Path(file_rel).name

        if "/" in source_rel:
            # Модули во вложенных папках пропускаем — для учебного
            # пайплайна ограничиваемся плоской структурой проекта.
            continue

        module_name = Path(source_rel).stem
        source_path = project_path / source_rel
        if not source_path.exists():
            errors.append(f"{source_rel}: файл не найден на диске, пропущен")
            continue

        source_code = source_path.read_text(encoding="utf-8", errors="replace")
        prompt = build_test_prompt(spec, module_name, source_code)

        try:
            raw_response = call_ollama(system_prompt, prompt, host, model)
            test_body = clean_code_response(raw_response)

            header = (
                "import sys\n"
                "import os\n"
                "import pytest\n"
                "sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))\n"
                f"import {module_name}\n"
                f"from {module_name} import *\n\n"
            )
            test_file_content = header + test_body

            test_relative_path = f"{project_dir}/tests/test_{module_name}.py"
            write_file(test_relative_path, test_file_content)
            tests_written.append(test_relative_path)
        except PathEscapeError as exc:
            errors.append(f"{source_rel}: запрещённый путь ({exc})")
        except requests.exceptions.ConnectionError:
            errors.append(f"{source_rel}: не удалось подключиться к Ollama")
        except requests.exceptions.Timeout:
            errors.append(f"{source_rel}: Ollama не успела ответить за отведённое время")
        except Exception as exc:
            errors.append(f"{source_rel}: {exc}")

    if not tests_written:
        return {
            "tests_written": [],
            "errors": errors or ["Не удалось сгенерировать ни одного теста"],
            "passed": 0,
            "failed": 0,
            "collection_errors": 0,
            "raw_output": "",
        }

    pytest_result = run_pytest(project_dir)

    return {
        "tests_written": tests_written,
        "errors": errors,
        "passed": pytest_result.get("passed", 0),
        "failed": pytest_result.get("failed", 0),
        "collection_errors": pytest_result.get("collection_errors", 0),
        "raw_output": pytest_result.get("raw_output", ""),
    }


def main() -> None:
    system_prompt = load_system_prompt()
    config = load_config()
    ollama_cfg = config.get("ollama", {})
    host = ollama_cfg.get("host", "http://localhost:11434")
    model = ollama_cfg.get("default_model", "qwen2.5-coder:3b")

    print(f"[tester] старт. host={host} model={model}", file=sys.stderr, flush=True)

    for raw_line in sys.stdin:
        raw_line = raw_line.strip()
        if not raw_line:
            continue

        try:
            message = json.loads(raw_line)
        except json.JSONDecodeError:
            print(json.dumps({
                "type": "agent_response",
                "agent": "tester",
                "error": "invalid_json",
                "raw": raw_line,
            }, ensure_ascii=False), flush=True)
            continue

        if message.get("type") == "shutdown":
            break

        spec = message.get("spec") or load_spec_from_memory()
        project_dir = message.get("project_dir")
        files_written = message.get("files_written")

        if not project_dir or not files_written:
            last_project = load_last_project_from_memory()
            if last_project:
                project_dir = project_dir or last_project.get("project_dir")
                files_written = files_written or last_project.get("files_written")

        if not spec or not project_dir or not files_written:
            response = {
                "type": "agent_response",
                "agent": "tester",
                "error": "no_project_context",
                "text": "Нет данных о проекте для тестирования (ни в сообщении, ни в общей памяти).",
            }
            print(json.dumps(response, ensure_ascii=False), flush=True)
            continue

        print(
            f"[tester] получен проект «{project_dir}», файлов на проверку: {len(files_written)}",
            file=sys.stderr, flush=True,
        )

        result = generate_tests_for_project(project_dir, files_written, spec, host, model, system_prompt)
        response = {
            "type": "agent_response",
            "agent": "tester",
            "project_dir": project_dir,
            **result,
        }
        print(json.dumps(response, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
