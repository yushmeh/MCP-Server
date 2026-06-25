"""
developer.py — агент-разработчик.

Принимает структурированное ТЗ (то же, что готовит агент-аналитик в
режиме ТЗ — см. workspace/prompts/analyst.md) и генерирует код для
каждого файла из списка "files" в ТЗ, используя Ollama. Готовые файлы
записываются в workspace/projects/<slug-проекта>/ через безопасные
инструменты файловой системы (core/fs_tools.py).

ТЗ может прийти двумя способами:
1. В самом сообщении от оркестратора: {"type": "build_request", "spec": {...}}
2. Если поле "spec" не передано — агент сам читает workspace/memory.json
   и берёт оттуда ключ "last_tech_spec" (общая память оркестратора).

Протокол общения с ядром:
вход (stdin):  {"type": "build_request", "spec": {...} | не указано}
выход (stdout): {"type": "agent_response", "agent": "developer",
                  "project_dir": "...", "files_written": [...], "errors": [...]}
"""

import json
import re
import sys
from datetime import datetime
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.fs_tools import create_directory, write_file, PathEscapeError  # noqa: E402

PROMPT_PATH = PROJECT_ROOT / "workspace" / "prompts" / "developer.md"
CONFIG_PATH = PROJECT_ROOT / "workspace" / "configs" / "config.json"
MEMORY_PATH = PROJECT_ROOT / "workspace" / "memory.json"


def load_system_prompt() -> str:
    if PROMPT_PATH.exists():
        return PROMPT_PATH.read_text(encoding="utf-8")
    return "Ты — агент-разработчик. Возвращай только код, без пояснений."


def load_config() -> dict:
    if CONFIG_PATH.exists() and CONFIG_PATH.stat().st_size > 0:
        try:
            return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


def load_spec_from_memory() -> dict | None:
    """Резервный способ получить ТЗ — напрямую из общей памяти
    оркестратора (workspace/memory.json), если оно не пришло в сообщении."""
    if MEMORY_PATH.exists() and MEMORY_PATH.stat().st_size > 0:
        try:
            memory = json.loads(MEMORY_PATH.read_text(encoding="utf-8"))
            return memory.get("last_tech_spec")
        except json.JSONDecodeError:
            return None
    return None


_CYRILLIC_TO_LATIN = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e",
    "ж": "zh", "з": "z", "и": "i", "й": "y", "к": "k", "л": "l", "м": "m",
    "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u",
    "ф": "f", "х": "h", "ц": "ts", "ч": "ch", "ш": "sh", "щ": "sch",
    "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "yu", "я": "ya",
}


def slugify(title: str) -> str:
    """Превращает название проекта (в т.ч. на кириллице) в безопасное,
    уникально читаемое имя папки."""
    text = (title or "project").strip().lower()
    text = "".join(_CYRILLIC_TO_LATIN.get(ch, ch) for ch in text)
    slug = re.sub(r"[^a-zA-Z0-9\-_]+", "_", text)
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug or "project"


def clean_code_response(raw_text: str) -> str:
    """Убирает markdown-обёртку и вступительные фразы, оставляя только код."""
    text = raw_text.strip()

    # Если ответ обёрнут в ```...``` — берём содержимое первого такого блока.
    fence_match = re.search(r"```[a-zA-Z]*\n(.*?)```", text, re.DOTALL)
    if fence_match:
        return fence_match.group(1).strip()

    # Без markdown-обёртки — отрезаем типичную вступительную фразу на
    # первой строке, если она явно не похожа на код.
    lines = text.splitlines()
    code_start_pattern = re.compile(r"^\s*(#|import|from|def|class|<|\{|//|/\*|using|package)")
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


def build_file_prompt(spec: dict, file_entry: dict, fix_context: str | None = None) -> str:
    """Формирует промпт для генерации ОДНОГО конкретного файла.

    :param fix_context: если передан — это отчёт о падении тестов
        (вывод pytest), который добавляется к промпту с просьбой
        исправить код так, чтобы тесты проходили.
    """
    prompt = (
        f"Проект: {spec.get('title', '')}\n"
        f"Описание: {spec.get('description', '')}\n"
        f"Требования: {', '.join(spec.get('requirements', []))}\n"
        f"Стек: {', '.join(spec.get('tech_stack', []))}\n\n"
        f"Сгенерируй содержимое файла: {file_entry.get('path')}\n"
        f"Назначение файла: {file_entry.get('description', '')}"
    )
    if fix_context:
        prompt += (
            "\n\nВНИМАНИЕ: предыдущая версия этого файла (или связанного с ним "
            "кода в проекте) не прошла автоматические тесты. Вот вывод тестов:\n"
            f"{fix_context}\n\n"
            "Исправь код этого файла так, чтобы устранить причину падения "
            "тестов, сохранив остальную функциональность и не убирая нужные "
            "элементы (импорты, функции), если они не являются причиной ошибки."
        )
    return prompt


def process_build_request(
    spec: dict,
    host: str,
    model: str,
    system_prompt: str,
    existing_project_dir: str | None = None,
    fix_context: str | None = None,
) -> dict:
    """Итерируется по списку файлов из ТЗ, генерирует и записывает каждый.

    :param existing_project_dir: если передан — генерация идёт в УЖЕ
        существующую папку проекта (режим исправления по отчёту
        тестировщика), без создания новой папки и без защиты от
        перезатирания (мы здесь сознательно перезаписываем файлы).
    :param fix_context: отчёт о падении тестов, передаётся в промпт
        каждого файла (см. build_file_prompt).
    """
    files = spec.get("files") or []
    if not files:
        # Резерв: если ТЗ не содержит явного списка файлов — генерируем
        # один main.py по общему описанию, чтобы пайплайн не падал.
        files = [{"path": "main.py", "description": spec.get("description", "")}]

    if existing_project_dir:
        project_dir = existing_project_dir
        create_directory(project_dir)
    else:
        project_dir = slugify(spec.get("title", "project"))

        # Защита от перезатирания: если папка с таким именем уже существует
        # и не пуста — добавляем суффикс с таймстампом, чтобы не потерять
        # результаты предыдущего запуска.
        existing = (PROJECT_ROOT / "workspace" / "projects" / project_dir)
        if existing.exists() and any(existing.iterdir()):
            project_dir = f"{project_dir}_{datetime.now():%Y%m%d_%H%M%S}"

        create_directory(project_dir)

    written: list[str] = []
    errors: list[str] = []

    for file_entry in files:
        file_path = file_entry.get("path")
        if not file_path:
            errors.append("Пропущен файл без указанного 'path'")
            continue

        full_relative_path = f"{project_dir}/{file_path}"
        prompt = build_file_prompt(spec, file_entry, fix_context=fix_context)

        try:
            raw_response = call_ollama(system_prompt, prompt, host, model)
            code = clean_code_response(raw_response)
            write_file(full_relative_path, code)
            written.append(full_relative_path)
        except PathEscapeError as exc:
            errors.append(f"{file_path}: запрещённый путь ({exc})")
        except requests.exceptions.ConnectionError:
            errors.append(f"{file_path}: не удалось подключиться к Ollama")
        except requests.exceptions.Timeout:
            errors.append(f"{file_path}: Ollama не успела ответить за отведённое время")
        except Exception as exc:
            errors.append(f"{file_path}: {exc}")

    return {
        "project_dir": project_dir,
        "files_written": written,
        "errors": errors,
    }


def main() -> None:
    system_prompt = load_system_prompt()
    config = load_config()
    ollama_cfg = config.get("ollama", {})
    host = ollama_cfg.get("host", "http://localhost:11434")
    model = ollama_cfg.get("default_model", "qwen2.5-coder:3b")

    print(f"[developer] старт. host={host} model={model}", file=sys.stderr, flush=True)

    for raw_line in sys.stdin:
        raw_line = raw_line.strip()
        if not raw_line:
            continue

        try:
            message = json.loads(raw_line)
        except json.JSONDecodeError:
            print(json.dumps({
                "type": "agent_response",
                "agent": "developer",
                "error": "invalid_json",
                "raw": raw_line,
            }, ensure_ascii=False), flush=True)
            continue

        if message.get("type") == "shutdown":
            break

        spec = message.get("spec") or load_spec_from_memory()
        if not spec:
            response = {
                "type": "agent_response",
                "agent": "developer",
                "error": "no_spec",
                "text": "Нет доступного ТЗ для разработки (ни в сообщении, ни в общей памяти).",
            }
            print(json.dumps(response, ensure_ascii=False), flush=True)
            continue

        message_type = message.get("type")

        if message_type == "fix_request":
            project_dir = message.get("project_dir")
            test_report = message.get("test_report", {})
            fix_context = (
                f"Пройдено: {test_report.get('passed', 0)}, "
                f"Упало: {test_report.get('failed', 0)}\n"
                f"Ошибки генерации тестов: {test_report.get('errors', [])}\n"
                f"Вывод pytest:\n{test_report.get('raw_output', '')}"
            )
            print(
                f"[developer] получен запрос на исправление проекта «{project_dir}» "
                f"(упало тестов: {test_report.get('failed', 0)})",
                file=sys.stderr, flush=True,
            )
            result = process_build_request(
                spec, host, model, system_prompt,
                existing_project_dir=project_dir,
                fix_context=fix_context,
            )
        else:
            print(
                f"[developer] получено ТЗ «{spec.get('title')}», файлов: {len(spec.get('files', []))}",
                file=sys.stderr, flush=True,
            )
            result = process_build_request(spec, host, model, system_prompt)

        response = {
            "type": "agent_response",
            "agent": "developer",
            **result,
        }
        print(json.dumps(response, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
