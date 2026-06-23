from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

# Обязательные поля JSON-схемы технического задания.
REQUIRED_FIELDS = ["title", "description", "requirements", "modules"]

# Поля, которые обязательно должны быть списками строк, если присутствуют.
LIST_FIELDS = ["requirements", "modules", "tech_stack"]


def _extract_json_block(raw_text: str) -> Optional[str]:
    """Достаёт JSON-объект из текста ответа модели."""
    md_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
    if md_match:
        return md_match.group(1)

    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return raw_text[start:end + 1]
    return None


def validate_llm_response(
    raw_text: str, logs_dir: str | Path
) -> Tuple[bool, Optional[dict], list[str]]:
    """Проверяет ответ LLM на соответствие JSON-схеме ТЗ."""
    errors: list[str] = []
    logs_dir = Path(logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)

    json_block = _extract_json_block(raw_text)
    if json_block is None:
        errors.append("Не найден JSON-объект в ответе модели")
        _log_invalid(raw_text, errors, logs_dir)
        return False, None, errors

    try:
        data = json.loads(json_block)
    except json.JSONDecodeError as exc:
        errors.append(f"Ошибка разбора JSON: {exc}")
        _log_invalid(raw_text, errors, logs_dir)
        return False, None, errors

    if not isinstance(data, dict):
        errors.append("Верхний уровень JSON должен быть объектом (dict)")
        _log_invalid(raw_text, errors, logs_dir)
        return False, None, errors

    for field in REQUIRED_FIELDS:
        if field not in data:
            errors.append(f"Отсутствует обязательное поле: '{field}'")

    for field in LIST_FIELDS:
        if field in data and not isinstance(data[field], list):
            errors.append(f"Поле '{field}' должно быть списком строк")

    if errors:
        _log_invalid(raw_text, errors, logs_dir)
        return False, data, errors

    return True, data, []


def _log_invalid(raw_text: str, errors: list[str], logs_dir: Path) -> None:
    """Записывает невалидный ответ LLM в logs/invalid_llm_responses.log."""
    log_file = logs_dir / "invalid_llm_responses.log"
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"=== {datetime.now().isoformat(timespec='seconds')} ===\n")
        f.write(f"Ошибки: {errors}\n")
        f.write("Исходный ответ модели:\n")
        f.write(raw_text + "\n")
        f.write("-" * 60 + "\n")
