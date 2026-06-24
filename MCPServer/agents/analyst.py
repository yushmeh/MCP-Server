import json
import sys
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).parent.parent
PROMPT_PATH = PROJECT_ROOT / "workspace" / "prompts" / "analyst.md"
CONFIG_PATH = PROJECT_ROOT / "workspace" / "configs" / "config.json"


def load_system_prompt() -> str:
    if PROMPT_PATH.exists():
        return PROMPT_PATH.read_text(encoding="utf-8")
    return "Ты — агент-аналитик. Превращай запрос пользователя в JSON-ТЗ."


def load_config() -> dict:
    if CONFIG_PATH.exists() and CONFIG_PATH.stat().st_size > 0:
        try:
            return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


def call_ollama(
    system_prompt: str,
    user_text: str,
    host: str,
    model: str,
    timeout: float = 280.0,
    force_json: bool = False,
) -> str:
    """Запрос к Ollama через REST API (/api/generate)."""
    url = f"{host.rstrip('/')}/api/generate"
    payload = {
        "model": model,
        "prompt": user_text,
        "system": system_prompt,
        "stream": False,
    }
    if force_json:
        payload["format"] = "json"

    response = requests.post(url, json=payload, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    return data.get("response", "")


def main() -> None:
    system_prompt = load_system_prompt()
    config = load_config()
    ollama_cfg = config.get("ollama", {})
    host = ollama_cfg.get("host", "http://localhost:11434")
    model = ollama_cfg.get("default_model", "qwen2.5-coder:3b")

    print(f"[analyst] старт. host={host} model={model}", file=sys.stderr, flush=True)

    for raw_line in sys.stdin:
        raw_line = raw_line.strip()
        if not raw_line:
            continue

        try:
            message = json.loads(raw_line)
        except json.JSONDecodeError:
            print(json.dumps({
                "type": "agent_response",
                "agent": "analyst",
                "error": "invalid_json",
                "raw": raw_line,
            }, ensure_ascii=False), flush=True)
            continue

        if message.get("type") == "shutdown":
            break

        user_text = message.get("text", "")
        want_spec = message.get("want_spec", False)

        try:
            llm_text = call_ollama(system_prompt, user_text, host, model, force_json=want_spec)
            response = {
                "type": "agent_response",
                "agent": "analyst",
                "text": llm_text,
                "source": "ollama",
            }
        except requests.exceptions.ConnectionError:
            response = {
                "type": "agent_response",
                "agent": "analyst",
                "error": "ollama_unreachable",
                "text": (
                    "Не удалось подключиться к Ollama на "
                    f"{host}. Проверь, что Ollama запущена (`ollama serve`)."
                ),
            }
        except requests.exceptions.Timeout:
            response = {
                "type": "agent_response",
                "agent": "analyst",
                "error": "ollama_timeout",
                "text": "Ollama не успела ответить за отведённое время. Попробуй ещё раз.",
            }
        except Exception as exc:
            response = {
                "type": "agent_response",
                "agent": "analyst",
                "error": "ollama_error",
                "text": f"Ошибка при обращении к Ollama: {exc}",
            }

        print(json.dumps(response, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
