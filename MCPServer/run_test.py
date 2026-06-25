"""
run_test.py — ручная проверка ядра MCP сервера.

Запускает test_agent как подпроцесс, посылает ему пару сообщений
от "пользователя" и печатает ответы. Также (если Ollama запущена)
делает один тестовый запрос к ней.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from core.orchestrator import Orchestrator  # noqa: E402


def main() -> None:
    orch = Orchestrator(
        workspace_dir=Path(__file__).parent / "workspace",
        logs_dir=Path(__file__).parent / "logs",
    )

    agent_script = Path(__file__).parent / "agents" / "test_agent.py"
    orch.start_agent("test_agent", agent_script)

    # Даём подпроцессу время подняться
    time.sleep(0.3)

    print("\n--- Проверка цепочки: пользователь -> test_agent ---")
    for text in ["Привет!", "Как дела?", "Запомни число 42"]:
        response = orch.handle_user_input(text)
        print(f"Пользователь: {text}")
        print(f"Агент:        {response}\n")

    print("--- Текущая общая память ---")
    print(orch.memory)

    print("\n--- Проверка связи с Ollama (опционально) ---")
    try:
        models = orch.list_ollama_models()
        print("Установленные модели Ollama:", models)
        if models:
            answer = orch.query_ollama("Скажи привет одним словом", model=models[0], timeout=180.0)
            print("Ответ модели:", answer)
    except Exception as exc:  # noqa: BLE001
        print("Ollama недоступна (это нормально, если ты её ещё не установил/не запустил):", exc)

    orch.shutdown()


if __name__ == "__main__":
    main()
