"""
test_agent.py — агент-заглушка для проверки работы Orchestrator.

Протокол общения с ядром (stdin/stdout):
  - ядро присылает в stdin одну JSON-строку вида:
        {"type": "user_input", "text": "...", "memory_snapshot": [...]}
  - агент отвечает в stdout одной JSON-строкой вида:
        {"type": "agent_response", "agent": "test_agent", "text": "..."}

Никакой реальной логики/модели тут нет — агент просто эхо-отвечает
и считает количество полученных сообщений. Этого достаточно, чтобы
проверить, что Orchestrator умеет запускать подпроцесс, посылать ему
данные и читать ответ обратно.
"""

import json
import sys


def main() -> None:
    message_count = 0

    # Читаем стандартный ввод построчно, пока ядро не закроет канал
    # (или не пришлёт сообщение типа "shutdown").
    for raw_line in sys.stdin:
        raw_line = raw_line.strip()
        if not raw_line:
            continue

        try:
            message = json.loads(raw_line)
        except json.JSONDecodeError:
            response = {
                "type": "agent_response",
                "agent": "test_agent",
                "error": "invalid_json",
                "raw": raw_line,
            }
            print(json.dumps(response, ensure_ascii=False), flush=True)
            continue

        message_count += 1

        if message.get("type") == "shutdown":
            break

        user_text = message.get("text", "")
        response = {
            "type": "agent_response",
            "agent": "test_agent",
            "text": f"[test_agent эхо #{message_count}]: {user_text}",
            "received_history_items": len(message.get("memory_snapshot", [])),
        }
        print(json.dumps(response, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
