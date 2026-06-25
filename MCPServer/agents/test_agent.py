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
