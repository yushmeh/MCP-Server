import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from core.orchestrator import Orchestrator  # noqa: E402

EXIT_COMMANDS = {"exit", "quit", "выход"}

# Ключевые слова, по которым мы сами (на уровне кода, а не надеясь на
# модель) включаем строгий режим ТЗ с валидацией и автозапуском
# агента-разработчика — это надёжнее, чем полагаться на то, что любая
# модель аккуратно соблюдёт инструкцию из системного промпта.
SPEC_TRIGGERS = ("тз", "техническое задание", "план проекта")


def wants_spec(user_text: str) -> bool:
    text = user_text.lower()
    return any(trigger in text for trigger in SPEC_TRIGGERS)


def main() -> None:
    orch = Orchestrator(
        workspace_dir=Path(__file__).parent / "workspace",
        logs_dir=Path(__file__).parent / "logs",
    )

    agent_script = Path(__file__).parent / "agents" / "analyst.py"
    orch.start_agent("analyst", agent_script)

    # Даём подпроцессу время подняться.
    time.sleep(0.5)

    print("Агент готов к диалогу. Для выхода набери: exit / quit / выход\n")

    try:
        while True:
            try:
                user_text = input("Вы: ").strip()
            except EOFError:
                break

            if not user_text:
                continue
            if user_text.lower() in EXIT_COMMANDS:
                break

            parse_as_spec = wants_spec(user_text)
            response = orch.handle_user_input(user_text, timeout=300.0, parse_as_spec=parse_as_spec)

            if response.get("error") == "agent_timeout":
                print("\n⏱ Агент не ответил за отведённое время. Попробуй ещё раз.\n")
                continue

            if parse_as_spec and response.get("parsed_spec") is not None:
                # Статус (✅/❌) и результат разработчика уже напечатаны
                # самим оркестратором — не дублируем сырой JSON в чате.
                print()
                continue

            print(f"Агент: {response.get('text', response)}\n")

    except KeyboardInterrupt:
        print("\nЗавершение по Ctrl+C...")

    finally:
        orch.shutdown()


if __name__ == "__main__":
    main()
