import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from core.orchestrator import Orchestrator  # noqa: E402

EXIT_COMMANDS = {"exit", "quit", "выход"}


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

            response = orch.handle_user_input(user_text, timeout=300.0)

            if response.get("error") == "agent_timeout":
                print("\n⏱ Агент не ответил за отведённое время. Попробуй ещё раз.\n")
                continue

            print(f"Агент: {response.get('text', response)}\n")

    except KeyboardInterrupt:
        print("\nЗавершение по Ctrl+C...")

    finally:
        orch.shutdown()


if __name__ == "__main__":
    main()
