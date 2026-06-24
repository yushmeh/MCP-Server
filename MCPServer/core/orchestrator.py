from __future__ import annotations

import json
import logging
import os
import queue
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import requests

from core.parser import validate_llm_response


class AgentHandle:
    """Обёртка над подпроцессом одного агента."""

    def __init__(self, name: str, process: subprocess.Popen):
        self.name = name
        self.process = process
        self.stdout_queue: "queue.Queue[str]" = queue.Queue()
        self._reader_thread = threading.Thread(
            target=self._read_stdout, daemon=True
        )
        self._reader_thread.start()

        # stderr агента раньше просто терялся в трубе — теперь печатаем
        # его в консоль ядра с префиксом, чтобы видеть ошибки/диагностику
        # из самого агента (например, причину отказа Ollama).
        self._stderr_thread = threading.Thread(
            target=self._read_stderr, daemon=True
        )
        self._stderr_thread.start()

    def _read_stdout(self) -> None:
        # Построчно читаем stdout агента и кладём в очередь,
        # пока процесс жив.
        assert self.process.stdout is not None
        for line in self.process.stdout:
            line = line.rstrip("\n")
            if line:
                self.stdout_queue.put(line)

    def _read_stderr(self) -> None:
        # Построчно читаем stderr агента и сразу печатаем в консоль ядра.
        if self.process.stderr is None:
            return
        for line in self.process.stderr:
            line = line.rstrip("\n")
            if line:
                print(f"[agent:{self.name}][stderr] {line}", file=sys.stderr, flush=True)

    def is_alive(self) -> bool:
        return self.process.poll() is None

    def send(self, message: dict) -> None:
        if not self.is_alive():
            raise RuntimeError(f"Агент '{self.name}' не запущен / уже завершился")
        assert self.process.stdin is not None
        self.process.stdin.write(json.dumps(message, ensure_ascii=False) + "\n")
        self.process.stdin.flush()

    def receive(self, timeout: float = 5.0) -> Optional[dict]:
        try:
            line = self.stdout_queue.get(timeout=timeout)
        except queue.Empty:
            return None
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            # Агент написал что-то не в формате JSON — отдаём как есть.
            return {"raw": line}

    def stop(self) -> None:
        if self.is_alive():
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()


class Orchestrator:
    def __init__(
        self,
        workspace_dir: str | Path = "workspace",
        logs_dir: str | Path = "logs",
        ollama_host: str = "http://localhost:11434",
    ):
        self.workspace_dir = Path(workspace_dir)
        self.logs_dir = Path(logs_dir)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        self.ollama_host = ollama_host.rstrip("/")

        self.agents: dict[str, AgentHandle] = {}
        # Порядок добавления агентов важен — первый агент в этом
        # списке получает ввод пользователя первым.
        self._agent_order: list[str] = []

        self.memory: dict[str, Any] = {}
        self._memory_path = self.workspace_dir / "memory.json"
        self._memory_lock = threading.Lock()
        self._load_memory()

        self.logger = self._setup_logging()
        self.logger.info("Orchestrator инициализирован. workspace=%s logs=%s",
                          self.workspace_dir, self.logs_dir)

    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger("mcp_orchestrator")
        logger.setLevel(logging.DEBUG)
        if logger.handlers:
            return logger  # повторная инициализация в тестах

        log_file = self.logs_dir / f"core_{datetime.now():%Y-%m-%d}.log"
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        console_handler = logging.StreamHandler(sys.stdout)

        fmt = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(fmt)
        console_handler.setFormatter(fmt)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        return logger

    def _load_memory(self) -> None:
        if self._memory_path.exists():
            try:
                self.memory = json.loads(self._memory_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                self.memory = {}

    def _save_memory(self) -> None:
        with self._memory_lock:
            self._memory_path.write_text(
                json.dumps(self.memory, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    def remember(self, key: str, value: Any) -> None:
        """Записать значение в общую память и сохранить на диск."""
        with self._memory_lock:
            self.memory[key] = value
        self._save_memory()
        self.logger.debug("memory[%s] = %r", key, value)

    def recall(self, key: str, default: Any = None) -> Any:
        return self.memory.get(key, default)

    def append_history(self, role: str, content: str) -> None:
        """Удобный хелпер: ведём общую историю диалога в памяти."""
        history = self.memory.setdefault("history", [])
        history.append({
            "role": role,
            "content": content,
            "ts": datetime.now().isoformat(timespec="seconds"),
        })
        self._save_memory()

    def start_agent(self, name: str, script_path: str | Path,
                     args: Optional[list[str]] = None) -> AgentHandle:
        """Запустить агента как подпроцесс."""
        script_path = Path(script_path)
        cmd = [sys.executable, str(script_path)] + (args or [])

        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"

        self.logger.info("Запуск агента '%s': %s", name, " ".join(cmd))
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,  # построчная буферизация
            env=env,
        )
        handle = AgentHandle(name, process)
        self.agents[name] = handle
        if name not in self._agent_order:
            self._agent_order.append(name)
        return handle

    def stop_agent(self, name: str) -> None:
        if name in self.agents:
            self.logger.info("Остановка агента '%s'", name)
            self.agents[name].stop()

    def stop_all_agents(self) -> None:
        for name in list(self.agents.keys()):
            self.stop_agent(name)

    def handle_user_input(self, text: str, timeout: float = 15.0, parse_as_spec: bool = False) -> dict:
        """Принять ввод пользователя и передать его первому агенту в цепочке."""
        self.logger.info("Получен ввод пользователя: %s", text)
        self.append_history("user", text)

        if not self._agent_order:
            self.logger.warning("Нет зарегистрированных агентов")
            return {"error": "no_agents_registered"}

        first_agent_name = self._agent_order[0]
        handle = self.agents[first_agent_name]

        message = {
            "type": "user_input",
            "text": text,
            "memory_snapshot": self.memory.get("history", [])[-10:],
            "want_spec": parse_as_spec,
        }
        handle.send(message)

        response = handle.receive(timeout=timeout)
        if response is None:
            self.logger.error("Агент '%s' не ответил за %.1f сек", first_agent_name, timeout)
            return {"error": "agent_timeout", "agent": first_agent_name}

        self.append_history(first_agent_name, json.dumps(response, ensure_ascii=False))

        # Валидация структурированного ТЗ — только если её явно запросили
        # (например, через run_analyst_spec.py). В обычном диалоговом режиме
        # агент отвечает свободным текстом, без принудительного JSON.
        if parse_as_spec and "text" in response:
            self._handle_analyst_response(response)

        return response

    def _handle_analyst_response(self, response: dict) -> None:
        """Валидирует ответ агента-аналитика и сохраняет ТЗ в общую память."""
        raw_text = response.get("text", "")
        is_valid, spec, errors = validate_llm_response(raw_text, self.logs_dir)

        response["parsed_spec"] = spec
        response["validation_errors"] = errors

        if is_valid:
            self.remember("last_tech_spec", spec)
            self.logger.info("Аналитик подготовил валидное ТЗ: %s", spec.get("title"))
            print(
                f"✅ Аналитик подготовил ТЗ «{spec.get('title')}» "
                f"({len(spec.get('requirements', []))} требований, "
                f"{len(spec.get('modules', []))} модулей)"
            )
            self._trigger_developer(spec)
        else:
            self.logger.warning("Невалидный ответ агента-аналитика: %s", errors)
            print(f"❌ Ответ Аналитика не прошёл валидацию: {errors}")
            preview = raw_text.strip()[:500]
            print(f"   Сырой ответ модели (первые 500 символов):\n   {preview}")

    def _trigger_developer(self, spec: dict) -> None:
        """Запускает (если ещё не запущен) агента-разработчика и передаёт
        ему готовое ТЗ для генерации кода."""
        agent_name = "developer"
        developer_script = self.workspace_dir.parent / "agents" / "developer.py"

        if agent_name not in self.agents or not self.agents[agent_name].is_alive():
            if not developer_script.exists():
                self.logger.warning("Скрипт агента-разработчика не найден: %s", developer_script)
                print(f"⚠️ Не найден агент-разработчик: {developer_script}")
                return
            self.start_agent(agent_name, developer_script)
            time.sleep(0.3)  # даём подпроцессу подняться

        print(f"🛠 Запускаю агента-разработчика для «{spec.get('title')}»...")
        handle = self.agents[agent_name]
        handle.send({"type": "build_request", "spec": spec})

        # Таймаут зависит от количества файлов: на медленном железе без
        # GPU каждый файл может генерироваться по 1-2 минуты, поэтому
        # фиксированных 300 секунд может не хватить уже на 3-4 файла.
        files_count = max(1, len(spec.get("files", [])))
        dev_timeout = max(300.0, files_count * 150.0)
        print(f"   (жду ответа разработчика до {int(dev_timeout)} сек на {files_count} файлов)")

        dev_response = handle.receive(timeout=dev_timeout)
        if dev_response is None:
            self.logger.error("Агент-разработчик не ответил за отведённое время")
            print("❌ Агент-разработчик не ответил за отведённое время")
            self._report_partial_project(spec)
            return

        self.remember("last_generated_project", dev_response)

        files_written = dev_response.get("files_written", [])
        errors = dev_response.get("errors", [])
        project_dir = dev_response.get("project_dir", "")

        self.logger.info(
            "Разработчик завершил работу: %s файлов записано, %s ошибок",
            len(files_written), len(errors),
        )

        if files_written:
            print(f"✅ Сгенерировано {len(files_written)} файлов в workspace/projects/{project_dir}/")
            for f in files_written:
                print(f"   • {f}")
        if errors:
            print(f"⚠️ Ошибки при генерации некоторых файлов: {errors}")
        if not files_written and not errors:
            print(f"⚠️ Разработчик не вернул результат: {dev_response}")

    def _report_partial_project(self, spec: dict) -> None:
        try:
            from agents.developer import slugify
            project_dir_guess = slugify(spec.get("title", "project"))
            project_path = self.workspace_dir / "projects" / project_dir_guess
            if project_path.exists():
                existing_files = sorted(
                    str(p.relative_to(project_path))
                    for p in project_path.rglob("*") if p.is_file()
                )
                if existing_files:
                    print(f"   ⏳ Разработчик может ещё работать в фоне. Уже записано: {existing_files}")
                    print(f"   Проверь папку через минуту-две: {project_path}")
        except Exception:
            pass

    def query_ollama(
        self,
        prompt: str,
        model: str = "qwen2.5-coder:3b",
        system: Optional[str] = None,
        stream: bool = False,
        timeout: float = 60.0,
    ) -> str:
        """Отправить запрос модели в Ollama (эндпоинт /api/generate)."""
        url = f"{self.ollama_host}/api/generate"
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
        }
        if system:
            payload["system"] = system

        self.logger.debug("Запрос к Ollama: model=%s, prompt[:80]=%r", model, prompt[:80])

        try:
            resp = requests.post(url, json=payload, timeout=timeout)
            resp.raise_for_status()
        except requests.exceptions.ConnectionError as exc:
            self.logger.error("Ollama недоступна по адресу %s: %s", self.ollama_host, exc)
            raise RuntimeError(
                f"Не удалось подключиться к Ollama на {self.ollama_host}. "
                "Проверь, что Ollama установлена и запущена (команда `ollama serve`)."
            ) from exc

        if stream:
            # При stream=True Ollama возвращает поток JSON-строк по одной
            # на каждый токен — склеиваем их в один текст.
            full_text = ""
            for line in resp.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                full_text += chunk.get("response", "")
                if chunk.get("done"):
                    break
            return full_text

        data = resp.json()
        return data.get("response", "")

    def list_ollama_models(self) -> list[str]:
        """Получить список локально установленных в Ollama моделей."""
        url = f"{self.ollama_host}/api/tags"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return [m["name"] for m in data.get("models", [])]

    def shutdown(self) -> None:
        self.logger.info("Завершение работы оркестратора...")
        self.stop_all_agents()
        self._save_memory()
        self.logger.info("Оркестратор остановлен.")
