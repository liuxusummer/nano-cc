from __future__ import annotations

import shlex
import subprocess
import threading
import uuid
from pathlib import Path
from queue import Empty, SimpleQueue
from typing import Any

from .models import BackgroundTaskNotification, BackgroundTaskRecord, BackgroundTaskStatus, utc_now


class BackgroundTaskManager:
    def __init__(self, *, root_dir: Path | None = None) -> None:
        self._records: dict[str, BackgroundTaskRecord] = {}
        self._processes: dict[str, subprocess.Popen[Any]] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._notifications: SimpleQueue[BackgroundTaskNotification] = SimpleQueue()
        self._lock = threading.Lock()
        self._root_dir = root_dir or Path(".trae/background_tasks")
        self._stop_event = threading.Event()
        self._runner: threading.Thread | None = None

    def record(self, record: BackgroundTaskRecord) -> None:
        with self._lock:
            self._records[record.task_id] = record

    def list_records(self) -> list[BackgroundTaskRecord]:
        with self._lock:
            return list(self._records.values())

    def run(self) -> None:
        with self._lock:
            if self._runner is not None:
                return
            self._root_dir.mkdir(parents=True, exist_ok=True)
            self._runner = threading.Thread(target=self._run_loop, daemon=True)
            self._runner.start()

    def flush_notifications(self) -> list[BackgroundTaskNotification]:
        drained: list[BackgroundTaskNotification] = []
        while True:
            try:
                drained.append(self._notifications.get(block=False))
            except Empty:
                break
        return drained

    def submit(self, argv: list[str], *, cwd: str | None = None, timeout_seconds: float | None = None) -> str:
        normalized_argv = _normalize_argv(argv)
        self.run()
        task_id = uuid.uuid4().hex
        record = BackgroundTaskRecord(
            task_id=task_id,
            status=BackgroundTaskStatus.PENDING.value,
            command=shlex.join(normalized_argv),
        )
        self.record(record)
        thread = threading.Thread(
            target=self._run_task,
            args=(task_id, normalized_argv, cwd, timeout_seconds),
            daemon=True,
        )
        with self._lock:
            self._threads[task_id] = thread
        thread.start()
        return task_id

    def status(self, task_id: str) -> BackgroundTaskRecord | None:
        with self._lock:
            record = self._records.get(task_id)
            return record

    def cancel(self, task_id: str) -> bool:
        with self._lock:
            record = self._records.get(task_id)
            process = self._processes.get(task_id)
            if record is None:
                return False
            if record.status in {
                BackgroundTaskStatus.COMPLETED.value,
                BackgroundTaskStatus.FAILED.value,
                BackgroundTaskStatus.CANCELLED.value,
            }:
                return True
            record.status = BackgroundTaskStatus.CANCELLED.value
            record.finished_at = utc_now()
            self._notifications.put(
                BackgroundTaskNotification(
                    task_id=record.task_id,
                    status=record.status,
                    finished_at=record.finished_at,
                )
            )
        try:
            if process is not None:
                process.terminate()
                process.wait(timeout=2)
        except Exception:
            try:
                if process is not None:
                    process.kill()
            except Exception:
                pass
        with self._lock:
            record = self._records.get(task_id)
            if record is None:
                return False
            record.status = BackgroundTaskStatus.CANCELLED.value
            record.finished_at = utc_now()
            record.exit_code = process.returncode if process is not None else record.exit_code
            self._processes.pop(task_id, None)
        return True

    def _run_loop(self) -> None:
        self._stop_event.wait()

    def _run_task(
        self,
        task_id: str,
        argv: list[str],
        cwd: str | None,
        timeout_seconds: float | None,
    ) -> None:
        task_dir = self._root_dir / task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = task_dir / "stdout.log"
        stderr_path = task_dir / "stderr.log"

        with self._lock:
            record = self._records.get(task_id)
            if record is None:
                return
            if record.status == BackgroundTaskStatus.CANCELLED.value:
                record.finished_at = record.finished_at or utc_now()
                self._notifications.put(
                    BackgroundTaskNotification(
                        task_id=record.task_id,
                        status=record.status,
                        finished_at=record.finished_at,
                        exit_code=record.exit_code,
                    )
                )
                return
            record.status = BackgroundTaskStatus.RUNNING.value
            record.started_at = utc_now()

        exit_code: int | None = None
        final_status = BackgroundTaskStatus.FAILED.value
        try:
            with stdout_path.open("w", encoding="utf-8") as stdout_f, stderr_path.open("w", encoding="utf-8") as stderr_f:
                process = subprocess.Popen(
                    argv,
                    cwd=cwd,
                    stdout=stdout_f,
                    stderr=stderr_f,
                    text=True,
                )
                with self._lock:
                    self._processes[task_id] = process
                try:
                    process.wait(timeout=timeout_seconds)
                except subprocess.TimeoutExpired:
                    try:
                        process.terminate()
                        process.wait(timeout=2)
                    except Exception:
                        try:
                            process.kill()
                        except Exception:
                            pass
                exit_code = process.returncode
                if exit_code == 0:
                    final_status = BackgroundTaskStatus.COMPLETED.value
        except Exception:
            final_status = BackgroundTaskStatus.FAILED.value

        output_tail = _read_tail(stdout_path)
        error_tail = _read_tail(stderr_path)

        with self._lock:
            record = self._records.get(task_id)
            if record is None:
                return
            if record.status != BackgroundTaskStatus.CANCELLED.value:
                record.status = final_status
            record.finished_at = utc_now()
            record.exit_code = exit_code
            record.output_tail = output_tail
            record.error_tail = error_tail
            self._processes.pop(task_id, None)

            self._notifications.put(
                BackgroundTaskNotification(
                    task_id=record.task_id,
                    status=record.status,
                    finished_at=record.finished_at,
                    exit_code=record.exit_code,
                    output_tail=record.output_tail,
                    error_tail=record.error_tail,
                )
            )


def _read_tail(path: Path, max_bytes: int = 8192) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    data = path.read_bytes()
    if len(data) > max_bytes:
        data = data[-max_bytes:]
    text = data.decode("utf-8", errors="replace")
    return text.strip() or None


def _normalize_argv(argv: list[str]) -> list[str]:
    if not isinstance(argv, list) or not argv:
        raise ValueError("argv is required")
    normalized: list[str] = []
    for item in argv:
        if not isinstance(item, str):
            raise ValueError("argv items must be strings")
        value = item.strip()
        if not value:
            raise ValueError("argv items must be non-empty strings")
        normalized.append(value)
    return normalized


BackgroundManager = BackgroundTaskManager
