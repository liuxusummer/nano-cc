from __future__ import annotations

import fcntl
import json
from datetime import datetime, timezone
from collections.abc import Iterable
from pathlib import Path
from typing import Any
from uuid import uuid4

from .models import TaskGraphItem, TaskStatus, TodoValidationError


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class TaskGraphStore:
    def __init__(self, root_dir: str | Path | None = None) -> None:
        self._root_dir = Path(root_dir) if root_dir is not None else Path(".agent/task_graph")

    @property
    def root_dir(self) -> Path:
        return self._root_dir

    @property
    def lock_path(self) -> Path:
        return self._root_dir / ".lock"

    @property
    def events_path(self) -> Path:
        return self._root_dir / "events.jsonl"

    def _lock_exclusive(self):
        self._root_dir.mkdir(parents=True, exist_ok=True)
        f = self.lock_path.open("a+")
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        return f

    def _lock_shared(self):
        self._root_dir.mkdir(parents=True, exist_ok=True)
        f = self.lock_path.open("a+")
        fcntl.flock(f.fileno(), fcntl.LOCK_SH)
        return f

    def _unlock(self, f) -> None:
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        finally:
            f.close()

    def _append_event(self, event_type: str, payload: dict[str, Any]) -> None:
        self._root_dir.mkdir(parents=True, exist_ok=True)
        record = {"ts": _utc_iso(), "type": str(event_type), "payload": payload, "id": uuid4().hex}
        with self.events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def has_valid_graph(self) -> bool:
        if not self._root_dir.exists():
            return False
        try:
            graph = self.load_graph()
            return bool(graph)
        except TodoValidationError:
            return False

    def load_graph(self) -> dict[str, TaskGraphItem]:
        if not self._root_dir.exists():
            return {}
        lock = self._lock_shared()
        try:
            graph = self._load_graph_unlocked()
            self.validate_graph(graph)
            return graph
        finally:
            self._unlock(lock)

    def write_graph(self, tasks: Iterable[TaskGraphItem | dict[str, Any]], merge: bool) -> dict[str, TaskGraphItem]:
        normalized = self._normalize_tasks(tasks)
        self._append_event("before", {"operation": "write_graph", "merge": bool(merge), "count": len(normalized)})
        lock = self._lock_exclusive()
        try:
            input_ids: set[str] = set()
            duplicates: list[str] = []
            for task in normalized:
                if task.id in input_ids:
                    duplicates.append(task.id)
                input_ids.add(task.id)
            if duplicates:
                raise TodoValidationError(
                    "duplicate_task_id",
                    "task id must be unique",
                    {"ids": sorted(set(duplicates))},
                )

            existing = self._load_graph_unlocked() if merge else {}
            if merge:
                combined = dict(existing)
                for task in normalized:
                    combined[task.id] = self._merge_runtime_fields(existing.get(task.id), task)
            else:
                combined = {task.id: task for task in normalized}

            self._normalize_mirrored_dependencies(combined)

            if self._graph_equals(existing if merge else self._load_graph_unlocked(), combined):
                self._append_event("keep", {"operation": "write_graph"})
                return existing if merge else self._load_graph_unlocked()

            self.validate_graph(combined)
            self._persist_graph(combined, replace=not merge)
            self._append_event("after", {"operation": "write_graph", "count": len(combined)})
            return combined
        except TodoValidationError as exc:
            self._append_event("failed", {"operation": "write_graph", "error": exc.to_dict()})
            raise
        except Exception as exc:
            self._append_event("failed", {"operation": "write_graph", "error": str(exc)})
            raise
        finally:
            self._unlock(lock)

    def query(self, mode: str) -> list[TaskGraphItem]:
        graph = self.load_graph()
        lowered = str(mode).strip().lower()
        if lowered not in {"ready", "blocked", "completed", "in_progress"}:
            raise TodoValidationError(
                "invalid_query_mode",
                "mode must be ready, blocked, in_progress, or completed",
            )

        tasks = sorted(graph.values(), key=lambda item: item.id)
        if lowered == "completed":
            return [task for task in tasks if task.status == TaskStatus.COMPLETED]
        if lowered == "in_progress":
            return [task for task in tasks if task.status == TaskStatus.IN_PROGRESS]
        if lowered == "ready":
            return [
                task
                for task in tasks
                if task.status == TaskStatus.PENDING and not task.blockedBy
            ]
        return [
            task
            for task in tasks
            if task.status == TaskStatus.PENDING and bool(task.blockedBy)
        ]

    def query_ready_unowned(self) -> list[TaskGraphItem]:
        graph = self.load_graph()
        tasks = sorted(graph.values(), key=lambda item: item.id)
        return [
            task
            for task in tasks
            if task.status == TaskStatus.PENDING and not task.blockedBy and (task.owner is None)
        ]

    def query_by_status(self, status: str) -> list[TaskGraphItem]:
        normalized_status = str(status).strip().lower()
        status_enum = TaskStatus(normalized_status)
        graph = self.load_graph()
        tasks = sorted(graph.values(), key=lambda item: item.id)
        return [task for task in tasks if task.status == status_enum]

    def create_task(
        self,
        content: str,
        *,
        task_id: str | None = None,
        status: TaskStatus = TaskStatus.PENDING,
        blocked_by: list[str] | None = None,
        blocks: list[str] | None = None,
        owner: str | None = None,
    ) -> TaskGraphItem:
        goal = str(content).strip()
        if not goal:
            raise TodoValidationError("invalid_task_content", "task content is required")

        normalized_id = str(task_id).strip() if task_id is not None else uuid4().hex
        if not normalized_id:
            raise TodoValidationError("invalid_task_id", "task id is required")

        self._append_event("before", {"operation": "create_task", "id": normalized_id})
        lock = self._lock_exclusive()
        try:
            graph = self._load_graph_unlocked()
            if normalized_id in graph:
                raise TodoValidationError(
                    "duplicate_task_id",
                    "task id must be unique",
                    {"id": normalized_id},
                )

            new_task = TaskGraphItem(
                id=normalized_id,
                content=goal,
                status=status,
                blockedBy=list(blocked_by or []),
                blocks=list(blocks or []),
                owner=(str(owner).strip() or None) if owner is not None else None,
            )

            updated: dict[str, TaskGraphItem] = {new_task.id: new_task}
            for dep_id in list(new_task.blockedBy):
                dep = graph.get(dep_id)
                if dep is None:
                    raise TodoValidationError(
                        "invalid_task_reference",
                        "task dependency references missing task",
                        {"id": new_task.id, "missing": dep_id},
                    )
                if new_task.id not in dep.blocks:
                    dep.blocks = [*dep.blocks, new_task.id]
                    updated[dep.id] = dep

            for blocked_id in list(new_task.blocks):
                target = graph.get(blocked_id)
                if target is None:
                    raise TodoValidationError(
                        "invalid_task_reference",
                        "task dependency references missing task",
                        {"id": new_task.id, "missing": blocked_id},
                    )
                if new_task.id not in target.blockedBy:
                    target.blockedBy = [*target.blockedBy, new_task.id]
                    updated[target.id] = target

            self.validate_graph(graph | updated)
            for item in updated.values():
                self._persist_task(item)
            self._append_event("after", {"operation": "create_task", "id": normalized_id})
            return new_task
        except TodoValidationError as exc:
            self._append_event("failed", {"operation": "create_task", "error": exc.to_dict(), "id": normalized_id})
            raise
        except Exception as exc:
            self._append_event("failed", {"operation": "create_task", "error": str(exc), "id": normalized_id})
            raise
        finally:
            self._unlock(lock)

    def complete(self, task_id: str) -> dict[str, Any]:
        normalized_id = str(task_id).strip()
        if not normalized_id:
            raise TodoValidationError("invalid_task_id", "task id is required")

        self._append_event("before", {"operation": "complete", "id": normalized_id})
        lock = self._lock_exclusive()
        try:
            graph = self._load_graph_unlocked()
            try:
                task = graph[normalized_id]
            except KeyError as exc:
                raise TodoValidationError(
                    "missing_task",
                    "task not found",
                    {"id": normalized_id},
                ) from exc

            if task.status == TaskStatus.COMPLETED:
                self._append_event("keep", {"operation": "complete", "id": normalized_id})
                return {"task": task, "unlocked": []}

            updated: dict[str, TaskGraphItem] = {task.id: task}
            unlocked: list[str] = []
            for blocked_id in list(task.blocks):
                blocked_task = graph.get(blocked_id)
                if blocked_task is None:
                    raise TodoValidationError(
                        "invalid_task_reference",
                        "task dependency references missing task",
                        {"id": task.id, "missing": blocked_id},
                    )
                if task.id in blocked_task.blockedBy:
                    blocked_task.blockedBy = [d for d in blocked_task.blockedBy if d != task.id]
                    updated[blocked_task.id] = blocked_task
                task.blocks = [t for t in task.blocks if t != blocked_id]
                updated[task.id] = task
                if blocked_task.status == TaskStatus.PENDING and not blocked_task.blockedBy:
                    unlocked.append(blocked_task.id)

            task.status = TaskStatus.COMPLETED
            updated[task.id] = task
            self.validate_graph(graph | updated)
            for item in updated.values():
                self._persist_task(item)
            self._append_event("task.completed", {"id": normalized_id})
            self._append_event("after", {"operation": "complete", "id": normalized_id})
            return {"task": task, "unlocked": sorted(set(unlocked))}
        except TodoValidationError as exc:
            self._append_event("failed", {"operation": "complete", "error": exc.to_dict(), "id": normalized_id})
            raise
        except Exception as exc:
            self._append_event("failed", {"operation": "complete", "error": str(exc), "id": normalized_id})
            raise
        finally:
            self._unlock(lock)

    def claim_task(self, task_id: str, owner_id: str) -> TaskGraphItem:
        normalized_id = str(task_id).strip()
        if not normalized_id:
            raise TodoValidationError("invalid_task_id", "task id is required")
        owner_value = str(owner_id).strip()
        if not owner_value:
            raise TodoValidationError("invalid_owner_id", "owner id is required")

        self._append_event("before", {"operation": "claim_task", "id": normalized_id, "owner": owner_value})
        lock = self._lock_exclusive()
        try:
            graph = self._load_graph_unlocked()
            try:
                task = graph[normalized_id]
            except KeyError as exc:
                raise TodoValidationError(
                    "missing_task",
                    "task not found",
                    {"id": normalized_id},
                ) from exc

            if task.owner == owner_value:
                self._append_event("keep", {"operation": "claim_task", "id": normalized_id, "owner": owner_value})
                return task

            task.owner = owner_value
            self._persist_task(task)
            self._append_event("after", {"operation": "claim_task", "id": normalized_id, "owner": owner_value})
            return task
        except TodoValidationError as exc:
            self._append_event("failed", {"operation": "claim_task", "error": exc.to_dict(), "id": normalized_id})
            raise
        except Exception as exc:
            self._append_event("failed", {"operation": "claim_task", "error": str(exc), "id": normalized_id})
            raise
        finally:
            self._unlock(lock)

    def start_task(self, task_id: str) -> TaskGraphItem:
        normalized_id = str(task_id).strip()
        if not normalized_id:
            raise TodoValidationError("invalid_task_id", "task id is required")

        self._append_event("before", {"operation": "start_task", "id": normalized_id})
        lock = self._lock_exclusive()
        try:
            graph = self._load_graph_unlocked()
            try:
                task = graph[normalized_id]
            except KeyError as exc:
                raise TodoValidationError(
                    "missing_task",
                    "task not found",
                    {"id": normalized_id},
                ) from exc

            if task.status == TaskStatus.COMPLETED:
                raise TodoValidationError(
                    "task_already_completed",
                    "task already completed",
                    {"id": normalized_id},
                )

            if task.status == TaskStatus.IN_PROGRESS:
                self._append_event("keep", {"operation": "start_task", "id": normalized_id})
                return task

            task.status = TaskStatus.IN_PROGRESS
            self._persist_task(task)
            self._append_event("after", {"operation": "start_task", "id": normalized_id})
            return task
        except TodoValidationError as exc:
            self._append_event("failed", {"operation": "start_task", "error": exc.to_dict(), "id": normalized_id})
            raise
        except Exception as exc:
            self._append_event("failed", {"operation": "start_task", "error": str(exc), "id": normalized_id})
            raise
        finally:
            self._unlock(lock)

    def release_task(self, task_id: str) -> TaskGraphItem:
        normalized_id = str(task_id).strip()
        if not normalized_id:
            raise TodoValidationError("invalid_task_id", "task id is required")

        self._append_event("before", {"operation": "release_task", "id": normalized_id})
        lock = self._lock_exclusive()
        try:
            graph = self._load_graph_unlocked()
            try:
                task = graph[normalized_id]
            except KeyError as exc:
                raise TodoValidationError(
                    "missing_task",
                    "task not found",
                    {"id": normalized_id},
                ) from exc

            if task.owner is None:
                self._append_event("keep", {"operation": "release_task", "id": normalized_id})
                return task

            task.owner = None
            self._persist_task(task)
            self._append_event("after", {"operation": "release_task", "id": normalized_id})
            return task
        except TodoValidationError as exc:
            self._append_event("failed", {"operation": "release_task", "error": exc.to_dict(), "id": normalized_id})
            raise
        except Exception as exc:
            self._append_event("failed", {"operation": "release_task", "error": str(exc), "id": normalized_id})
            raise
        finally:
            self._unlock(lock)

    def validate_graph(self, graph: dict[str, TaskGraphItem]) -> None:
        for task_id, task in graph.items():
            if not task_id:
                raise TodoValidationError("invalid_task_id", "task id is required")
            if task.id != task_id:
                raise TodoValidationError(
                    "invalid_task_id",
                    "task id mismatch",
                    {"expected": task_id, "actual": task.id},
                )
            if task_id in task.blockedBy or task_id in task.blocks:
                raise TodoValidationError(
                    "self_loop",
                    "task cannot depend on itself",
                    {"id": task_id},
                )
            for ref in [*task.blockedBy, *task.blocks]:
                if ref not in graph:
                    raise TodoValidationError(
                        "invalid_task_reference",
                        "task dependency references missing task",
                        {"id": task_id, "missing": ref},
                    )

        missing_mirror: list[dict[str, Any]] = []
        for task in graph.values():
            for blocked_id in task.blocks:
                target = graph[blocked_id]
                if task.id not in target.blockedBy:
                    missing_mirror.append(
                        {"type": "blocks", "from": task.id, "to": blocked_id, "missing": "blockedBy"}
                    )
            for dep_id in task.blockedBy:
                target = graph[dep_id]
                if task.id not in target.blocks:
                    missing_mirror.append(
                        {"type": "blockedBy", "from": task.id, "to": dep_id, "missing": "blocks"}
                    )
        if missing_mirror:
            raise TodoValidationError(
                "mirror_mismatch",
                "blockedBy and blocks must mirror each other",
                {"missing": missing_mirror},
            )

    def _normalize_tasks(self, tasks: Iterable[TaskGraphItem | dict[str, Any]]) -> list[TaskGraphItem]:
        if isinstance(tasks, (str, bytes)) or not isinstance(tasks, Iterable):
            raise TodoValidationError("invalid_task_list", "tasks must be an array")
        normalized: list[TaskGraphItem] = []
        for task in tasks:
            if isinstance(task, TaskGraphItem):
                normalized.append(task)
                continue
            if isinstance(task, dict):
                normalized.append(TaskGraphItem.from_mapping(task))
                continue
            raise TodoValidationError("invalid_task_item", "task item must be an object")
        return normalized

    def _merge_runtime_fields(self, existing: TaskGraphItem | None, incoming: TaskGraphItem) -> TaskGraphItem:
        if existing is None:
            return incoming
        return TaskGraphItem(
            id=incoming.id,
            content=incoming.content,
            status=existing.status,
            blockedBy=list(incoming.blockedBy),
            blocks=list(incoming.blocks),
            owner=existing.owner,
        )

    def _normalize_mirrored_dependencies(self, graph: dict[str, TaskGraphItem]) -> None:
        for task in graph.values():
            for blocked_id in task.blocks:
                target = graph.get(blocked_id)
                if target is None:
                    continue
                if task.id not in target.blockedBy:
                    target.blockedBy = [*target.blockedBy, task.id]
            for dep_id in task.blockedBy:
                target = graph.get(dep_id)
                if target is None:
                    continue
                if task.id not in target.blocks:
                    target.blocks = [*target.blocks, task.id]

    def _graph_equals(self, left: dict[str, TaskGraphItem], right: dict[str, TaskGraphItem]) -> bool:
        if left.keys() != right.keys():
            return False
        for key in left:
            if left[key].to_dict() != right[key].to_dict():
                return False
        return True

    def _load_graph_unlocked(self) -> dict[str, TaskGraphItem]:
        if not self._root_dir.exists():
            return {}
        if not self._root_dir.is_dir():
            raise TodoValidationError(
                "invalid_task_graph_root",
                "task graph root is not a directory",
                {"path": str(self._root_dir)},
            )

        graph: dict[str, TaskGraphItem] = {}
        seen_sources: dict[str, str] = {}
        for path in sorted(self._root_dir.glob("*.json")):
            if not path.is_file():
                continue
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise TodoValidationError(
                    "invalid_task_file",
                    "task file contains invalid json",
                    {"path": str(path), "error": str(exc)},
                ) from exc
            if not isinstance(raw, dict):
                raise TodoValidationError(
                    "invalid_task_file",
                    "task file must contain a json object",
                    {"path": str(path)},
                )
            item = TaskGraphItem.from_mapping(raw)
            expected_stem = path.stem
            if expected_stem != item.id:
                raise TodoValidationError(
                    "invalid_task_file",
                    "task file name does not match task id",
                    {"path": str(path), "expected": item.id, "actual": expected_stem},
                )
            if item.id in graph:
                raise TodoValidationError(
                    "duplicate_task_id",
                    "task id must be unique",
                    {"id": item.id, "paths": [seen_sources[item.id], str(path)]},
                )
            graph[item.id] = item
            seen_sources[item.id] = str(path)
        return graph

    def read_events(self) -> list[dict[str, Any]]:
        path = self.events_path
        if not path.exists():
            return []
        entries: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            loaded = json.loads(line)
            if isinstance(loaded, dict):
                entries.append(loaded)
        return entries

    def _persist_graph(self, graph: dict[str, TaskGraphItem], replace: bool) -> None:
        self._root_dir.mkdir(parents=True, exist_ok=True)
        if replace:
            for path in self._root_dir.glob("*.json"):
                if path.is_file():
                    path.unlink()
        for item in graph.values():
            self._persist_task(item)

    def _persist_task(self, task: TaskGraphItem) -> None:
        path = self._task_path(task.id)
        payload = {
            "id": task.id,
            "content": task.content,
            "status": task.status.value,
            "blockedBy": list(task.blockedBy),
            "blocks": list(task.blocks),
            "owner": task.owner if task.owner is not None else None,
        }
        tmp_path = path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tmp_path.replace(path)

    def _task_path(self, task_id: str) -> Path:
        normalized = str(task_id).strip()
        if not normalized:
            raise TodoValidationError("invalid_task_id", "task id is required")
        if "/" in normalized or "\\" in normalized or ".." in normalized:
            raise TodoValidationError("invalid_task_id", "task id contains invalid characters", {"id": normalized})
        return self._root_dir / f"{normalized}.json"

TaskSystem = TaskGraphStore
