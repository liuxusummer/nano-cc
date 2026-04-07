from __future__ import annotations

import fcntl
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping
from uuid import uuid4

from .models import TodoValidationError


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_iso() -> str:
    return _utc_now().isoformat()


def _normalize_token(value: object, *, field_name: str, error_code: str | None = None) -> str:
    token = str(value or "").strip()
    if not token:
        raise TodoValidationError(error_code or f"invalid_{field_name}", f"{field_name} is required")
    if "/" in token or "\\" in token or ".." in token:
        raise TodoValidationError(
            error_code or f"invalid_{field_name}",
            f"{field_name} contains invalid characters",
            {field_name: token},
        )
    return token


def _parse_dt(raw: object, *, field_name: str, default: datetime) -> datetime:
    if isinstance(raw, datetime):
        return raw
    if isinstance(raw, (str, bytes)) and str(raw).strip():
        try:
            return datetime.fromisoformat(str(raw))
        except ValueError as exc:
            raise TodoValidationError(
                f"invalid_{field_name}",
                f"invalid {field_name}: {raw}",
                {"value": str(raw)},
            ) from exc
    return default


def _ensure_unique(values: Iterable[str], *, field_name: str) -> None:
    seen: set[str] = set()
    dupes: set[str] = set()
    for value in values:
        if value in seen:
            dupes.add(value)
        seen.add(value)
    if dupes:
        raise TodoValidationError(
            f"duplicate_{field_name}",
            f"{field_name} must be unique",
            {"values": sorted(dupes)},
        )


@dataclass(slots=True)
class WorktreeBinding:
    id: str
    name: str
    task_id: str
    path: str
    base_ref: str = ""
    status: str = "active"
    created_at: datetime = field(default_factory=_utc_now)
    updated_at: datetime = field(default_factory=_utc_now)
    last_event_id: str | None = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> WorktreeBinding:
        binding_id = _normalize_token(data.get("id", ""), field_name="worktree_id", error_code="invalid_worktree_id")
        name = _normalize_token(data.get("name", ""), field_name="worktree_name", error_code="invalid_worktree_name")
        task_id = _normalize_token(
            data.get("task_id", data.get("taskId", "")),
            field_name="task_id",
            error_code="invalid_task_id",
        )
        path = str(data.get("path", "")).strip()
        if not path:
            raise TodoValidationError(
                "invalid_worktree_path",
                "worktree path is required",
                {"name": name, "task_id": task_id},
            )
        base_ref = str(data.get("base_ref", data.get("baseRef", "")) or "").strip()
        status = str(data.get("status", "active") or "active").strip()
        if status not in {"active", "kept", "creating", "removed", "broken"}:
            raise TodoValidationError(
                "invalid_worktree_status",
                "status must be active, kept, creating, removed, or broken",
                {"status": status, "name": name},
            )

        now = _utc_now()
        created_at = _parse_dt(data.get("created_at"), field_name="created_at", default=now)
        updated_at = _parse_dt(data.get("updated_at"), field_name="updated_at", default=created_at)
        last_event_id_raw = data.get("last_event_id", data.get("lastEventId"))
        last_event_id = str(last_event_id_raw).strip() if last_event_id_raw is not None else None

        return cls(
            id=binding_id,
            name=name,
            task_id=task_id,
            path=path,
            base_ref=base_ref,
            status=status,
            created_at=created_at,
            updated_at=updated_at,
            last_event_id=last_event_id or None,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "task_id": self.task_id,
            "path": self.path,
            "base_ref": self.base_ref,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "last_event_id": self.last_event_id,
        }


@dataclass(frozen=True, slots=True)
class WorktreeEventRef:
    entry: str

    def to_dict(self) -> dict[str, str]:
        return {"entry": self.entry}


def _derive_task_index(by_name: dict[str, WorktreeBinding]) -> dict[str, str]:
    active: dict[str, str] = {}
    for name, binding in by_name.items():
        if binding.status not in {"active", "kept", "creating"}:
            continue
        existing = active.get(binding.task_id)
        if existing is not None and existing != name:
            raise TodoValidationError(
                "duplicate_task_binding",
                "task_id is bound to multiple active worktrees",
                {"task_id": binding.task_id, "names": sorted({existing, name})},
            )
        active[binding.task_id] = name
    return dict(active)


@dataclass(slots=True)
class WorktreeIndex:
    by_name: dict[str, WorktreeBinding] = field(default_factory=dict)
    by_task: dict[str, str] = field(default_factory=dict)
    version: int = 1

    @classmethod
    def empty(cls) -> "WorktreeIndex":
        return cls(by_name={}, by_task={}, version=1)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "WorktreeIndex":
        version_raw = data.get("version", 1)
        try:
            version = int(version_raw)
        except (TypeError, ValueError) as exc:
            raise TodoValidationError("invalid_worktree_index", "version must be an integer") from exc
        if version != 1:
            raise TodoValidationError("unsupported_worktree_index", "unsupported index version", {"version": version})

        raw_by_name = data.get("by_name", data.get("byName", {}))
        if raw_by_name is None:
            raw_by_name = {}
        if not isinstance(raw_by_name, dict):
            raise TodoValidationError("invalid_worktree_index", "by_name must be an object")

        by_name: dict[str, WorktreeBinding] = {}
        for key, value in raw_by_name.items():
            if not isinstance(value, dict):
                raise TodoValidationError(
                    "invalid_worktree_record",
                    "worktree record must be an object",
                    {"name": str(key)},
                )
            binding = WorktreeBinding.from_mapping(value)
            if str(key) != binding.name:
                raise TodoValidationError(
                    "invalid_worktree_record",
                    "worktree name mismatch with key",
                    {"key": str(key), "name": binding.name},
                )
            by_name[binding.name] = binding

        _ensure_unique([binding.id for binding in by_name.values()], field_name="worktree_id")
        by_task = _derive_task_index(by_name)
        return cls(by_name=by_name, by_task=by_task, version=1)

    def refresh(self) -> None:
        self.by_task = _derive_task_index(self.by_name)

    def to_dict(self) -> dict[str, Any]:
        by_name_payload = {name: binding.to_dict() for name, binding in sorted(self.by_name.items(), key=lambda kv: kv[0])}
        by_task_payload = dict(sorted(self.by_task.items(), key=lambda kv: kv[0]))
        return {"version": self.version, "by_name": by_name_payload, "by_task": by_task_payload}


class WorktreeStore:
    def __init__(self, root_dir: str | Path | None = None) -> None:
        self._root_dir = Path(root_dir) if root_dir is not None else Path(".worktrees")
        self._lock_path = self._root_dir / ".lock"

    @property
    def root_dir(self) -> Path:
        return self._root_dir

    @property
    def index_path(self) -> Path:
        return self._root_dir / "index.json"

    @property
    def events_path(self) -> Path:
        return self._root_dir / "events.jsonl"

    def _lock_exclusive(self):
        self._lock_path.parent.mkdir(parents=True, exist_ok=True)
        f = self._lock_path.open("a+")
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        return f

    def _lock_shared(self):
        self._lock_path.parent.mkdir(parents=True, exist_ok=True)
        f = self._lock_path.open("a+")
        fcntl.flock(f.fileno(), fcntl.LOCK_SH)
        return f

    def _unlock(self, f):
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        finally:
            f.close()

    def new_worktree_id(self) -> str:
        millis = int(time.time() * 1000)
        return f"{millis}_{uuid4().hex[:8]}"

    def load_index(self) -> WorktreeIndex:
        lock = self._lock_shared()
        try:
            return self._load_index_unlocked()
        finally:
            self._unlock(lock)

    def write_index(self, items: Iterable[WorktreeBinding | dict[str, Any]], merge: bool) -> WorktreeIndex:
        self.append_event("worktree.index.write.before", {"merge": bool(merge)})
        lock = self._lock_exclusive()
        try:
            normalized = self._normalize_items(items)
            current = self._load_index_unlocked()
            by_name = dict(current.by_name) if merge else {}
            for binding in normalized:
                by_name[binding.name] = binding
            next_index = WorktreeIndex(by_name=by_name, by_task={}, version=1)
            next_index.refresh()
            if self._index_equals(current, next_index):
                self.append_event("worktree.index.write.keep", {})
                return current
            self._persist_index_unlocked(next_index)
            self.append_event("worktree.index.write.after", {})
            return next_index
        except TodoValidationError as exc:
            self.append_event("worktree.index.write.failed", {"error": exc.to_dict()})
            raise
        except Exception as exc:
            self.append_event("worktree.index.write.failed", {"error": str(exc)})
            raise
        finally:
            self._unlock(lock)

    def upsert(self, item: WorktreeBinding | dict[str, Any]) -> WorktreeBinding:
        self.append_event("worktree.index.upsert.before", {})
        lock = self._lock_exclusive()
        try:
            binding = item if isinstance(item, WorktreeBinding) else WorktreeBinding.from_mapping(item)
            current = self._load_index_unlocked()
            existing = current.by_name.get(binding.name)
            if existing is not None and existing.to_dict() == binding.to_dict():
                self.append_event("worktree.index.upsert.keep", {"name": binding.name})
                return existing
            next_by_name = dict(current.by_name)
            next_by_name[binding.name] = binding
            next_index = WorktreeIndex(by_name=next_by_name, by_task={}, version=1)
            next_index.refresh()
            self._persist_index_unlocked(next_index)
            self.append_event("worktree.index.upsert.after", {"name": binding.name})
            return binding
        except TodoValidationError as exc:
            self.append_event("worktree.index.upsert.failed", {"error": exc.to_dict()})
            raise
        except Exception as exc:
            self.append_event("worktree.index.upsert.failed", {"error": str(exc)})
            raise
        finally:
            self._unlock(lock)

    def reserve_creating(
        self,
        *,
        task_id: str,
        name: str,
        path: str,
        base_ref: str,
    ) -> WorktreeBinding:
        normalized_name = _normalize_token(name, field_name="worktree_name", error_code="invalid_worktree_name")
        normalized_task = _normalize_token(task_id, field_name="task_id", error_code="invalid_task_id")
        normalized_path = str(path).strip()
        if not normalized_path:
            raise TodoValidationError(
                "invalid_worktree_path",
                "worktree path is required",
                {"name": normalized_name, "task_id": normalized_task},
            )
        normalized_base_ref = str(base_ref or "").strip()

        self.append_event(
            "worktree.index.reserve.before",
            {"name": normalized_name, "task_id": normalized_task, "path": normalized_path},
        )
        lock = self._lock_exclusive()
        try:
            index = self._load_index_unlocked()
            existing = index.by_name.get(normalized_name)
            if existing is not None and existing.status in {"active", "kept", "creating"}:
                raise TodoValidationError(
                    "worktree_exists",
                    "worktree already exists",
                    {"name": normalized_name, "status": existing.status},
                )
            bound_name = index.by_task.get(normalized_task)
            if bound_name:
                raise TodoValidationError(
                    "task_already_bound",
                    "task already has an active worktree",
                    {"task_id": normalized_task, "name": bound_name},
                )

            binding = WorktreeBinding(
                id=self.new_worktree_id(),
                name=normalized_name,
                task_id=normalized_task,
                path=normalized_path,
                base_ref=normalized_base_ref,
                status="creating",
            )
            next_by_name = dict(index.by_name)
            next_by_name[normalized_name] = binding
            next_index = WorktreeIndex(by_name=next_by_name, by_task={}, version=1)
            next_index.refresh()
            self._persist_index_unlocked(next_index)
            self.append_event("worktree.index.reserve.after", {"name": normalized_name, "task_id": normalized_task})
            return binding
        except TodoValidationError as exc:
            self.append_event("worktree.index.reserve.failed", {"error": exc.to_dict()})
            raise
        except Exception as exc:
            self.append_event("worktree.index.reserve.failed", {"error": str(exc)})
            raise
        finally:
            self._unlock(lock)

    def release_creating(self, *, name: str) -> WorktreeBinding | None:
        normalized = _normalize_token(name, field_name="worktree_name", error_code="invalid_worktree_name")
        self.append_event("worktree.index.release.before", {"name": normalized})
        lock = self._lock_exclusive()
        try:
            index = self._load_index_unlocked()
            existing = index.by_name.get(normalized)
            if existing is None:
                self.append_event("worktree.index.release.failed", {"name": normalized, "error": "not found"})
                return None
            if existing.status != "creating":
                self.append_event("worktree.index.release.keep", {"name": normalized, "status": existing.status})
                return existing
            updated_payload = {**existing.to_dict(), "status": "removed", "updated_at": _utc_iso()}
            updated = WorktreeBinding.from_mapping(updated_payload)
            next_by_name = dict(index.by_name)
            next_by_name[normalized] = updated
            next_index = WorktreeIndex(by_name=next_by_name, by_task={}, version=1)
            next_index.refresh()
            self._persist_index_unlocked(next_index)
            self.append_event("worktree.index.release.after", {"name": normalized})
            return updated
        except TodoValidationError as exc:
            self.append_event("worktree.index.release.failed", {"name": normalized, "error": exc.to_dict()})
            raise
        except Exception as exc:
            self.append_event("worktree.index.release.failed", {"name": normalized, "error": str(exc)})
            raise
        finally:
            self._unlock(lock)

    def mark_removed(self, name: str) -> WorktreeBinding | None:
        normalized = _normalize_token(name, field_name="worktree_name", error_code="invalid_worktree_name")
        self.append_event("worktree.index.remove.before", {"name": normalized})
        lock = self._lock_exclusive()
        try:
            index = self._load_index_unlocked()
            existing = index.by_name.get(normalized)
            if existing is None:
                self.append_event("worktree.index.remove.failed", {"name": normalized, "error": "not found"})
                return None
            if existing.status == "removed":
                self.append_event("worktree.index.remove.keep", {"name": normalized})
                return existing
            updated_payload = {**existing.to_dict(), "status": "removed", "updated_at": _utc_iso()}
            updated = WorktreeBinding.from_mapping(updated_payload)
            next_by_name = dict(index.by_name)
            next_by_name[normalized] = updated
            next_index = WorktreeIndex(by_name=next_by_name, by_task={}, version=1)
            next_index.refresh()
            self._persist_index_unlocked(next_index)
            self.append_event("worktree.index.remove.after", {"name": normalized})
            return updated
        except TodoValidationError as exc:
            self.append_event("worktree.index.remove.failed", {"name": normalized, "error": exc.to_dict()})
            raise
        except Exception as exc:
            self.append_event("worktree.index.remove.failed", {"name": normalized, "error": str(exc)})
            raise
        finally:
            self._unlock(lock)

    def append_event(self, event_type: str, payload: Any) -> WorktreeEventRef:
        path = self.events_path
        path.parent.mkdir(parents=True, exist_ok=True)
        entry_id = uuid4().hex
        record: dict[str, Any] = {"ts": _utc_iso(), "type": str(event_type), "id": entry_id}
        if isinstance(payload, dict):
            record.update(payload)
        else:
            record["details"] = payload
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return WorktreeEventRef(entry=entry_id)

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

    def _normalize_items(self, items: Iterable[WorktreeBinding | dict[str, Any]]) -> list[WorktreeBinding]:
        if isinstance(items, (str, bytes)) or not isinstance(items, Iterable):
            raise TodoValidationError("invalid_worktree_list", "items must be an array")
        normalized: list[WorktreeBinding] = []
        for item in items:
            if isinstance(item, WorktreeBinding):
                normalized.append(item)
                continue
            if isinstance(item, dict):
                normalized.append(WorktreeBinding.from_mapping(item))
                continue
            raise TodoValidationError("invalid_worktree_item", "item must be an object")
        return normalized

    def _index_equals(self, left: WorktreeIndex, right: WorktreeIndex) -> bool:
        if left.version != right.version:
            return False
        if left.by_name.keys() != right.by_name.keys():
            return False
        if left.by_task != right.by_task:
            return False
        for key, binding in left.by_name.items():
            other = right.by_name.get(key)
            if other is None or binding.to_dict() != other.to_dict():
                return False
        return True

    def _load_index_unlocked(self) -> WorktreeIndex:
        path = self.index_path
        if not path.exists():
            return WorktreeIndex.empty()
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise TodoValidationError(
                "invalid_worktree_index", "index.json contains invalid json", {"path": str(path), "error": str(exc)}
            ) from exc
        if not isinstance(raw, dict):
            raise TodoValidationError(
                "invalid_worktree_index", "index.json must contain a json object", {"path": str(path)}
            )
        return WorktreeIndex.from_mapping(raw)

    def _persist_index(self, index: WorktreeIndex) -> None:
        lock = self._lock_exclusive()
        try:
            self._persist_index_unlocked(index)
        finally:
            self._unlock(lock)

    def _persist_index_unlocked(self, index: WorktreeIndex) -> None:
        path = self.index_path
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = index.to_dict()
        tmp_path = path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tmp_path.replace(path)
