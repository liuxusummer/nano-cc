from __future__ import annotations

import shlex
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .git_worktree import GitWorktree
from .models import TodoValidationError
from .task_system import TaskGraphStore
from .worktree_store import WorktreeBinding, WorktreeStore


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class WorktreeExecResult:
    returncode: int
    stdout: str
    stderr: str
    cwd: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "returncode": self.returncode,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "cwd": self.cwd,
        }


class WorktreeManager:
    def __init__(
        self,
        *,
        repo_dir: str | Path | None = None,
        store: WorktreeStore | None = None,
        task_store: TaskGraphStore | None = None,
        git: GitWorktree | None = None,
    ) -> None:
        self._repo_dir = Path(repo_dir) if repo_dir is not None else Path(".")
        self._store = store or WorktreeStore()
        self._task_store = task_store or TaskGraphStore()
        self._git = git or GitWorktree()

    @property
    def store(self) -> WorktreeStore:
        return self._store

    @property
    def repo_dir(self) -> Path:
        return self._repo_dir

    def _trees_dir(self) -> Path:
        return self._store.root_dir / "trees"

    def _resolve_by_name(self, name: str) -> WorktreeBinding:
        normalized = str(name).strip()
        if not normalized:
            raise TodoValidationError("invalid_worktree_name", "name is required")
        index = self._store.load_index()
        binding = index.by_name.get(normalized)
        if binding is None:
            raise TodoValidationError("missing_worktree", "worktree not found", {"name": normalized})
        return binding

    def _resolve_by_task(self, task_id: str) -> WorktreeBinding:
        normalized = str(task_id).strip()
        if not normalized:
            raise TodoValidationError("invalid_task_id", "task id is required")
        index = self._store.load_index()
        name = index.by_task.get(normalized)
        if not name:
            raise TodoValidationError("missing_worktree", "task has no active worktree", {"task_id": normalized})
        binding = index.by_name.get(name)
        if binding is None:
            raise TodoValidationError("missing_worktree", "worktree not found", {"name": name, "task_id": normalized})
        return binding

    def resolve(self, *, name: str | None = None, task_id: str | None = None) -> WorktreeBinding:
        if name and str(name).strip():
            return self._resolve_by_name(str(name))
        if task_id and str(task_id).strip():
            return self._resolve_by_task(str(task_id))
        raise TodoValidationError("invalid_worktree_ref", "either name or task_id is required")

    def create_and_bind(self, *, task_id: str, name: str, base_ref: str | None = None) -> dict[str, Any]:
        normalized_task = str(task_id).strip()
        normalized_name = str(name).strip()
        if not normalized_task:
            raise TodoValidationError("invalid_task_id", "task id is required")
        if not normalized_name:
            raise TodoValidationError("invalid_worktree_name", "name is required")

        graph = self._task_store.load_graph()
        task = graph.get(normalized_task)
        if task is None:
            raise TodoValidationError("missing_task", "task not found", {"id": normalized_task})

        trees_dir = self._trees_dir()
        path = trees_dir / normalized_name
        base_ref_value = str(base_ref).strip() if base_ref is not None else "HEAD"
        branch = f"wt/{normalized_name}"

        reserved = self._store.reserve_creating(
            task_id=normalized_task,
            name=normalized_name,
            path=str(path),
            base_ref=base_ref_value,
        )
        self._store.append_event(
            "worktree.create.before",
            {"task_id": normalized_task, "name": normalized_name, "path": str(path), "base_ref": base_ref_value},
        )
        try:
            self._git.add(
                repo_dir=self._repo_dir,
                worktree_dir=path,
                name=normalized_name,
                base_ref=base_ref_value,
                branch=branch,
                force=False,
            )
            path.mkdir(parents=True, exist_ok=True)

            started_task = self._task_store.start_task(normalized_task)

            binding = WorktreeBinding.from_mapping({**reserved.to_dict(), "status": "active", "updated_at": _utc_iso()})
            saved = self._store.upsert(binding)
            after_ref = self._store.append_event(
                "worktree.create.after",
                {
                    "id": saved.id,
                    "task_id": normalized_task,
                    "name": normalized_name,
                    "path": saved.path,
                    "base_ref": saved.base_ref,
                },
            )
            saved.last_event_id = after_ref.entry
            self._store.upsert(saved)

            return {"binding": saved, "task": started_task}
        except TodoValidationError as exc:
            self._store.append_event(
                "worktree.create.failed",
                {"task_id": normalized_task, "name": normalized_name, "path": str(path), "error": exc.to_dict()},
            )
            try:
                self._git.remove(repo_dir=self._repo_dir, worktree_dir=path, force=True)
            except Exception:
                pass
            self._store.release_creating(name=normalized_name)
            raise
        except Exception as exc:
            self._store.append_event(
                "worktree.create.failed",
                {"task_id": normalized_task, "name": normalized_name, "path": str(path), "error": str(exc)},
            )
            try:
                self._git.remove(repo_dir=self._repo_dir, worktree_dir=path, force=True)
            except Exception:
                pass
            self._store.release_creating(name=normalized_name)
            raise

    def keep(self, *, name: str) -> WorktreeBinding:
        binding = self._resolve_by_name(name)
        if binding.status == "kept":
            self._store.append_event("worktree.keep", {"name": binding.name, "task_id": binding.task_id})
            return binding
        updated = WorktreeBinding.from_mapping({**binding.to_dict(), "status": "kept", "updated_at": _utc_iso()})
        saved = self._store.upsert(updated)
        ref = self._store.append_event("worktree.keep", {"name": saved.name, "task_id": saved.task_id})
        saved.last_event_id = ref.entry
        self._store.upsert(saved)
        return saved

    def remove(self, *, name: str, complete_task: bool = True, force: bool = False) -> dict[str, Any]:
        binding = self._resolve_by_name(name)
        self._store.append_event(
            "worktree.remove.before",
            {"name": binding.name, "task_id": binding.task_id, "path": binding.path, "force": bool(force)},
        )
        try:
            self._git.remove(repo_dir=self._repo_dir, worktree_dir=Path(binding.path), force=force)
            shutil.rmtree(binding.path, ignore_errors=True)
            removed = self._store.mark_removed(binding.name) or binding
            after_ref = self._store.append_event(
                "worktree.remove.after",
                {"name": binding.name, "task_id": binding.task_id, "path": binding.path},
            )
            removed.last_event_id = after_ref.entry
            self._store.upsert(removed)
            completed: dict[str, Any] | None = None
            if complete_task:
                result = self._task_store.complete(binding.task_id)
                task = result["task"]
                self._store.append_event("task.completed", {"task_id": binding.task_id, "name": binding.name})
                completed = {"task": task, "unlocked": result.get("unlocked", [])}
            return {"binding": removed, "completed": completed}
        except TodoValidationError as exc:
            self._store.append_event(
                "worktree.remove.failed",
                {"name": binding.name, "task_id": binding.task_id, "path": binding.path, "error": exc.to_dict()},
            )
            raise
        except Exception as exc:
            self._store.append_event(
                "worktree.remove.failed",
                {"name": binding.name, "task_id": binding.task_id, "path": binding.path, "error": str(exc)},
            )
            raise

    def exec(
        self,
        *,
        name: str | None = None,
        task_id: str | None = None,
        argv: list[str],
        timeout_seconds: float | None = None,
    ) -> WorktreeExecResult:
        normalized_argv = _normalize_argv(argv)
        binding = self.resolve(name=name, task_id=task_id)
        cwd = Path(binding.path)
        if binding.status not in {"active", "kept"}:
            raise TodoValidationError(
                "invalid_worktree_status",
                "worktree is not active",
                {"name": binding.name, "status": binding.status},
            )
        if not cwd.exists():
            raise TodoValidationError("missing_worktree_dir", "worktree directory missing", {"path": str(cwd)})

        self._store.append_event(
            "worktree.exec.before",
            {
                "name": binding.name,
                "task_id": binding.task_id,
                "path": binding.path,
                "argv": normalized_argv,
                "command": shlex.join(normalized_argv),
            },
        )
        completed = subprocess.run(
            normalized_argv,
            cwd=str(cwd),
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout_seconds,
        )
        result = WorktreeExecResult(
            returncode=int(completed.returncode),
            stdout=completed.stdout or "",
            stderr=completed.stderr or "",
            cwd=str(cwd),
        )
        after_ref = self._store.append_event(
            "worktree.exec.after",
            {
                "name": binding.name,
                "task_id": binding.task_id,
                "path": binding.path,
                "returncode": result.returncode,
            },
        )
        updated = WorktreeBinding.from_mapping({**binding.to_dict(), "updated_at": _utc_iso(), "last_event_id": after_ref.entry})
        self._store.upsert(updated)
        return result

    def recover(self) -> dict[str, Any]:
        graph = self._task_store.load_graph()
        index = self._store.load_index()
        missing_paths: list[str] = []
        missing_tasks: list[str] = []
        updated: list[str] = []

        for binding in list(index.by_name.values()):
            if binding.status not in {"active", "kept"}:
                continue
            path = Path(binding.path)
            broken_reason: str | None = None
            if not path.exists():
                broken_reason = "missing_path"
                missing_paths.append(binding.name)
            elif binding.task_id not in graph:
                broken_reason = "missing_task"
                missing_tasks.append(binding.name)
            if broken_reason is None:
                continue
            payload = {**binding.to_dict(), "status": "broken", "updated_at": _utc_iso()}
            broken = WorktreeBinding.from_mapping(payload)
            ref = self._store.append_event(
                "worktree.recover.broken",
                {"name": broken.name, "task_id": broken.task_id, "path": broken.path, "reason": broken_reason},
            )
            broken.last_event_id = ref.entry
            self._store.upsert(broken)
            updated.append(broken.name)

        return {"missing_paths": missing_paths, "missing_tasks": missing_tasks, "updated": updated}


def _normalize_argv(argv: list[str]) -> list[str]:
    if not isinstance(argv, list) or not argv:
        raise TodoValidationError("invalid_argv", "argv must be a non-empty array of strings")
    normalized: list[str] = []
    for item in argv:
        if not isinstance(item, str):
            raise TodoValidationError("invalid_argv", "argv must be a non-empty array of strings")
        value = item.strip()
        if not value:
            raise TodoValidationError("invalid_argv", "argv must be a non-empty array of strings")
        normalized.append(value)
    return normalized
