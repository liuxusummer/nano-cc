import os
from pathlib import Path
from typing import Any

from ..core.models import TodoValidationError
from ..core.task_system import TaskGraphStore


def _store(args: dict[str, Any]) -> TaskGraphStore:
    root = args.get("root") or os.getcwd()
    return TaskGraphStore(root_dir=Path(root) / ".agent" / "task_graph")


def _summary(store: TaskGraphStore) -> dict[str, int]:
    return {
        "total": len(store.load_graph()),
        "ready": len(store.query("ready")),
        "blocked": len(store.query("blocked")),
        "in_progress": len(store.query("in_progress")),
        "completed": len(store.query("completed")),
    }


def _success(output: dict[str, Any]) -> dict[str, Any]:
    return {"success": True, "output": output, "error": None}


def _failure(exc: TodoValidationError) -> dict[str, Any]:
    return {"success": False, "output": None, "error": exc.to_dict()}


def todowrite(args: dict[str, Any]) -> dict[str, Any]:
    items = args.get("tasks") or args.get("todos") or []
    merge = bool(args.get("merge", False))
    store = _store(args)
    try:
        graph = store.write_graph(items, merge)
    except TodoValidationError as exc:
        return _failure(exc)
    return _success(
        {
            "tool": "todowrite",
            "summary": _summary(store),
            "tasks": [item.to_dict() for item in sorted(graph.values(), key=lambda item: item.id)],
        }
    )


def taskgraph_query(args: dict[str, Any]) -> dict[str, Any]:
    mode = str(args.get("mode", "")).strip()
    store = _store(args)
    try:
        tasks = store.query(mode)
    except TodoValidationError as exc:
        return _failure(exc)
    return _success({"tool": "taskgraph_query", "mode": mode, "tasks": [task.to_dict() for task in tasks]})


def task_complete(args: dict[str, Any]) -> dict[str, Any]:
    task_id = args.get("id")
    store = _store(args)
    try:
        result = store.complete(str(task_id or ""))
    except TodoValidationError as exc:
        return _failure(exc)
    return _success(
        {
            "tool": "task_complete",
            "task": result["task"].to_dict(),
            "unlocked": list(result["unlocked"]),
        }
    )


def task_start(args: dict[str, Any]) -> dict[str, Any]:
    task_id = args.get("id")
    store = _store(args)
    try:
        task = store.start_task(str(task_id or ""))
    except TodoValidationError as exc:
        return _failure(exc)
    return _success({"tool": "task_start", "task": task.to_dict()})


def task_claim(args: dict[str, Any]) -> dict[str, Any]:
    task_id = args.get("id")
    owner = args.get("owner")
    store = _store(args)
    try:
        task = store.claim_task(str(task_id or ""), str(owner or ""))
    except TodoValidationError as exc:
        return _failure(exc)
    return _success({"tool": "task_claim", "task": task.to_dict()})


def task_release(args: dict[str, Any]) -> dict[str, Any]:
    task_id = args.get("id")
    store = _store(args)
    try:
        task = store.release_task(str(task_id or ""))
    except TodoValidationError as exc:
        return _failure(exc)
    return _success({"tool": "task_release", "task": task.to_dict()})
