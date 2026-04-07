from __future__ import annotations

from pathlib import Path
from typing import Any, TYPE_CHECKING

from .models import TodoValidationError, ToolCall, ToolDefinition, ToolHandler, ToolResult
from .background_task_manager import BackgroundTaskManager
from .message_bus import MessageBus
from .skills_engine import SkillsEngine
from .spec_scaffold import scaffold_spec
from .subagent_manager import SubagentManager
from .task_system import TaskGraphStore
from .worktree_manager import WorktreeManager
from .worktree_store import WorktreeStore

if TYPE_CHECKING:
    from .teammate_manager import TeammateManager
from src.integrations.tavily_client import TavilyError, tavily_search


def _bash_handler(arguments: dict[str, object]) -> ToolResult:
    command = str(arguments.get("command", ""))
    return ToolResult(success=True, output={"tool": "bash", "command": command})


def _resolve_file_path(path: str, *, file_root: Path) -> Path:
    raw = str(path).strip()
    candidate = Path(raw)
    if candidate.is_absolute():
        return candidate.resolve(strict=False)
    resolved = (file_root / candidate).resolve(strict=False)
    root = file_root.resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        raise TodoValidationError(
            "invalid_file_path",
            "path escapes configured file root",
            {"path": raw, "root": str(root)},
        )
    return resolved


def _normalize_argv_argument(arguments: dict[str, Any]) -> list[str]:
    argv = arguments.get("argv")
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


def _read_file_handler(arguments: dict[str, object], *, file_root: Path) -> ToolResult:
    path = str(arguments.get("path", "")).strip()
    if not path:
        return ToolResult(success=False, error="missing path")

    file_path = _resolve_file_path(path, file_root=file_root)
    if not file_path.exists():
        return ToolResult(success=False, error=f"file not found: {path}")

    if not file_path.is_file():
        return ToolResult(success=False, error=f"not a file: {path}")

    content = file_path.read_text(encoding="utf-8")
    offset = arguments.get("offset")
    limit = arguments.get("limit")

    if offset is not None or limit is not None:
        lines = content.splitlines()
        start = max(int(offset or 1) - 1, 0)
        stop = start + int(limit) if limit is not None else None
        sliced_lines = lines[start:stop]
        content = "\n".join(sliced_lines)
        if sliced_lines and file_path.read_text(encoding="utf-8").endswith("\n"):
            content = f"{content}\n"

    return ToolResult(
        success=True, output={"tool": "read_file", "path": path, "content": content}
    )


def _write_file_handler(arguments: dict[str, object], *, file_root: Path) -> ToolResult:
    path = str(arguments.get("path", "")).strip()
    if not path:
        return ToolResult(success=False, error="missing path")

    if "content" not in arguments:
        return ToolResult(success=False, error="missing content")

    content = str(arguments.get("content", ""))
    file_path = _resolve_file_path(path, file_root=file_root)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content, encoding="utf-8")

    return ToolResult(
        success=True,
        output={
            "tool": "write_file",
            "path": path,
            "bytes_written": len(content.encode("utf-8")),
        },
    )


def _edit_file_handler(arguments: dict[str, object], *, file_root: Path) -> ToolResult:
    path = str(arguments.get("path", "")).strip()
    old_string = str(arguments.get("old_string", ""))
    if not path:
        return ToolResult(success=False, error="missing path")

    if not old_string:
        return ToolResult(success=False, error="missing old_string")

    if "new_string" not in arguments:
        return ToolResult(success=False, error="missing new_string")

    new_string = str(arguments.get("new_string", ""))
    file_path = _resolve_file_path(path, file_root=file_root)
    if not file_path.exists():
        return ToolResult(success=False, error=f"file not found: {path}")

    if not file_path.is_file():
        return ToolResult(success=False, error=f"not a file: {path}")

    original = file_path.read_text(encoding="utf-8")
    if old_string not in original:
        return ToolResult(success=False, error=f"text not found in file: {path}")

    updated = original.replace(old_string, new_string, 1)
    file_path.write_text(updated, encoding="utf-8")

    return ToolResult(
        success=True,
        output={"tool": "edit_file", "path": path, "replaced": 1},
    )


def _task_graph_summary(tasks: list[dict[str, Any]]) -> dict[str, int]:
    ready = 0
    blocked = 0
    in_progress = 0
    completed = 0
    for task in tasks:
        status = str(task.get("status", ""))
        blocked_by = task.get("blockedBy") or []
        if status == "completed":
            completed += 1
        elif status == "in_progress":
            in_progress += 1
        elif status == "pending":
            if blocked_by:
                blocked += 1
            else:
                ready += 1
    return {
        "total": len(tasks),
        "ready": ready,
        "blocked": blocked,
        "in_progress": in_progress,
        "completed": completed,
    }


def _todowrite_handler(arguments: dict[str, Any], store: TaskGraphStore) -> ToolResult:
    tasks_arg = arguments.get("tasks")
    if tasks_arg is None:
        tasks_arg = arguments.get("todos")
    if tasks_arg is None:
        raise TodoValidationError("invalid_task_list", "missing tasks")

    merge = arguments.get("merge", False)
    if not isinstance(merge, bool):
        raise TodoValidationError("invalid_merge_flag", "merge must be a boolean")

    graph = store.write_graph(tasks_arg, merge=merge)
    tasks = [item.to_dict() for item in sorted(graph.values(), key=lambda t: t.id)]
    return ToolResult(
        success=True,
        output={"tool": "todowrite", "summary": _task_graph_summary(tasks), "tasks": tasks},
    )


class ToolRegistry:
    def __init__(
        self,
        task_graph_store: TaskGraphStore | None = None,
        skills_engine: SkillsEngine | None = None,
        background_manager: BackgroundTaskManager | None = None,
        worktree_manager: WorktreeManager | None = None,
        teammate_manager: "TeammateManager | None" = None,
        message_bus: MessageBus | None = None,
        model_provider: Any | None = None,
    ) -> None:
        self._tools: dict[str, ToolDefinition] = {}
        self._handlers: dict[str, ToolHandler] = {}
        self._task_graph_store = task_graph_store or TaskGraphStore()
        self._file_root = self._task_graph_store.root_dir.parent.parent.resolve()
        self._subagent_manager = SubagentManager()
        self._model_provider = model_provider
        self._skills_engine = skills_engine or SkillsEngine(base_dir=self._file_root / "skills")
        self._background_manager = background_manager or BackgroundTaskManager()
        if teammate_manager is None:
            from .teammate_manager import TeammateManager

            if message_bus is not None:
                self._teammates = TeammateManager(root_dir=message_bus.root_dir, model_provider=self._model_provider)
            else:
                self._teammates = TeammateManager(
                    root_dir=self._file_root / ".agent" / "teammates",
                    model_provider=self._model_provider,
                )
        else:
            self._teammates = teammate_manager

        self._bus = message_bus or MessageBus(root_dir=self._teammates.root_dir)
        self._worktree_manager = worktree_manager

    @classmethod
    def with_builtin_tools(
        cls,
        task_graph_store: TaskGraphStore | None = None,
        *,
        skills_engine: SkillsEngine | None = None,
        background_manager: BackgroundTaskManager | None = None,
        worktree_manager: WorktreeManager | None = None,
        teammate_manager: "TeammateManager | None" = None,
        message_bus: MessageBus | None = None,
        model_provider: Any | None = None,
    ) -> ToolRegistry:
        task_store = task_graph_store or TaskGraphStore()
        manager = worktree_manager
        if manager is None:
            base_dir = task_store.root_dir.parent.parent
            manager = WorktreeManager(
                repo_dir=base_dir,
                store=WorktreeStore(root_dir=base_dir / ".worktrees"),
                task_store=task_store,
            )
        manager.recover()
        registry = cls(
            task_graph_store=task_store,
            skills_engine=skills_engine,
            background_manager=background_manager,
            worktree_manager=manager,
            teammate_manager=teammate_manager,
            message_bus=message_bus,
            model_provider=model_provider,
        )
        registry.register_builtin_tools()
        return registry

    def set_model_provider(self, provider: Any | None) -> None:
        self._model_provider = provider

    @property
    def task_graph_store(self) -> TaskGraphStore:
        return self._task_graph_store

    @property
    def background_manager(self) -> BackgroundTaskManager:
        return self._background_manager

    @property
    def message_bus(self) -> MessageBus:
        return self._bus

    @property
    def teammate_manager(self) -> "TeammateManager":
        return self._teammates

    def register(self, definition: ToolDefinition, handler: ToolHandler) -> None:
        if definition.name in self._tools:
            raise ValueError(f"tool already registered: {definition.name}")

        self._tools[definition.name] = definition
        self._handlers[definition.name] = handler

    def register_builtin_tools(self) -> None:
        self.register(
            ToolDefinition(name="bash", description="Dispatch bash command calls"),
            _bash_handler,
        )
        self.register(
            ToolDefinition(
                name="teammate_list",
                description="List teammates from roster",
            ),
            lambda arguments: self._handle_teammate_list(arguments),
        )
        self.register(
            ToolDefinition(
                name="teammate_spawn",
                description="Spawn teammate run for a goal",
                parameters={"id": "str", "goal": "str", "allowed_tools": "list[str]?", "max_iters": "int?"},
            ),
            lambda arguments: self._handle_teammate_spawn(arguments),
        )
        self.register(
            ToolDefinition(
                name="teammate_send",
                description="Send inbox message to teammate",
                parameters={"to": "str", "sender": "str", "content": "str"},
            ),
            lambda arguments: self._handle_teammate_send(arguments),
        )
        self.register(
            ToolDefinition(
                name="background_run",
                description="Run a background command as argv and return immediately",
                parameters={"argv": "list[str]", "cwd": "str?", "timeout_seconds": "float?"},
            ),
            lambda arguments: self._handle_background_run(arguments),
        )
        self.register(
            ToolDefinition(
                name="background_status",
                description="Query background task status by task_id",
                parameters={"id": "str"},
            ),
            lambda arguments: self._handle_background_status(arguments),
        )
        self.register(
            ToolDefinition(
                name="background_cancel",
                description="Cancel a background task by task_id",
                parameters={"id": "str"},
            ),
            lambda arguments: self._handle_background_cancel(arguments),
        )
        self.register(
            ToolDefinition(
                name="web_search",
                description="Search the web via Tavily",
                parameters={"query": "str", "max_results": "int?"},
            ),
            lambda arguments: self._handle_web_search(arguments),
        )
        self.register(
            ToolDefinition(
                name="todowrite",
                description="Write persistent task graph for the current session",
                parameters={"tasks": "list[object]", "merge": "bool"},
            ),
            lambda arguments: _todowrite_handler(arguments, self._task_graph_store),
        )
        self.register(
            ToolDefinition(
                name="taskgraph_query",
                description="Query task graph by mode: ready, blocked, in_progress, completed",
                parameters={"mode": "str"},
            ),
            lambda arguments: ToolResult(
                success=True,
                output={
                    "tool": "taskgraph_query",
                    "mode": str(arguments.get("mode", "")),
                    "tasks": [
                        task.to_dict()
                        for task in self._task_graph_store.query(str(arguments.get("mode", "")))
                    ],
                },
            ),
        )
        self.register(
            ToolDefinition(
                name="taskgraph_query_unowned",
                description="Query ready tasks without owner",
            ),
            lambda arguments: ToolResult(
                success=True,
                output={
                    "tool": "taskgraph_query_unowned",
                    "tasks": [task.to_dict() for task in self._task_graph_store.query_ready_unowned()],
                },
            ),
        )
        self.register(
            ToolDefinition(
                name="task_complete",
                description="Mark task completed and unlock downstream tasks",
                parameters={"id": "str"},
            ),
            lambda arguments: self._handle_task_complete(arguments),
        )
        self.register(
            ToolDefinition(
                name="create_task",
                description="Create and persist a new task from a goal",
                parameters={"goal": "str", "id": "str?", "blockedBy": "list[str]?", "blocks": "list[str]?", "owner": "str?"},
            ),
            lambda arguments: self._handle_create_task(arguments),
        )
        self.register(
            ToolDefinition(
                name="task_start",
                description="Move task status to in_progress",
                parameters={"id": "str"},
            ),
            lambda arguments: self._handle_task_start(arguments),
        )
        self.register(
            ToolDefinition(
                name="task_claim",
                description="Claim a task for an owner",
                parameters={"id": "str", "owner": "str?"},
            ),
            lambda arguments: self._handle_task_claim(arguments),
        )
        self.register(
            ToolDefinition(
                name="task_release",
                description="Release a task owner back to unowned",
                parameters={"id": "str"},
            ),
            lambda arguments: self._handle_task_release(arguments),
        )
        self.register(
            ToolDefinition(
                name="worktree_create_and_bind",
                description="Create git worktree under .worktrees/trees/<name> and bind to task",
                parameters={"task_id": "str", "name": "str", "base_ref": "str?"},
            ),
            lambda arguments: self._handle_worktree_create_and_bind(arguments),
        )
        self.register(
            ToolDefinition(
                name="worktree_exec",
                description="Execute argv command inside a worktree directory",
                parameters={"name": "str?", "task_id": "str?", "argv": "list[str]", "timeout_seconds": "float?"},
            ),
            lambda arguments: self._handle_worktree_exec(arguments),
        )
        self.register(
            ToolDefinition(
                name="worktree_keep",
                description="Mark a worktree as kept for later use",
                parameters={"name": "str"},
            ),
            lambda arguments: self._handle_worktree_keep(arguments),
        )
        self.register(
            ToolDefinition(
                name="worktree_remove",
                description="Remove a worktree directory and optionally complete its task",
                parameters={"name": "str", "complete_task": "bool?", "force": "bool?"},
            ),
            lambda arguments: self._handle_worktree_remove(arguments),
        )
        self.register(
            ToolDefinition(
                name="read_file",
                description="Read local file contents",
                parameters={"path": "str", "offset": "int?", "limit": "int?"},
            ),
            lambda arguments: _read_file_handler(arguments, file_root=self._file_root),
        )
        self.register(
            ToolDefinition(
                name="write_file",
                description="Write local file contents",
                parameters={"path": "str", "content": "str"},
            ),
            lambda arguments: _write_file_handler(arguments, file_root=self._file_root),
        )
        self.register(
            ToolDefinition(
                name="edit_file",
                description="Edit local file contents by replacement",
                parameters={"path": "str", "old_string": "str", "new_string": "str"},
            ),
            lambda arguments: _edit_file_handler(arguments, file_root=self._file_root),
        )
        self.register(
            ToolDefinition(
                name="spec_scaffold",
                description="Create a new spec skeleton under .trae/specs/<change_id>",
                parameters={"change_id": "str", "feature_name": "str?", "force": "bool?"},
            ),
            lambda arguments: self._handle_spec_scaffold(arguments),
        )
        self.register(
            ToolDefinition(
                name="skill_list",
                description="List available skills",
            ),
            lambda arguments: self._handle_skill_list(arguments),
        )
        self.register(
            ToolDefinition(
                name="skill_load",
                description="Load a skill by name",
                parameters={"name": "str"},
            ),
            lambda arguments: self._handle_skill_load(arguments),
        )
        self.register(
            ToolDefinition(
                name="compact",
                description="Compact active context and persist transcript",
                parameters={"reason": "str?"},
            ),
            lambda arguments: self._handle_compact(arguments),
        )
        self.register(
            ToolDefinition(
                name="spawn_subagent",
                description="Spawn a child agent with isolated context and scoped tool allowlist",
                parameters={
                    "goal": "str",
                    "allowed_tools": "list[str]",
                    "max_iters": "int?",
                    "max_tool_calls": "int?",
                },
            ),
            lambda arguments: self._handle_spawn_subagent(arguments),
        )
        self.register(
            ToolDefinition(
                name="idle",
                description="Idle sentinel tool",
            ),
            lambda arguments: ToolResult(success=True, output={"tool": "idle"}),
        )

    def list_tools(self) -> list[ToolDefinition]:
        return list(self._tools.values())

    def get_tool(self, name: str) -> ToolDefinition:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise KeyError(f"unknown tool: {name}") from exc

    def invoke(self, call: ToolCall) -> ToolResult:
        try:
            handler = self._handlers[call.name]
        except KeyError:
            return ToolResult(success=False, error=f"unknown tool: {call.name}")

        try:
            return handler(call.arguments)
        except TodoValidationError as exc:
            return ToolResult(success=False, output=None, error=exc.to_dict())

    def _handle_task_complete(self, arguments: dict[str, Any]) -> ToolResult:
        task_id = str(arguments.get("id", "")).strip()
        if not task_id:
            raise TodoValidationError("invalid_task_id", "task id is required")
        result = self._task_graph_store.complete(task_id)
        task = result["task"]
        unlocked = result["unlocked"]
        return ToolResult(
            success=True,
            output={
                "tool": "task_complete",
                "task": task.to_dict(),
                "unlocked": unlocked,
            },
        )

    def _handle_create_task(self, arguments: dict[str, Any]) -> ToolResult:
        goal = arguments.get("goal")
        if goal is None:
            goal = arguments.get("content")
        goal_value = str(goal or "").strip()
        if not goal_value:
            raise TodoValidationError("invalid_task_content", "goal is required")

        task_id_raw = arguments.get("id")
        task_id = str(task_id_raw).strip() if task_id_raw is not None else None
        blocked_by_raw = arguments.get("blockedBy", [])
        blocks_raw = arguments.get("blocks", [])
        owner_raw = arguments.get("owner")

        if blocked_by_raw is not None and not isinstance(blocked_by_raw, list):
            raise TodoValidationError("invalid_task_dependencies", "blockedBy must be an array of strings")
        if blocks_raw is not None and not isinstance(blocks_raw, list):
            raise TodoValidationError("invalid_task_dependencies", "blocks must be an array of strings")

        blocked_by = [str(item).strip() for item in (blocked_by_raw or [])]
        blocks = [str(item).strip() for item in (blocks_raw or [])]
        owner = str(owner_raw).strip() if owner_raw is not None else None

        task = self._task_graph_store.create_task(
            goal_value,
            task_id=task_id,
            blocked_by=blocked_by,
            blocks=blocks,
            owner=owner,
        )
        return ToolResult(success=True, output={"tool": "create_task", "task": task.to_dict()})

    def _handle_task_start(self, arguments: dict[str, Any]) -> ToolResult:
        task_id = str(arguments.get("id", "")).strip()
        if not task_id:
            raise TodoValidationError("invalid_task_id", "task id is required")
        task = self._task_graph_store.start_task(task_id)
        return ToolResult(success=True, output={"tool": "task_start", "task": task.to_dict()})

    def _handle_task_claim(self, arguments: dict[str, Any]) -> ToolResult:
        task_id = str(arguments.get("id", "")).strip()
        if not task_id:
            raise TodoValidationError("invalid_task_id", "task id is required")
        owner_raw = arguments.get("owner", "")
        owner_value = str(owner_raw).strip() or "system"
        task = self._task_graph_store.claim_task(task_id, owner_value)
        return ToolResult(success=True, output={"tool": "task_claim", "task": task.to_dict()})

    def _handle_task_release(self, arguments: dict[str, Any]) -> ToolResult:
        task_id = str(arguments.get("id", "")).strip()
        if not task_id:
            raise TodoValidationError("invalid_task_id", "task id is required")
        task = self._task_graph_store.release_task(task_id)
        return ToolResult(success=True, output={"tool": "task_release", "task": task.to_dict()})

    def _handle_worktree_create_and_bind(self, arguments: dict[str, Any]) -> ToolResult:
        task_id = arguments.get("task_id")
        if task_id is None:
            task_id = arguments.get("taskId")
        name = arguments.get("name")
        base_ref = arguments.get("base_ref")
        if base_ref is None:
            base_ref = arguments.get("baseRef")
        if not isinstance(task_id, str) or not task_id.strip():
            raise TodoValidationError("invalid_task_id", "task_id must be a non-empty string")
        if not isinstance(name, str) or not name.strip():
            raise TodoValidationError("invalid_worktree_name", "name must be a non-empty string")
        if base_ref is not None and not isinstance(base_ref, str):
            raise TodoValidationError("invalid_base_ref", "base_ref must be a string if provided")
        result = self._worktree_manager.create_and_bind(
            task_id=task_id.strip(),
            name=name.strip(),
            base_ref=base_ref.strip() if isinstance(base_ref, str) else None,
        )
        binding = result["binding"]
        task = result["task"]
        return ToolResult(
            success=True,
            output={
                "tool": "worktree_create_and_bind",
                "binding": binding.to_dict(),
                "task": task.to_dict(),
            },
        )

    def _handle_worktree_exec(self, arguments: dict[str, Any]) -> ToolResult:
        name = arguments.get("name")
        task_id = arguments.get("task_id")
        if task_id is None:
            task_id = arguments.get("taskId")
        argv = _normalize_argv_argument(arguments)
        if name is not None and not isinstance(name, str):
            raise TodoValidationError("invalid_worktree_name", "name must be a string if provided")
        if task_id is not None and not isinstance(task_id, str):
            raise TodoValidationError("invalid_task_id", "task_id must be a string if provided")
        timeout_raw = arguments.get("timeout_seconds")
        timeout_seconds: float | None = None
        if timeout_raw is not None:
            try:
                timeout_seconds = float(timeout_raw)
            except (TypeError, ValueError) as exc:
                raise TodoValidationError("invalid_timeout", "timeout_seconds must be a number") from exc
        result = self._worktree_manager.exec(
            name=name.strip() if isinstance(name, str) and name.strip() else None,
            task_id=task_id.strip() if isinstance(task_id, str) and task_id.strip() else None,
            argv=argv,
            timeout_seconds=timeout_seconds,
        )
        return ToolResult(
            success=True,
            output={"tool": "worktree_exec", "result": result.to_dict()},
        )

    def _handle_worktree_keep(self, arguments: dict[str, Any]) -> ToolResult:
        name = arguments.get("name")
        if not isinstance(name, str) or not name.strip():
            raise TodoValidationError("invalid_worktree_name", "name must be a non-empty string")
        binding = self._worktree_manager.keep(name=name.strip())
        return ToolResult(success=True, output={"tool": "worktree_keep", "binding": binding.to_dict()})

    def _handle_worktree_remove(self, arguments: dict[str, Any]) -> ToolResult:
        name = arguments.get("name")
        if not isinstance(name, str) or not name.strip():
            raise TodoValidationError("invalid_worktree_name", "name must be a non-empty string")
        complete_task = arguments.get("complete_task")
        if complete_task is None:
            complete_task = arguments.get("completeTask", True)
        if complete_task is not None and not isinstance(complete_task, bool):
            raise TodoValidationError("invalid_complete_task", "complete_task must be a boolean if provided")
        force = arguments.get("force", False)
        if force is not None and not isinstance(force, bool):
            raise TodoValidationError("invalid_force", "force must be a boolean if provided")
        result = self._worktree_manager.remove(
            name=name.strip(),
            complete_task=bool(complete_task) if complete_task is not None else True,
            force=bool(force),
        )
        binding = result["binding"]
        return ToolResult(
            success=True,
            output={
                "tool": "worktree_remove",
                "binding": binding.to_dict(),
                "completed": (
                    {
                        "task": result["completed"]["task"].to_dict(),
                        "unlocked": list(result["completed"].get("unlocked", [])),
                    }
                    if result.get("completed") is not None
                    else None
                ),
            },
        )

    def _handle_skill_list(self, arguments: dict[str, Any]) -> ToolResult:
        try:
            skills = self._skills_engine.list_skills()
        except TodoValidationError as exc:
            return ToolResult(success=False, output=exc.to_dict(), error=exc.message)
        return ToolResult(success=True, output={"tool": "skill_list", "skills": skills})

    def _handle_skill_load(self, arguments: dict[str, Any]) -> ToolResult:
        name = arguments.get("name")
        if not isinstance(name, str) or not name.strip():
            return self._invalid_tool_args(
                "invalid_skill_name",
                "name must be a non-empty string",
                {"name": name},
            )
        normalized = name.strip()
        if "/" in normalized or "\\" in normalized or ".." in normalized:
            return self._invalid_tool_args(
                "invalid_skill_name",
                "name contains invalid characters",
                {"name": normalized},
            )
        try:
            content = self._skills_engine.load_skill(normalized)
        except TodoValidationError as exc:
            return ToolResult(success=False, output=exc.to_dict(), error=exc.message)
        return ToolResult(
            success=True,
            output={"tool": "skill_load", "name": normalized, "content": content},
        )

    def _handle_spec_scaffold(self, arguments: dict[str, Any]) -> ToolResult:
        change_id = arguments.get("change_id")
        if change_id is None:
            change_id = arguments.get("changeId")
        feature_name = arguments.get("feature_name")
        if feature_name is None:
            feature_name = arguments.get("featureName")
        force_raw = arguments.get("force", False)
        force = bool(force_raw) if isinstance(force_raw, bool) else False

        if not isinstance(change_id, str):
            return self._invalid_tool_args(
                "invalid_change_id",
                "change_id must be a string",
                {"change_id": change_id},
            )
        if feature_name is not None and not isinstance(feature_name, str):
            return self._invalid_tool_args(
                "invalid_feature_name",
                "feature_name must be a string",
                {"feature_name": feature_name},
            )
        return scaffold_spec(change_id, feature_name, force=force)

    def _handle_spawn_subagent(self, arguments: dict[str, Any]) -> ToolResult:
        goal = arguments.get("goal")
        if not isinstance(goal, str) or not goal.strip():
            return self._invalid_tool_args(
                "invalid_goal",
                "goal must be a non-empty string",
                {"goal": goal},
            )

        allowed_tools = arguments.get("allowed_tools")
        if allowed_tools is None:
            allowed_tools = arguments.get("allowedTools")
        if not isinstance(allowed_tools, list) or any(not isinstance(item, str) for item in allowed_tools):
            return self._invalid_tool_args(
                "invalid_allowed_tools",
                "allowed_tools must be a list of strings",
                {"allowed_tools": allowed_tools},
            )

        max_iters_raw = arguments.get("max_iters", 30)
        max_tool_calls_raw = arguments.get("max_tool_calls", 50)
        try:
            max_iters = int(max_iters_raw)
        except (TypeError, ValueError):
            return self._invalid_tool_args(
                "invalid_max_iters",
                "max_iters must be an integer",
                {"max_iters": max_iters_raw},
            )
        try:
            max_tool_calls = int(max_tool_calls_raw)
        except (TypeError, ValueError):
            return self._invalid_tool_args(
                "invalid_max_tool_calls",
                "max_tool_calls must be an integer",
                {"max_tool_calls": max_tool_calls_raw},
            )
        if max_iters < 1:
            return self._invalid_tool_args(
                "invalid_max_iters",
                "max_iters must be >= 1",
                {"max_iters": max_iters},
            )
        if max_tool_calls < 1:
            return self._invalid_tool_args(
                "invalid_max_tool_calls",
                "max_tool_calls must be >= 1",
                {"max_tool_calls": max_tool_calls},
            )

        result = self._subagent_manager.run_subagent(
            goal=goal,
            allowed_tools=allowed_tools,
            registry_factory=ToolRegistry.with_builtin_tools,
            model_provider=self._model_provider,
            max_iters=max_iters,
            max_tool_calls=max_tool_calls,
        )

        ok = bool(result.get("ok")) if isinstance(result, dict) else False
        summary = str(result.get("summary", ""))
        stats = result.get("stats") if isinstance(result.get("stats"), dict) else {}
        if not ok:
            return ToolResult(
                success=False,
                error=summary or "subagent failed",
                output={
                    "code": "subagent_failed",
                    "message": summary or "subagent failed",
                    "details": {"stats": stats, "result_type": result.get("result_type"), "error": result.get("error")},
                },
            )
        return ToolResult(
            success=True,
            output={
                "tool": "spawn_subagent",
                "summary": summary,
                "stats": stats,
            },
        )

    def _invalid_tool_args(self, code: str, message: str, details: dict[str, Any]) -> ToolResult:
        return ToolResult(
            success=False,
            error=message,
            output={"code": code, "message": message, "details": details},
        )

    def _handle_compact(self, arguments: dict[str, Any]) -> ToolResult:
        from .context_compression import ContextCompression
        from .transcript_store import TranscriptStore

        store = TranscriptStore()
        transcript_id = store.new_transcript_id()
        compression = ContextCompression(store)
        reason = str(arguments.get("reason", "")).strip() or "manual"
        result = compression.manual_compact(transcript_id, [], reason)
        return ToolResult(success=True, output=result.to_tool_output())

    def _handle_teammate_list(self, arguments: dict[str, Any]) -> ToolResult:
        try:
            items = self._teammates.list()
        except TodoValidationError as exc:
            return self._tool_error(exc.code, exc.message, exc.details or {})
        return ToolResult(success=True, output={"tool": "teammate_list", "teammates": items})

    def _handle_teammate_spawn(self, arguments: dict[str, Any]) -> ToolResult:
        teammate_id = arguments.get("id")
        goal = arguments.get("goal")
        allowed_tools = arguments.get("allowed_tools", arguments.get("allowedTools", []))
        max_iters = arguments.get("max_iters")

        if not isinstance(teammate_id, str) or not teammate_id.strip():
            return self._tool_error(
                "invalid_teammate_id",
                "id must be a non-empty string",
                {"id": teammate_id},
            )
        if not isinstance(goal, str) or not goal.strip():
            return self._tool_error(
                "invalid_goal",
                "goal must be a non-empty string",
                {"goal": goal},
            )
        if allowed_tools is not None and not isinstance(allowed_tools, list):
            return self._tool_error(
                "invalid_allowed_tools",
                "allowed_tools must be a list of strings",
                {"allowed_tools": allowed_tools},
            )
        if isinstance(allowed_tools, list) and any(not isinstance(item, str) for item in allowed_tools):
            return self._tool_error(
                "invalid_allowed_tools",
                "allowed_tools must be a list of strings",
                {"allowed_tools": allowed_tools},
            )
        max_iters_value: int | None = None
        if max_iters is not None:
            try:
                max_iters_value = int(max_iters)
            except (TypeError, ValueError):
                return self._tool_error(
                    "invalid_max_iters",
                    "max_iters must be an integer",
                    {"max_iters": max_iters},
                )
        try:
            resp = self._teammates.spawn(
                str(teammate_id).strip(),
                str(goal).strip(),
                allowed_tools=allowed_tools or [],
                max_iters=max_iters_value,
            )
        except TodoValidationError as exc:
            return self._tool_error(exc.code, exc.message, exc.details or {})
        return ToolResult(success=True, output={"tool": "teammate_spawn", "result": resp})

    def _handle_teammate_send(self, arguments: dict[str, Any]) -> ToolResult:
        to = arguments.get("to")
        sender = arguments.get("sender")
        content = arguments.get("content")
        if not isinstance(to, str) or not to.strip():
            return self._tool_error("invalid_message_to", "message 'to' is required", {"to": to})
        if not isinstance(sender, str) or not sender.strip():
            return self._tool_error("invalid_message_sender", "sender is required", {"sender": sender})
        if not isinstance(content, str) or not content.strip():
            return self._tool_error("invalid_message_content", "content is required", {"content": content})
        try:
            msg = self._bus.send(str(to).strip(), str(sender).strip(), str(content).strip())
        except TodoValidationError as exc:
            return self._tool_error(exc.code, exc.message, exc.details or {})
        return ToolResult(success=True, output={"tool": "teammate_send", "message": msg})

    def _handle_background_run(self, arguments: dict[str, Any]) -> ToolResult:
        try:
            argv = _normalize_argv_argument(arguments)
        except TodoValidationError as exc:
            return self._tool_error(
                exc.code,
                exc.message,
                exc.details,
            )
        cwd = arguments.get("cwd")
        if cwd is not None and not isinstance(cwd, str):
            return self._tool_error(
                "invalid_cwd",
                "cwd must be a string if provided",
                {"cwd": cwd},
            )
        timeout_raw = arguments.get("timeout_seconds")
        timeout_seconds: float | None = None
        if timeout_raw is not None:
            try:
                timeout_seconds = float(timeout_raw)
            except (TypeError, ValueError):
                return self._tool_error(
                    "invalid_timeout",
                    "timeout_seconds must be a number",
                    {"timeout_seconds": timeout_raw},
                )
        task_id = self._background_manager.submit(argv, cwd=cwd, timeout_seconds=timeout_seconds)
        record = self._background_manager.status(task_id)
        return ToolResult(
            success=True,
            output={
                "tool": "background_run",
                "task_id": task_id,
                "status": record.status if record else "pending",
            },
        )

    def _handle_background_status(self, arguments: dict[str, Any]) -> ToolResult:
        task_id = arguments.get("id")
        if not isinstance(task_id, str) or not task_id.strip():
            return self._tool_error(
                "invalid_task_id",
                "id must be a non-empty string",
                {"id": task_id},
            )
        record = self._background_manager.status(task_id.strip())
        if record is None:
            return self._tool_error("not_found", "task not found", {"id": task_id})
        return ToolResult(success=True, output={"tool": "background_status", "record": record.to_dict()})

    def _handle_background_cancel(self, arguments: dict[str, Any]) -> ToolResult:
        task_id = arguments.get("id")
        if not isinstance(task_id, str) or not task_id.strip():
            return self._tool_error(
                "invalid_task_id",
                "id must be a non-empty string",
                {"id": task_id},
            )
        ok = self._background_manager.cancel(task_id.strip())
        record = self._background_manager.status(task_id.strip())
        return ToolResult(
            success=ok,
            output={
                "tool": "background_cancel",
                "ok": ok,
                "record": record.to_dict() if record else None,
            },
            error=None if ok else {"code": "cancel_failed", "message": "cancel failed", "details": {"id": task_id}},
        )

    def _handle_web_search(self, arguments: dict[str, Any]) -> ToolResult:
        query = arguments.get("query")
        if not isinstance(query, str) or not query.strip():
            return self._tool_error("invalid_query", "query must be a non-empty string", {"query": query})
        max_results_raw = arguments.get("max_results", 5)
        try:
            max_results = int(max_results_raw)
        except (TypeError, ValueError):
            return self._tool_error(
                "invalid_max_results",
                "max_results must be an integer",
                {"max_results": max_results_raw},
            )
        if max_results < 1 or max_results > 10:
            return self._tool_error(
                "invalid_max_results",
                "max_results must be between 1 and 10",
                {"max_results": max_results},
            )

        try:
            data = tavily_search(query.strip(), max_results=max_results)
        except TavilyError as exc:
            return self._tool_error(exc.code, exc.message, dict(exc.details or {}))
        except Exception as exc:
            return self._tool_error("upstream_error", "web_search failed", {"error": str(exc)})

        results_raw = data.get("results")
        results: list[dict[str, Any]] = []
        if isinstance(results_raw, list):
            for item in results_raw:
                if not isinstance(item, dict):
                    continue
                results.append(
                    {
                        "title": str(item.get("title", "") or ""),
                        "url": str(item.get("url", "") or ""),
                        "content": str(item.get("content", "") or ""),
                        "score": item.get("score"),
                    }
                )

        answer_raw = data.get("answer")
        answer = str(answer_raw) if isinstance(answer_raw, str) and answer_raw.strip() else None
        return ToolResult(
            success=True,
            output={
                "tool": "web_search",
                "query": query.strip(),
                "max_results": max_results,
                "answer": answer,
                "results": results,
            },
        )

    def _tool_error(self, code: str, message: str, details: dict[str, Any]) -> ToolResult:
        return ToolResult(success=False, output=None, error={"code": code, "message": message, "details": details})
