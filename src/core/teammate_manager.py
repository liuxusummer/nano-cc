from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Mapping

from .agent_loop import AgentLoop
from .message_bus import MessageBus
from .model_provider import ModelProvider, PlanState
from .models import TeammateRecord, TodoValidationError
from .tool_registry import ToolRegistry


class _RestrictedRegistry:
    def __init__(self, base: ToolRegistry, allowed_tools: set[str]) -> None:
        self._base = base
        self._allowed = allowed_tools

    @property
    def task_graph_store(self) -> Any:
        return self._base.task_graph_store

    @property
    def background_manager(self) -> Any:
        return self._base.background_manager

    def list_tools(self) -> list[Any]:
        tools = []
        for tool in self._base.list_tools():
            name = getattr(tool, "name", "")
            if isinstance(name, str) and name in self._allowed:
                tools.append(tool)
        return tools

    def invoke(self, call: Any) -> Any:
        from .models import ToolResult

        name = getattr(call, "name", "")
        if not isinstance(name, str) or name not in self._allowed:
            return ToolResult(success=False, error=f"tool not allowed: {name}")
        return self._base.invoke(call)


class _InboxInjectingProvider:
    def __init__(self, base: ModelProvider, bus: MessageBus, teammate_id: str) -> None:
        self._base = base
        self._bus = bus
        self._teammate_id = teammate_id

    def plan_next(self, state: PlanState) -> Any:
        inbox = self._bus.read_inbox(self._teammate_id)
        obs = state.get("observation") or {}
        if not isinstance(obs, dict):
            obs = {}
        obs["inbox"] = inbox.get("messages", [])
        state["observation"] = obs
        return self._base.plan_next(state)


class TeammateManager:
    def __init__(self, root_dir: str | Path | None = None, model_provider: ModelProvider | None = None) -> None:
        self._root_dir = Path(root_dir) if root_dir is not None else Path(".agent/teammates")
        self._bus = MessageBus(root_dir=self._root_dir)
        self._provider = model_provider
        self._runs: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    @property
    def root_dir(self) -> Path:
        return self._root_dir

    @property
    def roster_path(self) -> Path:
        return self._root_dir / "config.json"

    def _ensure_root(self) -> None:
        p = self._root_dir
        if p.exists() and not p.is_dir():
            raise TodoValidationError("invalid_teammates_root_dir", "teammates root_dir is not a directory", {"path": str(p)})
        p.mkdir(parents=True, exist_ok=True)

    def load(self) -> dict[str, Any]:
        self._ensure_root()
        path = self.roster_path
        if not path.exists():
            legacy = self._legacy_roster_path()
            if legacy is not None and legacy.exists():
                items = self._load_roster_items(legacy)
                self._write_roster(items)
                return {"teammates": items}
        if not path.exists():
            return {"teammates": []}
        try:
            items = self._load_roster_items(path)
        except json.JSONDecodeError:
            raise TodoValidationError("invalid_roster", "malformed config.json", {"path": str(path)})
        return {"teammates": items}

    def _legacy_roster_path(self) -> Path | None:
        if self._root_dir.name != "teammates":
            return None
        if self._root_dir.parent.name != ".agent":
            return None
        project_root = self._root_dir.parent.parent
        return project_root / ".trae" / "teammates" / "config.json"

    def _load_roster_items(self, path: Path) -> list[dict[str, Any]]:
        data = json.loads(path.read_text(encoding="utf-8"))
        raw_list = data.get("teammates") if isinstance(data, dict) else None
        if raw_list is None:
            return []
        items: list[dict[str, Any]] = []
        for raw in raw_list:
            if not isinstance(raw, Mapping):
                continue
            rec = TeammateRecord.from_mapping(raw)
            items.append(rec.to_dict())
        return items

    def list(self) -> list[dict[str, Any]]:
        return self.load()["teammates"]

    def _write_roster(self, items: list[dict[str, Any]]) -> None:
        self._ensure_root()
        path = self.roster_path
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps({"teammates": items}, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)

    def add(self, record: Mapping[str, Any]) -> dict[str, Any]:
        rec = TeammateRecord.from_mapping(record)
        current = self.list()
        for item in current:
            if item.get("id") == rec.id:
                raise TodoValidationError("duplicate_teammate_id", "teammate already exists", {"id": rec.id})
        current.append(rec.to_dict())
        self._write_roster(current)
        return {"ok": True, "record": rec.to_dict()}

    def update(self, teammate_id: str, patch: Mapping[str, Any]) -> dict[str, Any]:
        normalized_id = str(teammate_id).strip()
        if not normalized_id:
            raise TodoValidationError("invalid_teammate_id", "teammate id is required")
        current = self.list()
        updated: list[dict[str, Any]] = []
        found = None
        for item in current:
            if item.get("id") == normalized_id:
                merged = dict(item)
                for k in ("email", "name", "status"):
                    if k in patch:
                        merged[k] = patch[k]
                rec = TeammateRecord.from_mapping(merged)
                found = rec.to_dict()
                updated.append(found)
            else:
                updated.append(item)
        if found is None:
            raise TodoValidationError("unknown_teammate", "teammate not found", {"id": normalized_id})
        self._write_roster(updated)
        return {"ok": True, "record": found}

    def spawn(
        self,
        teammate_id: str,
        goal: str,
        allowed_tools: list[str] | None = None,
        max_iters: int | None = None,
    ) -> dict[str, Any]:
        normalized_id = str(teammate_id).strip()
        if not normalized_id:
            raise TodoValidationError("invalid_teammate_id", "teammate id is required")
        allowed_set = {str(n).strip() for n in (allowed_tools or []) if str(n).strip()}
        max_local_iters = max_iters if isinstance(max_iters, int) and max_iters > 0 else 30
        base_registry = ToolRegistry.with_builtin_tools()
        registry = _RestrictedRegistry(base_registry, allowed_set) if allowed_set else base_registry
        provider = self._provider
        injected_provider = _InboxInjectingProvider(provider, self._bus, normalized_id) if provider is not None else None
        with self._lock:
            self._runs[normalized_id] = {"status": "pending", "summary": None}
        loop = AgentLoop(registry=registry, model_provider=injected_provider)
        request = {"mode": "auto", "goal": str(goal).strip(), "max_iters": max_local_iters}
        if injected_provider is not None:
            request["strategy"] = "llm"
        thread = threading.Thread(target=self._run_loop, args=(normalized_id, loop, request), daemon=True)
        with self._lock:
            self._runs[normalized_id]["status"] = "running"
            self._runs[normalized_id]["thread"] = thread
        thread.start()
        return {"ok": True}

    def _summarize_resp(self, resp: Any) -> str:
        if isinstance(resp, dict):
            t = str(resp.get("type", "")).strip()
            if t == "final":
                return str(resp.get("content", ""))
            if t == "error":
                return str(resp.get("error", ""))
            if t == "tool_result":
                result = resp.get("result") or {}
                output = result.get("output") if isinstance(result, dict) else None
                return str(output)
        return str(resp)

    def _run_loop(self, teammate_id: str, loop: AgentLoop, request: dict[str, Any]) -> None:
        try:
            resp = loop.run(request, max_iters=int(request.get("max_iters", 30)))
            summary = self._summarize_resp(resp)
            with self._lock:
                rec = self._runs.get(teammate_id) or {}
                rec["status"] = "completed"
                rec["summary"] = summary
                self._runs[teammate_id] = rec
        except Exception as exc:
            with self._lock:
                rec = self._runs.get(teammate_id) or {}
                rec["status"] = "failed"
                rec["summary"] = str(exc)
                self._runs[teammate_id] = rec

    def status(self, teammate_id: str) -> dict[str, Any]:
        normalized_id = str(teammate_id).strip()
        with self._lock:
            rec = self._runs.get(normalized_id)
            if rec is None:
                return {"status": "unknown", "summary": None}
            thread = rec.get("thread")
            if isinstance(thread, threading.Thread) and thread.is_alive():
                rec["status"] = "running"
            return {"status": rec.get("status"), "summary": rec.get("summary")}
