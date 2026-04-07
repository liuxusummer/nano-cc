from __future__ import annotations

import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol


class _RegistryLike(Protocol):
    def list_tools(self) -> list[Any]: ...

    def invoke(self, call: Any) -> Any: ...

    @property
    def task_graph_store(self) -> Any: ...


RegistryFactory = Callable[[Any], _RegistryLike]


class _RestrictedRegistry:
    def __init__(
        self,
        base: _RegistryLike,
        allowed_tools: set[str],
        max_tool_calls: int,
    ) -> None:
        self._base = base
        self._allowed_tools = allowed_tools
        self._max_tool_calls = max_tool_calls
        self._tool_calls = 0

    @property
    def tool_calls(self) -> int:
        return self._tool_calls

    @property
    def task_graph_store(self) -> Any:
        return self._base.task_graph_store

    def list_tools(self) -> list[Any]:
        tools = []
        for tool in self._base.list_tools():
            name = getattr(tool, "name", "")
            if isinstance(name, str) and name in self._allowed_tools:
                tools.append(tool)
        return tools

    def invoke(self, call: Any) -> Any:
        from .models import ToolResult

        if self._tool_calls >= self._max_tool_calls:
            return ToolResult(success=False, error="tool call limit reached")

        name = getattr(call, "name", "")
        self._tool_calls += 1
        if not isinstance(name, str) or name not in self._allowed_tools:
            return ToolResult(success=False, error=f"tool not allowed: {name}")

        return self._base.invoke(call)


class SubagentManager:
    def run_subagent(
        self,
        goal: str,
        allowed_tools: list[str],
        registry_factory: RegistryFactory,
        model_provider: Any | None = None,
        max_iters: int = 30,
        max_tool_calls: int = 50,
    ) -> dict[str, Any]:
        from .agent_loop import AgentLoop
        from .task_system import TaskGraphStore

        normalized_goal = str(goal).strip()
        allowed = {str(name).strip() for name in allowed_tools if str(name).strip()}
        if max_iters < 1:
            max_iters = 1
        if max_tool_calls < 1:
            max_tool_calls = 1

        with tempfile.TemporaryDirectory(prefix="subagent_") as tmp_dir:
            store = TaskGraphStore(root_dir=Path(tmp_dir) / "task_graph")
            store.write_graph(
                [
                    {
                        "id": "plan",
                        "content": "subagent",
                        "status": "pending",
                        "blockedBy": [],
                        "blocks": [],
                    }
                ],
                merge=False,
            )

            base_registry = registry_factory(store)
            restricted = _RestrictedRegistry(base_registry, allowed, max_tool_calls=max_tool_calls)
            loop = AgentLoop(registry=restricted, model_provider=model_provider)

            request: dict[str, Any] = {"mode": "auto", "goal": normalized_goal, "max_iters": max_iters}
            if model_provider is not None:
                request["strategy"] = "llm"
            resp = loop.run(request, max_iters=max_iters)

            ok = False
            summary = ""
            error = None
            result_type = ""
            if isinstance(resp, dict):
                result_type = str(resp.get("type", "")).strip()
                if result_type == "final":
                    ok = True
                    summary = str(resp.get("content", ""))
                elif result_type == "tool_result":
                    result = resp.get("result") or {}
                    output = result.get("output") if isinstance(result, dict) else None
                    summary = str(output)
                    ok = bool(result.get("success")) if isinstance(result, dict) else False
                    if not ok:
                        error = str(result.get("error", "")) if isinstance(result, dict) else "tool failed"
                elif result_type == "error":
                    summary = str(resp.get("error", ""))
                    error = summary
                else:
                    summary = str(resp)
                    error = summary
            else:
                summary = str(resp)
                error = summary

            trace = resp.get("trace") if isinstance(resp, dict) else None
            iterations = len(trace) if isinstance(trace, list) else 1

            return {
                "ok": ok,
                "result_type": result_type,
                "error": error,
                "summary": summary,
                "stats": {"iterations": iterations, "tool_calls": restricted.tool_calls},
            }
