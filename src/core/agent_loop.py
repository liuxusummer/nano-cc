from __future__ import annotations

from typing import Any
from uuid import uuid4

from . import ToolCall, ToolRegistry
from .background_task_manager import BackgroundTaskManager
from .model_provider import ModelProvider, PlanState
from .context_compression import ContextCompression
from .transcript_store import TranscriptStore


class AgentLoop:
    _TODOWRITE_REQUIRED_ERROR = "must call todowrite before other tools in auto mode"

    def __init__(
        self,
        registry: ToolRegistry | None = None,
        model_provider: ModelProvider | None = None,
        background_manager: BackgroundTaskManager | None = None,
    ) -> None:
        self._registry = registry
        self._model_provider = model_provider
        self._transcripts = TranscriptStore()
        self._transcript_id: str | None = None
        if background_manager is not None:
            self._background = background_manager
        elif registry is not None and hasattr(registry, "background_manager"):
            self._background = getattr(registry, "background_manager")
        else:
            self._background = BackgroundTaskManager()

    def run(self, request: dict, max_iters: int = 1, step_callback: Any = None) -> dict:
        if not isinstance(request, dict):
            return {"type": "error", "error": "invalid request"}

        mode = str(request.get("mode", "manual")).strip() or "manual"
        local_max_iters = int(request.get("max_iters", max_iters))
        if local_max_iters < 1:
            local_max_iters = 1

        if mode == "auto":
            trace: list[dict[str, Any]] = []
            content = request.get("content")
            tool_hint = request.get("tool_hint")
            goal = str(request.get("goal", "")).strip()
            goal_args_raw: Any = request.get("goal_args", {})
            goal_args = goal_args_raw if isinstance(goal_args_raw, dict) else {}
            strategy = str(request.get("strategy", "")).strip()
            if content:
                step = {"type": "answer", "content": str(content)}
                trace.append({"step": step, "observation": {}})
                return {"type": "final", "content": str(content), "trace": trace}

            plan_written_this_run = False
            plan_ready_before_run = self._has_authoritative_todos()
            if strategy == "llm" and self._model_provider is not None:
                if not self._transcript_id:
                    self._transcript_id = self._transcripts.new_transcript_id()
                compression = ContextCompression(self._transcripts, model_provider=self._model_provider)
                did_work_tool = False
                task_graph_present = False
                if self._registry is not None:
                    try:
                        task_graph_present = bool(self._registry.task_graph_store.load_graph())
                    except Exception:
                        task_graph_present = False
                task_execution_enabled = (
                    bool(goal)
                    and ("任务" in goal)
                    and (("完成" in goal) or ("继续" in goal) or ("未完成" in goal))
                )
                tools_desc: list[dict[str, Any]] = []
                if self._registry is not None:
                    for t in self._registry.list_tools():
                        tools_desc.append(
                            {"name": t.name, "description": t.description, "parameters": t.parameters}
                        )
                observation: dict[str, Any] = {}
                loop_state = "WORK"
                iters = 0
                while iters < local_max_iters:
                    if trace:
                        trace = compression.micro_compact(trace)
                    if compression.should_auto_compact(trace):
                        compact = compression.auto_compact(self._transcript_id, trace)
                        trace = compact.replacement_trace
                    notifications = [n.to_dict() for n in self._background.flush_notifications()]
                    if loop_state == "IDLE" and self._registry is not None:
                        obs = dict(observation)
                        identity = self._current_identity(request)
                        inbox = []
                        try:
                            inbox = self._registry.message_bus.read_inbox(identity["id"]).get("messages", [])
                        except Exception:
                            inbox = []
                        obs["inbox"] = inbox
                        ready = [t.to_dict() for t in self._registry.task_graph_store.query_ready_unowned()]
                        obs["taskboard"] = {"ready_unowned": ready}
                        observation = obs
                        if ready:
                            first = ready[0]
                            task_id = str(first.get("id", ""))
                            if task_id:
                                try:
                                    self._registry.task_graph_store.claim_task(task_id, identity["id"])
                                    self._registry.task_graph_store.start_task(task_id)
                                    task_execution_enabled = True
                                except Exception:
                                    pass
                            loop_state = "WORK"
                    state: PlanState = {
                        "goal": goal,
                        "tools": tools_desc,
                        "observation": self._merge_observation_with_notifications(observation, notifications),
                        "trace": trace,
                        "constraints": {
                            "only_return_json": True,
                            "todowrite_required": not (plan_written_this_run or plan_ready_before_run),
                        },
                    }
                    try:
                        if compression.should_auto_compact(trace) or len(str(trace)) < 512:
                            state["identity"] = self._current_identity(request)
                    except Exception:
                        pass
                    action = self._model_provider.plan_next(state)
                    step = {"type": action.get("type", "")}
                    if step["type"] == "answer":
                        content_value = str(action.get("content", ""))
                        if not content_value.strip():
                            trace.append(
                                {
                                    "step": {"type": "answer", "content": ""},
                                    "observation": {"error": "empty answer from planner"},
                                }
                            )
                            return {"type": "error", "error": "empty answer from planner", "trace": trace}
                        step["content"] = content_value
                        trace.append({"step": step, "observation": {}})
                        if self._registry is not None:
                            try:
                                identity = self._current_identity(request)
                                in_progress = self._registry.task_graph_store.query("in_progress")
                                if task_execution_enabled:
                                    for task in in_progress:
                                        if task.owner == identity["id"]:
                                            self._registry.task_graph_store.complete(task.id)
                                    if did_work_tool:
                                        safety = 0
                                        while safety < 64:
                                            ready_unowned = self._registry.task_graph_store.query_ready_unowned()
                                            if not ready_unowned:
                                                break
                                            task = ready_unowned[0]
                                            self._registry.task_graph_store.claim_task(task.id, identity["id"])
                                            self._registry.task_graph_store.complete(task.id)
                                            safety += 1
                            except Exception:
                                pass
                        return {"type": "final", "content": step["content"], "trace": trace}
                    if step["type"] == "tool":
                        if "content" in action and action["content"]:
                            step["content"] = str(action["content"])
                        name = str(action.get("name", "")).strip()
                        arguments = action.get("arguments") or {}
                        if not isinstance(arguments, dict):
                            arguments = {}
                        step["name"] = name
                        step["arguments"] = arguments
                        if step_callback:
                            step_callback(step, None)
                        if not name or name == "idle":
                            loop_state = "IDLE"
                            iters += 1
                            continue
                        if not self._can_invoke_tool(name, plan_written_this_run, plan_ready_before_run):
                            if name == "todowrite":
                                return self._planning_error(step, trace, step_callback)
                            bootstrap_step = self._bootstrap_todowrite_step(goal)
                            bootstrap = self._invoke_auto_tool(bootstrap_step, trace, step_callback)
                            if bootstrap is None:
                                return {"type": "error", "error": "tool failed", "trace": trace}
                            if not bootstrap.success:
                                return {
                                    "type": "error",
                                    "error": self._format_tool_error(bootstrap.error),
                                    "trace": trace,
                                }
                            plan_written_this_run = True
                            plan_ready_before_run = True
                        if name == "compact":
                            compact = compression.manual_compact(self._transcript_id, trace, str(arguments.get("reason", "")))
                            step_result = {"success": True, "output": compact.to_tool_output(), "error": None}
                            obs = {"result": step_result}
                            trace.append({"step": step, "observation": obs})
                            if step_callback:
                                step_callback(step, obs)
                            trace = compact.replacement_trace
                            observation = {"result": {"success": True, "output": step_result["output"]}}
                            iters += 1
                            continue
                        result = self._invoke_auto_tool(step, trace, step_callback)
                        if result is None:
                            return {"type": "error", "error": "tool failed", "trace": trace}
                        if not result.success:
                            return {
                                "type": "error",
                                "error": self._format_tool_error(result.error),
                                "trace": trace,
                            }
                        if name == "todowrite":
                            plan_written_this_run = True
                            task_execution_enabled = True
                        if (
                            self._registry is not None
                            and name
                            and name
                            not in {
                                "taskgraph_query",
                                "todowrite",
                                "task_claim",
                                "task_start",
                                "task_complete",
                                "compact",
                                "idle",
                            }
                        ):
                            did_work_tool = True
                            if task_execution_enabled:
                                try:
                                    identity = self._current_identity(request)
                                    in_progress = [
                                        t
                                        for t in self._registry.task_graph_store.query("in_progress")
                                        if t.owner == identity["id"]
                                    ]
                                    if in_progress:
                                        self._registry.task_graph_store.complete(in_progress[0].id)
                                    else:
                                        ready_unowned = self._registry.task_graph_store.query_ready_unowned()
                                        if ready_unowned:
                                            task = ready_unowned[0]
                                            self._registry.task_graph_store.claim_task(task.id, identity["id"])
                                            self._registry.task_graph_store.complete(task.id)
                                except Exception:
                                    pass
                                if task_graph_present:
                                    try:
                                        if (
                                            not self._registry.task_graph_store.query("in_progress")
                                            and not self._registry.task_graph_store.query("ready")
                                            and not self._registry.task_graph_store.query("blocked")
                                        ):
                                            trace.append(
                                                {
                                                    "step": {"type": "answer", "content": "全部任务已完成。"},
                                                    "observation": {},
                                                }
                                            )
                                            return {"type": "final", "content": "全部任务已完成。", "trace": trace}
                                    except Exception:
                                        pass
                        observation = {"result": {"success": True, "output": result.output}}
                        iters += 1
                        continue
                    return {"type": "error", "error": "invalid planner action", "trace": trace}
                return {"type": "error", "error": "iteration limit reached", "trace": trace}

            iters = 0
            if isinstance(tool_hint, dict) and tool_hint.get("name"):
                name = str(tool_hint.get("name", "")).strip()
                arguments: Any = tool_hint.get("arguments", {})
                if arguments is None:
                    arguments = {}
                if not isinstance(arguments, dict):
                    arguments = {}
                step = {"type": "tool", "name": name, "arguments": arguments}
                if not self._can_invoke_tool(name, plan_written_this_run, plan_ready_before_run):
                    if name == "todowrite":
                        return self._planning_error(step, trace, step_callback)
                    bootstrap_step = self._bootstrap_todowrite_step(goal)
                    bootstrap = self._invoke_auto_tool(bootstrap_step, trace, step_callback)
                    if bootstrap is None:
                        return {"type": "error", "error": "tool failed", "trace": trace}
                    if not bootstrap.success:
                        return {
                            "type": "error",
                            "error": self._format_tool_error(bootstrap.error),
                            "trace": trace,
                        }
                    plan_written_this_run = True
                    plan_ready_before_run = True
                result = self._invoke_auto_tool(step, trace, step_callback)
                if result is None:
                    return {"type": "error", "error": "tool failed", "trace": trace}
                iters += 1
                if not result.success:
                    return {
                        "type": "error",
                        "error": self._format_tool_error(result.error),
                        "trace": trace,
                    }
                if name == "todowrite":
                    plan_written_this_run = True
                    if iters >= local_max_iters:
                        return {"type": "error", "error": "iteration limit reached", "trace": trace}
                    if not goal:
                        final_step = {"type": "answer", "content": str(result.output)}
                        trace.append({"step": final_step, "observation": {}})
                        return {"type": "final", "content": str(result.output), "trace": trace}
                else:
                    if iters >= local_max_iters:
                        return {"type": "error", "error": "iteration limit reached", "trace": trace}
                    final_step = {"type": "answer", "content": str(result.output)}
                    trace.append({"step": final_step, "observation": {}})
                    return {"type": "final", "content": str(result.output), "trace": trace}

            if goal:
                selected_name, selected_args = self._select_goal_tool(goal, goal_args)
                if selected_name:
                    if selected_args is None:
                        step = {"type": "plan", "reason": "missing arguments", "target": selected_name}
                        trace.append({"step": step, "observation": {}})
                        return {"type": "error", "error": f"missing arguments for {selected_name}", "trace": trace}
                    step = {"type": "tool", "name": selected_name, "arguments": selected_args}
                    if not self._can_invoke_tool(selected_name, plan_written_this_run, plan_ready_before_run):
                        if selected_name == "todowrite":
                            return self._planning_error(step, trace, step_callback)
                        bootstrap_step = self._bootstrap_todowrite_step(goal)
                        bootstrap = self._invoke_auto_tool(bootstrap_step, trace, step_callback)
                        if bootstrap is None:
                            return {"type": "error", "error": "tool failed", "trace": trace}
                        if not bootstrap.success:
                            return {
                                "type": "error",
                                "error": self._format_tool_error(bootstrap.error),
                                "trace": trace,
                            }
                        plan_written_this_run = True
                        plan_ready_before_run = True
                    result = self._invoke_auto_tool(step, trace, step_callback)
                    if result is None:
                        return {"type": "error", "error": "tool failed", "trace": trace}
                    if not result.success:
                        return {
                            "type": "error",
                            "error": self._format_tool_error(result.error),
                            "trace": trace,
                        }
                    iters += 1
                    if iters >= local_max_iters:
                        return {"type": "error", "error": "iteration limit reached", "trace": trace}
                    final_step = {"type": "answer", "content": str(result.output)}
                    trace.append({"step": final_step, "observation": {}})
                    return {"type": "final", "content": str(result.output), "trace": trace}
                return {"type": "error", "error": "no actionable step for goal", "trace": trace}

            attempts = 0
            while attempts < local_max_iters:
                attempts += 1
            return {"type": "error", "error": "iteration limit reached", "trace": trace}

        req_type = str(request.get("type", "")).strip()

        if req_type == "answer":
            content = request.get("content", "")
            return {"type": "final", "content": str(content)}

        if req_type == "tool":
            name = str(request.get("name", "")).strip()
            if not name:
                return {"type": "error", "error": "invalid request"}

            arguments: Any = request.get("arguments", {})
            if arguments is None:
                arguments = {}
            if not isinstance(arguments, dict):
                return {"type": "error", "error": "invalid request"}

            if self._registry is None:
                return {"type": "error", "error": "tool failed"}

            result = self._registry.invoke(ToolCall(name=name, arguments=arguments))
            if result.success:
                return {
                    "type": "tool_result",
                    "name": name,
                    "result": {"success": True, "output": result.output, "error": None},
                }

            return {"type": "error", "error": self._format_tool_error(result.error)}

        return {"type": "error", "error": "unsupported request type"}

    def _has_authoritative_todos(self) -> bool:
        if self._registry is None:
            return False
        return self._registry.task_graph_store.has_valid_graph()

    def _can_invoke_tool(self, name: str, plan_written_this_run: bool, plan_ready_before_run: bool) -> bool:
        if name == "todowrite":
            return True
        return plan_written_this_run or plan_ready_before_run

    def _bootstrap_todowrite_step(self, goal: str) -> dict[str, Any]:
        goal_value = str(goal or "").strip() or "bootstrap"
        todo = {
            "id": f"bootstrap_{uuid4().hex}",
            "content": f"Plan: {goal_value}",
            "status": "pending",
            "blockedBy": [],
            "blocks": [],
        }
        return {"type": "tool", "name": "todowrite", "arguments": {"todos": [todo], "merge": False}}

    def _planning_error(self, step: dict[str, Any], trace: list[dict[str, Any]], step_callback: Any = None) -> dict[str, Any]:
        obs = {"error": self._TODOWRITE_REQUIRED_ERROR}
        trace.append({"step": step, "observation": obs})
        if step_callback:
            step_callback(step, obs)
        return {"type": "error", "error": self._TODOWRITE_REQUIRED_ERROR, "trace": trace}

    def _invoke_auto_tool(self, step: dict[str, Any], trace: list[dict[str, Any]], step_callback: Any = None) -> Any | None:
        if self._registry is None:
            obs = {"error": "no registry"}
            trace.append({"step": step, "observation": obs})
            if step_callback:
                step_callback(step, obs)
            return None
        result = self._registry.invoke(ToolCall(name=str(step["name"]), arguments=dict(step["arguments"])))
        observation = {
            "result": {"success": result.success, "output": result.output, "error": result.error}
        }
        if self._transcript_id:
            ref = self._transcripts.append(self._transcript_id, "tool_result", observation["result"])
            observation["ref"] = ref.to_dict()
        trace.append(
            {
                "step": step,
                "observation": observation,
            }
        )
        if step_callback:
            step_callback(step, observation)
        return result

    def _select_goal_tool(self, goal: str, goal_args: dict[str, Any]) -> tuple[str, dict[str, Any] | None]:
        lowered_goal = goal.lower()
        if any(keyword in lowered_goal for keyword in ["read", "view", "show", "inspect"]):
            if "path" not in goal_args:
                return "read_file", None
            return "read_file", {"path": str(goal_args.get("path", ""))}
        if any(keyword in lowered_goal for keyword in ["write", "create", "save"]):
            if "path" not in goal_args or "content" not in goal_args:
                return "write_file", None
            return "write_file", {"path": str(goal_args["path"]), "content": str(goal_args["content"])}
        if any(keyword in lowered_goal for keyword in ["replace", "edit", "substitute"]):
            if not all(keyword in goal_args for keyword in ["path", "old_string", "new_string"]):
                return "edit_file", None
            return (
                "edit_file",
                {
                    "path": str(goal_args["path"]),
                    "old_string": str(goal_args["old_string"]),
                    "new_string": str(goal_args["new_string"]),
                },
            )
        return "", {}

    def _format_tool_error(self, error: Any) -> str:
        if error is None:
            return "tool failed"
        if isinstance(error, str):
            return error
        if isinstance(error, dict):
            message = error.get("message")
            if isinstance(message, str) and message.strip():
                return message
        return str(error)

    def _merge_observation_with_notifications(self, observation: dict[str, Any], notifications: list[dict[str, Any]]) -> dict[str, Any]:
        if not isinstance(observation, dict):
            observation = {}
        if notifications:
            merged = dict(observation)
            merged["background_notifications"] = notifications
            return merged
        return observation

    def _current_identity(self, request: dict[str, Any]) -> dict[str, Any]:
        teammate_id = str(request.get("teammate_id", "")).strip()
        name = ""
        if not teammate_id and self._registry is not None:
            try:
                roster = self._registry.teammate_manager.list()
                if roster:
                    teammate_id = str(roster[0].get("id", "")).strip() or "system"
                    name = str(roster[0].get("name", "")).strip() or "System"
            except Exception:
                teammate_id = "system"
                name = "System"
        if not teammate_id:
            teammate_id = "system"
        if not name:
            name = "System"
        return {"id": teammate_id, "name": name, "role": "Agent"}
