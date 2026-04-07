from __future__ import annotations

import json
import os
from typing import Any

from openai import OpenAI  # type: ignore

from src.core.model_provider import ModelProvider, PlanAction, PlanState


class OpenAIModelProvider(ModelProvider):
    def __init__(self) -> None:
        base_url = os.getenv("OPENAI_BASE_URL", "").strip()
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        self._model = os.getenv("OPENAI_MODEL", "glm-5")
        self._client = OpenAI(api_key=api_key, base_url=base_url or None)

    def plan_next(self, state: PlanState) -> PlanAction:
        tools_desc = state.get("tools", [])
        observation = state.get("observation", {})
        goal = state.get("goal", "")
        constraints = state.get("constraints", {}) or {}
        todowrite_required = bool(getattr(constraints, "get", lambda *_: False)("todowrite_required"))

        schema_hint = (
            "Return a single JSON object with keys: "
            "type ('answer'|'tool'), "
            "content (when type is 'answer'), "
            "name and arguments (when type is 'tool'). "
            "Only output the JSON, with no extra text. "
            "When type is 'answer', content MUST be a non-empty string. "
            "If you call tool 'todowrite', each todo.status MUST be one of "
            "'pending'|'in_progress'|'completed' (do NOT use 'ready')."
        )
        if todowrite_required:
            schema_hint = (
                f"{schema_hint} "
                "IMPORTANT: constraints.todowrite_required=true, so the next action MUST be a tool call to 'todowrite' "
                "with arguments {'todos':[{'id':str,'content':str,'status':'pending'|'in_progress'|'completed','blockedBy':[],'blocks':[]}], 'merge': false}."
            )

        system = (
            "You are an agent planner. Choose the next action as JSON. "
            "Available tools are listed with their names and parameters. "
            "Prefer a single, safe step."
        )
        if todowrite_required:
            system = (
                f"{system} "
                "When constraints.todowrite_required is true, do not call any other tool first. "
                "Return a todowrite tool action that writes a minimal todo plan."
            )

        user = json.dumps(
            {
                "goal": goal,
                "available_tools": tools_desc,
                "last_observation": observation,
                "constraints": constraints,
                "instruction": schema_hint,
            },
            ensure_ascii=False,
        )

        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0,
                response_format={"type": "json_object"},
            )
            content = (resp.choices[0].message.content or "").strip()
            data = json.loads(content)
        except Exception:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0,
            )
            content = (resp.choices[0].message.content or "").strip()
            try:
                start = content.find("{")
                end = content.rfind("}")
                data = json.loads(content[start : end + 1])
            except Exception:
                data = {"type": "answer", "content": "模型输出解析失败，请重试。"}

        action: PlanAction = {
            "type": str(data.get("type", "")),
        }
        if action["type"] == "answer":
            content = str(data.get("content", ""))
            action["content"] = content if content.strip() else "模型输出为空，请重试。"
        elif action["type"] == "tool":
            action["name"] = str(data.get("name", ""))
            args = data.get("arguments") or {}
            action["arguments"] = self._normalize_tool_arguments(
                name=action["name"],
                arguments=args if isinstance(args, dict) else {},
                goal=goal,
            )

        return action

    def _normalize_tool_arguments(self, *, name: str, arguments: dict[str, Any], goal: Any) -> dict[str, Any]:
        normalized = dict(arguments)
        tool_name = str(name or "").strip()
        goal_text = str(goal or "").strip()

        if tool_name == "web_search":
            query = normalized.get("query")
            if (not isinstance(query, str) or not query.strip()) and goal_text:
                normalized["query"] = goal_text

        return normalized
