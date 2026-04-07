from __future__ import annotations

from typing import Protocol, Any, TypedDict


class PlanState(TypedDict, total=False):
    goal: str
    tools: list[dict[str, Any]]
    observation: dict[str, Any]
    trace: list[dict[str, Any]]
    constraints: dict[str, Any]


class PlanAction(TypedDict, total=False):
    type: str
    content: str
    name: str
    arguments: dict[str, Any]


class ModelProvider(Protocol):
    def plan_next(self, state: PlanState) -> PlanAction:
        ...

