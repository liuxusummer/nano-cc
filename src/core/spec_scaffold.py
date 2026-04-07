from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .models import ToolResult


def _validate_change_id(change_id: str) -> str:
    normalized = change_id.strip()
    if not normalized:
        raise ValueError("change_id must be a non-empty string")
    if "/" in normalized or "\\" in normalized or ".." in normalized:
        raise ValueError("change_id contains invalid characters")
    return normalized


def _render_spec_md(feature_name: str) -> str:
    title = feature_name.strip() or "Feature"
    return "\n".join(
        [
            f"# {title} Spec",
            "",
            "## Why",
            "",
            "## Skills / 临时知识",
            "- 需要的知识：",
            "  - 无",
            "- 需要加载的 skills：",
            "  - 无",
            "- 加载时机：",
            "  - 无",
            "- 注入方式：",
            "  - 仅通过 `skill_load` 的 tool_result/observation 进入后续推理；不得写入 system prompt",
            "- 降级策略：",
            "  - 若 skill 不存在：改用检索工具定位资料，或先补充 `doc/skills/<name>.md`",
            "",
            "## What Changes",
            "",
            "## Impact",
            "",
            "## ADDED Requirements",
            "",
            "## MODIFIED Requirements",
            "",
            "## REMOVED Requirements",
            "",
        ]
    )


def _render_tasks_md() -> str:
    return "\n".join(
        [
            "# Tasks",
            "- [ ] Task 1: <fill me>",
            "",
            "# Task Dependencies",
            "",
        ]
    )


def _render_checklist_md() -> str:
    return "\n".join(
        [
            "- [ ] spec.md 包含 `## Skills / 临时知识` 段落，且无依赖时明确写“无”",
            "- [ ] 若引用 skill 内容，走 `skill_load` tool_result/observation 注入，不修改 system prompt",
        ]
    )


@dataclass(frozen=True, slots=True)
class SpecScaffoldResult:
    change_id: str
    root_dir: str
    files: dict[str, str]

    def to_tool_output(self) -> dict[str, object]:
        return {"tool": "spec_scaffold", "change_id": self.change_id, "root_dir": self.root_dir, "files": self.files}


def scaffold_spec(change_id: str, feature_name: str | None = None, *, force: bool = False) -> ToolResult:
    try:
        normalized = _validate_change_id(change_id)
    except ValueError as exc:
        return ToolResult(
            success=False,
            error=str(exc),
            output={"code": "invalid_change_id", "message": str(exc), "details": {"change_id": change_id}},
        )

    base_dir = Path(".trae/specs")
    target = base_dir / normalized
    if target.exists() and not force:
        return ToolResult(
            success=False,
            error="spec directory already exists",
            output={
                "code": "spec_exists",
                "message": "spec directory already exists",
                "details": {"path": str(target)},
            },
        )

    target.mkdir(parents=True, exist_ok=True)
    spec_path = target / "spec.md"
    tasks_path = target / "tasks.md"
    checklist_path = target / "checklist.md"

    spec_path.write_text(_render_spec_md(feature_name or normalized), encoding="utf-8")
    tasks_path.write_text(_render_tasks_md(), encoding="utf-8")
    checklist_path.write_text(_render_checklist_md(), encoding="utf-8")

    result = SpecScaffoldResult(
        change_id=normalized,
        root_dir=str(target),
        files={"spec.md": str(spec_path), "tasks.md": str(tasks_path), "checklist.md": str(checklist_path)},
    )
    return ToolResult(success=True, output=result.to_tool_output())
