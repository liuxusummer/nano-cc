from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any, TextIO


_ANSI_RESET = "\x1b[0m"
_ANSI_BOLD = "\x1b[1m"
_ANSI_DIM = "\x1b[2m"
_ANSI_RED = "\x1b[31m"
_ANSI_GREEN = "\x1b[32m"
_ANSI_YELLOW = "\x1b[33m"
_ANSI_BLUE = "\x1b[34m"
_ANSI_MAGENTA = "\x1b[35m"
_ANSI_CYAN = "\x1b[36m"

_SENSITIVE_KEYS = {
    "openai_api_key",
    "api_key",
    "authorization",
    "access_token",
    "refresh_token",
    "token",
    "password",
    "secret",
}


def sanitize(value: Any) -> Any:
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for k, v in value.items():
            key = str(k)
            if key.strip().lower() in _SENSITIVE_KEYS:
                out[key] = "***"
            else:
                out[key] = sanitize(v)
        return out
    if isinstance(value, (str, bytes)):
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [sanitize(v) for v in value]
    return value


def style(text: str, *, fg: str | None = None, bold: bool = False, dim: bool = False, enabled: bool = False) -> str:
    if not enabled:
        return text
    parts: list[str] = []
    if bold:
        parts.append(_ANSI_BOLD)
    if dim:
        parts.append(_ANSI_DIM)
    if fg == "red":
        parts.append(_ANSI_RED)
    elif fg == "green":
        parts.append(_ANSI_GREEN)
    elif fg == "yellow":
        parts.append(_ANSI_YELLOW)
    elif fg == "blue":
        parts.append(_ANSI_BLUE)
    elif fg == "magenta":
        parts.append(_ANSI_MAGENTA)
    elif fg == "cyan":
        parts.append(_ANSI_CYAN)
    if not parts:
        return text
    return "".join(parts) + text + _ANSI_RESET


def truncate_text(text: str, limit: int) -> tuple[str, bool]:
    if limit <= 0:
        return "", True
    if len(text) <= limit:
        return text, False
    return text[:limit], True


def format_json(value: Any, *, limit: int = 4000) -> str:
    sanitized = sanitize(value)
    text = json.dumps(sanitized, ensure_ascii=False, indent=2, sort_keys=False)
    truncated, was_truncated = truncate_text(text, limit=limit)
    if was_truncated:
        return f"{truncated}\n...[已截断]"
    return truncated


def print_line(out: TextIO, text: str = "") -> None:
    out.write(text)
    out.write("\n")


def render_todo_summary(
    out: TextIO,
    summary: Mapping[str, Any] | None,
    *,
    items: Sequence[Mapping[str, Any]] | None = None,
    color: bool = False,
) -> None:
    if not summary:
        return
    total = summary.get("total", 0)
    ready = summary.get("ready", 0)
    blocked = summary.get("blocked", 0)
    in_progress = summary.get("in_progress", 0)
    completed = summary.get("completed", 0)
    label = style("todo", fg="cyan", bold=True, enabled=color)
    print_line(
        out,
        f"{label}: total={total} ready={ready} blocked={blocked} in_progress={in_progress} completed={completed}",
    )
    if not items:
        return
    visible: list[Mapping[str, Any]] = []
    inprog: list[Mapping[str, Any]] = []
    for item in items:
        status = str(item.get("status", "")).strip()
        content = str(item.get("content", "")).strip()
        if not content:
            continue
        if status == "completed":
            continue
        if status == "in_progress":
            inprog.append(item)
        else:
            visible.append(item)
    # 优先输出 in_progress，高亮
    for item in inprog:
        content = str(item.get("content", "")).strip()
        status_text = style("in_progress", fg="yellow", bold=True, enabled=color)
        bullet = style(">", fg="yellow", bold=True, enabled=color)
        print_line(out, f"{bullet} [{status_text}] {content}")
    # 其余未完成项（pending/blocked）
    max_items = 5
    display = visible[:max_items]
    for item in display:
        status = str(item.get("status", "")).strip()
        content = str(item.get("content", "")).strip()
        if status == "pending":
            status_text = style("pending", fg="blue", enabled=color)
        elif status == "blocked":
            status_text = style("blocked", fg="red", enabled=color)
        else:
            status_text = status
        print_line(out, f"- [{status_text}] {content}")
    remaining = len(visible) - len(display)
    if remaining > 0:
        print_line(out, style(f"... 还有 {remaining} 项未显示", dim=True, enabled=color))


def iter_tool_steps(trace: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    if not trace:
        return []
    steps: list[dict[str, Any]] = []
    for item in trace:
        if not isinstance(item, dict):
            continue
        step = item.get("step")
        if not isinstance(step, dict):
            continue
        if str(step.get("type", "")).strip() != "tool":
            continue
        steps.append(item)
    return steps


def render_trace_summary(out: TextIO, trace: list[dict[str, Any]] | None, *, limit: int = 2000, color: bool = False) -> None:
    tool_items = iter_tool_steps(trace)
    if not tool_items:
        return
    print_line(out, style("tools:", fg="cyan", bold=True, enabled=color))
    for item in tool_items:
        step = item.get("step") or {}
        observation = item.get("observation") or {}
        name = str(step.get("name", "")).strip() or "<unknown>"
        result = observation.get("result") if isinstance(observation, dict) else None
        success = None
        output_preview = ""
        if isinstance(result, dict):
            success = result.get("success")
            output_preview = format_json(result.get("output"), limit=limit)
        if success is None:
            success_text = style("?", fg="yellow", enabled=color)
        elif bool(success):
            success_text = style("ok", fg="green", bold=True, enabled=color)
        else:
            success_text = style("fail", fg="red", bold=True, enabled=color)
        tool_name = style(name, fg="magenta", enabled=color)
        print_line(out, f"- {tool_name}: {success_text}")
        if output_preview:
            for line in output_preview.splitlines():
                print_line(out, style(f"  {line}", dim=True, enabled=color))
