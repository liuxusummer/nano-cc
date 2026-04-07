from __future__ import annotations

import argparse
import io
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TextIO

from src.cli.render import print_line, render_todo_summary, render_trace_summary, sanitize, style, truncate_text
from src.cli.session_store import SessionRef, SessionStore, SessionSummary
from src.core.task_system import TaskGraphStore


_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")

@dataclass(frozen=True, slots=True)
class CommandSpec:
    name: str
    aliases: tuple[str, ...]
    usage: str
    description: str
    append_space_on_complete: bool = False


@dataclass(frozen=True, slots=True)
class CommandSuggestion:
    name: str
    usage: str
    description: str
    completion: str


def list_command_specs() -> list[CommandSpec]:
    return [
        CommandSpec(
            name="help",
            aliases=("?",),
            usage="/help, /?",
            description="显示帮助",
            append_space_on_complete=False,
        ),
        CommandSpec(
            name="todos",
            aliases=(),
            usage="/todos [--mode <mode>]",
            description="查看 todo 列表",
            append_space_on_complete=True,
        ),
        CommandSpec(
            name="trace",
            aliases=(),
            usage="/trace [--last N]",
            description="查看当前会话最近 N 次 agent 轮次",
            append_space_on_complete=True,
        ),
        CommandSpec(
            name="clear",
            aliases=(),
            usage="/clear",
            description="清空当前界面显示",
            append_space_on_complete=False,
        ),
        CommandSpec(
            name="new-session",
            aliases=(),
            usage="/new-session",
            description="新建并切换到新对话",
            append_space_on_complete=False,
        ),
        CommandSpec(
            name="sessions",
            aliases=(),
            usage="/sessions",
            description="列出历史会话",
            append_space_on_complete=False,
        ),
        CommandSpec(
            name="exit",
            aliases=("quit",),
            usage="/exit, /quit",
            description="退出会话",
            append_space_on_complete=False,
        ),
    ]


def get_command_suggestions(input_text: str) -> list[CommandSuggestion]:
    text = str(input_text or "")
    if not text.startswith("/"):
        return []
    if text.startswith("//"):
        return []
    token_end = len(text)
    for i, ch in enumerate(text):
        if ch.isspace():
            token_end = i
            break
    if token_end != len(text):
        return []
    raw_token = text[1:token_end]
    prefix = raw_token.strip().lower()

    specs = list_command_specs()
    if not prefix:
        selected = specs
    else:
        selected = []
        for spec in specs:
            if spec.name.startswith(prefix):
                selected.append(spec)
                continue
            if any(alias.startswith(prefix) for alias in spec.aliases):
                selected.append(spec)
    suggestions: list[CommandSuggestion] = []
    for spec in selected:
        completion = f"/{spec.name}" + (" " if spec.append_space_on_complete else "")
        suggestions.append(
            CommandSuggestion(
                name=spec.name,
                usage=spec.usage,
                description=spec.description,
                completion=completion,
            )
        )
    return suggestions


def apply_command_completion(input_text: str, suggestion: CommandSuggestion) -> str:
    text = str(input_text or "")
    if not text.startswith("/") or text.startswith("//"):
        return text
    token_end = len(text)
    for i, ch in enumerate(text):
        if ch.isspace():
            token_end = i
            break
    rest = text[token_end:]
    return str(suggestion.completion) + rest


@dataclass(frozen=True, slots=True)
class SlashCommand:
    name: str
    args: list[str]
    raw_args: str
    raw_input: str


@dataclass(frozen=True, slots=True)
class SlashCommandDispatch:
    kind: Literal["not_command", "forward", "handled"]
    forward_text: str | None = None
    should_exit: bool = False
    stdout: str = ""
    stderr: str = ""
    ui_action: Literal["clear", "new_session", "show_sessions"] | None = None
    command_name: str | None = None
    raw_input: str | None = None


@dataclass(frozen=True, slots=True)
class SlashCommandContext:
    root: Path
    session_store: SessionStore
    session: SessionRef
    ui_mode: Literal["plain", "tui", "textual"] = "plain"
    color_out: bool = False
    color_err: bool = False


def _parse_slash_command(text: str) -> SlashCommand | None:
    raw = str(text or "").strip()
    if not raw.startswith("/"):
        return None
    if raw.startswith("//"):
        return None
    if raw.startswith("/?"):
        raw_args = raw[2:].lstrip()
        args = raw_args.split() if raw_args else []
        return SlashCommand(name="help", args=args, raw_args=raw_args, raw_input=raw)
    body = raw[1:]
    if not body:
        return None
    parts = body.split(maxsplit=1)
    name = parts[0].strip().lower()
    if not _NAME_RE.fullmatch(name):
        return None
    raw_args = parts[1] if len(parts) == 2 else ""
    args = raw_args.split() if raw_args else []
    return SlashCommand(name=name, args=args, raw_args=raw_args, raw_input=raw)


def _help_text() -> str:
    specs = list_command_specs()
    return "\n".join(
        [
            "可用命令：",
            "",
            *[f"{spec.usage:<22} {spec.description}" for spec in specs],
            "",
            "转义：输入以 // 开头时，将作为普通输入发送给 agent（去掉一个前导 /）。",
        ]
    )


def _command_entry(*, raw_input: str, command_name: str, stdout_text: str, stderr_text: str, limit: int = 4000) -> dict[str, Any]:
    out, out_trunc = truncate_text(stdout_text, limit=limit)
    err, err_trunc = truncate_text(stderr_text, limit=limit)
    output: dict[str, Any] = {"stdout": out, "stderr": err}
    if out_trunc:
        output["stdout_truncated"] = True
    if err_trunc:
        output["stderr_truncated"] = True
    return {
        "entry_type": "command",
        "raw_input": raw_input,
        "command_name": command_name,
        "command_output": sanitize(output),
    }


def render_session_summaries(
    summaries: list[SessionSummary],
    *,
    current_session_id: str | None,
    stdout: TextIO,
    color_out: bool,
) -> None:
    if not summaries:
        print_line(stdout, style("(无历史会话)", dim=True, enabled=color_out))
        return
    for index, summary in enumerate(summaries, start=1):
        current = " *" if summary.session.session_id == str(current_session_id or "") else ""
        last_input = summary.last_input
        if len(last_input) > 60:
            last_input = last_input[:57] + "..."
        updated = summary.updated_at or "-"
        suffix = f"  last={last_input}" if last_input else ""
        print_line(
            stdout,
            f"{style(f'[{index}]', fg='cyan', bold=True, enabled=color_out)} "
            f"{style(summary.session.session_id, fg='magenta', enabled=color_out)}{current} "
            f"{style(updated, dim=True, enabled=color_out)} "
            f"entries={summary.entry_count}{suffix}",
        )


def render_todos(
    root: Path,
    *,
    mode: str,
    stdout: TextIO,
    stderr: TextIO,
    color_out: bool,
    color_err: bool,
    session_store: SessionStore | None = None,
    session: SessionRef | None = None,
) -> int:
    resolved_store = session_store or SessionStore(root)
    store = TaskGraphStore(root_dir=resolved_store.resolve_task_graph_root(session))
    normalized_mode = str(mode or "all").strip().lower() or "all"
    try:
        if normalized_mode == "all":
            tasks = sorted(store.load_graph().values(), key=lambda item: item.id)
        else:
            tasks = store.query(normalized_mode)
    except Exception as exc:
        print_line(stderr, style(f"todos 读取失败: {exc}", fg="red", bold=True, enabled=color_err))
        return 1

    summary = _summarize_task_graph(store)
    render_todo_summary(stdout, summary, color=color_out)
    if not tasks:
        print_line(stdout, style("(空)", dim=True, enabled=color_out))
        return 0

    for task in tasks:
        status = str(getattr(task, "status", ""))
        owner = getattr(task, "owner", None)
        blocked_by = list(getattr(task, "blockedBy", []) or [])
        suffix = ""
        if owner:
            suffix += f" owner={owner}"
        if blocked_by:
            suffix += f" blockedBy={','.join(blocked_by)}"
        status_text = status
        if status == "completed":
            status_text = style(status, fg="green", enabled=color_out)
        elif status == "in_progress":
            status_text = style(status, fg="yellow", enabled=color_out)
        elif status == "pending":
            status_text = style(status, fg="blue", enabled=color_out)
        task_id = style(task.id, fg="magenta", enabled=color_out)
        print_line(stdout, f"- [{status_text}] {task_id}: {task.content}{suffix}")
    return 0


def render_trace_entries(
    entries: list[dict[str, Any]],
    *,
    stdout: TextIO,
    stderr: TextIO,
    color_out: bool,
    color_err: bool,
    session_id: str | None = None,
    include_commands: bool = True,
    last_n_agent: int | None = None,
) -> int:
    session_label = session_id or ""
    if session_label:
        print_line(stdout, f"{style('session', fg='cyan', bold=True, enabled=color_out)}: {style(session_label, fg='magenta', enabled=color_out)}")

    if not entries:
        print_line(stderr, style("会话为空。", fg="red", bold=True, enabled=color_err))
        return 1

    agent_entries = [e for e in entries if str(e.get("entry_type", "agent")).strip() != "command"]
    if last_n_agent is not None:
        n = int(last_n_agent)
        if n < 1:
            n = 1
        agent_entries = agent_entries[-n:]

    selected: list[dict[str, Any]] = []
    if include_commands and last_n_agent is None:
        selected = entries
    else:
        selected = agent_entries

    if not selected:
        print_line(stdout, style("(无可显示内容)", dim=True, enabled=color_out))
        return 0

    for i, entry in enumerate(selected, start=1):
        ts = str(entry.get("ts", ""))
        entry_type = str(entry.get("entry_type", "agent")).strip() or "agent"
        print_line(stdout, f"\n{style(f'[{i}]', fg='cyan', bold=True, enabled=color_out)} {style(ts, dim=True, enabled=color_out)}")
        if entry_type == "command":
            cmd_name = str(entry.get("command_name", "")).strip() or "<unknown>"
            raw_input = str(entry.get("raw_input", entry.get("user_input", "")))
            output = entry.get("command_output") if isinstance(entry.get("command_output"), dict) else {}
            stdout_text = str(output.get("stdout", "")) if isinstance(output, dict) else ""
            stderr_text = str(output.get("stderr", "")) if isinstance(output, dict) else ""
            print_line(stdout, f"{style('command', fg='cyan', bold=True, enabled=color_out)}: /{cmd_name}")
            if raw_input:
                print_line(stdout, f"{style('input', fg='cyan', bold=True, enabled=color_out)}: {raw_input}")
            if stdout_text:
                for line in stdout_text.splitlines():
                    print_line(stdout, line)
            if stderr_text:
                print_line(stdout, style("stderr:", fg="red", bold=True, enabled=color_out))
                for line in stderr_text.splitlines():
                    print_line(stdout, style(line, fg="red", enabled=color_out))
            continue

        user_input = str(entry.get("user_input", ""))
        resp = entry.get("agent_response") if isinstance(entry.get("agent_response"), dict) else {}
        resp_type = str(resp.get("type", ""))
        print_line(stdout, f"{style('user', fg='cyan', bold=True, enabled=color_out)}: {user_input}")
        if resp_type == "final":
            print_line(stdout, f"{style('final', fg='green', bold=True, enabled=color_out)}: {str(resp.get('content', ''))}")
        elif resp_type == "error":
            print_line(stdout, f"{style('error', fg='red', bold=True, enabled=color_out)}: {str(resp.get('error', ''))}")
        else:
            print_line(stdout, f"{style('result', fg='yellow', bold=True, enabled=color_out)}: {sanitize(resp)}")
        render_trace_summary(stdout, resp.get("trace") if isinstance(resp, dict) else None, color=color_out)
    return 0


def dispatch_slash_command(text: str, ctx: SlashCommandContext) -> SlashCommandDispatch:
    raw = str(text or "").strip()
    if raw.startswith("//"):
        return SlashCommandDispatch(kind="forward", forward_text=raw[1:])
    cmd = _parse_slash_command(raw)
    if cmd is None:
        return SlashCommandDispatch(kind="not_command")

    name = cmd.name
    if name in {"exit", "quit"}:
        return SlashCommandDispatch(kind="handled", should_exit=True, command_name=name, raw_input=cmd.raw_input)
    if name == "help":
        return SlashCommandDispatch(kind="handled", stdout=_help_text() + "\n", command_name=name, raw_input=cmd.raw_input)
    if name == "clear":
        return SlashCommandDispatch(kind="handled", ui_action="clear", command_name=name, raw_input=cmd.raw_input)
    if name == "new-session":
        return SlashCommandDispatch(kind="handled", ui_action="new_session", command_name=name, raw_input=cmd.raw_input)
    if name == "sessions":
        if ctx.ui_mode == "plain":
            out_buf = io.StringIO()
            render_session_summaries(
                ctx.session_store.list_session_summaries(limit=20),
                current_session_id=ctx.session.session_id,
                stdout=out_buf,
                color_out=ctx.color_out,
            )
            return SlashCommandDispatch(kind="handled", stdout=out_buf.getvalue(), command_name=name, raw_input=cmd.raw_input)
        return SlashCommandDispatch(kind="handled", ui_action="show_sessions", command_name=name, raw_input=cmd.raw_input)

    if name == "todos":
        parser = argparse.ArgumentParser(prog="/todos", add_help=False, exit_on_error=False)
        parser.add_argument("--mode", default="all", choices=["all", "ready", "blocked", "in_progress", "completed"])
        try:
            ns = parser.parse_args(cmd.args)
        except Exception as exc:
            return SlashCommandDispatch(
                kind="handled",
                stderr=f"参数错误: {exc}\n{_help_text()}\n",
                command_name=name,
                raw_input=cmd.raw_input,
            )
        out_buf = io.StringIO()
        err_buf = io.StringIO()
        render_todos(
            ctx.root,
            mode=str(getattr(ns, "mode", "all")),
            stdout=out_buf,
            stderr=err_buf,
            color_out=ctx.color_out,
            color_err=ctx.color_err,
            session_store=ctx.session_store,
            session=ctx.session,
        )
        return SlashCommandDispatch(kind="handled", stdout=out_buf.getvalue(), stderr=err_buf.getvalue(), command_name=name, raw_input=cmd.raw_input)

    if name == "trace":
        parser = argparse.ArgumentParser(prog="/trace", add_help=False, exit_on_error=False)
        parser.add_argument("--last", type=int, default=1)
        try:
            ns = parser.parse_args(cmd.args)
        except Exception as exc:
            return SlashCommandDispatch(
                kind="handled",
                stderr=f"参数错误: {exc}\n{_help_text()}\n",
                command_name=name,
                raw_input=cmd.raw_input,
            )
        entries = ctx.session_store.read_entries(ctx.session)
        out_buf = io.StringIO()
        err_buf = io.StringIO()
        render_trace_entries(
            entries,
            stdout=out_buf,
            stderr=err_buf,
            color_out=ctx.color_out,
            color_err=ctx.color_err,
            session_id=ctx.session.session_id,
            include_commands=False,
            last_n_agent=int(getattr(ns, "last", 1)),
        )
        return SlashCommandDispatch(kind="handled", stdout=out_buf.getvalue(), stderr=err_buf.getvalue(), command_name=name, raw_input=cmd.raw_input)

    return SlashCommandDispatch(
        kind="handled",
        stderr=f"未知命令: /{name}\n输入 /help 查看可用命令。\n",
        command_name=name,
        raw_input=cmd.raw_input,
    )


def build_command_session_entry(result: SlashCommandDispatch) -> dict[str, Any]:
    return _command_entry(
        raw_input=str(result.raw_input or ""),
        command_name=str(result.command_name or ""),
        stdout_text=str(result.stdout or ""),
        stderr_text=str(result.stderr or ""),
    )


def _summarize_task_graph(store: TaskGraphStore) -> dict[str, int]:
    try:
        graph = store.load_graph()
    except Exception:
        return {"total": 0, "ready": 0, "blocked": 0, "in_progress": 0, "completed": 0}
    total = 0
    ready = 0
    blocked = 0
    in_progress = 0
    completed = 0
    for item in graph.values():
        total += 1
        status = str(getattr(item, "status", ""))
        blocked_by = list(getattr(item, "blockedBy", []) or [])
        if status == "completed":
            completed += 1
        elif status == "in_progress":
            in_progress += 1
        elif status == "pending":
            if blocked_by:
                blocked += 1
            else:
                ready += 1
    return {"total": total, "ready": ready, "blocked": blocked, "in_progress": in_progress, "completed": completed}
