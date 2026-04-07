from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, TextIO

from src.cli.env_file import load_env_file
from src.cli.render import format_json, print_line, render_todo_summary, render_trace_summary, sanitize, style
from src.cli.slash_commands import SlashCommandContext, build_command_session_entry, dispatch_slash_command, render_todos, render_trace_entries
from src.cli.session_store import SessionRef, SessionStore
from src.core.agent_loop import AgentLoop
from src.core.task_system import TaskGraphStore
from src.core.tool_registry import ToolRegistry


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
    return {
        "total": total,
        "ready": ready,
        "blocked": blocked,
        "in_progress": in_progress,
        "completed": completed,
    }


def _list_task_graph_items(store: TaskGraphStore) -> list[dict[str, Any]]:
    try:
        graph = store.load_graph()
    except Exception:
        return []
    return [item.to_dict() for item in sorted(graph.values(), key=lambda item: item.id)]


def _require_openai_env(stderr: TextIO) -> bool:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if api_key:
        return True
    print_line(stderr, "缺少 OPENAI_API_KEY，无法在 strategy=llm 下启动。")
    print_line(stderr, "请设置环境变量 OPENAI_API_KEY（可选 OPENAI_BASE_URL / OPENAI_MODEL），或在项目根目录放置 .env 文件。")
    return False


def _is_tty(stream: TextIO) -> bool:
    isatty = getattr(stream, "isatty", None)
    if callable(isatty):
        try:
            return bool(isatty())
        except Exception:
            return False
    return False


def _tui_available(stdin: TextIO, stdout: TextIO) -> bool:
    if not (_is_tty(stdin) and _is_tty(stdout)):
        return False
    try:
        __import__("curses")
    except Exception:
        return False
    return True


def _textual_available(stdin: TextIO, stdout: TextIO) -> bool:
    if not (_is_tty(stdin) and _is_tty(stdout)):
        return False
    try:
        __import__("textual")
    except Exception:
        return False
    return True


def _clear_plain_screen(stdout: TextIO, *, color_out: bool, session_id: str) -> None:
    if _is_tty(stdout):
        stdout.write("\x1b[2J\x1b[H")
    print_line(stdout, f"{style('session', fg='cyan', bold=True, enabled=color_out)}: {style(session_id, fg='magenta', enabled=color_out)}")
    print_line(stdout, style("输入内容后回车发送；输入 exit/quit 退出。", dim=True, enabled=color_out))
    stdout.flush()


def _color_enabled(choice: str, stream: TextIO) -> bool:
    normalized = str(choice or "auto").strip().lower()
    if normalized == "never":
        return False
    if normalized == "always":
        return True
    if os.getenv("NO_COLOR") is not None:
        return False
    if os.getenv("FORCE_COLOR", "").strip():
        return True
    isatty = getattr(stream, "isatty", None)
    if callable(isatty):
        try:
            return bool(isatty())
        except Exception:
            return False
    return False


def _resolve_root_dir(raw_root: str) -> Path:
    raw = str(raw_root or ".").strip() or "."
    candidate = Path(raw).expanduser()
    root = candidate if candidate.is_absolute() else (Path.cwd() / candidate)
    root = root.resolve()
    if root.exists() and not root.is_dir():
        raise ValueError(f"--root 指向的路径不是目录: {root}")
    root.mkdir(parents=True, exist_ok=True)
    return root


def _ensure_within_root(root: Path, target: Path, *, label: str) -> Path:
    root_resolved = root.resolve()
    target_resolved = target.resolve(strict=False)
    try:
        target_resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise ValueError(f"{label} 路径越界，禁止逃逸 root: {target}") from exc
    return target_resolved


def _state_dir(root: Path) -> Path:
    return root / ".agent"


def _task_graph_store_for_session(
    session_store: SessionStore,
    session: SessionRef | None,
    *,
    allow_legacy_fallback: bool = False,
) -> TaskGraphStore:
    return TaskGraphStore(root_dir=session_store.resolve_task_graph_root(session, allow_legacy_fallback=allow_legacy_fallback))


def _migrate_state_dirs(root: Path) -> None:
    old = root / ".trae"
    new = root / ".agent"
    if not old.exists() or not old.is_dir():
        return
    if new.exists():
        return
    try:
        old.rename(new)
        return
    except Exception:
        pass
    try:
        new.mkdir(parents=True, exist_ok=True)
    except Exception:
        return
    for child in ("cli", "task_graph"):
        src = old / child
        dst = new / child
        if not src.exists() or dst.exists():
            continue
        try:
            src.rename(dst)
        except Exception:
            continue
    try:
        if old.exists() and old.is_dir() and not any(old.iterdir()):
            old.rmdir()
    except Exception:
        pass


def _build_loop(root: Path, *, mode: str, strategy: str, task_graph_store: TaskGraphStore | None = None) -> AgentLoop:
    store = task_graph_store or TaskGraphStore(root_dir=_state_dir(root) / "task_graph")
    if mode != "auto" or strategy != "llm":
        registry = ToolRegistry.with_builtin_tools(task_graph_store=store)
        return AgentLoop(registry)
    from src.integrations.openai_client import OpenAIModelProvider

    provider = OpenAIModelProvider()
    registry = ToolRegistry.with_builtin_tools(task_graph_store=store, model_provider=provider)
    return AgentLoop(registry, model_provider=provider)


def _cmd_todos(root: Path, args: argparse.Namespace, stdout: TextIO, stderr: TextIO, *, color_out: bool, color_err: bool) -> int:
    mode = str(getattr(args, "mode", "all")).strip().lower()
    session_store = SessionStore(root)
    session = None
    session_id = getattr(args, "session", None)
    if session_id:
        session = session_store.get_session(str(session_id))
        if session is None:
            print_line(
                stderr,
                style(f"未找到会话: {session_id}", fg="red", bold=True, enabled=color_err),
            )
            return 1
    else:
        session = session_store.find_last_session()
        if session is None:
            print_line(
                stderr,
                style("未找到会话。先运行 `start` 产生会话日志，或使用 `--session <id>` 指定。", fg="red", bold=True, enabled=color_err),
            )
            return 1
    return render_todos(
        root,
        mode=mode,
        stdout=stdout,
        stderr=stderr,
        color_out=color_out,
        color_err=color_err,
        session_store=session_store,
        session=session,
    )


def _cmd_trace(root: Path, args: argparse.Namespace, stdout: TextIO, stderr: TextIO, *, color_out: bool, color_err: bool) -> int:
    store = SessionStore(root)
    session = None
    session_id = getattr(args, "session", None)
    last = bool(getattr(args, "last", False))
    if session_id:
        session = store.get_session(str(session_id))
    elif last or True:
        session = store.find_last_session()

    if session is None:
        print_line(
            stderr,
            style("未找到会话。先运行 `start` 产生会话日志，或使用 `--session <id>` 指定。", fg="red", bold=True, enabled=color_err),
        )
        return 1

    entries = store.read_entries(session)
    return render_trace_entries(
        entries,
        stdout=stdout,
        stderr=stderr,
        color_out=color_out,
        color_err=color_err,
        session_id=session.session_id,
        include_commands=True,
        last_n_agent=None,
    )


def _cmd_start(root: Path, args: argparse.Namespace, stdin: TextIO, stdout: TextIO, stderr: TextIO, *, color_out: bool, color_err: bool) -> int:
    mode = str(getattr(args, "mode", "auto")).strip() or "auto"
    strategy = str(getattr(args, "strategy", "llm")).strip()
    if strategy == "none":
        strategy = ""
    max_iters = int(getattr(args, "max_iters", 20))
    if max_iters < 1:
        max_iters = 1
    ui = str(getattr(args, "ui", "auto")).strip().lower() or "auto"

    if mode == "auto" and strategy == "llm" and not _require_openai_env(stderr):
        return 2

    session_store = SessionStore(root)

    def build_session_loop(current_session: SessionRef) -> AgentLoop:
        return _build_loop(
            root,
            mode=mode,
            strategy=strategy,
            task_graph_store=_task_graph_store_for_session(session_store, current_session),
        )

    session = session_store.new_session()
    loop = build_session_loop(session)
    slash_ctx = SlashCommandContext(root=root, session_store=session_store, session=session, ui_mode="plain", color_out=color_out, color_err=color_err)
    if ui == "auto":
        if _textual_available(stdin, stdout):
            ui = "textual"
        elif _tui_available(stdin, stdout):
            ui = "tui"
        else:
            ui = "plain"

    if ui == "textual":
        if not _textual_available(stdin, stdout):
            print_line(
                stderr,
                style(
                    "Textual UI 需要在 TTY 终端运行（stdin/stdout 必须是 TTY，且 textual 可用）。",
                    fg="red",
                    bold=True,
                    enabled=color_err,
                ),
            )
            return 2
        from src.cli.textual_ui import run_textual_ui

        return run_textual_ui(
            root=root,
            loop=loop,
            build_loop_for_session=build_session_loop,
            session_store=session_store,
            session=session,
            mode=mode,
            strategy=strategy,
            max_iters=max_iters,
        )

    if ui == "tui":
        if not _tui_available(stdin, stdout):
            print_line(
                stderr,
                style(
                    "TUI 需要在 TTY 终端运行（stdin/stdout 必须是 TTY，且 curses 可用）。",
                    fg="red",
                    bold=True,
                    enabled=color_err,
                ),
            )
            return 2
        from src.cli.tui import run_tui

        return run_tui(
            root=root,
            loop=loop,
            build_loop_for_session=build_session_loop,
            session_store=session_store,
            session=session,
            mode=mode,
            strategy=strategy,
            max_iters=max_iters,
        )

    print_line(stdout, f"{style('session', fg='cyan', bold=True, enabled=color_out)}: {style(session.session_id, fg='magenta', enabled=color_out)}")
    print_line(stdout, style("输入内容后回车发送；输入 exit/quit 退出。", dim=True, enabled=color_out))

    round_idx = 0
    while True:
        try:
            stdout.write(style("> ", fg="cyan", bold=True, enabled=color_out))
            stdout.flush()
            line = stdin.readline()
        except KeyboardInterrupt:
            print_line(stdout, style("\n(中断)", fg="yellow", bold=True, enabled=color_out))
            break
        if not line:
            break
        text = line.strip()
        if not text:
            continue
        if text in {"exit", "quit", ":q"}:
            break
        dispatch = dispatch_slash_command(text, slash_ctx)
        if dispatch.kind == "forward":
            text = str(dispatch.forward_text or "").strip()
            if not text:
                continue
        elif dispatch.kind == "handled":
            if dispatch.ui_action == "clear":
                _clear_plain_screen(stdout, color_out=color_out, session_id=session.session_id)
            elif dispatch.ui_action == "new_session":
                session_store.append_round(session, build_command_session_entry(dispatch))
                session = session_store.new_session()
                loop = build_session_loop(session)
                slash_ctx = SlashCommandContext(root=root, session_store=session_store, session=session, ui_mode="plain", color_out=color_out, color_err=color_err)
                _clear_plain_screen(stdout, color_out=color_out, session_id=session.session_id)
                if dispatch.should_exit:
                    break
                continue
            if dispatch.stdout:
                stdout.write(dispatch.stdout)
            if dispatch.stderr:
                stderr.write(dispatch.stderr)
            session_store.append_round(session, build_command_session_entry(dispatch))
            if dispatch.should_exit:
                break
            continue

        round_idx += 1
        request: dict[str, Any] = {"mode": mode, "strategy": strategy, "goal": text, "max_iters": max_iters}
        resp = loop.run(request, max_iters=max_iters)
        try:
            store = _task_graph_store_for_session(session_store, session)
            summary = _summarize_task_graph(store)
            task_items = _list_task_graph_items(store)
        except Exception:
            summary = None
            task_items = []

        resp_type = str(resp.get("type"))
        if resp_type == "final":
            type_text = style(resp_type, fg="green", bold=True, enabled=color_out)
        elif resp_type == "error":
            type_text = style(resp_type, fg="red", bold=True, enabled=color_out)
        else:
            type_text = style(resp_type, fg="yellow", bold=True, enabled=color_out)
        print_line(stdout, f"\n{style(f'[{round_idx}]', fg='cyan', bold=True, enabled=color_out)} type={type_text}")
        if resp.get("type") == "final":
            print_line(stdout, str(resp.get("content", "")))
        elif resp.get("type") == "error":
            print_line(stdout, style(str(resp.get("error", "")), fg="red", enabled=color_out))
        else:
            print_line(stdout, format_json(resp, limit=4000))
        render_todo_summary(
            stdout,
            summary if isinstance(summary, dict) else None,
            items=task_items,
            color=color_out,
        )
        render_trace_summary(stdout, resp.get("trace") if isinstance(resp, dict) else None, color=color_out)

        session_store.append_round(
            session,
            {
                "entry_type": "agent",
                "user_input": text,
                "agent_response": sanitize(resp),
                "mode": mode,
                "strategy": strategy,
                "max_iters": max_iters,
            },
        )

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="src.cli")
    parser.add_argument("--root", default=".", help="项目根目录（默认当前目录）")
    parser.add_argument("--color", default="auto", choices=["auto", "always", "never"])
    parser.add_argument("--env-file", default=".env", help="加载环境变量文件（相对 root 或绝对路径）")
    parser.add_argument("--env-override", action="store_true", help="覆盖已存在的环境变量")
    sub = parser.add_subparsers(dest="command", required=True)

    start = sub.add_parser("start", help="启动交互式会话（REPL）")
    start.add_argument("--mode", default="auto", choices=["auto", "manual"])
    start.add_argument("--strategy", default="llm", choices=["llm", "none"])
    start.add_argument("--max-iters", type=int, default=20)
    start.add_argument("--ui", default="auto", choices=["auto", "textual", "tui", "plain"], help="界面模式：auto/textual/tui/plain（默认 auto）")

    todos = sub.add_parser("todos", help="查看 todo 列表")
    todos.add_argument("--mode", default="all", choices=["all", "ready", "blocked", "in_progress", "completed"])
    todos.add_argument("--session", default=None, help="指定会话 ID；默认最近会话")

    trace = sub.add_parser("trace", help="查看会话轨迹")
    trace.add_argument("--last", action="store_true", help="查看最近会话（默认）")
    trace.add_argument("--session", help="指定会话 id")

    return parser


def main(argv: list[str] | None = None, *, stdin: TextIO | None = None, stdout: TextIO | None = None, stderr: TextIO | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    stdin = sys.stdin if stdin is None else stdin
    stdout = sys.stdout if stdout is None else stdout
    stderr = sys.stderr if stderr is None else stderr

    parser = build_parser()
    args = parser.parse_args(argv)
    color_choice = str(getattr(args, "color", "auto"))
    color_out = _color_enabled(color_choice, stdout)
    color_err = _color_enabled(color_choice, stderr)

    try:
        root = _resolve_root_dir(str(getattr(args, "root", ".") or "."))
    except ValueError as exc:
        print_line(stderr, style(str(exc), fg="red", bold=True, enabled=color_err))
        return 2
    _migrate_state_dirs(root)

    env_file = str(getattr(args, "env_file", ".env") or ".env").strip() or ".env"
    override_env = bool(getattr(args, "env_override", False))
    env_candidate = Path(env_file).expanduser()
    if env_candidate.is_absolute():
        env_path = env_candidate
    else:
        try:
            env_path = _ensure_within_root(root, root / env_candidate, label="--env-file")
        except ValueError as exc:
            print_line(stderr, style(str(exc), fg="red", bold=True, enabled=color_err))
            return 2
    load_env_file(env_path, override=override_env)

    command = str(getattr(args, "command", "") or "").strip()
    if command == "start":
        return _cmd_start(root, args, stdin, stdout, stderr, color_out=color_out, color_err=color_err)
    if command == "todos":
        return _cmd_todos(root, args, stdout, stderr, color_out=color_out, color_err=color_err)
    if command == "trace":
        return _cmd_trace(root, args, stdout, stderr, color_out=color_out, color_err=color_err)

    print_line(stderr, style(f"unknown command: {command}", fg="red", bold=True, enabled=color_err))
    return 2
