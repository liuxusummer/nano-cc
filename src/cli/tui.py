from __future__ import annotations

import curses
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.cli.render import iter_tool_steps, sanitize
from src.cli.slash_commands import CommandSuggestion, SlashCommandContext, apply_command_completion, build_command_session_entry, dispatch_slash_command, get_command_suggestions
from src.cli.session_store import SessionRef, SessionStore, SessionSummary
from src.core.agent_loop import AgentLoop
from src.core.task_system import TaskGraphStore


@dataclass(slots=True)
class _UiState:
    input_buf: str
    chat_lines: list[str]
    trace_lines: list[str]
    round_idx: int
    command_suggestions: list[CommandSuggestion]
    command_selected_idx: int
    command_suppressed: bool
    command_last_input: str
    session_picker_active: bool
    session_picker_entries: list[SessionSummary]
    session_picker_selected_idx: int


def _wrap_lines(text: str, width: int) -> list[str]:
    width = max(int(width), 1)
    lines: list[str] = []
    for raw in str(text).splitlines() or [""]:
        wrapped = textwrap.wrap(raw, width=width, replace_whitespace=False, drop_whitespace=False)
        lines.extend(wrapped or [""])
    return lines


def _draw_box(stdscr, y: int, x: int, h: int, w: int, title: str) -> None:
    if h < 2 or w < 2:
        return
    try:
        stdscr.addstr(y, x, "+" + "-" * (w - 2) + "+")
        for row in range(1, h - 1):
            stdscr.addstr(y + row, x, "|")
            stdscr.addstr(y + row, x + w - 1, "|")
        stdscr.addstr(y + h - 1, x, "+" + "-" * (w - 2) + "+")
        if title and w >= 4:
            label = f" {title} "
            stdscr.addstr(y, x + 2, label[: max(w - 4, 0)])
    except curses.error:
        return


def _load_tasks(session_store: SessionStore, session: SessionRef) -> tuple[dict[str, int], list[dict[str, Any]]]:
    store = TaskGraphStore(root_dir=session_store.resolve_task_graph_root(session))
    try:
        graph = store.load_graph()
    except Exception:
        return {"total": 0, "ready": 0, "blocked": 0, "in_progress": 0, "completed": 0}, []

    total = 0
    ready = 0
    blocked = 0
    in_progress = 0
    completed = 0
    items: list[dict[str, Any]] = []
    for item in sorted(graph.values(), key=lambda t: t.id):
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
        items.append(item.to_dict())
    return {"total": total, "ready": ready, "blocked": blocked, "in_progress": in_progress, "completed": completed}, items


def _format_todo_lines(items: list[dict[str, Any]], max_lines: int) -> list[tuple[str, int]]:
    if max_lines <= 0:
        return []
    inprog: list[str] = []
    other: list[str] = []
    for item in items:
        status = str(item.get("status", "")).strip()
        content = str(item.get("content", "")).strip()
        if not content or status == "completed":
            continue
        if status == "in_progress":
            inprog.append(content)
        else:
            other.append(f"[{status}] {content}")
    lines: list[tuple[str, int]] = []
    for content in inprog:
        lines.append((f"> [in_progress] {content}", 1))
    visible = other[: max(0, max_lines - len(lines))]
    for line in visible:
        lines.append((line, 0))
    remaining = len(other) - len(visible)
    if remaining > 0 and len(lines) < max_lines:
        lines.append((f"... 还有 {remaining} 项未显示", 2))
    return lines[:max_lines]


def _format_trace_lines(resp: dict[str, Any], width: int, max_lines: int) -> list[str]:
    if max_lines <= 0:
        return []
    trace = resp.get("trace") if isinstance(resp, dict) else None
    items = iter_tool_steps(trace if isinstance(trace, list) else None)
    lines: list[str] = []
    for item in items:
        step = item.get("step") if isinstance(item, dict) else {}
        obs = item.get("observation") if isinstance(item, dict) else {}
        name = str((step or {}).get("name", "")).strip() or "<unknown>"
        result = (obs or {}).get("result") if isinstance(obs, dict) else None
        success = None
        if isinstance(result, dict):
            success = result.get("success")
        tag = "?" if success is None else ("ok" if bool(success) else "fail")
        lines.append(f"{name}: {tag}")
        if len(lines) >= max_lines:
            break
    return [line[: max(width, 1)] for line in lines]


def run_tui(
    *,
    root: Path,
    loop: AgentLoop,
    build_loop_for_session,
    session_store: SessionStore,
    session: SessionRef,
    mode: str,
    strategy: str,
    max_iters: int,
) -> int:
    state = _UiState(
        input_buf="",
        chat_lines=[],
        trace_lines=[],
        round_idx=0,
        command_suggestions=[],
        command_selected_idx=0,
        command_suppressed=False,
        command_last_input="",
        session_picker_active=False,
        session_picker_entries=[],
        session_picker_selected_idx=0,
    )
    slash_ctx = SlashCommandContext(root=root, session_store=session_store, session=session, ui_mode="tui", color_out=False, color_err=False)

    def _clear_session_view() -> None:
        state.input_buf = ""
        state.chat_lines = []
        state.trace_lines = []
        state.command_suggestions = []
        state.command_selected_idx = 0
        state.command_suppressed = False
        state.command_last_input = ""
        state.session_picker_active = False
        state.session_picker_entries = []
        state.session_picker_selected_idx = 0

    def _chat_lines_from_entries(entries: list[dict[str, Any]]) -> list[str]:
        lines: list[str] = []
        for entry in entries:
            entry_type = str(entry.get("entry_type", "agent")).strip() or "agent"
            if entry_type == "command":
                raw_input = str(entry.get("raw_input", "")).strip()
                output = entry.get("command_output") if isinstance(entry.get("command_output"), dict) else {}
                stdout_text = str(output.get("stdout", "")) if isinstance(output, dict) else ""
                stderr_text = str(output.get("stderr", "")) if isinstance(output, dict) else ""
                if raw_input:
                    lines.append(f"cmd: {raw_input}")
                if stdout_text:
                    lines.extend(stdout_text.splitlines())
                if stderr_text:
                    lines.append("stderr:")
                    lines.extend(stderr_text.splitlines())
                continue
            user_input = str(entry.get("user_input", "")).strip()
            if user_input:
                lines.append(f"user: {user_input}")
            resp = entry.get("agent_response") if isinstance(entry.get("agent_response"), dict) else {}
            resp_type = str(resp.get("type", "")).strip()
            if resp_type == "final":
                lines.append(f"final: {str(resp.get('content', ''))}")
            elif resp_type == "error":
                lines.append(f"error: {str(resp.get('error', ''))}")
            else:
                lines.append(f"result: {sanitize(resp)}")
        return lines

    def _trace_lines_from_entries(entries: list[dict[str, Any]], width: int, max_lines: int) -> list[str]:
        for entry in reversed(entries):
            if str(entry.get("entry_type", "agent")).strip() == "command":
                continue
            resp = entry.get("agent_response") if isinstance(entry.get("agent_response"), dict) else {}
            return _format_trace_lines(resp, width=width, max_lines=max_lines)
        return []

    def _show_session_picker() -> None:
        state.session_picker_entries = session_store.list_session_summaries(limit=20)
        selected_idx = 0
        for index, summary in enumerate(state.session_picker_entries):
            if summary.session.session_id == session.session_id:
                selected_idx = index
                break
        state.session_picker_active = True
        state.session_picker_selected_idx = selected_idx
        state.command_suggestions = []
        state.command_selected_idx = 0

    def _switch_to_session(target: SessionRef, width: int, max_trace_lines: int) -> None:
        nonlocal loop, session, slash_ctx
        session = target
        loop = build_loop_for_session(session)
        slash_ctx = SlashCommandContext(root=root, session_store=session_store, session=session, ui_mode="tui", color_out=False, color_err=False)
        entries = session_store.read_entries(session)
        _clear_session_view()
        state.chat_lines = _chat_lines_from_entries(entries)
        state.trace_lines = _trace_lines_from_entries(entries, width=width, max_lines=max_trace_lines)

    def _main(stdscr) -> int:
        nonlocal loop, session, slash_ctx
        curses.curs_set(1)
        stdscr.keypad(True)
        curses.noecho()
        curses.cbreak()

        while True:
            stdscr.erase()
            height, width = stdscr.getmaxyx()

            status_h = 1
            input_h = 2
            main_h = max(height - status_h - input_h, 1)
            top_h = max(min(main_h // 3, main_h - 1), 3)
            chat_h = max(main_h - top_h, 1)
            left_w = max(min(width // 3, max(width - 20, 10)), 20)
            right_w = max(width - left_w, 1)

            header = f"session={session.session_id}  mode={mode} strategy={strategy or 'none'} max_iters={max_iters}   keys: Enter=send  Ctrl-C=quit"
            stdscr.addstr(0, 0, header[: max(width - 1, 0)])

            todo_y = status_h
            todo_x = 0
            trace_y = status_h
            trace_x = left_w
            chat_y = status_h + top_h
            chat_x = 0
            input_y = status_h + top_h + chat_h

            _draw_box(stdscr, todo_y, todo_x, top_h, left_w, "Todo")
            _draw_box(stdscr, trace_y, trace_x, top_h, right_w, "Trace")
            _draw_box(stdscr, chat_y, chat_x, chat_h, width, "Chat")
            _draw_box(stdscr, input_y, 0, input_h, width, "Input")

            summary, items = _load_tasks(session_store, session)
            todo_title = f"todo: total={summary.get('total',0)} ready={summary.get('ready',0)} blocked={summary.get('blocked',0)} in_progress={summary.get('in_progress',0)} completed={summary.get('completed',0)}"
            try:
                stdscr.addstr(todo_y + 1, todo_x + 1, todo_title[: max(left_w - 2, 0)])
            except curses.error:
                pass
            todo_lines = _format_todo_lines(items, max_lines=max(top_h - 2, 0))
            for i, (line, style) in enumerate(todo_lines[: max(top_h - 2, 0)]):
                attr = curses.A_NORMAL
                if style == 1:
                    attr = curses.A_BOLD
                elif style == 2:
                    attr = curses.A_DIM
                try:
                    stdscr.addstr(todo_y + 2 + i, todo_x + 1, line[: max(left_w - 2, 0)], attr)
                except curses.error:
                    pass

            if state.input_buf != state.command_last_input:
                state.command_last_input = state.input_buf
                state.command_suppressed = False
            suggestions = [] if state.command_suppressed else get_command_suggestions(state.input_buf)
            if state.session_picker_active:
                _draw_box(stdscr, trace_y, trace_x, top_h, right_w, "Sessions (Enter=switch)")
                trace_lines = []
                max_lines = max(top_h - 2, 0)
                for i, summary in enumerate(state.session_picker_entries[:max_lines]):
                    prefix = "> " if i == state.session_picker_selected_idx else "  "
                    current = " *" if summary.session.session_id == session.session_id else ""
                    line = f"{prefix}{summary.session.session_id}{current}"
                    trace_lines.append(line)
            elif suggestions:
                if [s.completion for s in suggestions] != [s.completion for s in state.command_suggestions]:
                    state.command_selected_idx = 0
                state.command_suggestions = suggestions
                title = "Commands (Enter=complete)"
                _draw_box(stdscr, trace_y, trace_x, top_h, right_w, title)
                trace_lines = []
                max_lines = max(top_h - 2, 0)
                for i, s in enumerate(suggestions[:max_lines]):
                    prefix = "> " if i == state.command_selected_idx else "  "
                    trace_lines.append(f"{prefix}{s.usage}")
            else:
                state.command_suggestions = []
                state.command_selected_idx = 0
                _draw_box(stdscr, trace_y, trace_x, top_h, right_w, "Trace")
                trace_lines = state.trace_lines[: max(top_h - 2, 0)]
            for i, line in enumerate(trace_lines):
                try:
                    stdscr.addstr(trace_y + 1 + i, trace_x + 1, line[: max(right_w - 2, 0)])
                except curses.error:
                    pass

            chat_inner_h = max(chat_h - 2, 0)
            chat_inner_w = max(width - 2, 1)
            visible_chat = state.chat_lines[-chat_inner_h:]
            row = 0
            for entry in visible_chat:
                for line in _wrap_lines(entry, chat_inner_w):
                    if row >= chat_inner_h:
                        break
                    try:
                        stdscr.addstr(chat_y + 1 + row, 1, line[:chat_inner_w])
                    except curses.error:
                        pass
                    row += 1
                if row >= chat_inner_h:
                    break

            prompt = "> "
            input_inner_w = max(width - 2 - len(prompt), 1)
            try:
                stdscr.addstr(input_y + 1, 1, prompt)
                stdscr.addstr(input_y + 1, 1 + len(prompt), state.input_buf[-input_inner_w:])
                stdscr.move(input_y + 1, 1 + len(prompt) + min(len(state.input_buf), input_inner_w))
            except curses.error:
                pass

            stdscr.refresh()

            try:
                ch = stdscr.getch()
            except KeyboardInterrupt:
                break
            if ch in (3,):
                break
            if ch in (curses.KEY_UP, curses.KEY_DOWN) and state.session_picker_active:
                delta = -1 if ch == curses.KEY_UP else 1
                idx = state.session_picker_selected_idx + delta
                idx = max(0, min(idx, len(state.session_picker_entries) - 1))
                state.session_picker_selected_idx = idx
                continue
            if ch in (curses.KEY_UP, curses.KEY_DOWN) and state.command_suggestions:
                delta = -1 if ch == curses.KEY_UP else 1
                idx = state.command_selected_idx + delta
                idx = max(0, min(idx, len(state.command_suggestions) - 1))
                state.command_selected_idx = idx
                continue
            if ch == 27 and state.session_picker_active:
                state.session_picker_active = False
                state.session_picker_entries = []
                state.session_picker_selected_idx = 0
                continue
            if ch == 27 and state.command_suggestions:
                state.command_suppressed = True
                state.command_suggestions = []
                state.command_selected_idx = 0
                continue
            if ch in (10, 13):
                if state.session_picker_active and state.session_picker_entries:
                    idx = max(0, min(state.session_picker_selected_idx, len(state.session_picker_entries) - 1))
                    target = state.session_picker_entries[idx].session
                    _switch_to_session(target, width=max(right_w - 2, 1), max_trace_lines=max(top_h - 2, 0))
                    continue
                if state.command_suggestions:
                    idx = max(0, min(state.command_selected_idx, len(state.command_suggestions) - 1))
                    state.input_buf = apply_command_completion(state.input_buf, state.command_suggestions[idx])
                    state.command_suppressed = True
                    state.command_suggestions = []
                    state.command_selected_idx = 0
                    continue
                text = state.input_buf.strip()
                state.input_buf = ""
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
                        _clear_session_view()
                    elif dispatch.ui_action == "new_session":
                        session_store.append_round(session, build_command_session_entry(dispatch))
                        session = session_store.new_session()
                        loop = build_loop_for_session(session)
                        slash_ctx = SlashCommandContext(root=root, session_store=session_store, session=session, ui_mode="tui", color_out=False, color_err=False)
                        _clear_session_view()
                        if dispatch.should_exit:
                            break
                        continue
                    elif dispatch.ui_action == "show_sessions":
                        session_store.append_round(session, build_command_session_entry(dispatch))
                        _show_session_picker()
                        continue
                    else:
                        state.chat_lines.append(f"cmd: {str(dispatch.raw_input or '').strip()}")
                    if dispatch.stdout:
                        for line in str(dispatch.stdout).splitlines():
                            state.chat_lines.append(line)
                    if dispatch.stderr:
                        state.chat_lines.append("stderr:")
                        for line in str(dispatch.stderr).splitlines():
                            state.chat_lines.append(line)
                    session_store.append_round(session, build_command_session_entry(dispatch))
                    if str(dispatch.command_name or "") == "trace" and dispatch.stdout:
                        state.trace_lines = str(dispatch.stdout).splitlines()
                    if dispatch.should_exit:
                        break
                    continue
                state.round_idx += 1
                request: dict[str, Any] = {"mode": mode, "strategy": strategy, "goal": text, "max_iters": max_iters}
                resp = loop.run(request, max_iters=max_iters)
                state.chat_lines.append(f"user: {text}")
                resp_type = str(resp.get("type", "")).strip()
                if resp_type == "final":
                    state.chat_lines.append(f"final: {str(resp.get('content',''))}")
                elif resp_type == "error":
                    state.chat_lines.append(f"error: {str(resp.get('error',''))}")
                else:
                    state.chat_lines.append(f"result: {sanitize(resp)}")
                state.trace_lines = _format_trace_lines(resp, width=max(right_w - 2, 1), max_lines=max(top_h - 2, 0))
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
                continue
            if ch in (curses.KEY_BACKSPACE, 127, 8):
                state.input_buf = state.input_buf[:-1]
                continue
            if ch == curses.KEY_RESIZE:
                continue
            if 0 <= ch <= 255:
                char = chr(ch)
                if char.isprintable():
                    state.input_buf += char
                continue
            continue

        return 0

    try:
        return int(curses.wrapper(_main))
    except KeyboardInterrupt:
        return 0
