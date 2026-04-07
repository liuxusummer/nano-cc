from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from textual import work
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import Footer, Header, Input, Label, ListItem, ListView, RichLog, Static
from rich.markdown import Markdown

from src.cli.render import iter_tool_steps, sanitize, truncate_text
from src.cli.slash_commands import CommandSuggestion, SlashCommandContext, apply_command_completion, build_command_session_entry, dispatch_slash_command, get_command_suggestions
from src.cli.session_store import SessionRef, SessionStore, SessionSummary
from src.core.agent_loop import AgentLoop
from src.core.task_system import TaskGraphStore


@dataclass(slots=True)
class _TraceEntry:
    label: str
    output_preview: str


class AgentStep(Message):
    """Event fired when the agent completes a single step."""
    def __init__(self, step: dict[str, Any], observation: dict[str, Any] | None) -> None:
        self.step = step
        self.observation = observation
        super().__init__()


class AgentResponse(Message):
    """Event fired when the agent completes its run."""
    def __init__(self, text: str, response: dict[str, Any]) -> None:
        self.text = text
        self.response = response
        super().__init__()


class AgentStateChanged(Message):
    """Event fired when the agent starts or stops running."""
    def __init__(self, is_running: bool) -> None:
        self.is_running = is_running
        super().__init__()


class _HelpScreen(ModalScreen[None]):
    BINDINGS = [("escape", "dismiss", "Close")]

    def compose(self) -> ComposeResult:
        text = "\n".join(
            [
                "快捷键",
                "",
                "Tab / Shift+Tab: 切换焦点",
                "Enter: 发送输入框内容",
                "Up/Down: 列表导航（Todo / Trace）",
                "?: 打开帮助",
                "Esc: 关闭帮助",
                "",
                "输入 exit/quit 退出。",
            ]
        )
        yield Static(text, id="help")

    def action_dismiss(self) -> None:
        self.dismiss(None)


class _TextualCliApp(App[None]):
    CSS = """
    #root {
        height: 100%;
        background: $surface;
    }
    #status {
        padding: 0 1;
        color: $text-muted;
        background: $surface-darken-1;
        width: 100%;
    }
    #main {
        height: 1fr;
        margin: 1 1 0 1;
    }
    #left_panel {
        width: 25%;
        height: 100%;
        min-width: 20;
    }
    #todo_panel {
        height: 34%;
        border: round $primary;
        background: $panel;
        margin-bottom: 1;
    }
    #trace_panel {
        height: 33%;
        border: round $secondary;
        background: $panel;
        margin-bottom: 1;
    }
    #trace_preview {
        height: 33%;
        border: round $secondary;
        background: $panel;
    }
    #right_panel {
        width: 75%;
        height: 100%;
        margin-left: 1;
    }
    #chat_panel {
        height: 100%;
        border: round $accent;
        background: $panel;
    }
    #input {
        border: round $primary;
        margin: 0 1 1 1;
        background: $panel;
    }
    #command_panel {
        height: 8;
        border: round $secondary;
        margin: 0 1 0 1;
        background: $panel;
        display: none;
    }
    .title {
        padding: 0 1;
        color: $text;
        text-style: bold;
        background: $primary-darken-2;
        width: 100%;
    }
    .inprogress {
        color: $warning;
        text-style: bold;
    }
    """

    BINDINGS = [
        ("?", "help", "Help"),
    ]

    def __init__(
        self,
        *,
        root: Path,
        loop: AgentLoop,
        build_loop_for_session,
        session_store: SessionStore,
        session: SessionRef,
        mode: str,
        strategy: str,
        max_iters: int,
    ) -> None:
        super().__init__()
        self._root = Path(root)
        self._agent_loop = loop
        self._build_loop_for_session = build_loop_for_session
        self._session_store = session_store
        self._session = session
        self._mode = str(mode)
        self._strategy = str(strategy)
        self._max_iters = int(max_iters)
        self._trace_entries: list[_TraceEntry] = []
        self._agent_running = False
        self._command_suggestions: list[CommandSuggestion] = []
        self._command_selected_idx = 0
        self._command_suppressed_value: str | None = None
        self._session_picker_active = False
        self._session_picker_entries: list[SessionSummary] = []
        self._session_picker_selected_idx = 0

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        with Container(id="root"):
            yield Label(
                f"session={self._session.session_id}  mode={self._mode} strategy={self._strategy or 'none'} max_iters={self._max_iters}",
                id="status",
            )
            with Horizontal(id="main"):
                with Vertical(id="left_panel"):
                    with Vertical(id="todo_panel"):
                        yield Label("📋 Todo", classes="title")
                        yield ListView(id="todo_list")
                    with Vertical(id="trace_panel"):
                        yield Label("🔍 Trace (本轮)", classes="title")
                        yield ListView(id="trace_list")
                    with Vertical(id="trace_preview"):
                        yield Label("📄 Trace 输出预览", classes="title")
                        yield Static("", id="trace_output")
                with Vertical(id="right_panel"):
                    with Vertical(id="chat_panel"):
                        yield Label("💬 Chat", classes="title")
                        yield RichLog(id="chat_log", wrap=True, highlight=True, markup=True)
            with Vertical(id="command_panel"):
                yield Label("⌨️  Commands", classes="title")
                yield ListView(id="command_list")
            yield Input(placeholder="输入后回车发送（exit/quit 退出，? 查看帮助）", id="input")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#input", Input).focus()
        self.query_one("#todo_list", ListView).clear()
        self.query_one("#command_list", ListView).clear()
        status_label = self.query_one("#status", Label)
        status_label.update(f"session={self._session.session_id}  mode={self._mode} strategy={self._strategy or 'none'} max_iters={self._max_iters}")
        self.set_interval(1.0, self._refresh_todo)

    def action_help(self) -> None:
        self.push_screen(_HelpScreen())

    def _load_tasks(self) -> list[dict[str, Any]]:
        store = TaskGraphStore(root_dir=self._session_store.resolve_task_graph_root(self._session))
        try:
            graph = store.load_graph()
        except Exception:
            return []
        return [item.to_dict() for item in sorted(graph.values(), key=lambda t: t.id)]

    def _refresh_todo(self) -> None:
        todo_list = self.query_one("#todo_list", ListView)
        
        old_labels = []
        for item in todo_list.children:
            if isinstance(item, ListItem) and item.children:
                child = item.children[0]
                if isinstance(child, Label):
                    old_labels.append(str(getattr(child, "renderable", getattr(child, "text", str(child)))))
                
        tasks = self._load_tasks()
        inprog: list[dict[str, Any]] = []
        pending: list[dict[str, Any]] = []
        done: list[dict[str, Any]] = []
        for task in tasks:
            status = str(task.get("status", "")).strip()
            content = str(task.get("description", task.get("content", ""))).strip()
            if not content:
                continue
            if status == "in_progress":
                inprog.append(task)
            elif status == "completed":
                done.append(task)
            else:
                pending.append(task)
                
        new_labels = []
        for task in inprog + pending + done:
            status = str(task.get("status", "")).strip()
            content = str(task.get("description", task.get("content", ""))).strip()
            
            if status == "in_progress":
                icon = "[bold yellow]↻[/]"
                label_text = f"[bold yellow]{content}[/bold yellow]"
            elif status == "completed":
                icon = "[bold green]✓[/]"
                label_text = f"[dim][strike]{content}[/strike][/dim]"
            else:
                icon = "[dim]○[/]"
                label_text = f"[dim]{content}[/dim]"
                
            new_labels.append(f"{icon} {label_text}")
            
        if old_labels == new_labels:
            return
            
        todo_list.clear()
        for label in new_labels:
            todo_list.append(ListItem(Label(label)))

    def _clear_session_view(self) -> None:
        self.query_one("#chat_log", RichLog).clear()
        self.query_one("#trace_list", ListView).clear()
        self.query_one("#trace_output", Static).update("")
        self.query_one("#command_list", ListView).clear()
        input_widget = self.query_one("#input", Input)
        input_widget.value = ""
        self._trace_entries = []
        self._command_suggestions = []
        self._command_selected_idx = 0
        self._command_suppressed_value = None
        self._session_picker_active = False
        self._session_picker_entries = []
        self._session_picker_selected_idx = 0
        self._set_command_panel_visible(False)
        input_widget.focus()

    def _refresh_status(self) -> None:
        status_label = self.query_one("#status", Label)
        status_label.update(f"session={self._session.session_id}  mode={self._mode} strategy={self._strategy or 'none'} max_iters={self._max_iters}")

    def _start_new_session_view(self, session: SessionRef) -> None:
        self._session = session
        self._clear_session_view()
        self._refresh_status()

    def _show_session_picker(self) -> None:
        self._session_picker_entries = self._session_store.list_session_summaries(limit=20)
        self._session_picker_active = True
        self._session_picker_selected_idx = 0
        for index, summary in enumerate(self._session_picker_entries):
            if summary.session.session_id == self._session.session_id:
                self._session_picker_selected_idx = index
                break
        self._command_suggestions = []
        self._command_selected_idx = 0
        lst = self.query_one("#command_list", ListView)
        lst.clear()
        for summary in self._session_picker_entries:
            current = " *" if summary.session.session_id == self._session.session_id else ""
            last_input = summary.last_input
            if len(last_input) > 48:
                last_input = last_input[:45] + "..."
            extra = f" [dim]{last_input}[/dim]" if last_input else ""
            lst.append(ListItem(Label(f"[bold cyan]{summary.session.session_id}[/]{current}{extra}")))
        if self._session_picker_entries:
            lst.index = self._session_picker_selected_idx
        self._set_command_panel_visible(True)

    def _chat_history_entries(self, session: SessionRef) -> list[dict[str, Any]]:
        return self._session_store.read_entries(session)

    def _restore_session_history(self, session: SessionRef) -> None:
        entries = self._chat_history_entries(session)
        self._clear_session_view()
        chat = self.query_one("#chat_log", RichLog)
        latest_agent_response: dict[str, Any] | None = None
        for entry in entries:
            entry_type = str(entry.get("entry_type", "agent")).strip() or "agent"
            if entry_type == "command":
                raw_input = str(entry.get("raw_input", "")).strip()
                output = entry.get("command_output") if isinstance(entry.get("command_output"), dict) else {}
                out = str(output.get("stdout", "")) if isinstance(output, dict) else ""
                err = str(output.get("stderr", "")) if isinstance(output, dict) else ""
                if raw_input:
                    chat.write(f"\n[bold cyan]❯ {raw_input}[/bold cyan]\n")
                if out:
                    chat.write(Markdown(f"```\n{out.rstrip()}\n```"))
                    chat.write("")
                if err:
                    chat.write(Markdown(f"```text\n{err.rstrip()}\n```"))
                    chat.write("")
                continue
            user_input = str(entry.get("user_input", "")).strip()
            resp = entry.get("agent_response") if isinstance(entry.get("agent_response"), dict) else {}
            if user_input:
                chat.write(f"\n[bold cyan]❯ {user_input}[/bold cyan]\n")
            resp_type = str(resp.get("type", "")).strip()
            if resp_type == "final":
                chat.write("[bold yellow]🤖 Agent:[/bold yellow]")
                chat.write(Markdown(str(resp.get("content", ""))))
                chat.write("")
            elif resp_type == "error":
                chat.write(f"[bold red]✗ Error:[/bold red] {str(resp.get('error', ''))}")
                chat.write("")
            else:
                chat.write(f"[bold magenta]⚙ Result:[/bold magenta] {json.dumps(sanitize(resp), ensure_ascii=False)}")
                chat.write("")
            latest_agent_response = resp
        if latest_agent_response is not None:
            self._refresh_trace(latest_agent_response)
        else:
            self.query_one("#trace_list", ListView).clear()
            self.query_one("#trace_output", Static).update("")
            self._trace_entries = []
        self._refresh_todo()
        self._refresh_status()

    def _switch_to_session(self, session: SessionRef) -> None:
        self._session = session
        self._agent_loop = self._build_loop_for_session(session)
        self._restore_session_history(session)

    def _append_trace(self, step: dict[str, Any], obs: dict[str, Any]) -> None:
        trace_list = self.query_one("#trace_list", ListView)
        trace_output = self.query_one("#trace_output", Static)
        
        name = str((step or {}).get("name", "")).strip() or "<unknown>"
        result = (obs or {}).get("result") if isinstance(obs, dict) else None
        success = None
        output_preview = ""
        if isinstance(result, dict):
            success = result.get("success")
            output = result.get("output")
            text = json.dumps(sanitize(output), ensure_ascii=False, indent=2, sort_keys=False)
            truncated, was_truncated = truncate_text(text, limit=2000)
            output_preview = truncated + ("\n...[已截断]" if was_truncated else "")
        tag = "[bold yellow]?[/]" if success is None else ("[bold green]ok[/]" if bool(success) else "[bold red]fail[/]")
        label = f"[bold cyan]{name}[/]: {tag}"
        
        self._trace_entries.append(_TraceEntry(label=label, output_preview=output_preview))
        trace_list.append(ListItem(Label(label)))
        # 确保output_preview是字符串且不为None
        if output_preview is None:
            output_preview = ""
        # 尝试更新trace_output
        try:
            trace_output.update(str(output_preview))
        except Exception as e:
            # 如果更新失败，记录错误但不中断程序
            import traceback
            traceback.print_exc()

    def _refresh_trace(self, resp: dict[str, Any]) -> None:
        trace_list = self.query_one("#trace_list", ListView)
        trace_output = self.query_one("#trace_output", Static)
        
        trace_list.clear()
        self._trace_entries = []

        trace = resp.get("trace") if isinstance(resp, dict) else None
        tool_items = iter_tool_steps(trace if isinstance(trace, list) else None)
        for item in tool_items:
            step = item.get("step") if isinstance(item, dict) else {}
            obs = item.get("observation") if isinstance(item, dict) else {}
            name = str((step or {}).get("name", "")).strip() or "<unknown>"
            result = (obs or {}).get("result") if isinstance(obs, dict) else None
            success = None
            output_preview = ""
            if isinstance(result, dict):
                success = result.get("success")
                output = result.get("output")
                text = json.dumps(sanitize(output), ensure_ascii=False, indent=2, sort_keys=False)
                truncated, was_truncated = truncate_text(text, limit=2000)
                output_preview = truncated + ("\n...[已截断]" if was_truncated else "")
            tag = "[bold yellow]?[/]" if success is None else ("[bold green]ok[/]" if bool(success) else "[bold red]fail[/]")
            label = f"[bold cyan]{name}[/]: {tag}"
            self._trace_entries.append(_TraceEntry(label=label, output_preview=output_preview))
            trace_list.append(ListItem(Label(label)))

        if self._trace_entries:
            # 确保output_preview是字符串且不为None
            output_preview = self._trace_entries[0].output_preview
            if output_preview is None:
                output_preview = ""
            # 尝试更新trace_output
            try:
                trace_output.update(str(output_preview))
            except Exception as e:
                # 如果更新失败，记录错误但不中断程序
                import traceback
                traceback.print_exc()
        else:
            trace_output.update("")

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        if event.list_view.id != "trace_list":
            return
        idx = int(event.list_view.index or 0)
        if 0 <= idx < len(self._trace_entries):
            # 确保output_preview是字符串且不为None
            output_preview = self._trace_entries[idx].output_preview
            if output_preview is None:
                output_preview = ""
            # 尝试更新trace_output
            try:
                self.query_one("#trace_output", Static).update(str(output_preview))
            except Exception as e:
                # 如果更新失败，记录错误但不中断程序
                import traceback
                traceback.print_exc()

    def _set_command_panel_visible(self, visible: bool) -> None:
        panel = self.query_one("#command_panel", Vertical)
        panel.styles.display = "block" if visible else "none"

    def _refresh_command_suggestions(self, text: str) -> None:
        suggestions = get_command_suggestions(text)
        if not suggestions:
            self._command_suggestions = []
            self._command_selected_idx = 0
            self.query_one("#command_list", ListView).clear()
            self._set_command_panel_visible(False)
            return

        if [s.completion for s in suggestions] == [s.completion for s in self._command_suggestions]:
            self._set_command_panel_visible(True)
            return

        self._command_suggestions = suggestions
        self._command_selected_idx = 0
        lst = self.query_one("#command_list", ListView)
        lst.clear()
        for s in suggestions:
            lst.append(ListItem(Label(f"[bold cyan]{s.usage}[/] [dim]{s.description}[/dim]")))
        lst.index = 0
        self._set_command_panel_visible(True)

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id != "input":
            return
        if self._session_picker_active:
            self._session_picker_active = False
            self._session_picker_entries = []
            self._session_picker_selected_idx = 0
            self.query_one("#command_list", ListView).clear()
            self._set_command_panel_visible(False)
        value = str(event.value or "")
        if self._command_suppressed_value is not None:
            if value == self._command_suppressed_value:
                self._command_suggestions = []
                self._command_selected_idx = 0
                self.query_one("#command_list", ListView).clear()
                self._set_command_panel_visible(False)
                return
            self._command_suppressed_value = None
        self._refresh_command_suggestions(value)

    def on_key(self, event) -> None:
        if self._session_picker_active:
            key = str(getattr(event, "key", "") or "").lower()
            if key in {"up", "down"} and self._session_picker_entries:
                delta = -1 if key == "up" else 1
                idx = self._session_picker_selected_idx + delta
                idx = max(0, min(idx, len(self._session_picker_entries) - 1))
                self._session_picker_selected_idx = idx
                self.query_one("#command_list", ListView).index = idx
                event.stop()
                return
            if key == "escape":
                self._session_picker_active = False
                self._session_picker_entries = []
                self._session_picker_selected_idx = 0
                self.query_one("#command_list", ListView).clear()
                self._set_command_panel_visible(False)
                event.stop()
                return
            if key == "enter" and self._session_picker_entries:
                idx = max(0, min(self._session_picker_selected_idx, len(self._session_picker_entries) - 1))
                target = self._session_picker_entries[idx].session
                self._switch_to_session(target)
                event.stop()
                return
        if not self._command_suggestions:
            return
        panel = self.query_one("#command_panel", Vertical)
        if str(panel.styles.display) == "none":
            return

        key = str(getattr(event, "key", "") or "").lower()
        if key in {"up", "down"}:
            delta = -1 if key == "up" else 1
            idx = self._command_selected_idx + delta
            idx = max(0, min(idx, len(self._command_suggestions) - 1))
            self._command_selected_idx = idx
            self.query_one("#command_list", ListView).index = idx
            event.stop()
            return
        if key == "escape":
            self._command_suggestions = []
            self._command_selected_idx = 0
            self.query_one("#command_list", ListView).clear()
            self._set_command_panel_visible(False)
            event.stop()
            return
        if key == "enter":
            idx = max(0, min(self._command_selected_idx, len(self._command_suggestions) - 1))
            suggestion = self._command_suggestions[idx]
            input_widget = self.query_one("#input", Input)
            new_value = apply_command_completion(str(input_widget.value or ""), suggestion)
            input_widget.value = new_value
            input_widget.focus()
            self._command_suppressed_value = new_value
            self._command_suggestions = []
            self._command_selected_idx = 0
            self.query_one("#command_list", ListView).clear()
            self._set_command_panel_visible(False)
            event.stop()
            return

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if self._agent_running:
            return

        text = str(event.value or "").strip()
        event.input.value = ""
        self._command_suggestions = []
        self._command_selected_idx = 0
        self._command_suppressed_value = None
        self.query_one("#command_list", ListView).clear()
        self._set_command_panel_visible(False)
        if not text:
            return
        if text in {"exit", "quit", ":q"}:
            self.exit()
            return

        slash_ctx = SlashCommandContext(root=self._root, session_store=self._session_store, session=self._session, ui_mode="textual", color_out=False, color_err=False)
        dispatch = dispatch_slash_command(text, slash_ctx)
        if dispatch.kind == "forward":
            text = str(dispatch.forward_text or "").strip()
            if not text:
                return
        elif dispatch.kind == "handled":
            if dispatch.ui_action == "clear":
                self._clear_session_view()
                self._session_store.append_round(self._session, build_command_session_entry(dispatch))
                if dispatch.should_exit:
                    self.exit()
                return
            if dispatch.ui_action == "new_session":
                self._session_store.append_round(self._session, build_command_session_entry(dispatch))
                new_session = self._session_store.new_session()
                self._agent_loop = self._build_loop_for_session(new_session)
                self._start_new_session_view(new_session)
                if dispatch.should_exit:
                    self.exit()
                return
            if dispatch.ui_action == "show_sessions":
                self._session_store.append_round(self._session, build_command_session_entry(dispatch))
                self._show_session_picker()
                if dispatch.should_exit:
                    self.exit()
                return
            chat = self.query_one("#chat_log", RichLog)
            raw_input = str(dispatch.raw_input or "").strip()
            chat.write(f"\n[bold cyan]❯ {raw_input}[/bold cyan]\n")
            out = str(dispatch.stdout or "")
            err = str(dispatch.stderr or "")
            if out:
                chat.write(Markdown(f"```\n{out.rstrip()}\n```"))
                chat.write("")
            if err:
                chat.write(Markdown(f"```text\n{err.rstrip()}\n```"))
                chat.write("")
            self._session_store.append_round(self._session, build_command_session_entry(dispatch))
            if dispatch.should_exit:
                self.exit()
            return

        chat = self.query_one("#chat_log", RichLog)
        chat.write(f"\n[bold cyan]❯ {text}[/bold cyan]\n")
        self.query_one("#trace_list", ListView).clear()
        self.query_one("#trace_output", Static).update("")
        self._trace_entries = []
        request: dict[str, Any] = {"mode": self._mode, "strategy": self._strategy, "goal": text, "max_iters": self._max_iters}

        self._run_agent_task(text, request)

    @work(thread=True)
    def _run_agent_task(self, text: str, request: dict[str, Any]) -> None:
        self.post_message(AgentStateChanged(is_running=True))
        
        def step_callback(step: dict[str, Any], obs: dict[str, Any]) -> None:
            self.post_message(AgentStep(step=step, observation=obs))
            
        try:
            resp = self._agent_loop.run(request, max_iters=self._max_iters, step_callback=step_callback)
            self.post_message(AgentResponse(text=text, response=resp))
        except Exception as e:
            self.post_message(AgentResponse(text=text, response={"type": "error", "error": str(e)}))
        finally:
            self.post_message(AgentStateChanged(is_running=False))

    def on_agent_state_changed(self, event: AgentStateChanged) -> None:
        self._agent_running = event.is_running
        input_widget = self.query_one("#input", Input)
        input_widget.disabled = event.is_running

        status_label = self.query_one("#status", Label)
        base_status = f"session={self._session.session_id}  mode={self._mode} strategy={self._strategy or 'none'} max_iters={self._max_iters}"
        if event.is_running:
            status_label.update(f"{base_status}  [bold yellow](Agent Running...)[/]")
        else:
            status_label.update(base_status)
            input_widget.focus()

    def on_agent_step(self, event: AgentStep) -> None:
        chat = self.query_one("#chat_log", RichLog)
        step = event.step
        obs = event.observation
        if obs is None:
            content = step.get("content")
            if content:
                chat.write("[bold yellow]🤖 Agent:[/bold yellow]")
                chat.write(Markdown(str(content)))
                chat.write("")
                
            step_type = step.get("type")
            if step_type == "tool":
                name = step.get("name", "unknown")
                args = step.get("arguments", {})
                args_str = json.dumps(args, ensure_ascii=False)
                if len(args_str) > 100:
                    args_str = args_str[:97] + "..."
                chat.write(f"[dim]🛠️  Calling tool:[/] [bold cyan]{name}[/] [dim]{args_str}[/]")
        else:
            result = obs.get("result", {})
            if result:
                success = result.get("success")
                if success is True:
                    chat.write("[dim][bold green]✓[/bold green] Tool finished.[/dim]")
                elif success is False:
                    chat.write(f"[dim][bold red]✗[/bold red] Tool failed: {result.get('error')}[/dim]")
            self._append_trace(step, obs)

    def on_agent_response(self, event: AgentResponse) -> None:
        resp = event.response
        text = event.text
        
        chat = self.query_one("#chat_log", RichLog)
        resp_type = str(resp.get("type", "")).strip()
        if resp_type == "final":
            chat.write("[bold yellow]🤖 Agent:[/bold yellow]")
            chat.write(Markdown(str(resp.get('content', ''))))
            chat.write("")
        elif resp_type == "error":
            chat.write(f"[bold red]✗ Error:[/bold red] {str(resp.get('error', ''))}")
            chat.write("")
        elif resp_type == "tool_result":
            chat.write(f"[bold green]✓ Tool '{resp.get('name')}' succeeded[/bold green]")
            chat.write(Markdown(str(resp.get("result", {}).get("output", ""))))
            chat.write("")
        else:
            chat.write(f"[bold magenta]⚙ Result:[/bold magenta] {json.dumps(sanitize(resp), ensure_ascii=False)}")
            chat.write("")

        self._refresh_trace(resp if isinstance(resp, dict) else {})
        self._refresh_todo()
        self._session_store.append_round(
            self._session,
            {
                "entry_type": "agent",
                "user_input": text,
                "agent_response": sanitize(resp),
                "mode": self._mode,
                "strategy": self._strategy,
                "max_iters": self._max_iters,
            },
        )


def run_textual_ui(
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
    app = _TextualCliApp(
        root=root,
        loop=loop,
        build_loop_for_session=build_loop_for_session,
        session_store=session_store,
        session=session,
        mode=mode,
        strategy=strategy,
        max_iters=max_iters,
    )
    app.run()
    return 0
