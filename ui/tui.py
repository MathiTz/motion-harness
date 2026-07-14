"""
Motion Harness — Professional TUI
==================================
Built on Textual with native theme switching and runtime provider selection.

Screens:
  - ProviderSelect: Pick a provider/model at startup
  - MainScreen:     Tabbed hub (Chat, Tasks, Skills, Memory, Settings)

Key features:
  - Themes cascade through every widget via Textual's ``$variable`` system
  - Ctrl+T cycles themes instantly
  - Settings tab has a SelectableDropdown for switching provider/model at runtime
  - Ctrl+C cancels the current request; Ctrl+Q quits

Launch:  python main.py              → TUI (default)
         python main.py --chat       → old REPL
         python main.py --provider X → TUI with pre-selected provider
"""

from __future__ import annotations

import asyncio
import os
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Optional

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import (
    Button,
    Header,
    Footer,
    Input,
    Label,
    ListItem,
    ListView,
    Select,
    Static,
    TabbedContent,
    TabPane,
)

from core.config import ConfigManager
from core.orchestrator import TaskManager, TaskRequest
from core.providers import ModelConfig
from main import MotionAgent
from ui.themes import ThemeRegistry

WORKSPACE = os.getenv("MOTION_WORKSPACE", os.getcwd())
DASHBOARD_URL = "https://localhost:7860/"
DASHBOARD_ADMIN_KEY = "ME27dXc6uoEC_dWXJCyPVDPN"


# ─── Shared state ─────────────────────────────────────────────────────────────

class AppState:
    """Reactive state shared across all screens."""

    def __init__(self) -> None:
        self.agent: Optional[MotionAgent] = None
        self.task_manager: Optional[TaskManager] = None
        self.config_manager: ConfigManager = ConfigManager()
        self.current_provider_id: str = ""
        self.current_theme: str = "one_dark"
        self.caveman_enabled: bool = True

    def reconnect(self, provider_id: str) -> None:
        """Re-create the agent and task manager for a new provider/model."""
        cfg = self.config_manager.get_provider_config(provider_id)
        model_config = ModelConfig(
            name=cfg.get("name", provider_id),
            endpoint=cfg["endpoint"],
            api_key=cfg.get("api_key"),
            provider_type=cfg.get("provider_type", "cloud"),
            options=cfg.get("options", {}),
        )
        # Close old connections if any
        if self.agent:
            try:
                self.agent.memory.close()
            except Exception:
                pass
        self.agent = MotionAgent(model_config)
        self.task_manager = TaskManager(model_config, WORKSPACE)
        self.current_provider_id = provider_id

    @staticmethod
    def build_provider_options() -> list[tuple[str, str]]:
        """Return [(display_label, provider_id), ...] for Select dropdowns.
        Only includes providers that have a configured API key (or are local)."""
        cm = ConfigManager()
        providers = cm.list_providers()
        options: list[tuple[str, str]] = []
        for pid, name, models, is_default, has_key in providers:
            if not has_key:
                continue
            if len(models) > 1:
                for m in models:
                    full = f"{pid}/{m}"
                    options.append((f"{name} → {m}", full))
            elif models:
                full = f"{pid}/{models[0]}" if models[0] != "?" else pid
                options.append((f"{name}", full))
            else:
                options.append((name, pid))
        return options

    @staticmethod
    def build_all_provider_info() -> list[tuple[str, str, list, bool, bool]]:
        """Return all providers: (pid, name, models, is_default, has_key)."""
        cm = ConfigManager()
        return cm.list_providers()


# ─── Small widgets ────────────────────────────────────────────────────────────

class ChatMessage(Static):
    """A single message in the chat log with formatted sender labels."""

    PREFIX = {"user": "You", "agent": "Agent", "system": "⚡"}

    def __init__(self, text: str, sender: str = "user", **kwargs) -> None:
        safe = text.replace("[", "\\[").replace("]", "\\]")
        prefix = self.PREFIX.get(sender, sender)
        if sender == "user":
            formatted = f"[bold cyan]{prefix}:[/] {safe}"
        elif sender == "agent":
            formatted = f"[bold green]{prefix}:[/] {safe}"
        else:
            formatted = f"[dim]{prefix}:[/] {safe}"
        super().__init__(formatted, classes=f"msg {sender}_msg", **kwargs)


class TaskRow(Static):
    """One row in the task panel."""

    ICON = {"PENDING": "⏳", "RUNNING": "⚙️", "COMPLETED": "✅", "FAILED": "❌"}
    COLOR = {"PENDING": "dim", "RUNNING": "bold yellow", "COMPLETED": "bold green", "FAILED": "bold red"}

    def __init__(self, task_id: str, prompt: str, status: str, **kwargs) -> None:
        self._task_id = task_id
        self._prompt = prompt
        self._status = status
        icon = self.ICON.get(status, "?")
        color = self.COLOR.get(status, "")
        display = f"[dim]{task_id}[/]  {icon} [{color}]{status}[/]  [dim]{prompt[:50]}[/]"
        super().__init__(display, id=f"task-{task_id}", **kwargs)


class ProviderOption(ListItem):
    """A selectable provider row on the startup screen."""

    def __init__(self, provider_id: str, name: str, models: list, is_default: bool, has_key: bool = True, **kwargs) -> None:
        self.provider_id = provider_id
        self.models = models
        self.has_key = has_key
        marker = " ← default" if is_default else ""
        lock = "" if has_key else " [dim red]🔒 no key[/]"
        label = f"⚡ {name}{marker}{lock}"
        if len(models) > 1:
            label += f"  [dim]({', '.join(models[:3])}{'…' if len(models) > 3 else ''})[/dim]"
        super().__init__(Label(label), **kwargs)


# ─── Provider selection screen ────────────────────────────────────────────────

class ProviderSelectScreen(Screen):
    """Pick a provider/model at startup."""

    CSS = """
    #provider_screen {
        align: center middle;
    }
    #provider_box {
        width: 72;
        height: auto;
        max-height: 85%;
        border: round $primary;
        border-title: " Motion Harness ";
        padding: 1 3;
        background: $surface;
        overflow-y: auto;
    }
    #provider_title {
        text-align: center;
        text-style: bold;
        color: $primary;
        margin-bottom: 0;
    }
    #provider_subtitle {
        text-align: center;
        color: $text-muted;
        margin-bottom: 1;
    }
    #provider_list {
        height: auto;
        max-height: 22;
        border: solid $border;
        padding: 0 1;
        background: $background;
    }
    #provider_status {
        text-align: center;
        color: $text-muted;
        margin-top: 1;
    }
    """

    BINDINGS = [
        Binding("enter", "select", "Select"),
        Binding("q", "quit", "Quit"),
    ]

    def __init__(self, state: AppState, **kwargs) -> None:
        super().__init__(**kwargs)
        self.state = state

    def compose(self) -> ComposeResult:
        with Container(id="provider_screen"):
            with Container(id="provider_box"):
                yield Label("⚡ Motion Harness", id="provider_title")
                yield Label("Select a provider:", id="provider_subtitle")
                yield ListView(id="provider_list")
                yield Label("↑↓ Navigate · Enter Select · Q Quit", id="provider_status")

    def on_mount(self) -> None:
        providers = self.state.config_manager.list_providers()
        lv = self.query_one("#provider_list", ListView)
        for pid, name, models, is_default, has_key in providers:
            lv.append(ProviderOption(pid, name, models, is_default, has_key))

    def action_select(self) -> None:
        lv = self.query_one("#provider_list", ListView)
        idx = lv.index
        if idx is None:
            return
        option = lv.children[idx]
        if not isinstance(option, ProviderOption):
            return

        if not option.has_key:
            self.notify("🔒 No API key configured for this provider", severity="warning")
            return

        provider_id = option.provider_id
        models = option.models
        provider_cfg = self.state.config_manager.get_provider_config(provider_id)
        default_model = provider_cfg.get("default_model") or (models[0] if models else None)
        full_id = f"{provider_id}/{default_model}" if default_model else provider_id

        try:
            self.state.reconnect(full_id)
        except Exception as e:
            self.notify(f"Connection failed: {e}", severity="error")
            return

        self.app.switch_screen(MainScreen(self.state))


# ─── Main hub screen (tabbed) ────────────────────────────────────────────────

class MainScreen(Screen):
    """The main hub with Chat, Tasks, Skills, Memory, Settings tabs."""

    CSS = """
    #main_tabs { height: 1fr; }
    TabbedContent TabPane {
        padding: 0 1;
    }
    """

    BINDINGS = [
        Binding("ctrl+t", "toggle_theme", "Theme"),
        Binding("ctrl+q", "quit", "Quit"),
    ]

    def __init__(self, state: AppState, **kwargs) -> None:
        super().__init__(**kwargs)
        self.state = state

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with TabbedContent(id="main_tabs"):
            with TabPane("💬 Chat", id="chat_tab"):
                yield ChatPane(self.state)
            with TabPane("⚙️ Tasks", id="tasks_tab"):
                yield TasksPane(self.state)
            with TabPane("🎓 Skills", id="skills_tab"):
                yield SkillsPane(self.state)
            with TabPane("🧠 Memory", id="memory_tab"):
                yield MemoryPane(self.state)
            with TabPane("🔧 Settings", id="settings_tab"):
                yield SettingsPane(self.state)
        yield Footer()

    def action_toggle_theme(self) -> None:
        themes = ThemeRegistry.theme_ids()
        idx = themes.index(self.state.current_theme)
        self.state.current_theme = themes[(idx + 1) % len(themes)]
        # Use Textual's native theme system — cascades through ALL CSS vars
        self.app.theme = self.state.current_theme
        self.notify(f"Theme → {self.state.current_theme}")


# ─── Chat pane ────────────────────────────────────────────────────────────────

class ChatPane(Container):
    """Message history + input bar."""

    CSS = """
    ChatPane {
        layout: vertical;
        height: 1fr;
    }
    #chat_log {
        height: 1fr;
        border: round $primary;
        border-title: " Chat ";
        padding: 0 1;
        overflow-y: auto;
        scrollbar-size: 1 1;
        background: $background;
    }
    #chat_input_row {
        height: auto;
        padding: 1 0 0 0;
    }
    #chat_input {
        height: 3;
        border: round $primary;
    }
    .msg {
        margin: 0 0;
        padding: 0 1;
    }
    .user_msg {
        color: $primary;
        text-style: bold;
        background: $surface;
        padding: 0 1;
        margin: 0 0;
    }
    .agent_msg {
        color: $foreground;
        background: transparent;
        padding: 0 1;
        margin: 0 0;
    }
    .system_msg {
        color: $text-muted;
        text-style: italic;
        padding: 0 1;
        margin: 0 0;
    }
    """

    def __init__(self, state: AppState, **kwargs) -> None:
        super().__init__(**kwargs)
        self.state = state

    def compose(self) -> ComposeResult:
        yield VerticalScroll(id="chat_log")
        with Horizontal(id="chat_input_row"):
            yield Input(placeholder="Type a message… (Enter to send)", id="chat_input")

    def on_mount(self) -> None:
        name = self.state.agent.provider.config.name if self.state.agent else "?"
        log = self.query_one("#chat_log", VerticalScroll)
        log.border_title = "Chat"
        log.mount(ChatMessage(f"⚡ Motion Harness — connected to {name}", sender="system"))
        self.query_one("#chat_input", Input).focus()

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text:
            return
        event.input.value = ""

        log = self.query_one("#chat_log", VerticalScroll)
        log.mount(ChatMessage(text, sender="user"))
        thinking = ChatMessage("⚙️ Thinking…", sender="system")
        log.mount(thinking)
        log.scroll_end(animate=False)

        self._run_agent(text, thinking)

    @work(exclusive=True, name="agent_chat")
    async def _run_agent(self, prompt: str, thinking: ChatMessage) -> None:
        log = self.query_one("#chat_log", VerticalScroll)
        try:
            response = await self.state.agent.run(prompt, target="user")
            thinking.remove()
            log.mount(ChatMessage(response, sender="agent"))
        except asyncio.CancelledError:
            thinking.remove()
            log.mount(ChatMessage("⏹ Cancelled.", sender="system"))
        except Exception as e:
            thinking.remove()
            log.mount(ChatMessage(f"❌ {e}", sender="system"))
        finally:
            log.scroll_end(animate=False)


# ─── Tasks pane ───────────────────────────────────────────────────────────────

class TasksPane(Container):
    """Live task orchestration dashboard."""

    CSS = """
    #tasks_container {
        height: 1fr;
        border: round $primary;
        border-title: " Tasks ";
        padding: 1;
        background: $background;
    }
    #tasks_header {
        height: auto;
        margin-bottom: 1;
        color: $primary;
        text-style: bold;
    }
    #task_list {
        height: 1fr;
        scrollbar-size: 1 1;
    }
    #task_input_row {
        height: auto;
        padding: 1 0 0 0;
    }
    #task_input {
        border: round $primary;
    }
    """

    def __init__(self, state: AppState, **kwargs) -> None:
        super().__init__(**kwargs)
        self.state = state

    def compose(self) -> ComposeResult:
        with Vertical(id="tasks_container"):
            yield Label("⚙️ Task Orchestrator", id="tasks_header")
            yield VerticalScroll(id="task_list")
            with Horizontal(id="task_input_row"):
                yield Input(placeholder="Spawn a new task… (Enter to submit)", id="task_input")

    def on_mount(self) -> None:
        self._update_header()

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        prompt = event.value.strip()
        if not prompt:
            return
        event.input.value = ""

        tm = self.state.task_manager
        if not tm:
            self.notify("No task manager", severity="error")
            return

        request = TaskRequest(prompt=prompt)
        task_id = await tm.spawn_task(request)

        tl = self.query_one("#task_list", VerticalScroll)
        tl.mount(TaskRow(task_id, prompt, "PENDING"))
        self.notify(f"Task {task_id} spawned")
        self._wait_for_task(task_id, prompt)

    @work(exclusive=False, name="task_wait")
    async def _wait_for_task(self, task_id: str, prompt: str) -> None:
        tm = self.state.task_manager
        if not tm:
            return
        status = await tm.wait_for_task(task_id)
        try:
            self.query_one(f"#task-{task_id}", TaskRow).remove()
        except Exception:
            pass
        self.query_one("#task_list", VerticalScroll).mount(
            TaskRow(task_id, prompt, status.status)
        )
        self._update_header()

    def _update_header(self) -> None:
        tm = self.state.task_manager
        if not tm:
            return
        s = tm.get_status()
        try:
            self.query_one("#tasks_header", Label).update(
                f"⚙️ Task Orchestrator — {s['active_workers']}/{s['max_workers']} workers"
            )
        except Exception:
            pass


# ─── Skills pane ──────────────────────────────────────────────────────────────

class SkillsPane(Container):
    """Browse crystallized skills from the skills/ directory."""

    CSS = """
    #skills_container {
        height: 1fr;
        border: round $primary;
        border-title: " Skills ";
        padding: 1;
        background: $background;
    }
    #skills_header {
        color: $primary;
        text-style: bold;
    }
    #skills_list {
        height: 1fr;
        scrollbar-size: 1 1;
    }
    #skills_search {
        height: auto;
        padding: 0 0 1 0;
    }
    #skills_search_input {
        border: round $primary;
    }
    .skill_entry {
        padding: 0 1;
        margin: 0 0;
    }
    """

    def __init__(self, state: AppState, **kwargs) -> None:
        super().__init__(**kwargs)
        self.state = state

    def compose(self) -> ComposeResult:
        with Vertical(id="skills_container"):
            yield Label("🎓 Crystallized Skills", id="skills_header")
            with Horizontal(id="skills_search"):
                yield Input(placeholder="Search skills…", id="skills_search_input")
            yield VerticalScroll(id="skills_list")

    def on_mount(self) -> None:
        self._load_skills()

    def _load_skills(self, query: str = "") -> None:
        skills_dir = Path(WORKSPACE) / "skills"
        sl = self.query_one("#skills_list", VerticalScroll)
        for child in list(sl.children):
            child.remove()

        if not skills_dir.exists():
            sl.mount(Static("[dim]No skills yet. Skills crystallize automatically after successful tasks.[/]", classes="skill_entry"))
            return

        md_files = sorted(skills_dir.glob("*.md"))
        if query:
            md_files = [f for f in md_files if query in f.stem.lower() or query in f.read_text(errors="replace").lower()]

        if not md_files:
            sl.mount(Static(f"[dim]No skills matching '{query}'.[/]", classes="skill_entry"))
            return

        for f in md_files:
            name = f.stem.replace("_", " ").title()
            sz = f.stat().st_size
            mtime = datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            sl.mount(Static(f"[bold]{name}[/]  [dim]{sz}B · {mtime}[/]", classes="skill_entry"))

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        self._load_skills(query=event.value.strip().lower())


# ─── Memory pane ──────────────────────────────────────────────────────────────

class MemoryPane(Container):
    """Search the hybrid memory store (semantic + keyword)."""

    CSS = """
    #memory_container {
        height: 1fr;
        border: round $primary;
        border-title: " Memory ";
        padding: 1;
        background: $background;
    }
    #memory_results {
        height: 1fr;
        scrollbar-size: 1 1;
    }
    #memory_search_row {
        height: auto;
        padding: 0 0 1 0;
    }
    #memory_search_input {
        border: round $primary;
    }
    #memory_header {
        color: $primary;
        text-style: bold;
    }
    .memory_entry {
        padding: 0 1;
        margin: 0 0;
    }
    """

    def __init__(self, state: AppState, **kwargs) -> None:
        super().__init__(**kwargs)
        self.state = state

    def compose(self) -> ComposeResult:
        with Vertical(id="memory_container"):
            yield Label("🧠 Memory Search", id="memory_header")
            with Horizontal(id="memory_search_row"):
                yield Input(placeholder="Search memories… (Enter to search)", id="memory_search_input")
            yield VerticalScroll(id="memory_results")

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        query = event.value.strip()
        if not query:
            return

        rp = self.query_one("#memory_results", VerticalScroll)
        for child in list(rp.children):
            child.remove()
        rp.mount(Static("[dim]Searching…[/]", classes="memory_entry"))

        if not self.state.agent:
            for child in list(rp.children):
                child.remove()
            rp.mount(Static("[red]No agent available[/]", classes="memory_entry"))
            return

        try:
            chunks = await self.state.agent.retriever.retrieve(query, top_k=10)
            for child in list(rp.children):
                child.remove()
            if not chunks:
                rp.mount(Static("[dim]No memories found.[/]", classes="memory_entry"))
                return
            for i, chunk in enumerate(chunks):
                content = chunk.get("content", "")[:300]
                score = chunk.get("score", 0)
                mtype = chunk.get("type", "?")
                rp.mount(Static(f"[bold]#{i+1}[/] [dim]{mtype} · score={score:.3f}[/]\n{content}", classes="memory_entry"))
        except Exception as e:
            for child in list(rp.children):
                child.remove()
            rp.mount(Static(f"[red]Error: {e}[/]", classes="memory_entry"))


# ─── Settings pane ────────────────────────────────────────────────────────────

class SettingsPane(Container):
    """Provider/model selector (dropdown), theme selector, Caveman toggle, etc."""

    CSS = """
    #settings_container {
        height: 1fr;
        border: round $primary;
        border-title: " Settings ";
        padding: 1 2;
        background: $background;
        scrollbar-size: 1 1;
    }
    .settings_label {
        text-style: bold;
        color: $primary;
        margin-top: 1;
        margin-bottom: 0;
    }
    .settings_section {
        margin-top: 0;
        margin-bottom: 1;
        padding: 0 0;
    }
    .settings_row {
        margin-top: 0;
        height: auto;
        color: $text-muted;
    }
    #provider_select {
        margin-top: 0;
        margin-bottom: 0;
    }
    #theme_select {
        margin-top: 0;
        margin-bottom: 0;
    }
    .settings_btn {
        margin-top: 0;
        margin-right: 1;
    }
    #settings_provider {
        color: $foreground;
    }
    #settings_caveman {
        color: $foreground;
    }
    #settings_workers {
        color: $text-muted;
    }
    """

    def __init__(self, state: AppState, **kwargs) -> None:
        super().__init__(**kwargs)
        self.state = state

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="settings_container"):
            yield Label("🔧 Settings", classes="settings_label")

            # ── Provider / Model selector ──────────────────────────────
            yield Label("Provider / Model", classes="settings_label")
            yield Label("  Switch the active model at runtime:", classes="settings_row")
            options = AppState.build_provider_options()
            # Select needs (display, value) tuples and a default value
            valid_values = {v for _, v in options}
            default_val = self.state.current_provider_id if self.state.current_provider_id in valid_values else (options[0][1] if options else Select.BLANK)
            yield Select(options, value=default_val, id="provider_select")
            # Show a human-friendly label for the active provider
            active_label = default_val
            for label, val in options:
                if val == default_val:
                    active_label = label
                    break
            yield Label(f"  Active: {active_label}", id="settings_provider")

            # ── Theme ────────────────────────────────────────────────────
            yield Label("Theme", classes="settings_label")
            theme_options = [(ThemeRegistry.get_theme(tid).name, tid) for tid in ThemeRegistry.theme_ids()]
            yield Select(theme_options, value=self.state.current_theme, id="theme_select")
            yield Button("Toggle Theme (Ctrl+T)", id="btn_toggle_theme", classes="settings_btn")

            # ── Caveman ──────────────────────────────────────────────────
            yield Label("Caveman Compression", classes="settings_label")
            cstatus = "ON" if self.state.caveman_enabled else "OFF"
            yield Label(f"  Status: {cstatus}", id="settings_caveman")
            yield Button("Toggle Caveman", id="btn_toggle_caveman", classes="settings_btn")

            # ── Workspace ────────────────────────────────────────────────
            yield Label("Workspace", classes="settings_label")
            yield Label(f"  {WORKSPACE}", classes="settings_row")

            # ── Dashboard ────────────────────────────────────────────────
            yield Label("Dashboard", classes="settings_label")
            yield Button("Open Dashboard ↗", id="btn_dashboard", classes="settings_btn")

            # ── Workers ──────────────────────────────────────────────────
            yield Label("Workers", classes="settings_label")
            w = f"  {os.cpu_count() * 2} max (CPU-aware)" if self.state.task_manager else "  Not initialized"
            yield Label(w, id="settings_workers")

    async def on_select_changed(self, event: Select.Changed) -> None:
        """Handle dropdown changes for provider and theme selectors."""
        if event.select.id == "provider_select":
            new_provider_id = event.value
            if new_provider_id == Select.BLANK:
                return
            # Verify API key is available before switching
            if not self.state.config_manager.has_api_key(new_provider_id):
                self.notify("🔒 No API key configured for this provider", severity="warning")
                # Revert the Select to the current provider
                valid_values = {v for _, v in AppState.build_provider_options()}
                current = self.state.current_provider_id if self.state.current_provider_id in valid_values else (list(valid_values)[0] if valid_values else "")
                self.query_one("#provider_select", Select).value = current
                return
            try:
                self.state.reconnect(new_provider_id)
                # Find the display label for the new provider
                new_label = new_provider_id
                for label, val in AppState.build_provider_options():
                    if val == new_provider_id:
                        new_label = label
                        break
                self.query_one("#settings_provider", Label).update(f"  Active: {new_label}")
                self.notify(f"Switched to {new_label}")
                # Update chat welcome message
                try:
                    chat_pane = self.app.screen.query_one(ChatPane)
                    log = chat_pane.query_one("#chat_log", VerticalScroll)
                    log.mount(ChatMessage(f"⚡ Switched to {new_label}", sender="system"))
                    log.scroll_end(animate=False)
                except Exception:
                    pass
            except Exception as e:
                self.notify(f"Failed to switch: {e}", severity="error")

        elif event.select.id == "theme_select":
            new_theme = event.value
            if new_theme == Select.BLANK:
                return
            self.state.current_theme = new_theme
            self.app.theme = new_theme  # Textual native — cascades everywhere
            self.notify(f"Theme → {ThemeRegistry.get_theme(new_theme).name}")

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id

        if bid == "btn_toggle_theme":
            themes = ThemeRegistry.theme_ids()
            idx = themes.index(self.state.current_theme)
            self.state.current_theme = themes[(idx + 1) % len(themes)]
            self.app.theme = self.state.current_theme
            # Sync the Select dropdown too
            self.query_one("#theme_select", Select).value = self.state.current_theme
            self.notify(f"Theme → {ThemeRegistry.get_theme(self.state.current_theme).name}")

        elif bid == "btn_toggle_caveman":
            self.state.caveman_enabled = not self.state.caveman_enabled
            if self.state.agent:
                self.state.agent.caveman.enabled = self.state.caveman_enabled
            s = "ON" if self.state.caveman_enabled else "OFF"
            self.query_one("#settings_caveman", Label).update(f"  Status: {s}")
            self.notify(f"Caveman: {s}")

        elif bid == "btn_dashboard":
            webbrowser.open(f"{DASHBOARD_URL}?key={DASHBOARD_ADMIN_KEY}")
            self.notify("Opening dashboard in browser…")


# ─── The App ──────────────────────────────────────────────────────────────────

class MotionTUI(App):
    """The top-level Motion Harness TUI application.

    Themes are registered as Textual-native themes so that setting
    ``self.theme = "dracula"`` cascades through every CSS ``$variable``
    in every widget — borders, backgrounds, accents, everything.
    """

    CSS = """
    Screen { background: $background; color: $foreground; }
    Header { background: $surface; }
    Footer { background: $surface; }
    TabbedContent { height: 1fr; }
    .msg { margin: 0 0; padding: 0 1; }
    .user_msg { color: $primary; text-style: bold; }
    .agent_msg { color: $foreground; }
    .system_msg { color: $text-muted; text-style: italic; }
    """

    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit"),
        Binding("ctrl+c", "request_cancel", "Cancel"),
    ]

    def __init__(self, model_config: Optional[ModelConfig] = None, provider_id: str = "", workspace: str = WORKSPACE, **kwargs) -> None:
        super().__init__(**kwargs)
        self.state = AppState()
        self._model_config = model_config
        self._provider_id = provider_id
        self._workspace = workspace

    def on_mount(self) -> None:
        # Register all themes with Textual's native system
        for tid in ThemeRegistry.theme_ids():
            ttheme = ThemeRegistry.get_textual_theme(tid)
            self.register_theme(ttheme)

        # Set initial theme
        self.theme = self.state.current_theme

        if self._model_config:
            self.state.agent = MotionAgent(self._model_config)
            self.state.task_manager = TaskManager(self._model_config, self._workspace)
            # Use the explicit provider_id (e.g. "ollama-cloud/gemma3:12b")
            # rather than model_config.name (a display name like "Ollama Cloud (gemma3:12b)")
            # which won't match Select option values.
            if self._provider_id:
                self.state.current_provider_id = self._provider_id
            else:
                # Fallback: try to match against known options
                options = AppState.build_provider_options()
                self.state.current_provider_id = options[0][1] if options else ""
            self.push_screen(MainScreen(self.state))
        else:
            self.push_screen(ProviderSelectScreen(self.state))

    def action_request_cancel(self) -> None:
        """Cancel the running agent chat — not a quit."""
        try:
            worker = self.workers.get("agent_chat")
            if worker:
                worker.cancel()
                self.notify("Request cancelled")
        except Exception:
            pass

    async def on_unmount(self) -> None:
        """Graceful shutdown: close provider and DB connections."""
        if self.state.agent:
            try:
                await asyncio.wait_for(self.state.agent.provider.close(), timeout=2.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass
            self.state.agent.memory.close()


def launch_tui(model_config: Optional[ModelConfig] = None, provider_id: str = "") -> None:
    """Entry point called from main.py."""
    app = MotionTUI(model_config=model_config, provider_id=provider_id)
    app.run()


if __name__ == "__main__":
    config = ModelConfig(name="Motion-TUI", endpoint="http://localhost", provider_type="local")
    launch_tui(model_config=config)