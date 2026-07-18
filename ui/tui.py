"""
Motion Harness — Professional TUI
==================================
Built on Textual with native theme switching and runtime provider selection.

Screens:
  - ProviderSelect: Pick a provider/model at startup
  - MainScreen:     Tabbed hub (Chat, Tasks, Skills, KB, Memory, Settings)

Key features:
  - Themes cascade through every widget via Textual's ``$variable`` system
  - Ctrl+T cycles themes instantly
  - Settings tab has a SelectableDropdown for switching provider/model at runtime
  - Ctrl+C cancels the current request; Ctrl+Q quits
  - KB tab: knowledge base for reference docs that don't become skills

Launch:  python main.py              → TUI (default)
         python main.py --chat       → old REPL
         python main.py --provider X → TUI with pre-selected provider
"""

from __future__ import annotations

import asyncio
import os
import re
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
from rich.console import Group
from rich.markdown import Markdown as RichMarkdown
from rich.text import Text

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
from core.orchestrator import TaskManager, TaskRequest, TaskStatus
from core.providers import ModelConfig
from main import MotionAgent
from ui.themes import ThemeRegistry

WORKSPACE = os.getenv("MOTION_WORKSPACE", os.getcwd())
KB_DIR = os.path.join(WORKSPACE, "knowledge")
DASHBOARD_URL = "https://localhost:7860/"
DASHBOARD_ADMIN_KEY = "ME27dXc6uoEC_dWXJCyPVDPN"


def _suppress_logging() -> None:
    """Redirect root logging to a file so it doesn't bleed into the TUI."""
    import logging
    log_path = os.path.join(WORKSPACE, "motion.log")
    handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    root = logging.getLogger()
    # Remove any StreamHandler (e.g. basicConfig's stderr handler)
    root.handlers = [h for h in root.handlers if not isinstance(h, logging.StreamHandler) or isinstance(h, logging.FileHandler)]
    root.addHandler(handler)
    root.setLevel(logging.INFO)


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
        self.auto_synthesis_enabled: bool = False
        self.ui_mode: str = "conservative"
        self.show_activity_rail: bool = True
        self.show_trace_panel: bool = False
        self.last_agent_response: str = ""
        self.last_turn_metrics: dict = {}
        self.session_metrics: dict = {
            "turns": 0,
            "prompt_tokens_est": 0,
            "output_tokens_est": 0,
            "total_tokens_est": 0,
            "estimated_cost_usd": 0.0,
        }

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
        self.agent.auto_skill_synthesis = self.auto_synthesis_enabled
        self.task_manager = TaskManager(model_config, WORKSPACE)
        self.current_provider_id = provider_id

    @staticmethod
    def build_provider_options() -> list[tuple[str, str]]:
        """Return [(display_label, provider_id), ...] for Select dropdowns.
        Only includes providers that have a configured API key (or are local)."""
        cm = ConfigManager()
        providers = cm.list_providers()
        providers_cfg = cm.get("providers", {}) or {}
        def _provider_priority(pid: str) -> tuple[int, str]:
            cfg = providers_cfg.get(pid, {})
            is_local = cfg.get("provider_type") == "local"
            is_ollama = "ollama" in pid.lower() or "ollama" in str(cfg.get("endpoint", "")).lower()
            # Lower tuple sorts first: local/ollama first, then alphabetic.
            return (0 if (is_local or is_ollama) else 1, pid)
        providers = sorted(providers, key=lambda p: _provider_priority(p[0]))
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


def _slugify_name(name: str) -> str:
    value = name.strip().lower().replace(" ", "_")
    return re.sub(r"[^a-z0-9_-]+", "", value)


def _skills_dir() -> Path:
    return Path(WORKSPACE) / "skills"


def _extract_reasoning_and_answer(text: str) -> tuple[str, str]:
    """Extract <think>...</think> blocks if present; return (reasoning, answer)."""
    if "<think>" not in text:
        return "", text
    reasoning_parts: list[str] = []
    answer = text
    while "<think>" in answer and "</think>" in answer:
        start = answer.find("<think>")
        end = answer.find("</think>", start)
        if end == -1:
            break
        chunk = answer[start + len("<think>"):end].strip()
        if chunk:
            reasoning_parts.append(chunk)
        answer = (answer[:start] + answer[end + len("</think>"):]).strip()
    return "\n\n".join(reasoning_parts).strip(), answer.strip()


# ─── Chat message widgets ─────────────────────────────────────────────────────

class UserMessage(Static):
    """A user message card — primary accent header, markdown body."""
    DEFAULT_CSS = """
    UserMessage {
        background: $surface;
        border-left: heavy $primary;
        border-right: blank;
        border-top: blank;
        border-bottom: blank;
        padding: 1 2;
        margin: 1 8 0 0;
        color: $text;
    }
    """


class ReasoningMessage(Static):
    """Collapsible-style block used to surface model reasoning stream."""
    DEFAULT_CSS = """
    ReasoningMessage {
        background: $panel;
        border-left: heavy $warning;
        border-right: blank;
        border-top: blank;
        border-bottom: blank;
        padding: 1 2;
        margin: 1 0 0 8;
        color: $text-muted;
        text-style: dim italic;
    }
    """

class AgentMessage(Static):
    """An agent message card — accent header, markdown body."""
    DEFAULT_CSS = """
    AgentMessage {
        background: $surface;
        border-left: heavy $accent;
        border-right: blank;
        border-top: blank;
        border-bottom: blank;
        padding: 1 2;
        margin: 1 0 0 8;
        color: $text;
    }
    """

class SystemMessage(Static):
    """A system/info message — muted, single-line metadata."""
    DEFAULT_CSS = """
    SystemMessage {
        color: $text-muted;
        text-style: dim;
        padding: 0 1;
        margin: 0 0 0 0;
    }
    """


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
        background: $surface;
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
        self.query_one("#provider_box", Container).border_title = " Motion Harness "
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


# ─── Global activity rail + main hub ─────────────────────────────────────────

class ActivityRail(Vertical):
    """Right-side global activity rail for live task visibility."""

    DEFAULT_CSS = """
    ActivityRail {
        width: 36;
        min-width: 30;
        max-width: 42;
        border-left: heavy $border;
        padding: 1 1;
        background: $panel;
    }
    #activity_header {
        color: $accent;
        text-style: bold;
        margin-bottom: 1;
    }
    #activity_list {
        height: 1fr;
        scrollbar-size: 1 1;
    }
    .activity_row {
        padding: 0 1;
        margin: 0 0 1 0;
        color: $text;
        background: $surface;
        border: round $border;
    }
    """

    def __init__(self, state: AppState, **kwargs) -> None:
        super().__init__(**kwargs)
        self.state = state

    def compose(self) -> ComposeResult:
        yield Label("⚡ Live Activity", id="activity_header")
        yield VerticalScroll(id="activity_list")

    def on_mount(self) -> None:
        self.set_interval(0.5, self.refresh_activity)
        self.refresh_activity()

    def refresh_activity(self) -> None:
        task_manager = self.state.task_manager
        if not task_manager:
            return

        status = task_manager.get_status()
        tasks = list(status["tasks"].values())
        tasks.sort(key=lambda t: t.start_time or datetime.min, reverse=True)
        running = sum(1 for t in tasks if t.status == "RUNNING")
        self.query_one("#activity_header", Label).update(f"⚡ Live Activity · {running} running")

        container = self.query_one("#activity_list", VerticalScroll)
        for child in list(container.children):
            child.remove()

        if not tasks:
            container.mount(Static("[dim]No tasks yet[/]", classes="activity_row"))
            return

        for task in tasks[:20]:
            icon = TaskRow.ICON.get(task.status, "•")
            color = TaskRow.COLOR.get(task.status, "white")
            snippet = task.prompt.replace("[", "\\[").replace("]", "\\]")
            snippet = snippet[:46] + "…" if len(snippet) > 46 else snippet
            latest = task.logs[-1] if task.logs else ""
            latest = latest.replace("[", "\\[").replace("]", "\\]")
            latest = latest[:58] + "…" if len(latest) > 58 else latest
            row = f"{icon} [{color}]{task.status}[/] [dim]{task.task_id}[/]\n{snippet}"
            if latest:
                row += f"\n[dim]{latest}[/]"
            container.mount(Static(row, classes="activity_row"))


# ─── Shortcuts overlay ────────────────────────────────────────────────────────

class ShortcutsOverlay(Screen):
    """Unified shortcuts help overlay generated from real bindings."""

    CSS = """
    ShortcutsOverlay {
        align: center middle;
    }
    #shortcuts_box {
        width: 78;
        max-height: 80%;
        border: round $primary;
        background: $surface;
        padding: 1 2;
        scrollbar-size: 1 1;
    }
    #shortcuts_title {
        color: $primary;
        text-style: bold;
        text-align: center;
        margin-bottom: 1;
    }
    #shortcuts_body {
        height: auto;
        max-height: 70%;
        scrollbar-size: 1 1;
    }
    .shortcut_section {
        color: $accent;
        text-style: bold;
        margin-top: 1;
        margin-bottom: 0;
    }
    .shortcut_row {
        color: $text;
        padding: 0 1;
    }
    #shortcuts_hint {
        color: $text-muted;
        text-align: center;
        margin-top: 1;
    }
    """

    BINDINGS = [
        Binding("escape", "dismiss_overlay", "Close"),
        Binding("question_sign", "dismiss_overlay", "Close"),
        Binding("q", "dismiss_overlay", "Close"),
    ]

    def __init__(self, main_screen: "MainScreen", **kwargs) -> None:
        super().__init__(**kwargs)
        self._main = main_screen

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="shortcuts_box"):
            yield Label("⌨️  Keyboard Shortcuts", id="shortcuts_title")
            yield VerticalScroll(id="shortcuts_body")
            yield Label("Press ? / Esc / q to close", id="shortcuts_hint")

    def on_mount(self) -> None:
        self.query_one("#shortcuts_box", VerticalScroll).border_title = " Shortcuts "
        body = self.query_one("#shortcuts_body", VerticalScroll)
        sections = self._collect_bindings()
        for section_title, rows in sections.items():
            body.mount(Label(section_title, classes="shortcut_section"))
            for key, label in rows:
                body.mount(Static(f"  [bold]{key}[/]  [dim]·[/]  {label}", classes="shortcut_row"))

    def _collect_bindings(self) -> dict:
        groups: dict[str, list[tuple[str, str]]] = {}
        seen_keys: set[str] = set()
        global_bindings = getattr(self._main, "BINDINGS", []) or []
        groups["Global"] = []
        for b in global_bindings:
            key = getattr(b, "key", "")
            label = getattr(b, "description", "") or key
            if not key or key in seen_keys:
                continue
            seen_keys.add(key)
            groups["Global"].append(self._pretty_key(key, label))
        try:
            chat = self._main.query_one(ChatPane)
            chat_bindings = getattr(chat, "BINDINGS", []) or []
            groups["Chat"] = []
            for b in chat_bindings:
                key = getattr(b, "key", "")
                label = getattr(b, "description", "") or key
                if not key or key in seen_keys:
                    continue
                seen_keys.add(key)
                groups["Chat"].append(self._pretty_key(key, label))
        except Exception:
            pass
        return {k: v for k, v in groups.items() if v}

    @staticmethod
    def _pretty_key(key: str, label: str) -> tuple[str, str]:
        pretty = (
            key.replace("ctrl+", "Ctrl+")
                .replace("alt+", "Alt+")
                .replace("shift+", "Shift+")
                .replace("_", " ")
        )
        pretty = pretty.replace("question sign", "?")
        return pretty, label

    def action_dismiss_overlay(self) -> None:
        self.app.pop_screen()


class MainScreen(Screen):
    """The main hub with Chat, Tasks, Skills, Memory, Settings tabs."""

    CSS = """
    #main_shell { height: 1fr; background: $background; }
    #main_body { width: 1fr; padding: 0 1 0 0; }
    #main_tabs { height: 1fr; }
    #session_metrics_footer {
        height: auto;
        color: $text-muted;
        background: $background;
        border-top: solid $border;
        padding: 0 2;
    }
    TabbedContent TabPane {
        padding: 0 0;
    }
    """

    BINDINGS = [
        Binding("ctrl+t", "toggle_theme", "Theme", priority=True),
        Binding("ctrl+b", "toggle_activity_rail", "Rail", priority=True),
        Binding("ctrl+right", "next_tab", "Next Tab", priority=True),
        Binding("ctrl+left", "prev_tab", "Prev Tab", priority=True),
        Binding("ctrl+1", "goto_tab('chat_tab')", "Chat", priority=True),
        Binding("ctrl+2", "goto_tab('tasks_tab')", "Tasks", priority=True),
        Binding("ctrl+3", "goto_tab('skills_tab')", "Skills", priority=True),
        Binding("ctrl+4", "goto_tab('kb_tab')", "KB", priority=True),
        Binding("ctrl+5", "goto_tab('memory_tab')", "Memory", priority=True),
        Binding("ctrl+6", "goto_tab('settings_tab')", "Settings", priority=True),
        Binding("alt+1", "goto_tab('chat_tab')", "Chat", priority=True),
        Binding("alt+2", "goto_tab('tasks_tab')", "Tasks", priority=True),
        Binding("alt+3", "goto_tab('skills_tab')", "Skills", priority=True),
        Binding("alt+4", "goto_tab('kb_tab')", "KB", priority=True),
        Binding("alt+5", "goto_tab('memory_tab')", "Memory", priority=True),
        Binding("alt+6", "goto_tab('settings_tab')", "Settings", priority=True),
        Binding("f1", "goto_tab('chat_tab')", "Chat", priority=True),
        Binding("f2", "goto_tab('tasks_tab')", "Tasks", priority=True),
        Binding("f3", "goto_tab('skills_tab')", "Skills", priority=True),
        Binding("f4", "goto_tab('kb_tab')", "KB", priority=True),
        Binding("f5", "goto_tab('memory_tab')", "Memory", priority=True),
        Binding("f6", "goto_tab('settings_tab')", "Settings", priority=True),
        Binding("ctrl+]", "next_tab", "Next Tab", priority=True),
        Binding("ctrl+[", "prev_tab", "Prev Tab", priority=True),
        Binding("ctrl+q", "quit", "Quit", priority=True),
        Binding("question_sign", "show_shortcuts", "Shortcuts", priority=True),
    ]

    def __init__(self, state: AppState, **kwargs) -> None:
        super().__init__(**kwargs)
        self.state = state

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="main_shell"):
            with Vertical(id="main_body"):
                with TabbedContent(id="main_tabs"):
                    with TabPane("💬 Chat", id="chat_tab"):
                        yield ChatPane(self.state)
                    with TabPane("⚙️ Tasks", id="tasks_tab"):
                        yield TasksPane(self.state)
                    with TabPane("🎓 Skills", id="skills_tab"):
                        yield SkillsPane(self.state)
                    with TabPane("📚 KB", id="kb_tab"):
                        yield KBPane(self.state)
                    with TabPane("🧠 Memory", id="memory_tab"):
                        yield MemoryPane(self.state)
                    with TabPane("🔧 Settings", id="settings_tab"):
                        yield SettingsPane(self.state)
            yield ActivityRail(self.state, id="activity_rail")
        yield Label("", id="session_metrics_footer")
        yield Footer()

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        # Chat-local buttons (trace/copy/stats/save) are handled in ChatPane.
        # No global tab buttons remain; tab headers are the canonical nav.
        pass

    def on_mount(self) -> None:
        if not self.state.show_activity_rail:
            self.query_one("#activity_rail", ActivityRail).styles.display = "none"
        self.refresh_session_footer()

    def refresh_session_footer(self) -> None:
        s = self.state.session_metrics or {}
        provider_hint = self.state.current_provider_id or "unknown"
        text = (
            f"Session · turns={s.get('turns', 0)} · "
            f"prompt≈{s.get('prompt_tokens_est', 0)} tok · "
            f"output≈{s.get('output_tokens_est', 0)} tok · "
            f"total≈{s.get('total_tokens_est', 0)} tok · "
            f"cost≈${s.get('estimated_cost_usd', 0.0):.4f} · "
            f"provider={provider_hint}"
        )
        try:
            self.query_one("#session_metrics_footer", Label).update(text)
        except Exception:
            pass

    def action_toggle_theme(self) -> None:
        themes = ThemeRegistry.theme_ids()
        idx = themes.index(self.state.current_theme)
        self.state.current_theme = themes[(idx + 1) % len(themes)]
        self.app.theme = self.state.current_theme
        self.notify(f"Theme → {self.state.current_theme}")

    def action_toggle_activity_rail(self) -> None:
        rail = self.query_one("#activity_rail", ActivityRail)
        self.state.show_activity_rail = not self.state.show_activity_rail
        rail.styles.display = "block" if self.state.show_activity_rail else "none"

    def _activate_tab(self, tab_id: str) -> None:
        tabs = self.query_one("#main_tabs", TabbedContent)
        tabs.active = tab_id
        focus_targets = {
            "chat_tab": "#chat_input",
            "tasks_tab": "#task_input",
            "skills_tab": "#skills_search_input",
            "kb_tab": "#kb_search_input",
            "memory_tab": "#memory_search_input",
        }
        selector = focus_targets.get(tab_id)
        if not selector:
            return
        try:
            self.query_one(selector, Input).focus()
        except Exception:
            pass

    def action_goto_tab(self, tab_id: str) -> None:
        self._activate_tab(tab_id)

    def action_next_tab(self) -> None:
        tabs = self.query_one("#main_tabs", TabbedContent)
        order = ["chat_tab", "tasks_tab", "skills_tab", "kb_tab", "memory_tab", "settings_tab"]
        current = tabs.active or order[0]
        try:
            idx = order.index(current)
        except ValueError:
            idx = 0
        self._activate_tab(order[(idx + 1) % len(order)])

    def action_prev_tab(self) -> None:
        tabs = self.query_one("#main_tabs", TabbedContent)
        order = ["chat_tab", "tasks_tab", "skills_tab", "kb_tab", "memory_tab", "settings_tab"]
        current = tabs.active or order[0]
        try:
            idx = order.index(current)
        except ValueError:
            idx = 0
        self._activate_tab(order[(idx - 1) % len(order)])

    def action_show_shortcuts(self) -> None:
        self.app.push_screen(ShortcutsOverlay(self))


# ─── Chat pane ────────────────────────────────────────────────────────────────

class ChatPane(Vertical):
    """Message history + input bar."""
    BINDINGS = [
        Binding("ctrl+shift+t", "toggle_trace_panel", "Trace", priority=True),
        Binding("ctrl+shift+c", "copy_last_response", "Copy", priority=True),
        Binding("f8", "toggle_trace_panel", "Trace", priority=True),
        Binding("f9", "copy_last_response", "Copy", priority=True),
    ]

    DEFAULT_CSS = """
    ChatPane {
        height: 1fr;
    }
    #chat_body {
        height: 1fr;
        padding: 0 0 1 0;
    }
    #chat_log {
        height: 1fr;
        width: 3fr;
        border: round $primary;
        padding: 1 2;
        scrollbar-size: 1 1;
        background: $surface;
    }
    #trace_panel {
        width: 42;
        height: 1fr;
        border: round $border;
        background: $panel;
        padding: 1 1;
        margin-left: 1;
    }
    #trace_header {
        color: $text-muted;
        text-style: bold;
        margin-bottom: 0;
    }
    #trace_log {
        height: 1fr;
        scrollbar-size: 1 1;
        border-top: solid $border;
        padding-top: 1;
    }
    #trace_summary_chip {
        height: auto;
        width: auto;
        padding: 0 2;
        margin: 0 0 1 0;
        background: $panel;
        border: round $border;
        color: $text-muted;
    }
    #chat_input_row {
        height: auto;
        padding: 1 1 1 1;
        background: $panel;
        border: solid $border;
    }
    #chat_controls {
        height: auto;
        width: auto;
        padding: 0 0 0 1;
        border-left: solid $border;
        margin-left: 1;
    }
    #chat_primary_actions, #chat_secondary_actions {
        height: auto;
        width: auto;
    }
    #chat_secondary_actions {
        margin-left: 1;
    }
    #chat_metrics {
        color: $text-muted;
        margin: 0 0 0 0;
        padding: 0 2;
    }
    #chat_input {
        height: 3;
        border: round $primary;
        background: $surface;
        width: 1fr;
    }
    .chat_btn {
        margin-left: 1;
        min-width: 10;
    }
    """

    def __init__(self, state: AppState, **kwargs) -> None:
        super().__init__(**kwargs)
        self.state = state
        self._last_trace_stage: str = ""

    def compose(self) -> ComposeResult:
        with Horizontal(id="chat_body"):
            yield VerticalScroll(id="chat_log")
            with Vertical(id="trace_panel"):
                yield Label("Interaction Trace", id="trace_header")
                yield VerticalScroll(id="trace_log")
        yield Label("", id="trace_summary_chip")
        yield Label("", id="chat_metrics")
        with Horizontal(id="chat_input_row"):
            yield Input(placeholder="Type a message… (Enter to send)", id="chat_input")
            with Horizontal(id="chat_controls"):
                with Horizontal(id="chat_primary_actions"):
                    yield Button("Trace", id="chat_toggle_trace_btn", classes="chat_btn")
                    yield Button("Copy", id="chat_copy_btn", classes="chat_btn")
                    yield Button("Stats", id="chat_stats_btn", classes="chat_btn")
                with Horizontal(id="chat_secondary_actions"):
                    yield Button("Save Skill", id="chat_save_skill_btn", classes="chat_btn")

    def on_click(self, event) -> None:
        if getattr(event.control, "id", None) == "trace_summary_chip":
            self._set_trace_panel_visible(True)

    def on_mount(self) -> None:
        name = self.state.agent.provider.config.name if self.state.agent else "?"
        log = self.query_one("#chat_log", VerticalScroll)
        log.border_title = " Chat "
        log.mount(SystemMessage(f"⚡ Motion Harness — connected to {name}"))
        log.mount(SystemMessage("Tip: Ctrl/Alt+1..6 or F1-F6 tabs · F8 Trace · F9 Copy · /skill save <name>"))
        self._append_trace("session_start", f"provider={name}")
        self._set_trace_panel_visible(self.state.show_trace_panel)
        self._refresh_metrics_bar()
        self.query_one("#chat_input", Input).focus()

    def _set_trace_panel_visible(self, visible: bool) -> None:
        self.state.show_trace_panel = visible
        panel = self.query_one("#trace_panel", Vertical)
        panel.styles.display = "block" if visible else "none"
        chip = self.query_one("#trace_summary_chip", Label)
        chip.styles.display = "none" if visible else "block"
        button = self.query_one("#chat_toggle_trace_btn", Button)
        button.label = "Trace On" if visible else "Trace"
        self._refresh_trace_chip()
        self.notify("Trace panel shown" if visible else "Trace panel hidden")
        # Preserve composer focus: expanding the trace panel must never steal focus
        # from the input unless the user explicitly clicked into the panel.
        try:
            self.query_one("#chat_input", Input).focus()
        except Exception:
            pass

    def _refresh_trace_chip(self) -> None:
        try:
            trace_log = self.query_one("#trace_log", VerticalScroll)
            chip = self.query_one("#trace_summary_chip", Label)
        except Exception:
            return
        count = len(trace_log.children)
        last_stage = self._last_trace_stage
        chip.update(f" trace · {count} events · {last_stage} " if last_stage else f" trace · {count} events ")

    def action_toggle_trace_panel(self) -> None:
        self._set_trace_panel_visible(not self.state.show_trace_panel)

    def _copy_last_response(self) -> None:
        text = (self.state.last_agent_response or "").strip()
        if not text:
            self.notify("No assistant response to copy yet.", severity="warning")
            return
        copy_fn = getattr(self.app, "copy_to_clipboard", None)
        if callable(copy_fn):
            try:
                copy_fn(text)
                self.notify("Copied last response to clipboard.")
                return
            except Exception:
                pass
        try:
            input_box = self.query_one("#chat_input", Input)
            input_box.value = text[:10000]
            input_box.focus()
            self.notify("Clipboard unavailable; response inserted into input for manual copy.", severity="warning")
            return
        except Exception:
            pass
        self.notify("Clipboard copy unavailable in this terminal.", severity="warning")

    def action_copy_last_response(self) -> None:
        self._copy_last_response()

    @staticmethod
    def _render_user_markdown(timestamp: str, text: str):
        safe = text.strip() or "_Empty message._"
        return Group(
            Text(f"{timestamp} • You", style="bold"),
            RichMarkdown(safe),
        )

    @staticmethod
    def _render_agent_markdown(timestamp: str, answer: str):
        safe_answer = answer.strip() or "_No response content._"
        return Group(
            Text(f"{timestamp} • Motion", style="dim"),
            RichMarkdown(safe_answer),
        )

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text:
            return
        event.input.value = ""

        log = self.query_one("#chat_log", VerticalScroll)
        if text.startswith("/skill"):
            await self._handle_skill_command(text, log)
            log.scroll_end(animate=False)
            return
        ts = datetime.now().strftime("%H:%M:%S")
        user_msg = UserMessage("")
        user_msg.update(self._render_user_markdown(ts, text))
        log.mount(user_msg)
        thinking = SystemMessage("⚙️ Thinking…")
        live_response = AgentMessage(f"[dim]{ts} • Motion[/]\n")
        log.mount(thinking)
        log.mount(live_response)
        log.scroll_end(animate=False)
        self._run_agent(text, thinking, live_response)

    def _refresh_metrics_bar(self) -> None:
        m = self.state.last_turn_metrics or {}
        s = self.state.session_metrics or {}
        provider = m.get("provider_type") or (
            getattr(self.state.agent.provider.config, "provider_type", "") if self.state.agent else ""
        )
        last_total = m.get("total_tokens_est", 0)
        session_total = s.get("total_tokens_est", 0)
        cost = s.get("estimated_cost_usd", 0.0)
        cost_text = "$0.00 local" if provider == "local" else f"${cost:.4f} est"
        text = (
            f"Last≈{last_total} tok · Session≈{session_total} tok · "
            f"Turns={s.get('turns', 0)} · Spend={cost_text}"
        )
        try:
            self.query_one("#chat_metrics", Label).update(text)
        except Exception:
            pass
    def _append_trace(self, event_type: str, detail: str = "") -> None:
        trace_log = self.query_one("#trace_log", VerticalScroll)
        ts = datetime.now().strftime("%H:%M:%S")
        label_map = {
            "memory_recall_start": "🧠 memory.recall.start",
            "memory_recall_done": "🧠 memory.recall.done",
            "model_start": "🤖 model.start",
            "stream_chunk": "🌊 stream.chunk",
            "model_done": "🤖 model.done",
            "finalize": "✅ finalize",
            "skill_synthesis_start": "🎓 skill.synthesis.start",
            "skill_synthesis_done": "🎓 skill.synthesis.done",
            "skill_synthesis_error": "🎓 skill.synthesis.error",
            "session_start": "⚡ session.start",
            "interaction_start": "▶ interaction.start",
            "interaction_error": "❌ interaction.error",
            "interaction_cancelled": "⏹ interaction.cancelled",
        }
        label = label_map.get(event_type, event_type)
        safe_detail = (detail or "").replace("[", "\\[").replace("]", "\\]")
        line = f"[dim]{ts}[/] {label}"
        if safe_detail:
            line += f" [dim]· {safe_detail[:220]}[/]"
        self._last_trace_stage = f"{label}"
        trace_log.mount(SystemMessage(line))
        trace_log.scroll_end(animate=False)
        self.query_one("#trace_header", Label).update(
            f"Interaction Trace ({len(trace_log.children)})"
        )
        self._refresh_trace_chip()

    @work(exclusive=True, name="agent_chat")
    async def _run_agent(self, prompt: str, thinking: SystemMessage, live_response: AgentMessage) -> None:
        log = self.query_one("#chat_log", VerticalScroll)
        chunks: list[str] = []
        first_chunk = True
        header_ts = datetime.now().strftime("%H:%M:%S")
        header = f"[dim]{header_ts} • Motion[/]\n"
        reasoning_widget: Optional[ReasoningMessage] = None
        current_raw = ""
        self._append_trace("interaction_start", prompt[:120])

        async def on_stream_chunk(chunk: str) -> None:
            nonlocal first_chunk
            if not chunk:
                return
            nonlocal reasoning_widget, current_raw
            chunks.append(chunk)
            current_raw = "".join(chunks)
            if first_chunk:
                first_chunk = False
                try:
                    thinking.remove()
                except Exception:
                    pass
            reasoning, answer = _extract_reasoning_and_answer(current_raw)
            if reasoning:
                if reasoning_widget is None:
                    reasoning_widget = ReasoningMessage("[dim]Reasoning stream[/]\n")
                    log.mount(reasoning_widget, before=live_response)
                reasoning_widget.update(f"[dim]Reasoning[/]\n{reasoning[:2500]}")
            live_response.update(header + (answer or ""))
        async def on_trace_event(*args) -> None:
            event_type = "trace"
            payload: Dict[str, Any] = {}
            if len(args) == 2:
                event_type = str(args[0])
                payload = args[1] if isinstance(args[1], dict) else {}
            elif len(args) == 1 and isinstance(args[0], dict):
                payload = args[0]
                event_type = str(payload.get("stage") or payload.get("event") or "trace")
            detail_parts: list[str] = []
            if isinstance(payload, dict):
                for key in ("query", "target", "provider", "model", "task_id", "status"):
                    value = payload.get(key)
                    if value is not None and value != "":
                        detail_parts.append(f"{key}={value}")
                        if len(detail_parts) >= 2:
                            break
            self._append_trace(event_type, ", ".join(detail_parts))
        try:
            response = await self.state.agent.run(
                prompt,
                target="user",
                on_stream_chunk=on_stream_chunk,
                on_trace_event=on_trace_event,
            )
            if not chunks:
                try:
                    thinking.remove()
                except Exception:
                    pass
                raw = response or ""
                reasoning, answer = _extract_reasoning_and_answer(raw)
                if reasoning:
                    reasoning_widget = ReasoningMessage(f"[dim]Reasoning[/]\n{reasoning[:2500]}")
                    log.mount(reasoning_widget, before=live_response)
                live_response.update(self._render_agent_markdown(header_ts, answer or ""))
                self.state.last_agent_response = answer or ""
            else:
                reasoning, answer = _extract_reasoning_and_answer("".join(chunks))
                self.state.last_agent_response = answer or ""
                live_response.update(self._render_agent_markdown(header_ts, answer or ""))
            est_prompt_tokens = max(1, len(prompt) // 4)
            est_output_tokens = max(1, len(self.state.last_agent_response or "") // 4)
            provider_type = getattr(self.state.agent.provider.config, "provider_type", "")
            est_cost_usd = 0.0 if provider_type == "local" else None
            self.state.last_turn_metrics = {
                "prompt_tokens_est": est_prompt_tokens,
                "output_tokens_est": est_output_tokens,
                "total_tokens_est": est_prompt_tokens + est_output_tokens,
                "estimated_cost_usd": est_cost_usd,
                "provider_type": provider_type,
            }
            session = self.state.session_metrics
            session["turns"] += 1
            session["prompt_tokens_est"] += est_prompt_tokens
            session["output_tokens_est"] += est_output_tokens
            session["total_tokens_est"] += est_prompt_tokens + est_output_tokens
            if isinstance(est_cost_usd, (int, float)):
                session["estimated_cost_usd"] += float(est_cost_usd)
            self._refresh_metrics_bar()
            main_screen = self.screen
            if isinstance(main_screen, MainScreen):
                main_screen.refresh_session_footer()
        except asyncio.CancelledError:
            try:
                thinking.remove()
            except Exception:
                pass
            live_response.remove()
            log.mount(SystemMessage("⏹ Cancelled."))
            self._append_trace("interaction_cancelled")
        except Exception as e:
            try:
                thinking.remove()
            except Exception:
                pass
            live_response.remove()
            log.mount(SystemMessage(f"❌ {e}"))
            self._append_trace("interaction_error", str(e))
        finally:
            log.scroll_end(animate=False)

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id
        log = self.query_one("#chat_log", VerticalScroll)
        if bid == "chat_save_skill_btn":
            await self._handle_skill_command("/skill save skill_from_chat", log)
            log.scroll_end(animate=False)
            return
        if bid == "chat_toggle_trace_btn":
            self.action_toggle_trace_panel()
            return
        if bid == "chat_copy_btn":
            self._copy_last_response()
            return
        if bid == "chat_stats_btn":
            m = self.state.last_turn_metrics or {}
            if not m:
                self.notify("No interaction stats yet. Send a message first.", severity="warning")
                return
            cost = m.get("estimated_cost_usd")
            cost_text = "$0.00 (local model)" if cost == 0.0 else "N/A"
            msg = (
                f"📊 Last turn — prompt≈{m.get('prompt_tokens_est', 0)} tok, "
                f"output≈{m.get('output_tokens_est', 0)} tok, "
                f"total≈{m.get('total_tokens_est', 0)} tok, cost≈{cost_text}"
            )
            log.mount(SystemMessage(msg))
            s = self.state.session_metrics
            log.mount(SystemMessage(
                f"📦 Session — turns={s.get('turns', 0)}, total≈{s.get('total_tokens_est', 0)} tok, "
                f"cost≈${s.get('estimated_cost_usd', 0.0):.4f}"
            ))
            log.scroll_end(animate=False)

    async def _handle_skill_command(self, text: str, log: VerticalScroll) -> None:
        parts = text.split(maxsplit=2)
        if len(parts) < 2:
            log.mount(SystemMessage("Usage: /skill save <name>  or  /skill delete <name>"))
            return
        action = parts[1].strip().lower()
        if action not in {"save", "delete"}:
            log.mount(SystemMessage("Unknown /skill action. Use save or delete."))
            return
        if len(parts) < 3 or not parts[2].strip():
            log.mount(SystemMessage("Provide a skill name, e.g. /skill save refactor_parser"))
            return

        skill_name = _slugify_name(parts[2])
        if not skill_name:
            log.mount(SystemMessage("Skill name can only include letters, numbers, '-' and '_'"))
            return
        skills_dir = _skills_dir()
        skills_dir.mkdir(parents=True, exist_ok=True)
        skill_path = skills_dir / f"{skill_name}.md"

        if action == "delete":
            if skill_path.exists():
                skill_path.unlink()
                log.mount(SystemMessage(f"🗑 Deleted skill: {skill_name}"))
            else:
                log.mount(SystemMessage(f"Skill not found: {skill_name}"))
            return

        content = self.state.last_agent_response.strip()
        if not content:
            log.mount(SystemMessage("No recent agent reply to save yet. Ask something first, then run /skill save <name>."))
            return
        with open(skill_path, "w", encoding="utf-8") as f:
            f.write(f"# {parts[2].strip()}\n\n{content}\n")
        log.mount(SystemMessage(f"✅ Saved skill from last reply: {skill_name}"))


# ─── Task detail screen ──────────────────────────────────────────────────────

class TaskDetailScreen(Screen):
    """Push-screen overlay showing full task detail."""

    DEFAULT_CSS = """
    TaskDetailScreen {
        align: center middle;
    }
    #detail_box {
        width: 80;
        height: 85%;
        border: round $primary;
        background: $surface;
        padding: 1 2;
        overflow-y: auto;
        scrollbar-size: 1 1;
    }
    #detail_header {
        text-style: bold;
        color: $primary;
        margin-bottom: 1;
    }
    #detail_back_btn {
        margin-right: 1;
    }
    .detail_section {
        color: $primary;
        text-style: bold;
        margin-top: 1;
        margin-bottom: 0;
    }
    .detail_prompt {
        color: $foreground;
        background: $surface;
        padding: 0 1;
        margin: 0 0;
    }
    .detail_result {
        color: $foreground;
        padding: 0 1;
        margin: 0 0;
    }
    .detail_error {
        color: red;
        text-style: bold;
        padding: 0 1;
        margin: 0 0;
    }
    .detail_meta {
        color: $text-muted;
        padding: 0 1;
        margin: 0 0;
    }
    .detail_log_line {
        color: $text-muted;
        padding: 0 1;
        margin: 0 0;
    }
    """

    BINDINGS = [
        Binding("escape", "pop_screen", "Back"),
    ]

    def __init__(self, task_id: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self._task_id = task_id
        self._task_manager: Optional[TaskManager] = None

    def compose(self) -> ComposeResult:
        with Vertical(id="detail_box"):
            yield Button("← Back", id="detail_back_btn", classes="settings_btn")
            yield Static("", id="detail_header")
            yield Static("", classes="detail_meta")

    def _render_task(self, task: TaskStatus) -> None:
        status_icon = TaskRow.ICON.get(task.status, "?")
        status_color = TaskRow.COLOR.get(task.status, "")
        header = f"{status_icon} [{status_color}]{task.status}[/] — {self._task_id}"
        self.query_one("#detail_header", Static).update(header)

        meta = ""
        if task.start_time:
            meta += f"Started {task.start_time.strftime('%H:%M:%S')}"
        if task.duration:
            meta += f"  ·  Duration: {task.duration}"
        if task.artifact_path:
            safe_path = task.artifact_path.replace("[", "\\[").replace("]", "\\]")
            meta += f"\n📄 {safe_path}"
        self.query_one(".detail_meta", Static).update(meta)

        box = self.query_one("#detail_box", Vertical)
        fixed_ids = {"detail_back_btn", "detail_header"}
        for child in list(box.children):
            if child.id in fixed_ids or "detail_meta" in child.classes:
                continue
            child.remove()

        box.mount(Label("Prompt", classes="detail_section"))
        safe_prompt = task.prompt.replace("[", "\\[").replace("]", "\\]")
        box.mount(Static(safe_prompt, classes="detail_prompt"))

        if task.result:
            box.mount(Label("Result", classes="detail_section"))
            safe_result = task.result.replace("[", "\\[").replace("]", "\\]")
            if len(safe_result) > 3000:
                safe_result = safe_result[:3000] + "…"
            box.mount(Static(safe_result, classes="detail_result"))

        if task.error:
            box.mount(Label("Error", classes="detail_section"))
            safe_error = task.error.replace("[", "\\[").replace("]", "\\]")
            box.mount(Static(safe_error, classes="detail_error"))

        if task.conversation:
            box.mount(Label("Conversation", classes="detail_section"))
            for turn in task.conversation:
                role = turn.get("role", "?")
                content = turn.get("content", "").replace("[", "\\[").replace("]", "\\]")
                role_icon = {"user": "👤", "agent": "🤖", "system": "⚡"}.get(role, "•")
                box.mount(Static(f"{role_icon} [{status_color if role == 'agent' else 'cyan'}]{role.title()}[/]: {content[:2000]}", classes="detail_result"))

        if task.logs:
            box.mount(Label("Log", classes="detail_section"))
            for line in task.logs:
                safe_line = line.replace("[", "\\[").replace("]", "\\]")
                box.mount(Static(safe_line, classes="detail_log_line"))

    async def _on_task_progress(self, status: TaskStatus) -> None:
        if status.task_id != self._task_id:
            return
        self._render_task(status)

    def on_mount(self) -> None:
        self.query_one("#detail_box", Vertical).border_title = " Task Detail "
        state = None
        app = self.app
        if isinstance(app, MotionTUI) and hasattr(app, "state"):
            state = app.state
        if not state or not state.task_manager:
            self.query_one("#detail_header", Static).update(f"Task {self._task_id}")
            return

        self._task_manager = state.task_manager
        self._task_manager.subscribe(self._task_id, self._on_task_progress)
        task = self._task_manager.get_task(self._task_id)
        if not task:
            self.query_one("#detail_header", Static).update(f"Task {self._task_id} — not found")
            return
        self._render_task(task)

    def on_unmount(self) -> None:
        if self._task_manager:
            self._task_manager.unsubscribe(self._task_id, self._on_task_progress)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "detail_back_btn":
            self.app.pop_screen()

    def action_pop_screen(self) -> None:
        self.app.pop_screen()


class TaskRow(Static):
    """A clickable task row in the task list."""

    DEFAULT_CSS = """
    TaskRow {
        padding: 1 1;
        margin: 0 0 1 0;
        background: $background 30%;
        border: round $border;
    }
    TaskRow:hover {
        background: $primary 15%;
        border: round $primary;
    }
    """

    ICON = {"PENDING": "⏳", "RUNNING": "⚙️", "COMPLETED": "✅", "FAILED": "❌"}
    COLOR = {"PENDING": "dim", "RUNNING": "bold yellow", "COMPLETED": "bold green", "FAILED": "bold red"}

    def __init__(self, task_id: str, prompt: str, status: str, **kwargs) -> None:
        self._task_id = task_id
        self._prompt = prompt
        self._status = status
        icon = self.ICON.get(status, "?")
        color = self.COLOR.get(status, "")
        display = f"{icon} [{color}]{status}[/] [dim]{task_id}[/]\n  {self._truncate(prompt, 80)}"
        super().__init__(display, id=f"task-{task_id}", **kwargs)

    @staticmethod
    def _truncate(text: str, max_len: int) -> str:
        safe = text.replace("[", "\\[").replace("]", "\\]")
        return safe[:max_len] + "…" if len(safe) > max_len else safe

    def update_status(self, status: str, prompt: str | None = None) -> None:
        if prompt:
            self._prompt = prompt
        self._status = status
        icon = self.ICON.get(status, "?")
        color = self.COLOR.get(status, "")
        display = f"{icon} [{color}]{status}[/] [dim]{self._task_id}[/]\n  {self._truncate(self._prompt, 80)}"
        self.update(display)

    def on_click(self) -> None:
        self.app.push_screen(TaskDetailScreen(self._task_id))


class TasksPane(Vertical):
    """Live task orchestration dashboard."""

    DEFAULT_CSS = """
    TasksPane {
        height: 1fr;
    }
    #tasks_container {
        height: 1fr;
        border: round $border;
        padding: 1;
        background: $surface;
    }
    #tasks_header_row {
        height: auto;
        margin-bottom: 1;
    }
    #tasks_header {
        color: $primary;
        text-style: bold;
        width: 1fr;
    }
    #tasks_count {
        color: $text-muted;
        width: auto;
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
        border: round $border;
    }
    """

    def __init__(self, state: AppState, **kwargs) -> None:
        super().__init__(**kwargs)
        self.state = state

    def compose(self) -> ComposeResult:
        with Vertical(id="tasks_container"):
            with Horizontal(id="tasks_header_row"):
                yield Label("⚙️ Tasks", id="tasks_header")
                yield Label("", id="tasks_count")
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
        task_id = await tm.spawn_task(request, progress_callback=self._on_task_progress)
        tl = self.query_one("#task_list", VerticalScroll)
        tl.mount(TaskRow(task_id, prompt, "PENDING"))
        self.notify(f"Task {task_id} spawned")
        self._update_header()
        self._wait_for_task(task_id, prompt)

    def _on_task_progress(self, status: 'TaskStatus') -> None:
        """Called by TaskManager on status transitions."""
        try:
            row = self.query_one(f"#task-{status.task_id}", TaskRow)
            row.update_status(status.status)
        except Exception:
            pass
        self._update_header()

    @work(exclusive=False, name="task_wait")
    async def _wait_for_task(self, task_id: str, prompt: str) -> None:
        tm = self.state.task_manager
        if not tm:
            return
        status = await tm.wait_for_task(task_id)
        # Update the TaskRow with final status (pane may not be mounted if user switched tabs)
        try:
            existing = self.query_one(f"#task-{task_id}", TaskRow)
            existing.update_status(status.status, prompt)
        except Exception:
            try:
                self.query_one("#task_list", VerticalScroll).mount(
                    TaskRow(task_id, prompt, status.status)
                )
            except Exception:
                pass
        self._update_header()

    def _update_header(self) -> None:
        tm = self.state.task_manager
        if not tm:
            return
        s = tm.get_status()
        total = len(s["tasks"])
        running = sum(1 for t in s["tasks"].values() if t.status == "RUNNING")
        done = sum(1 for t in s["tasks"].values() if t.status in ("COMPLETED", "FAILED"))
        try:
            self.query_one("#tasks_header", Label).update("⚙️ Tasks")
            self.query_one("#tasks_count", Label).update(
                f"{running} running · {done}/{total} done" if total else "no tasks yet"
            )
        except Exception:
            pass


# ─── Skills pane ──────────────────────────────────────────────────────────────

class SkillFileRow(Static):
    """Clickable skill row that loads content into the editor form."""

    def __init__(self, skill_path: Path, label: str, **kwargs) -> None:
        super().__init__(label, classes="skill_entry", **kwargs)
        self.skill_path = skill_path

    def on_click(self) -> None:
        try:
            pane = self.app.screen.query_one(SkillsPane)
            pane._load_skill_file(self.skill_path)
        except Exception:
            pass

class SkillsPane(Container):
    """Browse crystallized skills from the skills/ directory."""

    CSS = """
    #skills_container {
        height: 1fr;
        border: round $border;
        padding: 1;
        background: $surface;
    }
    #skills_header {
        color: $primary;
        text-style: bold;
        margin-bottom: 0;
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
        border: round $border;
    }
    #skills_editor_row {
        height: auto;
        padding: 1 0 0 0;
    }
    #skills_title_input {
        margin-right: 1;
    }
    #skills_content_input {
        border: round $border;
        margin-top: 1;
    }
    #skills_status {
        color: $text-muted;
        margin-top: 1;
    }
    .skill_entry {
        padding: 0 1;
        margin: 0 0 1 0;
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
            with Horizontal(id="skills_editor_row"):
                yield Input(placeholder="Skill name…", id="skills_title_input")
                yield Button("Use Last Reply", id="skills_use_last_reply_btn", classes="settings_btn")
                yield Button("Refine Draft", id="skills_refine_btn", classes="settings_btn")
                yield Button("Save/Update", id="skills_save_btn", classes="settings_btn")
                yield Button("Delete", id="skills_delete_btn", classes="settings_btn")
                yield Button("Clear", id="skills_clear_btn", classes="settings_btn")
            yield Input(placeholder="Skill content…", id="skills_content_input")
            yield Label("Tip: In chat, use /skill save <name> to save the last reply as a skill.", id="skills_status")
            yield VerticalScroll(id="skills_list")

    def on_mount(self) -> None:
        self.query_one("#skills_container", Vertical).border_title = " Skills "
        self._load_skills()

    def _load_skills(self, query: str = "") -> None:
        skills_dir = _skills_dir()
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
            sl.mount(SkillFileRow(f, f"[bold]{name}[/]  [dim]{sz}B · {mtime}[/]"))

    def _load_skill_file(self, skill_path: Path) -> None:
        try:
            raw = skill_path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            self.notify(f"Could not read skill: {e}", severity="error")
            return
        self.query_one("#skills_title_input", Input).value = skill_path.stem
        self.query_one("#skills_content_input", Input).value = raw.replace("\n", " ")[:8000]
        self.query_one("#skills_status", Label).update(f"Selected: {skill_path.name}")

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id
        if bid not in {"skills_save_btn", "skills_delete_btn", "skills_refine_btn", "skills_use_last_reply_btn", "skills_clear_btn"}:
            return
        if bid == "skills_clear_btn":
            self.query_one("#skills_title_input", Input).value = ""
            self.query_one("#skills_content_input", Input).value = ""
            self.query_one("#skills_status", Label).update("Draft cleared.")
            return
        if bid == "skills_use_last_reply_btn":
            content = (self.state.last_agent_response or "").strip()
            if not content:
                self.notify("No recent reply yet. Ask something in Chat first.", severity="warning")
                return
            self.query_one("#skills_content_input", Input).value = content.replace("\n", " ")[:8000]
            self.query_one("#skills_status", Label).update("Loaded last assistant reply into draft.")
            return
        title = self.query_one("#skills_title_input", Input).value.strip()
        if not title:
            self.notify("Enter a skill name first", severity="warning")
            return
        slug = _slugify_name(title)
        if not slug:
            self.notify("Invalid skill name", severity="warning")
            return
        skills_dir = _skills_dir()
        skills_dir.mkdir(parents=True, exist_ok=True)
        skill_path = skills_dir / f"{slug}.md"

        content = self.query_one("#skills_content_input", Input).value.strip()
        if bid == "skills_refine_btn":
            if not content:
                self.notify("Add draft content first, then refine.", severity="warning")
                return
            if not self.state.agent:
                self.notify("No active agent to refine draft.", severity="error")
                return
            prompt = (
                "Refine the following draft into a concise, reusable skill with clear steps and constraints. "
                "Return plain markdown only.\n\n"
                f"Skill name: {title}\n\nDraft:\n{content}"
            )
            self.query_one("#skills_status", Label).update("Refining draft…")
            try:
                refined = await self.state.agent.run(prompt, target="user")
                self.query_one("#skills_content_input", Input).value = (refined or "").replace("\n", " ")[:8000]
                self.query_one("#skills_status", Label).update("Draft refined. Review, edit, then Save/Update.")
                self.notify("Skill draft refined")
            except Exception as e:
                self.query_one("#skills_status", Label).update(f"Refine failed: {e}")
                self.notify(f"Refine failed: {e}", severity="error")
            return

        if bid == "skills_delete_btn":
            if not skill_path.exists():
                self.notify(f"Skill not found: {slug}", severity="warning")
                return
            skill_path.unlink()
            self.query_one("#skills_status", Label).update(f"Deleted: {skill_path.name}")
            self.notify(f"Deleted skill: {slug}")
            self._load_skills()
            return

        content = self.query_one("#skills_content_input", Input).value.strip()
        if not content:
            self.notify("Enter skill content first", severity="warning")
            return
        if not content.startswith("# "):
            content = f"# {title}\n\n{content}"
        with open(skill_path, "w", encoding="utf-8") as f:
            f.write(content.strip() + "\n")
        self.query_one("#skills_status", Label).update(f"Saved: {skill_path.name}")
        self.notify(f"Saved skill: {slug}")
        self._load_skills()

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "skills_search_input":
            self._load_skills(query=event.value.strip().lower())


# ─── Knowledge Base pane ──────────────────────────────────────────────────────
class KBEntryRow(Static):
    """Clickable KB row that loads entry content into editor fields."""

    def __init__(self, kb_path: Path, label: str, **kwargs) -> None:
        super().__init__(label, classes="kb_entry", **kwargs)
        self.kb_path = kb_path

    def on_click(self) -> None:
        try:
            pane = self.app.screen.query_one(KBPane)
            pane._load_kb_file(self.kb_path)
        except Exception:
            pass

class KBPane(Container):
    """Browse and search the knowledge base (reference docs that aren't skills)."""

    CSS = """
    #kb_container {
        height: 1fr;
        border: round $border;
        padding: 1;
        background: $surface;
    }
    #kb_header {
        color: $primary;
        text-style: bold;
        margin-bottom: 0;
    }
    #kb_list {
        height: 1fr;
        scrollbar-size: 1 1;
    }
    #kb_search {
        height: auto;
        padding: 0 0 1 0;
    }
    #kb_search_input {
        border: round $border;
    }
    #kb_add_row {
        height: auto;
        padding: 0 0 0 0;
    }
    #kb_add_title {
        height: 3;
        margin-right: 1;
    }
    #kb_add_btn {
        margin-right: 1;
    }
    #kb_delete_btn {
        margin-right: 1;
    }
    #kb_add_area {
        height: 5;
        margin-top: 1;
        border: round $border;
    }
    #kb_mode_select {
        margin-top: 1;
    }
    #kb_memory_search_row {
        height: auto;
        margin-top: 1;
    }
    #kb_memory_search_input {
        border: round $border;
    }
    #kb_memory_results {
        height: 8;
        scrollbar-size: 1 1;
        border: solid $border;
        background: $panel;
        padding: 0 1;
        margin-top: 1;
    }
    #kb_status {
        color: $text-muted;
        margin-top: 1;
    }
    .kb_entry {
        padding: 0 1;
        margin: 0 0 1 0;
    }
    """

    def __init__(self, state: AppState, **kwargs) -> None:
        super().__init__(**kwargs)
        self.state = state

    def compose(self) -> ComposeResult:
        with Vertical(id="kb_container"):
            yield Label("📚 Knowledge Base", id="kb_header")
            with Horizontal(id="kb_search"):
                yield Input(placeholder="Search knowledge base…", id="kb_search_input")
            yield VerticalScroll(id="kb_list")
            with Horizontal(id="kb_add_row"):
                yield Input(placeholder="Title for new entry…", id="kb_add_title")
                yield Button("Use Last Reply", id="kb_use_last_reply_btn", classes="settings_btn")
                yield Button("Save/Update", id="kb_add_btn", classes="settings_btn")
                yield Button("Delete", id="kb_delete_btn", classes="settings_btn")
                yield Button("Clear", id="kb_clear_btn", classes="settings_btn")
            yield Input(placeholder="Content or paste text…", id="kb_add_area")
            yield Select(
                [("Save + index to memory", "index"), ("Save only (no indexing)", "save")],
                value="index",
                id="kb_mode_select",
            )
            with Horizontal(id="kb_memory_search_row"):
                yield Input(placeholder="Search indexed memory from this tab…", id="kb_memory_search_input")
            yield VerticalScroll(id="kb_memory_results")
            yield Label("KB indexing mode controls whether entries are annexed into memory.", id="kb_status")

    def on_mount(self) -> None:
        self.query_one("#kb_container", Vertical).border_title = " Knowledge Base "
        self._load_kb()

    def _load_kb(self, query: str = "") -> None:
        kb_dir = Path(KB_DIR)
        kl = self.query_one("#kb_list", VerticalScroll)
        for child in list(kl.children):
            child.remove()

        if not kb_dir.exists():
            kl.mount(Static("[dim]No knowledge base entries yet. Use the form below to add one.[/]", classes="kb_entry"))
            return

        md_files = sorted(kb_dir.glob("*.md"))
        if query:
            md_files = [f for f in md_files if query in f.stem.lower() or query in f.read_text(errors="replace").lower()]

        if not md_files:
            kl.mount(Static(f"[dim]No entries matching '{query}'.[/]", classes="kb_entry"))
            return

        for f in md_files:
            name = f.stem.replace("_", " ").title()
            sz = f.stat().st_size
            mtime = datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            entry_type = "note"
            first_line = f.read_text(errors="replace").split("\n", 1)[0].lower()
            if first_line.startswith("http"):
                entry_type = "url"
            elif first_line.startswith("#"):
                entry_type = "doc"
            icon = {"doc": "📄", "url": "🔗", "snippet": "✂️", "note": "📝"}.get(entry_type, "📄")
            kl.mount(KBEntryRow(f, f"{icon} [bold]{name}[/]  [dim]{sz}B · {mtime}[/]"))

    def _load_kb_file(self, kb_path: Path) -> None:
        try:
            raw = kb_path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            self.notify(f"Could not read KB entry: {e}", severity="error")
            return
        title = kb_path.stem.replace("_", " ")
        body = raw
        if raw.startswith("# "):
            lines = raw.splitlines()
            title = lines[0][2:].strip() or title
            body = "\n".join(lines[2:]).strip()
        self.query_one("#kb_add_title", Input).value = title
        self.query_one("#kb_add_area", Input).value = body.replace("\n", " ")[:8000]
        self.query_one("#kb_status", Label).update(f"Selected: {kb_path.name}")

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "kb_search_input":
            self._load_kb(query=event.value.strip().lower())
        elif event.input.id == "kb_memory_search_input":
            await self._run_memory_search(event.value.strip())

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id
        if bid not in {"kb_add_btn", "kb_delete_btn", "kb_use_last_reply_btn", "kb_clear_btn"}:
            return
        if bid == "kb_clear_btn":
            self.query_one("#kb_add_title", Input).value = ""
            self.query_one("#kb_add_area", Input).value = ""
            self.query_one("#kb_status", Label).update("Draft cleared.")
            return
        if bid == "kb_use_last_reply_btn":
            content = (self.state.last_agent_response or "").strip()
            if not content:
                self.notify("No recent reply yet. Ask something in Chat first.", severity="warning")
                return
            self.query_one("#kb_add_area", Input).value = content.replace("\n", " ")[:8000]
            self.query_one("#kb_status", Label).update("Loaded last assistant reply into KB draft.")
            return
        title_input = self.query_one("#kb_add_title", Input)
        content_input = self.query_one("#kb_add_area", Input)
        title = title_input.value.strip()
        if not title:
            self.notify("Enter a title for the KB entry", severity="warning")
            return

        kb_dir = Path(KB_DIR)
        kb_dir.mkdir(parents=True, exist_ok=True)
        safe_name = _slugify_name(title)
        if not safe_name:
            self.notify("Invalid KB title", severity="warning")
            return
        file_path = kb_dir / f"{safe_name}.md"

        if bid == "kb_delete_btn":
            if not file_path.exists():
                self.notify(f"KB entry not found: {safe_name}", severity="warning")
                return
            file_path.unlink()
            self.query_one("#kb_status", Label).update(f"Deleted: {file_path.name}")
            self.notify(f"Deleted KB entry: {title}")
            self._load_kb()
            return

        content = content_input.value.strip()
        if not content:
            self.notify("Enter content for the KB entry", severity="warning")
            return
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"# {title}\n\n{content}\n")

        mode = self.query_one("#kb_mode_select", Select).value
        indexed = False
        if mode == "index" and self.state.agent:
            from memory.db import MemoryChunk
            try:
                embedding = await self.state.agent.get_embedding(content)
                self.state.agent.memory.add_memory(MemoryChunk(
                    content=content,
                    embedding=embedding,
                    metadata={"file": str(file_path), "type": "KB"},
                    mem_type="DOC",
                ))
                indexed = True
            except Exception:
                indexed = False

        title_input.value = ""
        content_input.value = ""
        annexed = "indexed to memory" if indexed else ("saved only" if mode == "save" else "saved (index failed)")
        self.query_one("#kb_status", Label).update(f"Saved: {file_path.name} · {annexed}")
        self.notify(f"Saved KB entry: {title} ({annexed})")
        self._load_kb()

    async def _run_memory_search(self, query: str) -> None:
        results = self.query_one("#kb_memory_results", VerticalScroll)
        for child in list(results.children):
            child.remove()
        if not query:
            results.mount(Static("[dim]Type a query and press Enter to search indexed memory.[/]", classes="kb_entry"))
            return
        if not self.state.agent:
            results.mount(Static("[red]No agent available[/]", classes="kb_entry"))
            return
        try:
            chunks = await self.state.agent.retriever.retrieve(query, top_k=6)
            if not chunks:
                results.mount(Static("[dim]No indexed memory results.[/]", classes="kb_entry"))
                return
            for i, chunk in enumerate(chunks):
                content = (chunk.get("content", "") or "").replace("[", "\\[").replace("]", "\\]")
                score = chunk.get("score", 0)
                results.mount(Static(f"[bold]#{i+1}[/] [dim]score={score:.3f}[/]\n{content[:240]}", classes="kb_entry"))
        except Exception as e:
            results.mount(Static(f"[red]Memory search failed: {e}[/]", classes="kb_entry"))


# ─── Memory pane ──────────────────────────────────────────────────────────────

class MemoryPane(Container):
    """Search the hybrid memory store (semantic + keyword)."""

    CSS = """
    #memory_container {
        height: 1fr;
        border: round $border;
        padding: 1;
        background: $surface;
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
        border: round $border;
    }
    #memory_header {
        color: $primary;
        text-style: bold;
        margin-bottom: 0;
    }
    .memory_entry {
        padding: 0 1;
        margin: 0 0 1 0;
        border-top: solid $border;
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

    def on_mount(self) -> None:
        self.query_one("#memory_container", Vertical).border_title = " Memory "

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
        border: round $border;
        padding: 1 2;
        background: $surface;
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

            yield Label("Provider / Model", classes="settings_label")
            yield Label("  Switch the active model at runtime:", classes="settings_row")
            options = AppState.build_provider_options()
            valid_values = {v for _, v in options}
            default_val = self.state.current_provider_id if self.state.current_provider_id in valid_values else (options[0][1] if options else Select.BLANK)
            yield Select(options, value=default_val, id="provider_select")
            active_label = default_val
            for label, val in options:
                if val == default_val:
                    active_label = label
                    break
            yield Label(f"  Active: {active_label}", id="settings_provider")

            yield Label("Theme", classes="settings_label")
            theme_options = [(ThemeRegistry.get_theme(tid).name, tid) for tid in ThemeRegistry.theme_ids()]
            yield Select(theme_options, value=self.state.current_theme, id="theme_select")
            yield Button("Toggle Theme (Ctrl+T)", id="btn_toggle_theme", classes="settings_btn")

            yield Label("Interface Style", classes="settings_label")
            yield Select(
                [("Conservative", "conservative"), ("Experimental", "experimental")],
                value=self.state.ui_mode,
                id="ui_mode_select",
            )

            yield Label("Global Activity Rail", classes="settings_label")
            rail_state = "ON" if self.state.show_activity_rail else "OFF"
            yield Label(f"  Status: {rail_state} (Ctrl+B)", id="settings_activity_rail")
            yield Button("Toggle Activity Rail", id="btn_toggle_activity_rail", classes="settings_btn")

            yield Label("Caveman Compression", classes="settings_label")
            cstatus = "ON" if self.state.caveman_enabled else "OFF"
            yield Label(f"  Status: {cstatus}", id="settings_caveman")
            yield Button("Toggle Caveman", id="btn_toggle_caveman", classes="settings_btn")

            yield Label("Auto Skill Synthesis", classes="settings_label")
            syn_status = "ON" if self.state.auto_synthesis_enabled else "OFF"
            yield Label(f"  Status: {syn_status} (crystallize skills from each reply)", id="settings_synthesis")
            yield Button("Toggle Auto-Synthesis", id="btn_toggle_synthesis", classes="settings_btn")

            yield Label("Workspace", classes="settings_label")
            yield Label(f"  {WORKSPACE}", classes="settings_row")

            yield Label("Dashboard", classes="settings_label")
            yield Button("Open Dashboard ↗", id="btn_dashboard", classes="settings_btn")

            yield Label("Workers", classes="settings_label")
            w = f"  {os.cpu_count() * 2} max (CPU-aware)" if self.state.task_manager else "  Not initialized"
            yield Label(w, id="settings_workers")

    async def on_select_changed(self, event: Select.Changed) -> None:
        """Handle dropdown changes for provider and theme selectors."""
        if event.select.id == "provider_select":
            new_provider_id = event.value
            if new_provider_id == Select.BLANK:
                return
            if not self.state.config_manager.has_api_key(new_provider_id):
                self.notify("🔒 No API key configured for this provider", severity="warning")
                valid_values = {v for _, v in AppState.build_provider_options()}
                current = self.state.current_provider_id if self.state.current_provider_id in valid_values else (list(valid_values)[0] if valid_values else "")
                self.query_one("#provider_select", Select).value = current
                return
            try:
                self.state.reconnect(new_provider_id)
                new_label = new_provider_id
                for label, val in AppState.build_provider_options():
                    if val == new_provider_id:
                        new_label = label
                        break
                self.query_one("#settings_provider", Label).update(f"  Active: {new_label}")
                self.notify(f"Switched to {new_label}")
                try:
                    chat_pane = self.app.screen.query_one(ChatPane)
                    log = chat_pane.query_one("#chat_log", VerticalScroll)
                    log.mount(SystemMessage(f"⚡ Switched to {new_label}"))
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
            self.app.theme = new_theme
            self.notify(f"Theme → {ThemeRegistry.get_theme(new_theme).name}")
        elif event.select.id == "ui_mode_select":
            mode = event.value
            if mode == Select.BLANK:
                return
            self.state.ui_mode = mode
            self.app.set_class(mode == "experimental", "experimental-ui")
            self.notify(f"UI mode → {mode}")

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id

        if bid == "btn_toggle_theme":
            themes = ThemeRegistry.theme_ids()
            idx = themes.index(self.state.current_theme)
            self.state.current_theme = themes[(idx + 1) % len(themes)]
            self.app.theme = self.state.current_theme
            self.query_one("#theme_select", Select).value = self.state.current_theme
            self.notify(f"Theme → {ThemeRegistry.get_theme(self.state.current_theme).name}")

        elif bid == "btn_toggle_caveman":
            self.state.caveman_enabled = not self.state.caveman_enabled
            if self.state.agent:
                self.state.agent.caveman.enabled = self.state.caveman_enabled
            s = "ON" if self.state.caveman_enabled else "OFF"
            self.query_one("#settings_caveman", Label).update(f"  Status: {s}")
            self.notify(f"Caveman: {s}")

        elif bid == "btn_toggle_synthesis":
            self.state.auto_synthesis_enabled = not self.state.auto_synthesis_enabled
            if self.state.agent:
                self.state.agent.auto_skill_synthesis = self.state.auto_synthesis_enabled
            s = "ON" if self.state.auto_synthesis_enabled else "OFF"
            self.query_one("#settings_synthesis", Label).update(
                f"  Status: {s} (crystallize skills from each reply)"
            )
            self.notify(f"Auto Skill Synthesis: {s}")

        elif bid == "btn_dashboard":
            webbrowser.open(f"{DASHBOARD_URL}?key={DASHBOARD_ADMIN_KEY}")
            self.notify("Opening dashboard in browser…")
        elif bid == "btn_toggle_activity_rail":
            try:
                if isinstance(self.app.screen, MainScreen):
                    self.app.screen.action_toggle_activity_rail()
                rail_state = "ON" if self.state.show_activity_rail else "OFF"
                self.query_one("#settings_activity_rail", Label).update(f"  Status: {rail_state} (Ctrl+B)")
            except Exception:
                pass


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
    .experimental-ui TaskRow {
        padding: 1 2;
        margin: 0 0 1 0;
    }
    .experimental-ui #chat_log {
        border: heavy $primary;
    }
    .experimental-ui #tasks_container {
        border: heavy $primary;
    }
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
        # Redirect logging to file so it doesn't bleed into the TUI
        _suppress_logging()

        # Register all themes with Textual's native system
        for tid in ThemeRegistry.theme_ids():
            ttheme = ThemeRegistry.get_textual_theme(tid)
            self.register_theme(ttheme)

        # Set initial theme
        self.theme = self.state.current_theme
        self.set_class(self.state.ui_mode == "experimental", "experimental-ui")

        if self._model_config:
            self.state.agent = MotionAgent(self._model_config)
            self.state.task_manager = TaskManager(self._model_config, self._workspace)
            if self._provider_id:
                self.state.current_provider_id = self._provider_id
            else:
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