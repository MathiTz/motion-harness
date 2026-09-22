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

Visual guardrails (R3 — Ops Rounded):
  - Maximum one permanent border per major region.
  - No adjacent parallel separators within 1 row of each other.
  - Spacing scale: 0, 1, 2 row gaps (2 reserved for section breaks).
  - Persistent accent area target below ~8-10% of screen.
  - Rounded corners on conversation bubbles and grouped setting blocks.
  - Both User and Motion replies render as explicit rounded balloons.
  - Accent is reserved for active tab, focused input, and high-signal state.

Launch:  python main.py              → TUI (default)
         python main.py --chat       → old REPL
         python main.py --provider X → TUI with pre-selected provider
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional
from rich.console import Group
from rich.markdown import Markdown as RichMarkdown
from rich.syntax import Syntax
from rich.style import Style
from rich.text import Text

from textual import events, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.screen import Screen
from textual.widgets import (
    Footer,
    Header,
    Input,
    Label,
    ListItem,
    ListView,
    LoadingIndicator,
    Select,
    Static,
)

from core.config import ConfigManager
from core.context import estimate_tokens
from core.orchestrator import TaskManager, TaskRequest
from core.pricing import format_cost, turn_cost
from core import trajectory as traj
from core.session import state_dir
from core.providers import ModelConfig
from core.session import SessionStore, state_dir
from core.skills import SkillLibrary, slugify
from core.toolstate import ToolSession
from core import auth
from main import MotionAgent, REPO_DIR
from ui.themes import ThemeRegistry

WORKSPACE = os.getenv("MOTION_WORKSPACE", os.getcwd())
KB_DIR = os.path.join(WORKSPACE, "knowledge")
logger = logging.getLogger(__name__)


def _suppress_logging() -> None:
    """Redirect root logging to a file so it doesn't bleed into the TUI."""
    import logging
    # The harness's own log lives with the harness, not in the user's project.
    log_path = os.path.join(REPO_DIR, "motion.log")
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
        self.mcp_manager = self._build_mcp_manager()
        self.current_provider_id: str = ""
        self.current_theme: str = "opencode"
        self.caveman_enabled: bool = True
        self.auto_synthesis_enabled: bool = False
        self.ui_mode: str = "conservative"
        self.show_activity_rail: bool = True
        self.show_trace_panel: bool = False
        # One-time nudge shown on the first agent turn pointing at F8/trace,
        # since the trace panel exists but is easy to miss (#9 in the issue
        # report: user found it by accident after asking for a checklist).
        self.seen_first_turn_hint: bool = False
        # When enabled, the agent's intermediate tool-loop responses (its
        # visible "thinking" between tool calls) are shown inline in the chat
        # log, not just summarized in the trace panel. Off by default since
        # it can be noisy; toggle with F7 or the command palette.
        self.show_thinking: bool = False
        # New asks start in "plan" (read-only, discuss-first). The agent only
        # gains write access ("build") once the user explicitly confirms via
        # a build-trigger phrase (see _is_build_trigger) or the Tab toggle.
        self.agent_mode: str = "plan"
        # Start mode: config `default_agent_mode: build` skips the discuss-first
        # plan step (risky commands still ask for approval either way).
        _configured_mode = self.config_manager.get("default_agent_mode")
        if _configured_mode in ("plan", "build"):
            self.agent_mode = _configured_mode
        self.busy: bool = False  # True while an agent response is streaming
        # Prompts submitted while busy are queued here instead of cancelling
        # the in-flight agent worker (which @work(exclusive=True) would
        # otherwise do). _run_agent drains this after each turn completes.
        self.message_queue: list[str] = []
        self.last_agent_response: str = ""
        self.prompt_history: list[str] = []  # submitted prompts for up/down recall
        self._history_index: int = -1
        # Paths outside the workspace root the user has approved "for this
        # session" via the out-of-workspace PermissionScreen (FR in the
        # issue report). Holds resolved Path objects (matching what
        # WorkspaceTools compares against) and is passed to
        # MotionAgent.run() on every turn so approvals persist without
        # re-prompting.
        # Approvals, undo checkpoints, read-tracking and todos that must
        # survive across turns. allowed_workspace_paths aliases the shared set.
        self.tool_session = ToolSession()
        self.allowed_workspace_paths: set[Path] = self.tool_session.allowed_paths
        # Per-session JSONL transcript path (prompt + full response per
        # turn), created lazily on first write. Only used when the user has
        # opted in via track_interactions in config.yml (asked once, at
        # first launch - see TrackingConsentScreen).
        self._session_store: Optional[SessionStore] = None
        self.todos: list[dict] = []
        # Edits made during the most recent turn, for the /diff command, and
        # whether to show them inline as they happen (config: show_diffs).
        self.turn_diffs: list[tuple[str, str, int, int]] = []
        # One record per model step of every turn this session (see core/trajectory.py),
        # plus the last turn's full message transcript for `/trajectory save full`.
        self.trajectory: list[dict] = []
        self.trajectory_turn: int = 0
        self.last_transcript: Optional[dict] = None
        self.show_diffs: bool = bool(self.config_manager.get("show_diffs", True))
        self.session_context: str = ""  # rolling, bounded summary of the session
        self._context_turns: list[tuple[str, str]] = []  # recent turns used to build context
        self.conversation_turns: list[tuple[str, str]] = []  # (prompt, response)
        # Files attached via `/attach <path>`; their extracted text / base64
        # is injected into the next prompt so the model can see the content.
        self.attachments: list[dict] = []
        self.last_turn_metrics: dict = {}
        self.session_metrics: dict = {
            "turns": 0,
            "prompt_tokens_est": 0,
            "output_tokens_est": 0,
            "total_tokens_est": 0,
            "estimated_cost_usd": 0.0,
            "unpriced_turns": 0,
        }

    def _build_mcp_manager(self):
        """Build an MCP manager from config.yml's optional `mcp:` block."""
        try:
            from core.mcp import MCPManager
        except Exception:
            return None
        servers_cfg = (self.config_manager.get("mcp") or {}).get("servers") or {}
        if not servers_cfg:
            return None
        try:
            return MCPManager(servers_cfg)
        except Exception:
            return None

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
                asyncio.get_running_loop().create_task(self.agent.provider.close())
            except RuntimeError:
                pass
            try:
                self.agent.memory.close()
            except Exception:
                pass
        self.agent = self.make_agent(model_config)
        self.task_manager = TaskManager(model_config, WORKSPACE, self.mcp_manager, self.config_manager.data)
        self.current_provider_id = provider_id

    def make_agent(self, model_config: ModelConfig) -> MotionAgent:
        """Build an agent wired to this session's config (permissions, memory,
        MCP)."""
        agent = MotionAgent(model_config, mcp_manager=self.mcp_manager)
        agent.auto_skill_synthesis = self.auto_synthesis_enabled
        agent.permissions_config = self.config_manager.data
        from core.agent_config import configure_agent

        configure_agent(agent, self.config_manager.get, self.config_manager)
        agent.auto_remember = bool(self.config_manager.get("remember_turns", True))
        try:
            agent.recall_timeout = float(self.config_manager.get("recall_timeout", 2.0))
        except (TypeError, ValueError):
            pass
        return agent

    @property
    def context_window(self) -> int:
        return int(getattr(getattr(self.agent, "provider", None), "context_window", 32768) or 32768)

    def history_tokens(self) -> int:
        """Rough token size of the conversation history sent with each turn."""
        return sum(estimate_tokens(p) + estimate_tokens(r) for p, r in self.conversation_turns[-8:])

    def needs_compaction(self, threshold: float = 0.6) -> bool:
        return self.history_tokens() >= self.context_window * threshold

    async def compact_with_model(self) -> str:
        """Replace the running conversation with a model-written summary.
        Returns the summary ("" if there was nothing to compact)."""
        if not self.conversation_turns or self.agent is None:
            return ""
        summary = await self.agent.summarize(self.conversation_turns)
        if not summary:
            return ""
        self.conversation_turns = [("[Summary of the conversation so far]", summary)]
        self._context_turns = list(self.conversation_turns)
        self.session_context = "[Compacted context]\n"
        self.session_metrics["total_tokens_est"] = 0
        self.session_metrics["prompt_tokens_est"] = 0
        self.session_metrics["output_tokens_est"] = 0
        return summary

    def new_session(self) -> None:
        """Start a fresh conversation (keeps provider, approvals and config)."""
        self.conversation_turns = []
        self._context_turns = []
        self.session_context = ""
        self.attachments.clear()
        self.todos = []
        self.trajectory = []
        self.trajectory_turn = 0
        self.last_transcript = None
        self.last_agent_response = ""
        self.message_queue.clear()
        self.tool_session.todos = []
        self.tool_session.read_files.clear()
        self.tool_session.checkpoints.entries.clear()
        self._session_store = None
        for k in self.session_metrics:
            self.session_metrics[k] = 0.0 if k == "estimated_cost_usd" else 0

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

    def record_prompt(self, prompt: str) -> None:
        """Record a submitted prompt for up/down history recall."""
        if not prompt:
            return
        if not self.prompt_history or self.prompt_history[-1] != prompt:
            self.prompt_history.append(prompt)
        self._history_index = len(self.prompt_history)

    def history_previous(self, current: str) -> str:
        """Return the previous prompt in history, or the current draft."""
        if not self.prompt_history:
            return current
        if self._history_index < 0:
            self._history_index = len(self.prompt_history)
        # Save the draft the first time we move up from the empty "new prompt" slot.
        if self._history_index == len(self.prompt_history):
            self._history_draft = current
        self._history_index = max(0, self._history_index - 1)
        return self.prompt_history[self._history_index]

    def history_next(self, current: str) -> str:
        """Return the next prompt in history, or the current draft."""
        if not self.prompt_history:
            return current
        if self._history_index >= len(self.prompt_history) - 1:
            self._history_index = len(self.prompt_history)
            return getattr(self, "_history_draft", current)
        self._history_index += 1
        return self.prompt_history[self._history_index]

    def update_session_context(self, prompt: str, response: str) -> None:
        """Maintain a rolling, bounded summary of the session.

        Keeps the most recent turns verbatim and folds older turns into a
        compact summary so the model always has aligned, up-to-date context
        without unbounded history growth.
        """
        self._context_turns.append((prompt, response))
        # Keep the last N turns verbatim; fold everything older into a summary.
        KEEP = 4
        if len(self._context_turns) > KEEP:
            older = self._context_turns[:-KEEP]
            self._context_turns = self._context_turns[-KEEP:]
            folded = "\n".join(f"Q: {p}\nA: {r[:200]}" for p, r in older)
            self.session_context = f"[Prior context]\n{folded}\n\n[Recent turns]\n"
        else:
            self.session_context = "[Recent turns]\n"

    def should_compact(self, context_window: int = 8192, threshold: float = 0.95) -> bool:
        """Return True when the session has consumed most of the context window."""
        if context_window <= 0:
            return False
        used = self.session_metrics.get("total_tokens_est", 0)
        return used >= context_window * threshold

    def compact(self) -> None:
        """Fold all current turns into a single prior-context summary."""
        if not self._context_turns:
            return
        folded = "\n".join(f"Q: {p}\nA: {r[:200]}" for p, r in self._context_turns)
        self.session_context = f"[Compacted context]\n{folded}\n"
        self._context_turns = []
        self.session_metrics["total_tokens_est"] = 0
        self.session_metrics["prompt_tokens_est"] = 0
        self.session_metrics["output_tokens_est"] = 0

    def build_context_prompt(self) -> str:
        """Return the session context block, used as a *query* to the vector DB."""
        if not self._context_turns and not self.session_context:
            return ""
        recent = "\n".join(f"Q: {p}\nA: {r[:300]}" for p, r in self._context_turns)
        return f"{self.session_context}{recent}"

    def context_summary(self) -> str:
        """Short one-line summary for the UI context row."""
        if not self._context_turns:
            return "No context yet"
        n = len(self._context_turns)
        last = self._context_turns[-1][0]
        return f"{n} turn{'s' if n != 1 else ''} · last: {last[:60]}"

    @staticmethod
    def build_all_provider_info() -> list[tuple[str, str, list, bool, bool]]:
        """Return all providers: (pid, name, models, is_default, has_key)."""
        cm = ConfigManager()
        return cm.list_providers()

    def log_interaction(self, prompt: str, response: str) -> None:
        """Append one turn (prompt + full response) to this session's JSONL
        transcript under <workspace>/.motion/sessions/, if the user has opted
        into tracking. The same transcript powers /resume."""
        if not self.config_manager.get("track_interactions"):
            return
        try:
            if self._session_store is None:
                self._session_store = SessionStore(WORKSPACE)
            self._session_store.append({
                "provider": self.current_provider_id,
                "prompt": prompt,
                "response": response,
            })
        except Exception:
            pass


def _slugify_name(name: str) -> str:
    return slugify(name)


def _skills_dir() -> Path:
    """Where /skill save writes: project-local, under the self-ignoring .motion/."""
    return state_dir(WORKSPACE, "skills")


# Phrases that count as an explicit go-ahead to start creating/editing files.
# Kept intentionally narrow (word-boundary anchored) to avoid false positives
# on messages that merely *describe* a build without asking for one yet.
_BUILD_TRIGGER_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\blet'?s build\b",
        r"\blet'?s do (it|this)\b",
        r"\blet'?s ship (it|this)\b",
        r"\bgo ahead\b",
        r"\bbuild it\b",
        r"\bbuild this\b",
        r"\bbuild that\b",
        r"\bship it\b",
        r"\bimplement it\b",
        r"\bimplement this\b",
        r"\bstart building\b",
        r"\byou can build\b",
        r"\byes,? build\b",
        r"\bmake it happen\b",
    )
]


def _is_build_trigger(text: str) -> bool:
    """Return True if the message is an explicit confirmation to start building."""
    return any(pattern.search(text) for pattern in _BUILD_TRIGGER_PATTERNS)


def _extract_reasoning_and_answer(text: str, streaming: bool = False) -> tuple[str, str]:
    """Extract <think>...</think> blocks if present; return (reasoning, answer).

    With ``streaming=True`` an unclosed trailing <think> (the model is still
    thinking) counts as reasoning instead of leaking into the answer."""
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
    if streaming and "<think>" in answer:
        idx = answer.find("<think>")
        pending = answer[idx + len("<think>"):].strip()
        if pending:
            reasoning_parts.append(pending)
        answer = answer[:idx].strip()
    return "\n\n".join(reasoning_parts).strip(), answer.strip()


# ─── Chat message widgets ─────────────────────────────────────────────────────

class UserMessage(Static):
    """User message — thin primary left accent bar, no box."""
    DEFAULT_CSS = """
    UserMessage {
        background: transparent;
        color: $text;
        border-left: thick $primary;
        padding: 0 2;
        margin: 1 0 0 1;
    }
    """

class ReasoningMessage(Static):
    """opencode-style Thinking block — muted header + dim italic body."""
    DEFAULT_CSS = """
    ReasoningMessage {
        background: $panel;
        color: $text-muted;
        border-left: solid $warning;
        padding: 0 2;
        margin: 0 0 1 1;
    }
    """

class ThinkingMessage(Static):
    """Opt-in live view of the agent's intermediate tool-loop responses.

    Distinct from ReasoningMessage (which renders <think> blocks from the
    final answer): this shows the model's visible text between tool calls
    as it works, when available and when the user has enabled it (F7 /
    command palette "Toggle agent thinking").
    """
    DEFAULT_CSS = """
    ThinkingMessage {
        background: $panel;
        color: $text-muted;
        border-left: solid $secondary;
        padding: 0 2;
        margin: 0 0 1 1;
    }
    """

class StepsMessage(Static):
    """Always-visible live view of tool activity (list/read/write/replace).

    Unlike ThinkingMessage (the model's free-form intermediate text, opt-in
    via show_thinking), this shows concrete tool operations - "wrote x.py",
    "read y.py" - so long-running tasks always show visible progress instead
    of a blank chat while many tool calls run in the background.
    """
    DEFAULT_CSS = """
    StepsMessage {
        background: $panel;
        color: $text-muted;
        border-left: solid $success;
        padding: 0 2;
        margin: 0 0 1 1;
    }
    """

class DiffMessage(Static):
    """A file edit shown as a colored unified diff (what the agent changed)."""
    DEFAULT_CSS = """
    DiffMessage {
        background: $panel;
        border-left: solid $warning;
        padding: 0 2;
        margin: 0 0 1 1;
    }
    """

    def show(self, path: str, diff: str, added: int, removed: int, code_theme: str, max_lines: int = 24) -> None:
        self.path = path
        self.diff_text = diff
        lines = diff.splitlines()
        body = "\n".join(lines[:max_lines])
        if len(lines) > max_lines:
            body += f"\n… {len(lines) - max_lines} more line(s) — /diff shows the full change"
        header = Text(f"± {path}  ", style="bold")
        header.append(f"+{added}", style="green")
        header.append(" ")
        header.append(f"−{removed}", style="red")
        self.update(Group(header, Syntax(body, "diff", theme=code_theme, word_wrap=True, background_color="default")))


class AgentMessage(Static):
    """Agent reply — no box, clean text flow, spaced below the user prompt.

    The color is intentionally unset so the Rich Markdown visual keeps its
    own theme-aware token colors (syntax highlighting) like opencode.
    """
    DEFAULT_CSS = """
    AgentMessage {
        background: transparent;
        padding: 0 2;
        margin: 1 0 1 1;
    }
    """

class SystemMessage(Static):
    """System/info message — muted, single-line."""
    DEFAULT_CSS = """
    SystemMessage {
        color: $text-muted;
        text-style: dim;
        padding: 0 2;
        margin: 0 0 0 0;
    }
    """


class ComposerSubmitted(Message):
    """Posted by ChatComposer when the user presses Enter."""

    def __init__(self, text: str) -> None:
        self.text = text
        super().__init__()


class ChatComposer(Static, can_focus=True):
    """Compact two-row prompt panel mirroring opencode's Prompt component.

    Row 0: single-line editable input with a themed cursor.
    Row 1: meta line (agent · model · provider) rendered as markup.
    The whole panel has a thick left border in the agent-mode color.
    """

    DEFAULT_CSS = """
    ChatComposer {
        height: auto;
        min-height: 4;
        max-height: 12;
        width: 1fr;
        background: $surface;
        color: $text;
        padding: 1 2 1 2;
        border: blank;
        border-left: solid $secondary;
        margin: 0;
    }
    ChatComposer:focus {
        border: blank;
        border-left: solid $secondary;
        background: $surface;
    }
    """

    # Pastes longer than this are collapsed to a "[LINES N]" placeholder
    # instead of dumping the raw text into the composer.
    PASTE_COLLAPSE_THRESHOLD = 100

    def __init__(self, state: AppState, placeholder: str = "Ask anything…", **kwargs) -> None:
        super().__init__(**kwargs)
        self._state = state
        self._placeholder = placeholder
        self.value: str = ""
        self.cursor_position: int = 0
        self.meta_markup: str = ""
        # Maps a "[LINES N]" placeholder literally embedded in self.value to
        # the real pasted text it stands in for. Expanded back on submit.
        self._pasted_blocks: dict[str, str] = {}
        # Slash-command + skill completions shown while typing a "/".
        self._suggestions: list[str] = []
        self._suggestion_index: int = 0

    # Slash commands offered during "/" completion.
    SLASH_COMMANDS = [
        ("/attach", "attach a file (path or browse)"),
        ("/clear", "drop all attached files"),
        ("/auth", "manage provider API keys"),
        ("/skill", "list/show/save/delete reusable skills"),
        ("/compact", "summarize the conversation to free context"),
        ("/undo", "revert the file changes of the last turn"),
        ("/diff", "show the last turn's edits (on|off toggles inline diffs)"),
        ("/trajectory", "steps, tokens and tools of the last turn (copy|save|all)"),
        ("/tracking", "save session transcripts locally (on|off)"),
        ("/effort", "reasoning effort for the model (low|medium|high|off)"),
        ("/budget", "per-turn limits (steps N | tokens N | cost X | seconds N | off)"),
        ("/new", "start a fresh conversation"),
        ("/resume", "list or reload a saved session"),
        ("/todos", "show the agent's task list"),
        ("/mcp", "show connected MCP servers and tools"),
        ("/jobs", "background processes (stop <id|all>)"),
        ("/synthesize", "toggle auto skill crystallization"),
        ("/parallel", "run sub-tasks on background workers"),
        ("/tools", "list available agent tools"),
        ("/help", "show commands + tools"),
    ]

    def set_meta(self, markup: str) -> None:
        """Update the second (meta) row."""
        self.meta_markup = markup
        self.refresh()

    def _update_suggestions(self) -> None:
        """Rebuild the completion list for slash-commands and saved skills."""
        self._suggestions = []
        self._suggestion_index = 0
        if not self.value.startswith("/"):
            return
        token = self.value[1:]  # everything after "/"
        for cmd, desc in self.SLASH_COMMANDS:
            if cmd[1:].startswith(token):
                self._suggestions.append(f"{cmd} — {desc}")
        # Saved skills (project + global) also autocomplete after "/skill ".
        try:
            if self.value.startswith("/skill "):
                query = self.value[len("/skill "):].strip().lower()
                for name, _desc in SkillLibrary.for_workspace(WORKSPACE).index():
                    if name.lower().startswith(query):
                        self._suggestions.append(f"/skill {name}")
        except Exception:
            pass

    def _current_suggestion(self) -> str:
        if self._suggestions:
            return self._suggestions[self._suggestion_index % len(self._suggestions)]
        return ""

    def _apply_suggestion(self) -> None:
        self._update_suggestions()
        if not self._suggestions:
            return
        text = self._current_suggestion()
        # Use just the "/command" part (strip the description).
        suggestion = text.split("—")[0].strip()
        # If the user is typing "/skill ..." and a named skill completes, fill
        # the full "/skill <name>" so args aren't clobbered or left dangling.
        if suggestion.startswith("/skill ") and len(suggestion) > len("/skill "):
            self.value = suggestion
        elif suggestion.startswith("/skill"):
            self.value = "/skill "
        else:
            token = suggestion.split()[0]
            self.value = token
        self.cursor_position = len(self.value)
        self._invalidate_layout()

    def _suggestions_markup(self) -> str:
        if not self._suggestions:
            return ""
        lines = [self._current_suggestion()]
        # dim trailing hints for the other suggestions (max a few)
        for i, s in enumerate(self._suggestions):
            if i == self._suggestion_index % len(self._suggestions):
                continue
            if len(lines) >= 5:
                break
            title = s.split("—")[0].strip()
            lines.append("[dim]" + title + "[/dim]")
        return "[blue]" + lines[0] + "[/]" + ("\n" + "\n".join(lines[1:]) if len(lines) > 1 else "")


    def _wrap_value(self) -> list[str]:
        """Soft-wrap the input value to the available content width."""
        width = max(10, self.size.width - self.styles.padding.left - self.styles.padding.right)
        if not self.value:
            return [self._placeholder]
        lines: list[str] = []
        for paragraph in self.value.split("\n"):
            if not paragraph:
                lines.append("")
                continue
            while len(paragraph) > width:
                lines.append(paragraph[:width])
                paragraph = paragraph[width:]
            lines.append(paragraph)
        return lines

    def _cursor_to_wrapped(self, width: int) -> tuple[int, int]:
        """Map a flat cursor offset to (wrapped_line_index, column)."""
        pos = min(self.cursor_position, len(self.value))
        paragraphs = self.value.split("\n")
        seen = 0
        for p, para in enumerate(paragraphs):
            para_len = len(para)
            if pos <= seen + para_len:
                col_in_para = pos - seen
                # How many wrapped lines preceded this paragraph?
                wline = sum(max(1, -(-len(q) // width)) for q in paragraphs[:p])
                wrapped_row = wline + (col_in_para // width)
                wrapped_col = col_in_para % width
                return wrapped_row, wrapped_col
            seen += para_len + 1
        # Cursor at very end.
        wline = sum(max(1, -(-len(q) // width)) for q in paragraphs)
        last = paragraphs[-1] if paragraphs else ""
        return max(0, wline - 1), len(last) % width

    def render(self) -> Text:
        width = max(10, self.size.width - self.styles.padding.left - self.styles.padding.right)
        wrapped = self._wrap_value()
        lines = [Text(line, style=self.rich_style) for line in wrapped]

        if self.has_focus:
            wline, wcol = self._cursor_to_wrapped(width)
            if wline >= len(lines):
                wline = len(lines) - 1
            if not lines[wline].plain:
                lines[wline] = Text(" ", style=self.rich_style)
            if wcol >= len(lines[wline].plain):
                lines[wline].append(" ")
                wcol = len(lines[wline].plain) - 1
            theme = self.app.get_theme(self.app.theme)
            from textual.color import Color
            primary = Color.parse(theme.primary).rich_color
            surface_color = theme.surface or theme.background or "#1e1e1e"
            surface = Color.parse(surface_color).rich_color
            lines[wline].stylize(Style(bgcolor=primary, color=surface), wcol, wcol + 1)

        line1 = Text.from_markup(self.meta_markup) if self.meta_markup else Text("")
        result = lines[0]
        for extra in lines[1:]:
            result = Text.assemble(result, "\n", extra)
        result = Text.assemble(result, "\n\n", line1)
        # Render slash-command / skill suggestions as a popup when typing "/".
        if self.value.startswith("/"):
            self._update_suggestions()
        if self._suggestions:
            popup = Text.from_markup(self._suggestions_markup())
            result = Text.assemble(result, "\n", popup)
        return result

    def _invalidate_layout(self) -> None:
        # Invalidate the cached content height so the panel re-sizes with the
        # number of (soft-wrapped) input lines.
        try:
            self._content_height_cache = None
        except Exception:
            pass
        self.refresh(repaint=True, layout=True)

    def _insert(self, char: str) -> None:
        pos = self.cursor_position
        self.value = self.value[:pos] + char + self.value[pos:]
        self.cursor_position = min(len(self.value), pos + len(char))
        self._invalidate_layout()

    def on_paste(self, event: events.Paste) -> None:
        """Insert terminal bracketed-paste text at the current cursor.

        Pastes over PASTE_COLLAPSE_THRESHOLD chars are collapsed to a
        "[LINES N]" placeholder so a large paste doesn't blow up the
        composer's display; the real text is substituted back in on submit.
        """
        event.stop()
        event.prevent_default()
        if not event.text:
            return
        # Preserve multiline prompts while normalizing terminal line endings.
        text = event.text.replace("\r\n", "\n").replace("\r", "\n")
        if len(text) <= self.PASTE_COLLAPSE_THRESHOLD:
            self._insert(text)
            return
        line_count = text.count("\n") + 1
        placeholder = f"[LINES {line_count}]"
        # Disambiguate same-line-count pastes within one draft so expansion
        # on submit maps each placeholder back to its own original text.
        suffix = 1
        unique = placeholder
        while unique in self._pasted_blocks:
            suffix += 1
            unique = f"[LINES {line_count}#{suffix}]"
        self._pasted_blocks[unique] = text
        self._insert(unique)

    def _delete(self) -> None:
        pos = self.cursor_position
        if pos < len(self.value):
            self.value = self.value[:pos] + self.value[pos + 1:]
            self._invalidate_layout()

    def _backspace(self) -> None:
        pos = self.cursor_position
        if pos > 0:
            self.value = self.value[: pos - 1] + self.value[pos:]
            self.cursor_position = pos - 1
            self._invalidate_layout()

    def _expand_pasted_placeholders(self, text: str) -> str:
        """Substitute "[LINES N]" placeholders back to their real pasted text."""
        if not self._pasted_blocks:
            return text
        for placeholder, original in self._pasted_blocks.items():
            text = text.replace(placeholder, original)
        return text

    def on_key(self, event) -> None:
        self._update_suggestions()
        if event.key == "tab" and self._suggestion_active():
            event.prevent_default()
            self._next_suggestion()
            return
        if event.key in ("up", "down") and self._suggestion_active():
            event.prevent_default()
            self._cycle_suggestion(1 if event.key == "down" else -1)
            return
        if (event.key == "enter" or event.key == "ctrl+s"):
            event.prevent_default()
            # Enter completes a partially typed command, but a command that is
            # already fully typed (e.g. "/help") must submit - otherwise it would
            # re-apply the same suggestion forever and never run.
            if self._suggestion_active() and self.value.strip() != self._current_suggestion().split("—")[0].strip():
                self._apply_suggestion()
                return
            expanded = self._expand_pasted_placeholders(self.value)
            self._pasted_blocks.clear()
            self.post_message(ComposerSubmitted(expanded))
            return
        if event.key == "up":
            event.prevent_default()
            self.value = self._state.history_previous(self.value)
            self.cursor_position = len(self.value)
            self.refresh()
        elif event.key == "down":
            event.prevent_default()
            self.value = self._state.history_next(self.value)
            self.cursor_position = len(self.value)
            self.refresh()
        elif event.key == "left":
            event.prevent_default()
            self.cursor_position = max(0, self.cursor_position - 1)
            self.refresh()
        elif event.key == "right":
            event.prevent_default()
            self.cursor_position = min(len(self.value), self.cursor_position + 1)
            self.refresh()
        elif event.key == "home":
            event.prevent_default()
            self.cursor_position = 0
            self.refresh()
        elif event.key == "end":
            event.prevent_default()
            self.cursor_position = len(self.value)
            self.refresh()
        elif event.key == "backspace":
            event.prevent_default()
            self._backspace()
        elif event.key == "delete":
            event.prevent_default()
            self._delete()
        elif event.character is not None and event.is_printable:
            event.prevent_default()
            self._insert(event.character)
        self._update_suggestions()

    def _suggestion_active(self) -> bool:
        return bool(self._suggestions)

    def _next_suggestion(self) -> None:
        if self._suggestions:
            self._suggestion_index = (self._suggestion_index + 1) % len(self._suggestions)
            self._invalidate_layout()

    def _cycle_suggestion(self, delta: int) -> None:
        if self._suggestions:
            self._suggestion_index = (self._suggestion_index + delta) % len(self._suggestions)
            self._invalidate_layout()

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
        border: round $border;
        padding: 1 3;
        background: $surface;
        overflow-y: auto;
    }
    #provider_title {
        text-align: center;
        text-style: bold;
        color: $text;
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
        # Config may have changed on disk since this ConfigManager was last
        # loaded (manual edit, or the agent itself writing config.yml).
        self.state.config_manager.reload()
        self.query_one("#provider_box", Container).border_title = " Motion Harness "
        providers = self.state.config_manager.list_providers()
        lv = self.query_one("#provider_list", ListView)
        for pid, name, models, is_default, has_key in providers:
            lv.append(ProviderOption(pid, name, models, is_default, has_key))
        if lv.children:
            lv.index = 0
        lv.focus()

    def action_select(self) -> None:
        lv = self.query_one("#provider_list", ListView)
        idx = lv.index
        if idx is None:
            return
        option = lv.children[idx]
        if not isinstance(option, ProviderOption):
            return

        if not option.has_key:
            # Instead of a dead-end warning, let the user enter a key right
            # here and reconnect automatically once it's saved.
            provider_id = option.provider_id
            models = option.models

            def _connect_after_auth() -> None:
                self.state.config_manager.reload()
                if not self.state.config_manager.has_api_key(provider_id):
                    self.notify("No API key saved; provider still locked.", severity="warning")
                    return
                provider_cfg = self.state.config_manager.get_provider_config(provider_id)
                default_model = provider_cfg.get("default_model") or (models[0] if models else None)
                full_id = f"{provider_id}/{default_model}" if default_model else provider_id
                try:
                    self.state.reconnect(full_id)
                except Exception as e:
                    self.notify(f"Connection failed: {e}", severity="error")
                    return
                try:
                    self.state.config_manager.set("last_provider", full_id)
                except Exception:
                    pass
                self.app.switch_screen(MainScreen(self.state))

            self.app.push_screen(AuthInputScreen(provider_id, self.state, on_saved=_connect_after_auth))
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
        try:
            self.state.config_manager.set("last_provider", full_id)
        except Exception:
            pass

        self.app.switch_screen(MainScreen(self.state))


# ─── Global activity rail + main hub ─────────────────────────────────────────

class ContextPanel(Vertical):
    """Right-side context panel showing session context and recent turns.

    Mirrors opencode's context/side panel: it reflects the rolling session
    summary and the most recent user/assistant turns so the operator can see
    what the model is working against.
    """

    DEFAULT_CSS = """
    ContextPanel {
        width: 38;
        min-width: 30;
        max-width: 44;
        border-left: blank;
        margin-left: 2;
        padding: 1 1;
        background: $panel;
    }
    #context_header {
        color: $text-muted;
        text-style: bold;
        margin-bottom: 1;
        padding: 0 1;
        background: transparent;
        border: blank;
    }
    #context_body {
        height: 1fr;
        scrollbar-size: 1 1;
    }
    .context_section {
        color: $text-muted;
        text-style: bold;
        margin-top: 1;
        margin-bottom: 0;
        padding: 0 1;
    }
    .context_turn {
        padding: 0 1;
        margin: 0 0 1 0;
        color: $text;
        background: transparent;
        border: blank;
    }
    .context_turn:first-child {
        border-top: blank;
    }
    """

    def __init__(self, state: AppState, **kwargs) -> None:
        super().__init__(**kwargs)
        self.state = state

    def compose(self) -> ComposeResult:
        yield Label("Context", id="context_header")
        yield VerticalScroll(id="context_body")

    def update_current_steps(self, lines: list[str]) -> None:
        """Live-update a "Current turn" checklist fed by the same tool-op
        lines shown inline in chat, so the user has an at-a-glance view of
        what step the agent is on without needing to open the trace panel
        (#9 in the issue report: no checklist to follow along with).
        """
        try:
            container = self.query_one("#context_body", VerticalScroll)
        except Exception:
            return
        try:
            existing = self.query_one("#current_steps_block", Static)
        except Exception:
            existing = None
        if not lines:
            if existing is not None:
                existing.remove()
            return
        text = "[bold]Current turn[/]\n" + "\n".join(f"• {s}" for s in lines[-8:])
        if existing is not None:
            existing.update(text)
        else:
            container.mount(Static(text, id="current_steps_block", classes="context_turn"), before=0)

    def update_todos(self, todos: list) -> None:
        """Show the model's todo_write checklist (kept at the top of the panel)."""
        try:
            container = self.query_one("#context_body", VerticalScroll)
        except Exception:
            return
        try:
            existing = self.query_one("#todos_block", Static)
        except Exception:
            existing = None
        if not todos:
            if existing is not None:
                existing.remove()
            return
        marks = {"completed": "[green]✓[/]", "in_progress": "[yellow]▶[/]", "pending": "[dim]○[/]"}
        lines = [
            f"{marks.get(t.get('status'), '○')} {str(t.get('content', '')).replace('[', chr(92) + '[')[:60]}"
            for t in todos[:12]
        ]
        text = "[bold]Todo[/]\n" + "\n".join(lines)
        if existing is not None:
            existing.update(text)
        else:
            container.mount(Static(text, id="todos_block", classes="context_turn"), before=0)

    def refresh_context(self) -> None:
        container = self.query_one("#context_body", VerticalScroll)
        for child in list(container.children):
            child.remove()
        if getattr(self.state, "todos", None):
            self.update_todos(self.state.todos)

        summary = self.state.context_summary()
        container.mount(Static(f"[dim]session · {summary}[/]", classes="context_turn"))

        # Resolve the theme primary to a hex for Rich markup.
        try:
            theme = self.app.get_theme(self.app.theme)
            primary = theme.primary or "#fab283"
        except Exception:
            primary = "#fab283"

        turns = getattr(self.state, "conversation_turns", None) or []
        if turns:
            container.mount(Label("Recent turns", classes="context_section"))
            for prompt, response in turns[-6:]:
                p = prompt.replace("[", "\\[").replace("]", "\\]")
                p = p[:44] + "…" if len(p) > 44 else p
                r = (response or "").replace("[", "\\[").replace("]", "\\]")
                r = r[:44] + "…" if len(r) > 44 else r
                container.mount(
                    Static(f"[{primary} bold]Q[/] {p}\n[dim]A[/] {r or '…'}", classes="context_turn")
                )
        else:
            container.mount(Static("[dim]No turns yet — start a conversation.[/]", classes="context_turn"))


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
        border: round $border;
        background: $surface;
        padding: 1 2;
        scrollbar-size: 1 1;
    }
    #shortcuts_title {
        color: $text;
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
        color: $text-muted;
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


# ─── Command palette + model dialog ─────────────────────────────────────────

class CommandPalette(Screen):
    """opencode-style command palette (Ctrl+K)."""

    CSS = """
    CommandPalette {
        align: center middle;
    }
    #palette_box {
        width: 72;
        max-height: 80%;
        border: round $border;
        background: $surface;
        padding: 1 2;
        scrollbar-size: 1 1;
    }
    #palette_input {
        margin-bottom: 1;
        border: solid $border;
    }
    #palette_list {
        height: auto;
        max-height: 60%;
        scrollbar-size: 1 1;
        padding: 0 1;
        border: blank;
    }
    #palette_hint {
        color: $text-muted;
        text-align: center;
        margin-top: 1;
    }
    """

    BINDINGS = [
        Binding("escape", "dismiss_palette", "Close", priority=True),
        Binding("up", "nav_up", "Up", show=False),
        Binding("down", "nav_down", "Down", show=False),
    ]

    def __init__(self, main_screen: "MainScreen", **kwargs) -> None:
        super().__init__(**kwargs)
        self._main = main_screen
        self._commands: list[tuple[str, str]] = []
        self._selection_index: Optional[int] = None

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="palette_box"):
            yield Input(placeholder="Type a command…", id="palette_input")
            yield ListView(id="palette_list")
            yield Label("↑↓ Navigate · Esc to close", id="palette_hint")

    def on_mount(self) -> None:
        self.query_one("#palette_box", VerticalScroll).border_title = " Commands "
        self._commands = [
            ("Switch model…", "model"),
            ("Toggle agent (build/plan)", "agent"),
            ("Theme menu", "theme"),
            ("Show shortcuts", "shortcuts"),
            ("Toggle trace panel", "trace"),
            ("Toggle agent thinking", "thinking"),
            ("Copy last response", "copy"),
            ("Copy last code block", "copy_code"),
            ("Copy trace log", "copy_trace"),
            ("Show turn trajectory (/trajectory)", "trajectory"),
            ("Copy turn trajectory", "copy_trajectory"),
            ("Toggle interaction tracking", "tracking"),
            ("Toggle context panel", "context"),
            ("Manage API keys (/auth)", "auth"),
            ("Close menu (Esc)", "close"),
        ]
        lv = self.query_one("#palette_list", ListView)
        for label, _ in self._commands:
            lv.append(ListItem(Label(label)))
        if lv.children:
            lv.index = 0
            self._selection_index = 0
        self.query_one("#palette_input", Input).focus()

    def on_input_changed(self, event: Input.Changed) -> None:
        query = event.value.strip().lower()
        lv = self.query_one("#palette_list", ListView)
        first_visible: Optional[int] = None
        for idx, (label, _) in enumerate(self._commands):
            visible = query in label.lower()
            item = lv.children[idx]
            item.display = visible
            if visible and first_visible is None:
                first_visible = idx
        # Re-clamp the highlight: with display-toggling, Textual keeps the
        # stale index, so point it at the first *visible* row. If nothing
        # matches, clear the highlight so Enter has no target.
        self._selection_index = first_visible
        if first_visible is None:
            lv.index = None
        else:
            lv.index = first_visible

    def action_nav_up(self) -> None:
        # The Input owns focus while typing, so arrow keys land here instead
        # of on the (sibling, unfocused) ListView - forward them manually,
        # stepping only over visible (non-filtered) rows.
        self._move_selection(-1)

    def action_nav_down(self) -> None:
        self._move_selection(1)

    def _move_selection(self, delta: int) -> None:
        lv = self.query_one("#palette_list", ListView)
        order = [i for i, (label, _) in enumerate(self._commands) if lv.children[i].display]
        if not order:
            self._selection_index = None
            lv.index = None
            return
        current = self._selection_index
        if current is None or current not in order:
            idx = order[0] if delta >= 0 else order[-1]
        else:
            pos = order.index(current)
            pos = max(0, min(len(order) - 1, pos + delta))
            idx = order[pos]
        self._selection_index = idx
        lv.index = idx

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        item = event.item
        if item is None:
            return
        if not item.display:
            return
        self._activate_label(item)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        # Enter in the input box: pick the highlighted / first filtered command.
        lv = self.query_one("#palette_list", ListView)
        item = lv.highlighted_child
        if item is None or not item.display:
            return
        event.stop()
        self._activate_label(item)

    def _activate_label(self, item) -> None:
        # Match by position rather than parsing the row's label text, which
        # isn't portable across Textual versions (Label.content only exists in
        # newer releases - older ones store it as Label._content). Items are
        # appended in _commands order and we track the highlight index, so
        # resolve straight to the command by index.
        idx = self._selection_index
        if idx is not None and 0 <= idx < len(self._commands):
            self._run(self._commands[idx][1])

    def _run(self, action: str) -> None:
        # Subscreen-opening commands (model, auth, theme, shortcuts) push
        # their own screen with the palette left open underneath, so Esc from
        # that submenu returns to the command palette instead of dumping you
        # back to the chat - handy for picking a different command. Immediate
        # toggles (agent/trace/thinking/copy/context) run right away and
        # close the palette first.
        if action in ("model", "auth", "theme", "shortcuts"):
            if action == "model":
                self._main.action_open_model_dialog()
            elif action == "auth":
                self._main.action_open_auth()
            elif action == "theme":
                self._main.action_open_theme_menu()
            elif action == "shortcuts":
                self._main.action_show_shortcuts()
            return
        self.app.pop_screen()
        if action == "close":
            return
        if action == "agent":
            try:
                self._main.query_one(ChatPane)._toggle_agent_mode()
            except Exception:
                pass
        elif action == "trace":
            try:
                self._main.query_one(ChatPane).action_toggle_trace_panel()
            except Exception:
                pass
        elif action == "thinking":
            try:
                self._main.query_one(ChatPane).action_toggle_thinking()
            except Exception:
                pass
        elif action == "copy":
            try:
                self._main.query_one(ChatPane).action_copy_last_response()
            except Exception:
                pass
        elif action == "copy_code":
            try:
                self._main.query_one(ChatPane).action_copy_last_code_block()
            except Exception:
                pass
        elif action == "context":
            self._main.action_toggle_context_panel()
        elif action in ("copy_trace", "trajectory", "copy_trajectory", "tracking"):
            try:
                chat = self._main.query_one(ChatPane)
                if action == "copy_trace":
                    chat.action_copy_trace()
                elif action == "trajectory":
                    chat.show_trajectory()
                elif action == "copy_trajectory":
                    chat.copy_trajectory()
                else:
                    chat.set_tracking("toggle")
            except Exception:
                pass

    def action_dismiss_palette(self) -> None:
        self.app.pop_screen()


class ModelDialog(Screen):
    """opencode-style model/provider switcher (Ctrl+O).

    Lists every model individually (all cloud models for a provider appear
    once its API key is set) with a live search filter.
    """

    CSS = """
    ModelDialog {
        align: center middle;
    }
    #model_box {
        width: 72;
        max-height: 80%;
        border: round $border;
        background: $surface;
        padding: 1 2;
        scrollbar-size: 1 1;
    }
    #model_title {
        color: $text;
        text-style: bold;
        text-align: center;
        margin-bottom: 1;
    }
    #model_input {
        margin-bottom: 1;
        border: solid $border;
    }
    #model_list {
        height: auto;
        max-height: 60%;
        scrollbar-size: 1 1;
        padding: 0 1;
        border: blank;
    }
    #model_hint {
        color: $text-muted;
        text-align: center;
        margin-top: 1;
    }
    """

    BINDINGS = [
        Binding("escape", "dismiss_model", "Close", priority=True),
        Binding("ctrl+r", "refresh_models", "Refresh", priority=True),
        Binding("ctrl+n", "add_model", "Add model", priority=True),
        Binding("up", "nav_up", "Up", show=False),
        Binding("down", "nav_down", "Down", show=False),
    ]

    def __init__(self, state: AppState, **kwargs) -> None:
        super().__init__(**kwargs)
        self.state = state
        self._entries: list[tuple[str, str]] = []  # (label, full_id)
        self._selection_index: Optional[int] = None
        self._populated = False  # ensures rows are mounted exactly once

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="model_box"):
            yield Label("Select Model", id="model_title")
            yield Input(placeholder="Search models…", id="model_input")
            yield ListView(id="model_list")
            yield Label("Type to filter · ↑↓ Navigate · Enter to select · Ctrl+R refresh · Ctrl+N add custom model · Esc to close", id="model_hint")

    def on_mount(self) -> None:
        # Config may have changed on disk (manual edit, or the agent itself
        # editing config.yml) since this ConfigManager was last loaded.
        self.state.config_manager.reload()
        self._entries = self._collect_entries()
        self.run_worker(self._populate_models(), thread=False)
        self.query_one("#model_input", Input).focus()

    def _render_models(self, query: str) -> None:
        q = query.strip().lower()
        lv = self.query_one("#model_list", ListView)
        # Populate/rebuild the rows when they don't match the current entry
        # set (first frame from on_mount, or after a refresh added models).
        # append()/clear() are async in Textual 3.x, so rows are added via the
        # awaitable _set_model_rows_async() (called from the populate/rebuild
        # helpers); live typing afterwards only toggles display (never
        # re-mounts), which preserves the ListView's selection.
        if not lv.children:
            return
        first_visible: Optional[int] = None
        for idx in range(min(len(self._entries), len(lv.children))):
            label, _full = self._entries[idx]
            visible = (not q) or q in label.lower()
            lv.children[idx].display = visible
            if visible and first_visible is None:
                first_visible = idx
        if first_visible is None:
            self._selection_index = None
            lv.index = None
        else:
            self._selection_index = first_visible
            lv.index = first_visible

    async def _set_model_rows_async(self, rows: list[ModelOption]) -> None:
        """Awaitable variant of row population for Textual 3.x and 8.x.

        Textual 3.x (used by the `motion` launcher) only accepts one ListItem
        per append() call, unlike 8.x which accepts a splat, so we loop and
        await each one to ensure children are present when we re-render."""
        lv = self.query_one("#model_list", ListView)
        if lv.children:
            await lv.clear()
        for row in rows:
            await lv.append(row)

    async def _populate_models(self) -> None:
        """Mount the entry rows once, awaiting so children exist before the
        first filter is applied; then re-render to set the initial highlight."""
        if self._populated:
            return
        self._populated = True
        await self._set_model_rows_async([ModelOption(label, full) for label, full in self._entries])
        query = self.query_one("#model_input", Input).value
        self._render_models(query)

    async def _rebuild_models(self, query: str) -> None:
        """Rebuild the row set from the current entries (after a refresh or
        screen resume where the model list may have changed), then re-render."""
        lv = self.query_one("#model_list", ListView)
        if lv.children:
            await self._set_model_rows_async([ModelOption(label, full) for label, full in self._entries])
        self._render_models(query)

    def on_screen_resume(self) -> None:
        """Re-sync the list whenever this dialog becomes active again, e.g.
        after returning from AddModelScreen."""
        self.state.config_manager.reload()
        self._entries = self._collect_entries()
        # on_mount's _populate_models owns the very first population; resume
        # (which also fires on first push) must not race it and re-append.
        if not self._populated:
            return
        try:
            query = self.query_one("#model_input", Input).value
        except Exception:
            query = ""
        self.run_worker(self._rebuild_models(query), thread=False)

    def action_add_model(self) -> None:
        self.app.push_screen(AddModelScreen(self.state))

    @work
    async def action_refresh_models(self) -> None:
        """Scrape the latest Ollama Cloud model list, persist it, and re-render."""
        self.notify("Refreshing model list…")
        try:
            from core.catalog import update_ollama_cloud_models
            added = await update_ollama_cloud_models()
        except Exception:
            added = 0
        # Reload the running config manager so newly persisted models appear
        # immediately and survive restarts.
        self.state.config_manager.reload()
        self._entries = self._collect_entries()
        await self._rebuild_models(self.query_one("#model_input", Input).value)
        if added:
            self.notify(f"Found {added} new model{'s' if added != 1 else ''}")
        else:
            self.notify("Model list is up to date")

    def _collect_entries(self) -> list[tuple[str, str]]:
        entries: list[tuple[str, str]] = []
        providers_cfg = (self.state.config_manager.get("providers") or {})
        for pid, name, models, is_default, has_key in AppState.build_all_provider_info():
            if not has_key:
                continue
            cfg = providers_cfg.get(pid, {}) or {}
            cfg_models = cfg.get("models", {}) or {}
            if models:
                for m in models:
                    full = f"{pid}/{m}"
                    meta = cfg_models.get(m, {}) or {}
                    label = f"{name} → {m}"
                    if meta.get("context_window"):
                        ctx = meta["context_window"]
                        label += f" · {ctx//1000}k ctx"
                    if meta.get("input_mtok") is not None and meta.get("output_mtok") is not None:
                        label += f" · ${meta['input_mtok']}/${meta['output_mtok']}/M"
                    entries.append((label, full))
            else:
                entries.append((f"{name}", pid))
        return entries

    def on_input_changed(self, event: Input.Changed) -> None:
        self._render_models(event.value)

    def action_nav_up(self) -> None:
        # The search Input owns focus, so forward arrow keys to the sibling
        # ListView, which never receives them directly while unfocused -
        # stepping only over visible (non-filtered) rows.
        self._move_selection(-1)

    def action_nav_down(self) -> None:
        self._move_selection(1)

    def _move_selection(self, delta: int) -> None:
        lv = self.query_one("#model_list", ListView)
        order = [i for i in range(len(lv.children)) if lv.children[i].display]
        if not order:
            lv.index = None
            return
        current = getattr(self, "_selection_index", None)
        if current is None or current not in order:
            idx = order[0] if delta >= 0 else order[-1]
        else:
            pos = order.index(current)
            pos = max(0, min(len(order) - 1, pos + delta))
            idx = order[pos]
        self._selection_index = idx
        lv.index = idx

    def on_input_submitted(self, event: Input.Submitted) -> None:
        lv = self.query_one("#model_list", ListView)
        item = lv.highlighted_child
        if item is None or not item.display:
            return
        event.stop()
        self._select(item)

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        if event.item is not None and event.item.display:
            self._select(event.item)

    def _select(self, item) -> None:
        if not isinstance(item, ModelOption):
            return
        full_id = item.full_id
        try:
            self.state.reconnect(full_id)
            # Persist so the next `motion` launch reconnects to this model
            # instead of always falling back to the catalog default.
            try:
                self.state.config_manager.set("last_provider", full_id)
            except Exception:
                pass
            self.notify(f"Switched to {full_id}")
            # self.app.screen is this dialog (or the command palette beneath
            # it if opened that way) - not MainScreen - so the old lookup
            # never matched and the composer meta / footer never refreshed.
            # Walk the whole screen stack to find MainScreen instead.
            try:
                for screen in self.app.screen_stack:
                    if isinstance(screen, MainScreen):
                        pane = screen.query_one(ChatPane)
                        pane._refresh_meta()
                        pane._refresh_connection_line()
                        screen.refresh_session_footer()
                        break
            except Exception:
                pass
        except Exception as e:
            self.notify(f"Connection failed: {e}", severity="error")
        self.app.pop_screen()

    def action_dismiss_model(self) -> None:
        self.app.pop_screen()


class ModelOption(ListItem):
    """A single selectable model row in the model dialog."""

    def __init__(self, label: str, full_id: str, **kwargs) -> None:
        self.full_id = full_id
        super().__init__(Label(label), **kwargs)


class FileEntry(ListItem):
    """A file/directory row in the file picker."""

    def __init__(self, path: Path, is_dir: bool, **kwargs) -> None:
        self.path = path
        self.is_dir = is_dir
        icon = "📁 " if is_dir else "📄 "
        label = f"{icon}{path.name}"
        if is_dir:
            label += "/"
        super().__init__(Label(label), **kwargs)


class FilePickerScreen(Screen):
    """Interactive file browser for attaching files.

    Navigate directories with ↑/↓ + Enter, go up with backspace, select a
    file with Enter to attach it (dismisses with the selected path).
    """

    CSS = """
    FilePickerScreen {
        align: center middle;
    }
    #picker_box {
        width: 80;
        height: 70%;
        border: round $border;
        background: $surface;
        padding: 1 2;
    }
    #picker_title {
        color: $text;
        text-style: bold;
        text-align: center;
        margin-bottom: 1;
    }
    #picker_path {
        color: $text-muted;
        margin-bottom: 1;
    }
    #picker_list {
        height: 1fr;
        border: solid $border;
        padding: 0 1;
        background: $surface;
    }
    #picker_hint {
        color: $text-muted;
        text-align: center;
        margin-top: 1;
    }
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel", priority=True),
        Binding("backspace", "go_up", "Up dir", show=False),
        Binding("up", "nav_up", "Up", show=False),
        Binding("down", "nav_down", "Down", show=False),
        Binding("enter", "choose", "Select", show=False),
    ]

    def __init__(self, start_dir: Path = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._cwd = (start_dir or Path.cwd()).expanduser()
        if not self._cwd.is_dir():
            self._cwd = Path.cwd()

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="picker_box"):
            yield Label("Attach a file", id="picker_title")
            yield Label("", id="picker_path")
            yield ListView(id="picker_list")
            yield Label("↑↓ navigate · Enter select · backspace up · Esc cancel", id="picker_hint")

    def on_mount(self) -> None:
        self._reload()

    def _reload(self) -> None:
        title = self.query_one("#picker_title", Label)
        path_label = self.query_one("#picker_path", Label)
        path_label.update(str(self._cwd))
        lv = self.query_one("#picker_list", ListView)
        lv.clear()
        items = []
        try:
            children = sorted(self._cwd.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        except Exception:
            children = []
        for child in children:
            try:
                if child.is_dir():
                    items.append(FileEntry(child, True))
                elif child.is_file():
                    items.append(FileEntry(child, False))
            except Exception:
                continue
        for it in items:
            lv.append(it)
        if len(self._cwd.parts) > 1:
            self._up_entry = FileEntry(self._cwd.parent, True)
        lv.index = 0
        lv.focus()

    def action_nav_up(self) -> None:
        self.query_one("#picker_list", ListView).action_cursor_up()

    def action_nav_down(self) -> None:
        self.query_one("#picker_list", ListView).action_cursor_down()

    def action_go_up(self) -> None:
        parent = self._cwd.parent
        if parent and parent.is_dir() and parent != self._cwd:
            self._cwd = parent
            self._reload()

    def action_choose(self) -> None:
        lv = self.query_one("#picker_list", ListView)
        item = lv.highlighted_child
        if not isinstance(item, FileEntry):
            return
        if item.is_dir:
            self._cwd = item.path
            self._reload()
        else:
            self.dismiss(item.path)

    def action_cancel(self) -> None:
        self.dismiss(None)


class AddModelScreen(Screen):
    """Add a custom model to a provider (fixes: only config.yml models were
    usable, with no in-app way to add one - #7 in the issue report)."""

    CSS = """
    AddModelScreen {
        align: center middle;
    }
    #addmodel_box {
        width: 72;
        border: round $border;
        background: $surface;
        padding: 1 2;
    }
    #addmodel_title {
        color: $text;
        text-style: bold;
        text-align: center;
        margin-bottom: 1;
    }
    #addmodel_label {
        color: $text-muted;
        margin-top: 1;
    }
    .addmodel_input {
        margin-bottom: 1;
        border: solid $border;
    }
    #addmodel_hint {
        color: $text-muted;
        text-align: center;
    }
    """

    BINDINGS = [
        Binding("escape", "dismiss_add_model", "Close", priority=True),
    ]

    def __init__(self, state: AppState, **kwargs) -> None:
        super().__init__(**kwargs)
        self.state = state

    def compose(self) -> ComposeResult:
        with Container(id="addmodel_box"):
            yield Label("Add Custom Model", id="addmodel_title")
            yield Label("Provider id (e.g. ollama-cloud):", id="addmodel_label")
            yield Input(placeholder="provider id…", id="addmodel_provider_input", classes="addmodel_input")
            yield Label("Model name:", id="addmodel_label_2")
            yield Input(placeholder="model name…", id="addmodel_model_input", classes="addmodel_input")
            yield Label("Enter in either field to save · Esc to cancel", id="addmodel_hint")

    def on_mount(self) -> None:
        # Pre-fill the provider field with the currently connected provider
        # for convenience (the common case: adding another model for it).
        base_provider = (self.state.current_provider_id or "").split("/", 1)[0]
        if base_provider:
            self.query_one("#addmodel_provider_input", Input).value = base_provider
        self.query_one("#addmodel_model_input", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        provider_id = self.query_one("#addmodel_provider_input", Input).value.strip()
        model_name = self.query_one("#addmodel_model_input", Input).value.strip()
        if not provider_id or not model_name:
            self.notify("Provide both a provider id and a model name.", severity="warning")
            return
        try:
            self.state.config_manager.add_model(provider_id, model_name)
        except Exception as e:
            self.notify(f"Could not add model: {e}", severity="error")
            return
        self.notify(f"Added {model_name} to {provider_id}")
        self.app.pop_screen()

    def action_dismiss_add_model(self) -> None:
        self.app.pop_screen()


class PermissionOption(ListItem):
    """A choice row in the out-of-workspace permission prompt."""

    def __init__(self, label: str, choice: str, **kwargs) -> None:
        self.choice = choice
        super().__init__(Label(label), **kwargs)


class PermissionScreen(Screen):
    """Modal asking the user to approve a tool call that would touch a path
    outside the workspace root, instead of the previous hard failure (FR in
    the issue report: "não acessa coisa fora do workspace, tinha que pedir
    permissão"). Returned via ``dismiss()`` so callers can ``await
    app.push_screen_wait(PermissionScreen(path))``.
    """

    CSS = """
    PermissionScreen {
        align: center middle;
    }
    #permission_box {
        width: 76;
        border: round $warning;
        background: $surface;
        padding: 1 2;
    }
    #permission_title {
        color: $warning;
        text-style: bold;
        margin-bottom: 1;
    }
    #permission_path {
        color: $text;
        margin-bottom: 1;
    }
    #permission_list {
        height: auto;
        border: blank;
        padding: 0 1;
    }
    #permission_hint {
        color: $text-muted;
        text-align: center;
        margin-top: 1;
    }
    """

    BINDINGS = [
        Binding("escape", "deny", "Deny", priority=True),
    ]

    def __init__(self, path: str, title: str = "⚠ Out-of-workspace access requested", detail: str = "", **kwargs) -> None:
        super().__init__(**kwargs)
        self.path = path
        self.title_text = title
        self.detail = detail

    def compose(self) -> ComposeResult:
        with Container(id="permission_box"):
            yield Label(self.title_text, id="permission_title")
            safe = self.path.replace("[", "\\[")
            body = f"[dim]{safe}[/]"
            if self.detail:
                body += f"\n[dim italic]{self.detail.replace('[', chr(92) + '[')}[/]"
            yield Static(body, id="permission_path")
            yield ListView(id="permission_list")
            yield Label("Enter to choose · Esc to deny", id="permission_hint")

    def on_mount(self) -> None:
        lv = self.query_one("#permission_list", ListView)
        lv.append(PermissionOption("Allow once", "once"))
        lv.append(PermissionOption("Allow for this session", "session"))
        lv.append(PermissionOption("Deny", "deny"))
        lv.index = 0
        lv.focus()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        choice = event.item.choice if isinstance(event.item, PermissionOption) else "deny"
        self.dismiss(choice)

    def action_deny(self) -> None:
        self.dismiss("deny")


class AskUserScreen(Screen):
    """Modal for the model's ``ask_user`` tool: pick a suggested answer or type
    one. Dismisses with the answer string (None if cancelled)."""

    CSS = """
    AskUserScreen { align: center middle; }
    #ask_box { width: 80; max-height: 80%; border: round $primary; background: $surface; padding: 1 2; }
    #ask_title { color: $primary; text-style: bold; margin-bottom: 1; }
    #ask_question { color: $text; margin-bottom: 1; }
    #ask_list { height: auto; max-height: 10; border: blank; padding: 0 1; }
    #ask_input { margin-top: 1; border: solid $border; }
    #ask_hint { color: $text-muted; text-align: center; margin-top: 1; }
    """

    BINDINGS = [Binding("escape", "cancel", "Skip", priority=True)]

    def __init__(self, question: str, options: list[str], **kwargs) -> None:
        super().__init__(**kwargs)
        self.question = question
        self.options = options

    def compose(self) -> ComposeResult:
        with Container(id="ask_box"):
            yield Label("❓ The agent has a question", id="ask_title")
            yield Static(self.question.replace("[", "\\["), id="ask_question")
            if self.options:
                yield ListView(id="ask_list")
            yield Input(placeholder="Type an answer and press Enter…", id="ask_input")
            yield Label("Enter to answer · Esc to skip", id="ask_hint")

    def on_mount(self) -> None:
        if self.options:
            lv = self.query_one("#ask_list", ListView)
            for opt in self.options:
                item = ListItem(Label(opt))
                item.answer = opt  # type: ignore[attr-defined]
                lv.append(item)
            lv.index = 0
            lv.focus()
        else:
            self.query_one("#ask_input", Input).focus()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        self.dismiss(getattr(event.item, "answer", None))

    def on_input_submitted(self, event: Input.Submitted) -> None:
        value = event.value.strip()
        if value:
            self.dismiss(value)

    def action_cancel(self) -> None:
        self.dismiss(None)


class TrackingOption(ListItem):
    """A choice row in the interaction-tracking consent prompt."""

    def __init__(self, label: str, choice: bool, **kwargs) -> None:
        self.choice = choice
        super().__init__(Label(label), **kwargs)


class TrackingConsentScreen(Screen):
    """One-time, first-launch prompt asking whether to log full session
    interactions (prompt + full response) to a local JSONL file.

    Shown once (config_manager.get("track_interactions") is None means it
    has never been answered); the choice is persisted to config.yml so this
    never asks again. Reachable later via the command palette ("Toggle
    interaction tracking") to change the choice.
    """

    CSS = """
    TrackingConsentScreen {
        align: center middle;
    }
    #tracking_box {
        width: 76;
        border: round $border;
        background: $surface;
        padding: 1 2;
    }
    #tracking_title {
        color: $text;
        text-style: bold;
        margin-bottom: 1;
    }
    #tracking_body {
        color: $text-muted;
        margin-bottom: 1;
    }
    #tracking_list {
        height: auto;
        border: blank;
        padding: 0 1;
    }
    #tracking_hint {
        color: $text-muted;
        text-align: center;
        margin-top: 1;
    }
    """

    BINDINGS = [
        Binding("escape", "decline", "No thanks", priority=True),
    ]

    def compose(self) -> ComposeResult:
        with Container(id="tracking_box"):
            yield Label("💾 Track session interactions?", id="tracking_title")
            yield Static(
                "Save every prompt + full response this session to a local "
                "JSONL file (sessions/<timestamp>.jsonl) for your own review "
                "or later context reuse. Nothing leaves your machine. You can "
                "change this anytime with /tracking on|off or Ctrl+K → "
                "Toggle interaction tracking.",
                id="tracking_body",
            )
            yield ListView(id="tracking_list")
            yield Label("Enter to choose · Esc = No thanks", id="tracking_hint")

    def on_mount(self) -> None:
        lv = self.query_one("#tracking_list", ListView)
        lv.append(TrackingOption("Yes, track this and future sessions", True))
        lv.append(TrackingOption("No thanks", False))
        lv.index = 0
        lv.focus()

    def _choose(self, enabled: bool) -> None:
        try:
            self.app.state.config_manager.set("track_interactions", enabled)
        except Exception:
            pass
        self.notify(
            "Interaction tracking enabled" if enabled
            else "Interaction tracking disabled — turn it on later with /tracking on",
            timeout=8,
        )
        self.app.pop_screen()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        choice = event.item.choice if isinstance(event.item, TrackingOption) else False
        self._choose(choice)

    def action_decline(self) -> None:
        self._choose(False)


class AuthInputScreen(Screen):
    """Prompt for an API key for a provider (opencode-style auth)."""

    CSS = """
    AuthInputScreen {
        align: center middle;
    }
    #auth_box {
        width: 72;
        border: round $border;
        background: $surface;
        padding: 1 2;
    }
    #auth_title {
        color: $text;
        text-style: bold;
        text-align: center;
        margin-bottom: 1;
    }
    #auth_input {
        margin-bottom: 1;
        border: solid $border;
    }
    #auth_hint {
        color: $text-muted;
        text-align: center;
    }
    """

    BINDINGS = [
        Binding("escape", "dismiss_auth", "Close", priority=True),
    ]

    def __init__(
        self,
        provider: str,
        state: AppState,
        on_saved: Optional[Callable[[], None]] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.provider = provider
        self.state = state
        self.on_saved = on_saved

    def compose(self) -> ComposeResult:
        with Container(id="auth_box"):
            yield Label(f"API key for {self.provider}", id="auth_title")
            yield Input(placeholder="Paste your API key…", password=True, id="auth_input")
            yield Label("Enter to save · Esc to cancel", id="auth_hint")

    def on_mount(self) -> None:
        self.query_one("#auth_input", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        key = event.value.strip()
        if not key:
            self.notify("No key entered; nothing saved.", severity="warning")
            self.app.pop_screen()
            return
        auth.set_key(self.provider, key)
        self.notify(f"Saved API key for {self.provider}")
        self.app.pop_screen()
        if self.on_saved:
            try:
                self.on_saved()
            except Exception:
                pass

    def action_dismiss_auth(self) -> None:
        self.app.pop_screen()


class AuthListScreen(Screen):
    """List providers and their key status; pick one to log in/out."""

    CSS = """
    AuthListScreen {
        align: center middle;
    }
    #authlist_box {
        width: 72;
        max-height: 80%;
        border: round $border;
        background: $surface;
        padding: 1 2;
        scrollbar-size: 1 1;
    }
    #authlist_title {
        color: $text;
        text-style: bold;
        text-align: center;
        margin-bottom: 1;
    }
    #authlist_list {
        height: auto;
        max-height: 60%;
        scrollbar-size: 1 1;
        padding: 0 1;
        border: blank;
    }
    #authlist_hint {
        color: $text-muted;
        text-align: center;
        margin-top: 1;
    }
    """

    BINDINGS = [
        Binding("escape", "dismiss_authlist", "Close", priority=True),
    ]

    def __init__(self, state: AppState, **kwargs) -> None:
        super().__init__(**kwargs)
        self.state = state

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="authlist_box"):
            yield Label("Manage API Keys", id="authlist_title")
            yield ListView(id="authlist_list")
            yield Label("Enter to set a key · Esc to close", id="authlist_hint")

    def on_mount(self) -> None:
        self.state.config_manager.reload()
        self.query_one("#authlist_box", VerticalScroll).border_title = " Auth "
        lv = self.query_one("#authlist_list", ListView)
        for pid, name, models, is_default, has_key in self.state.config_manager.list_providers():
            status = "🔑" if has_key else "🔒"
            lv.append(AuthOption(pid, name, has_key, status))
        if lv.children:
            lv.index = 0
        lv.focus()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        option = event.item
        if not isinstance(option, AuthOption):
            return
        self.app.pop_screen()
        self.app.push_screen(AuthInputScreen(option.provider_id, self.state))

    def action_dismiss_authlist(self) -> None:
        self.app.pop_screen()


class AuthOption(ListItem):
    """A provider row in the auth list."""

    def __init__(self, provider_id: str, name: str, has_key: bool, status: str, **kwargs) -> None:
        self.provider_id = provider_id
        label = f"{status} {name}  [dim]({provider_id})[/]"
        super().__init__(Label(label), **kwargs)


class ThemeOption(ListItem):
    """A selectable theme row in the theme menu."""

    def __init__(self, theme_id: str, **kwargs) -> None:
        self.theme_id = theme_id
        super().__init__(Label(theme_id), **kwargs)


class ThemeMenuScreen(Screen):
    """Theme picker menu (replaces blind Ctrl+T cycling - #2 in the issue
    report). Live-previews the highlighted theme; Enter confirms (and
    persists it), Esc reverts to whatever theme was active on open."""

    CSS = """
    ThemeMenuScreen {
        align: center middle;
    }
    #theme_box {
        width: 48;
        max-height: 80%;
        border: round $border;
        background: $surface;
        padding: 1 2;
        scrollbar-size: 1 1;
    }
    #theme_title {
        color: $text;
        text-style: bold;
        text-align: center;
        margin-bottom: 1;
    }
    #theme_list {
        height: auto;
        max-height: 60%;
        scrollbar-size: 1 1;
        padding: 0 1;
        border: blank;
    }
    #theme_hint {
        color: $text-muted;
        text-align: center;
        margin-top: 1;
    }
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel", priority=True),
        Binding("enter", "confirm", "Apply", priority=True),
    ]

    def __init__(self, main_screen: "MainScreen", **kwargs) -> None:
        super().__init__(**kwargs)
        self._main = main_screen
        self._original_theme = main_screen.state.current_theme

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="theme_box"):
            yield Label("Select Theme", id="theme_title")
            yield ListView(id="theme_list")
            yield Label("↑↓ preview · Enter to apply · Esc to cancel", id="theme_hint")

    def on_mount(self) -> None:
        lv = self.query_one("#theme_list", ListView)
        themes = ThemeRegistry.theme_ids()
        for idx, tid in enumerate(themes):
            lv.append(ThemeOption(tid))
            if tid == self._original_theme:
                lv.index = idx
        if lv.index is None and lv.children:
            lv.index = 0
        lv.focus()

    def on_list_view_highlighted(self, event: ListView.Highlighted) -> None:
        if isinstance(event.item, ThemeOption):
            self.app.theme = event.item.theme_id

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        if isinstance(event.item, ThemeOption):
            self._main.set_theme(event.item.theme_id)
            self.notify(f"Theme → {event.item.theme_id}")
        self.app.pop_screen()

    def action_confirm(self) -> None:
        lv = self.query_one("#theme_list", ListView)
        item = lv.highlighted_child
        if isinstance(item, ThemeOption):
            self._main.set_theme(item.theme_id)
            self.notify(f"Theme → {item.theme_id}")
        self.app.pop_screen()

    def action_cancel(self) -> None:
        # Revert the live preview if the user backs out without confirming.
        self.app.theme = self._original_theme
        self.app.pop_screen()


class MainScreen(Screen):
    """The main chat screen with a right-hand context panel."""

    CSS = """
    #main_shell { height: 1fr; background: $background; }
    #session_metrics_footer {
        height: auto;
        color: $text-muted;
        background: $background;
        border-top: blank;
        padding: 0 2;
        text-style: dim;
    }
    """

    BINDINGS = [
        Binding("ctrl+t", "open_theme_menu", "Theme", priority=True),
        Binding("ctrl+b", "toggle_context_panel", "Context", priority=True),
        Binding("ctrl+k", "open_command_palette", "Commands", priority=True),
        Binding("ctrl+o", "open_model_dialog", "Model", priority=True),
        Binding("ctrl+a", "open_auth", "Auth", priority=True),
        Binding("ctrl+q", "quit", "Quit", priority=True),
        Binding("question_sign", "show_shortcuts", "Shortcuts", priority=True),
    ]

    def __init__(self, state: AppState, **kwargs) -> None:
        super().__init__(**kwargs)
        self.state = state

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="main_shell"):
            yield ChatPane(self.state)
            yield ContextPanel(self.state, id="context_panel")
        yield Label("", id="session_metrics_footer")
        yield Footer()

    def on_mount(self) -> None:
        if not self.state.show_activity_rail:
            self.query_one("#context_panel", ContextPanel).styles.display = "none"
        self.refresh_session_footer()
        self.refresh_context_panel()

    def refresh_context_panel(self) -> None:
        try:
            panel = self.query_one("#context_panel", ContextPanel)
            panel.refresh_context()
        except Exception:
            pass

    def action_open_command_palette(self) -> None:
        self.app.push_screen(CommandPalette(self))

    def action_open_model_dialog(self) -> None:
        self.app.push_screen(ModelDialog(self.state))

    def action_open_auth(self) -> None:
        self.app.push_screen(AuthListScreen(self.state))

    def action_open_theme_menu(self) -> None:
        self.app.push_screen(ThemeMenuScreen(self))

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

    def set_theme(self, theme_id: str) -> None:
        """Apply a theme and persist it so it's restored on the next launch."""
        self.state.current_theme = theme_id
        self.app.theme = theme_id
        try:
            self.state.config_manager.set("default_theme", theme_id)
        except Exception:
            pass

    def action_toggle_context_panel(self) -> None:
        panel = self.query_one("#context_panel", ContextPanel)
        self.state.show_activity_rail = not self.state.show_activity_rail
        panel.styles.display = "block" if self.state.show_activity_rail else "none"

    def action_show_shortcuts(self) -> None:
        self.app.push_screen(ShortcutsOverlay(self))


# ─── Chat pane ────────────────────────────────────────────────────────────────

class ChatPane(Vertical):
    """Message history + input bar."""
    BINDINGS = [
        Binding("ctrl+shift+t", "toggle_trace_panel", "Trace", priority=True),
        Binding("ctrl+shift+c", "copy_last_response", "Copy", priority=True),
        Binding("ctrl+shift+k", "copy_last_code_block", "Copy code", priority=True),
        Binding("f7", "toggle_thinking", "Thinking", priority=True),
        Binding("f8", "toggle_trace_panel", "Trace", priority=True),
        Binding("f9", "copy_last_response", "Copy", priority=True),
        Binding("f10", "copy_trace", "Copy trace", priority=True),
        Binding("tab", "toggle_agent_mode", "Agent", priority=True),
        Binding("ctrl+e", "open_external_editor", "Editor", priority=True),
    ]

    DEFAULT_CSS = """
    ChatPane {
        height: 1fr;
        background: $background;
        padding: 0;
    }
    #chat_body {
        height: 1fr;
        padding: 0;
    }
    #chat_log {
        height: 1fr;
        width: 3fr;
        border: blank;
        padding: 1 2;
        scrollbar-size: 1 1;
        background: $background;
    }
    #trace_panel {
        width: 40;
        height: 1fr;
        border: blank;
        background: $panel;
        padding: 1 1;
        margin-left: 1;
    }
    #trace_header {
        color: $text-muted;
        text-style: bold;
        margin-bottom: 1;
        padding: 0 1;
        background: transparent;
        border: blank;
    }
    #trace_log {
        height: 1fr;
        scrollbar-size: 1 1;
        padding-top: 1;
    }
    #trace_log > * {
        border-top: blank;
        padding-top: 0;
        margin-top: 0;
    }
    #trace_summary_chip {
        height: auto;
        width: auto;
        padding: 0 2;
        margin: 0 0 1 0;
        background: transparent;
        border: blank;
        color: $text-muted;
        text-style: dim;
    }
    /*
      Prompt layout modeled on opencode's Prompt component:
        - Surface panel holds the single-line input + meta row.
        - Thick left border in the agent/model color.
        - Status row below shows spinner + token/cost + shortcuts.
    */
    #chat_status {
        height: 1;
        width: 1fr;
        padding: 0 0 0 2;
        background: $background;
        color: $text-muted;
        text-style: dim;
        content-align: left middle;
    }
    #chat_status_spinner {
        width: 3;
        height: 1;
        color: $primary;
        margin-right: 1;
        display: none;
    }
    #chat_status_spinner.busy {
        display: block;
    }
    """

    def __init__(self, state: AppState, **kwargs) -> None:
        super().__init__(**kwargs)
        self.state = state
        self._last_trace_stage: str = ""
        # Parallel to self.state.message_queue: the "📥 Queued" notice widget
        # mounted for each queued prompt, so it can be removed once that
        # prompt is dequeued and starts running (or discarded on cancel).
        self._queued_notices: list[SystemMessage] = []
        # Trace lines are buffered; widgets are only mounted while the panel is
        # visible (mounting one widget per event was a measurable UI cost).
        self._trace_lines: list[str] = []
        self._trace_count = 0
        # Live view of the running turn, read by the status-line timer.
        self._turn: dict = {}

    def compose(self) -> ComposeResult:
        with Horizontal(id="chat_body"):
            yield VerticalScroll(id="chat_log")
            with Vertical(id="trace_panel"):
                yield Label("Interaction Trace", id="trace_header")
                yield VerticalScroll(id="trace_log")
        yield Label("", id="trace_summary_chip")
        yield ChatComposer(
            self.state,
            placeholder="Ask anything…",
            id="chat_input",
        )
        with Horizontal(id="chat_status"):
            yield LoadingIndicator(id="chat_status_spinner")
            yield Label("", id="chat_status_text")

    def on_click(self, event) -> None:
        if getattr(event.control, "id", None) == "trace_summary_chip":
            self._set_trace_panel_visible(True)

    def on_mount(self) -> None:
        log = self.query_one("#chat_log", VerticalScroll)
        log.mount(SystemMessage("⚡ Motion Harness", id="connection_line"))
        log.mount(SystemMessage("Tip: Ctrl+K commands · Ctrl+O model · Tab agent · F7 thinking · F8 trace · F9 copy · /skill save <name>"))
        self._refresh_connection_line()
        self._append_trace("session_start", f"provider={self.state.current_provider_id}")
        self._set_trace_panel_visible(self.state.show_trace_panel)
        self._refresh_meta()
        self.query_one("#chat_input", ChatComposer).focus()

    def _refresh_connection_line(self) -> None:
        """Update the intro "connected to" line whenever the provider/model
        changes, so it never lags behind a reconnect."""
        try:
            line = self.query_one("#connection_line", SystemMessage)
        except Exception:
            return
        pid = self.state.current_provider_id or "?"
        base, _, model = pid.partition("/")
        if model:
            text = f"⚡ Motion Harness — connected to {base} · {model}"
        else:
            text = f"⚡ Motion Harness — connected to {pid}"
        line.update(text)

    def _set_trace_panel_visible(self, visible: bool) -> None:
        self.state.show_trace_panel = visible
        panel = self.query_one("#trace_panel", Vertical)
        panel.styles.display = "block" if visible else "none"
        chip = self.query_one("#trace_summary_chip", Label)
        chip.styles.display = "none" if visible else "block"
        if visible:
            self._rebuild_trace_panel()
        self._refresh_trace_chip()
        self.notify("Trace panel shown" if visible else "Trace panel hidden")
        # Preserve composer focus: expanding the trace panel must never steal focus
        # from the input unless the user explicitly clicked into the panel.
        try:
            self.query_one("#chat_input", ChatComposer).focus()
        except Exception:
            pass

    def _refresh_trace_chip(self) -> None:
        try:
            chip = self.query_one("#trace_summary_chip", Label)
        except Exception:
            return
        count = self._trace_count
        last_stage = self._last_trace_stage
        chip.update(f" trace · {count} events · {last_stage} " if last_stage else f" trace · {count} events ")

    def action_toggle_trace_panel(self) -> None:
        self._set_trace_panel_visible(not self.state.show_trace_panel)

    def action_toggle_thinking(self) -> None:
        """Toggle inline display of the agent's intermediate tool-loop text.

        This is opt-in and separate from the trace panel: when enabled, the
        model's visible responses between tool calls (if any - some
        providers/tasks never produce intermediate text) are rendered as a
        live "thinking" bubble in the chat log, not just summarized in trace.
        """
        self.state.show_thinking = not self.state.show_thinking
        self.notify(f"Agent thinking {'shown' if self.state.show_thinking else 'hidden'}")

    def _toggle_agent_mode(self) -> None:
        """Switch between build (full access) and plan (read-only) agents."""
        self.state.agent_mode = "plan" if self.state.agent_mode == "build" else "build"
        self._refresh_meta()
        self.notify(f"Agent → {self.state.agent_mode}")

    def action_toggle_agent_mode(self) -> None:
        self._toggle_agent_mode()

    def action_open_external_editor(self) -> None:
        """Open the composer draft in $EDITOR and read it back (opencode Ctrl+E)."""
        self._open_external_editor()

    @work(thread=True, exclusive=True)
    def _open_external_editor(self) -> None:
        """Run $EDITOR in a worker thread while the app is suspended.

        Previously this called subprocess.call(...) directly on the UI/event
        loop thread, which froze Textual's rendering and input handling for
        the whole duration the editor was open - keystrokes (including :q)
        landed in whichever process actually had the terminal, and the app
        appeared to hang or exit. `App.suspend()` properly hands the terminal
        to the child editor and restores Textual's terminal mode (and theme)
        on return; running it in a worker thread keeps the event loop free.
        """
        import subprocess
        import tempfile
        editor = os.environ.get("EDITOR") or os.environ.get("VISUAL") or "vi"

        def _get_draft() -> str:
            return self.query_one("#chat_input", ChatComposer).value

        def _apply(content: str) -> None:
            input_box = self.query_one("#chat_input", ChatComposer)
            input_box.value = content
            input_box.focus()
            self.notify("Editor content loaded into composer.")

        def _fail(message: str) -> None:
            self.notify(message, severity="error")

        try:
            draft = self.app.call_from_thread(_get_draft)
        except Exception:
            return
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as f:
            f.write(draft)
            path = f.name
        try:
            with self.app.suspend():
                subprocess.call([editor, path])
            with open(path, encoding="utf-8") as f:
                content = f.read()
            self.app.call_from_thread(_apply, content)
        except Exception as e:
            self.app.call_from_thread(_fail, f"Could not open editor: {e}")
        finally:
            try:
                os.remove(path)
            except Exception:
                pass

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
            input_box = self.query_one("#chat_input", ChatComposer)
            input_box.value = text[:10000]
            input_box.focus()
            self.notify("Clipboard unavailable; response inserted into input for manual copy.", severity="warning")
            return
        except Exception:
            pass
        self.notify("Clipboard copy unavailable in this terminal.", severity="warning")

    def action_copy_last_response(self) -> None:
        self._copy_last_response()

    # ── trace / trajectory / tracking ────────────────────────────────────────
    def _copy_text(self, text: str, what: str) -> None:
        """Copy to the clipboard, falling back to the input box (as the response copy does)."""
        copy_fn = getattr(self.app, "copy_to_clipboard", None)
        if callable(copy_fn):
            try:
                copy_fn(text)
                self.notify(f"Copied {what} to clipboard.")
                return
            except Exception:
                pass
        try:
            input_box = self.query_one("#chat_input", ChatComposer)
            input_box.value = text[:10000]
            input_box.focus()
            self.notify(f"Clipboard unavailable; {what} inserted into input for manual copy.", severity="warning")
        except Exception:
            self.notify("Clipboard copy unavailable in this terminal.", severity="warning")

    def action_copy_trace(self) -> None:
        """Copy the whole interaction-trace log as plain text (the panel itself can't be selected)."""
        if not self._trace_lines:
            self.notify("The trace log is empty.", severity="warning")
            return
        plain = "\n".join(Text.from_markup(line).plain for line in self._trace_lines)
        self._copy_text(plain, f"trace log ({len(self._trace_lines)} lines)")

    def _trajectory_records(self, everything: bool = False) -> list[dict]:
        recs = self.state.trajectory
        return recs if everything else traj.turn_records(recs)

    def _trajectory_title(self, everything: bool) -> str:
        if everything:
            return f"Trajectory (whole session, {self.state.trajectory_turn} turns)"
        return f"Trajectory of turn {self.state.trajectory_turn}"

    def show_trajectory(self, everything: bool = False) -> None:
        log = self.query_one("#chat_log", VerticalScroll)
        recs = self._trajectory_records(everything)
        text = traj.render(recs, self._trajectory_title(everything))
        widget = Static(Text(text), classes="trajectory")
        widget.styles.border_left = ("solid", "orange")
        widget.styles.padding = (0, 2)
        widget.styles.margin = (0, 0, 1, 1)
        log.mount(widget)
        if recs:
            log.mount(SystemMessage("  /trajectory copy · /trajectory save [full] · /trajectory all"))
        log.scroll_end(animate=False)

    def copy_trajectory(self, everything: bool = False) -> None:
        recs = self._trajectory_records(everything)
        if not recs:
            self.notify("No steps recorded yet.", severity="warning")
            return
        self._copy_text(traj.render(recs, self._trajectory_title(everything)), "trajectory")

    def save_trajectory(self, full: bool = False, everything: bool = False) -> Optional[Path]:
        import json

        recs = self._trajectory_records(everything)
        if not recs:
            self.notify("No steps recorded yet.", severity="warning")
            return None
        last = self.state.last_transcript or {}
        doc = traj.to_json(
            self.state.trajectory if everything else recs,
            turn=None if everything else recs[0].get("turn"),
            provider=self.state.current_provider_id,
            system_prompt=last.get("system_prompt") if full else None,
            messages=last.get("messages") if full else None,
        )
        if everything:
            doc["steps"], doc["summary"], doc["insights"] = recs, traj.totals(recs), traj.insights(recs)
        directory = state_dir(WORKSPACE, "trajectories")  # self-ignoring: never lands in the user's commits
        name = f"session-{datetime.now():%Y%m%d-%H%M%S}.json" if everything else f"turn-{recs[0].get('turn', 0)}-{datetime.now():%H%M%S}.json"
        path = directory / name
        path.write_text(json.dumps(doc, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        return path

    def _trajectory_command(self, arg: str, log: VerticalScroll) -> None:
        words = arg.lower().split()
        everything = "all" in words
        if "copy" in words:
            self.copy_trajectory(everything)
        elif "save" in words:
            path = self.save_trajectory(full="full" in words, everything=everything)
            if path:
                log.mount(SystemMessage(f"💾 Saved {path}" + ("" if "full" in words else "  (add 'full' to include every message sent to the model)")))
        elif words and not everything:
            log.mount(SystemMessage("Usage: /trajectory [all] | copy [all] | save [full] [all]"))
        else:
            self.show_trajectory(everything)

    def set_tracking(self, mode: str = "toggle") -> None:
        """on | off | toggle | status. This is also how to undo the first-launch 'No thanks'."""
        cm = self.state.config_manager
        log = self.query_one("#chat_log", VerticalScroll)
        mode = (mode or "status").lower()
        if mode not in ("on", "off", "toggle", "status"):
            log.mount(SystemMessage("Usage: /tracking [on|off]"))
            return
        current = bool(cm.get("track_interactions"))
        enabled = {"on": True, "off": False, "status": current}.get(mode, not current)
        if enabled != current:
            try:
                cm.set("track_interactions", enabled)
            except Exception as exc:
                log.mount(SystemMessage(f"⚠ Could not save the setting: {exc}"))
                return
            self.state._session_store = None  # next turn starts a fresh transcript file
        state = (
            f"ON — every turn is saved under {WORKSPACE}/.motion/sessions/" if enabled else "OFF — nothing is saved"
        )
        log.mount(SystemMessage(f"💾 Interaction tracking is {state}. Change it with /tracking {'off' if enabled else 'on'}."))
        log.scroll_end(animate=False)

    def _copy_last_code_block(self) -> None:
        """Copy the last fenced code block from the last agent response.

        Whole-response copy (F9/Ctrl+Shift+C) is often too coarse when all
        the user wants is the code the agent just wrote - this extracts just
        that (#11 in the issue report).
        """
        text = self.state.last_agent_response or ""
        blocks = re.findall(r"```[^\n]*\n(.*?)```", text, re.DOTALL)
        if not blocks:
            self.notify("No code block found in the last response.", severity="warning")
            return
        code = blocks[-1].strip("\n")
        copy_fn = getattr(self.app, "copy_to_clipboard", None)
        if callable(copy_fn):
            try:
                copy_fn(code)
                self.notify("Copied last code block to clipboard.")
                return
            except Exception:
                pass
        try:
            input_box = self.query_one("#chat_input", ChatComposer)
            input_box.value = code[:10000]
            input_box.focus()
            self.notify("Clipboard unavailable; code block inserted into input for manual copy.", severity="warning")
            return
        except Exception:
            pass
        self.notify("Clipboard copy unavailable in this terminal.", severity="warning")

    def action_copy_last_code_block(self) -> None:
        self._copy_last_code_block()

    @staticmethod
    def _theme_code_style(theme_name: str) -> str:
        """Map a TUI theme to a Pygments code style for Rich Markdown."""
        mapping = {
            "opencode": "github-dark",
            "dracula": "dracula",
            "nord": "nord",
            "one_dark": "one-dark",
            "omni_dark": "dracula",
            "solarized_light": "solarized-light",
            # Pygments ships no native Catppuccin style; "dracula" is the
            # closest purple/pink-accented dark style for the three dark
            # flavors, and "solarized-light" is the closest light one.
            "catppuccin_mocha": "dracula",
            "catppuccin_macchiato": "dracula",
            "catppuccin_frappe": "dracula",
            "catppuccin_latte": "solarized-light",
        }
        return mapping.get(theme_name, "default")

    def _render_user_markdown(self, timestamp: str, text: str):
        safe = text.strip() or "_Empty message._"
        return Group(
            Text(f"you  {timestamp}", style="dim"),
            RichMarkdown(safe, code_theme=self._theme_code_style(self.app.theme)),
        )

    def _render_agent_markdown(self, timestamp: str, answer: str):
        safe_answer = answer.strip() or "_No response content._"
        return RichMarkdown(safe_answer, code_theme=self._theme_code_style(self.app.theme))

    def _agent_color(self) -> str:
        """Return the agent-mode accent color as a CSS variable string.

        opencode uses blue for the build agent and orange for the plan agent.
        """
        return "$secondary" if self.state.agent_mode == "build" else "$warning"

    def _refresh_meta(self) -> None:
        """Update the composer meta row: agent · model · provider.

        Mirrors opencode's prompt footer which shows:
          AgentName · modelName providerName
        and uses the agent color as the left border highlight.
        """
        try:
            composer = self.query_one("#chat_input", ChatComposer)
        except Exception:
            return
        agent = self.state.agent_mode
        agent_label = agent.capitalize()
        provider_id = self.state.current_provider_id or ""
        # current_provider_id holds "provider/model"; split so the meta line
        # shows the actual model and the bare provider. ModelConfig.name is
        # the provider's display name, so it can't substitute for the model.
        if "/" in provider_id:
            base_provider, model = provider_id.split("/", 1)
        else:
            base_provider, model = provider_id, ""
        agent_color = self._agent_color()
        # Resolve CSS variable name to a concrete hex color for Rich markup.
        theme = self.app.get_theme(self.app.theme)
        agent_hex = self._agent_hex(theme, agent_color)
        model_disp = model or (self.state.agent.provider.config.name if self.state.agent else "?")
        provider_disp = base_provider or provider_id
        # opencode-style meta: "Build · deepseek-v4-flash ollama-cloud"
        meta_markup = (
            f"[{agent_hex} bold]{agent_label}[/] [dim]·[/] {model_disp} "
            f"[dim]{provider_disp}[/]"
        )
        composer.set_meta(meta_markup)
        # Tint the left border of the composer with the agent color.
        self._refresh_agent_color_accent()
        self._refresh_status()

    @staticmethod
    def _agent_hex(theme, agent_color: str) -> str:
        """Resolve an agent-mode CSS variable name to a concrete hex color."""
        if agent_color == "$warning":
            return theme.warning or theme.primary or "#f5a742"
        if agent_color == "$secondary":
            return theme.secondary or theme.primary or "#5c9cf5"
        return theme.primary or "#fab283"

    def _refresh_agent_color_accent(self) -> None:
        """Apply the current agent-mode accent color to the prompt border."""
        theme = self.app.get_theme(self.app.theme)
        agent_color = self._agent_color()
        hex_color = self._agent_hex(theme, agent_color)
        composer = self.query_one("#chat_input", ChatComposer)
        composer.styles.border_left = ("solid", hex_color)

    @staticmethod
    def _fmt_tokens(n: int) -> str:
        return f"{n / 1000:.1f}k" if n >= 1000 else str(n)

    def _refresh_status(self) -> None:
        """Update the status row below the composer.

        While a turn runs this is a live readout - phase, elapsed time, step,
        time-to-first-token, the latest command output line - so waiting on the
        model is never silent. Idle, it shows the last turn's timing and totals.
        """
        try:
            status_text = self.query_one("#chat_status_text", Label)
            spinner = self.query_one("#chat_status_spinner", LoadingIndicator)
        except Exception:
            return
        m = self.state.last_turn_metrics or {}
        s = self.state.session_metrics or {}
        last_total = m.get("total_tokens_est", 0)
        session_total = s.get("total_tokens_est", 0)
        turns = s.get("turns", 0)
        cost = s.get("estimated_cost_usd", 0.0)
        if isinstance(cost, (int, float)) and cost > 0:
            cost_part = f"  [dim]·[/]  [$warning]{format_cost(cost)}[/]"
            if s.get("unpriced_turns"):
                cost_part += f" [dim]+ {s['unpriced_turns']} unpriced[/]"
        elif s.get("unpriced_turns"):
            cost_part = "  [dim]·  cost n/a (model has no pricing; set input_mtok/output_mtok)[/]"
        else:
            cost_part = ""
        if self.state.busy:
            spinner.set_class(True, "busy")
            t = self._turn or {}
            elapsed = time.monotonic() - t.get("started", time.monotonic())
            bits = [f"[bold]{t.get('phase', 'working')}[/] {elapsed:.1f}s"]
            if t.get("step"):
                bits.append(f"step {t['step']}")
            if t.get("ttft") is not None:
                bits.append(f"first token {t['ttft']:.1f}s")
            if t.get("tokens"):
                bits.append(f"{self._fmt_tokens(t['tokens'])} tok")
            if self.state.message_queue:
                bits.append(f"{len(self.state.message_queue)} queued")
            running_jobs = len(self.state.tool_session.jobs.running())
            if running_jobs:
                bits.append(f"⚙ {running_jobs} job{'s' if running_jobs != 1 else ''}")
            out = (t.get("out") or "").replace("[", "\\[").replace("]", "\\]")
            if out:
                bits.append(f"[dim]{out[:70]}[/]")
            status_text.update("  [dim]·[/]  ".join(bits) + "   [dim]Esc cancels[/]")
        else:
            spinner.set_class(False, "busy")
            timing = ""
            if m.get("elapsed_s") is not None:
                timing = f"[dim]last turn[/] {m['elapsed_s']:.1f}s"
                if m.get("ttft_s") is not None:
                    timing += f" [dim](first token {m['ttft_s']:.1f}s)[/]"
                last_cost = m.get("estimated_cost_usd")
                if isinstance(last_cost, (int, float)) and last_cost > 0:
                    timing += f" [$warning]{format_cost(last_cost)}[/]"
                timing += "  [dim]·[/]  "
            running_jobs = len(self.state.tool_session.jobs.running())
            if running_jobs:
                timing += f"⚙ {running_jobs} job{'s' if running_jobs != 1 else ''} (/jobs)  [dim]·[/]  "
            status_text.update(
                f"{timing}"
                f"[dim]turns[/] {turns}  [dim]·[/]  "
                f"[dim]last[/] {self._fmt_tokens(last_total)} tok  [dim]·[/]  "
                f"[dim]session[/] {self._fmt_tokens(session_total)} tok{cost_part}"
            )

    async def on_composer_submitted(self, event: ComposerSubmitted) -> None:
        await self._submit_composer(event.text)

    async def _submit_composer(self, text: str = "") -> None:
        input_box = self.query_one("#chat_input", ChatComposer)
        text = (text or input_box.value).strip()
        if not text:
            return
        input_box.value = ""
        input_box.cursor_position = 0
        input_box.refresh()
        input_box.focus()
        self.state.record_prompt(text)

        log = self.query_one("#chat_log", VerticalScroll)
        if text.startswith("/skill"):
            # Plan agent is read-only: no skill writes.
            if self.state.agent_mode == "plan" and ("save" in text or "delete" in text):
                log.mount(SystemMessage("⛔ Plan agent is read-only — skill writes disabled."))
                log.scroll_end(animate=False)
                return
            await self._handle_skill_command(text, log)
            log.scroll_end(animate=False)
            return
        if text.startswith("/auth"):
            await self._handle_auth_command(text, log)
            log.scroll_end(animate=False)
            return
        if text.startswith("/tools") or text.strip() == "/help":
            await self._handle_tools_command()
            return
        if text.split()[0] in ("/compact", "/undo", "/new", "/resume", "/todos", "/mcp", "/diff", "/jobs", "/trajectory", "/tracking", "/effort", "/budget"):
            await self._handle_session_command(text, log)
            log.scroll_end(animate=False)
            return
        if text.startswith("/attach") or text.startswith("/clear"):
            if text.startswith("/attach") and ("/attach" == text.strip()):
                # No path given -> open the interactive file picker.
                picked = await self.app.push_screen_wait(FilePickerScreen(Path(WORKSPACE)))
                log.scroll_end(animate=False)
                if picked is not None:
                    await self._handle_attach_command(f"/attach {picked}", log)
                return
            await self._handle_attach_command(text, log)
            log.scroll_end(animate=False)
            return
        if text.startswith("/synthesize"):
            await self._handle_synthesize_command(text, log)
            log.scroll_end(animate=False)
            return
        if text.startswith("/parallel"):
            await self._handle_parallel_command(text, log)
            log.scroll_end(animate=False)
            return
        if self.state.agent_mode == "plan" and _is_build_trigger(text):
            self.state.agent_mode = "build"
            self._refresh_meta()
            log.mount(SystemMessage("🔧 Build confirmed — switching to Build mode, I'll create/edit files now."))
        ts = datetime.now().strftime("%H:%M:%S")
        user_msg = UserMessage("")
        user_msg.update(self._render_user_markdown(ts, text))
        log.mount(user_msg)
        if self.state.busy:
            # Don't call _run_agent here: it's @work(exclusive=True), so a
            # second call would cancel the in-flight turn instead of running
            # alongside it. Queue instead - the running worker drains this
            # queue itself once its current turn finishes.
            self.state.message_queue.append(text)
            notice = SystemMessage(
                f"📥 Queued (#{len(self.state.message_queue)}) — will run once the "
                "current task finishes."
            )
            self._queued_notices.append(notice)
            log.mount(notice)
            log.scroll_end(animate=False)
            return
        live_response = AgentMessage("")
        log.mount(live_response)
        log.scroll_end(animate=False)
        prompt_for_agent, images, display_prompt = self._consume_attachments(text)
        self._run_agent(prompt_for_agent, live_response, display_prompt, images)

    MAX_ATTACHED_TEXT_CHARS = 60_000

    def _consume_attachments(self, text: str) -> tuple[str, list[dict], str]:
        """Turn pending attachments into (prompt, images, display_prompt) for
        ONE turn, then clear them.

        Attachments used to be re-sent with every later message (including
        hundreds of KB of base64), inflating every request for the rest of the
        session. Now they go with the message they were attached to; the
        conversation history keeps only a short note about them.
        """
        atts, self.state.attachments = self.state.attachments, []
        if not atts:
            return text, [], text
        blocks: list[str] = []
        images: list[dict] = []
        names: list[str] = []
        budget = self.MAX_ATTACHED_TEXT_CHARS
        for att in atts:
            path = att.get("path", "")
            name = Path(path).name if path else "?"
            names.append(name)
            if att.get("type") == "text" and att.get("content"):
                content = att["content"][:budget]
                budget = max(0, budget - len(content))
                note = " (truncated)" if len(att["content"]) > len(content) else ""
                blocks.append(f"[Attached file: {path}{note}]\n{content}")
            elif att.get("type") == "image":
                blocks.append(f"[Attached image: {name}] (absolute path: {path})")
                if att.get("data"):
                    images.append({"name": name, "mime": att.get("mime", "image/png"), "data": att["data"]})
            else:
                blocks.append(f"[Attached file: {name}] ({path}) ({att.get('summary','')})")
        header = "<attachments>\n" + "\n\n".join(blocks) + "\n</attachments>\n\n"
        return header + text, images, f"{text}\n[attached: {', '.join(names)}]"

    def _attach_context(self, text: str) -> str:
        """Prompt text with attachments included (kept for callers/tests)."""
        return self._consume_attachments(text)[0]

    _TRACE_LABELS = {
        "memory_recall_start": "🧠 memory.recall.start",
        "memory_recall_done": "🧠 memory.recall.done",
        "memory_recall_timeout": "🧠 memory.recall.timeout",
        "model_start": "🤖 model.start",
        "model_step": "⏱ model.step",
        "model_done": "🤖 model.done",
        "turn_start": "▶ turn.start",
        "turn_done": "🏁 turn.done",
        "finalize": "✅ finalize",
        "skill_synthesis_start": "🎓 skill.synthesis.start",
        "skill_synthesis_done": "🎓 skill.synthesis.done",
        "skill_synthesis_error": "🎓 skill.synthesis.error",
        "session_start": "⚡ session.start",
        "interaction_start": "▶ interaction.start",
        "interaction_error": "❌ interaction.error",
        "interaction_cancelled": "⏹ interaction.cancelled",
        "tool_start": "🔧 about to run",
        "tool_done": "✅ finished",
        "tool_error": "❌ failed",
        "tool_progress": "📶 tool.progress",
        "provider_error": "⛔ provider.error",
        "usage": "🧮 usage",
        "step_cap_hit": "⚠️ step_cap.hit",
        "loop_warning": "⚠️ loop.warning",
        "permission_request": "🔐 permission",
        "context_compacted": "🗜 context.compacted",
        "native_tools_disabled": "🔁 native_tools.disabled",
        "todo_update": "☑ todo.update",
        "sandbox": "🧱 sandbox",
        "subagent_start": "🧩 subagent.start",
        "subagent_done": "🧩 subagent.done",
        "exploration_nudge": "💡 nudge",
        "budget_hit": "⏱ budget.hit",
        "hook_blocked": "🪝 hook.blocked",
        "hook_output": "🪝 hook.output",
        "failover": "🔀 failover",
        "failover_skipped": "🔀 failover.skipped",
    }
    TRACE_BUFFER_MAX = 400
    TRACE_WIDGET_MAX = 250

    def _append_trace(self, event_type: str, detail: str = "", **extra) -> None:
        if event_type == "stream_chunk":  # one per token: pure noise
            return
        ts = datetime.now().strftime("%H:%M:%S")
        label = self._TRACE_LABELS.get(event_type, event_type)
        safe_detail = (detail or "").replace("[", "\\[").replace("]", "\\]")
        line = f"[dim]{ts}[/] {label}"
        if safe_detail:
            line += f" [dim]· {safe_detail[:220]}[/]"
        # Append extra context (provider, model, prompt_preview, etc.) so trace
        # errors are explicit without breaking the compact line format.
        if extra:
            context_parts = []
            for key in ("provider", "model", "prompt_preview", "path", "tool", "error", "raw_preview"):
                value = extra.get(key)
                if value is not None and value != "":
                    safe_value = str(value).replace("[", "\\[").replace("]", "\\]")
                    context_parts.append(f"{key}={safe_value[:120]}")
            if context_parts:
                line += " [dim]" + " ".join(context_parts) + "[/]"
        self._last_trace_stage = f"{label}"
        self._trace_count += 1
        self._trace_lines.append(line)
        if len(self._trace_lines) > self.TRACE_BUFFER_MAX:
            del self._trace_lines[: -self.TRACE_BUFFER_MAX]
        if self.state.show_trace_panel:
            self._mount_trace_line(line)
        else:
            self._refresh_trace_chip()

    def _mount_trace_line(self, line: str) -> None:
        try:
            trace_log = self.query_one("#trace_log", VerticalScroll)
            trace_log.mount(SystemMessage(line))
            children = list(trace_log.children)
            for old in children[: -self.TRACE_WIDGET_MAX]:
                old.remove()
            trace_log.scroll_end(animate=False)
            self.query_one("#trace_header", Label).update(f"Interaction Trace ({self._trace_count})")
        except Exception:
            pass
        self._refresh_trace_chip()

    def _rebuild_trace_panel(self) -> None:
        try:
            trace_log = self.query_one("#trace_log", VerticalScroll)
            for child in list(trace_log.children):
                child.remove()
            for line in self._trace_lines[-self.TRACE_WIDGET_MAX:]:
                trace_log.mount(SystemMessage(line))
            trace_log.scroll_end(animate=False)
            self.query_one("#trace_header", Label).update(f"Interaction Trace ({self._trace_count})")
        except Exception:
            pass

    @work(exclusive=True, name="agent_chat")
    async def _run_agent(
        self,
        prompt: str,
        live_response: AgentMessage,
        display_prompt: Optional[str] = None,
        images: Optional[list] = None,
    ) -> None:
        """Run one turn, then drain any prompts queued while it was busy.

        Looping here (rather than re-invoking this @work(exclusive=True)
        method) means a queued follow-up runs in the SAME worker instead of
        starting a second worker that would cancel this one.
        """
        log = self.query_one("#chat_log", VerticalScroll)
        self.state.busy = True
        self._refresh_status()
        try:
            while True:
                cancelled = await self._run_agent_turn(prompt, live_response, log, display_prompt, images)
                display_prompt, images = None, None
                if cancelled:
                    if self.state.message_queue:
                        dropped = len(self.state.message_queue)
                        self.state.message_queue.clear()
                        for notice in self._queued_notices:
                            notice.remove()
                        self._queued_notices.clear()
                        log.mount(SystemMessage(
                            f"⏹ Discarded {dropped} queued message(s) due to cancellation."
                        ))
                    break
                if not self.state.message_queue:
                    break
                prompt = self.state.message_queue.pop(0)
                if self._queued_notices:
                    self._queued_notices.pop(0).remove()
                live_response = AgentMessage("")
                log.mount(live_response)
                log.scroll_end(animate=False)
        finally:
            self.state.busy = False
            self._refresh_status()
            log.scroll_end(animate=False)

    async def _maybe_auto_compact(self, log: VerticalScroll) -> None:
        """Summarize older turns when the history nears the model's window,
        instead of letting every request grow until the provider rejects it."""
        if not self.state.needs_compaction():
            return
        log.mount(SystemMessage(
            f"🗜 Context is filling up (~{self._fmt_tokens(self.state.history_tokens())} tok of "
            f"{self._fmt_tokens(self.state.context_window)}) — summarizing earlier turns…"
        ))
        try:
            summary = await self.state.compact_with_model()
        except Exception as e:
            log.mount(SystemMessage(f"⚠ Auto-compact failed ({e}); continuing without it."))
            return
        if summary:
            log.mount(SystemMessage("✓ Conversation compacted into a summary."))
            self._refresh_context_panel_safe()

    def _refresh_context_panel_safe(self) -> None:
        try:
            main_screen = self.screen
            if isinstance(main_screen, MainScreen):
                main_screen.refresh_context_panel()
        except Exception:
            pass

    async def _run_agent_turn(
        self,
        prompt: str,
        live_response: AgentMessage,
        log: VerticalScroll,
        display_prompt: Optional[str] = None,
        images: Optional[list] = None,
    ) -> bool:
        """Run a single agent turn. Returns True if it was cancelled."""
        header_ts = datetime.now().strftime("%H:%M:%S")
        st = self._turn = {
            "started": time.monotonic(), "phase": "thinking", "step": 0, "tool": "",
            "out": "", "ttft": None, "tokens": 0,
        }
        # Live output buffers. `committed` is final-answer text delivered as a
        # plain chunk (legacy providers); `step_buf` is the current model
        # step's streamed text, discarded from the answer if tool calls follow.
        committed: list[str] = []
        step_buf = ""
        think_buf = ""
        think_t0: Optional[float] = None
        think_last = 0.0
        dirty = False
        reasoning_widget: Optional[ReasoningMessage] = None
        thinking_widget: Optional[ThinkingMessage] = None
        thinking_steps: list[str] = []
        steps_widget: Optional[StepsMessage] = None
        step_lines: list[str] = []
        # Real provider-reported usage accumulated across every request made
        # during this turn (each tool-loop step + the final completion).
        turn_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        has_real_usage = False
        summary_info: dict = {}
        self.state.turn_diffs = []
        self.state.trajectory_turn += 1
        if not self.state.seen_first_turn_hint:
            self.state.seen_first_turn_hint = True
            self.notify(
                "Tip: press F8 to watch live tool-by-tool traces (or F7 for the model's "
                "intermediate thinking) while this runs.",
                title="Trace panel",
                timeout=8,
            )
        self._append_trace("interaction_start", prompt[:120])

        def render_live(answer: str):
            # Re-parsing a long Markdown document on every token is quadratic;
            # past a few KB show plain text until the final render.
            if len(answer) > 5000:
                return Text(answer)
            return self._render_agent_markdown(header_ts, answer or "")

        def show_thinking_step(text: str) -> None:
            nonlocal thinking_widget
            text = text.strip()
            if not text:
                return
            thinking_steps.append(text)
            if self.state.show_thinking:
                if thinking_widget is None:
                    thinking_widget = ThinkingMessage("")
                    log.mount(thinking_widget, before=live_response)
                preview = "\n\n".join(f"› {s}" for s in thinking_steps[-6:])
                thinking_widget.update(Text(preview[:3000], style="dim italic"))

        def flush() -> None:
            nonlocal dirty, reasoning_widget
            dirty = False
            raw = "".join(committed) + step_buf
            inline_reasoning, answer = _extract_reasoning_and_answer(raw, streaming=True)
            reasoning = think_buf
            if inline_reasoning:
                reasoning = f"{think_buf}\n\n{inline_reasoning}" if think_buf else inline_reasoning
            if reasoning:
                if reasoning_widget is None:
                    reasoning_widget = ReasoningMessage("")
                    log.mount(reasoning_widget, before=live_response)
                reasoning_widget.update(Text(reasoning[-1500:], style="dim italic"))
            live_response.update(render_live(answer))
            log.scroll_end(animate=False)

        async def renderer() -> None:
            # Coalesce token-rate updates into ~12 renders/second.
            try:
                while True:
                    await asyncio.sleep(0.08)
                    if dirty:
                        flush()
            except asyncio.CancelledError:
                pass

        def on_stream_chunk(chunk: str) -> None:
            nonlocal step_buf, think_buf, dirty, think_t0, think_last, steps_widget
            if not chunk:
                return
            if chunk.startswith("_delta_ "):
                step_buf += chunk[8:]
                st["phase"] = "answering"
                dirty = True
                return
            if chunk.startswith("_endstep_"):
                show_thinking_step(step_buf)
                step_buf = ""
                st["phase"] = "running tools"
                dirty = True
                return
            if chunk.startswith("_think_ "):
                think_buf += chunk[8:]
                now = time.monotonic()
                if think_t0 is None:
                    think_t0 = now
                think_last = now
                st["phase"] = "thinking"
                dirty = True
                return
            if chunk.startswith("_out_ "):
                st["out"] = chunk[6:]
                return
            # Internal progress markers from the tool loop. "_tool_" (concrete
            # operations like "wrote x.py") is always shown inline so long tasks
            # show live progress; "_step_" (legacy providers' free-form step
            # text) stays opt-in (F7).
            if chunk.startswith("_step_ "):
                step_text = chunk[len("_step_ "):].strip()
                self._append_trace("tool_progress", step_text[:220])
                show_thinking_step(step_text)
                if self.state.show_thinking:
                    log.scroll_end(animate=False)
                return
            if chunk.startswith("_tool_ "):
                tool_text = chunk[len("_tool_ "):].strip()
                self._append_trace("tool_progress", tool_text[:220])
                st["out"] = ""
                if tool_text:
                    step_lines.append(tool_text)
                    if steps_widget is None:
                        steps_widget = StepsMessage("")
                        log.mount(steps_widget, before=live_response)
                    preview = "\n".join(f"• {s}" for s in step_lines[-8:])
                    steps_widget.update(Text(preview[:3000], style="dim"))
                    log.scroll_end(animate=False)
                    try:
                        main_screen = self.screen
                        if isinstance(main_screen, MainScreen):
                            main_screen.query_one("#context_panel", ContextPanel).update_current_steps(step_lines)
                    except Exception:
                        pass
                return
            committed.append(chunk)
            dirty = True

        def on_trace_event(*args) -> None:
            nonlocal has_real_usage
            event_type = "trace"
            payload: Dict[str, Any] = {}
            if len(args) == 2:
                event_type = str(args[0])
                payload = args[1] if isinstance(args[1], dict) else {}
            elif len(args) == 1 and isinstance(args[0], dict):
                payload = args[0]
                event_type = str(payload.get("stage") or payload.get("event") or "trace")

            def _tool_detail(prefix: str, tool: str, path: str, error: str = "") -> str:
                detail = f"{tool or 'tool'} {prefix}"
                if path:
                    detail += f" on `{path}`"
                if error:
                    detail += f": {error}"
                return detail

            detail = ""
            if event_type == "failover" and payload.get("provider"):
                self.state.current_provider_id = str(payload["provider"])  # the status line follows the switch
                self.notify(f"Switched to {payload['provider']} (the previous provider failed)", severity="warning", timeout=8)
            if event_type == "step_record":
                record = dict(payload.get("record") or {})
                if record:
                    record["turn"] = self.state.trajectory_turn
                    self.state.trajectory.append(record)
                return
            if event_type == "usage":
                has_real_usage = True
                turn_usage["prompt_tokens"] += int(payload.get("prompt_tokens") or 0)
                turn_usage["completion_tokens"] += int(payload.get("completion_tokens") or 0)
                turn_usage["total_tokens"] += int(payload.get("total_tokens") or 0)
                st["tokens"] = turn_usage["total_tokens"]
                self._append_trace(
                    event_type,
                    f"+{payload.get('prompt_tokens', 0)} prompt / +{payload.get('completion_tokens', 0)} completion",
                )
                return
            if event_type == "model_step":
                st["step"] = payload.get("step", st["step"])
                if st["ttft"] is None and payload.get("ttft_ms") is not None:
                    st["ttft"] = payload["ttft_ms"] / 1000
                ttft = payload.get("ttft_ms")
                detail = f"step {payload.get('step')}: {payload.get('duration_ms', 0) / 1000:.1f}s" + (
                    f" (first token {ttft / 1000:.1f}s)" if ttft is not None else ""
                )
            elif event_type == "turn_done":
                self.state.last_transcript = {
                    "system_prompt": payload.get("system_prompt"),
                    "messages": payload.get("transcript"),
                }
                summary_info.update({k: v for k, v in payload.items() if k not in ("transcript", "system_prompt")})
                detail = payload.get("message", "")
            elif event_type in {"tool_start", "tool_done", "tool_error"}:
                tool = payload.get("tool", "")
                path = payload.get("path", "")
                error = payload.get("error", "")
                if event_type == "tool_start":
                    st["phase"] = f"running {tool}"
                    detail = _tool_detail("about to run", tool, path)
                elif event_type == "tool_done":
                    st["phase"] = "thinking"
                    detail = _tool_detail("finished", tool, path)
                    diff = payload.get("diff") or ""
                    if diff and not payload.get("created"):
                        added, removed = int(payload.get("added") or 0), int(payload.get("removed") or 0)
                        self.state.turn_diffs.append((path, diff, added, removed))
                        if self.state.show_diffs:
                            widget = DiffMessage("")
                            widget.show(path, diff, added, removed, self._theme_code_style(self.app.theme))
                            log.mount(widget, before=live_response)
                            log.scroll_end(animate=False)
                else:
                    st["phase"] = "thinking"
                    detail = _tool_detail("failed", tool, path, error)
            elif event_type == "tool_progress":
                detail = payload.get("message", "")
            else:
                detail = payload.get("message", "")
                if not detail:
                    parts: list[str] = []
                    for key in ("query", "target", "provider", "model", "task_id", "status"):
                        value = payload.get(key)
                        if value is not None and value != "":
                            parts.append(f"{key}={value}")
                            if len(parts) >= 2:
                                break
                    detail = ", ".join(parts)
            self._append_trace(event_type, detail[:220])

        async def on_permission_request(path: str) -> str:
            """Modal approval for a tool call that would touch a path outside
            the workspace root. Returns "once", "session", or "deny"."""
            st["phase"] = "waiting for you"
            try:
                choice = await self.app.push_screen_wait(PermissionScreen(str(path)))
            except Exception:
                choice = "deny"
            st["phase"] = "thinking"
            return choice if choice in ("once", "session") else "deny"

        async def on_approval(kind: str, subject: str, reason: str) -> str:
            """Modal approval for a risky shell command or a fetch to a
            private-network address."""
            title = {
                "command": "⚠ The agent wants to run a risky command",
                "network": "⚠ The agent wants to reach a private network address",
            }.get(kind, "⚠ Approval needed")
            st["phase"] = "waiting for you"
            try:
                choice = await self.app.push_screen_wait(PermissionScreen(subject, title=title, detail=reason))
            except Exception:
                choice = "deny"
            st["phase"] = "thinking"
            return choice if choice in ("once", "session") else "deny"

        async def on_ask_user(question: str, options: list) -> Optional[str]:
            st["phase"] = "waiting for you"
            try:
                return await self.app.push_screen_wait(AskUserScreen(question, options))
            except Exception:
                return None
            finally:
                st["phase"] = "thinking"

        def on_todo(todos: list) -> None:
            self.state.todos = todos
            try:
                main_screen = self.screen
                if isinstance(main_screen, MainScreen):
                    main_screen.query_one("#context_panel", ContextPanel).update_todos(todos)
            except Exception:
                pass

        renderer_task = asyncio.create_task(renderer())
        status_timer = self.set_interval(0.25, self._refresh_status)
        try:
            await self._maybe_auto_compact(log)
            # Reference prior conversation so the model isn't left to guess:
            # the context query pulls related memory AND the last turns keep
            # the model grounded in what was already said. Capped at the most
            # recent 8 turns so a long session cannot drown out the latest
            # user message.
            history: list[dict[str, str]] = []
            turns = (getattr(self.state, "conversation_turns", None) or [])[-8:]
            for p, r in turns:
                history.append({"role": "user", "content": p})
                if r:
                    history.append({"role": "assistant", "content": r})
            context_query = self.state.build_context_prompt()
            response = await self.state.agent.run(
                prompt,
                target="user",
                on_stream_chunk=on_stream_chunk,
                on_trace_event=on_trace_event,
                history=history or None,
                context_query=context_query or None,
                workspace=WORKSPACE,
                agent_mode=self.state.agent_mode,
                on_permission_request=on_permission_request,
                allowed_paths=self.state.allowed_workspace_paths,
                session=self.state.tool_session,
                on_ask_user=on_ask_user,
                on_approval=on_approval,
                on_todo=on_todo,
                images=images or None,
            )
            renderer_task.cancel()
            reasoning, answer = _extract_reasoning_and_answer(response or "")
            self.state.last_agent_response = answer or ""
            live_response.update(self._render_agent_markdown(header_ts, answer or ""))
            elapsed = time.monotonic() - st["started"]
            # Collapse the reasoning block to a one-line "thought for Ns" (F7
            # keeps the full text visible) so the answer stays the focus. This
            # also covers turns that finished before the first render tick.
            all_reasoning = "\n\n".join(x for x in (think_buf, reasoning) if x).strip()
            if all_reasoning or reasoning_widget is not None:
                if reasoning_widget is None:
                    reasoning_widget = ReasoningMessage("")
                    log.mount(reasoning_widget, before=live_response)
                secs = (think_last - think_t0) if think_t0 is not None else elapsed
                if self.state.show_thinking:
                    reasoning_widget.update(Text(all_reasoning[-2500:], style="dim italic"))
                else:
                    reasoning_widget.update(Text(
                        f"▸ thought for {max(secs, 0.1):.1f}s  (F7 shows reasoning)", style="dim italic"
                    ))
            if has_real_usage:
                est_prompt_tokens = turn_usage["prompt_tokens"]
                est_output_tokens = turn_usage["completion_tokens"]
            else:
                est_prompt_tokens = max(1, len(prompt) // 4)
                est_output_tokens = max(1, len(self.state.last_agent_response or "") // 4)
            provider_type = getattr(self.state.agent.provider.config, "provider_type", "")
            est_cost_usd = turn_cost(
                provider_type,
                getattr(self.state.agent.provider.config, "options", {}) or {},
                est_prompt_tokens,
                est_output_tokens,
            )
            self.state.last_turn_metrics = {
                "prompt_tokens_est": est_prompt_tokens,
                "output_tokens_est": est_output_tokens,
                "total_tokens_est": est_prompt_tokens + est_output_tokens,
                "estimated_cost_usd": est_cost_usd,
                "provider_type": provider_type,
                "tokens_are_real": has_real_usage,
                "elapsed_s": elapsed,
                "ttft_s": st["ttft"],
                "tool_calls": summary_info.get("tool_calls", 0),
            }
            session = self.state.session_metrics
            session["turns"] += 1
            session["prompt_tokens_est"] += est_prompt_tokens
            session["output_tokens_est"] += est_output_tokens
            session["total_tokens_est"] += est_prompt_tokens + est_output_tokens
            if isinstance(est_cost_usd, (int, float)):
                session["estimated_cost_usd"] += float(est_cost_usd)
            else:
                session["unpriced_turns"] = session.get("unpriced_turns", 0) + 1
            # Record the turn for the context panel + rolling session context.
            recorded = display_prompt or prompt
            self.state.conversation_turns.append((recorded, self.state.last_agent_response))
            self.state.update_session_context(recorded, self.state.last_agent_response)
            self.state.log_interaction(recorded, self.state.last_agent_response)
            log.scroll_end(animate=False)
            self._refresh_status()
            main_screen = self.screen
            if isinstance(main_screen, MainScreen):
                main_screen.refresh_session_footer()
                main_screen.refresh_context_panel()
            return False
        except asyncio.CancelledError:
            live_response.remove()
            log.mount(SystemMessage("⏹ Cancelled."))
            self._append_trace(
                "interaction_cancelled",
                "user interrupted",
                prompt_preview=prompt[:120],
            )
            return True
        except Exception as e:
            live_response.remove()
            error_detail = f"{type(e).__name__}: {e}"
            provider_id = self.state.current_provider_id or "unknown"
            model_name = self.state.agent.provider.config.name if self.state.agent else "?"
            # Log the full traceback to motion.log (not just the short message
            # shown in-app) so a "it just stopped" report can be diagnosed
            # after the fact even without a live repro.
            logger.exception("Agent turn failed (provider=%s, model=%s)", provider_id, model_name)
            log.mount(SystemMessage(f"❌ {error_detail}"))
            self._append_trace(
                "interaction_error",
                error_detail,
                provider=provider_id,
                model=model_name,
                prompt_preview=prompt[:120],
            )
            if "provider" in error_detail.lower() and "timed out" in error_detail.lower():
                log.mount(SystemMessage(
                    "💡 Provider timed out. Try again, check your connection, "
                    "or switch providers with /auth or the provider picker."
                ))
            return False
        finally:
            renderer_task.cancel()
            status_timer.stop()
            self._turn = {}

    async def _handle_tools_command(self) -> None:
        """Show the available tools and commands (opencode-style /tools | /help)."""
        log = self.query_one("#chat_log", VerticalScroll)
        lines = [
            "Agent tools:",
            "  files:    list_files · glob_files · grep · read_file · write_file · replace_in_file",
            "  run:      run_command · run_script · run_python",
            "  web:      web_fetch · web_search   (results are treated as untrusted)",
            "  other:    read_image · todo_write · ask_user · use_skill · memory_save/get · MCP tools",
            "  jobs:     job_start · job_output · job_list · job_stop  (dev servers, watchers)",
            "",
            "Slash commands:",
            "  /attach [path]              attach a file to your next message (sent once)",
            "  /compact                    summarize the conversation to free context",
            "  /undo                       revert the file changes made in the last turn",
            "  /diff [on|off]              show the last turn's edits / toggle inline diffs",
            "  /trajectory [copy|save|all] steps, tokens and tool results of the last turn (F10 copies the trace log)",
            "  /tracking [on|off]          save session transcripts locally (asked once at first launch)",
            "  /effort [low|medium|high|off]  how hard reasoning models think (lower = faster and cheaper)",
            "  /budget [steps N|tokens N|cost X|seconds N|off]  stop a turn that uses more than this",
            "  /new                        start a fresh conversation",
            "  /resume [id]                list saved sessions / reload one",
            "  /todos                      show the agent's task list",
            "  /skill list|show|save|delete   manage reusable skills",
            "  /mcp                        connected MCP servers and tools",
            "  /jobs [stop <id|all>]       background processes the agent started",
            "  /parallel a ; b ; c         run sub-tasks on background workers",
            "  /synthesize on|off          toggle auto-crystallization into skills",
            "  /auth list|login|logout     manage provider API keys",
            "",
            "Read tools also work in Plan mode; write/run tools need Build (Tab).",
            "Risky shell commands (rm -r, sudo, git push, …) always ask first.",
        ]
        for line in lines:
            log.mount(SystemMessage(line))
        log.scroll_end(animate=False)

    async def _handle_session_command(self, text: str, log: VerticalScroll) -> None:
        parts = text.split(maxsplit=1)
        cmd = parts[0]
        arg = parts[1].strip() if len(parts) > 1 else ""
        busy_only = ("/compact", "/undo", "/new", "/resume")
        if cmd in busy_only and self.state.busy:
            log.mount(SystemMessage(f"⛔ {cmd} can't run while the agent is working — press Esc to cancel first."))
            return

        if cmd == "/compact":
            if not self.state.conversation_turns:
                log.mount(SystemMessage("Nothing to compact yet."))
                return
            log.mount(SystemMessage(f"🗜 Summarizing {len(self.state.conversation_turns)} turn(s)…"))
            log.scroll_end(animate=False)
            self.run_worker(self._do_compact(log), exclusive=False)
            return

        if cmd == "/undo":
            lines = self.state.tool_session.checkpoints.undo_last_turn()
            if not lines:
                log.mount(SystemMessage("Nothing to undo — no file changes recorded."))
                return
            log.mount(SystemMessage("↩ Reverted the last turn's file changes:"))
            for line in lines:
                log.mount(SystemMessage(f"  {line}"))
            return

        if cmd == "/new":
            self.state.new_session()
            for child in list(log.children):
                if getattr(child, "id", None) != "connection_line":
                    child.remove()
            log.mount(SystemMessage("✨ New conversation. (Approvals and settings are kept.)"))
            self._refresh_context_panel_safe()
            self._refresh_status()
            return

        if cmd == "/resume":
            if not self.state.config_manager.get("track_interactions"):
                log.mount(SystemMessage(
                    "Session history is off. Enable it from the command palette (Toggle interaction tracking) "
                    "and future sessions can be resumed."
                ))
                return
            if not arg:
                sessions = SessionStore.list_sessions(WORKSPACE)
                if not sessions:
                    log.mount(SystemMessage("No saved sessions in this workspace yet."))
                    return
                log.mount(SystemMessage("Saved sessions (use /resume <id>):"))
                for sess in sessions:
                    log.mount(SystemMessage(
                        f"  {sess['id']}  {sess['modified']}  {sess['turns']} turn(s)  {sess['first_prompt']}"
                    ))
                return
            turns = SessionStore.load(WORKSPACE, arg)
            if not turns:
                log.mount(SystemMessage(f"No session named '{arg}'. Use /resume to list them."))
                return
            self.state.new_session()
            for child in list(log.children):
                if getattr(child, "id", None) != "connection_line":
                    child.remove()
            for rec in turns:
                ts = str(rec.get("timestamp", ""))[11:19]
                user_msg = UserMessage("")
                user_msg.update(self._render_user_markdown(ts, rec.get("prompt", "")))
                log.mount(user_msg)
                agent_msg = AgentMessage("")
                agent_msg.update(self._render_agent_markdown(ts, rec.get("response", "")))
                log.mount(agent_msg)
                self.state.conversation_turns.append((rec.get("prompt", ""), rec.get("response", "")))
                self.state.update_session_context(rec.get("prompt", ""), rec.get("response", ""))
            self.state._session_store = SessionStore(WORKSPACE, Path(arg).stem)
            log.mount(SystemMessage(f"↻ Resumed {len(turns)} turn(s) from {arg}. New turns append to that session."))
            self._refresh_context_panel_safe()
            return

        if cmd == "/trajectory":
            self._trajectory_command(arg, log)
            return

        if cmd == "/budget":
            from core.budget import Budget

            budget = self.state.agent.budget or Budget()
            words = arg.lower().split()
            fields = {"steps": ("max_steps", int), "tokens": ("max_tokens", int), "cost": ("max_cost_usd", float),
                      "seconds": ("max_seconds", float)}
            if words == ["off"]:
                budget = Budget()
            elif words:
                if len(words) != 2 or words[0] not in fields:
                    log.mount(SystemMessage("Usage: /budget [steps N | tokens N | cost X | seconds N | off]"))
                    return
                name, kind = fields[words[0]]
                try:
                    value = kind(words[1].lstrip("$"))
                    if value <= 0:
                        raise ValueError
                except ValueError:
                    log.mount(SystemMessage(f"Usage: /budget {words[0]} <positive number>"))
                    return
                setattr(budget, name, value)
            self.state.agent.budget = budget
            log.mount(SystemMessage(
                f"⏱ Per-turn budget: {budget.describe()}"
                + (" — a turn that reaches it gets one last tool-free step to answer." if budget.active else
                   " — set one with /budget steps 12 (or tokens / cost / seconds).")
            ))
            return

        if cmd == "/effort":
            options = self.state.agent.provider.config.options
            level = arg.lower()
            if level in ("low", "medium", "high"):
                options["reasoning_effort"] = level
            elif level in ("off", "default", "none"):
                options.pop("reasoning_effort", None)
            elif level:
                log.mount(SystemMessage("Usage: /effort [low|medium|high|off]"))
                return
            current = options.get("reasoning_effort")
            log.mount(SystemMessage(
                f"🧠 Reasoning effort: {current or 'model default'}"
                + ("" if level else " — change it with /effort low|medium|high|off")
                + " (only models that support it act on this; others ignore it)"
            ))
            return

        if cmd == "/tracking":
            self.set_tracking(arg or "status")
            return

        if cmd == "/diff":
            if arg in ("on", "off"):
                self.state.show_diffs = arg == "on"
                log.mount(SystemMessage(f"Inline diffs {'shown' if self.state.show_diffs else 'hidden'} for this session."))
                return
            if not self.state.turn_diffs:
                log.mount(SystemMessage("No file edits in the last turn."))
                return
            style = self._theme_code_style(self.app.theme)
            for path, diff, added, removed in self.state.turn_diffs:
                widget = DiffMessage("")
                widget.show(path, diff, added, removed, style, max_lines=200)
                log.mount(widget)
            return

        if cmd == "/jobs":
            jobs = self.state.tool_session.jobs
            parts_ = arg.split()
            if parts_ and parts_[0] == "stop":
                target = parts_[1] if len(parts_) > 1 else ""
                if target == "all":
                    n = await jobs.stop_all()
                    log.mount(SystemMessage(f"■ Stopped {n} job(s)."))
                elif target:
                    try:
                        info = await jobs.stop(target)
                        log.mount(SystemMessage(f"■ {info['job_id']} {info['status']}"))
                    except Exception as e:
                        log.mount(SystemMessage(f"⚠ {e}"))
                else:
                    log.mount(SystemMessage("Usage: /jobs stop <id|all>"))
                self._refresh_status()
                return
            listing = jobs.listing()
            if not listing:
                log.mount(SystemMessage("No background jobs. The agent starts them with job_start (dev servers, watchers)."))
            for j in listing:
                mark = "●" if j["status"] == "running" else "○"
                log.mount(SystemMessage(f"  {mark} {j['job_id']}  {j['status']:<11} {j['uptime_s']}s  {j['lines']} lines  {j['command'][:70]}"))
            return

        if cmd == "/todos":
            todos = self.state.todos
            if not todos:
                log.mount(SystemMessage("No todo list yet — the agent creates one for multi-step tasks."))
                return
            marks = {"completed": "✓", "in_progress": "▶", "pending": "○"}
            for t in todos:
                log.mount(SystemMessage(f"  {marks.get(t.get('status'), '○')} {t.get('content', '')}"))
            return

        if cmd == "/mcp":
            mgr = self.state.mcp_manager
            if mgr is None or not mgr.servers:
                log.mount(SystemMessage("No MCP servers configured. Add an `mcp: servers:` block to config.yml."))
                return
            index = mgr.tool_index()
            for name in sorted(mgr.servers):
                tools = [t for srv, t, _ in index if srv == name]
                if name in mgr.errors:
                    log.mount(SystemMessage(f"  ✗ {name}: {mgr.errors[name]}"))
                elif tools:
                    log.mount(SystemMessage(f"  ✓ {name}: {len(tools)} tool(s) — {', '.join(tools[:8])}{' …' if len(tools) > 8 else ''}"))
                else:
                    log.mount(SystemMessage(f"  … {name}: connecting"))
            return

    async def _do_compact(self, log: VerticalScroll) -> None:
        try:
            before = self.state.history_tokens()
            summary = await self.state.compact_with_model()
        except Exception as e:
            log.mount(SystemMessage(f"⚠ Compact failed: {e}"))
            return
        if not summary:
            log.mount(SystemMessage("⚠ The model returned an empty summary; nothing changed."))
            return
        log.mount(SystemMessage(
            f"✓ Compacted: ~{self._fmt_tokens(before)} → ~{self._fmt_tokens(self.state.history_tokens())} tok of history."
        ))
        self._refresh_context_panel_safe()
        log.scroll_end(animate=False)

    async def _handle_attach_command(self, text: str, log: VerticalScroll) -> None:
        parts = text.split(maxsplit=1)
        cmd = parts[0]
        if cmd == "/clear":
            count = len(self.state.attachments)
            self.state.attachments.clear()
            log.mount(SystemMessage(f"🗑 Cleared {count} attached file(s)."))
            return
        if len(parts) < 2 or not parts[1].strip():
            log.mount(SystemMessage("Usage: /attach <path>   (or /clear to drop attachments)"))
            return
        path_str = parts[1].strip()
        # Resolve relative paths against the workspace (not the process CWD)
        # and store the absolute path so downstream read_image/read_file calls
        # find the real file regardless of where the agent runs.
        candidate = Path(path_str).expanduser()
        if not candidate.is_absolute():
            candidate = Path(WORKSPACE) / candidate
        path = candidate.resolve()
        if not path.is_file():
            log.mount(SystemMessage(f"⚠ File not found: {path_str}"))
            return
        try:
            info = self._extract_attachment(path)
        except Exception as e:
            log.mount(SystemMessage(f"⚠ Could not read attachment: {e}"))
            return
        info["path"] = str(path)  # absolute path for agent tool use
        self.state.attachments.append(info)
        label = info.get("summary", path.name)
        log.mount(SystemMessage(f"📎 Attached [{len(self.state.attachments)}]: {label}"))
        self.notify(f"Attached {path.name}")

    def _extract_attachment(self, path: Path) -> dict:
        """Return attachment content for one file: base64 data-url for images,
        extracted text for doc/xlsx/pdf, plain text otherwise."""
        mime, _ = Path(path.name).suffix.lower().lstrip("."), None
        suffix = path.suffix.lower().lstrip(".")
        text_content = None
        if suffix in {"txt", "md", "csv", "json", "py", "yml", "yaml", "toml"}:
            text_content = path.read_text(encoding="utf-8", errors="replace")[:200_000]
        elif suffix in {"docx"}:
            text_content = self._extract_docx(path)
        elif suffix in {"xlsx"}:
            text_content = self._extract_xlsx(path)
        elif suffix in {"pdf"}:
            text_content = self._extract_pdf(path)
        elif suffix in {"png", "jpg", "jpeg", "gif", "webp"}:
            raw = path.read_bytes()
            if not raw:
                raise ValueError("image file is empty")
            if len(raw) > 5 * 1024 * 1024:
                raise ValueError(f"image is {len(raw) // (1024 * 1024)} MB; most providers accept at most 5 MB")
            b64 = base64.b64encode(raw).decode("ascii")
            mime_t = {
                "png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
                "gif": "image/gif", "webp": "image/webp",
            }.get(suffix, "image/png")
            return {
                "path": str(path), "type": "image", "mime": mime_t,
                "data": b64,  # complete: a truncated base64 image is corrupt
                "summary": f"{path.name} (image {len(raw) // 1024}k)",
            }
        else:
            # Fallback: read bytes, describe size since text decoding is unsafe.
            raw = path.read_bytes()
            return {
                "path": str(path), "type": "binary",
                "summary": f"{path.name} (binary {len(raw) // 1024}k, text extraction unsupported)",
            }
        return {
            "path": str(path), "type": "text", "content": text_content or "",
            "summary": f"{path.name} ({len(text_content or '') // 1024}k extracted text)",
        }

    def _extract_docx(self, path: Path) -> str:
        try:
            from docx import Document
        except ImportError:
            return f"[docx text extraction requires python-docx; raw file at {path}]"
        doc = Document(str(path))
        blocks = [p.text for p in doc.paragraphs if p.text.strip()]
        for table in doc.tables:
            for row in table.rows:
                blocks.append(" | ".join(c.text for c in row.cells))
        return "\n".join(blocks)[:200_000]

    def _extract_xlsx(self, path: Path) -> str:
        try:
            import openpyxl
        except ImportError:
            return f"[xlsx text extraction requires openpyxl; raw file at {path}]"
        wb = openpyxl.load_workbook(str(path), read_only=True, data_only=True)
        out = []
        for sheet in wb.sheetnames:
            ws = wb[sheet]
            rows = []
            for row in ws.iter_rows(values_only=True):
                cells = [("" if c is None else str(c)) for c in row]
                if any(cells):
                    rows.append(" | ".join(cells))
            out.append(f"# Sheet: {sheet}\n" + "\n".join(rows))
        return "\n\n".join(out)[:200_000]

    def _extract_pdf(self, path: Path) -> str:
        try:
            import pypdf
        except ImportError:
            return f"[pdf text extraction requires pypdf; raw file at {path}]"
        # pypdf >= 3 dropped PdfFileReader/getNumPages; use the current API.
        reader = pypdf.PdfReader(str(path))
        text = []
        for page in reader.pages:
            text.append(page.extract_text() or "")
        return "\n".join(t for t in text if t)[:200_000]

    async def _handle_synthesize_command(self, text: str, log: VerticalScroll) -> None:
        parts = text.split(maxsplit=1)
        arg = parts[1].strip().lower() if len(parts) > 1 else ""
        if arg == "on":
            self.state.auto_synthesis_enabled = True
            if self.state.agent:
                self.state.agent.auto_skill_synthesis = True
            log.mount(SystemMessage("🎓 Auto skill synthesis ENABLED — successful tasks will crystallize into skills."))
        elif arg == "off":
            self.state.auto_synthesis_enabled = False
            if self.state.agent:
                self.state.agent.auto_skill_synthesis = False
            log.mount(SystemMessage("🎓 Auto skill synthesis DISABLED."))
        else:
            state = "on" if self.state.auto_synthesis_enabled else "off"
            log.mount(SystemMessage(f"🎓 Auto skill synthesis is currently {state}. Usage: /synthesize on|off"))

    async def _handle_parallel_command(self, text: str, log: VerticalScroll) -> None:
        parts = text.split(maxsplit=1)
        if len(parts) < 2 or not parts[1].strip():
            log.mount(SystemMessage("Usage: /parallel <subtask 1> ; <subtask 2> ; ..."))
            return
        subtasks = [p.strip() for p in parts[1].split(";") if p.strip()]
        if len(subtasks) < 2:
            log.mount(SystemMessage("Provide at least two sub-tasks separated by ';' to run in parallel."))
            return
        if not self.state.agent:
            log.mount(SystemMessage("No active agent to parallelize on."))
            return
        tm = self.state.task_manager
        if tm is None:
            log.mount(SystemMessage("Task manager unavailable."))
            return
        log.mount(SystemMessage(f"⚡ Spawning {len(subtasks)} parallel sub-tasks…"))
        log.scroll_end(animate=False)

        def make_reporter():
            announced = False

            async def report(status) -> None:
                nonlocal announced
                if announced or status.status not in ("COMPLETED", "FAILED"):
                    return
                announced = True
                try:
                    if status.status == "COMPLETED":
                        snippet = (status.result or "").strip().replace("\n", " ")[:220]
                        msg = f"✅ task {status.task_id} finished in {status.duration}: {snippet}"
                    else:
                        msg = f"❌ task {status.task_id} failed: {status.error}"
                    if status.artifact_path:
                        msg += f"\n   full transcript: {status.artifact_path}"
                    log.mount(SystemMessage(msg))
                    log.scroll_end(animate=False)
                except Exception:
                    pass

            return report

        for prompt in subtasks:
            req = TaskRequest(prompt=prompt, model_id=self.state.current_provider_id or None)
            task_id = await tm.spawn_task(req, progress_callback=make_reporter())
            log.mount(SystemMessage(f"  › {task_id}: {prompt[:80]}"))
            log.scroll_end(animate=False)
        log.mount(SystemMessage(
            "Background tasks run non-interactively: risky commands are refused. Results appear here as they finish."
        ))
        log.scroll_end(animate=False)

    async def _handle_skill_command(self, text: str, log: VerticalScroll) -> None:
        parts = text.split(maxsplit=2)
        if len(parts) < 2:
            log.mount(SystemMessage("Usage: /skill list | show <name> | save <name> | delete <name>"))
            return
        action = parts[1].strip().lower()
        if action == "list":
            index = SkillLibrary.for_workspace(WORKSPACE).index()
            if not index:
                log.mount(SystemMessage("No skills saved yet. Use /skill save <name> after a good reply."))
            for name, desc in index:
                log.mount(SystemMessage(f"  {name} — {desc}"))
            return
        if action == "show":
            name = parts[2].strip() if len(parts) > 2 else ""
            content = SkillLibrary.for_workspace(WORKSPACE).get(name) if name else None
            if content is None:
                log.mount(SystemMessage(f"Skill not found: {name or '(no name given)'}"))
            else:
                for line in content.splitlines()[:30]:
                    log.mount(SystemMessage(f"  {line}"))
            return
        if action not in {"save", "delete"}:
            log.mount(SystemMessage("Unknown /skill action. Use list, show, save or delete."))
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

    async def _handle_auth_command(self, text: str, log: VerticalScroll) -> None:
        parts = text.split(maxsplit=2)
        action = parts[1].strip().lower() if len(parts) > 1 else ""
        if action == "list":
            keys = auth.list_keys()
            if not keys:
                log.mount(SystemMessage("No API keys stored. Use /auth login <provider>."))
                return
            lines = ["Stored API keys:"]
            for provider, key in sorted(keys.items()):
                masked = f"{key[:4]}…{key[-4:]}" if len(key) > 8 else "…"
                lines.append(f"  {provider}: {masked}")
            log.mount(SystemMessage("\n".join(lines)))
            return
        if action == "logout":
            provider = parts[2].strip() if len(parts) > 2 else ""
            if not provider:
                log.mount(SystemMessage("Usage: /auth logout <provider>"))
                return
            if auth.remove_key(provider):
                log.mount(SystemMessage(f"🗑 Removed API key for {provider}."))
            else:
                log.mount(SystemMessage(f"No stored key for {provider}."))
            return
        if action == "login":
            provider = parts[2].strip() if len(parts) > 2 else ""
            if not provider:
                log.mount(SystemMessage("Usage: /auth login <provider>"))
                return
            try:
                self.state.config_manager.get_provider_config(provider)
            except ValueError:
                log.mount(SystemMessage(f"Unknown provider: {provider}"))
                return
            self.app.push_screen(AuthInputScreen(provider, self.state))
            return
        log.mount(SystemMessage("Usage: /auth list | /auth login <provider> | /auth logout <provider>"))



# ─── The App ──────────────────────────────────────────────────────────────────

class MotionTUI(App):
    """The top-level Motion Harness TUI application.

    Themes are registered as Textual-native themes so that setting
    ``self.theme = "dracula"`` cascades through every CSS ``$variable``
    in every widget — borders, backgrounds, accents, everything.
    """

    CSS = """
    Screen { background: $background; color: $foreground; }
    Header {
        background: $background;
        border-bottom: blank;
        color: $text-muted;
        text-style: bold;
        padding: 0 1;
    }
    Header.-header-tall { height: 3; }
    Footer {
        background: $background;
        border-top: blank;
        color: $text-muted;
        padding: 0 1;
    }
    """

    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit"),
        Binding("ctrl+c", "request_cancel", "Cancel"),
        Binding("escape", "request_cancel", "Cancel"),
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

        # Restore the last-saved theme (config.yml's default_theme), falling
        # back to AppState's built-in default when unset/unknown.
        saved_theme = self.state.config_manager.get("default_theme")
        if saved_theme and saved_theme in ThemeRegistry.theme_ids():
            self.state.current_theme = saved_theme
        self.theme = self.state.current_theme
        self.set_class(self.state.ui_mode == "experimental", "experimental-ui")

        if self._model_config:
            self.state.agent = self.state.make_agent(self._model_config)
            self.state.task_manager = TaskManager(self._model_config, self._workspace, self.state.mcp_manager, self.state.config_manager.data)
            if self._provider_id:
                self.state.current_provider_id = self._provider_id
            else:
                options = AppState.build_provider_options()
                self.state.current_provider_id = options[0][1] if options else ""
            self.push_screen(MainScreen(self.state))
        else:
            self.push_screen(ProviderSelectScreen(self.state))

        if self.state.mcp_manager is not None:
            self.run_worker(self._init_mcp(), exclusive=False)

        # One-time tracking consent on first launch: ask before first use so
        # the user gets a clear choice, but never nag again afterwards.
        if self.state.config_manager.get("track_interactions") is None:
            self.call_after_refresh(self._ask_tracking_consent)

    async def _init_mcp(self) -> None:
        """Connect MCP servers in the background so startup isn't delayed."""
        mgr = self.state.mcp_manager
        try:
            await mgr.initialize_all()
        except Exception as e:
            self.notify(f"MCP setup failed: {e}", severity="warning")
            return
        tools = len(mgr.tool_index())
        if tools:
            self.notify(f"MCP: {tools} tool(s) from {len(mgr.servers) - len(mgr.errors)} server(s)")
        for name, err in mgr.errors.items():
            self.notify(f"MCP server '{name}' failed: {err[:120]}", severity="warning", timeout=10)

    def _ask_tracking_consent(self) -> None:
        """Show the tracking consent prompt over the current screen."""
        try:
            self.push_screen(TrackingConsentScreen())
        except Exception:
            pass

    def action_request_cancel(self) -> None:
        """Cancel the running agent chat — not a quit.

        (This used to call ``self.workers.get(...)``, which does not exist on
        Textual's WorkerManager; the resulting AttributeError was swallowed, so
        Esc / Ctrl+C never cancelled anything.)
        """
        cancelled = False
        for worker in list(self.workers):
            if worker.name == "agent_chat" and not worker.is_finished:
                worker.cancel()
                cancelled = True
        if cancelled:
            self.notify("Request cancelled")

    async def on_unmount(self) -> None:
        """Graceful shutdown: close provider, MCP and DB connections."""
        try:  # never leave the agent's background processes running after we exit
            await asyncio.wait_for(self.state.tool_session.jobs.stop_all(), timeout=5.0)
        except Exception:
            pass
        if self.state.mcp_manager is not None:
            try:
                await asyncio.wait_for(self.state.mcp_manager.close_all(), timeout=3.0)
            except Exception:
                pass
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