from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Input, Static, ScrollableContainer, ListItem, ListView, Label
from textual.containers import Container, Vertical, Horizontal
from textual.binding import Binding
from ui.themes import ThemeRegistry
from core.providers import ModelConfig, ProviderFactory
from core.orchestrator import TaskManager, TaskRequest
from main import MotionAgent
import asyncio
import os

WORKSPACE = os.getenv("MOTION_WORKSPACE", os.getcwd())

class MotionTUI(App):
    """
    Professional TUI for Motion Harness.
    Chat interface with theme support and parallel task orchestration.
    """
    CSS = """
    Screen {
        background: $background;
        color: $foreground;
    }
    #chat_container {
        height: 1fr;
        border: solid $border;
        padding: 1;
        overflow-y: scroll;
    }
    #task_panel {
        height: auto;
        max-height: 8;
        border: solid $border;
        padding: 0 1;
        overflow-y: scroll;
    }
    #input_area {
        height: 3;
        border-top: solid $border;
        padding: 0 1;
    }
    .message {
        margin: 1 0;
        padding: 0 1;
    }
    .user_msg {
        color: $accent;
        text-style: bold;
    }
    .agent_msg {
        color: $foreground;
    }
    .task_status {
        color: $secondary;
    }
    """

    BINDINGS = [
        Binding("ctrl+t", "toggle_theme", "Toggle Theme"),
        Binding("ctrl+s", "show_status", "Task Status"),
        Binding("ctrl+c", "quit", "Quit"),
    ]

    def __init__(self, model_config: ModelConfig, workspace: str = WORKSPACE):
        super().__init__()
        self.model_config = model_config
        self.agent = MotionAgent(model_config)
        self.task_manager = TaskManager(model_config, workspace)
        self.current_theme_name = "one_dark"

    def on_mount(self) -> None:
        self.apply_theme(self.current_theme_name)

    def apply_theme(self, theme_name: str):
        theme = ThemeRegistry.get_theme(theme_name)
        self.theme_vars = {
            "background": theme.background,
            "foreground": theme.foreground,
            "accent": theme.accent,
            "secondary": theme.secondary,
            "border": theme.border,
            "highlight": theme.highlight,
        }
        self.screen.styles.background = theme.background
        self.screen.styles.color = theme.foreground
        self.title = f"Motion Harness - {theme.name}"

    def compose(self) -> ComposeResult:
        yield Header()
        with Container(id="main_container"):
            yield ScrollableContainer(id="task_panel")
            yield ScrollableContainer(id="chat_container")
            yield Horizontal(
                Input(placeholder="Enter prompt... (Ctrl+Enter to send)", id="user_input"),
                id="input_area"
            )
        yield Footer()

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        user_text = event.value
        if not user_text:
            return

        chat = self.query_one("#chat_container")
        chat.mount(Static(f"You: {user_text}", classes="message user_msg"))
        event.input.value = ""

        # Spawn task through the orchestrator for parallel tracking
        task_id = await self.task_manager.spawn_task(
            TaskRequest(prompt=user_text)
        )

        task_panel = self.query_one("#task_panel")
        task_panel.mount(Static(f"Task {task_id}: PENDING", classes="task_status", id=f"task-{task_id}"))

        # Poll for task completion
        asyncio.create_task(self._poll_task(task_id, chat, task_panel))

    async def _poll_task(self, task_id: str, chat: ScrollableContainer, task_panel: ScrollableContainer) -> None:
        """Poll the task manager until the task completes."""
        while True:
            status = self.task_manager.tasks.get(task_id)
            if not status:
                break
            if status.status in ("COMPLETED", "FAILED"):
                # Update the task panel
                try:
                    task_widget = self.query_one(f"#task-{task_id}")
                    task_widget.update(f"Task {task_id}: {status.status}")
                except Exception:
                    pass

                if status.status == "COMPLETED" and status.result:
                    chat.mount(Static(f"Agent: {status.result}", classes="message agent_msg"))
                elif status.status == "FAILED" and status.error:
                    chat.mount(Static(f"Error: {status.error}", classes="message agent_msg"))
                break
            await asyncio.sleep(0.3)

        chat.scroll_end()

    def action_toggle_theme(self) -> None:
        themes = list(ThemeRegistry.THEMES.keys())
        idx = themes.index(self.current_theme_name)
        self.current_theme_name = themes[(idx + 1) % len(themes)]
        self.apply_theme(self.current_theme_name)
        self.notify(f"Theme changed to {self.current_theme_name}")

    def action_show_status(self) -> None:
        status = self.task_manager.get_status()
        self.notify(f"Workers: {status['active_workers']}/{status['max_workers']} | Tasks: {len(status['tasks'])}")

if __name__ == "__main__":
    config = ModelConfig(name="Motion-TUI", endpoint="http://localhost", provider_type="local")
    app = MotionTUI(config)
    app.run()
