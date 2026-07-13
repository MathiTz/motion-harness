from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Input, Static, ScrollableContainer, ListItem, ListView
from textual.containers import Container, Vertical, Horizontal
from textual.binding import Binding
from ui.themes import ThemeRegistry
from core.providers import ModelConfig
from core.orchestrator import TaskManager, TaskRequest
from main import MotionAgent
import asyncio
import os

class MotionTUI(App):
    """
    Professional TUI for Motion Harness.
    Implements a chat interface with theme support and real-time streaming simulation.
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
    """

    BINDINGS = [
        Binding("ctrl+t", "toggle_theme", "Toggle Theme"),
        Binding("ctrl+c", "quit", "Quit"),
    ]

    def __init__(self, model_config: ModelConfig):
        super().__init__()
        self.model_config = model_config
        self.agent = MotionAgent(model_config)
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
        # Set basic styles directly to avoid the AttributeError
        self.screen.styles.background = theme.background
        self.screen.styles.color = theme.foreground
        self.title = f"Motion Harness - {theme.name}"

    def compose(self) -> ComposeResult:
        yield Header()
        with Container(id="main_container"):
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

        # Display user message
        chat = self.query_one("#chat_container")
        chat.mount(Static(f"You: {user_text}", classes="message user_msg"))
        
        # Clear input
        event.input.value = ""

        # Agent Processing
        chat.mount(Static("Agent is thinking...", id="thinking_msg", classes="message agent_msg"))
        
        try:
            # Call the agent loop
            response = await self.agent.run(user_text, target="user")
            
            # Remove thinking indicator and show result
            self.query_one("#thinking_msg").remove()
            chat.mount(Static(f"Agent: {response}", classes="message agent_msg"))
        except Exception as e:
            self.query_one("#thinking_msg").remove()
            chat.mount(Static(f"Error: {str(e)}", classes="message agent_msg"))
        
        chat.scroll_end()

    def action_toggle_theme(self) -> None:
        themes = list(ThemeRegistry.THEMES.keys())
        idx = themes.index(self.current_theme_name)
        self.current_theme_name = themes[(idx + 1) % len(themes)]
        self.apply_theme(self.current_theme_name)
        self.notify(f"Theme changed to {self.current_theme_name}")

if __name__ == "__main__":
    # Configuration for TUI launch
    config = ModelConfig(name="Motion-TUI", endpoint="http://localhost", provider_type="local")
    app = MotionTUI(config)
    app.run()
