"""Tests for the ChatComposer widget behavior.

Covers:
  - Enter posts a ComposerSubmitted message (submit)
  - Ctrl+S also submits
  - Up/down navigates prompt history
"""
import pytest
from rich.style import Style

from textual.app import App, ComposeResult
from textual.message import Message

from ui.tui import AppState, ChatComposer, ComposerSubmitted


class _ComposerHarness(App):
    """Minimal app that hosts a ChatComposer and records submitted messages."""

    def __init__(self):
        super().__init__()
        self.submitted = []
        self.state = AppState()

    def compose(self) -> ComposeResult:
        yield ChatComposer(self.state, id="composer")

    def on_composer_submitted(self, event: ComposerSubmitted) -> None:
        self.submitted.append(event.text)
        self.state.record_prompt(event.text)


@pytest.mark.asyncio
async def test_enter_submits_message():
    app = _ComposerHarness()
    async with app.run_test() as pilot:
        composer = app.query_one("#composer", ChatComposer)
        composer.focus()
        composer.value = "hello world"
        composer.cursor_position = len(composer.value)
        await pilot.press("enter")
        # The message should have been posted with the composer text.
        assert app.submitted == ["hello world"]


@pytest.mark.asyncio
async def test_ctrl_s_submits_message():
    app = _ComposerHarness()
    async with app.run_test() as pilot:
        composer = app.query_one("#composer", ChatComposer)
        composer.focus()
        composer.value = "send via ctrl+s"
        composer.cursor_position = len(composer.value)
        await pilot.press("ctrl+s")
        assert app.submitted == ["send via ctrl+s"]


@pytest.mark.asyncio
async def test_up_down_navigates_history():
    app = _ComposerHarness()
    async with app.run_test() as pilot:
        composer = app.query_one("#composer", ChatComposer)
        composer.focus()
        composer.value = "draft"
        composer.cursor_position = len(composer.value)
        await pilot.press("enter")
        composer.value = ""
        composer.cursor_position = 0
        await pilot.press("up")
        assert composer.value == "draft"
        await pilot.press("down")
        assert composer.value == ""
        await pilot.press("down")
        assert composer.value == ""


def test_prompt_history_records_and_navigates():
    state = AppState()
    state.record_prompt("first")
    state.record_prompt("second")
    # Previous goes back through history.
    assert state.history_previous("") == "second"
    assert state.history_previous("") == "first"
    # Next returns to the current draft.
    assert state.history_next("") == "second"
    assert state.history_next("") == ""


def test_prompt_history_dedupes_consecutive():
    state = AppState()
    state.record_prompt("same")
    state.record_prompt("same")
    assert state.prompt_history == ["same"]
