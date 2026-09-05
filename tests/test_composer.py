"""Tests for the ChatComposer widget behavior.

Covers:
  - Enter posts a ComposerSubmitted message (submit)
  - Ctrl+S also submits
  - Up/down navigates prompt history
"""
import pytest
from rich.style import Style
from textual import events

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

@pytest.mark.asyncio
async def test_paste_inserts_multiline_text_at_cursor():
    app = _ComposerHarness()
    async with app.run_test() as pilot:
        composer = app.query_one("#composer", ChatComposer)
        composer.focus()
        composer.value = "before  after"
        composer.cursor_position = len("before ")
        composer.post_message(events.Paste("line 1\r\nline 2"))
        await pilot.pause()

        assert composer.value == "before line 1\nline 2 after"
        assert composer.cursor_position == len("before line 1\nline 2")


@pytest.mark.asyncio
async def test_large_paste_collapses_to_lines_placeholder():
    # Regression: pasting something large (e.g. a stack trace or file
    # content) used to dump the whole thing into the composer inline. Pastes
    # over the threshold should collapse to a "[LINES N]" placeholder.
    app = _ComposerHarness()
    async with app.run_test() as pilot:
        composer = app.query_one("#composer", ChatComposer)
        composer.focus()
        big_text = "\n".join(f"line {i}" for i in range(30))  # well over 100 chars
        assert len(big_text) > composer.PASTE_COLLAPSE_THRESHOLD
        composer.post_message(events.Paste(big_text))
        await pilot.pause()

        assert composer.value == "[LINES 30]"
        assert composer._pasted_blocks["[LINES 30]"] == big_text


@pytest.mark.asyncio
async def test_short_paste_is_not_collapsed():
    app = _ComposerHarness()
    async with app.run_test() as pilot:
        composer = app.query_one("#composer", ChatComposer)
        composer.focus()
        short_text = "a" * composer.PASTE_COLLAPSE_THRESHOLD  # exactly at threshold
        composer.post_message(events.Paste(short_text))
        await pilot.pause()

        assert composer.value == short_text
        assert composer._pasted_blocks == {}


@pytest.mark.asyncio
async def test_submitting_expands_paste_placeholder_to_original_text():
    app = _ComposerHarness()
    async with app.run_test() as pilot:
        composer = app.query_one("#composer", ChatComposer)
        composer.focus()
        big_text = "x" * 250
        composer.post_message(events.Paste(big_text))
        await pilot.pause()
        assert composer.value == "[LINES 1]"

        await pilot.press("enter")

        # The model still receives the real pasted content, not the placeholder.
        assert app.submitted == [big_text]
        # Placeholder bookkeeping is cleared after submit.
        assert composer._pasted_blocks == {}


@pytest.mark.asyncio
async def test_duplicate_line_count_pastes_get_distinct_placeholders():
    app = _ComposerHarness()
    async with app.run_test() as pilot:
        composer = app.query_one("#composer", ChatComposer)
        composer.focus()
        first = "a" * 150
        second = "b" * 150
        composer.post_message(events.Paste(first))
        await pilot.pause()
        composer.post_message(events.Paste(second))
        await pilot.pause()

        assert composer.value == "[LINES 1][LINES 1#2]"

        await pilot.press("enter")
        assert app.submitted == [first + second]


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
