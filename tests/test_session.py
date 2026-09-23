"""SessionStore: JSONL transcript persistence and turn reconciliation for /resume.

Issue #14's crash-recovery gap: a turn that never reached the old single `append(prompt, response)`
call - because the harness process died mid-tool-call, before any code ran to write it - vanished
from the transcript with no trace at all, and `/resume` had no way to tell the user something had
been attempted. `SessionStore.start_turn` now writes a marker before any tool call runs; `_read`
turns an unmatched one into a visible `interrupted: True` placeholder instead of the turn silently
disappearing. See tests/test_tui_flow.py for the same behavior exercised through the live TUI.
"""
from pathlib import Path

from core.session import INTERRUPTED_MARKER, SessionStore


def test_matched_turn_start_and_end_reconcile_into_one_clean_turn(tmp_path: Path):
    store = SessionStore(tmp_path, "s1")
    turn_id = store.start_turn("hi")
    store.append({"type": "turn_end", "turn_id": turn_id, "prompt": "hi", "response": "hello", "provider": "p"})
    turns = SessionStore.load(tmp_path, "s1")
    assert len(turns) == 1
    assert turns[0]["prompt"] == "hi" and turns[0]["response"] == "hello" and turns[0]["provider"] == "p"
    assert "type" not in turns[0] and "turn_id" not in turns[0] and "interrupted" not in turns[0]


def test_unmatched_turn_start_becomes_an_interrupted_placeholder(tmp_path: Path):
    store = SessionStore(tmp_path, "s1")
    store.start_turn("do something risky")
    # No turn_end is ever written - simulates the process dying right here.
    turns = SessionStore.load(tmp_path, "s1")
    assert len(turns) == 1
    assert turns[0]["prompt"] == "do something risky"
    assert turns[0]["response"] == INTERRUPTED_MARKER
    assert turns[0]["interrupted"] is True


def test_interrupted_turn_keeps_its_chronological_position(tmp_path: Path):
    store = SessionStore(tmp_path, "s1")
    store.append({"prompt": "first", "response": "ok"})  # pre-existing (legacy) complete record
    store.start_turn("second, never finishes")
    store.append({"prompt": "third", "response": "also ok"})  # a later, unrelated legacy record
    turns = SessionStore.load(tmp_path, "s1")
    assert [t["prompt"] for t in turns] == ["first", "second, never finishes", "third"]
    assert not turns[0].get("interrupted") and turns[1]["interrupted"] is True and not turns[2].get("interrupted")


def test_list_sessions_counts_the_interrupted_turn(tmp_path: Path):
    store = SessionStore(tmp_path, "s1")
    store.start_turn("never finishes")
    listing = SessionStore.list_sessions(tmp_path)
    assert listing[0]["turns"] == 1 and listing[0]["first_prompt"] == "never finishes"


def test_legacy_sessions_with_no_type_field_still_load_unchanged(tmp_path: Path):
    """Regression guard: sessions written before turn_start/turn_end reconciliation existed must
    keep loading exactly as before - _read must not require every record to carry a "type"."""
    store = SessionStore(tmp_path, "s1")
    store.append({"prompt": "hi", "response": "hello", "provider": "p"})
    turns = SessionStore.load(tmp_path, "s1")
    assert turns == [{"timestamp": turns[0]["timestamp"], "prompt": "hi", "response": "hello", "provider": "p"}]
