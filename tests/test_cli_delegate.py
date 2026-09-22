"""CLI-delegate providers (Claude Code / Codex), driven against stand-in scripts on PATH - never the
real binaries or a real login. Verifies argv construction, NDJSON streaming, session continuity,
cancellation, and the failure/timeout paths, all without a network call or a real subscription."""
import asyncio
import json
import os
import stat
import sys
import textwrap
from pathlib import Path

import pytest

from core.cli_delegate import (
    ClaudeCLIProvider,
    CLIDelegateError,
    CodexCLIProvider,
    claude_cli_available,
    clear_detection_cache,
    codex_cli_available,
)
from core.providers import ModelConfig

PY = sys.executable


def make_config(**opts):
    return ModelConfig(name="x", endpoint="", provider_type="cli", options=opts)


@pytest.fixture
def fake_bin(tmp_path, monkeypatch):
    """Writes a Python script as an executable `<name>` on a PATH prepended ahead of everything else,
    then restores PATH and clears the availability cache on teardown."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()

    def install(name: str, body: str) -> Path:
        path = bin_dir / name
        path.write_text(f"#!{PY}\n{body}\n")
        path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        return path

    old_path = os.environ.get("PATH", "")
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{old_path}")
    clear_detection_cache()
    yield install
    clear_detection_cache()


def ndjson_script(*lines, exit_code=0, stderr="") -> str:
    """A script body that prints each line as NDJSON to stdout and exits with the given code."""
    payload = json.dumps([json.dumps(l) for l in lines])
    return textwrap.dedent(f"""
        import sys, json
        lines = json.loads({payload!r})
        for l in lines:
            print(l, flush=True)
        {"sys.stderr.write(" + repr(stderr) + ")" if stderr else ""}
        sys.exit({exit_code})
    """)


# ── availability detection ──────────────────────────────────────────────────

def test_detection_reflects_path_and_is_cached(fake_bin):
    assert claude_cli_available() is False and codex_cli_available() is False
    fake_bin("claude", "pass")
    assert claude_cli_available() is False                          # cached from the check above
    clear_detection_cache()
    assert claude_cli_available() is True and codex_cli_available() is False


# ── Claude Code delegate ─────────────────────────────────────────────────────

CLAUDE_HAPPY = [
    {"type": "system", "subtype": "init", "session_id": "sess-abc"},
    {"type": "stream_event", "event": {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Look"}}},
    {"type": "stream_event", "event": {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "ing..."}}},
    {"type": "assistant", "message": {"content": [{"type": "tool_use", "name": "Read", "input": {"file_path": "a.py"}}]}},
    {"type": "stream_event", "event": {"type": "content_block_delta", "delta": {"type": "text_delta", "text": " Found it."}}},
    {"type": "result", "subtype": "success", "session_id": "sess-abc", "total_cost_usd": 0.0142,
     "usage": {"input_tokens": 500, "output_tokens": 40}, "result": "Looking... Found it."},
]


async def test_claude_delegate_happy_path(tmp_path, fake_bin):
    fake_bin("claude", ndjson_script(*CLAUDE_HAPPY))
    p = ClaudeCLIProvider(make_config())
    events = []
    result = await p.run_delegate("investigate", mode="build", workspace=str(tmp_path), on_event=events.append)
    assert result.text == "Looking... Found it."
    assert result.session_ref == "sess-abc" and result.cost_usd == pytest.approx(0.0142)
    assert result.usage == {"prompt_tokens": 500, "completion_tokens": 40, "total_tokens": 540}
    assert result.tool_lines == ["🔧 Read(file_path=a.py)"]
    kinds = [e.kind for e in events]
    assert kinds.count("text") == 3 and "reasoning" in kinds        # live text deltas + tool visibility
    assert "".join(e.text for e in events if e.kind == "text") == "Looking... Found it."


async def test_claude_delegate_session_continuity(tmp_path, fake_bin):
    calls = []
    fake_bin("claude", textwrap.dedent(f"""
        import sys, json
        calls_path = {str(tmp_path / "calls.json")!r}
        import os
        prior = json.load(open(calls_path)) if os.path.exists(calls_path) else []
        prior.append(sys.argv[1:])
        json.dump(prior, open(calls_path, "w"))
        resumed = "--resume" in sys.argv
        sid = "sess-1"
        print(json.dumps({{"type": "system", "subtype": "init", "session_id": sid}}))
        print(json.dumps({{"type": "result", "result": ("second " if resumed else "first "), "session_id": sid}}))
    """))
    p = ClaudeCLIProvider(make_config())
    r1 = await p.run_delegate("one", mode="build", workspace=str(tmp_path))
    assert r1.text == "first" and p.session_ref is None             # run_delegate doesn't self-update; caller does (see agent_loop)
    p.session_ref = r1.session_ref
    r2 = await p.run_delegate("two", mode="build", workspace=str(tmp_path))
    assert r2.text == "second"
    calls = json.loads((tmp_path / "calls.json").read_text())
    assert "--resume" not in calls[0] and "--resume" in calls[1] and calls[1][calls[1].index("--resume") + 1] == "sess-1"


async def test_claude_delegate_plan_mode_only_allows_read_only_tools(tmp_path, fake_bin):
    seen = {}
    fake_bin("claude", textwrap.dedent(f"""
        import sys, json
        json.dump(sys.argv[1:], open({str(tmp_path / "argv.json")!r}, "w"))
        print(json.dumps({{"type": "result", "result": "ok"}}))
    """))
    p = ClaudeCLIProvider(make_config())
    await p.run_delegate("look only", mode="plan", workspace=str(tmp_path))
    argv = json.loads((tmp_path / "argv.json").read_text())
    tools = argv[argv.index("--allowedTools") + 1]
    assert "Write" not in tools and "Edit" not in tools and "Bash" not in tools and "Read" in tools
    assert "--permission-mode" not in argv                          # no acceptEdits in plan mode


async def test_claude_delegate_error_result_raises(tmp_path, fake_bin):
    fake_bin("claude", ndjson_script(
        {"type": "system", "subtype": "init", "session_id": "s1"},
        {"type": "result", "is_error": True, "result": "usage limit reached", "session_id": "s1"},
    ))
    p = ClaudeCLIProvider(make_config())
    with pytest.raises(CLIDelegateError, match="usage limit reached"):
        await p.run_delegate("hi", mode="build", workspace=str(tmp_path))


async def test_claude_delegate_falls_back_to_result_text_if_no_deltas_streamed(tmp_path, fake_bin):
    """A very short reply can arrive with no partial-message deltas at all - the final result's own
    text must still be used, not silently dropped."""
    fake_bin("claude", ndjson_script(
        {"type": "system", "subtype": "init", "session_id": "s1"},
        {"type": "result", "result": "OK.", "session_id": "s1"},
    ))
    p = ClaudeCLIProvider(make_config())
    result = await p.run_delegate("hi", mode="build", workspace=str(tmp_path))
    assert result.text == "OK."


# ── Codex delegate ───────────────────────────────────────────────────────────

CODEX_HAPPY = [
    {"type": "thread.started", "thread_id": "th-1"},
    {"type": "turn.started"},
    {"type": "item.completed", "item": {"id": "i1", "type": "command_execution", "command": "ls"}},
    {"type": "item.completed", "item": {"id": "i2", "type": "agent_message", "text": "Done listing."}},
    {"type": "turn.completed", "usage": {"input_tokens": 300, "output_tokens": 12}},
]


async def test_codex_delegate_happy_path(tmp_path, fake_bin):
    fake_bin("codex", ndjson_script(*CODEX_HAPPY))
    p = CodexCLIProvider(make_config())
    events = []
    result = await p.run_delegate("list files", mode="build", workspace=str(tmp_path), on_event=events.append)
    assert result.text == "Done listing." and result.session_ref == "th-1"
    assert result.cost_usd is None                                  # subscription usage: no dollar figure invented
    assert result.usage == {"prompt_tokens": 300, "completion_tokens": 12, "total_tokens": 312}
    assert result.tool_lines == ["🔧 command_execution(ls)"]


async def test_codex_delegate_sandbox_mode_mirrors_plan_build(tmp_path, fake_bin):
    fake_bin("codex", textwrap.dedent(f"""
        import sys, json
        json.dump(sys.argv[1:], open({str(tmp_path / "argv.json")!r}, "w"))
        print(json.dumps({{"type": "turn.completed", "usage": {{}}}}))
        print(json.dumps({{"type": "item.completed", "item": {{"type": "agent_message", "text": "ok"}}}}))
    """))
    p = CodexCLIProvider(make_config())
    await p.run_delegate("x", mode="plan", workspace=str(tmp_path))
    argv = json.loads((tmp_path / "argv.json").read_text())
    assert argv[argv.index("--sandbox") + 1] == "read-only"
    await p.run_delegate("x", mode="build", workspace=str(tmp_path))
    argv = json.loads((tmp_path / "argv.json").read_text())
    assert argv[argv.index("--sandbox") + 1] == "workspace-write"


async def test_codex_delegate_failed_turn_raises(tmp_path, fake_bin):
    fake_bin("codex", ndjson_script(
        {"type": "thread.started", "thread_id": "th-1"},
        {"type": "turn.failed", "message": "sandbox denied the command"},
    ))
    p = CodexCLIProvider(make_config())
    with pytest.raises(CLIDelegateError, match="sandbox denied"):
        await p.run_delegate("x", mode="build", workspace=str(tmp_path))


# ── shared machinery: errors, timeouts, cancellation ─────────────────────────

async def test_missing_binary_raises_immediately(tmp_path):
    p = ClaudeCLIProvider(make_config())
    with pytest.raises(CLIDelegateError, match="claude"):
        await p.run_delegate("hi", mode="build", workspace=str(tmp_path))


async def test_nonzero_exit_with_no_output_raises_with_stderr(tmp_path, fake_bin):
    fake_bin("claude", "import sys; sys.stderr.write('auth expired, run `claude login`\\n'); sys.exit(1)")
    p = ClaudeCLIProvider(make_config())
    with pytest.raises(CLIDelegateError, match="auth expired"):
        await p.run_delegate("hi", mode="build", workspace=str(tmp_path))


async def test_nonzero_exit_after_a_real_result_is_not_treated_as_failure(tmp_path, fake_bin):
    """A process that streamed a proper result and then exited oddly still counts as done - matching
    what an interactive run would show."""
    fake_bin("claude", ndjson_script(
        {"type": "system", "subtype": "init", "session_id": "s1"},
        {"type": "result", "result": "finished anyway", "session_id": "s1"},
        exit_code=1,
    ))
    p = ClaudeCLIProvider(make_config())
    result = await p.run_delegate("hi", mode="build", workspace=str(tmp_path))
    assert result.text == "finished anyway"


async def test_a_stalled_process_times_out_and_is_killed(tmp_path, fake_bin):
    fake_bin("claude", "import time; time.sleep(30)")
    p = ClaudeCLIProvider(make_config(timeout=0.5))
    with pytest.raises(CLIDelegateError, match="no output"):
        await p.run_delegate("hi", mode="build", workspace=str(tmp_path))
    assert p._proc is None


async def test_cancelling_the_turn_kills_the_subprocess(tmp_path, fake_bin):
    pidfile = tmp_path / "pid"
    fake_bin("claude", textwrap.dedent(f"""
        import os, time
        open({str(pidfile)!r}, "w").write(str(os.getpid()))
        time.sleep(30)
    """))
    p = ClaudeCLIProvider(make_config())
    task = asyncio.ensure_future(p.run_delegate("hi", mode="build", workspace=str(tmp_path)))
    for _ in range(50):
        if pidfile.exists():
            break
        await asyncio.sleep(0.05)
    pid = int(pidfile.read_text())
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0.2)
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)                                              # the process is really gone


async def test_malformed_json_lines_are_skipped_not_fatal(tmp_path, fake_bin):
    fake_bin("claude", textwrap.dedent("""
        print("not json at all")
        print("")
        import json
        print(json.dumps({"type": "system", "subtype": "init", "session_id": "s1"}))
        print(json.dumps({"type": "result", "result": "fine", "session_id": "s1"}))
    """))
    p = ClaudeCLIProvider(make_config())
    result = await p.run_delegate("hi", mode="build", workspace=str(tmp_path))
    assert result.text == "fine"
