"""Agent loop tests with a scripted streaming provider."""
import asyncio
import copy
import os
import time
from pathlib import Path

import httpx
import pytest

from core.agent_loop import _ToolTagFilter
from core.context import compact_messages, estimate_tokens, messages_tokens, trim_old_tool_results
from core.providers import BaseProvider, ModelConfig, NativeToolsUnsupported, StreamEvent, ToolCall
from core.toolstate import ToolSession
from core.workspace_tools import WorkspaceTools
from main import MotionAgent


def text(*chunks):
    return [StreamEvent("text", text=c) for c in chunks]


def call(cid, tool_name, **args):
    return [StreamEvent("tool_call", tool_call=ToolCall(cid, tool_name, args))]


def calls(*items):
    return [StreamEvent("tool_call", tool_call=ToolCall(cid, name, args)) for cid, name, args in items]


class Scripted(BaseProvider):
    def __init__(self, steps, native=True, vision=False, window=32768):
        super().__init__(ModelConfig(
            name="s", endpoint="http://x", provider_type="local",
            options={"native_tools": native, "model": "m", "vision": vision, "context_window": window},
        ))
        self.steps = list(steps)
        self.requests = []

    async def chat_stream(self, messages, system_prompt="", tools=None, **kw):
        self.requests.append({"messages": copy.deepcopy(messages), "system": system_prompt, "tools": tools})
        step = self.steps.pop(0)
        if isinstance(step, Exception):
            raise step
        for ev in step:
            yield ev
            await asyncio.sleep(0)


class EmptyRetriever:
    async def retrieve(self, query, top_k=5):
        return []


def make_agent(provider):
    agent = MotionAgent(ModelConfig(name="t", endpoint="http://x", provider_type="local"), memory_path=":memory:")
    agent.provider = provider
    agent.retriever = EmptyRetriever()
    return agent


async def run(agent, prompt="do the thing", **kw):
    chunks, traces = [], []
    kw.setdefault("workspace", None)
    resp = await agent.run(
        prompt,
        on_stream_chunk=lambda c: chunks.append(c),
        on_trace_event=lambda stage, payload: traces.append((stage, payload)),
        **kw,
    )
    return resp, chunks, traces


def tool_msgs(request):
    return [m for m in request["messages"] if m["role"] == "tool"]


# ── native tool loop ────────────────────────────────────────────────────────

async def test_native_loop_streams_and_keeps_original_prompt(tmp_path: Path):
    (tmp_path / "a.txt").write_text("hello")
    p = Scripted([
        text("Let me look") + call("c1", "read_file", path="a.txt"),
        text("It says ", "hello."),
    ])
    resp, chunks, traces = await run(make_agent(p), "What does a.txt say?", workspace=str(tmp_path), agent_mode="build")

    assert resp == "It says hello."
    # The original request is still in the conversation on every step (regression:
    # it used to vanish after the first tool call).
    for req in p.requests:
        assert req["messages"][0] == {"role": "user", "content": "What does a.txt say?"}
    # Native tool schemas were sent, and the tool result came back as a tool message.
    assert {t["name"] for t in p.requests[0]["tools"]} >= {"read_file", "grep", "run_command", "todo_write"}
    second = p.requests[1]["messages"]
    assert [m["role"] for m in second] == ["user", "assistant", "tool"]
    assert second[1]["tool_calls"][0]["name"] == "read_file"
    assert second[2]["tool_call_id"] == "c1" and '"hello"' in second[2]["content"]
    assert not second[2]["content"].startswith("<motion_tool_result>")  # bare JSON in native mode
    # Streaming markers: narration is flagged as intermediate, the answer is not.
    assert chunks[:2] == ["_delta_ Let me look", "_endstep_ "]
    assert "_delta_ It says " in chunks and "_delta_ hello." in chunks
    assert chunks[-1] != "It says hello."  # the final text is not re-emitted
    assert any(c.startswith("_tool_ read `a.txt`") for c in chunks)
    stages = [s for s, _ in traces]
    assert stages[0] == "turn_start" and "turn_done" in stages and stages.count("model_step") == 2


async def test_parallel_safe_calls_run_concurrently(tmp_path: Path, monkeypatch):
    for n in "ab":
        (tmp_path / f"{n}.txt").write_text(n)
    windows = []

    orig = WorkspaceTools.aexecute

    async def slow(self, name, arguments, on_output=None):
        start = time.monotonic()
        await asyncio.sleep(0.2)
        result = await orig(self, name, arguments, on_output)
        windows.append((start, time.monotonic()))
        return result

    monkeypatch.setattr(WorkspaceTools, "aexecute", slow)
    p = Scripted([
        calls(("1", "read_file", {"path": "a.txt"}), ("2", "read_file", {"path": "b.txt"})),
        text("done"),
    ])
    t0 = time.monotonic()
    await run(make_agent(p), workspace=str(tmp_path))
    assert time.monotonic() - t0 < 0.38  # two 0.2s calls overlapped
    assert windows[0][0] < windows[1][1] and windows[1][0] < windows[0][1]
    results = tool_msgs(p.requests[1])
    assert [m["tool_call_id"] for m in results] == ["1", "2"]  # order preserved


async def test_mutating_calls_run_sequentially_in_order(tmp_path: Path):
    p = Scripted([
        calls(("1", "write_file", {"path": "x.txt", "content": "1"}),
              ("2", "run_command", {"command": "cat x.txt"})),
        text("ok"),
    ])
    await run(make_agent(p), workspace=str(tmp_path), agent_mode="build")
    out = tool_msgs(p.requests[1])[1]["content"]
    assert '"stdout": "1"' in out  # the write happened before the command ran


async def test_run_command_does_not_block_the_event_loop(tmp_path: Path):
    p = Scripted([call("1", "run_command", command="sleep 1"), text("done")])
    ticks = []

    async def heartbeat():
        while True:
            ticks.append(time.monotonic())
            await asyncio.sleep(0.05)

    hb = asyncio.create_task(heartbeat())
    await run(make_agent(p), workspace=str(tmp_path), agent_mode="build")
    hb.cancel()
    gaps = [b - a for a, b in zip(ticks, ticks[1:])]
    assert len(ticks) > 12  # ~20 expected; the old blocking call gave 0-1
    assert max(gaps) < 0.5


async def test_cancelling_a_turn_kills_the_running_command(tmp_path: Path):
    pidfile = tmp_path / "pid"
    p = Scripted([call("1", "run_command", command=f"echo $$ > {pidfile}; sleep 30"), text("never")])
    task = asyncio.create_task(run(make_agent(p), workspace=str(tmp_path), agent_mode="build"))
    for _ in range(100):
        if pidfile.exists() and pidfile.read_text().strip():
            break
        await asyncio.sleep(0.05)
    pid = int(pidfile.read_text())
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0.2)
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


async def test_streaming_output_lines_reach_the_ui(tmp_path: Path):
    p = Scripted([call("1", "run_command", command="echo first; sleep 0.7; echo second"), text("ok")])
    _, chunks, _ = await run(make_agent(p), workspace=str(tmp_path), agent_mode="build")
    assert any(c.startswith("_out_ run_command: ") for c in chunks)


# ── fallbacks ───────────────────────────────────────────────────────────────

async def test_falls_back_to_text_tools_when_native_tools_rejected(tmp_path: Path):
    (tmp_path / "f.txt").write_text("data")
    p = Scripted([
        NativeToolsUnsupported("model does not support tools", 400, "tools"),
        text('<motion_tool>{"name":"read_file","arguments":{"path":"f.txt"}}</motion_tool>'),
        text("got data"),
    ])
    resp, chunks, traces = await run(make_agent(p), "read f", workspace=str(tmp_path), agent_mode="build")
    assert resp == "got data"
    assert p.native_tools is False
    assert p.requests[1]["tools"] is None
    assert "<motion_tool>" in p.requests[1]["system"]
    assert any(s == "native_tools_disabled" for s, _ in traces)
    # XML mode: tool result is a user message with the envelope.
    assert p.requests[2]["messages"][-1]["content"].startswith("<motion_tool_result>")
    assert p.requests[2]["messages"][0]["content"] == "read f"


async def test_xml_stream_hides_tool_markup_from_the_user(tmp_path: Path):
    (tmp_path / "f.txt").write_text("z")
    p = Scripted([
        text("Checking. <mo", 'tion_tool>{"name":"read_file",', '"arguments":{"path":"f.txt"}}</motion_tool>'),
        text("All ", "good"),
    ], native=False)
    resp, chunks, _ = await run(make_agent(p), workspace=str(tmp_path), agent_mode="build")
    assert resp == "All good"
    shown = "".join(c[len("_delta_ "):] for c in chunks if c.startswith("_delta_ "))
    assert "motion_tool" not in shown and "Checking." in shown


def test_tool_tag_filter_passes_ordinary_angle_brackets():
    f = _ToolTagFilter()
    out = f.feed("use a <div> and x < 5, then ") + f.feed("List<int> ") + f.flush()
    assert out == "use a <div> and x < 5, then List<int> "
    g = _ToolTagFilter()
    assert g.feed("ok <list_files>{") == "ok " and g.feed("...") == ""


async def test_provider_timeout_returns_friendly_message(tmp_path: Path):
    p = Scripted([httpx.ReadTimeout("slow")])
    resp, _, traces = await run(make_agent(p), workspace=str(tmp_path))
    assert resp.startswith("⚠️") and "timed out" in resp
    assert any(s == "provider_error" for s, _ in traces)


async def test_malformed_native_arguments_are_reported_to_the_model(tmp_path: Path):
    bad = [StreamEvent("tool_call", tool_call=ToolCall("1", "read_file", {}, parse_error="invalid JSON arguments: x"))]
    p = Scripted([bad, text("recovered")])
    resp, _, _ = await run(make_agent(p), workspace=str(tmp_path))
    assert resp == "recovered"
    assert "invalid JSON" in tool_msgs(p.requests[1])[0]["content"]


# ── permissions & safety ────────────────────────────────────────────────────

async def test_risky_command_needs_approval_and_is_refused_without_a_ui(tmp_path: Path):
    (tmp_path / "build").mkdir()
    p = Scripted([call("1", "run_command", command="rm -rf build"), text("ok")])
    await run(make_agent(p), workspace=str(tmp_path), agent_mode="build")
    assert (tmp_path / "build").exists()
    assert "needs user approval" in tool_msgs(p.requests[1])[0]["content"]


async def test_risky_command_runs_after_user_approves_once(tmp_path: Path):
    (tmp_path / "build").mkdir()
    asked = []

    def approve(kind, subject, reason):
        asked.append((kind, subject))
        return "once"

    p = Scripted([call("1", "run_command", command="rm -rf build"), text("ok")])
    await run(make_agent(p), workspace=str(tmp_path), agent_mode="build", on_approval=approve)
    assert not (tmp_path / "build").exists()
    assert asked == [("command", "rm -rf build")]


async def test_session_approval_is_remembered_across_turns(tmp_path: Path):
    asked = []
    session = ToolSession()

    async def approve(kind, subject, reason):
        asked.append(subject)
        return "session"

    for _ in range(2):
        (tmp_path / "d").mkdir(exist_ok=True)
        p = Scripted([call("1", "run_command", command="rm -rf d"), text("ok")])
        await run(make_agent(p), workspace=str(tmp_path), agent_mode="build", on_approval=approve, session=session)
    assert asked == ["rm -rf d"]  # second turn did not re-prompt


async def test_catastrophic_command_is_always_refused(tmp_path: Path):
    p = Scripted([call("1", "run_command", command="rm -rf /"), text("ok")])
    await run(make_agent(p), workspace=str(tmp_path), agent_mode="build", on_approval=lambda *a: "session")
    assert "command refused" in tool_msgs(p.requests[1])[0]["content"]


async def test_overwriting_an_unread_file_is_blocked_until_read(tmp_path: Path):
    (tmp_path / "keep.txt").write_text("precious")
    p = Scripted([
        call("1", "write_file", path="keep.txt", content="oops"),
        call("2", "read_file", path="keep.txt"),
        call("3", "write_file", path="keep.txt", content="fine"),
        text("done"),
    ])
    await run(make_agent(p), workspace=str(tmp_path), agent_mode="build")
    assert "has not been read" in tool_msgs(p.requests[1])[0]["content"]
    assert (tmp_path / "keep.txt").read_text() == "fine"


async def test_undo_restores_the_whole_turn(tmp_path: Path):
    (tmp_path / "old.txt").write_text("v1")
    session = ToolSession()
    p = Scripted([
        call("1", "read_file", path="old.txt"),
        calls(("2", "write_file", {"path": "old.txt", "content": "v2"}),
              ("3", "write_file", {"path": "new.txt", "content": "n"})),
        text("done"),
    ])
    await run(make_agent(p), workspace=str(tmp_path), agent_mode="build", session=session)
    assert (tmp_path / "old.txt").read_text() == "v2" and (tmp_path / "new.txt").exists()
    lines = session.checkpoints.undo_last_turn()
    assert len(lines) == 2
    assert (tmp_path / "old.txt").read_text() == "v1" and not (tmp_path / "new.txt").exists()


async def test_plan_mode_hides_and_blocks_mutating_tools(tmp_path: Path):
    p = Scripted([call("1", "write_file", path="x", content="y"), text("Plan: …")])
    resp, _, _ = await run(make_agent(p), workspace=str(tmp_path), agent_mode="plan")
    names = {t["name"] for t in p.requests[0]["tools"]}
    assert "read_file" in names and "write_file" not in names and "run_command" not in names
    assert not (tmp_path / "x").exists()
    assert "disabled in plan mode" in tool_msgs(p.requests[1])[0]["content"]
    assert "read-only Plan mode" in p.requests[1]["messages"][-1]["content"]
    assert resp == "Plan: …"


async def test_out_of_workspace_permission_flow(tmp_path: Path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "o.txt").write_text("secret")
    ws = tmp_path / "ws"
    ws.mkdir()
    allowed = set()
    p = Scripted([call("1", "read_file", path="../outside/o.txt"), text("ok")])
    await run(make_agent(p), workspace=str(ws), on_permission_request=lambda path: "session", allowed_paths=allowed)
    assert '"secret"' in tool_msgs(p.requests[1])[0]["content"]
    assert (outside / "o.txt").resolve() in allowed


async def test_subprocess_environment_hides_api_keys(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-should-not-leak")
    p = Scripted([call("1", "run_command", command="echo key=[$OPENAI_API_KEY]"), text("ok")])
    await run(make_agent(p), workspace=str(tmp_path), agent_mode="build")
    out = tool_msgs(p.requests[1])[0]["content"]
    assert "sk-should-not-leak" not in out and "key=[]" in out


# ── meta tools ──────────────────────────────────────────────────────────────

async def test_todo_write_updates_session_and_notifies_ui(tmp_path: Path):
    seen = []
    session = ToolSession()
    p = Scripted([
        call("1", "todo_write", todos=[{"content": "a", "status": "completed"}, {"content": "b", "status": "in_progress"}]),
        text("done"),
    ])
    _, _, traces = await run(make_agent(p), workspace=str(tmp_path), on_todo=seen.append, session=session)
    assert seen[0][1] == {"content": "b", "status": "in_progress"}
    assert session.todos == seen[0]
    assert any(s == "todo_update" for s, _ in traces)


async def test_ask_user_round_trip_and_unavailable(tmp_path: Path):
    async def answer(question, options):
        assert question == "Which DB?" and options == ["sqlite", "pg"]
        return "sqlite"

    p = Scripted([call("1", "ask_user", question="Which DB?", options=["sqlite", "pg"]), text("using sqlite")])
    resp, _, _ = await run(make_agent(p), workspace=str(tmp_path), on_ask_user=answer)
    assert '"answer": "sqlite"' in tool_msgs(p.requests[1])[0]["content"]

    q = Scripted([call("1", "ask_user", question="Which DB?"), text("assuming")])
    await run(make_agent(q), workspace=str(tmp_path))
    assert "no interactive user" in tool_msgs(q.requests[1])[0]["content"]


async def test_memory_save_persists_in_the_memory_db(tmp_path: Path):
    agent = make_agent(Scripted([call("1", "memory_save", key="k", text="remember me"), text("saved")]))
    await run(agent, workspace=str(tmp_path))
    assert agent.memory.get_note("k") == "remember me"
    # ...and is recallable by a later session through keyword search.
    assert agent.memory.keyword_search("remember")


async def test_grep_and_glob_tools_via_agent(tmp_path: Path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "m.py").write_text("def main():\n    pass\n")
    p = Scripted([call("1", "grep", pattern="def main", glob="*.py"), text("found")])
    await run(make_agent(p), workspace=str(tmp_path))
    assert '"path": "src/m.py"' in tool_msgs(p.requests[1])[0]["content"]


# ── vision ──────────────────────────────────────────────────────────────────

PNG = bytes.fromhex("89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4890000000d49444154789c6360000002000001e221bc330000000049454e44ae426082")


async def test_attached_images_become_native_image_parts_for_vision_models(tmp_path: Path):
    p = Scripted([text("a pixel")], vision=True)
    await run(make_agent(p), "what is this?", workspace=str(tmp_path),
              images=[{"name": "x.png", "mime": "image/png", "data": "QUJD"}])
    content = p.requests[0]["messages"][0]["content"]
    assert content[0] == {"type": "text", "text": "what is this?"}
    assert content[1] == {"type": "image", "mime": "image/png", "data": "QUJD"}


async def test_images_degrade_gracefully_for_text_only_models(tmp_path: Path):
    p = Scripted([text("sorry")], vision=False)
    await run(make_agent(p), "what is this?", workspace=str(tmp_path),
              images=[{"name": "x.png", "mime": "image/png", "data": "QUJD"}])
    content = p.requests[0]["messages"][0]["content"]
    assert isinstance(content, str) and "does not accept image input" in content


async def test_read_image_tool_attaches_pixels_only_when_model_can_see(tmp_path: Path):
    (tmp_path / "x.png").write_bytes(PNG)
    p = Scripted([call("1", "read_image", path="x.png"), text("seen")], vision=True)
    await run(make_agent(p), workspace=str(tmp_path))
    msgs = p.requests[1]["messages"]
    assert "data_url" not in msgs[2]["content"]  # no base64 dumped into the text
    assert msgs[-1]["role"] == "user" and msgs[-1]["content"][1]["type"] == "image"

    q = Scripted([call("1", "read_image", path="x.png"), text("blind")], vision=False)
    await run(make_agent(q), workspace=str(tmp_path))
    assert "cannot view images" in tool_msgs(q.requests[1])[0]["content"]


# ── recall & context ────────────────────────────────────────────────────────

async def test_slow_memory_recall_never_stalls_the_turn(tmp_path: Path):
    class Slow:
        async def retrieve(self, query, top_k=5):
            await asyncio.sleep(5)
            return [{"content": "LATE_MEMORY_MARKER"}]

    agent = make_agent(Scripted([text("hi")]))
    agent.retriever = Slow()
    agent.recall_timeout = 0.1
    t0 = time.monotonic()
    resp, _, traces = await run(agent, workspace=str(tmp_path))
    assert time.monotonic() - t0 < 1.5 and resp == "hi"
    assert any(s == "memory_recall_timeout" for s, _ in traces)
    assert "LATE_MEMORY_MARKER" not in agent.provider.requests[0]["system"]


async def test_failing_retriever_does_not_fail_the_turn(tmp_path: Path):
    class Broken:
        async def retrieve(self, query, top_k=5):
            raise RuntimeError("db locked")

    agent = make_agent(Scripted([text("still works")]))
    agent.retriever = Broken()
    resp, _, _ = await run(agent, workspace=str(tmp_path))
    assert resp == "still works"


async def test_project_instructions_and_environment_reach_the_system_prompt(tmp_path: Path):
    (tmp_path / "AGENTS.md").write_text("Always use tabs.")
    p = Scripted([text("ok")])
    await run(make_agent(p), workspace=str(tmp_path))
    system = p.requests[0]["system"]
    assert "Always use tabs." in system and "Working directory:" in system and "Today's date:" in system


async def test_saved_skills_are_advertised_and_loadable(tmp_path: Path):
    skills = tmp_path / ".motion" / "skills"
    skills.mkdir(parents=True)
    (skills / "deploy.md").write_text("# Deploy\nRun make ship.\n")
    p = Scripted([call("1", "use_skill", name="deploy"), text("ok")])
    await run(make_agent(p), workspace=str(tmp_path))
    assert "- deploy:" in p.requests[0]["system"]
    assert "make ship" in tool_msgs(p.requests[1])[0]["content"]


async def test_old_tool_output_is_trimmed_between_steps(tmp_path: Path):
    (tmp_path / "big.txt").write_text("x" * 50_000)
    steps = [call(str(i), "read_file", path="big.txt") for i in range(9)] + [text("done")]
    p = Scripted(steps, window=10_000_000)
    await run(make_agent(p), workspace=str(tmp_path))
    last = p.requests[-1]["messages"]
    tools = [m["content"] for m in last if m["role"] == "tool"]
    assert "trimmed" in tools[0] and len(tools[0]) < 2500  # oldest was cut
    assert "trimmed" not in tools[-1]  # newest is intact


async def test_context_is_compacted_when_it_nears_the_window(tmp_path: Path):
    (tmp_path / "big.txt").write_text("x" * 50_000)
    steps = [call(str(i), "read_file", path="big.txt") for i in range(9)] + [text("done")]
    p = Scripted(steps, window=32768)
    resp, _, traces = await run(make_agent(p), workspace=str(tmp_path))
    assert resp == "done"
    assert any(s == "context_compacted" for s, _ in traces)
    last = p.requests[-1]["messages"]
    assert last[0]["content"] == "do the thing"  # original request always survives
    assert any("trimmed to save context" in str(m["content"]) for m in last)
    for i, m in enumerate(last):  # tool results still follow the call that produced them
        if m["role"] == "tool":
            assert last[i - 1]["role"] in ("assistant", "tool")


def test_trim_and_compaction_keep_tool_pairs_valid():
    msgs = [{"role": "user", "content": "task"}]
    for i in range(12):
        msgs.append({"role": "assistant", "content": "", "tool_calls": [{"id": str(i), "name": "read_file", "arguments": {"path": f"f{i}"}}]})
        msgs.append({"role": "tool", "tool_call_id": str(i), "name": "read_file", "content": "y" * 4000})
    before = messages_tokens(msgs)
    assert compact_messages(msgs, "", window_tokens=before // 2, start_index=0)
    assert msgs[0]["content"] == "task" and "trimmed to save context" in msgs[1]["content"]
    # every remaining tool message still follows the assistant message that issued it
    for i, m in enumerate(msgs):
        if m["role"] == "tool":
            assert msgs[i - 1]["role"] in ("assistant", "tool")
    assert messages_tokens(msgs) < before

    small = [{"role": "user", "content": "hi"}]
    assert not compact_messages(small, "", 10_000, 0)
    assert trim_old_tool_results([{"role": "tool", "content": "z" * 9000}] * 2, keep_recent=1) > 0
    assert estimate_tokens("abcd" * 10) == 10


async def test_auto_remember_stores_substantive_turns_in_background(tmp_path: Path):
    agent = make_agent(Scripted([call("1", "list_files"), text("There are no files here.")]))
    agent.auto_remember = True
    await run(agent, "please list the files in here", workspace=str(tmp_path))
    await asyncio.gather(*list(agent._bg_tasks))
    hits = agent.memory.keyword_search("list files")
    assert hits and "please list the files" in hits[0][1]
