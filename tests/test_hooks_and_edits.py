"""User hooks (pre/post tool) and the atomic multi-file edit tool."""
import json
import stat
import sys
from pathlib import Path

import pytest

from core.hooks import Hooks
from core.workspace_tools import WorkspaceToolError, WorkspaceTools
from tests.test_agent_loop import Scripted, call, calls, make_agent, run, text, tool_msgs

PY = sys.executable


def hooks_from(cfg: dict) -> Hooks:
    return Hooks.from_config({"hooks": cfg}.get)


def script(tmp_path: Path, name: str, body: str) -> str:
    path = tmp_path / name
    path.write_text(f"#!/bin/sh\n{body}\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return str(path)


# ── config ──────────────────────────────────────────────────────────────────

def test_config_parsing_reports_bad_entries_instead_of_crashing():
    h = hooks_from({"pre_tool": [{"command": "true"}, {"match": "(", "command": "true"}, {"match": "x"}, "junk"],
                    "post_tool": [{"match": "write_file", "command": "true", "timeout": "5"}]})
    assert len(h.hooks) == 2 and len(h.problems) == 3 and h                       # the two valid ones survive
    assert [x.event for x in h.hooks] == ["pre_tool", "post_tool"] and h.hooks[1].timeout == 5.0
    assert not Hooks.from_config({}.get) and Hooks.from_config({"hooks": "x"}.get).problems


def test_match_is_a_full_regex_on_the_tool_name():
    h = hooks_from({"pre_tool": [{"match": "write_file|edit_files", "command": "true"}]})
    assert h.matching("pre_tool", "write_file") and h.matching("pre_tool", "edit_files")
    assert not h.matching("pre_tool", "read_file") and not h.matching("pre_tool", "write_file_x")
    assert not h.matching("post_tool", "write_file")


# ── running hooks ───────────────────────────────────────────────────────────

async def test_hook_gets_the_call_as_json_and_env(tmp_path: Path):
    out = tmp_path / "seen.json"
    cmd = script(tmp_path, "h.sh", f'cat > {out}; echo "$MOTION_HOOK_EVENT $MOTION_TOOL" >> {out}.env')
    r = await hooks_from({"pre_tool": [{"command": cmd}]}).run("pre_tool", "write_file", {"path": "a.txt", "content": "x"}, str(tmp_path))
    assert not r.blocked
    seen = json.loads(out.read_text())
    assert seen["tool"] == "write_file" and seen["arguments"]["path"] == "a.txt" and seen["event"] == "pre_tool"
    assert Path(f"{out}.env").read_text().strip() == "pre_tool write_file"


async def test_pre_hook_failure_blocks_with_its_output_and_success_allows(tmp_path: Path):
    deny = script(tmp_path, "deny.sh", 'echo "generated files are read-only" >&2; exit 3')
    ok = script(tmp_path, "ok.sh", "exit 0")
    r = await hooks_from({"pre_tool": [{"command": deny}]}).run("pre_tool", "write_file", {}, str(tmp_path))
    assert r.blocked and "read-only" in r.output
    assert not (await hooks_from({"pre_tool": [{"command": ok}]}).run("pre_tool", "write_file", {}, str(tmp_path))).blocked
    silent = await hooks_from({"pre_tool": [{"command": "exit 9"}]}).run("pre_tool", "x", {}, str(tmp_path))
    assert silent.blocked and "exited 9" in silent.output                              # a reason is always given


async def test_a_hook_that_hangs_times_out_and_fails_closed(tmp_path: Path):
    import time

    t0 = time.monotonic()
    r = await hooks_from({"pre_tool": [{"command": "sleep 30", "timeout": 1}]}).run("pre_tool", "x", {}, str(tmp_path))
    assert r.blocked and "timed out" in r.output and time.monotonic() - t0 < 6


async def test_a_missing_command_blocks_rather_than_silently_skipping_the_guard(tmp_path: Path):
    r = await hooks_from({"pre_tool": [{"command": "/no/such/guard-binary"}]}).run("pre_tool", "x", {}, str(tmp_path))
    assert r.blocked


async def test_post_hooks_never_block_and_report_failures(tmp_path: Path):
    r = await hooks_from({"post_tool": [{"command": "echo formatted; exit 4"}]}).run("post_tool", "write_file", {}, str(tmp_path), result={"path": "a"})
    assert not r.blocked and "formatted" in r.output and "exited 4" in r.output


async def test_hooks_run_in_the_workspace_and_output_is_bounded(tmp_path: Path):
    r = await hooks_from({"post_tool": [{"command": "pwd; head -c 100000 /dev/zero | tr '\\0' 'x'"}]}).run("post_tool", "x", {}, str(tmp_path))
    assert r.output.startswith(str(tmp_path.resolve())) or r.output.startswith(str(tmp_path)) and len(r.output) <= 2000


# ── inside the agent loop ───────────────────────────────────────────────────

async def test_pre_hook_blocks_the_tool_and_the_model_sees_why(tmp_path: Path):
    guard = script(tmp_path, "guard.sh", 'echo "do not touch lockfiles" >&2; exit 1')
    agent = make_agent(Scripted([call("1", "write_file", path="package-lock.json", content="{}"), text("understood")]))
    agent.hooks = hooks_from({"pre_tool": [{"match": "write_file", "command": guard}]})
    resp, _, traces = await run(agent, workspace=str(tmp_path), agent_mode="build")
    assert resp == "understood" and not (tmp_path / "package-lock.json").exists()
    assert "blocked by a pre_tool hook: do not touch lockfiles" in tool_msgs(agent.provider.requests[1])[0]["content"]
    assert any(s == "hook_blocked" for s, _ in traces)


async def test_pre_hook_only_affects_matching_tools(tmp_path: Path):
    (tmp_path / "a.txt").write_text("hi")
    agent = make_agent(Scripted([call("1", "read_file", path="a.txt"), text("read it")]))
    agent.hooks = hooks_from({"pre_tool": [{"match": "write_file", "command": "exit 1"}]})
    resp, _, _ = await run(agent, workspace=str(tmp_path), agent_mode="build")
    assert resp == "read it" and '"content"' in tool_msgs(agent.provider.requests[1])[0]["content"]


async def test_post_hook_output_is_added_to_the_result_the_model_sees(tmp_path: Path):
    fmt = script(tmp_path, "fmt.sh", 'echo "formatted with ruff: 1 file changed"')
    agent = make_agent(Scripted([call("1", "write_file", path="a.py", content="x=1\n"), text("done")]))
    agent.hooks = hooks_from({"post_tool": [{"match": "write_file", "command": fmt}]})
    _, _, traces = await run(agent, workspace=str(tmp_path), agent_mode="build")
    assert "formatted with ruff" in tool_msgs(agent.provider.requests[1])[0]["content"]
    assert (tmp_path / "a.py").exists() and any(s == "hook_output" for s, _ in traces)


async def test_hooks_also_guard_subagent_tool_calls(tmp_path: Path):
    from tests.test_subagents import Router, task

    (tmp_path / "f.txt").write_text("x")
    p = Router(lead=[task("1", "alpha", "general"), text("done")],
               subs={"alpha": [call("s1", "write_file", path="o.txt", content="no"), text("blocked")]})
    agent = make_agent(p)
    agent.hooks = hooks_from({"pre_tool": [{"match": "write_file", "command": "echo nope; exit 1"}]})
    await run(agent, workspace=str(tmp_path), agent_mode="build")
    assert not (tmp_path / "o.txt").exists() and "blocked by a pre_tool hook" in tool_msgs(p.sub_requests[1])[0]["content"]


# ── edit_files ──────────────────────────────────────────────────────────────

def workspace_with(tmp_path: Path, **files) -> WorkspaceTools:
    for name, content in files.items():
        (tmp_path / name.replace("__", ".")).write_text(content)
    tools = WorkspaceTools(tmp_path)
    for name in files:
        tools.session.read_files.add((tmp_path / name.replace("__", ".")).resolve())
    return tools


def test_edit_files_changes_several_files_in_one_call(tmp_path: Path):
    t = workspace_with(tmp_path, a__py="import foo\nfoo.run()\n", b__py="from x import foo\n")
    r = t.execute("edit_files", {"edits": [
        {"path": "a.py", "old": "import foo", "new": "import bar"},
        {"path": "a.py", "old": "foo.run()", "new": "bar.run()"},
        {"path": "b.py", "old": "import foo", "new": "import bar"},
    ]})
    assert (tmp_path / "a.py").read_text() == "import bar\nbar.run()\n" and (tmp_path / "b.py").read_text() == "from x import bar\n"
    assert r["edits"] == 3 and [f["path"] for f in r["files"]] == ["a.py", "b.py"] and r["files"][0]["replacements"] == 2
    assert r["lines_added"] == 3 and "-import foo" in r["_diff"] and "+from x import bar" in r["_diff"]


@pytest.mark.parametrize("bad_edit,message", [
    ({"path": "b.py", "old": "NOPE", "new": "x"}, "old text was not found"),
    ({"path": "b.py", "old": "dup", "new": "x"}, "occurs 2 times"),
    ({"path": "missing.py", "old": "a", "new": "b"}, "file does not exist"),
    ({"path": "b.py", "old": "", "new": "x"}, "old must be a non-empty string"),
    ({"path": "b.py", "old": "dup", "new": 5}, "new must be a string"),
])
def test_one_bad_edit_writes_nothing_at_all(tmp_path: Path, bad_edit, message):
    t = workspace_with(tmp_path, a__py="alpha\n", b__py="dup dup\n")
    with pytest.raises(WorkspaceToolError, match=message) as exc:
        t.execute("edit_files", {"edits": [{"path": "a.py", "old": "alpha", "new": "ALPHA"}, bad_edit]})
    assert "edit 2" in str(exc.value) and "nothing was written" in str(exc.value)
    assert (tmp_path / "a.py").read_text() == "alpha\n" and (tmp_path / "b.py").read_text() == "dup dup\n"   # the good edit was NOT applied


def test_later_edits_see_earlier_ones_and_replace_all_counts(tmp_path: Path):
    t = workspace_with(tmp_path, a__py="one two two\n")
    r = t.execute("edit_files", {"edits": [
        {"path": "a.py", "old": "two", "new": "2", "replace_all": True},
        {"path": "a.py", "old": "one 2", "new": "1 2"},
    ]})
    assert (tmp_path / "a.py").read_text() == "1 2 2\n" and r["files"][0]["replacements"] == 3
    with pytest.raises(WorkspaceToolError, match="not found"):                       # the first edit already consumed "two"
        t.execute("edit_files", {"edits": [{"path": "a.py", "old": "2", "new": "x", "replace_all": True}, {"path": "a.py", "old": "2", "new": "y"}]})


def test_edit_files_input_validation_and_read_before_write(tmp_path: Path):
    t = workspace_with(tmp_path, a__py="x\n")
    for bad in (None, [], "x", [5]):
        with pytest.raises(WorkspaceToolError):
            t.execute("edit_files", {"edits": bad})
    with pytest.raises(WorkspaceToolError, match="too many edits"):
        t.execute("edit_files", {"edits": [{"path": "a.py", "old": "x", "new": "y"}] * 51})
    (tmp_path / "unread.py").write_text("keep\n")
    with pytest.raises(WorkspaceToolError, match="has not been read"):
        WorkspaceTools(tmp_path, enforce_read_before_write=True).execute(
            "edit_files", {"edits": [{"path": "unread.py", "old": "keep", "new": "lose"}]})
    assert (tmp_path / "unread.py").read_text() == "keep\n"


def test_edit_files_needs_build_mode_and_stays_inside_the_workspace(tmp_path: Path):
    t = workspace_with(tmp_path, a__py="x\n")
    with pytest.raises(WorkspaceToolError, match="plan mode"):
        WorkspaceTools(tmp_path, read_only=True).execute("edit_files", {"edits": [{"path": "a.py", "old": "x", "new": "y"}]})
    from core.workspace_tools import OutOfWorkspaceError
    with pytest.raises(OutOfWorkspaceError):
        t.execute("edit_files", {"edits": [{"path": "../outside.py", "old": "x", "new": "y"}]})
    assert "edit_files" not in {s["name"] for s in WorkspaceTools(tmp_path, read_only=True).tool_schemas()}
    assert "edit_files" in {s["name"] for s in t.tool_schemas()}


async def test_edit_files_in_the_loop_is_undoable_and_described(tmp_path: Path):
    from core.toolstate import ToolSession

    (tmp_path / "a.py").write_text("one\n")
    (tmp_path / "b.py").write_text("two\n")
    session = ToolSession()
    steps = [call("r1", "read_file", path="a.py"), call("r2", "read_file", path="b.py"),
             call("e", "edit_files", edits=[{"path": "a.py", "old": "one", "new": "1"}, {"path": "b.py", "old": "two", "new": "2"}]),
             text("done")]
    _, chunks, _ = await run(make_agent(Scripted(steps)), workspace=str(tmp_path), agent_mode="build", session=session)
    assert (tmp_path / "a.py").read_text() == "1\n" and (tmp_path / "b.py").read_text() == "2\n"
    assert any("edited 2 file(s)" in c for c in chunks)
    assert len(session.checkpoints.undo_last_turn()) == 2 and (tmp_path / "a.py").read_text() == "one\n" and (tmp_path / "b.py").read_text() == "two\n"
