"""Background jobs: start / read / wait / stop, safety, and the agent+TUI surface."""
import asyncio
import os
import sys
import time
from pathlib import Path

import pytest

from core.jobs import MAX_RUNNING_JOBS, JobManager
from core.toolstate import ToolSession
from core.workspace_tools import WorkspaceToolError, WorkspaceTools
from tests.test_agent_loop import Scripted, call, calls, make_agent, run, text, tool_msgs
from tests.test_tui_flow import send, system_lines, tui_app, wait_idle
from core.sandbox import Sandbox

PY = sys.executable


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def tools_for(ws: Path, **kw):
    session = kw.pop("session", None) or ToolSession()
    return WorkspaceTools(ws, session=session, **kw), session


async def start(t, command, **kw):
    return await t.aexecute("job_start", {"command": command, **kw})


# ── manager / tools ─────────────────────────────────────────────────────────

async def test_job_returns_immediately_and_output_is_incremental(tmp_path: Path):
    t, s = tools_for(tmp_path)
    try:
        t0 = time.monotonic()
        job = await start(t, f'{PY} -u -c "import time\nfor i in range(3): print(f\'tick {{i}}\'); time.sleep(0.4)\nprint(\'bye\')"', name="ticker")
        assert time.monotonic() - t0 < 1.0 and job["status"] == "running" and job["name"] == "ticker"
        chunks, status = [], "running"
        for _ in range(40):                                    # poll like an agent would
            out = await t.aexecute("job_output", {"job_id": job["job_id"], "wait_seconds": 5})
            if out["output"]:
                chunks.append(out["output"])
            status = out["status"]
            if status != "running":
                break
        assert status == "exited(0)"
        assert len(chunks) >= 2                                # it did not wait for the whole run
        # every line arrives exactly once, in order: nothing repeated, nothing lost
        assert "\n".join(chunks).splitlines() == ["tick 0", "tick 1", "tick 2", "bye"]
        everything = await t.aexecute("job_output", {"job_id": job["job_id"], "all": True})
        assert everything["output"].splitlines() == ["tick 0", "tick 1", "tick 2", "bye"] and everything["lines_skipped"] == 0
    finally:
        await s.jobs.stop_all()


async def test_stderr_is_merged_and_exit_codes_are_reported(tmp_path: Path):
    t, s = tools_for(tmp_path)
    try:
        job = await start(t, "echo out; echo err 1>&2; exit 3")
        out = await t.aexecute("job_output", {"job_id": job["job_id"], "wait_seconds": 5, "all": True})
        for _ in range(20):
            if out["status"] != "running":
                break
            await asyncio.sleep(0.1)
            out = await t.aexecute("job_output", {"job_id": job["job_id"], "all": True})
        assert out["status"] == "exited(3)" and {"out", "err"} <= set(out["output"].split())
    finally:
        await s.jobs.stop_all()


async def test_wait_returns_early_when_the_job_exits_and_times_out_when_quiet(tmp_path: Path):
    t, s = tools_for(tmp_path)
    try:
        quick = await start(t, "sleep 0.3")
        t0 = time.monotonic()
        r = await t.aexecute("job_output", {"job_id": quick["job_id"], "wait_seconds": 20})
        assert time.monotonic() - t0 < 5 and r["status"].startswith("exited")
        quiet = await start(t, "sleep 30")
        t0 = time.monotonic()
        r = await t.aexecute("job_output", {"job_id": quiet["job_id"], "wait_seconds": 0.5})
        assert 0.4 < time.monotonic() - t0 < 3 and r["status"] == "running" and r["output"] == ""
    finally:
        await s.jobs.stop_all()


async def test_stop_kills_the_whole_process_tree(tmp_path: Path):
    pidfile = tmp_path / "child.pid"
    t, s = tools_for(tmp_path)
    job = await start(t, f"sleep 60 & echo $! > {pidfile}; wait")     # a background CHILD of the job shell
    for _ in range(100):
        if pidfile.exists() and pidfile.read_text().strip():
            break
        await asyncio.sleep(0.05)
    child, shell = int(pidfile.read_text()), job["pid"]
    assert alive(child) and alive(shell)
    info = await t.aexecute("job_stop", {"job_id": job["job_id"]})
    await asyncio.sleep(0.2)
    assert info["was_running"] and not alive(child) and not alive(shell)
    again = await t.aexecute("job_stop", {"job_id": job["job_id"]})
    assert again["was_running"] is False                                # stopping twice is harmless
    await s.jobs.stop_all()


async def test_stubborn_processes_are_force_killed(tmp_path: Path):
    m = JobManager()
    try:
        job = await m.start(["/bin/sh", "-c", "trap '' TERM; while true; do sleep 1; done"], command="stubborn",
                            cwd=str(tmp_path), env=dict(os.environ))
        await asyncio.sleep(0.3)
        t0 = time.monotonic()
        info = await m.stop(job.id, grace=0.5)
        assert time.monotonic() - t0 < 5 and not alive(job.proc.pid) and info["was_running"]
    finally:
        await m.stop_all()


async def test_running_job_limit_and_unknown_ids(tmp_path: Path):
    t, s = tools_for(tmp_path)
    try:
        for _ in range(MAX_RUNNING_JOBS):
            await start(t, "sleep 30")
        with pytest.raises(WorkspaceToolError, match="too many running jobs"):
            await start(t, "sleep 30")
        with pytest.raises(WorkspaceToolError, match="unknown job"):
            await t.aexecute("job_output", {"job_id": "job99"})
        listing = (await t.aexecute("job_list", {}))["jobs"]
        assert len(listing) == MAX_RUNNING_JOBS and all(j["status"] == "running" for j in listing)
    finally:
        assert await s.jobs.stop_all() == MAX_RUNNING_JOBS


async def test_output_buffer_is_bounded_and_reports_skipped_lines(tmp_path: Path):
    t, s = tools_for(tmp_path)
    try:
        job = await start(t, f'{PY} -c "for i in range(5000): print(i)"')
        for _ in range(100):
            out = await t.aexecute("job_output", {"job_id": job["job_id"], "all": True, "lines": 5})
            if out["status"] != "running":
                break
            await asyncio.sleep(0.1)
        assert out["output"].splitlines() == ["4995", "4996", "4997", "4998", "4999"]
        fresh = s.jobs.get(job["job_id"])
        assert len(fresh.lines) == 2000 and fresh.total_lines == 5000   # ring buffer, nothing unbounded
        fresh.read_cursor = 0
        missed = await t.aexecute("job_output", {"job_id": job["job_id"], "lines": 100})
        assert missed["lines_skipped"] == 5000 - 100                     # says how much it could not show
    finally:
        await s.jobs.stop_all()


# ── safety ──────────────────────────────────────────────────────────────────

async def test_jobs_go_through_the_same_command_policy(tmp_path: Path):
    t, s = tools_for(tmp_path)
    (tmp_path / "d").mkdir()
    with pytest.raises(WorkspaceToolError, match="needs user approval"):
        await start(t, "rm -rf d")
    assert (tmp_path / "d").exists()
    with pytest.raises(WorkspaceToolError, match="refused"):
        await start(t, "rm -rf ~")           # refused before anything can run
    approved, s2 = tools_for(tmp_path, approve=lambda *a: "once")
    try:
        job = await start(approved, "rm -rf d")
        await approved.aexecute("job_output", {"job_id": job["job_id"], "wait_seconds": 3})
        assert not (tmp_path / "d").exists()
    finally:
        await s2.jobs.stop_all()


async def test_plan_mode_can_read_jobs_but_not_start_or_stop_them(tmp_path: Path):
    t, s = tools_for(tmp_path, read_only=True)
    assert (await t.aexecute("job_list", {}))["jobs"] == []
    for name, args in (("job_start", {"command": "sleep 1"}), ("job_stop", {"job_id": "job1"})):
        with pytest.raises(WorkspaceToolError, match="plan mode"):
            await t.aexecute(name, args)
    names = {x["name"] for x in t.tool_schemas()}
    assert {"job_output", "job_list"} <= names and not ({"job_start", "job_stop"} & names)


@pytest.mark.skipif(not Sandbox(".").active, reason="no OS sandbox here")
async def test_jobs_run_inside_the_write_sandbox(tmp_path: Path):
    outside = Path(__file__).resolve().parents[2] / f".motion_job_test_{os.getpid()}"
    outside.mkdir()
    try:
        t, s = tools_for(tmp_path, sandbox=Sandbox(tmp_path))
        job = await start(t, f"echo pwned > {outside}/x; echo rc=$?")
        out = await t.aexecute("job_output", {"job_id": job["job_id"], "wait_seconds": 10})   # waits for the first line
        assert not (outside / "x").exists() and "rc=1" in out["output"]
        await s.jobs.stop_all()
    finally:
        import shutil
        shutil.rmtree(outside, ignore_errors=True)


# ── agent + TUI ─────────────────────────────────────────────────────────────

async def test_agent_can_start_a_server_use_it_and_stop_it_across_steps(tmp_path: Path):
    (tmp_path / "index.html").write_text("hello from the server")
    server = f"{PY} -u -m http.server 8765 --bind 127.0.0.1"
    fetch = f"{PY} -c \"import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:8765/index.html').read().decode())\""
    session = ToolSession()
    p = Scripted([
        call("1", "job_start", command=server, name="web"),
        call("2", "job_output", job_id="job1", wait_seconds=10),
        call("3", "run_command", command=fetch),
        call("4", "job_stop", job_id="job1"),
        text("served and stopped"),
    ])
    try:
        resp, _, _ = await run(make_agent(p), workspace=str(tmp_path), agent_mode="build", session=session)
        assert resp == "served and stopped"
        assert "Serving HTTP" in tool_msgs(p.requests[2])[-1]["content"]
        assert "hello from the server" in tool_msgs(p.requests[3])[-1]["content"]
        assert session.jobs.get("job1").status != "running"
    finally:
        await session.jobs.stop_all()


async def test_jobs_outlive_a_turn_and_show_in_the_tui(tmp_path, monkeypatch):
    steps = [call("1", "job_start", command="sleep 60", name="sleeper"), text("started it")]
    async with tui_app(tmp_path, monkeypatch, steps) as (app, pilot):
        jobs = app.state.tool_session.jobs
        try:
            await send(app, pilot, "start a sleeper")
            await wait_idle(app, pilot)
            assert len(jobs.running()) == 1                                   # still alive after the turn ended
            from tests.test_tui_flow import _text_of
            assert "1 job" in _text_of(app.screen.query_one("#chat_status_text"))
            await send(app, pilot, "/jobs")
            await pilot.pause(0.1)
            assert any("job1" in t and "running" in t and "sleep 60" in t for t in system_lines(app))
            pid = jobs.get("job1").proc.pid
            await send(app, pilot, "/jobs stop job1")
            await pilot.pause(0.5)
            assert not jobs.running() and not alive(pid)
            await send(app, pilot, "/jobs stop nope")
            await pilot.pause(0.1)
            assert any("unknown job" in t for t in system_lines(app))
        finally:
            await jobs.stop_all()


async def test_quitting_the_app_stops_background_jobs(tmp_path, monkeypatch):
    steps = [call("1", "job_start", command="sleep 60"), text("ok")]
    pid = None
    async with tui_app(tmp_path, monkeypatch, steps) as (app, pilot):
        await send(app, pilot, "go")
        await wait_idle(app, pilot)
        pid = app.state.tool_session.jobs.get("job1").proc.pid
        assert alive(pid)
    await asyncio.sleep(0.5)                                                  # app exited -> on_unmount ran
    assert not alive(pid)
