"""Regression coverage for core/command_watchdog.py (issue #14): a run_command child used to
survive the harness being killed uncatchably (kill -9, OOM, a crash) as an orphan, reparented to
init, still running - proven with a real `kill -9` reproduction before the fix existed. Every
run_command/run_script/run_python child is now wrapped in a small supervisor (see
core/workspace_tools.py's `_watchdog_enabled`/`_WATCHDOG_SCRIPT`) that kills the whole process tree
if the harness process disappears. These tests cover: normal execution is unaffected, the existing
timeout/cancellation kill paths still reach the real command through the extra wrapper layer, the
MOTION_DISABLE_COMMAND_WATCHDOG escape hatch and non-POSIX fallback skip the wrapper, sandboxed
execution still composes with it, and - the key new regression test - a real external SIGKILL of
the harness process still takes the real command down with it.
"""
import asyncio
import os
import signal
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import core.workspace_tools as workspace_tools
from core.sandbox import Sandbox
from core.workspace_tools import WorkspaceToolError, WorkspaceTools

needs_posix = pytest.mark.skipif(os.name != "posix", reason="watchdog is POSIX-only")
REAL_SANDBOX = Sandbox(".").active
REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def spy_exec(monkeypatch):
    """Records the argv of every asyncio.create_subprocess_exec call while still running it for
    real, so a test can assert on how a command was wrapped without losing the actual subprocess
    behavior other assertions in the same test need."""
    calls: list[list[str]] = []
    real = asyncio.create_subprocess_exec

    async def spy(*argv, **kwargs):
        calls.append([str(a) for a in argv])
        return await real(*argv, **kwargs)

    monkeypatch.setattr(workspace_tools.asyncio, "create_subprocess_exec", spy)
    return calls


@needs_posix
async def test_normal_command_runs_correctly_through_the_watchdog_wrapper(tmp_path: Path, spy_exec):
    result = await WorkspaceTools(tmp_path).aexecute("run_command", {"command": "echo hi"})
    assert (result["exit_code"], result["stdout"].strip()) == (0, "hi")
    assert any(str(workspace_tools._WATCHDOG_SCRIPT) in call for call in spy_exec)


@needs_posix
async def test_timeout_kills_the_real_command_not_just_the_watchdog(tmp_path: Path):
    pidfile = tmp_path / "pid"
    with pytest.raises(WorkspaceToolError, match="timed out"):
        await WorkspaceTools(tmp_path).aexecute(
            "run_command", {"command": f"echo $$ > {pidfile}; sleep 20", "timeout": 1}
        )
    pid = int(pidfile.read_text().strip())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


@needs_posix
async def test_task_cancellation_still_kills_the_real_command(tmp_path: Path):
    pidfile = tmp_path / "pid"
    tools = WorkspaceTools(tmp_path)
    task = asyncio.create_task(
        tools.aexecute("run_command", {"command": f"echo $$ > {pidfile}; sleep 30", "timeout": 60})
    )
    for _ in range(100):
        if pidfile.exists() and pidfile.read_text().strip():
            break
        await asyncio.sleep(0.05)
    pid = int(pidfile.read_text().strip())
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0.2)
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


@needs_posix
async def test_disable_env_var_skips_the_watchdog_wrapper(tmp_path: Path, monkeypatch, spy_exec):
    monkeypatch.setenv("MOTION_DISABLE_COMMAND_WATCHDOG", "1")
    result = await WorkspaceTools(tmp_path).aexecute("run_command", {"command": "echo hi"})
    assert result["stdout"].strip() == "hi"
    # still runs directly via create_subprocess_exec (/bin/sh -c ...) - just without the watchdog
    # script prepended to argv.
    assert spy_exec == [["/bin/sh", "-c", "echo hi"]]


def test_watchdog_disabled_helper_reports_false_on_non_posix(monkeypatch):
    """`_watchdog_enabled()` is the single gate `_arun` uses to decide whether to wrap a command;
    on a non-POSIX OS (Windows) it must report False so `_arun` falls back to its pre-existing,
    unwrapped `create_subprocess_shell`/`create_subprocess_exec` calls. Tested at this level rather
    than by faking `os.name` through a full WorkspaceTools call: `os.name` is also read by pathlib
    itself (it picks WindowsPath vs PosixPath from it), so faking it any more broadly breaks
    unrelated Path construction on a real POSIX machine instead of exercising the fallback.
    """
    monkeypatch.setattr(workspace_tools.os, "name", "nt")
    assert workspace_tools._watchdog_enabled() is False


def test_watchdog_disabled_helper_reports_false_via_env_var(monkeypatch):
    monkeypatch.setenv("MOTION_DISABLE_COMMAND_WATCHDOG", "1")
    assert workspace_tools._watchdog_enabled() is False


@needs_posix
@pytest.mark.skipif(not REAL_SANDBOX, reason="no working OS sandbox on this machine")
async def test_sandboxed_execution_still_composes_with_the_watchdog(tmp_path: Path, spy_exec):
    tools = WorkspaceTools(tmp_path, sandbox=Sandbox(tmp_path))
    result = await tools.aexecute("run_command", {"command": "echo hi"})
    assert (result["exit_code"], result["stdout"].strip()) == (0, "hi")
    assert any(str(workspace_tools._WATCHDOG_SCRIPT) in call for call in spy_exec)


@needs_posix
async def test_real_external_kill_of_the_harness_still_kills_the_command(tmp_path: Path):
    """The key regression test: reproduces the original bug for real. Runs a run_command call in
    its own OS process (standing in for 'the harness'), SIGKILLs that process from here the way an
    OOM killer or `kill -9` would, and asserts the real command it spawned dies with it instead of
    surviving as an orphan - which is exactly what happened before core/command_watchdog.py existed.
    """
    pidfile = tmp_path / "pid"
    driver = tmp_path / "driver.py"
    driver.write_text(
        textwrap.dedent(
            f"""
            import asyncio
            import sys
            sys.path.insert(0, {str(REPO_ROOT)!r})
            import core.workspace_tools as wt
            wt._WATCHDOG_POLL_SECONDS = 0.2  # keep this test fast
            asyncio.run(wt.WorkspaceTools({str(tmp_path)!r}).aexecute(
                "run_command", {{"command": "echo $$ > {pidfile}; sleep 30", "timeout": 60}}
            ))
            """
        )
    )
    harness = subprocess.Popen([sys.executable, str(driver)])
    try:
        for _ in range(100):
            if pidfile.exists() and pidfile.read_text().strip():
                break
            await asyncio.sleep(0.05)
        else:
            pytest.fail("driver process never started the real command")
        real_cmd_pid = int(pidfile.read_text().strip())

        os.kill(harness.pid, signal.SIGKILL)  # simulate the harness itself being killed uncatchably
        harness.wait(timeout=5)

        for _ in range(50):  # poll interval is 0.2s; give it generous margin
            try:
                os.kill(real_cmd_pid, 0)
            except ProcessLookupError:
                break
            await asyncio.sleep(0.1)
        else:
            pytest.fail("real command survived the harness process being killed")
    finally:
        if harness.poll() is None:
            harness.kill()
        try:
            os.kill(real_cmd_pid, signal.SIGKILL)
        except (NameError, ProcessLookupError):
            pass
