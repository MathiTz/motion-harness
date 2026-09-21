"""Sandbox (OS-level write confinement) and Python-code policy tests."""
import asyncio
import os
import shutil
import uuid
from pathlib import Path
from unittest.mock import patch

import pytest

import core.sandbox as sbx
from core.permissions import CommandPolicy
from core.sandbox import Sandbox, bwrap_argv, seatbelt_profile
from core.workspace_tools import WorkspaceToolError, WorkspaceTools

REAL = Sandbox(".").active
needs_sandbox = pytest.mark.skipif(not REAL, reason="no working OS sandbox on this machine")


@pytest.fixture
def outside():
    """A directory OUTSIDE the workspace and outside every temp/cache dir
    (temp dirs are deliberately writable, so tmp_path can't serve); a sibling
    of the repo checkout, never the user's home directory."""
    d = Path(__file__).resolve().parents[2] / f".motion_sandbox_test_{uuid.uuid4().hex[:8]}"
    d.mkdir()
    (d / "victim.txt").write_text("keep me")
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


def tools_for(ws: Path, **kw) -> WorkspaceTools:
    return WorkspaceTools(ws, sandbox=Sandbox(ws, protected=kw.pop("protected", ())), **kw)


# ── construction (runs everywhere) ──────────────────────────────────────────

def test_seatbelt_profile_denies_writes_then_reallows_and_protects_secrets(tmp_path: Path):
    secret_dir, secret_file = tmp_path / "auth", tmp_path / ".env"
    secret_dir.mkdir()
    secret_file.write_text("K=1")
    profile = seatbelt_profile([tmp_path / 'we"ird'], [secret_dir, secret_file])
    lines = profile.splitlines()
    assert lines[:3] == ["(version 1)", "(allow default)", "(deny file-write*)"]
    assert '(subpath "%s")' % (str(tmp_path / 'we"ird').replace('"', '\\"')) in lines[3]  # quotes escaped
    assert lines.index(lines[3]) < len(lines) - 2  # allow comes after the blanket deny
    assert f'(deny file-read* (subpath "{secret_dir}"))' in lines
    assert f'(deny file-read* (literal "{secret_file}"))' in lines


def test_bwrap_argv_layout(tmp_path: Path):
    ws, missing, secret_dir = tmp_path / "ws", tmp_path / "nope", tmp_path / "s"
    ws.mkdir(); secret_dir.mkdir()
    argv = bwrap_argv([ws, missing], [secret_dir])
    assert argv[:4] == ["bwrap", "--ro-bind", "/", "/"] and argv[-2:] == ["--die-with-parent", "--"]
    assert ["--bind", str(ws), str(ws)] == argv[argv.index("--bind"):argv.index("--bind") + 3]
    assert str(missing) not in argv  # non-existent paths can't be bind-mounted
    assert ["--tmpfs", str(secret_dir)] == argv[argv.index("--tmpfs"):argv.index("--tmpfs") + 2]


def test_writable_set_includes_workspace_approved_paths_temp_and_caches(tmp_path: Path):
    approved = tmp_path / "approved"
    paths = {str(p) for p in Sandbox(tmp_path).writable_paths([approved])}
    assert str(tmp_path.resolve()) in paths and str(approved) in paths
    assert "/tmp" in paths or "/private/tmp" in paths
    assert str(Path.home() / ".cache") in paths and str(Path.home() / ".npm") in paths
    assert str(Path.home()) not in paths and str(Path.home() / "Documents") not in paths


def test_mode_off_disables_and_wrap_is_passthrough(tmp_path: Path):
    off = Sandbox(tmp_path, mode="off")
    assert not off.active and "disabled" in off.describe()
    assert off.wrap("echo hi", shell=True) == ["/bin/sh", "-c", "echo hi"]
    assert off.wrap(["python", "-c", "1"], shell=False) == ["python", "-c", "1"]


def test_backend_detection_probes_for_real_and_degrades_gracefully():
    class Done:
        def __init__(self, rc): self.returncode = rc

    def detect(platform, which, rc):
        sbx._detect_backend.cache_clear()
        try:
            with patch.object(sbx.sys, "platform", platform), \
                 patch.object(sbx.shutil, "which", return_value=which), \
                 patch.object(sbx.subprocess, "run", return_value=Done(rc)):
                return sbx._detect_backend()
        finally:
            sbx._detect_backend.cache_clear()

    assert detect("darwin", "/usr/bin/sandbox-exec", 0) == "seatbelt"
    assert detect("darwin", "/usr/bin/sandbox-exec", 71) is None   # nested sandbox: probe fails => off
    assert detect("linux", "/usr/bin/bwrap", 0) == "bwrap"
    assert detect("linux", "/usr/bin/bwrap", 1) is None            # userns disabled => off
    assert detect("linux", None, 0) is None                        # not installed
    assert detect("win32", None, 0) is None                        # no backend on Windows


def test_unavailable_sandbox_is_described_honestly(tmp_path: Path):
    with patch.object(sbx, "_detect_backend", return_value=None):
        s = Sandbox(tmp_path)
        assert not s.active and "NOT write-confined" in s.describe()
        assert s.wrap("x", shell=True) == ["/bin/sh", "-c", "x"]


def test_system_prompt_only_mentions_the_sandbox_when_it_is_active(tmp_path: Path):
    with patch.object(sbx, "_detect_backend", return_value="seatbelt"):
        assert "write sandbox" in WorkspaceTools(tmp_path, sandbox=Sandbox(tmp_path)).system_instructions(True)
        assert "write sandbox" not in WorkspaceTools(tmp_path, sandbox=Sandbox(tmp_path), read_only=True).system_instructions(True)
    with patch.object(sbx, "_detect_backend", return_value=None):
        assert "write sandbox" not in WorkspaceTools(tmp_path, sandbox=Sandbox(tmp_path)).system_instructions(True)
    assert "write sandbox" not in WorkspaceTools(tmp_path).system_instructions(True)


# ── real enforcement (needs sandbox-exec / bwrap) ───────────────────────────

@needs_sandbox
async def test_shell_cannot_write_or_delete_outside_the_workspace(tmp_path: Path, outside: Path):
    t = tools_for(tmp_path)
    w = await t.aexecute("run_command", {"command": f"echo pwned > {outside}/new.txt"})
    assert w["exit_code"] != 0 and not (outside / "new.txt").exists()
    assert "sandbox" in w["sandbox_note"].lower()
    d = await t.aexecute("run_command", {"command": f"rm {outside}/victim.txt"})
    assert d["exit_code"] != 0 and (outside / "victim.txt").read_text() == "keep me"


@needs_sandbox
async def test_python_cannot_bypass_the_sandbox(tmp_path: Path, outside: Path):
    """The exact hole found in review: run_python used to delete anything."""
    approve = lambda *a: "once"   # even with the user approving the code prompt...
    t = tools_for(tmp_path, approve=approve)
    r = await t.aexecute("run_python", {"code": f"import shutil; shutil.rmtree({str(outside)!r})"})
    assert r["exit_code"] != 0 and "PermissionError" in r["stderr"]
    assert (outside / "victim.txt").exists()                        # ...the OS still says no
    (tmp_path / "s.py").write_text(f"import pathlib; pathlib.Path({str(outside / 'victim.txt')!r}).write_text('x')")
    s = await t.aexecute("run_script", {"path": "s.py"})
    assert s["exit_code"] != 0 and (outside / "victim.txt").read_text() == "keep me"


@needs_sandbox
async def test_normal_development_still_works_inside_the_sandbox(tmp_path: Path):
    t = tools_for(tmp_path)
    ok = await t.aexecute("run_command", {"command": "mkdir -p src && echo hi > src/a.txt && cat src/a.txt"})
    assert ok["exit_code"] == 0 and ok["stdout"].strip() == "hi"
    tmp = await t.aexecute("run_command", {"command": f"echo t > /tmp/motion_sbx_{uuid.uuid4().hex} && echo ok"})
    assert tmp["stdout"].strip() == "ok"                         # temp is writable
    py = await t.aexecute("run_python", {"code": "import json, asyncio, sqlite3; open('o.json','w').write(json.dumps([1])); print('ran')"})
    assert py["stdout"].strip() == "ran" and (tmp_path / "o.json").exists()
    assert (await t.aexecute("run_command", {"command": "ls /etc | head -1; git --version"}))["exit_code"] == 0  # reads + tools fine


@needs_sandbox
async def test_user_approved_paths_become_writable(tmp_path: Path, outside: Path):
    t = tools_for(tmp_path)
    assert (await t.aexecute("run_command", {"command": f"touch {outside}/a"}))["exit_code"] != 0
    t.allowed_paths.add(outside.resolve())                        # what "allow for this session" does
    assert (await t.aexecute("run_command", {"command": f"touch {outside}/a"}))["exit_code"] == 0
    assert (outside / "a").exists()


@needs_sandbox
async def test_harness_secrets_are_unreadable_to_commands(tmp_path: Path):
    secrets = tmp_path.parent / f"sec_{uuid.uuid4().hex[:6]}"
    secrets.mkdir()
    (secrets / "auth.json").write_text('{"k": "sk-topsecret"}')
    env_file = tmp_path.parent / f"envfile_{uuid.uuid4().hex[:6]}"
    env_file.write_text("KEY=sk-topsecret")
    try:
        # Approve the policy prompt ("touches credentials") so the request reaches
        # the OS sandbox, which is the layer under test.
        t = tools_for(tmp_path, protected=[secrets, env_file], approve=lambda *a: "once")
        for cmd in (f"cat {secrets}/auth.json", f"cat {env_file}"):
            r = await t.aexecute("run_command", {"command": cmd})
            assert r["exit_code"] != 0 and "sk-topsecret" not in r["stdout"]
        r = await t.aexecute("run_python", {"code": f"print(open({str(secrets / 'auth.json')!r}).read())"})
        assert "sk-topsecret" not in r["stdout"] and r["exit_code"] != 0
    finally:
        shutil.rmtree(secrets, ignore_errors=True)
        env_file.unlink(missing_ok=True)


@needs_sandbox
async def test_sandboxed_command_is_still_killed_on_cancel(tmp_path: Path):
    pidfile = tmp_path / "pid"
    t = tools_for(tmp_path)
    task = asyncio.create_task(t.aexecute("run_command", {"command": f"echo $$ > {pidfile}; sleep 30"}))
    for _ in range(100):
        if pidfile.exists() and pidfile.read_text().strip():
            break
        await asyncio.sleep(0.05)
    pid = int(pidfile.read_text())
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0.3)
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


async def test_sandbox_off_lets_commands_write_anywhere(tmp_path: Path, outside: Path):
    t = WorkspaceTools(tmp_path, sandbox=Sandbox(tmp_path, mode="off"))
    r = await t.aexecute("run_command", {"command": f"echo x > {outside}/free.txt"})
    assert r["exit_code"] == 0 and (outside / "free.txt").exists()


# ── code policy (runs everywhere) ───────────────────────────────────────────

@pytest.mark.parametrize("code,decision", [
    ("print(6*7)", "allow"),
    ("import json; print(json.dumps({'a': 1}))", "allow"),
    ("open('out.txt','w').write('x')", "allow"),
    ("import shutil; shutil.rmtree('build')", "ask"),
    ("from shutil import rmtree\nrmtree('x')", "ask"),
    ("import os; os.remove('f')", "ask"),
    ("import os; os.unlink('f')", "ask"),
    ("from pathlib import Path; Path('f').unlink()", "ask"),
    ("import os; os.system('ls')", "ask"),
    ("import subprocess; subprocess.run(['ls'])", "ask"),
    ("import os; os.system('rm -rf /')", "deny"),
    ("import os; os.system('rm -rf ~')", "deny"),
    ("import subprocess; subprocess.run('mkfs.ext4 /dev/sda', shell=True)", "deny"),
])
def test_python_code_policy(code, decision):
    assert CommandPolicy().decide_code(code)[0] == decision


def test_code_approval_is_remembered_per_snippet():
    p = CommandPolicy()
    code = "import shutil; shutil.rmtree('build')"
    assert p.decide_code(code)[0] == "ask"
    p.approved.add(p.code_key(code))
    assert p.decide_code(code)[0] == "allow"
    assert p.decide_code(code + " # edited")[0] == "ask"      # a different snippet asks again


async def test_run_python_asks_before_deleting_and_refuses_without_a_ui(tmp_path: Path):
    (tmp_path / "victim").mkdir()
    code = "import shutil; shutil.rmtree('victim')"
    with pytest.raises(WorkspaceToolError, match="needs user approval"):
        await WorkspaceTools(tmp_path).aexecute("run_python", {"code": code})
    assert (tmp_path / "victim").exists()

    asked = []

    def approve(kind, subject, reason):
        asked.append(subject)
        return "once"

    await WorkspaceTools(tmp_path, approve=approve).aexecute("run_python", {"code": code})
    assert not (tmp_path / "victim").exists() and "python snippet" in asked[0]

    (tmp_path / "victim").mkdir()
    with pytest.raises(WorkspaceToolError, match="denied"):
        await WorkspaceTools(tmp_path, approve=lambda *a: "deny").aexecute("run_python", {"code": code})


async def test_run_script_content_is_scanned_too(tmp_path: Path):
    (tmp_path / "danger.py").write_text("import shutil\nshutil.rmtree('victim')\n")
    (tmp_path / "victim").mkdir()
    with pytest.raises(WorkspaceToolError, match="needs user approval"):
        await WorkspaceTools(tmp_path).aexecute("run_script", {"path": "danger.py"})
    assert (tmp_path / "victim").exists()
    (tmp_path / "safe.py").write_text("print('fine')\n")
    assert "fine" in (await WorkspaceTools(tmp_path).aexecute("run_script", {"path": "safe.py"}))["stdout"]


def _forbid_execution(monkeypatch):
    """Tests of *refusal* must never be able to run the dangerous thing if the
    refusal regresses: replace the process runner with one that fails loudly.
    (An earlier version of this test executed `rm -rf ~` for real.)"""
    async def boom(self, *a, **k):
        raise AssertionError("a refused command reached the process runner")

    monkeypatch.setattr(WorkspaceTools, "_arun", boom)


@pytest.mark.parametrize("code", [
    "import os; os.system('rm -rf ~')",
    "import os; os.system(\"rm -rf $HOME\")",
    "import os; os.system('rm -rf \"$HOME\"')",
    "import os; os.system('rm -rf /')",
    "import os; os.system('rm -rf /*')",
    "import os; os.system('rm -rf ~/')",
    "import subprocess; subprocess.run('rm -rf ~/*', shell=True)",
])
async def test_catastrophic_code_is_refused_even_if_the_user_would_approve(tmp_path: Path, monkeypatch, code):
    _forbid_execution(monkeypatch)
    t = WorkspaceTools(tmp_path, approve=lambda *a: "session")
    with pytest.raises(WorkspaceToolError, match="code refused"):
        await t.aexecute("run_python", {"code": code})


@pytest.mark.parametrize("cmd", [
    "rm -rf ~", "rm -rf ~/", "rm -rf ~/*", "rm -rf $HOME", 'rm -rf "$HOME"', "rm -rf ${HOME}", "rm -rf /", "rm -rf /*",
    "rm -fr /", "rm -r -f /", "sudo rm -rf /", "echo hi; rm -rf ~; echo done", "cd /tmp && rm -rf /",
])
async def test_catastrophic_shell_commands_are_refused_without_running(tmp_path: Path, monkeypatch, cmd):
    _forbid_execution(monkeypatch)
    t = WorkspaceTools(tmp_path, approve=lambda *a: "session")
    with pytest.raises(WorkspaceToolError, match="refused"):
        await t.aexecute("run_command", {"command": cmd})


# ── credential stores and network ───────────────────────────────────────────

def test_default_protected_paths_cover_credential_stores_but_not_ssh():
    from core.sandbox import CREDENTIAL_PATHS, default_protected_paths

    names = {str(p) for p in default_protected_paths()}
    home = str(Path.home())
    for rel in CREDENTIAL_PATHS:
        assert f"{home}/{rel}" in names
    assert f"{home}/.aws" in names and f"{home}/.kube" in names and f"{home}/.config/gh" in names
    assert f"{home}/.ssh" not in names                                     # git push over ssh must keep working


def test_allow_and_deny_lists_adjust_what_is_protected(tmp_path: Path):
    from core.sandbox import default_protected_paths, sandbox_settings

    home = Path.home()
    allowed = {str(p) for p in default_protected_paths(allow_read=["~/.aws"])}
    assert f"{home}/.aws" not in allowed and f"{home}/.kube" in allowed       # opt back in, only that one
    denied = {str(p) for p in default_protected_paths(deny_read=["~/.ssh", str(tmp_path / "x")])}
    assert f"{home}/.ssh" in denied and str(tmp_path / "x") in denied
    # the harness's own secrets can never be allowed
    from core import auth
    assert str(auth.AUTH_DIR) in {str(p) for p in default_protected_paths(allow_read=[str(auth.AUTH_DIR)])}
    cfg = {"sandbox_allow_read": "~/.aws", "sandbox_network": "deny", "sandbox_deny_read": ["~/.ssh"]}
    s = sandbox_settings(cfg.get)
    assert s == {"mode": "auto", "allow_read": ["~/.aws"], "deny_read": ["~/.ssh"], "network": False}
    assert sandbox_settings({}.get)["network"] is True


def test_network_block_appears_in_both_backends_and_in_the_description(tmp_path: Path):
    assert "(deny network*)" in seatbelt_profile([tmp_path], [], network=False)
    assert "(deny network*)" not in seatbelt_profile([tmp_path], [])
    assert "--unshare-net" in bwrap_argv([tmp_path], [], network=False)
    assert "--unshare-net" not in bwrap_argv([tmp_path], [])
    with patch.object(sbx, "_detect_backend", return_value="seatbelt"):
        assert "network blocked" in Sandbox(tmp_path, network=False).describe()
        assert "network blocked" not in Sandbox(tmp_path).describe()
        assert "credential folders unreadable" in Sandbox(tmp_path).describe()


def test_blocked_hint_explains_the_credential_and_network_cases():
    assert "sandbox_allow_read" in sbx.BLOCKED_HINT and "sandbox_network" in sbx.BLOCKED_HINT


@needs_sandbox
async def test_a_credentials_folder_cannot_be_read_by_shell_or_python(tmp_path: Path, outside: Path):
    creds = outside / ".aws"
    creds.mkdir()
    (creds / "credentials").write_text("aws_secret_access_key=TOPSECRET")
    t = WorkspaceTools(tmp_path, sandbox=Sandbox(tmp_path, protected=[creds]), approve=lambda *a: "once")
    r = await t.aexecute("run_command", {"command": f"cat {creds}/credentials"})
    assert r["exit_code"] != 0 and "TOPSECRET" not in r["stdout"] + r["stderr"]
    p = await t.aexecute("run_python", {"code": f"print(open({str(creds / 'credentials')!r}).read())"})
    assert "TOPSECRET" not in p["stdout"] and p["exit_code"] != 0
    ok = await t.aexecute("run_command", {"command": f"cat {outside}/victim.txt"})     # everything else stays readable
    assert ok["stdout"].strip() == "keep me"


@needs_sandbox
async def test_network_can_be_blocked_for_commands(tmp_path: Path):
    import socket
    import threading

    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(4)
    port = server.getsockname()[1]
    threading.Thread(target=lambda: [server.accept() for _ in range(3)], daemon=True).start()
    code = f"import socket; s=socket.create_connection(('127.0.0.1', {port}), timeout=3); print('CONNECTED')"
    try:
        open_net = WorkspaceTools(tmp_path, sandbox=Sandbox(tmp_path), approve=lambda *a: "once")
        assert "CONNECTED" in (await open_net.aexecute("run_python", {"code": code}))["stdout"]
        closed = WorkspaceTools(tmp_path, sandbox=Sandbox(tmp_path, network=False), approve=lambda *a: "once")
        r = await closed.aexecute("run_python", {"code": code})
        assert "CONNECTED" not in r["stdout"] and r["exit_code"] != 0
    finally:
        server.close()
