"""Headless (`motion -p`) tests: in-process with scripted providers, and real
subprocess runs against a local OpenAI-compatible fake server."""
import io
import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from core.headless import EXIT_FAILED, EXIT_OK, HeadlessUsageError, run_headless
from core.providers import StreamEvent
from tests.test_agent_loop import EmptyRetriever, Scripted, call, make_agent, text

REPO = Path(__file__).resolve().parents[1]


# ── in-process ──────────────────────────────────────────────────────────────

def factory(steps):
    def make():
        agent = make_agent(Scripted(steps))
        agent.provider.config.provider_type = "cloud"
        agent.provider.config.options.update(input_mtok=1.0, output_mtok=2.0)
        return agent
    return make


async def go(prompt, steps, **kw):
    out, err = io.StringIO(), io.StringIO()
    code = await run_headless(prompt, out=out, err=err, agent_factory=factory(steps), **kw)
    return code, out.getvalue(), err.getvalue()


async def test_text_mode_prints_only_the_answer(tmp_path):
    code, out, err = await go("hi", [text("Hello ", "world")], workspace=str(tmp_path))
    assert (code, out, err) == (EXIT_OK, "Hello world\n", "")


async def test_verbose_shows_tool_activity_on_stderr_not_stdout(tmp_path):
    (tmp_path / "f.txt").write_text("x")
    code, out, err = await go("read", [call("1", "read_file", path="f.txt"), text("done")], workspace=str(tmp_path), verbose=True)
    assert out == "done\n" and "read `f.txt`" in err


async def test_json_mode_reports_usage_cost_and_stats(tmp_path):
    (tmp_path / "f.txt").write_text("x")
    usage = [StreamEvent("usage", usage={"prompt_tokens": 1000, "completion_tokens": 500, "total_tokens": 1500})]
    code, out, _ = await go("read", [call("1", "read_file", path="f.txt") + usage, text("all done") + usage],
                            workspace=str(tmp_path), output_format="json")
    data = json.loads(out)
    assert code == EXIT_OK and data["ok"] is True and data["result"] == "all done" and data["error"] is None
    assert data["tool_calls"] == 1 and data["steps"] == 2 and data["mode"] == "build"
    assert data["usage"] == {"prompt_tokens": 2000, "completion_tokens": 1000, "total_tokens": 3000}
    assert data["cost_usd"] == pytest.approx((2000 * 1.0 + 1000 * 2.0) / 1e6)
    assert data["elapsed_s"] is not None and data["workspace"] == str(tmp_path)


async def test_stream_json_emits_events_then_the_result(tmp_path):
    (tmp_path / "f.txt").write_text("x")
    steps = [text("Looking") + call("1", "read_file", path="f.txt"), text("Fin", "al")]
    code, out, _ = await go("go", steps, workspace=str(tmp_path), output_format="stream-json")
    events = [json.loads(l) for l in out.splitlines()]
    kinds = [e["type"] for e in events]
    assert kinds[-1] == "result" and events[-1]["result"] == "Final"
    assert "text" in kinds and "tool" in kinds and "step_end" in kinds
    assert kinds.index("step_end") < kinds.index("tool")           # narration ends before the tool runs
    assert "".join(e["text"] for e in events if e["type"] == "text") == "LookingFinal"


async def test_plan_flag_makes_the_run_read_only(tmp_path):
    steps = [call("1", "write_file", path="x.txt", content="no"), text("Plan: ...")]
    code, out, _ = await go("build it", steps, workspace=str(tmp_path), plan=True, output_format="json")
    assert json.loads(out)["mode"] == "plan" and not (tmp_path / "x.txt").exists()


async def test_risky_commands_are_refused_because_nothing_can_approve_them(tmp_path):
    (tmp_path / "d").mkdir()
    agent_steps = [call("1", "run_command", command="rm -rf d"), text("could not")]
    code, out, _ = await go("clean", agent_steps, workspace=str(tmp_path))
    assert code == EXIT_OK and (tmp_path / "d").exists()


async def test_provider_failure_gives_exit_1_and_an_error(tmp_path):
    import httpx

    code, out, err = await go("hi", [httpx.ReadTimeout("slow")], workspace=str(tmp_path))
    assert code == EXIT_FAILED and out == "" and "timed out" in err
    code, out, _ = await go("hi", [httpx.ReadTimeout("slow")], workspace=str(tmp_path), output_format="json")
    data = json.loads(out)
    assert code == EXIT_FAILED and data["ok"] is False and "timed out" in data["error"]


async def test_unexpected_exception_is_reported_not_raised(tmp_path):
    class Boom(Scripted):
        async def chat_stream(self, *a, **k):
            raise RuntimeError("kaboom")
            yield  # pragma: no cover

    def make():
        agent = make_agent(Boom([]))
        return agent

    out, err = io.StringIO(), io.StringIO()
    code = await run_headless("hi", out=out, err=err, agent_factory=make, workspace=str(tmp_path), output_format="json")
    assert code == EXIT_FAILED and "RuntimeError: kaboom" in json.loads(out.getvalue())["error"]


@pytest.mark.parametrize("kwargs,match", [
    ({"prompt": "   "}, "empty prompt"),
    ({"prompt": "x", "output_format": "yaml"}, "output-format"),
    ({"prompt": "x", "workspace": "/definitely/not/a/dir"}, "not a directory"),
])
async def test_usage_errors_raise_before_any_model_call(kwargs, match):
    prompt = kwargs.pop("prompt")
    with pytest.raises(HeadlessUsageError, match=match):
        await run_headless(prompt, agent_factory=lambda: pytest.fail("agent must not be built"), **kwargs)


async def test_headless_runs_do_not_write_long_term_memory_by_default(tmp_path):
    agent_holder = {}

    def make():
        agent = factory([call("1", "list_files"), text("There are no files in this workspace at all.")])()
        agent_holder["a"] = agent
        return agent

    out = io.StringIO()
    await run_headless("please list the files here", out=out, err=io.StringIO(), agent_factory=make, workspace=str(tmp_path))
    assert agent_holder["a"].auto_remember is False


# ── real subprocess against a fake OpenAI-compatible server ─────────────────

class FakeLLM:
    """Serves /v1/chat/completions as SSE. Behaviour is chosen per request by `script`."""

    def __init__(self, script):
        self.script, self.requests = script, []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["content-length"])))
                outer.requests.append(body)
                status, chunks = outer.script(body, len(outer.requests))
                self.send_response(status)
                self.send_header("content-type", "text/event-stream" if status == 200 else "application/json")
                self.end_headers()
                if status != 200:
                    self.wfile.write(b'{"error": "boom"}')
                    return
                for c in chunks:
                    self.wfile.write(f"data: {json.dumps(c)}\n\n".encode())
                self.wfile.write(b"data: [DONE]\n\n")

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()


def delta(**d):
    return {"choices": [{"delta": d}]}


USAGE = {"choices": [], "usage": {"prompt_tokens": 1000, "completion_tokens": 500}}


def read_then_answer(body, n):
    if body["messages"][-1]["role"] == "tool":
        assert "PELICAN-42" in body["messages"][-1]["content"]      # the tool really ran in the workspace
        return 200, [delta(content="The codeword is "), delta(content="PELICAN-42."), USAGE]
    call_ = {"index": 0, "id": "c1", "function": {"name": "read_file", "arguments": json.dumps({"path": "notes.txt"})}}
    return 200, [delta(tool_calls=[call_]), USAGE]


@pytest.fixture
def cli(tmp_path):
    (tmp_path / "ws").mkdir()
    (tmp_path / "ws" / "notes.txt").write_text("The launch codeword is PELICAN-42.\n")
    servers = []

    def run(script, *args, stdin=None, max_retries=0, extra_env=None):
        fake = FakeLLM(script)
        servers.append(fake)
        cfg = tmp_path / "config.yml"
        cfg.write_text(
            "providers:\n  default: fake\n  fake:\n    name: Fake\n"
            f"    endpoint: {fake.url}\n    api_key: k\n    provider_type: proxy\n    default_model: m\n"
            f"    models:\n      m: {{temperature: 0.1, input_mtok: 1.0, output_mtok: 2.0, max_retries: {max_retries}}}\n"
            "track_interactions: false\n"
        )
        env = {**os.environ, "MOTION_CONFIG": str(cfg), "HOME": str(tmp_path / "home"), "PYTHONPATH": str(REPO),
               "MOTION_WORKSPACE": str(tmp_path / "ws"),
               "MOTION_DEFAULT_PROVIDER": "fake",  # beats any developer .env
               **(extra_env or {})}
        (tmp_path / "home").mkdir(exist_ok=True)
        proc = subprocess.run(
            [sys.executable, str(REPO / "main.py"), *args], input=stdin, capture_output=True, text=True,
            env=env, cwd=str(tmp_path), timeout=60,
        )
        return proc, fake

    yield run
    for s in servers:
        s.close()


def test_cli_text_mode_end_to_end(cli):
    proc, fake = cli(read_then_answer, "-p", "What is the codeword in notes.txt?")
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == "The codeword is PELICAN-42.\n"
    names = {t["function"]["name"] for t in fake.requests[0]["tools"]}
    assert {"read_file", "grep", "write_file"} <= names
    assert fake.requests[0]["messages"][0]["role"] == "system" and fake.requests[0]["stream"] is True


def test_cli_json_mode_reports_real_usage_and_cost(cli):
    proc, _ = cli(read_then_answer, "-p", "codeword?", "--output-format", "json")
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    assert data["ok"] and data["result"] == "The codeword is PELICAN-42." and data["tool_calls"] == 1
    assert data["usage"] == {"prompt_tokens": 2000, "completion_tokens": 1000, "total_tokens": 3000}
    assert data["cost_usd"] == pytest.approx(0.004) and data["model"] == "m"


def test_cli_stream_json_and_verbose(cli):
    proc, _ = cli(read_then_answer, "-p", "codeword?", "--output-format", "stream-json", "--verbose")
    assert proc.returncode == 0
    kinds = [json.loads(l)["type"] for l in proc.stdout.splitlines()]
    assert kinds[-1] == "result" and "text" in kinds and "tool" in kinds
    assert "read `notes.txt`" in proc.stderr


def test_cli_plan_flag_hides_write_tools_from_the_model(cli):
    proc, fake = cli(read_then_answer, "-p", "codeword?", "--plan")
    assert proc.returncode == 0
    names = {t["function"]["name"] for t in fake.requests[0]["tools"]}
    assert "read_file" in names and not ({"write_file", "run_command", "replace_in_file"} & names)


def test_cli_prompt_from_stdin_and_stdin_as_context(cli):
    proc, fake = cli(read_then_answer, "-p", "-", stdin="What is the codeword in notes.txt?")
    assert proc.returncode == 0 and fake.requests[0]["messages"][-1]["content"] == "What is the codeword in notes.txt?"
    proc, fake = cli(read_then_answer, "-p", "Summarize:", "--stdin", stdin="line one\nline two\n")
    assert proc.returncode == 0
    sent = fake.requests[0]["messages"][-1]["content"]
    assert sent.startswith("Summarize:") and "<stdin>\nline one\nline two\n\n</stdin>" in sent


def test_cli_provider_error_gives_exit_1_and_message_on_stderr(cli):
    proc, _ = cli(lambda body, n: (500, []), "-p", "hello")
    assert proc.returncode == 1 and proc.stdout == "" and "500" in proc.stderr
    proc, _ = cli(lambda body, n: (500, []), "-p", "hello", "--output-format", "json")
    assert proc.returncode == 1 and json.loads(proc.stdout)["ok"] is False


def test_cli_transient_errors_are_retried(cli):
    def flaky(body, n):
        return (503, []) if n < 3 else (200, [delta(content="recovered"), USAGE])

    proc, fake = cli(flaky, "-p", "hello", max_retries=3)
    assert proc.returncode == 0 and proc.stdout == "recovered\n" and len(fake.requests) == 3


def test_cli_usage_errors_exit_2(cli):
    assert cli(read_then_answer, "-p", "   ")[0].returncode == 2                      # empty prompt
    assert cli(read_then_answer, "-p", "x", "--workspace", "/no/such/dir")[0].returncode == 2
    assert cli(read_then_answer, "-p", "x", "--provider", "does-not-exist")[0].returncode == 2
    assert cli(read_then_answer, "-p", "x", "--output-format", "yaml")[0].returncode == 2   # argparse


def test_cli_risky_command_is_refused_and_reported_to_the_model(cli, tmp_path):
    (tmp_path / "ws" / "keep").mkdir()

    def script(body, n):
        if body["messages"][-1]["role"] == "tool":
            assert "needs user approval" in body["messages"][-1]["content"]
            return 200, [delta(content="I was not allowed to."), USAGE]
        c = {"index": 0, "id": "c", "function": {"name": "run_command", "arguments": json.dumps({"command": "rm -rf keep"})}}
        return 200, [delta(tool_calls=[c]), USAGE]

    proc, _ = cli(script, "-p", "delete keep")
    assert proc.returncode == 0 and proc.stdout == "I was not allowed to.\n"
    assert (tmp_path / "ws" / "keep").exists()


def test_cli_config_permissions_can_preapprove_a_command(cli, tmp_path):
    (tmp_path / "ws" / "keep").mkdir()

    def script(body, n):
        if body["messages"][-1]["role"] == "tool":
            return 200, [delta(content="done"), USAGE]
        c = {"index": 0, "id": "c", "function": {"name": "run_command", "arguments": json.dumps({"command": "rm -rf keep"})}}
        return 200, [delta(tool_calls=[c]), USAGE]

    # The fixture writes the config; append a permissions rule through MOTION_CONFIG's file.
    proc, fake = cli(script, "-p", "x", extra_env={})   # first, confirm it is refused
    assert (tmp_path / "ws" / "keep").exists()
    cfg = tmp_path / "config.yml"
    cfg.write_text(cfg.read_text() + 'permissions:\n  commands:\n    allow: ["rm -rf keep"]\n')
    env = {**os.environ, "MOTION_CONFIG": str(cfg), "HOME": str(tmp_path / "home"), "PYTHONPATH": str(REPO),
           "MOTION_WORKSPACE": str(tmp_path / "ws"), "MOTION_DEFAULT_PROVIDER": "fake"}
    fake2 = FakeLLM(script)
    cfg.write_text(cfg.read_text().replace(fake.url, fake2.url))
    try:
        proc = subprocess.run([sys.executable, str(REPO / "main.py"), "-p", "x"], capture_output=True, text=True, env=env, cwd=str(tmp_path), timeout=60)
    finally:
        fake2.close()
    assert proc.returncode == 0, proc.stderr
    assert not (tmp_path / "ws" / "keep").exists()
