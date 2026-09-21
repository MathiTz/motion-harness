from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.providers import ModelConfig
from core.workspace_tools import WorkspaceToolError, WorkspaceTools, parse_tool_call
from main import MotionAgent


def test_write_and_replace_file(tmp_path: Path) -> None:
    tools = WorkspaceTools(tmp_path)

    result = tools.execute(
        "write_file",
        {"path": "src/app.py", "content": "print('old')\n"},
    )
    assert result["path"] == "src/app.py"
    assert (tmp_path / "src/app.py").read_text() == "print('old')\n"

    tools.execute(
        "replace_in_file",
        {
            "path": "src/app.py",
            "old": "print('old')",
            "new": "print('new')",
        },
    )
    assert (tmp_path / "src/app.py").read_text() == "print('new')\n"


def test_tools_reject_paths_outside_workspace(tmp_path: Path) -> None:
    tools = WorkspaceTools(tmp_path)
    with pytest.raises(WorkspaceToolError, match="escapes"):
        tools.execute("write_file", {"path": "../outside.txt", "content": "no"})


def test_plan_mode_rejects_writes(tmp_path: Path) -> None:
    tools = WorkspaceTools(tmp_path, read_only=True)
    with pytest.raises(WorkspaceToolError, match="plan mode"):
        tools.execute("write_file", {"path": "blocked.txt", "content": "no"})


def test_plan_mode_instructions_do_not_demand_writes(tmp_path: Path) -> None:
    # Regression: Plan-mode instructions must never tell the model it "MUST"
    # write files, since write_file/replace_in_file always fail read-only.
    # Otherwise the model just punts back to the user instead of planning.
    instructions = WorkspaceTools(tmp_path, read_only=True).instructions
    assert "MUST actually write files" not in instructions
    assert "DISABLED" in instructions
    assert "PLAN" in instructions
    # Plan mode must still advertise the read/search tools it can use.
    assert "read_file" in instructions
    assert "web_fetch" in instructions
    # Mutation/run tools are NOT advertised as available; they're gated.
    assert "- run_command:" not in instructions


def test_build_mode_instructions_still_demand_writes(tmp_path: Path) -> None:
    instructions = WorkspaceTools(tmp_path, read_only=False).instructions
    assert "write_file" in instructions
    assert "run_script" in instructions
    assert "run_python" in instructions
    assert "web_fetch" in instructions
    assert "web_search" in instructions
    # Advertises execution so the model doesn't punt on user scripts.
    assert "run_script" in instructions or "run_python" in instructions


def test_parse_tool_call() -> None:
    call = parse_tool_call(
        '<motion_tool>{"name":"read_file","arguments":{"path":"README.md"}}</motion_tool>'
    )
    assert call == ("read_file", {"path": "README.md"})


def test_parse_tool_call_tolerates_malformed_opening_tag() -> None:
    call = parse_tool_call(
        '<motion_tool">{"name":"list_files","arguments":{"path":".","pattern":"*"}}</motion_tool>'
    )
    assert call == ("list_files", {"path": ".", "pattern": "*"})
def test_parse_tool_call_accepts_direct_xml_tool_tag() -> None:
    call = parse_tool_call(
        'I will inspect the project.\n<list_files>{"path": ".", "pattern": "*"}</list_files>'
    )
    assert call == ("list_files", {"path": ".", "pattern": "*"})


def test_parse_tool_call_accepts_direct_write_tag() -> None:
    call = parse_tool_call(
        '<write_file>{"path":"app.py","content":"print(1)\\n"}</write_file>'
    )
    assert call == ("write_file", {"path": "app.py", "content": "print(1)\n"})


def test_parse_recovers_misspelled_envelope_and_unescaped_source_quotes() -> None:
    response = (
        '<motion_tilte>{"name":"write_file","arguments":'
        '{"path":"models/product.py","content":""""Product data model."""\\n\\n'
        'from dataclasses import dataclass\\n\\n'
        '@dataclass\\nclass Product:\\n    name: str\\n'
        '    def to_dict(self):\\n        return {"name": self.name}\\n"}}'
        '</motion_tool>'
    )

    call = parse_tool_call(response)

    assert call is not None
    name, arguments = call
    assert name == "write_file"
    assert arguments["path"] == "models/product.py"
    assert arguments["content"].startswith('"""Product data model."""\n\n')
    assert 'return {"name": self.name}' in arguments["content"]


def test_parse_tool_call_rejects_unparseable_markup() -> None:
    with pytest.raises(WorkspaceToolError, match="malformed"):
        parse_tool_call("<motion_tool>not json</motion_tool>")


def test_parse_tool_call_tolerates_space_after_opening_bracket() -> None:
    # Regression: a model emitted "< motion_tool>" (stray space after "<").
    # This previously matched none of the tool patterns AND slipped past the
    # malformed-tag leak guard (which checked for the literal substring
    # "<motion_"), so parse_tool_call() returned None and the raw tool-call
    # JSON was shown to the user as if it were the final answer.
    call = parse_tool_call(
        '< motion_tool>{"name":"write_file","arguments":'
        '{"path":"requirements.txt","content":"fastapi\\nuvicorn\\n"}}</motion_tool>'
    )
    assert call == ("write_file", {"path": "requirements.txt", "content": "fastapi\nuvicorn\n"})


def test_parse_tool_call_tolerates_space_before_closing_bracket() -> None:
    call = parse_tool_call(
        '<motion_tool>{"name":"read_file","arguments":{"path":"a.txt"}}< /motion_tool>'
    )
    assert call == ("read_file", {"path": "a.txt"})


def test_parse_tool_call_raises_instead_of_leaking_when_truly_malformed() -> None:
    # A stray-space opening tag with no valid closing tag at all must still
    # be caught by the leak guard (raise) rather than silently returning
    # None, which would let the raw markup leak to the user as final text.
    with pytest.raises(WorkspaceToolError, match="malformed"):
        parse_tool_call('< motion_tool>{"name":"write_file"')


def test_parse_tool_call_handles_dsml_channel_leak() -> None:
    # Regression: deepseek-v4-flash leaked its own internal tool-channel
    # markup verbatim - "<| DSML | tool:write_file>{...}</ | DSML | tool>" -
    # which matched none of the motion_tool/direct-tag patterns nor the old
    # leak guard, so the raw call (including a huge README) was shown as the
    # final chat answer instead of being executed.
    call = parse_tool_call(
        '<| DSML | tool:write_file>{"path":"README.md","content":"# Title\\n"}'
        '</ | DSML | tool>'
    )
    assert call == ("write_file", {"path": "README.md", "content": "# Title\n"})


def test_parse_tool_call_generic_guard_catches_unknown_envelope_names() -> None:
    # Defense in depth: whatever exotic envelope name a model invents next,
    # a tag wrapping something shaped like write_file's arguments (path +
    # content) must never be shown as final text - it should raise instead
    # of returning None, so main.py's retry loop kicks in.
    with pytest.raises(WorkspaceToolError, match="malformed"):
        parse_tool_call(
            '<some_unknown_channel>{"path":"a.txt","content":"x"}</some_unknown_channel>'
        )


def test_parse_tool_call_does_not_flag_prose_with_html_and_json_keys() -> None:
    # Regression: after inspecting files, a model naturally summarizes using
    # words like "arguments", "path", and "content" alongside HTML-like "<".
    # The old broad leak guard ("<" anywhere + JSON keys anywhere) treated this
    # as a malformed tool call and aborted the turn. It must return None so the
    # prose is displayed as the final answer.
    response = (
        "I inspected the project. web/app.py has a route that takes "
        '`path` and `content` as arguments, and arguments={"x": 1}. '
        "The HTML uses `<form>` tags. Let me know what you'd like to change."
    )
    assert parse_tool_call(response) is None


class _ToolCallingProvider:
    class _Config:
        provider_type = "local"

    def __init__(self) -> None:
        self.config = self._Config()
        self.calls = 0

    async def complete(self, prompt, system_prompt="", history=None, **kwargs):
        self.calls += 1
        if self.calls == 1:
            assert "write_file" in system_prompt
            return (
                '<motion_tool>{"name":"write_file","arguments":'
                '{"path":"generated.txt","content":"created by agent\\n"}}</motion_tool>'
            )
        assert history
        assert "motion_tool_result" in history[-1]["content"]
        return "Created `generated.txt`."

    async def close(self):
        pass


class _EmptyRetriever:
    async def retrieve(self, query, top_k=5):
        return []


async def test_agent_executes_write_tool(tmp_path: Path) -> None:
    agent = MotionAgent(
        ModelConfig(name="test", endpoint="http://localhost", provider_type="local"),
        memory_path=":memory:",
    )
    agent.provider = _ToolCallingProvider()
    agent.retriever = _EmptyRetriever()

    response = await agent.run(
        "Create generated.txt",
        workspace=str(tmp_path),
        agent_mode="build",
    )

    assert response == "Created `generated.txt`."
    assert (tmp_path / "generated.txt").read_text() == "created by agent\n"


class _EmptyFinalProvider(_ToolCallingProvider):
    async def complete(self, prompt, system_prompt="", history=None, **kwargs):
        self.calls += 1
        if self.calls == 1:
            return (
                '<motion_tool>{"name":"write_file","arguments":'
                '{"path":"fallback.txt","content":"done\\n"}}</motion_tool>'
            )
        return ""


async def test_agent_summarizes_tools_when_model_final_is_empty(tmp_path: Path) -> None:
    agent = MotionAgent(
        ModelConfig(name="test", endpoint="http://localhost", provider_type="local"),
        memory_path=":memory:",
    )
    agent.provider = _EmptyFinalProvider()
    agent.retriever = _EmptyRetriever()

    response = await agent.run(
        "Create fallback.txt",
        workspace=str(tmp_path),
        agent_mode="build",
    )

    assert response == "Completed filesystem changes:\n- wrote `fallback.txt`"
    assert (tmp_path / "fallback.txt").read_text() == "done\n"
    assert agent.provider.calls == 4


class _PlanModeBlockedWriteProvider:
    """Simulates a model that first tries to write, gets blocked, then plans."""

    class _Config:
        provider_type = "local"

    def __init__(self) -> None:
        self.config = self._Config()
        self.calls = 0
        self.seen_histories: list[list] = []

    async def complete(self, prompt, system_prompt="", history=None, **kwargs):
        self.calls += 1
        self.seen_histories.append(history or [])
        if self.calls == 1:
            assert "MUST actually write files" not in system_prompt
            return (
                '<motion_tool>{"name":"write_file","arguments":'
                '{"path":"scraper.py","content":"pass\\n"}}</motion_tool>'
            )
        # Second turn: the loop must have nudged it away from retrying the
        # write and toward answering with a real plan.
        assert "read-only Plan mode" in history[-1]["content"]
        return "Plan: create scraper.py, storage.py, and a CLI entrypoint."

    async def close(self):
        pass


async def test_plan_mode_recovers_from_blocked_write_with_real_plan(tmp_path: Path) -> None:
    agent = MotionAgent(
        ModelConfig(name="test", endpoint="http://localhost", provider_type="local"),
        memory_path=":memory:",
    )
    agent.provider = _PlanModeBlockedWriteProvider()
    agent.retriever = _EmptyRetriever()

    response = await agent.run(
        "We'll create a market scraper",
        workspace=str(tmp_path),
        agent_mode="plan",
    )

    assert response == "Plan: create scraper.py, storage.py, and a CLI entrypoint."
    assert not (tmp_path / "scraper.py").exists()


class _NeverFinishesProvider:
    """Simulates a model that keeps writing files and never stops on its own,
    forcing the tool loop to hit its MAX_TOOL_STEPS cap."""

    class _Config:
        provider_type = "local"

    def __init__(self) -> None:
        self.config = self._Config()
        self.calls = 0

    async def complete(self, prompt, system_prompt="", history=None, **kwargs):
        self.calls += 1
        return (
            '<motion_tool>{"name":"write_file","arguments":'
            f'{{"path":"file{self.calls}.txt","content":"content {self.calls}\\n"}}}}'
            '</motion_tool>'
        )

    async def close(self):
        pass


async def test_agent_reports_progress_when_tool_call_cap_is_hit(tmp_path: Path) -> None:
    # Regression: hitting the tool-call cap used to discard all completed
    # work and reply with a bare "Stopped after 12 tool calls. Please narrow
    # the task and try again." even when files had already been written.
    from main import MAX_TOOL_STEPS

    agent = MotionAgent(
        ModelConfig(name="test", endpoint="http://localhost", provider_type="local"),
        memory_path=":memory:",
    )
    agent.provider = _NeverFinishesProvider()
    agent.retriever = _EmptyRetriever()

    response = await agent.run(
        "Build a large multi-file project",
        workspace=str(tmp_path),
        agent_mode="build",
    )

    assert f"internal safety limit ({MAX_TOOL_STEPS} tool calls)" in response
    assert "wrote `file1.txt`" in response
    assert f"wrote `file{MAX_TOOL_STEPS}.txt`" in response
    assert "continue" in response.lower()
    # The files were actually written to disk despite the loop not finishing.
    assert (tmp_path / "file1.txt").read_text() == "content 1\n"
    assert (tmp_path / f"file{MAX_TOOL_STEPS}.txt").exists()


class _FailsOnceThenPlansProvider:
    """Simulates a model that emits a malformed tool call once, then recovers."""

    class _Config:
        provider_type = "local"

    def __init__(self) -> None:
        self.config = self._Config()
        self.calls = 0

    async def complete(self, prompt, system_prompt="", history=None, **kwargs):
        self.calls += 1
        if self.calls == 1:
            return "<motion_tool>not valid json</motion_tool>"
        return "Plan: fix the scraper and run it."

    async def close(self):
        pass


async def test_agent_continues_after_malformed_tool_call(tmp_path: Path) -> None:
    # Regression: tool errors should enrich context and let the loop continue,
    # not hard-stop the interaction.
    agent = MotionAgent(
        ModelConfig(name="test", endpoint="http://localhost", provider_type="local"),
        memory_path=":memory:",
    )
    agent.provider = _FailsOnceThenPlansProvider()
    agent.retriever = _EmptyRetriever()

    response = await agent.run(
        "Review the scraper",
        workspace=str(tmp_path),
        agent_mode="plan",
    )

    assert response == "Plan: fix the scraper and run it."
    assert agent.provider.calls == 2


class _ToolFailsOnceThenWritesProvider:
    """Simulates write_file failing once, then succeeding."""

    class _Config:
        provider_type = "local"

    def __init__(self) -> None:
        self.config = self._Config()
        self.calls = 0

    async def complete(self, prompt, system_prompt="", history=None, **kwargs):
        self.calls += 1
        if self.calls == 1:
            return (
                '<motion_tool>{"name":"write_file","arguments":'
                '{"path":"../out.txt","content":"first\\n"}}</motion_tool>'
            )
        if self.calls == 2:
            return (
                '<motion_tool>{"name":"write_file","arguments":'
                '{"path":"out.txt","content":"second\\n"}}</motion_tool>'
            )
        return "Created out.txt after retrying."

    async def close(self):
        pass


async def test_agent_continues_after_tool_execution_error(tmp_path: Path) -> None:
    # Regression: a tool execution error should be fed back as context so the
    # model can recover, not stop the loop immediately.
    agent = MotionAgent(
        ModelConfig(name="test", endpoint="http://localhost", provider_type="local"),
        memory_path=":memory:",
    )
    agent.provider = _ToolFailsOnceThenWritesProvider()
    agent.retriever = _EmptyRetriever()

    # First write fails because the path escapes the workspace.
    response = await agent.run(
        "Create out.txt",
        workspace=str(tmp_path),
        agent_mode="build",
    )

    assert "Created out.txt after retrying." in response
    assert (tmp_path / "out.txt").read_text() == "second\n"
    assert agent.provider.calls == 3


# ── New toolset: run_script / run_python / read_image / memory / env ─────────

def test_run_script_executes_file(tmp_path: Path) -> None:
    script = tmp_path / "hello.py"
    script.write_text("import sys; print('hi', sys.argv[1])")
    tools = WorkspaceTools(tmp_path, read_only=False)
    result = tools.execute("run_script", {"path": "hello.py", "args": ["there"]})
    assert result["exit_code"] == 0
    assert "hi there" in result["stdout"]


def test_run_script_missing_file(tmp_path: Path) -> None:
    tools = WorkspaceTools(tmp_path, read_only=False)
    with pytest.raises(WorkspaceToolError):
        tools.execute("run_script", {"path": "nope.py"})

def test_run_script_denied_in_plan_mode(tmp_path: Path) -> None:
    tools = WorkspaceTools(tmp_path, read_only=True)
    with pytest.raises(WorkspaceToolError):
        tools.execute("run_script", {"path": "hello.py", "args": []})


def test_run_python_executes_snippet(tmp_path: Path) -> None:
    tools = WorkspaceTools(tmp_path, read_only=False)
    result = tools.execute("run_python", {"code": "print(6*7)"})
    assert result["exit_code"] == 0
    assert "42" in result["stdout"]


def test_glob_files_finds_matches(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("")
    (tmp_path / "src" / "b.txt").write_text("")
    tools = WorkspaceTools(tmp_path, read_only=False)
    result = tools.execute("glob_files", {"pattern": "src/*.py"})
    assert "src/a.py" in result["files"]


def test_read_image_returns_base64(tmp_path: Path) -> None:
    png = tmp_path / "x.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 20)
    tools = WorkspaceTools(tmp_path, read_only=False)
    result = tools.execute("read_image", {"path": "x.png"})
    assert result["format"] == "png"
    assert result["mime"] == "image/png"
    assert result["data_url"].startswith("data:image/png;base64,")


def test_memory_save_get(tmp_path: Path) -> None:
    tools = WorkspaceTools(tmp_path, read_only=False)
    tools.execute("memory_save", {"key": "k", "text": "hello world"})
    got = tools.execute("memory_get", {"key": "k"})
    assert got["found"] is True
    assert got["text"] == "hello world"


def test_env_var_whitelist(tmp_path: Path) -> None:
    tools = WorkspaceTools(tmp_path, read_only=False)
    import os
    os.environ["PYTHONPATH"] = "/some/path"
    result = tools.execute("env_var", {"name": "PYTHONPATH"})
    assert result["value"] == "/some/path"
    with pytest.raises(WorkspaceToolError):
        tools.execute("env_var", {"name": "API_CLAUDE_KEY"})


def test_web_fetch(tmp_path: Path) -> None:
    tools = WorkspaceTools(tmp_path, read_only=False)
    with patch("core.workspace_tools.httpx.get") as mock_get:
        resp = MagicMock()
        resp.status_code = 200
        resp.text = "<html><body>Hello world</body></html>"
        resp.headers = {"content-type": "text/html"}
        mock_get.return_value = resp
        result = tools.execute("web_fetch", {"url": "https://example.com"})
    assert result["status"] == 200
    assert "Hello world" in result["text"]


def test_web_search_ddg(tmp_path: Path) -> None:
    tools = WorkspaceTools(tmp_path, read_only=False)
    with patch("core.workspace_tools.httpx.get") as mock_get:
        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status.return_value = None
        resp.text = (
            '<a class="result__a" href="https://example.com">Example</a>'
        )
        mock_get.return_value = resp
        result = tools.execute("web_search", {"query": "python docs"})
    assert result["count"] >= 1
    assert result["results"][0]["title"] == "Example"


class _ScriptRunnerProvider:
    """Model: call run_script, then produce a final answer citing the output."""
    class _Config:
        provider_type = "local"
    def __init__(self) -> None:
        self.config = self._Config()
        self.calls = 0
    async def complete(self, prompt, system_prompt="", history=None, **kwargs):
        self.calls += 1
        if self.calls == 1:
            assert "run_script" in system_prompt
            return (
                '<motion_tool>{"name":"run_script","arguments":'
                '{"path":"hello.py","args":["world"]}}</motion_tool>'
            )
        assert history
        last = history[-1]["content"]
        assert "motion_tool_result" in last
        assert "hello world" in last  # real captured stdout propagated back
        return "The script ran and printed its output."
    async def close(self):
        pass


async def test_agent_runs_user_script_tool(tmp_path: Path) -> None:
    """End-to-end: the agent must actually run a user script (not hand code
    back) and receive the captured output — the bug this feature fixes."""
    (tmp_path / "hello.py").write_text("import sys; print('hello', sys.argv[1])\n")
    agent = MotionAgent(
        ModelConfig(name="test", endpoint="http://localhost", provider_type="local"),
        memory_path=":memory:",
    )
    agent.provider = _ScriptRunnerProvider()
    agent.retriever = _EmptyRetriever()

    response = await agent.run(
        "Run hello.py with arg 'world'",
        workspace=str(tmp_path),
        agent_mode="build",
    )
    assert "ran and printed" in response
    assert agent.provider.calls == 2


class _RunPythonProvider:
    class _Config:
        provider_type = "local"
    def __init__(self) -> None:
        self.config = self._Config()
        self.calls = 0
    async def complete(self, prompt, system_prompt="", history=None, **kwargs):
        self.calls += 1
        if self.calls == 1:
            return (
                '<motion_tool>{"name":"run_python","arguments":'
                '{"code":"print(6*7)"}}</motion_tool>'
            )
        assert history
        last = history[-1]["content"]
        assert "motion_tool_result" in last
        assert "42" in last
        return "Six times seven is 42."
    async def close(self):
        pass


async def test_agent_runs_python_snippet(tmp_path: Path) -> None:
    agent = MotionAgent(
        ModelConfig(name="test", endpoint="http://localhost", provider_type="local"),
        memory_path=":memory:",
    )
    agent.provider = _RunPythonProvider()
    agent.retriever = _EmptyRetriever()
    response = await agent.run("compute 6*7", workspace=str(tmp_path), agent_mode="build")
    assert "42" in response
    assert agent.provider.calls == 2
