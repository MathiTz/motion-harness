from pathlib import Path

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


def test_build_mode_instructions_still_demand_writes(tmp_path: Path) -> None:
    instructions = WorkspaceTools(tmp_path, read_only=False).instructions
    assert "MUST actually write files" in instructions


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
