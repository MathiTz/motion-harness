"""MCP client tested against a real, independently-maintained third-party server (issue #15).

tests/test_mcp.py's own docstring is honest about its scope: "MCP client tests against a tiny
stdio echo server" - a test double written for this suite. This file instead runs the official
filesystem reference server from https://github.com/modelcontextprotocol/servers
(`@modelcontextprotocol/server-filesystem`, run via `npx`), proving our MCP client (core/mcp.py)
actually interoperates with a real server someone else maintains, not just our own double.

Needs Node/npx and, on first run in a fresh environment, network access to fetch the package from
the npm registry (npx caches it after that). Skipped automatically wherever npx isn't on PATH -
this is not run in CI (see docs/compatibility.md and .github/workflows/ci.yml, which doesn't
install Node). It's a maintainer/local, on-demand check, same as scripts/live_check.py, not a merge
gate - the mock-transport tests already cover the protocol mechanics on every PR.
"""
import shutil

import pytest

from core.mcp import MCPManager

pytestmark = pytest.mark.skipif(shutil.which("npx") is None, reason="needs Node/npx on PATH (not installed in CI)")


def fs_config(root: str) -> dict:
    return {"fs": {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", root]}}


async def test_discovers_real_tools_and_reads_a_real_file(tmp_path):
    (tmp_path / "greeting.txt").write_text("hello from a real MCP server\n")
    mgr = MCPManager(fs_config(str(tmp_path)))
    try:
        await mgr.initialize_all()
        assert mgr.errors == {}
        tools = {t[1] for t in mgr.tool_index()}
        # The reference server's real tool set - not anything this test suite invented.
        assert {"read_text_file", "write_file", "list_directory"} <= tools

        result = await mgr.call_tool("fs", "read_text_file", {"path": str(tmp_path / "greeting.txt")})
        assert result["text"] == "hello from a real MCP server\n"
    finally:
        await mgr.close_all()


async def test_writes_through_the_real_server_land_on_disk(tmp_path):
    mgr = MCPManager(fs_config(str(tmp_path)))
    try:
        await mgr.initialize_all()
        await mgr.call_tool("fs", "write_file", {"path": str(tmp_path / "written.txt"), "content": "written via MCP"})
        assert (tmp_path / "written.txt").read_text() == "written via MCP"
    finally:
        await mgr.close_all()


async def test_the_real_servers_own_errors_surface_as_mcp_error(tmp_path):
    from core.mcp import MCPError

    mgr = MCPManager(fs_config(str(tmp_path)))
    try:
        await mgr.initialize_all()
        with pytest.raises(MCPError):
            await mgr.call_tool("fs", "read_text_file", {"path": str(tmp_path / "does-not-exist.txt")})
    finally:
        await mgr.close_all()
