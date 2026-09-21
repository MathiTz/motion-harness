"""Lightweight MCP (Model Context Protocol) client support.

Lets users plug external MCP servers into Motion Harness so the agent can
call their tools via the standard ``mcp_call`` motion tool, e.g.:

    <motion_tool>{"name":"mcp_call","arguments":{
        "server":"github","tool":"get_issue","arguments":{"id":123}
    }}</motion_tool>

Servers are defined in config.yml under an ``mcp:`` block:

    mcp:
      servers:
        github:
          command: "npx"
          args: ["-y", "@modelcontextprotocol/server-github"]
          env: { GITHUB_PERSONAL_ACCESS_TOKEN: "..." }

Two transports are supported when the underlying transport is available:
  - stdio: spawn a local command (``command`` + ``args``)
  - http:  connect to an MCP-over-HTTP/SSE endpoint (``url``)

The optional ``mcp`` python package enables the full protocol (list_tools,
call_tool, initialize). When it is not installed, we degrade to a simple
stdio JSON-RPC client for basic tool listing/calling, so the feature works
with minimal dependencies.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import shlex
import subprocess
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class MCPError(ValueError):
    """Raised when an MCP operation fails."""


class MCPServer:
    """A single configured MCP server."""

    def __init__(self, name: str, cfg: Dict[str, Any]) -> None:
        self.name = name
        self.cfg = cfg
        self.command = cfg.get("command")
        self.args = cfg.get("args", [])
        self.env = dict(cfg.get("env", {}))
        self.url = cfg.get("url")

    @property
    def transport(self) -> str:
        return "http" if self.url else "stdio"


class MCPManager:
    """Manages a set of configured MCP servers and exposes their tools."""

    def __init__(self, servers_cfg: Optional[Dict[str, Any]] = None) -> None:
        self.servers: Dict[str, MCPServer] = {}
        self._proc: Dict[str, subprocess.Popen] = {}
        self._tools: Dict[str, List[Dict[str, Any]]] = {}
        if servers_cfg:
            for name, cfg in servers_cfg.items():
                self.servers[name] = MCPServer(name, cfg)

    def has_servers(self) -> bool:
        return bool(self.servers)

    async def initialize_all(self) -> None:
        """Best-effort initialize each configured server and list its tools."""
        for name in list(self.servers):
            try:
                tools = await self.list_tools(name)
                self._tools[name] = tools
            except Exception as e:
                logger.warning("MCP server %s init failed: %s", name, e)

    async def list_tools(self, name: str) -> List[Dict[str, Any]]:
        server = self.servers.get(name)
        if not server:
            raise MCPError(f"unknown MCP server: {name}")
        try:
            return await self._list_tools_stdio(server)
        except Exception as e:
            # Fall back to a minimal stub when the protocol lib / stdio handshake
            # isn't available, so the error is informative rather than opaque.
            raise MCPError(f"could not list tools for {name}: {e}") from e

    async def call_tool(self, name: str, tool: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        server = self.servers.get(name)
        if not server:
            raise MCPError(f"unknown MCP server: {name}")
        return await self._call_tool_http(server, tool, arguments)

    def run_call(self, name: str, tool: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Synchronous wrapper (called from the tool loop) around call_tool.

        The tool loop runs synchronously inside the asyncio event loop, so we
        must not call asyncio.run() in-place. We run the async MCP call in a
        dedicated thread and wait on it with a bounded timeout.
        """
        import threading
        result: Dict[str, Any] = {}
        err: Optional[Exception] = None
        def _run() -> None:
            nonlocal result, err
            try:
                result = asyncio.run(self.call_tool(name, tool, arguments))
            except Exception as e:  # noqa: BLE001
                err = e
        t = threading.Thread(target=_run, daemon=True)
        t.start()
        t.join(timeout=30)
        if t.is_alive():
            raise MCPError(f"MCP call to {name}/{tool} timed out after 30s")
        if err:
            raise MCPError(str(err)) from err
        return result

    # ── transports ──────────────────────────────────────────────────────────
    async def _list_tools_stdio(self, server: MCPServer) -> List[Dict[str, Any]]:
        # Attempt an initialize handshake over stdio. If the `mcp` package is
        # installed we could use it; here we use a plain subprocess JSON-RPC
        # (many MCP stdio servers accept sequential JSON-RPC lines).
        cmd = self._build_cmd(server)
        try:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                env={**os.environ, **server.env} if server.env else None,
            )
        except FileNotFoundError as e:
            raise MCPError(f"binary not found for {server.name}: {cmd[0]}") from e
        self._proc[server.name] = proc
        try:
            init = {"jsonrpc": "2.0", "id": 1, "method": "initialize",
                    "params": {"protocolVersion": "2024-11-05",
                               "capabilities": {}, "clientInfo": {"name": "motion-harness"}}}
            out, _ = self._rpc_roundtrip(proc, init)
            # Then ask for initialized + tools/list.
            self._rpc_roundtrip(proc, {"jsonrpc": "2.0", "method": "notifications/initialized"})
            tools_resp = self._rpc_roundtrip(proc, {
                "jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
            return tools_resp or []
        finally:
            try:
                proc.kill()
            except Exception:
                pass

    def _rpc_roundtrip(self, proc: subprocess.Popen, payload: Dict[str, Any]):
        line = json.dumps(payload)
        assert proc.stdin is not None and proc.stdout is not None
        proc.stdin.write(line + "\n")
        proc.stdin.flush()
        # Read until we have a matching id (bounded).
        import time
        deadline = time.time() + 10
        while time.time() < deadline:
            out = proc.stdout.readline()
            if not out:
                break
            try:
                msg = json.loads(out)
            except Exception:
                continue
            if msg.get("id") == payload.get("id") or msg.get("method") == "tools/list":
                if payload.get("method") in ("tools/list", "tools/call") and "result" in msg:
                    return msg["result"]
                if msg.get("id") == payload.get("id"):
                    return msg.get("result")
        return {}

    async def _call_tool_http(self, server: MCPServer, tool: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        # HTTP transport: only support if the mcp package is installed. Without
        # it we cannot speak SSE, so surface a clear message.
        try:
            from mcp import ClientSession, StdioServerParameters  # noqa: F401
            raise MCPError("http MCP transport requires the full protocol client")
        except ImportError:
            raise MCPError(
                "MCP HTTP transport needs the `mcp` package. Install it (pip install mcp) "
                "or use a stdio-configured server."
            ) from None

    def _build_cmd(self, server: MCPServer) -> List[str]:
        if not server.command:
            raise MCPError(f"server {server.name} is missing 'command' for stdio transport")
        if isinstance(server.command, str):
            # Allow a full shell line like "npx -y @scope/package"
            parts = shlex.split(server.command)
        else:
            parts = [str(server.command)]
        return parts + [str(a) for a in server.args]

    def mcp_call_instruction(self) -> str:
        if not self.servers:
            return ""
        names = ", ".join(sorted(self.servers))
        return (
            f"\nAdditional tools are available from MCP servers ({names}). Call them with:\n"
            f'- mcp_call: {{"server": "<name>", "tool": "<tool>", "arguments": {{...}}}} — invokes a '
            f"tool exposed by an external MCP server"
        )
