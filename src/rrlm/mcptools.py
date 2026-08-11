"""MCP servers as host-side agent tools (opt-in ``mcp`` extra).

An MCP (Model Context Protocol) server exposes tools over stdio; this module
mounts them into a solve as ordinary awaitable host tools, so the agent calls
``await <tool_name>(...)`` from the REPL exactly like any other host tool and
the call is bridged to the server. Connections live for the duration of one
solve (opened lazily inside :func:`rrlm.asolve`, closed when the run ends) and
each server runs as a real subprocess the caller names explicitly::

    from rrlm import solve
    from rrlm.mcptools import MCPServerSpec

    result = solve(
        "Look up the vendor in the CRM and report its status.",
        data=text,
        mcp=[MCPServerSpec(command="crm-mcp-server", allow=("lookup_vendor",))],
    )

Security note: MCP tools run host-side with this process's permissions on
every backend (the sandbox only isolates the generated Python, not host
tools). ``allow`` narrows a server to named tools; prefer it for servers that
expose more than the task needs.
"""

from __future__ import annotations

import asyncio
import shlex
from collections.abc import Mapping
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class MCPServerSpec:
    """One stdio MCP server to mount for a solve.

    ``allow`` is a tool-name allowlist (None mounts every tool the server
    lists); ``prefix`` namespaces the mounted names (``prefix="crm_"`` turns
    ``lookup`` into ``crm_lookup``) to avoid clashes between servers.
    """

    command: str
    args: tuple[str, ...] = ()
    env: Mapping[str, str] | None = None
    allow: tuple[str, ...] | None = None
    prefix: str = ""

    @classmethod
    def parse(cls, spec: str) -> MCPServerSpec:
        """Build a spec from a CLI string: a shell-split command line."""
        parts = shlex.split(spec)
        if not parts:
            raise ValueError("empty --mcp server command")
        return cls(command=parts[0], args=tuple(parts[1:]))


@dataclass
class MountedTool:
    """A server tool ready to mount: the callable plus its self-description."""

    name: str
    description: str
    call: Any = field(repr=False)


def _require_mcp():
    try:
        import mcp  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "MCP support needs the 'mcp' extra: uv sync --extra mcp "
            "(or: pip install 'rrlm[mcp]')"
        ) from exc


def extract_text(result) -> str:
    """Flatten a CallToolResult's content into text; raise on server errors.

    A tool error must surface as an exception so the REPL sees a failure it
    can react to, not an error message that reads like a successful answer.
    """
    parts = [
        getattr(item, "text")
        for item in getattr(result, "content", ())
        if getattr(item, "text", None) is not None
    ]
    text = "\n".join(parts)
    if getattr(result, "isError", False):
        raise RuntimeError(f"MCP tool failed: {text or 'no error detail provided'}")
    return text


def select_tools(listed: list, spec: MCPServerSpec) -> list:
    """Apply the spec's allowlist to a server's listed tools.

    Naming a tool that the server does not expose is a config error and fails
    loudly - a silent partial mount would read as the model ignoring a tool.
    """
    if spec.allow is None:
        return list(listed)
    by_name = {tool.name: tool for tool in listed}
    missing = [name for name in spec.allow if name not in by_name]
    if missing:
        known = ", ".join(sorted(by_name)) or "none"
        raise ValueError(
            f"MCP server {spec.command!r} does not expose {', '.join(missing)} "
            f"(available: {known})"
        )
    return [by_name[name] for name in spec.allow]


async def connect_mcp_servers(
    stack: AsyncExitStack, specs: list[MCPServerSpec]
) -> list[MountedTool]:
    """Connect every server on the given stack; return the tools to mount.

    The stack owns the connections: when the caller's ``async with`` exits,
    sessions close and the server subprocesses end. Wrappers carry the MCP
    tool's name and description so the model sees what the server declared.
    """
    _require_mcp()
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    mounted: list[MountedTool] = []
    for spec in specs:
        params = StdioServerParameters(
            command=spec.command,
            args=list(spec.args),
            env=dict(spec.env) if spec.env else None,
        )
        read, write = await stack.enter_async_context(stdio_client(params))
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        listed = (await session.list_tools()).tools
        for tool in select_tools(listed, spec):
            mounted.append(_mount(session, tool, spec.prefix))
    return mounted


def _mount(session, tool, prefix: str) -> MountedTool:
    name = f"{prefix}{tool.name}"
    description = tool.description or f"MCP tool {tool.name}"
    # The ClientSession is bound to the loop it was opened on (this one), but
    # predict-rlm's host-tool bridge awaits tools with asyncio.run() in its
    # own worker thread - a different loop. Calling the session there would
    # hang forever, so route every call back to the owning loop and hand the
    # result across with a thread-safe future. Same-loop callers skip the hop.
    owner_loop = asyncio.get_running_loop()

    async def call(**kwargs):
        coro = session.call_tool(tool.name, kwargs or None)
        if asyncio.get_running_loop() is owner_loop:
            result = await coro
        else:
            result = await asyncio.wrap_future(
                asyncio.run_coroutine_threadsafe(coro, owner_loop)
            )
        return extract_text(result)

    call.__name__ = name
    call.__qualname__ = name
    call.__doc__ = description
    return MountedTool(name=name, description=description, call=call)


def mcp_tools_note(mounted: list[MountedTool]) -> str:
    """One instruction block advertising the mounted tools to the model."""
    lines = [
        "Extra host tools are mounted in this REPL (await them like "
        "`result = await tool_name(arg=value)`; they return text):"
    ]
    lines += [f"- {tool.name}: {tool.description}" for tool in mounted]
    return "\n".join(lines)
