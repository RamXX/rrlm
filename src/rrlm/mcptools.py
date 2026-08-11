"""MCP servers as host-side agent tools (opt-in ``mcp`` extra).

This module mounts an MCP (Model Context Protocol) server's tools into a
solve as ordinary awaitable host tools: the agent calls
``await <tool_name>(...)`` from the REPL exactly like any other host tool and
the call is bridged to the server. Connections live for the duration of one
solve (opened lazily inside :func:`rrlm.asolve`, closed when the run ends)::

    from rrlm import solve
    from rrlm.mcptools import MCPServerSpec

    result = solve(
        "Look up the vendor in the CRM and report its status.",
        data=text,
        mcp=[
            # remote server over streamable HTTP (the preferred transport)
            MCPServerSpec(url="https://crm.example.com/mcp",
                          headers={"Authorization": "Bearer ..."},
                          allow=("lookup_vendor",)),
            # local server as a stdio subprocess
            MCPServerSpec(command="local-tools-server"),
        ],
    )

Transports: streamable HTTP for URLs (current best practice), ``stdio`` for
local subprocess servers, and legacy HTTP+SSE via ``transport="sse"`` for
remote servers that have not migrated yet. Protocol generations are the SDK's
concern: the client negotiates the version at ``initialize``, so servers on
the current stateless-HTTP revision and older stateful servers both work.

Security note: MCP tools run host-side with this process's permissions on
every backend (the sandbox only isolates the generated Python, not host
tools). ``allow`` narrows a server to named tools; prefer it for servers that
expose more than the task needs, and prefer ``https`` URLs with explicit
``headers`` auth over ad-hoc local proxies.
"""

from __future__ import annotations

import asyncio
import shlex
from collections.abc import Mapping
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Any

TRANSPORTS = ("stdio", "http", "sse")


@dataclass(frozen=True)
class MCPServerSpec:
    """One MCP server to mount for a solve: local stdio or remote HTTP.

    Exactly one of ``command`` (a local stdio server subprocess) or ``url``
    (a remote server) must be set. ``transport`` is inferred - ``stdio`` for
    commands, ``http`` (streamable HTTP, the current best practice) for URLs -
    and only needs to be spelled out as ``"sse"`` for remote servers still on
    the legacy HTTP+SSE transport. ``headers`` go on every HTTP request
    (authorization, API keys). Protocol-generation compatibility is the SDK's
    job, not the spec's: the client negotiates the version at initialize, so
    current stateless-HTTP servers and older stateful ones both work.

    ``allow`` is a tool-name allowlist (None mounts every tool the server
    lists); ``prefix`` namespaces the mounted names (``prefix="crm_"`` turns
    ``lookup`` into ``crm_lookup``) to avoid clashes between servers.
    """

    command: str | None = None
    args: tuple[str, ...] = ()
    env: Mapping[str, str] | None = None
    url: str | None = None
    headers: Mapping[str, str] | None = None
    transport: str | None = None
    allow: tuple[str, ...] | None = None
    prefix: str = ""

    def __post_init__(self) -> None:
        if bool(self.command) == bool(self.url):
            raise ValueError("an MCP server spec needs exactly one of command= or url=")
        transport = self.resolved_transport()
        if transport not in TRANSPORTS:
            raise ValueError(f"unknown MCP transport {transport!r}: choose one of {TRANSPORTS}")
        if transport == "stdio" and not self.command:
            raise ValueError("transport='stdio' needs command=")
        if transport in ("http", "sse") and not self.url:
            raise ValueError(f"transport={transport!r} needs url=")

    def resolved_transport(self) -> str:
        if self.transport:
            return self.transport
        return "stdio" if self.command else "http"

    @classmethod
    def parse(cls, spec: str) -> MCPServerSpec:
        """Build a spec from a CLI string.

        ``http(s)://...`` is a remote streamable-HTTP server; ``sse+http(s)://...``
        forces the legacy SSE transport for servers that have not migrated;
        anything else is a shell-split local stdio command line.
        """
        text = spec.strip()
        if text.startswith(("http://", "https://")):
            return cls(url=text)
        if text.startswith("sse+"):
            return cls(url=text.removeprefix("sse+"), transport="sse")
        parts = shlex.split(text)
        if not parts:
            raise ValueError("empty --mcp server spec")
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
    sessions close and stdio server subprocesses end. Wrappers carry the MCP
    tool's name and description so the model sees what the server declared.

    Protocol generations are handled by the SDK, not here: ``initialize``
    negotiates the version with the server, so servers on the current
    stateless-HTTP revision and servers still on older stateful revisions
    both work through the same code path.
    """
    _require_mcp()
    from mcp import ClientSession

    mounted: list[MountedTool] = []
    for spec in specs:
        read, write = await _open_transport(stack, spec)
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        listed = (await session.list_tools()).tools
        for tool in select_tools(listed, spec):
            mounted.append(_mount(session, tool, spec.prefix))
    return mounted


async def _open_transport(stack: AsyncExitStack, spec: MCPServerSpec):
    """Open the spec's transport on the stack; return its (read, write) pair.

    Every SDK transport yields the same stream pair, so the session layer
    above is transport-agnostic. Streamable HTTP is the default for URLs
    (current best practice); ``sse`` covers remote servers that have not yet
    migrated off the legacy HTTP+SSE transport; ``stdio`` runs a local
    subprocess.
    """
    transport = spec.resolved_transport()
    if transport == "stdio":
        from mcp import StdioServerParameters
        from mcp.client.stdio import stdio_client

        params = StdioServerParameters(
            command=spec.command,
            args=list(spec.args),
            env=dict(spec.env) if spec.env else None,
        )
        return await stack.enter_async_context(stdio_client(params))
    if transport == "http":
        from mcp.client.streamable_http import create_mcp_http_client, streamable_http_client

        http_client = await stack.enter_async_context(
            create_mcp_http_client(headers=dict(spec.headers) if spec.headers else None)
        )
        return await stack.enter_async_context(
            streamable_http_client(spec.url, http_client=http_client)
        )
    from mcp.client.sse import sse_client

    return await stack.enter_async_context(
        sse_client(spec.url, headers=dict(spec.headers) if spec.headers else None)
    )


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
