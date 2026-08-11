"""A real, offline MCP server for rrlm integration tests (stdio transport).

Like ``stub_server.py``, this is not a mock: it is the official ``mcp`` SDK
serving over stdio in a genuine subprocess. Tests mount it through
``rrlm.mcptools`` and the agent's REPL awaits its tools for real; determinism
comes from the tools' trivial semantics, not from patching anything.

Tools: ``add(a, b)`` (exact arithmetic the test can verify end-to-end) and
``fail(reason)`` (always errors, to exercise the tool-error path).
"""

from __future__ import annotations

from mcp.server.mcpserver import MCPServer

server = MCPServer("rrlm-test-tools")


@server.tool(description="Add two integers and return the sum.")
def add(a: int, b: int) -> int:
    return a + b


@server.tool(description="Always fails; used to test error propagation.")
def fail(reason: str = "requested") -> str:
    raise RuntimeError(f"deliberate failure: {reason}")


if __name__ == "__main__":
    server.run(transport="stdio")
