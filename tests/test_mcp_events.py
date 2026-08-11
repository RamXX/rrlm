"""MCP tool mounting and structured progress events: unit + integration.

The MCP integration tests run the full chain with no mocks: the stub LM
returns canned REPL code that awaits an MCP tool, the tool executes in a real
MCP server subprocess (``tests/mcp_stub_server.py``, official SDK, stdio),
and the answer comes back through predict-rlm's host-tool bridge. Event tests
observe the callback stream on both the native and the engine path.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from rrlm import solve
from rrlm.events import EventEmitter
from rrlm.mcptools import MCPServerSpec, extract_text, mcp_tools_note, select_tools
from rrlm.solve import asolve

TESTS_DIR = Path(__file__).resolve().parent
MCP_SERVER = MCPServerSpec(command=sys.executable, args=(str(TESTS_DIR / "mcp_stub_server.py"),))
DATA = "alpha\nbeta\ngamma\n"


# --- unit: specs, selection, extraction -------------------------------------


def test_spec_parse_shell_splits():
    spec = MCPServerSpec.parse("uvx some-server --flag 'a b'")
    assert spec.command == "uvx"
    assert spec.args == ("some-server", "--flag", "a b")


def test_spec_parse_rejects_empty():
    with pytest.raises(ValueError, match="empty"):
        MCPServerSpec.parse("   ")


def _tool(name: str):
    return SimpleNamespace(name=name, description=f"{name} tool")


def test_select_tools_defaults_to_all():
    tools = [_tool("a"), _tool("b")]
    assert select_tools(tools, MCPServerSpec(command="x")) == tools


def test_select_tools_applies_allowlist_in_order():
    tools = [_tool("a"), _tool("b"), _tool("c")]
    picked = select_tools(tools, MCPServerSpec(command="x", allow=("c", "a")))
    assert [t.name for t in picked] == ["c", "a"]


def test_select_tools_missing_name_fails_loudly():
    with pytest.raises(ValueError, match="does not expose ghost"):
        select_tools([_tool("a")], MCPServerSpec(command="x", allow=("ghost",)))


def test_extract_text_joins_content():
    result = SimpleNamespace(
        content=[SimpleNamespace(text="one"), SimpleNamespace(text="two")], isError=False
    )
    assert extract_text(result) == "one\ntwo"


def test_extract_text_raises_on_tool_error():
    result = SimpleNamespace(content=[SimpleNamespace(text="boom")], isError=True)
    with pytest.raises(RuntimeError, match="boom"):
        extract_text(result)


def test_mcp_tools_note_lists_every_tool():
    from rrlm.mcptools import MountedTool

    note = mcp_tools_note(
        [MountedTool(name="add", description="Add ints.", call=None)]
    )
    assert "await" in note and "- add: Add ints." in note


# --- unit: events -----------------------------------------------------------


def test_emitter_isolates_run_from_broken_callback():
    def broken(event):
        raise RuntimeError("display bug")

    EventEmitter(broken).emit("run_started")  # must not raise


def test_engine_path_emits_run_events():
    events: list[dict] = []
    result = asyncio.run(
        asolve("count-lines", DATA, engine="reference", on_event=events.append)
    )
    assert result["answer"] == 3
    names = [e["event"] for e in events]
    assert names == ["run_started", "run_finished"]
    assert events[0]["engine"] == "reference"
    assert events[1]["error"] is None


def test_engine_rejects_mcp():
    with pytest.raises(ValueError, match="mutually exclusive"):
        asyncio.run(asolve("x", DATA, engine="reference", mcp=[MCP_SERVER]))


# --- integration: real MCP server, real stack --------------------------------


@pytest.mark.integration
def test_mcp_tool_call_end_to_end(stub_model):
    model = stub_model("mcptool")
    result = solve(
        "add the numbers with the mounted tool", DATA,
        main_model=model, backend="supervisor", max_iterations=5, mcp=[MCP_SERVER],
    )
    assert result["error"] is None
    assert result["answer"] == "42"  # add(19, 23) computed in the MCP server


@pytest.mark.integration
def test_mcp_allowlist_missing_tool_fails_the_run(stub_model):
    model = stub_model("mcptool")
    spec = MCPServerSpec(
        command=MCP_SERVER.command, args=MCP_SERVER.args, allow=("no_such_tool",)
    )
    result = solve(
        "irrelevant", DATA,
        main_model=model, backend="supervisor", max_iterations=5, mcp=[spec],
    )
    assert result["error"] is not None and "does not expose" in result["error"]


@pytest.mark.integration
def test_native_run_emits_llm_call_events(stub_model):
    model = stub_model("submit")
    events: list[dict] = []
    result = solve(
        "compute", DATA,
        main_model=model, backend="supervisor", max_iterations=5,
        on_event=events.append,
    )
    assert result["error"] is None
    names = [e["event"] for e in events]
    assert names[0] == "run_started" and names[-1] == "run_finished"
    calls = [e for e in events if e["event"] == "llm_call"]
    assert calls and all(c["role"] in ("main", "sub") for c in calls)
    assert events[0]["backend"] == "supervisor"


@pytest.mark.integration
def test_spawn_emits_spawn_events(stub_model):
    model = stub_model("spawn")
    events: list[dict] = []
    result = solve(
        "delegate", DATA,
        main_model=model, backend="supervisor", max_depth=1, max_iterations=5,
        on_event=events.append,
    )
    assert result["error"] is None
    names = [e["event"] for e in events]
    assert "spawn_started" in names and "spawn_finished" in names
    assert names.index("spawn_started") < names.index("spawn_finished")
