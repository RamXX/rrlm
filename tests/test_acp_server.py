"""rrlm-acp tests: protocol dispatch unit-level, both backends, end to end.

The unit tests drive :class:`rrlm.acp_server.AcpServer` with an injected fake
session and a list-collecting writer, covering the v1 surface an ACP client
exercises: initialize, session/new, session/prompt with streamed updates,
session/cancel, and the error paths. The pi-backend tests run a real fake-pi
subprocess (``tests/fake_pi.py``) speaking the documented pi RPC protocol
over real pipes. The e2e tests run the installed ``rrlm-acp`` console script
as a genuine subprocess: harness mode against the offline stub LM's
``session`` scenario (namespace persistence across ACP turns), and pi mode
against fake-pi (conversation continuity in one long-lived process) - which
is Buzz's usage shape ("any command that speaks ACP over stdio").
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from rrlm.acp_server import (
    INTERNAL_ERROR,
    INVALID_PARAMS,
    METHOD_NOT_FOUND,
    PARSE_ERROR,
    PROTOCOL_VERSION,
    AcpServer,
    harness_agent_factory,
    pi_agent_factory,
    split_prompt,
)
from tests.conftest import write_stub_pi_config

DATA = "alpha\nbeta\ngamma\n"
FAKE_PI = str(Path(__file__).resolve().parent / "fake_pi.py")


class FakeSession:
    """Records calls; returns canned results and emits canned events."""

    def __init__(self, **defaults):
        self.defaults = defaults
        self.calls: list = []
        self.closed = False

    async def asolve(self, instruction, data="", **options):
        self.calls.append((instruction, data, options))
        on_event = options.get("on_event")
        if on_event:
            on_event({"event": "run_started", "instruction_chars": len(instruction)})
            on_event({"event": "llm_call", "role": "sub", "cost_usd": 0.01})
            on_event({"event": "run_finished", "error": None, "wall_clock_s": 0.5})
        return {"answer": "42", "error": None}

    def close(self):
        self.closed = True


def make_server(session_factory=FakeSession):
    sent: list[dict] = []
    factory = harness_agent_factory({"max_llm_calls": 7}, session_factory=session_factory)
    server = AcpServer(agent_factory=factory, send=sent.append)
    return server, sent


def make_pi_server():
    sent: list[dict] = []
    factory = pi_agent_factory([sys.executable, FAKE_PI])
    server = AcpServer(agent_factory=factory, send=sent.append, steering=True)
    return server, sent


async def open_session(server, sent) -> str:
    await server.handle_message({"jsonrpc": "2.0", "id": 0, "method": "initialize",
                                 "params": {"protocolVersion": PROTOCOL_VERSION}})
    await server.handle_message({"jsonrpc": "2.0", "id": 1, "method": "session/new",
                                 "params": {"cwd": os.getcwd(), "mcpServers": []}})
    return sent[-1]["result"]["sessionId"]


async def finish_prompt(server, session_id):
    task = server._sessions[session_id].prompt_task
    if task is not None:
        await task


def updates_of(sent) -> list[dict]:
    return [m["params"]["update"] for m in sent if m.get("method") == "session/update"]


def prompt_message(session_id, blocks, rid=2):
    return {"jsonrpc": "2.0", "id": rid, "method": "session/prompt",
            "params": {"sessionId": session_id, "prompt": blocks}}


# --- unit: handshake and dispatch --------------------------------------------


def test_initialize_negotiates_version_with_no_auth():
    server, sent = make_server()
    asyncio.run(server.handle_message(
        {"jsonrpc": "2.0", "id": 0, "method": "initialize",
         "params": {"protocolVersion": 99}}))
    result = sent[-1]["result"]
    assert result["protocolVersion"] == PROTOCOL_VERSION  # min(requested, ours)
    assert result["authMethods"] == []
    assert result["agentCapabilities"]["loadSession"] is False
    assert result["agentCapabilities"]["promptCapabilities"]["embeddedContext"] is True
    assert result["agentInfo"]["name"] == "rrlm"
    assert result["serverInfo"] == result["agentInfo"]
    assert "_meta" not in result  # the harness cannot steer; do not advertise

    asyncio.run(server.handle_message(
        {"jsonrpc": "2.0", "id": 1, "method": "initialize",
         "params": {"protocolVersion": 1}}))
    assert sent[-1]["result"]["protocolVersion"] == 1  # a v1 client stays v1


def test_pi_initialize_advertises_steering():
    server, sent = make_pi_server()
    asyncio.run(server.handle_message(
        {"jsonrpc": "2.0", "id": 0, "method": "initialize",
         "params": {"protocolVersion": 2}}))
    result = sent[-1]["result"]
    assert result["protocolVersion"] == 2
    assert result["_meta"] == {"steering": {"supported": True}}


def test_session_new_creates_session_with_defaults():
    server, sent = make_server()
    session_id = asyncio.run(open_session(server, sent))
    assert session_id.startswith("rrlm-")
    assert server._sessions[session_id].agent._session.defaults == {"max_llm_calls": 7}


def test_unknown_method_answers_method_not_found():
    server, sent = make_server()
    asyncio.run(server.handle_message({"jsonrpc": "2.0", "id": 5, "method": "session/load",
                                       "params": {}}))
    assert sent[-1]["error"]["code"] == METHOD_NOT_FOUND


def test_unknown_notification_is_silently_ignored():
    server, sent = make_server()
    asyncio.run(server.handle_message({"jsonrpc": "2.0", "method": "session/whatever",
                                       "params": {}}))
    assert sent == []


def test_malformed_line_answers_parse_error():
    server, sent = make_server()
    asyncio.run(server.handle_line("this is not json"))
    assert sent[-1]["id"] is None
    assert sent[-1]["error"]["code"] == PARSE_ERROR


# --- unit: the harness prompt turn -------------------------------------------


def test_prompt_streams_updates_then_answers_end_turn():
    server, sent = make_server()

    async def run():
        session_id = await open_session(server, sent)
        await server.handle_message(prompt_message(session_id, [
            {"type": "text", "text": "count the lines"},
            {"type": "resource", "resource": {"uri": "buzz://data", "text": DATA}},
        ]))
        await finish_prompt(server, session_id)
        return session_id

    session_id = asyncio.run(run())

    session = server._sessions[session_id].agent._session
    assert session.calls[0][0] == "count the lines"
    assert session.calls[0][1] == f"[buzz://data]\n{DATA}"

    updates = updates_of(sent)
    kinds = [u["sessionUpdate"] for u in updates]
    assert kinds[0] == "tool_call"
    assert "tool_call_update" in kinds
    assert kinds[-1] == "agent_message_chunk"
    assert updates[-1]["content"]["text"] == "42"
    # The completed tool_call_update carries the run summary.
    done = [u for u in updates if u["sessionUpdate"] == "tool_call_update"][-1]
    assert done["status"] == "completed"

    # The response arrives after every update, and ends the turn.
    assert sent[-1] == {"jsonrpc": "2.0", "id": 2, "result": {"stopReason": "end_turn"}}


def test_prompt_without_text_is_invalid_params():
    server, sent = make_server()

    async def run():
        session_id = await open_session(server, sent)
        await server.handle_message(prompt_message(session_id, [
            {"type": "resource", "resource": {"uri": "x://d", "text": DATA}}]))

    asyncio.run(run())
    assert sent[-1]["error"]["code"] == INVALID_PARAMS


def test_prompt_for_unknown_session_is_invalid_params():
    server, sent = make_server()
    asyncio.run(server.handle_message(prompt_message("rrlm-nope",
                                                     [{"type": "text", "text": "hi"}])))
    assert sent[-1]["error"]["code"] == INVALID_PARAMS


def test_failed_solve_reports_the_error_in_the_message():
    class Failing(FakeSession):
        async def asolve(self, instruction, data="", **options):
            return {"answer": None, "error": "budget exceeded"}

    server, sent = make_server(Failing)

    async def run():
        session_id = await open_session(server, sent)
        await server.handle_message(prompt_message(session_id,
                                                   [{"type": "text", "text": "go"}]))
        await finish_prompt(server, session_id)

    asyncio.run(run())
    assert "budget exceeded" in updates_of(sent)[-1]["content"]["text"]
    assert sent[-1]["result"] == {"stopReason": "end_turn"}


def test_solve_exception_becomes_internal_error_response():
    class Broken(FakeSession):
        async def asolve(self, *a, **k):
            raise RuntimeError("boom")

    server, sent = make_server(Broken)

    async def run():
        session_id = await open_session(server, sent)
        await server.handle_message(prompt_message(session_id,
                                                   [{"type": "text", "text": "go"}]))
        await finish_prompt(server, session_id)

    asyncio.run(run())
    assert sent[-1]["error"]["code"] == INTERNAL_ERROR
    assert "boom" in sent[-1]["error"]["message"]


def test_cancel_answers_the_pending_prompt_with_cancelled():
    class Hanging(FakeSession):
        async def asolve(self, *a, **k):
            await asyncio.Event().wait()

    server, sent = make_server(Hanging)

    async def run():
        session_id = await open_session(server, sent)
        await server.handle_message(prompt_message(session_id,
                                                   [{"type": "text", "text": "go"}]))
        await asyncio.sleep(0)  # let the turn start hanging
        await server.handle_message({"jsonrpc": "2.0", "method": "session/cancel",
                                     "params": {"sessionId": session_id}})
        await finish_prompt(server, session_id)

    asyncio.run(run())
    assert sent[-1] == {"jsonrpc": "2.0", "id": 2, "result": {"stopReason": "cancelled"}}


def test_aclose_cancels_and_releases_sessions():
    server, sent = make_server()

    async def run():
        session_id = await open_session(server, sent)
        agent = server._sessions[session_id].agent
        await server.aclose()
        return agent

    agent = asyncio.run(run())
    assert server._sessions == {}
    assert agent._session.closed is True


# --- unit: content block splitting -------------------------------------------


def test_split_prompt_reads_local_resource_links(tmp_path):
    payload = tmp_path / "d.txt"
    payload.write_text(DATA)
    instruction, data = split_prompt([
        {"type": "text", "text": "count"},
        {"type": "resource_link", "uri": payload.as_uri(), "name": "d.txt"},
    ])
    assert instruction == "count"
    assert data == f"[{payload.as_uri()}]\n{DATA}"


def test_split_prompt_notes_unfetchable_links():
    instruction, data = split_prompt([
        {"type": "text", "text": "count"},
        {"type": "resource_link", "uri": "https://example.com/x"},
    ])
    assert "not fetched" in instruction
    assert data == ""


def test_split_prompt_ignores_unknown_blocks():
    instruction, data = split_prompt([
        {"type": "image", "data": "...", "mimeType": "image/png"},
        {"type": "text", "text": "hello"},
    ])
    assert (instruction, data) == ("hello", "")


# --- integration: the pi backend against a real fake-pi subprocess ------------


@pytest.mark.integration
def test_pi_turns_share_one_process_and_stream_tools():
    server, sent = make_pi_server()

    async def run():
        session_id = await open_session(server, sent)
        await server.handle_message(prompt_message(session_id,
                                                   [{"type": "text", "text": "hello"}], rid=2))
        await finish_prompt(server, session_id)
        await server.handle_message(prompt_message(session_id,
                                                   [{"type": "text", "text": "again"}], rid=3))
        await finish_prompt(server, session_id)
        await server.aclose()

    asyncio.run(run())
    updates = updates_of(sent)
    chunks = [u["content"]["text"] for u in updates
              if u["sessionUpdate"] == "agent_message_chunk"]
    # turn-N counts prompts inside ONE fake-pi process: the conversation
    # (unlike the harness) lives in the subprocess and survives across turns.
    assert chunks == ["turn-1", "turn-2"]
    tool_calls = [u for u in updates if u["sessionUpdate"] == "tool_call"]
    assert tool_calls and tool_calls[0]["kind"] == "execute"
    assert "bash" in tool_calls[0]["title"]
    # The tool's details.summary (usage metrics) rides in the completed update.
    done = [u for u in updates if u["sessionUpdate"] == "tool_call_update"
            and u.get("status") == "completed"]
    assert "calls=3 tokens=10+5" in done[0]["content"][0]["content"]["text"]
    ends = [m["result"]["stopReason"] for m in sent if "result" in m and "stopReason"
            in m.get("result", {})]
    assert ends == ["end_turn", "end_turn"]


@pytest.mark.integration
def test_pi_stages_embedded_resources_to_files():
    server, sent = make_pi_server()

    async def run():
        session_id = await open_session(server, sent)
        await server.handle_message(prompt_message(session_id, [
            {"type": "text", "text": "use the data"},
            {"type": "resource", "resource": {"uri": "buzz://data", "text": DATA}},
        ]))
        await finish_prompt(server, session_id)
        await server.aclose()

    asyncio.run(run())
    text = "".join(u["content"]["text"] for u in updates_of(sent)
                   if u["sessionUpdate"] == "agent_message_chunk")
    # fake-pi read the staged file and reported its length: the data traveled
    # by file, not inline through pi's context.
    staged = f"[buzz://data]\n{DATA}"
    assert f"data-len={len(staged)}" in text


@pytest.mark.integration
def test_pi_cancel_aborts_and_answers_cancelled():
    server, sent = make_pi_server()

    async def run():
        session_id = await open_session(server, sent)
        await server.handle_message(prompt_message(session_id,
                                                   [{"type": "text", "text": "HANG"}]))
        await asyncio.sleep(0.2)  # let fake-pi accept the prompt and start hanging
        await server.handle_message({"jsonrpc": "2.0", "method": "session/cancel",
                                     "params": {"sessionId": session_id}})
        await finish_prompt(server, session_id)
        await server.aclose()

    asyncio.run(run())
    assert sent[-1] == {"jsonrpc": "2.0", "id": 2, "result": {"stopReason": "cancelled"}}


@pytest.mark.integration
def test_pi_ui_dialogs_are_auto_cancelled():
    server, sent = make_pi_server()

    async def run():
        session_id = await open_session(server, sent)
        await server.handle_message(prompt_message(session_id,
                                                   [{"type": "text", "text": "DIALOG"}]))
        await finish_prompt(server, session_id)
        await server.aclose()

    asyncio.run(run())
    chunks = [u["content"]["text"] for u in updates_of(sent)
              if u["sessionUpdate"] == "agent_message_chunk"]
    # fake-pi blocked on a select dialog; the bridge answered cancelled=true
    # on its own, so the turn completed with the documented default.
    assert chunks == ["cancelled"]
    assert sent[-1]["result"] == {"stopReason": "end_turn"}


# --- model listing and switching (the Buzz model-picker probe) ----------------


@pytest.mark.integration
def test_pi_session_new_advertises_models_both_ways():
    server, sent = make_pi_server()

    async def run():
        session_id = await open_session(server, sent)
        await server.aclose()
        return session_id

    asyncio.run(run())
    result = sent[1]["result"]
    option = result["configOptions"][0]
    assert option["category"] == "model" and option["type"] == "select"
    assert [o["value"] for o in option["options"]] == ["stub/model-a", "stub/model-b"]
    assert option["options"][0]["displayName"] == "Stub A"
    assert option["currentValue"] == "stub/model-a"
    assert result["models"] == {
        "availableModels": [{"modelId": "stub/model-a", "name": "Stub A"},
                            {"modelId": "stub/model-b", "name": "Stub B"}],
        "currentModelId": "stub/model-a",
    }


@pytest.mark.integration
def test_pi_set_config_option_switches_the_model():
    server, sent = make_pi_server()

    async def run():
        session_id = await open_session(server, sent)
        await server.handle_message({"jsonrpc": "2.0", "id": 2,
                                     "method": "session/set_config_option",
                                     "params": {"sessionId": session_id,
                                                "configId": "model",
                                                "value": "stub/model-b"}})
        await server.handle_message({"jsonrpc": "2.0", "id": 3,
                                     "method": "session/set_model",
                                     "params": {"sessionId": session_id,
                                                "modelId": "stub/model-a"}})
        await server.handle_message({"jsonrpc": "2.0", "id": 4,
                                     "method": "session/set_model",
                                     "params": {"sessionId": session_id,
                                                "modelId": "stub/nope"}})
        await server.aclose()

    asyncio.run(run())
    stable = next(m for m in sent if m.get("id") == 2)
    assert stable["result"]["configOptions"][0]["currentValue"] == "stub/model-b"
    legacy = next(m for m in sent if m.get("id") == 3)
    assert legacy["result"] == {"sessionId": legacy["result"]["sessionId"],
                                "modelId": "stub/model-a"}
    unknown = next(m for m in sent if m.get("id") == 4)
    assert unknown["error"]["code"] == INVALID_PARAMS


def test_harness_lists_models_from_pi_config(tmp_path, monkeypatch):
    agent_dir = tmp_path / "agent"
    write_stub_pi_config(agent_dir, "http://127.0.0.1:1", "submit")
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(agent_dir))
    server, sent = make_server()

    async def run():
        session_id = await open_session(server, sent)
        await server.handle_message({"jsonrpc": "2.0", "id": 2,
                                     "method": "session/set_config_option",
                                     "params": {"sessionId": session_id,
                                                "configId": "model",
                                                "value": "stub/stub-model"}})
        await server.handle_message(prompt_message(session_id,
                                                   [{"type": "text", "text": "go"}], rid=3))
        await finish_prompt(server, session_id)
        return session_id

    session_id = asyncio.run(run())
    result = sent[1]["result"]
    assert result["models"]["availableModels"] == [
        {"modelId": "stub/stub-model", "name": "stub/stub-model"}]
    # The switch rides into subsequent solves as a main_model override.
    session = server._sessions[session_id].agent._session
    assert session.calls[0][2]["main_model"] == "stub/stub-model"


def test_harness_session_new_survives_empty_pi_config(tmp_path, monkeypatch):
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(tmp_path / "nowhere"))
    server, sent = make_server()
    asyncio.run(open_session(server, sent))
    result = sent[1]["result"]
    assert "sessionId" in result
    assert "configOptions" not in result and "models" not in result


# --- buzz-contract behaviors: steering, usage, system prompt, cancel ack ------


@pytest.mark.integration
def test_pi_steering_injects_into_the_running_turn():
    server, sent = make_pi_server()

    async def run():
        session_id = await open_session(server, sent)
        await server.handle_message(prompt_message(session_id,
                                                   [{"type": "text", "text": "HANG"}]))
        await asyncio.sleep(0.2)  # let fake-pi accept the prompt and start hanging
        await server.handle_message({"jsonrpc": "2.0", "id": 5,
                                     "method": "_session/steering",
                                     "params": {"sessionId": session_id, "prompt": [
                                         {"type": "text", "text": "also do this"}]}})
        await server.handle_message({"jsonrpc": "2.0", "id": 6, "method": "session/cancel",
                                     "params": {"sessionId": session_id}})
        await finish_prompt(server, session_id)
        await server.aclose()

    asyncio.run(run())
    steer = next(m for m in sent if m.get("id") == 5)
    assert steer["result"] == {"outcome": "injected"}
    # cancel sent as a request gets an explicit null ack, and the prompt
    # still resolves cancelled.
    cancel_ack = next(m for m in sent if m.get("id") == 6)
    assert cancel_ack["result"] is None
    assert sent[-1]["result"] == {"stopReason": "cancelled"}


def test_steering_without_a_running_turn_is_invalid_params():
    server, sent = make_server()

    async def run():
        session_id = await open_session(server, sent)
        await server.handle_message({"jsonrpc": "2.0", "id": 5,
                                     "method": "_session/steering",
                                     "params": {"sessionId": session_id, "prompt": [
                                         {"type": "text", "text": "hello"}]}})

    asyncio.run(run())
    assert sent[-1]["error"]["code"] == INVALID_PARAMS


@pytest.mark.integration
def test_pi_usage_accumulates_and_rides_before_the_response():
    server, sent = make_pi_server()

    async def run():
        session_id = await open_session(server, sent)
        for rid in (2, 3):
            await server.handle_message(prompt_message(session_id,
                                                       [{"type": "text", "text": "hi"}],
                                                       rid=rid))
            await finish_prompt(server, session_id)
        await server.aclose()

    asyncio.run(run())
    usage = [m["params"]["update"] for m in sent
             if m.get("method") == "_goose/unstable/session/update"]
    assert [u["accumulatedInputTokens"] for u in usage] == [10, 20]
    assert usage[-1]["accumulatedTotalTokens"] == 30
    assert usage[-1]["accumulatedCost"] == 0.02
    # Ordering is load-bearing: usage before the prompt response, each turn.
    kinds = [("usage" if m.get("method") == "_goose/unstable/session/update" else
              "response" if "result" in m and "stopReason" in m.get("result", {}) else None)
             for m in sent]
    kinds = [k for k in kinds if k]
    assert kinds == ["usage", "response", "usage", "response"]


@pytest.mark.integration
def test_pi_receives_the_session_system_prompt():
    server, sent = make_pi_server()

    async def run():
        await server.handle_message({"jsonrpc": "2.0", "id": 0, "method": "initialize",
                                     "params": {"protocolVersion": 2}})
        await server.handle_message({"jsonrpc": "2.0", "id": 1, "method": "session/new",
                                     "params": {"cwd": os.getcwd(), "mcpServers": [],
                                                "systemPrompt": "You are the rrlm agent."}})
        session_id = sent[-1]["result"]["sessionId"]
        agent = server._sessions[session_id].agent
        await server.aclose()
        return agent

    agent = asyncio.run(run())
    flag = agent._argv.index("--append-system-prompt")
    staged = Path(agent._argv[flag + 1])
    # The prompt is staged to a file (argv cannot carry Buzz's 512KB cap)
    # and the file rides the spawn argv; aclose removed the staging dir.
    assert staged.name == "system-prompt.md"


def test_keepalive_updates_flow_while_a_turn_hangs(monkeypatch):
    import rrlm.acp_server as acp

    monkeypatch.setattr(acp, "KEEPALIVE_INTERVAL_S", 0.03)

    class Hanging(FakeSession):
        async def asolve(self, *a, **k):
            await asyncio.Event().wait()

    server, sent = make_server(Hanging)

    async def run():
        session_id = await open_session(server, sent)
        await server.handle_message(prompt_message(session_id,
                                                   [{"type": "text", "text": "go"}]))
        await asyncio.sleep(0.2)
        await server.handle_message({"jsonrpc": "2.0", "method": "session/cancel",
                                     "params": {"sessionId": session_id}})
        await finish_prompt(server, session_id)

    asyncio.run(run())
    keepalives = [u for u in updates_of(sent) if u == {"sessionUpdate": "keepalive"}]
    assert len(keepalives) >= 2


# --- e2e: the real console script over the real protocol ----------------------


class AcpProc:
    """NDJSON JSON-RPC driver for one rrlm-acp subprocess."""

    def __init__(self, env: dict, extra_args: list[str] | None = None):
        exe = shutil.which("rrlm-acp") or str(Path(sys.executable).parent / "rrlm-acp")
        self.proc = subprocess.Popen(
            [exe, *(extra_args or [])], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, env=env,
        )
        self._next_id = 0
        self.updates: list[dict] = []

    def request(self, method: str, params: dict) -> dict:
        """Send one request; collect updates until its response arrives."""
        self._next_id += 1
        rid = self._next_id
        self._write({"jsonrpc": "2.0", "id": rid, "method": method, "params": params})
        while True:
            message = self._read()
            if message.get("id") == rid:
                return message
            if message.get("method") == "session/update":
                self.updates.append(message["params"])

    def _write(self, message: dict) -> None:
        self.proc.stdin.write(json.dumps(message) + "\n")
        self.proc.stdin.flush()

    def _read(self) -> dict:
        line = self.proc.stdout.readline()
        assert line, f"agent closed unexpectedly: {self.proc.stderr.read()}"
        return json.loads(line)

    def message_chunks(self) -> list[str]:
        return [u["update"]["content"]["text"] for u in self.updates
                if u["update"]["sessionUpdate"] == "agent_message_chunk"]


def clean_env() -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith("RRLM_")}
    env.pop("OPENROUTER_API_KEY", None)
    return env


@pytest.fixture
def acp_env(tmp_path, stub_base_url) -> dict:
    """A clean subprocess env whose Pi config points at the stub's session mode."""
    agent_dir = tmp_path / "agent"
    model = write_stub_pi_config(agent_dir, stub_base_url, "session")
    env = clean_env()
    env["PI_CODING_AGENT_DIR"] = str(agent_dir)
    env["RRLM_MAIN"] = model
    return env


@pytest.mark.e2e
def test_acp_agent_end_to_end(acp_env, tmp_path):
    agent = AcpProc(acp_env)
    try:
        init = agent.request("initialize", {"protocolVersion": 1,
                                            "clientCapabilities": {}})
        assert init["result"]["protocolVersion"] == 1

        new = agent.request("session/new", {"cwd": str(tmp_path), "mcpServers": []})
        session_id = new["result"]["sessionId"]

        # Turn 1: instruction + embedded data. The namespace counter is the
        # persistence proof, through the ACP protocol.
        first = agent.request("session/prompt", {"sessionId": session_id, "prompt": [
            {"type": "text", "text": "start a counter"},
            {"type": "resource", "resource": {"uri": "buzz://data", "text": DATA}},
        ]})
        assert first["result"]["stopReason"] == "end_turn"

        # Turn 2: no data; the session namespace carries it.
        second = agent.request("session/prompt", {"sessionId": session_id, "prompt": [
            {"type": "text", "text": "increment it"},
        ]})
        assert second["result"]["stopReason"] == "end_turn"
        assert agent.message_chunks() == ["1", "2"]

        # Progress streamed as one tool_call per turn, completed at the end.
        tool_calls = [u["update"] for u in agent.updates
                      if u["update"]["sessionUpdate"] == "tool_call"]
        assert len(tool_calls) == 2
        completed = [u["update"] for u in agent.updates
                     if u["update"]["sessionUpdate"] == "tool_call_update"
                     and u["update"]["status"] == "completed"]
        assert len(completed) == 2

        # A malformed line answers a parse error and does not desynchronize.
        agent.proc.stdin.write("this is not json\n")
        agent.proc.stdin.flush()
        bad = agent._read()
        assert bad["id"] is None and bad["error"]["code"] == PARSE_ERROR

        third = agent.request("session/prompt", {"sessionId": session_id, "prompt": [
            {"type": "text", "text": "still alive"},
        ]})
        assert third["result"]["stopReason"] == "end_turn"

        # stdin EOF: everything tears down, exit 0, nothing lingers.
        agent.proc.stdin.close()
        assert agent.proc.wait(timeout=15) == 0
    finally:
        if agent.proc.poll() is None:
            agent.proc.kill()


@pytest.mark.e2e
def test_acp_pi_mode_end_to_end(tmp_path):
    """--pi over the console script, with fake-pi standing in for pi."""
    agent = AcpProc(clean_env(),
                    ["--pi", "--pi-command", f"{sys.executable} {FAKE_PI}"])
    try:
        assert agent.request("initialize",
                             {"protocolVersion": 1})["result"]["protocolVersion"] == 1
        session_id = agent.request("session/new", {"cwd": str(tmp_path),
                                                   "mcpServers": []})["result"]["sessionId"]
        first = agent.request("session/prompt", {"sessionId": session_id, "prompt": [
            {"type": "text", "text": "hello"}]})
        second = agent.request("session/prompt", {"sessionId": session_id, "prompt": [
            {"type": "text", "text": "again"}]})
        assert (first["result"], second["result"]) == ({"stopReason": "end_turn"},) * 2
        assert agent.message_chunks() == ["turn-1", "turn-2"]

        agent.proc.stdin.close()
        assert agent.proc.wait(timeout=15) == 0
    finally:
        if agent.proc.poll() is None:
            agent.proc.kill()
