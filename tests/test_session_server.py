"""rrlm-session tests: the protocol unit-level, and the real server end to end.

The integration test drives the installed ``rrlm-session`` console script as a
genuine subprocess against the offline stub LM's ``session`` scenario: the
namespace counter must climb across protocol requests (1, 2), survive within
the process, drop back to 1 after ``reset``, and the server must keep serving
after a malformed line. That is the Pi extension's exact usage shape.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from rrlm.session_server import PROTOCOL, handle_request
from tests.conftest import write_stub_pi_config

DATA = "alpha\nbeta\ngamma\n"


class FakeSession:
    """Records calls; returns canned results. Unit tests only."""

    def __init__(self):
        self.calls: list = []
        self.resets = 0

    def solve(self, instruction, data="", **options):
        self.calls.append((instruction, data, options))
        return {"answer": "ok", "error": None}

    def reset(self):
        self.resets += 1


# --- unit: request dispatch --------------------------------------------------


def test_ping_reports_protocol():
    response = handle_request(FakeSession(), {"id": 7, "op": "ping"})
    assert response["id"] == 7
    assert response["result"]["protocol"] == PROTOCOL


def test_reset_dispatches():
    session = FakeSession()
    response = handle_request(session, {"id": 1, "op": "reset"})
    assert response["result"] == {"ok": True}
    assert session.resets == 1


def test_solve_defaults_to_op_solve_and_passes_data():
    session = FakeSession()
    response = handle_request(session, {"id": 2, "instruction": "count", "data": DATA})
    assert response["result"]["answer"] == "ok"
    assert session.calls == [("count", DATA, {})]


def test_solve_reads_data_file(tmp_path):
    payload = tmp_path / "d.txt"
    payload.write_text(DATA)
    session = FakeSession()
    handle_request(session, {"id": 3, "instruction": "count", "data_file": str(payload)})
    assert session.calls[0][1] == DATA


def test_solve_rejects_both_data_forms():
    response = handle_request(
        FakeSession(), {"id": 4, "instruction": "x", "data": "a", "data_file": "/b"}
    )
    assert "not both" in response["error"]


def test_solve_requires_instruction():
    response = handle_request(FakeSession(), {"id": 5, "data": "a"})
    assert "instruction" in response["error"]


def test_solve_maps_answer_type_names():
    session = FakeSession()
    handle_request(
        session, {"id": 6, "instruction": "n?", "options": {"answer_type": "int"}}
    )
    assert session.calls[0][2] == {"answer_type": int}


def test_solve_rejects_unknown_answer_type():
    response = handle_request(
        FakeSession(), {"id": 7, "instruction": "x", "options": {"answer_type": "uuid"}}
    )
    assert "unknown answer_type" in response["error"]


def test_unknown_op_is_an_error_response():
    response = handle_request(FakeSession(), {"id": 8, "op": "explode"})
    assert "unknown op" in response["error"]


def test_request_exception_becomes_error_response():
    class Broken(FakeSession):
        def solve(self, *a, **k):
            raise RuntimeError("boom")

    response = handle_request(Broken(), {"id": 9, "instruction": "x"})
    assert response["error"] == "RuntimeError: boom"


# --- integration: the real server over the real protocol ---------------------


class SessionProc:
    """Line-protocol driver for one rrlm-session subprocess."""

    def __init__(self, env: dict):
        exe = shutil.which("rrlm-session") or str(Path(sys.executable).parent / "rrlm-session")
        self.proc = subprocess.Popen(
            [exe], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, env=env,
        )
        self._next_id = 0

    def request(self, raw: dict | str) -> dict:
        line = raw if isinstance(raw, str) else json.dumps(raw)
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()
        out = self.proc.stdout.readline()
        assert out, f"server closed unexpectedly: {self.proc.stderr.read()}"
        return json.loads(out)

    def solve(self, instruction: str, data: str | None = None) -> dict:
        self._next_id += 1
        req: dict = {"id": self._next_id, "op": "solve", "instruction": instruction}
        if data is not None:
            req["data"] = data
        return self.request(req)


@pytest.fixture
def session_env(tmp_path, stub_base_url) -> dict:
    """A clean subprocess env whose Pi config points at the stub's session mode."""
    agent_dir = tmp_path / "agent"
    model = write_stub_pi_config(agent_dir, stub_base_url, "session")
    env = {k: v for k, v in os.environ.items() if not k.startswith("RRLM_")}
    env.pop("OPENROUTER_API_KEY", None)
    env["PI_CODING_AGENT_DIR"] = str(agent_dir)
    env["RRLM_MAIN"] = model
    return env


@pytest.mark.e2e
def test_session_server_end_to_end(session_env):
    server = SessionProc(session_env)
    try:
        assert server.request({"id": 0, "op": "ping"})["result"]["ok"] is True

        # The namespace counter is the persistence proof, through the protocol.
        first = server.solve("start a counter", DATA)
        second = server.solve("increment it")
        assert first["result"]["error"] is None
        assert (first["result"]["answer"], second["result"]["answer"]) == ("1", "2")

        # reset clears the namespace; the session keeps serving.
        assert server.request({"id": 90, "op": "reset"})["result"] == {"ok": True}
        assert server.solve("start over", DATA)["result"]["answer"] == "1"

        # A malformed line answers an error and does NOT desynchronize: the
        # next solve still sees the post-reset namespace (counter continues).
        bad = server.request("this is not json")
        assert bad["id"] is None and "bad request line" in bad["error"]
        assert server.solve("still alive", DATA)["result"]["answer"] == "2"

        # close: acknowledged, then a clean exit.
        assert server.request({"id": 99, "op": "close"})["result"] == {"ok": True}
        assert server.proc.wait(timeout=15) == 0
    finally:
        if server.proc.poll() is None:
            server.proc.kill()


@pytest.mark.e2e
def test_session_server_exits_cleanly_on_stdin_eof(session_env):
    """Host death = pipe EOF = automatic cleanup; nothing must linger."""
    server = SessionProc(session_env)
    server.proc.stdin.close()
    assert server.proc.wait(timeout=15) == 0
