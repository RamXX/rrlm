"""Unit tests for rrlm.solve internals: data resolution and cost reconciliation.

These are unit tests, so the heavy collaborators (model resolution, LM build, the
RLM run) are stubbed to isolate the function under test. The end-to-end behaviour
with no stubbing lives in test_integration_solve.py / test_e2e_cli.py.
"""

from __future__ import annotations

import importlib
import io
import types

import httpx
import respx

from rrlm.metrics import CallRecord
from rrlm.openrouter import GENERATION_URL

# Force the submodule (rrlm.solve), not the re-exported solve function.
S = importlib.import_module("rrlm.solve")


def test_read_data_none_returns_empty():
    assert S._read_data(None) == ""


def test_read_data_literal_passthrough():
    assert S._read_data("inline payload") == "inline payload"


def test_read_data_from_file(tmp_path):
    p = tmp_path / "payload.txt"
    p.write_text("file body")
    assert S._read_data(f"@{p}") == "file body"


def test_read_data_from_stdin(monkeypatch):
    monkeypatch.setattr("sys.stdin", io.StringIO("piped in"))
    assert S._read_data("-") == "piped in"


class _LM:
    def __init__(self):
        self.history: list = []


def _fake_model():
    return types.SimpleNamespace(
        ref="openrouter/x", is_local=False, supports_reasoning=False,
        max_tokens=4096, api_key="k",
    )


@respx.mock
def test_solve_reconciles_openrouter_gen_ids(monkeypatch):
    """When a call carries an OpenRouter ``gen-`` id, solve() reconciles its cost."""
    monkeypatch.setattr(S, "load_env", lambda: "or-key")
    monkeypatch.setattr(S, "resolve_model", lambda m=None: _fake_model())
    monkeypatch.setattr(S, "build_lm", lambda *a, **k: _LM())

    rec = CallRecord(role="main", gen_id="gen-xyz")
    # Return the record only once (for the main role) to avoid double-counting.
    monkeypatch.setattr(
        S, "harvest_lm_history",
        lambda lm, role, start=0: [rec] if role == "main" else [],
    )

    class _RLM:
        spawn_stats: dict = {}

        async def acall(self, *, task, data):
            return types.SimpleNamespace(answer="A", trace=None)

    monkeypatch.setattr(S, "build_rlm", lambda *a, **k: _RLM())
    respx.get(GENERATION_URL).mock(
        return_value=httpx.Response(200, json={"data": {"total_cost": 0.05}})
    )

    result = S.solve("task", "data")
    assert result["answer"] == "A"
    assert rec.cost_usd == 0.05  # reconcile() ran against the mocked endpoint
    assert abs(result["usage"]["cost_usd"] - 0.05) < 1e-9


def test_solve_skips_reconcile_for_local_gen_ids(monkeypatch):
    """Local/foreign gen ids never trigger an OpenRouter lookup (offline-safe)."""
    monkeypatch.setattr(S, "load_env", lambda: "")
    monkeypatch.setattr(S, "resolve_model", lambda m=None: _fake_model())
    monkeypatch.setattr(S, "build_lm", lambda *a, **k: _LM())
    rec = CallRecord(role="main", gen_id="chatcmpl-local-1")
    monkeypatch.setattr(
        S, "harvest_lm_history",
        lambda lm, role, start=0: [rec] if role == "main" else [],
    )

    class _RLM:
        spawn_stats: dict = {}

        async def acall(self, *, task, data):
            return types.SimpleNamespace(answer="B", trace=None)

    monkeypatch.setattr(S, "build_rlm", lambda *a, **k: _RLM())
    # No respx mock installed: if reconcile tried to hit the network this would
    # raise. It must not, because the gen id is not an OpenRouter one.
    result = S.solve("task", "data")
    assert result["answer"] == "B"
    assert rec.cost_usd is None
