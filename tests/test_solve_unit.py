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


def test_solve_files_not_found_fails_before_model_resolution(tmp_path):
    """A missing input file raises immediately, before any Pi/model lookup."""
    import pytest

    with pytest.raises(FileNotFoundError, match="nope.pdf"):
        S.solve("read it", files=[tmp_path / "nope.pdf"])


def test_asolve_many_rejects_empty_instructions():
    import asyncio

    import pytest

    with pytest.raises(ValueError, match="non-empty"):
        asyncio.run(S.asolve_many([], "data"))


def test_json_default_handles_pydantic_and_fallback():
    from pydantic import BaseModel

    class Point(BaseModel):
        x: int

    assert S._json_default(Point(x=3)) == {"x": 3}
    assert S._json_default(object()).startswith("<object")


def test_env_float_parses_and_rejects(monkeypatch):
    monkeypatch.setenv("RRLM_MAX_COST", "0.25")
    assert S._env_float("RRLM_MAX_COST") == 0.25
    monkeypatch.setenv("RRLM_MAX_COST", "not-a-number")
    assert S._env_float("RRLM_MAX_COST") is None
    monkeypatch.delenv("RRLM_MAX_COST")
    assert S._env_float("RRLM_MAX_COST") is None


# --- CLI argument routing (solve itself is stubbed; parsing is real) -------- #
def _capture_solve_kwargs(monkeypatch, argv):
    captured = {}

    def fake_solve(*args, **kwargs):
        captured["args"] = args
        captured.update(kwargs)
        return {"answer": "ok", "error": None}

    monkeypatch.setattr(S, "solve", fake_solve)
    monkeypatch.setattr("sys.argv", argv)
    S.main()
    return captured


def test_cli_answer_type_json_maps_to_dict(monkeypatch):
    captured = _capture_solve_kwargs(
        monkeypatch, ["rrlm-solve", "-i", "q", "--answer-type", "json"]
    )
    assert captured["answer_type"] is dict


def test_cli_answer_type_list_maps_to_list_of_str(monkeypatch):
    captured = _capture_solve_kwargs(
        monkeypatch, ["rrlm-solve", "-i", "q", "--answer-type", "list"]
    )
    assert captured["answer_type"] == list[str]


def test_cli_max_cost_flag_and_env(monkeypatch):
    captured = _capture_solve_kwargs(
        monkeypatch, ["rrlm-solve", "-i", "q", "--max-cost", "0.10"]
    )
    assert captured["max_cost_usd"] == 0.10
    monkeypatch.setenv("RRLM_MAX_COST", "0.05")
    captured = _capture_solve_kwargs(monkeypatch, ["rrlm-solve", "-i", "q"])
    assert captured["max_cost_usd"] == 0.05


def test_cli_doctrine_file(monkeypatch, tmp_path):
    doc = tmp_path / "winner.txt"
    doc.write_text("Optimized doctrine text.", encoding="utf-8")
    captured = _capture_solve_kwargs(
        monkeypatch, ["rrlm-solve", "-i", "q", "--doctrine", str(doc)]
    )
    assert captured["doctrine"] == "Optimized doctrine text."


def test_cli_routes_multiple_instructions_to_solve_many(monkeypatch):
    captured = {}

    def fake_many(instructions, data, **kwargs):
        captured["instructions"] = instructions
        return {"answer": ["a", "b"], "answers": ["a", "b"], "error": None}

    monkeypatch.setattr(S, "solve_many", fake_many)
    monkeypatch.setattr("sys.argv", ["rrlm-solve", "-i", "q1", "-i", "q2"])
    S.main()
    assert captured["instructions"] == ["q1", "q2"]
