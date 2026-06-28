"""Unit tests for rrlm.solve guardrails: hard wall-clock timeout + cap threading.
Pure unit tests -- the heavy deps (model resolution, LM build, RLM run) are stubbed,
so no model/sandbox is needed."""
from __future__ import annotations

import asyncio
import importlib
import types

# `import rrlm.solve as S` would bind the re-exported `solve` *function* (rrlm/__init__
# exports it), not the module -- force the submodule so monkeypatch targets module globals.
S = importlib.import_module("rrlm.solve")


class _LM:
    def __init__(self):
        self.history: list = []

    def __call__(self, *a, **k):
        return ["x"]


def _fake_model():
    return types.SimpleNamespace(
        ref="local/x", is_local=True, supports_reasoning=False, max_tokens=4096, api_key="local"
    )


def _patch_common(monkeypatch):
    monkeypatch.setattr(S, "load_env", lambda: "")
    monkeypatch.setattr(S, "resolve_model", lambda m=None: _fake_model())
    monkeypatch.setattr(S, "build_lm", lambda *a, **k: _LM())
    monkeypatch.setattr(S, "harvest_lm_history", lambda *a, **k: [])
    monkeypatch.setattr(S, "summarize", lambda recs: {})


def test_timeout_aborts(monkeypatch):
    _patch_common(monkeypatch)

    class _SlowRLM:
        spawn_stats: dict = {}

        async def acall(self, *, task, data):
            await asyncio.sleep(5)  # far longer than the timeout
            return types.SimpleNamespace(answer="late", trace=None)

    monkeypatch.setattr(S, "build_rlm", lambda *a, **k: _SlowRLM())
    r = S.solve("task", "data", timeout_s=0.3)
    assert r["answer"] == ""
    assert r["error"] and "Timeout" in r["error"]
    assert r["wall_clock_s"] < 3  # aborted promptly, didn't wait the full 5s


def test_no_timeout_completes(monkeypatch):
    _patch_common(monkeypatch)

    class _FastRLM:
        spawn_stats: dict = {}

        async def acall(self, *, task, data):
            return types.SimpleNamespace(answer="B", trace=None)

    monkeypatch.setattr(S, "build_rlm", lambda *a, **k: _FastRLM())
    r = S.solve("task", "data", timeout_s=10)
    assert r["error"] is None and r["answer"] == "B"


def test_caps_thread_into_config(monkeypatch):
    _patch_common(monkeypatch)
    captured: dict = {}

    class _FastRLM:
        spawn_stats: dict = {}

        async def acall(self, *, task, data):
            return types.SimpleNamespace(answer="ok", trace=None)

    def fake_build_rlm(cfg, *a, **k):
        captured["max_llm_calls"] = cfg.max_llm_calls
        captured["max_iterations"] = cfg.max_iterations
        return _FastRLM()

    monkeypatch.setattr(S, "build_rlm", fake_build_rlm)
    S.solve("t", "d", max_llm_calls=7, max_iterations=9)
    assert captured == {"max_llm_calls": 7, "max_iterations": 9}
