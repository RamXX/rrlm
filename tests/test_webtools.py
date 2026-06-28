"""Tests for rrlm.webtools and its wiring into the harness and CLI.

Fully offline and deterministic: the optional 'web' extra (ddgs + trafilatura) is
not installed in CI, so the search backend is faked and HTTP is mocked with respx.
No network, no mocks inside the tool's own logic (only its external collaborators
are stubbed), so these are honest unit tests of the formatting/IO behaviour.
"""

from __future__ import annotations

import asyncio
import importlib
import sys
import types

import httpx
import respx

from rrlm.config import HarnessConfig
from rrlm.harness import build_lm, build_rlm
from rrlm.pi_config import ResolvedModel

W = importlib.import_module("rrlm.webtools")
S = importlib.import_module("rrlm.solve")


# --- helpers --------------------------------------------------------------- #
def _fake_ddgs_cls(hits=None, raise_exc=None):
    """A DDGS-shaped context manager class returning canned hits."""

    class _D:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def text(self, query, max_results=5):
            if raise_exc is not None:
                raise raise_exc
            return list(hits or [])

    return _D


def _inject_trafilatura(monkeypatch, extract):
    mod = types.ModuleType("trafilatura")
    mod.extract = extract
    monkeypatch.setitem(sys.modules, "trafilatura", mod)


def _model(max_tokens: int = 65_536, **kw) -> ResolvedModel:
    return ResolvedModel(
        ref="test/model", provider="openrouter", model_id="test/model",
        litellm_id="openrouter/test/model", api_key="sk-or-test",
        max_tokens=max_tokens, reasoning_style="openrouter", **kw,
    )


# --- web_tools / web_available --------------------------------------------- #
def test_web_tools_returns_named_callables():
    tools = W.web_tools()
    names = {fn.__name__ for fn in tools}
    assert names == {"web_search", "fetch"}


def test_web_available_false_without_ddgs(monkeypatch):
    monkeypatch.setattr(W, "_import_ddgs", lambda: None)
    assert W.web_available() is False


def test_web_available_true_when_both_present(monkeypatch):
    monkeypatch.setattr(W, "_import_ddgs", lambda: _fake_ddgs_cls())
    _inject_trafilatura(monkeypatch, lambda html, **k: "x")
    assert W.web_available() is True


# --- web_search ------------------------------------------------------------ #
def test_web_search_empty_query():
    assert asyncio.run(W.web_search("   ")) == "web_search: empty query"


def test_web_search_missing_extra(monkeypatch):
    monkeypatch.setattr(W, "_import_ddgs", lambda: None)
    out = asyncio.run(W.web_search("anything"))
    assert "optional 'web' extra" in out


def test_web_search_formats_hits(monkeypatch):
    hits = [
        {"title": "Paris", "href": "https://en.wikipedia.org/wiki/Paris",
         "body": "Paris is the capital of France."},
        {"title": "France", "url": "https://example.com/france", "body": "x" * 500},
    ]
    monkeypatch.setattr(W, "_import_ddgs", lambda: _fake_ddgs_cls(hits=hits))
    out = asyncio.run(W.web_search("capital of France", max_results=2))
    assert out.startswith("1. Paris")
    assert "https://en.wikipedia.org/wiki/Paris" in out
    assert "2. France" in out
    # the second result's url is read from 'url' when 'href' is absent
    assert "https://example.com/france" in out
    # snippet is truncated to 300 chars
    assert "x" * 300 in out and "x" * 301 not in out


def test_web_search_no_results(monkeypatch):
    monkeypatch.setattr(W, "_import_ddgs", lambda: _fake_ddgs_cls(hits=[]))
    out = asyncio.run(W.web_search("zxqw"))
    assert "no results" in out


def test_web_search_clamps_max_results(monkeypatch):
    seen = {}

    class _D:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def text(self, query, max_results=5):
            seen["n"] = max_results
            return [{"title": "t", "href": "u", "body": "b"}]

    monkeypatch.setattr(W, "_import_ddgs", lambda: _D)
    asyncio.run(W.web_search("q", max_results=999))
    assert seen["n"] == 10  # clamped to the [1, 10] window


def test_web_search_handles_backend_exception(monkeypatch):
    monkeypatch.setattr(
        W, "_import_ddgs", lambda: _fake_ddgs_cls(raise_exc=RuntimeError("rate limited"))
    )
    out = asyncio.run(W.web_search("q"))
    assert out.startswith("web_search error: RuntimeError")


# --- fetch ----------------------------------------------------------------- #
def test_fetch_invalid_url():
    out = asyncio.run(W.fetch("not-a-url"))
    assert out.startswith("fetch: invalid url")


@respx.mock
def test_fetch_happy_path(monkeypatch):
    respx.get("https://example.com/a").mock(
        return_value=httpx.Response(200, html="<html><body><p>hi</p></body></html>")
    )
    _inject_trafilatura(monkeypatch, lambda html, **k: "Clean article text.")
    out = asyncio.run(W.fetch("https://example.com/a"))
    assert out == "Clean article text."


@respx.mock
def test_fetch_truncates(monkeypatch):
    respx.get("https://example.com/long").mock(
        return_value=httpx.Response(200, html="<html></html>")
    )
    _inject_trafilatura(monkeypatch, lambda html, **k: "y" * 20_000)
    out = asyncio.run(W.fetch("https://example.com/long", max_chars=1000))
    assert "truncated at 1000 chars" in out
    assert out.count("y") == 1000


@respx.mock
def test_fetch_http_error(monkeypatch):
    respx.get("https://example.com/boom").mock(return_value=httpx.Response(500))
    _inject_trafilatura(monkeypatch, lambda html, **k: "never reached")
    out = asyncio.run(W.fetch("https://example.com/boom"))
    assert out.startswith("fetch error:")


@respx.mock
def test_fetch_falls_back_to_tag_strip(monkeypatch):
    respx.get("https://example.com/b").mock(
        return_value=httpx.Response(200, html="<html><body><p>Raw body text</p></body></html>")
    )
    _inject_trafilatura(monkeypatch, lambda html, **k: None)  # extraction yields nothing
    out = asyncio.run(W.fetch("https://example.com/b"))
    assert "Raw body text" in out
    assert "<" not in out  # tags stripped


@respx.mock
def test_fetch_missing_trafilatura(monkeypatch):
    respx.get("https://example.com/c").mock(
        return_value=httpx.Response(200, html="<html></html>")
    )
    monkeypatch.setitem(sys.modules, "trafilatura", None)  # import raises ImportError
    out = asyncio.run(W.fetch("https://example.com/c"))
    assert "optional 'web' extra" in out


# --- harness wiring -------------------------------------------------------- #
def test_build_rlm_web_registers_tools():
    cfg = HarnessConfig(web=True)
    main = build_lm(_model(), cfg.main_max_tokens, cfg.temperature)
    rlm = build_rlm(cfg, main, main)
    names = set(rlm.tools.keys()) if isinstance(rlm.tools, dict) else {
        t.__name__ for t in rlm.tools
    }
    assert "web_search" in names
    assert "fetch" in names


def test_build_rlm_without_web_omits_tools():
    cfg = HarnessConfig(web=False)
    main = build_lm(_model(), cfg.main_max_tokens, cfg.temperature)
    rlm = build_rlm(cfg, main, main)
    names = set(rlm.tools.keys()) if isinstance(rlm.tools, dict) else {
        t.__name__ for t in rlm.tools
    }
    assert "web_search" not in names
    assert "fetch" not in names


def test_web_skill_is_a_skill():
    from predict_rlm import Skill

    from rrlm.playbooks import web_skill

    skill = web_skill()
    assert isinstance(skill, Skill)
    assert "web_search" in skill.instructions


# --- CLI flag / env -------------------------------------------------------- #
def _capture_web(monkeypatch, argv):
    captured = {}

    def fake_solve(*a, **k):
        captured.update(k)
        return {"answer": "ok", "error": None}

    monkeypatch.setattr(S, "solve", fake_solve)
    monkeypatch.setattr(sys, "argv", argv)
    S.main()
    return captured


def test_cli_web_flag(monkeypatch):
    captured = _capture_web(monkeypatch, ["rrlm-solve", "-i", "q", "--web"])
    assert captured["web"] is True


def test_cli_web_env(monkeypatch):
    monkeypatch.setenv("RRLM_WEB", "1")
    captured = _capture_web(monkeypatch, ["rrlm-solve", "-i", "q"])
    assert captured["web"] is True


def test_cli_web_default_off(monkeypatch):
    monkeypatch.delenv("RRLM_WEB", raising=False)
    captured = _capture_web(monkeypatch, ["rrlm-solve", "-i", "q"])
    assert captured["web"] is False
