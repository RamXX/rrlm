"""Tests for rrlm.webtools and its wiring into the harness and CLI.

Fully offline and deterministic: the optional 'web' extra (ddgs + trafilatura) is
not installed in CI, so the search backend is faked and HTTP is mocked with respx.
No network, no mocks inside the tool's own logic (only its external collaborators
are stubbed), so these are honest unit tests of the formatting/IO behaviour.
"""

from __future__ import annotations

import asyncio
import importlib
import ipaddress
import socket
import sys
import types

import httpx
import pytest
import respx

from rrlm.config import HarnessConfig
from rrlm.harness import build_lm, build_rlm
from rrlm.pi_config import ResolvedModel

W = importlib.import_module("rrlm.webtools")
S = importlib.import_module("rrlm.solve")


@pytest.fixture(autouse=True)
def _offline_dns(monkeypatch):
    """Keep the suite offline: the SSRF guard resolves hostnames, so answer
    locally. IP literals resolve to themselves; every name maps to a fixed,
    globally-routable address (documentation ranges are NOT is_global)."""

    def fake_getaddrinfo(host, *args, **kwargs):
        try:
            ipaddress.ip_address(host)
            ip = host
        except ValueError:
            ip = "93.184.216.34"
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 0))]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)


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


# --- SSRF guard ------------------------------------------------------------ #
def test_fetch_blocks_loopback_literal():
    out = asyncio.run(W.fetch("http://127.0.0.1:9999/admin"))
    assert out.startswith("fetch blocked:")
    assert "non-public address" in out


def test_fetch_blocks_metadata_endpoint():
    # The classic cloud-credential SSRF target must be unreachable.
    out = asyncio.run(W.fetch("http://169.254.169.254/latest/meta-data/"))
    assert out.startswith("fetch blocked:")


def test_fetch_blocks_private_range():
    out = asyncio.run(W.fetch("http://10.0.0.8/internal"))
    assert out.startswith("fetch blocked:")


def test_fetch_blocks_hostname_resolving_private(monkeypatch):
    def resolve_private(host, *a, **k):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("192.168.1.5", 0))]

    monkeypatch.setattr(socket, "getaddrinfo", resolve_private)
    out = asyncio.run(W.fetch("https://internal.corp.example/"))
    assert out.startswith("fetch blocked:")
    assert "internal.corp.example" in out


@respx.mock
def test_fetch_blocks_redirect_to_private(monkeypatch):
    """A public page 302-ing to an internal address must be refused: every
    redirect hop passes the guard, not just the first URL."""
    respx.get("https://example.com/redir").mock(
        return_value=httpx.Response(302, headers={"location": "http://10.0.0.9/secret"})
    )
    _inject_trafilatura(monkeypatch, lambda html, **k: "never reached")
    out = asyncio.run(W.fetch("https://example.com/redir"))
    assert out.startswith("fetch blocked:")
    assert "10.0.0.9" in out


@respx.mock
def test_fetch_follows_public_redirects(monkeypatch):
    respx.get("https://example.com/old").mock(
        return_value=httpx.Response(301, headers={"location": "https://example.com/new"})
    )
    respx.get("https://example.com/new").mock(
        return_value=httpx.Response(200, html="<html><body><p>moved here</p></body></html>")
    )
    _inject_trafilatura(monkeypatch, lambda html, **k: "moved here")
    assert asyncio.run(W.fetch("https://example.com/old")) == "moved here"


@respx.mock
def test_fetch_gives_up_after_too_many_redirects(monkeypatch):
    respx.get("https://example.com/loop").mock(
        return_value=httpx.Response(302, headers={"location": "https://example.com/loop"})
    )
    _inject_trafilatura(monkeypatch, lambda html, **k: "never reached")
    out = asyncio.run(W.fetch("https://example.com/loop"))
    assert "redirects" in out and out.startswith("fetch error:")


@respx.mock
def test_fetch_private_allowed_with_env(monkeypatch):
    """RRLM_WEB_ALLOW_PRIVATE=1 is the trusted-intranet escape hatch."""
    monkeypatch.setenv("RRLM_WEB_ALLOW_PRIVATE", "1")
    respx.get("http://127.0.0.1:9999/status").mock(
        return_value=httpx.Response(200, html="<html><body>intranet ok</body></html>")
    )
    _inject_trafilatura(monkeypatch, lambda html, **k: "intranet ok")
    assert asyncio.run(W.fetch("http://127.0.0.1:9999/status")) == "intranet ok"


def test_fetch_unresolvable_host_not_blocked_by_guard(monkeypatch):
    """Resolution failure is not a block: the request itself fails with a
    clearer connection error."""

    def raise_gaierror(host, *a, **k):
        raise OSError("name or service not known")

    monkeypatch.setattr(socket, "getaddrinfo", raise_gaierror)
    out = asyncio.run(W.fetch("https://no-such-host.invalid/"))
    assert out.startswith("fetch error:")  # connect failure, not "fetch blocked"


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
