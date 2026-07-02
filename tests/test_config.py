"""Unit tests for rrlm.config: harness config, backend resolution, env loading."""

from __future__ import annotations

import pytest

from rrlm.config import HarnessConfig, load_env, resolve_backend


def test_harness_config_as_dict_round_trips_defaults():
    d = HarnessConfig().as_dict()
    assert d["backend"] == "supervisor"
    assert d["max_depth"] == 2
    assert d["max_iterations"] == 30
    assert d["max_llm_calls"] == 50
    assert d["max_spawns"] == 16
    assert d["max_cost_usd"] is None


def test_resolve_backend_default_is_supervisor(monkeypatch):
    monkeypatch.delenv("RRLM_BACKEND", raising=False)
    assert resolve_backend(None) == "supervisor"


def test_resolve_backend_env_and_arg_precedence(monkeypatch):
    monkeypatch.setenv("RRLM_BACKEND", "jspi")
    assert resolve_backend(None) == "jspi"  # env fills the gap
    assert resolve_backend("sbx") == "sbx"  # explicit arg wins over env


def test_resolve_backend_rejects_unknown(monkeypatch):
    monkeypatch.delenv("RRLM_BACKEND", raising=False)
    with pytest.raises(ValueError, match="unknown backend"):
        resolve_backend("firecracker")


def test_load_env_returns_stripped_openrouter_key(monkeypatch):
    # load_dotenv does not override an already-set var, so the explicit value wins.
    monkeypatch.setenv("OPENROUTER_API_KEY", "  sk-or-test  ")
    assert load_env() == "sk-or-test"


def test_load_env_empty_when_absent(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    # An empty/missing key is not fatal; load_env returns an empty string.
    assert isinstance(load_env(), str)


def test_load_env_reads_cwd_dotenv(monkeypatch, tmp_path):
    # Installed (uv tool) users have no checkout .env; the CWD one must work.
    monkeypatch.delenv("RRLM_CWD_PROBE", raising=False)
    (tmp_path / ".env").write_text("RRLM_CWD_PROBE=from-cwd\n")
    monkeypatch.chdir(tmp_path)
    load_env()
    import os

    assert os.environ.get("RRLM_CWD_PROBE") == "from-cwd"
