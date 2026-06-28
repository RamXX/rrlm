"""Unit tests for rrlm.config: harness config serialization and env loading."""

from __future__ import annotations

from rrlm.config import HarnessConfig, load_env


def test_harness_config_as_dict_round_trips_defaults():
    d = HarnessConfig().as_dict()
    assert d["backend"] == "jspi"
    assert d["max_depth"] == 2
    assert d["max_iterations"] == 30
    assert d["max_llm_calls"] == 50


def test_load_env_returns_stripped_openrouter_key(monkeypatch):
    # load_dotenv does not override an already-set var, so the explicit value wins.
    monkeypatch.setenv("OPENROUTER_API_KEY", "  sk-or-test  ")
    assert load_env() == "sk-or-test"


def test_load_env_empty_when_absent(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    # An empty/missing key is not fatal; load_env returns an empty string.
    assert isinstance(load_env(), str)
