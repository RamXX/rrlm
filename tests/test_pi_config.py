"""Pi-config model resolution tests -- no network, isolated temp Pi config."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rrlm.pi_config import resolve_model

_MODELS = {
    "providers": {
        "lmstudio": {
            "baseUrl": "http://localhost:1234/v1",
            "api": "openai-completions",
            "apiKey": "lm-studio",
            "models": [
                {
                    "id": "qwen/qwen3.6-27b",
                    "reasoning": True,
                    "contextWindow": 131072,
                    "maxTokens": 16384,
                }
            ],
        },
        "omlx": {
            "baseUrl": "http://127.0.0.1:8004/v1",
            "api": "openai-completions",
            "apiKey": "1234",
            "models": [{"id": "gemma-4-31B", "reasoning": False, "maxTokens": 32768}],
        },
    }
}


def _write(agent_dir: Path, *, models=None, settings=None, auth=None, current=None) -> Path:
    agent_dir.mkdir(parents=True, exist_ok=True)
    if models is not None:
        (agent_dir / "models.json").write_text(json.dumps(models))
    if settings is not None:
        (agent_dir / "settings.json").write_text(json.dumps(settings))
    if auth is not None:
        (agent_dir / "auth.json").write_text(json.dumps(auth))
    if current is not None:
        (agent_dir.parent / "config.json").write_text(json.dumps(current))
    return agent_dir


@pytest.fixture
def agent_dir(tmp_path: Path) -> Path:
    return tmp_path / "agent"


def test_custom_provider_local_openai_compatible(agent_dir: Path):
    _write(agent_dir, models=_MODELS)
    m = resolve_model("lmstudio/qwen/qwen3.6-27b", pi_dir=agent_dir)
    assert m.provider == "lmstudio"
    assert m.model_id == "qwen/qwen3.6-27b"
    assert m.litellm_id == "openai/qwen/qwen3.6-27b"
    assert m.api_base == "http://localhost:1234/v1"
    assert m.api_key == "lm-studio"
    assert m.is_local is True
    assert m.needs_json_schema is True  # LM Studio (port 1234)
    assert m.reasoning_style == "chat_template"
    assert m.supports_reasoning is True
    assert m.context_window == 131072
    assert m.max_tokens == 16384


def test_bare_model_id_finds_provider(agent_dir: Path):
    _write(agent_dir, models=_MODELS)
    m = resolve_model("gemma-4-31B", pi_dir=agent_dir)
    assert m.provider == "omlx"
    assert m.is_local is True
    assert m.needs_json_schema is False  # not LM Studio


def test_default_from_current_config_json(agent_dir: Path):
    _write(
        agent_dir,
        models=_MODELS,
        current={
            "provider": "openai",
            "model": "mythos-26b",
            "baseUrl": "http://localhost:1234/v1",
            "apiKey": "lm-studio",
        },
    )
    m = resolve_model(None, pi_dir=agent_dir)
    assert m.model_id == "mythos-26b"
    assert m.litellm_id == "openai/mythos-26b"
    assert m.api_base == "http://localhost:1234/v1"
    assert m.is_local is True


def test_default_from_settings_when_no_config(agent_dir: Path):
    _write(agent_dir, models=_MODELS, settings={"defaultProvider": "omlx", "defaultModel": "gemma-4-31B"})
    m = resolve_model(None, pi_dir=agent_dir)
    assert m.provider == "omlx"
    assert m.model_id == "gemma-4-31B"


def test_builtin_openrouter_uses_env_key(agent_dir: Path, monkeypatch):
    _write(agent_dir, models={"providers": {}})
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-xyz")
    m = resolve_model("openrouter/qwen/qwen3.6-27b", pi_dir=agent_dir)
    assert m.litellm_id == "openrouter/qwen/qwen3.6-27b"
    assert m.api_key == "sk-or-xyz"
    assert m.is_local is False
    assert m.reasoning_style == "openrouter"


def test_apikey_env_reference_and_auth_precedence(agent_dir: Path, monkeypatch):
    models = {
        "providers": {
            "zai": {
                "baseUrl": "https://api.z.ai/v4",
                "api": "openai-completions",
                "apiKey": "$ZAI_API_KEY",
                "models": [{"id": "glm-5"}],
            }
        }
    }
    monkeypatch.setenv("ZAI_API_KEY", "from-env")
    _write(agent_dir, models=models)
    assert resolve_model("zai/glm-5", pi_dir=agent_dir).api_key == "from-env"

    # auth.json entry trumps the models.json key
    _write(agent_dir, models=models, auth={"zai": {"type": "api_key", "key": "from-auth"}})
    assert resolve_model("zai/glm-5", pi_dir=agent_dir).api_key == "from-auth"


def test_apikey_shell_command(agent_dir: Path):
    models = {
        "providers": {
            "custom": {
                "baseUrl": "http://localhost:9000/v1",
                "api": "openai-completions",
                "apiKey": "!printf secret123",
                "models": [{"id": "m"}],
            }
        }
    }
    _write(agent_dir, models=models)
    assert resolve_model("custom/m", pi_dir=agent_dir).api_key == "secret123"


def test_response_schema_override(agent_dir: Path, monkeypatch):
    _write(agent_dir, models=_MODELS)
    monkeypatch.setenv("RRLM_RESPONSE_SCHEMA", "0")
    assert resolve_model("lmstudio/qwen/qwen3.6-27b", pi_dir=agent_dir).needs_json_schema is False


def test_unresolvable_reference_raises(agent_dir: Path):
    _write(agent_dir, models={"providers": {}})
    with pytest.raises(ValueError, match="resolve"):
        resolve_model("nope-no-provider", pi_dir=agent_dir)
