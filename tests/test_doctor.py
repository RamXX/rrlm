"""Tests for rrlm-doctor: the environment report.

Runs against a temp Pi config; the local-server check pings the real offline
stub server (a live localhost endpoint), so 'up' is genuinely observed.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from rrlm.doctor import main, report


def test_doctor_ready_with_stub_default(stub_model):
    model_ref = stub_model("submit")
    agent_dir = Path(os.environ["PI_CODING_AGENT_DIR"])
    provider, model_id = model_ref.split("/", 1)
    (agent_dir / "settings.json").write_text(
        json.dumps({"defaultProvider": provider, "defaultModel": model_id})
    )
    out = report()
    assert "[ok] python" in out
    assert "[ok] predict-rlm" in out
    assert f"[ok] default model resolves: {provider}/{model_id}" in out
    assert "supervisor: host CPython (default" in out
    assert "verdict: ready" in out
    # the stub is a live localhost server; the ping must see it up
    assert f"[ok] {provider}: up at" in out


def test_doctor_no_default_model(tmp_path, monkeypatch):
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(tmp_path / "empty-agent"))
    for var in ("OPENROUTER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    out = report()
    assert "[--] no default model" in out
    assert "verdict: no default model" in out
    assert "rrlm-solve --main" in out  # the report tells the user the way out


def test_doctor_down_local_server(tmp_path, monkeypatch):
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir(parents=True)
    (agent_dir / "models.json").write_text(json.dumps({
        "providers": {
            "deadlocal": {
                "baseUrl": "http://127.0.0.1:1/v1",
                "api": "openai-completions",
                "apiKey": "x",
                "models": [{"id": "m"}],
            }
        }
    }))
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(agent_dir))
    out = report()
    assert "[--] deadlocal: not responding at http://127.0.0.1:1/v1" in out


def test_doctor_credentials_reported_without_values(tmp_path, monkeypatch):
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(tmp_path / "none"))
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-supersecret-value")
    out = report()
    assert "[ok] OPENROUTER_API_KEY set" in out
    assert "supersecret" not in out  # never print credential values


def test_doctor_main_prints(capsys, tmp_path, monkeypatch):
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(tmp_path / "none"))
    main()
    assert "verdict:" in capsys.readouterr().out
