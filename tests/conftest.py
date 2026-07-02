"""Shared fixtures for the rrlm test suite.

The integration and e2e tests run against a real, offline OpenAI-compatible stub
server (``tests/stub_server.py``) started as a genuine subprocess. There is no
in-process patching of the LLM call path: litellm sends real HTTP, the stub
answers with canned chat-completion JSON, and predict-rlm executes the returned
code for real. That is what makes these tests valid integration / e2e coverage
while staying fully offline, deterministic, and credential-free.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent
STUB_PATH = TESTS_DIR / "stub_server.py"

# rrlm reads these from the environment (CLI + extension parity); a developer's
# shell must never leak them into the suite, or tests stop being deterministic.
_RRLM_ENV_VARS = (
    "RRLM_MAIN", "RRLM_SUB", "RRLM_BACKEND", "RRLM_WEB", "RRLM_TIMEOUT",
    "RRLM_MAX_COST", "RRLM_TRACE_DIR", "RRLM_SBX_NAME", "RRLM_RESPONSE_SCHEMA",
    "RRLM_WEB_ALLOW_PRIVATE", "RRLM_RUNS_DIR", "RRLM_GEPA_DATASET",
)


@pytest.fixture(autouse=True)
def _clean_rrlm_env(monkeypatch):
    for var in _RRLM_ENV_VARS:
        monkeypatch.delenv(var, raising=False)

# Slow-mode server-side delay. Comfortably larger than the timeouts the timeout
# test uses, so the wall-clock ceiling fires before the stub would answer.
STUB_SLOW_SECONDS = 2.0


@pytest.fixture(scope="session")
def stub_base_url() -> str:
    """Start the offline stub server once for the session; yield its base URL."""
    proc = subprocess.Popen(
        [sys.executable, str(STUB_PATH), "--port", "0",
         "--slow-seconds", str(STUB_SLOW_SECONDS)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    # Wait for the readiness line carrying the OS-assigned port.
    host = port = None
    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        line = proc.stdout.readline()
        if not line:
            if proc.poll() is not None:
                err = proc.stderr.read()
                raise RuntimeError(f"stub server exited early: {err}")
            continue
        if line.startswith("STUB_READY"):
            _, host, port = line.split()
            break
    if not port:
        proc.terminate()
        raise RuntimeError("stub server did not become ready in time")

    base = f"http://{host}:{port}"
    try:
        yield base
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def write_stub_pi_config(
    agent_dir: Path,
    base_url: str,
    mode: str,
    *,
    model_id: str = "stub-model",
    provider: str = "stub",
    reasoning: bool = False,
    max_tokens: int = 4096,
    context_window: int = 8192,
) -> str:
    """Write a minimal Pi models.json pointing one provider at the stub.

    The chosen scenario (submit/predict/never/slow) is encoded in the provider
    baseUrl path prefix, which the stub server routes on. Returns the Pi model
    reference (``provider/model_id``) callers pass as ``main_model``.
    """
    agent_dir.mkdir(parents=True, exist_ok=True)
    models = {
        "providers": {
            provider: {
                "baseUrl": f"{base_url}/{mode}/v1",
                "api": "openai-completions",
                "apiKey": "stub-key",
                "models": [
                    {
                        "id": model_id,
                        "reasoning": reasoning,
                        "maxTokens": max_tokens,
                        "contextWindow": context_window,
                    }
                ],
            }
        }
    }
    (agent_dir / "models.json").write_text(json.dumps(models))
    return f"{provider}/{model_id}"


@pytest.fixture
def stub_model(tmp_path, monkeypatch, stub_base_url):
    """Return a factory: ``configure(mode) -> model_ref`` wired to the stub.

    Each call (re)writes a temp Pi config for the given scenario and points
    ``PI_CODING_AGENT_DIR`` at it, so ``rrlm`` resolves the model to the stub
    with no real credentials. Also clears OpenRouter creds so nothing can reach
    the network even if a code path tried.
    """
    agent_dir = tmp_path / "agent"
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(agent_dir))
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    def configure(mode: str, **kwargs) -> str:
        return write_stub_pi_config(agent_dir, stub_base_url, mode, **kwargs)

    return configure
