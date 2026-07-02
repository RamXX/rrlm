"""Harness budgets, paths, and environment loading.

Model configuration is no longer a registry here, it is resolved from Pi's own
config (see ``rrlm.pi_config``), so rrlm runs whatever models Pi provides.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _default_runs_dir() -> Path:
    """Benchmark artifacts dir: env override, else the checkout, else the CWD.

    When rrlm is installed as a tool, ``PROJECT_ROOT`` points inside the
    virtualenv (no pyproject.toml there), so fall back to the caller's CWD
    instead of writing runs/ into site-packages.
    """
    env = os.environ.get("RRLM_RUNS_DIR")
    if env:
        return Path(env)
    if (PROJECT_ROOT / "pyproject.toml").exists():
        return PROJECT_ROOT / "runs"
    return Path.cwd() / "runs"


RUNS_DIR = _default_runs_dir()

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

BACKENDS = ("supervisor", "jspi", "sbx")


def resolve_backend(arg: str | None) -> str:
    """Backend precedence: explicit arg > ``RRLM_BACKEND`` env > ``supervisor``.

    ``supervisor`` (host CPython) is the default because it needs no extra
    runtime (no Deno, no Docker) and is the fastest; pick ``jspi`` or ``sbx``
    when the data or task is untrusted and you want the code sandboxed.
    """
    backend = (arg or os.environ.get("RRLM_BACKEND", "")).strip() or "supervisor"
    if backend not in BACKENDS:
        raise ValueError(
            f"unknown backend {backend!r}: choose one of {', '.join(BACKENDS)}"
        )
    return backend


@dataclass(frozen=True)
class HarnessConfig:
    """Budgets and knobs for one run. Model identity lives in ResolvedModel."""

    main_model: str = ""  # reference string, for logging/run-id only
    sub_model: str = ""
    max_depth: int = 2  # rlm_spawn recursion budget (0 = no recursion)
    max_iterations: int = 30  # REPL turns per agent
    # Sub-LM (predict) call budget. Enforced twice: per agent by predict-rlm,
    # and globally across the whole spawn tree at the shared LM (see
    # rrlm.harness.RunBudget), so it is a real per-run ceiling.
    max_llm_calls: int = 50
    # Global ceiling on rlm_spawn invocations across the whole tree. When it is
    # reached, further spawn calls return a refusal string so the orchestrator
    # can finish within its own REPL instead of dying.
    max_spawns: int = 16
    # Soft USD ceiling for the whole run, checked before each LM call from the
    # best per-call cost figures available (OpenRouter inline usage.cost or
    # litellm's estimate). Local models report no cost and do not count.
    # None = unlimited. The run can overshoot by at most one in-flight call.
    max_cost_usd: float | None = None
    main_max_tokens: int = 16_384  # per-turn output cap for the orchestrator
    sub_max_tokens: int = 8_192  # per-call output cap for predict()
    temperature: float = 0.2
    backend: str = "supervisor"  # "supervisor" (host CPython) | "jspi" (Deno/WASM) | "sbx" (Docker)
    reasoning: str = "default"  # default | off | low | medium | high
    # Per-REPL-turn sandbox wall-clock cap. predict-rlm defaults to 300s, which
    # assumes cloud-speed concurrent leaves; local serial leaves need a wide
    # fan-out to fit, so callers raise this for local endpoints.
    sandbox_exec_timeout: float = 300.0
    # Per-turn action-generation re-asks on a parse/validation failure. Only
    # applied when the installed predict-rlm supports it (feature-detected).
    max_action_retries: int = 0
    # Give the agent live web retrieval (host-side web_search/fetch tools + a
    # doctrine to retrieve-and-verify instead of answering from memory). Opt-in;
    # needs the optional 'web' extra (ddgs + trafilatura). See rrlm.webtools.
    web: bool = False

    def as_dict(self) -> dict:
        return asdict(self)


def load_env() -> str:
    """Load ``.env`` and return the OpenRouter key (may be empty).

    Looks in the checkout root first (development), then the caller's CWD
    (installed use, where PROJECT_ROOT points inside the virtualenv). Neither
    overrides variables already set in the environment.

    The OpenRouter key is only needed for the optional no-Pi path and for cost
    reconciliation against OpenRouter's generation endpoint. Model credentials
    themselves come from Pi config, so absence is not fatal.
    """
    load_dotenv(PROJECT_ROOT / ".env")
    load_dotenv(Path.cwd() / ".env")
    return os.environ.get("OPENROUTER_API_KEY", "").strip()
