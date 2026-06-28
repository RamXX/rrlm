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
RUNS_DIR = PROJECT_ROOT / "runs"

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


@dataclass(frozen=True)
class HarnessConfig:
    """Budgets and knobs for one run. Model identity lives in ResolvedModel."""

    main_model: str = ""  # reference string, for logging/run-id only
    sub_model: str = ""
    max_depth: int = 2  # rlm_spawn recursion budget (0 = no recursion)
    max_iterations: int = 30  # REPL turns per agent
    max_llm_calls: int = 50  # sub-LM call budget per agent
    main_max_tokens: int = 16_384  # per-turn output cap for the orchestrator
    sub_max_tokens: int = 8_192  # per-call output cap for predict()
    temperature: float = 0.2
    backend: str = "jspi"  # "jspi" (Deno/WASM) | "sbx" (Docker) | "supervisor" (local CPython)
    reasoning: str = "default"  # default | off | low | medium | high
    # Per-REPL-turn sandbox wall-clock cap. predict-rlm defaults to 300s, which
    # assumes cloud-speed concurrent leaves; local serial leaves need a wide
    # fan-out to fit, so callers raise this for local endpoints.
    sandbox_exec_timeout: float = 300.0
    # Per-turn action-generation re-asks on a parse/validation failure. Only
    # applied when the installed predict-rlm supports it (feature-detected).
    max_action_retries: int = 0

    def as_dict(self) -> dict:
        return asdict(self)


def load_env() -> str:
    """Load ``.env`` and return the OpenRouter key (may be empty).

    The OpenRouter key is only needed for the optional no-Pi path and for cost
    reconciliation against OpenRouter's generation endpoint. Model credentials
    themselves come from Pi config, so absence is not fatal.
    """
    load_dotenv(PROJECT_ROOT / ".env")
    return os.environ.get("OPENROUTER_API_KEY", "").strip()
