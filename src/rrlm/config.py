"""Experiment configuration: model registry, harness budgets, paths."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = PROJECT_ROOT / "runs"

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


@dataclass(frozen=True)
class ModelSpec:
    """An OpenRouter model under test."""

    slug: str  # OpenRouter slug, e.g. "qwen/qwen3.7-max"
    context_length: int  # advertised; see context_is_total
    max_output_tokens: int  # hard model limit
    provider_order: tuple[str, ...] | None = None  # pin OpenRouter providers
    context_is_total: bool = False  # True when context_length covers in+out combined
    provider_prefix: str = "openrouter"  # litellm provider, e.g. "openai" for local vllm
    api_base: str | None = None  # set for local endpoints (ollama/vllm); skips reconcile
    # LM Studio rejects response_format json_object but accepts json_schema; set
    # this so DSPy emits json_schema (via litellm) instead of json_object.
    supports_response_schema: bool = False
    notes: str = ""

    @property
    def litellm_id(self) -> str:
        return f"{self.provider_prefix}/{self.slug}"

    @property
    def is_openrouter(self) -> bool:
        return self.provider_prefix == "openrouter"


MODELS: dict[str, ModelSpec] = {
    "qwen3.7-max": ModelSpec(
        slug="qwen/qwen3.7-max",
        context_length=1_000_000,
        max_output_tokens=65_536,
        notes="$1.25/M in, $3.75/M out (2026-06); thinking model",
    ),
    "qwen3.7-plus": ModelSpec(
        slug="qwen/qwen3.7-plus",
        context_length=1_000_000,
        max_output_tokens=65_536,
        notes="$0.32/M in, $1.28/M out (2026-06); thinking model",
    ),
    "gemma-4-26b": ModelSpec(
        slug="google/gemma-4-26b-a4b-it",
        context_length=262_100,
        max_output_tokens=32_768,
        provider_order=("cloudflare", "siliconflow"),
        context_is_total=True,  # 262.1K covers input+output combined
        notes="locally runnable class; low-context arm of the hypothesis",
    ),
    "qwen3.6-27b": ModelSpec(
        slug="qwen/qwen3.6-27b",
        context_length=262_100,
        max_output_tokens=65_536,
        provider_order=("chutes", "deepinfra"),
        context_is_total=True,
        notes="locally runnable class; low-context arm of the hypothesis",
    ),
    # Local DFlash/MLX servers (see ~/workspace/serve-models.sh). Uncensored
    # variants of the same architectures as qwen3.6-27b / gemma-4-26b, served
    # via mlx_lm.server. Slug must equal the server's --model value: mlx_lm
    # loads the model named in each request. Reasoning toggled locally through
    # chat_template_kwargs.enable_thinking, not the OpenRouter reasoning field.
    "heretic-27b-local": ModelSpec(
        slug="/Users/ramirosalas/.lmstudio/models/bi0h4z4rd88/"
        "Qwen3.6-27B-uncensored-heretic-v2-Native-MTP-Preserved-oQ8-mtp",
        context_length=262_100,
        max_output_tokens=65_536,
        context_is_total=True,
        provider_prefix="openai",
        api_base="http://127.0.0.1:8770/v1",
        notes="local Qwen3.6-27B uncensored (heretic) via mlx_lm.server :8770",
    ),
    "qwen3.6-27b-official-local": ModelSpec(
        slug="qwen/qwen3.6-27b",
        context_length=262_100,
        max_output_tokens=65_536,
        context_is_total=True,
        provider_prefix="openai",
        api_base="http://127.0.0.1:1234/v1",  # LM Studio
        supports_response_schema=True,  # LM Studio needs json_schema, not json_object
        notes="official Qwen3.6-27B via LM Studio; orchestrator-fidelity control",
    ),
    "mtp-27b-local": ModelSpec(
        slug="mtplx-qwen36-27b-optimized-quality",
        context_length=262_100,
        max_output_tokens=65_536,
        context_is_total=True,
        provider_prefix="openai",
        api_base="http://127.0.0.1:8000/v1",  # MTP qwen3.6-27b server
        notes="MTP-optimized Qwen3.6-27B (:8000); reasoning model, thinking not "
        "cleanly disablable; accepts json_object so no schema flag needed",
    ),
    "qwen3.6-27b-dflash-local": ModelSpec(
        slug="mlx-community/Qwen3.6-27B-8bit",
        context_length=262_100,
        max_output_tokens=65_536,
        context_is_total=True,
        provider_prefix="openai",
        api_base="http://127.0.0.1:8770/v1",  # dflash-qwen36.sh serve (DFlash + Q8)
        notes="official Qwen3.6-27B Q8 MLX via DFlash speculative decoding; "
        "reliable+fast orchestrator (replaces uncensored heretic and MTPLX)",
    ),
    "qwen3.6-27b-mlx-local": ModelSpec(
        slug="mlx-community/Qwen3.6-27B-8bit",
        context_length=262_100,
        max_output_tokens=65_536,
        context_is_total=True,
        provider_prefix="openai",
        api_base="http://127.0.0.1:8772/v1",  # serve-qwen36.sh (mlx_lm, stochastic)
        notes="official Qwen3.6-27B Q8 via mlx_lm.server (stochastic sampling); "
        "isolation test vs DFlash determinism",
    ),
    "supergemma-26b-local": ModelSpec(
        slug="Jiunsong/supergemma4-26b-uncensored-mlx-4bit-v2",
        context_length=262_100,
        max_output_tokens=32_768,
        context_is_total=True,
        provider_prefix="openai",
        api_base="http://127.0.0.1:8771/v1",
        notes="local gemma-4-26b uncensored (supergemma) via mlx_lm.server :8771",
    ),
}


@dataclass(frozen=True)
class HarnessConfig:
    """Budgets and knobs for one experimental condition."""

    main_model: str = "qwen3.7-max"
    sub_model: str = "qwen3.7-max"
    max_depth: int = 2  # rlm_spawn recursion budget (0 = no recursion)
    max_iterations: int = 30  # REPL turns per agent
    max_llm_calls: int = 50  # sub-LM call budget per agent
    main_max_tokens: int = 16_384  # per-turn output cap for the orchestrator
    sub_max_tokens: int = 8_192  # per-call output cap for predict()
    temperature: float = 0.2
    backend: str = "jspi"  # "jspi" (Deno/WASM) | "sbx" (Docker) | "supervisor" (real local CPython)
    reasoning: str = "default"  # OpenRouter reasoning: default | off | low | medium | high
    # Per-REPL-turn sandbox wall-clock cap. predict-rlm defaults to 300s, which
    # assumes cloud-speed concurrent leaves; local serial leaves need a wide
    # fan-out to fit, so the runner raises this for local endpoints.
    sandbox_exec_timeout: float = 300.0
    # Per-turn action-generation re-asks on a parse/validation failure (0.7+).
    # Absorbs intermittent malformed/empty turns that otherwise abort the run.
    max_action_retries: int = 0

    def as_dict(self) -> dict:
        return asdict(self)


def load_env() -> str:
    """Load .env and return the OpenRouter API key, failing loudly if absent."""
    load_dotenv(PROJECT_ROOT / ".env")
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Create a .env file at the project root "
            "(see .env.example)."
        )
    return key
