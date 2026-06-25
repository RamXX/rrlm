"""Resolve model references against the local Pi (pi-coding-agent) config.

rrlm runs whatever models Pi is already configured with -- local servers,
OpenRouter, OpenAI, Anthropic, z.ai, and so on -- instead of carrying its own
model registry. This module reads Pi's config files and turns a model reference
into the connection details a ``dspy.LM`` needs.

Pi config locations (override the agent dir with ``$PI_CODING_AGENT_DIR``)::

    ~/.pi/config.json          current default {provider, model, baseUrl, apiKey}
    ~/.pi/agent/models.json    custom providers + models
    ~/.pi/agent/settings.json  defaultProvider / defaultModel
    ~/.pi/agent/auth.json      credentials (read for resolution only)

A reference is ``provider/model`` (``openrouter/qwen/qwen3.6-27b``), a bare model
id (``glm-5``), or ``None`` to use Pi's current default. Key resolution follows
Pi's order -- ``auth.json`` entry, then environment variable, then the
``models.json`` ``apiKey`` -- and understands Pi's key formats: a literal string,
a ``$VAR`` / ``VAR`` environment reference, or a ``!command`` whose stdout is the
key. ``auth.json`` is read only to resolve credentials; its contents are never
logged.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

# Built-in Pi providers mapped to their litellm prefix and credential sources.
# Pi knows these providers' base URLs and model catalogs natively; litellm does
# too, so we only need the prefix plus where to find the key. Source: the pi
# skill's references/providers-and-models.md (env var + auth.json key columns).
_BUILTIN_PROVIDERS: dict[str, tuple[str, str, str]] = {
    # pi name:        (litellm prefix, env var,             auth.json key)
    "anthropic": ("anthropic", "ANTHROPIC_API_KEY", "anthropic"),
    "openai": ("openai", "OPENAI_API_KEY", "openai"),
    "google": ("gemini", "GEMINI_API_KEY", "google"),
    "gemini": ("gemini", "GEMINI_API_KEY", "google"),
    "openrouter": ("openrouter", "OPENROUTER_API_KEY", "openrouter"),
    "deepseek": ("deepseek", "DEEPSEEK_API_KEY", "deepseek"),
    "mistral": ("mistral", "MISTRAL_API_KEY", "mistral"),
    "groq": ("groq", "GROQ_API_KEY", "groq"),
    "xai": ("xai", "XAI_API_KEY", "xai"),
    "cerebras": ("cerebras", "CEREBRAS_API_KEY", "cerebras"),
    "fireworks": ("fireworks_ai", "FIREWORKS_API_KEY", "fireworks"),
}

# Pi `api` kind -> litellm provider prefix for custom (models.json) providers.
_API_TO_LITELLM: dict[str, str] = {
    "openai-completions": "openai",
    "openai-responses": "openai",
    "anthropic-messages": "anthropic",
    "google-generative-ai": "gemini",
}

_LOCAL_HOSTS = ("localhost", "127.0.0.1", "0.0.0.0", "::1")
_ENV_NAME = re.compile(r"^[A-Z][A-Z0-9_]*$")

DEFAULT_CONTEXT_WINDOW = 128_000
DEFAULT_MAX_TOKENS = 32_768


@dataclass(frozen=True)
class ResolvedModel:
    """A model reference resolved to everything ``build_lm`` needs."""

    ref: str  # the original reference, for messages/logging
    provider: str  # pi provider name
    model_id: str  # id sent to the API
    litellm_id: str  # e.g. "openrouter/qwen/qwen3.6-27b", "openai/glm-5"
    api_key: str
    api_base: str | None = None  # set for openai-compatible endpoints
    context_window: int = DEFAULT_CONTEXT_WINDOW
    max_tokens: int = DEFAULT_MAX_TOKENS
    supports_reasoning: bool = False
    # How to request reasoning on/off: OpenRouter uses a `reasoning` body field;
    # OpenAI-compatible chat servers (qwen/glm/gemma family) use a chat-template
    # kwarg; anything else can't be toggled generically.
    reasoning_style: str = "none"  # "openrouter" | "chat_template" | "none"
    is_local: bool = False
    needs_json_schema: bool = False  # LM Studio rejects response_format json_object
    openrouter_routing: dict | None = None  # optional provider pin from models.json

    @property
    def is_openrouter(self) -> bool:
        return self.provider == "openrouter"


@dataclass(frozen=True)
class PiConfig:
    """Parsed Pi configuration; all sections optional/empty when absent."""

    agent_dir: Path | None = None
    current_default: dict = field(default_factory=dict)  # ~/.pi/config.json
    providers: dict = field(default_factory=dict)  # models.json "providers"
    settings: dict = field(default_factory=dict)  # settings.json
    auth: dict = field(default_factory=dict)  # auth.json


def pi_agent_dir() -> Path:
    """Pi agent dir: ``$PI_CODING_AGENT_DIR`` or ``~/.pi/agent``."""
    override = os.environ.get("PI_CODING_AGENT_DIR")
    return Path(override) if override else Path.home() / ".pi" / "agent"


def _load_json(path: Path) -> dict:
    try:
        with path.open(encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def load_pi_config(pi_dir: str | Path | None = None) -> PiConfig:
    """Load Pi's config files. Missing files yield empty sections, not errors."""
    agent_dir = Path(pi_dir) if pi_dir else pi_agent_dir()
    models = _load_json(agent_dir / "models.json")
    # ~/.pi/config.json sits one level above the agent dir in the default layout.
    top = agent_dir.parent / "config.json"
    return PiConfig(
        agent_dir=agent_dir,
        current_default=_load_json(top),
        providers=models.get("providers", {}) if isinstance(models, dict) else {},
        settings=_load_json(agent_dir / "settings.json"),
        auth=_load_json(agent_dir / "auth.json"),
    )


def _resolve_key_string(raw: str) -> str:
    """Resolve a Pi key string: literal | $VAR / VAR | !command."""
    s = (raw or "").strip()
    if not s:
        return ""
    if s.startswith("!"):
        try:
            out = subprocess.run(
                s[1:], shell=True, capture_output=True, text=True, timeout=30
            )
            return out.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""
    if s.startswith("$"):
        return os.environ.get(s[1:], "")
    if _ENV_NAME.match(s):  # looks like an env var name (e.g. OPENAI_API_KEY)
        return os.environ.get(s, "")
    return s  # literal (e.g. "lm-studio", "ollama", "1234")


def _resolve_key(raw: str | None, auth_entry: object) -> str:
    """auth.json entry wins over the models.json/literal key (Pi's order)."""
    if isinstance(auth_entry, dict):
        if auth_entry.get("type", "api_key") == "api_key" and auth_entry.get("key"):
            resolved = _resolve_key_string(str(auth_entry["key"]))
            if resolved:
                return resolved
    elif isinstance(auth_entry, str) and auth_entry:
        resolved = _resolve_key_string(auth_entry)
        if resolved:
            return resolved
    return _resolve_key_string(raw or "")


def _is_local_url(url: str | None) -> bool:
    return bool(url) and any(h in url for h in _LOCAL_HOSTS)


def _needs_json_schema(provider: str, base_url: str | None) -> bool:
    """LM Studio's OpenAI endpoint rejects response_format json_object.

    Force on with ``RRLM_RESPONSE_SCHEMA=1`` for any other server that needs it.
    """
    override = os.environ.get("RRLM_RESPONSE_SCHEMA", "").strip().lower()
    if override in ("1", "true", "on", "yes"):
        return True
    if override in ("0", "false", "off", "no"):
        return False
    hay = f"{provider} {base_url or ''}".lower()
    return "lmstudio" in hay or "lm-studio" in hay or ":1234" in hay


def _reasoning_style(provider: str, api_kind: str) -> str:
    if provider == "openrouter":
        return "openrouter"
    if api_kind in ("openai-completions", "openai-responses"):
        return "chat_template"
    return "none"


def known_providers(cfg: PiConfig) -> set[str]:
    return set(_BUILTIN_PROVIDERS) | set(cfg.providers)


def _from_current_default(cfg: PiConfig, ref: str) -> ResolvedModel | None:
    """Build directly from ~/.pi/config.json (the model Pi is running now)."""
    cd = cfg.current_default
    model = cd.get("model")
    base = cd.get("baseUrl")
    if not (model and base):
        return None
    provider = cd.get("provider", "openai")
    key = _resolve_key(cd.get("apiKey"), cfg.auth.get(provider))
    return ResolvedModel(
        ref=ref,
        provider=provider,
        model_id=model,
        litellm_id=f"openai/{model}",
        api_key=key or "not-needed",
        api_base=base,
        is_local=_is_local_url(base),
        reasoning_style="chat_template",
        needs_json_schema=_needs_json_schema(provider, base),
    )


def _resolve_custom(cfg: PiConfig, provider: str, model_id: str, ref: str) -> ResolvedModel:
    pdef = cfg.providers[provider]
    base = pdef.get("baseUrl")
    api_kind = pdef.get("api", "openai-completions")
    models = pdef.get("models", [])
    mdef: dict = {}
    for m in models:
        if isinstance(m, dict) and (m.get("id") == model_id or m.get("name") == model_id):
            mdef = m
            api_kind = m.get("api", api_kind)
            break
    prefix = _API_TO_LITELLM.get(api_kind, "openai")
    key = _resolve_key(pdef.get("apiKey"), cfg.auth.get(provider))
    return ResolvedModel(
        ref=ref,
        provider=provider,
        model_id=model_id,
        litellm_id=f"{prefix}/{model_id}",
        api_key=key or "not-needed",
        api_base=base,
        context_window=int(mdef.get("contextWindow", DEFAULT_CONTEXT_WINDOW)),
        max_tokens=int(mdef.get("maxTokens", DEFAULT_MAX_TOKENS)),
        supports_reasoning=bool(mdef.get("reasoning", False)),
        reasoning_style=_reasoning_style(provider, api_kind),
        is_local=_is_local_url(base),
        needs_json_schema=_needs_json_schema(provider, base),
    )


def _resolve_builtin(cfg: PiConfig, provider: str, model_id: str, ref: str) -> ResolvedModel:
    prefix, env_var, auth_key = _BUILTIN_PROVIDERS[provider]
    # A models.json entry may override a built-in provider's base URL / models.
    override = cfg.providers.get(provider, {}) if isinstance(cfg.providers, dict) else {}
    base = override.get("baseUrl")
    mdef: dict = {}
    for m in override.get("models", []):
        if isinstance(m, dict) and (m.get("id") == model_id or m.get("name") == model_id):
            mdef = m
            break
    overrides = override.get("modelOverrides", {})
    if isinstance(overrides, dict) and model_id in overrides:
        mdef = {**mdef, **overrides[model_id]}
    routing = None
    compat = mdef.get("compat", {}) if isinstance(mdef, dict) else {}
    if isinstance(compat, dict):
        routing = compat.get("openRouterRouting")
    key = _resolve_key(override.get("apiKey"), cfg.auth.get(auth_key)) or os.environ.get(env_var, "")
    return ResolvedModel(
        ref=ref,
        provider=provider,
        model_id=model_id,
        litellm_id=f"{prefix}/{model_id}" if not base else f"openai/{model_id}",
        api_key=key or "not-needed",
        api_base=base,
        context_window=int(mdef.get("contextWindow", DEFAULT_CONTEXT_WINDOW)),
        max_tokens=int(mdef.get("maxTokens", DEFAULT_MAX_TOKENS)),
        supports_reasoning=bool(mdef.get("reasoning", provider in ("anthropic", "openai", "google"))),
        reasoning_style=_reasoning_style(provider, "openai-completions"),
        is_local=_is_local_url(base),
        needs_json_schema=_needs_json_schema(provider, base),
        openrouter_routing=routing,
    )


def _split_ref(ref: str, cfg: PiConfig) -> tuple[str, str]:
    """Split a reference into (provider, model_id).

    ``provider/model`` when the first segment is a known provider; otherwise the
    whole reference is treated as a bare model id and matched against configured
    providers (then Pi's default provider).
    """
    providers = known_providers(cfg)
    if "/" in ref:
        head, tail = ref.split("/", 1)
        if head in providers:
            return head, tail
    # bare model id: find the provider that defines it
    for pname, pdef in cfg.providers.items():
        for m in pdef.get("models", []) if isinstance(pdef, dict) else []:
            if isinstance(m, dict) and (m.get("id") == ref or m.get("name") == ref):
                return pname, ref
    default_provider = cfg.settings.get("defaultProvider") or cfg.current_default.get("provider")
    if default_provider:
        return default_provider, ref
    raise ValueError(
        f"cannot resolve model reference {ref!r}: no provider prefix and no match "
        f"in Pi config. Use 'provider/model' (e.g. 'openrouter/qwen/qwen3.6-27b') "
        f"or configure the model in {cfg.agent_dir}/models.json."
    )


def resolve_model(ref: str | None, *, pi_dir: str | Path | None = None) -> ResolvedModel:
    """Resolve a model reference (or ``None`` for Pi's default) to a ResolvedModel."""
    cfg = load_pi_config(pi_dir)

    if not ref:
        # Prefer the model Pi is running right now (~/.pi/config.json), which
        # carries its own baseUrl + key.
        from_default = _from_current_default(cfg, ref="<pi-default>")
        if from_default is not None:
            return from_default
        provider = cfg.settings.get("defaultProvider")
        model = cfg.settings.get("defaultModel")
        if not (provider and model):
            raise ValueError(
                "no model specified and no Pi default found. Pass an explicit "
                "model (e.g. 'openrouter/qwen/qwen3.6-27b') or set defaultProvider/"
                f"defaultModel in {cfg.agent_dir}/settings.json."
            )
        ref = f"{provider}/{model}"

    provider, model_id = _split_ref(ref, cfg)
    if provider in cfg.providers and provider not in _BUILTIN_PROVIDERS:
        return _resolve_custom(cfg, provider, model_id, ref)
    if provider in _BUILTIN_PROVIDERS:
        return _resolve_builtin(cfg, provider, model_id, ref)
    # provider present in models.json AND a built-in (override case): custom wins
    if provider in cfg.providers:
        return _resolve_custom(cfg, provider, model_id, ref)
    raise ValueError(
        f"unknown provider {provider!r} for reference {ref!r}. Known providers: "
        f"{', '.join(sorted(known_providers(cfg))) or '(none configured)'}."
    )
