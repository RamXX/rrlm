"""rrlm-doctor: check everything rrlm needs and say what is missing.

Prints one line per check, ``[ok]`` / ``[--]`` (absent but optional) /
``[!!]`` (a problem for the default path), then a one-line verdict. Never
prints credential values. Exit code 0 always; this is a report, not a gate.
"""

from __future__ import annotations

import importlib.metadata
import os
import shutil
import sys

from rrlm.pi_config import _BUILTIN_PROVIDERS, load_pi_config, resolve_model

_OK, _NO, _BAD = "[ok]", "[--]", "[!!]"


def _pkg_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _check_versions(lines: list[str]) -> None:
    lines.append("versions")
    lines.append(f"  {_OK} python {sys.version.split()[0]}")
    for pkg in ("rrlm", "dspy", "predict-rlm", "litellm"):
        v = _pkg_version(pkg)
        lines.append(f"  {_OK} {pkg} {v}" if v else f"  {_BAD} {pkg} not installed")


def _check_pi(lines: list[str]) -> bool:
    """Report Pi config state; True when a default model resolves."""
    lines.append("pi config")
    cfg = load_pi_config()
    if cfg.agent_dir and cfg.agent_dir.exists():
        lines.append(f"  {_OK} agent dir {cfg.agent_dir}")
    else:
        lines.append(f"  {_NO} no Pi agent dir at {cfg.agent_dir} (Pi not set up; "
                     "explicit 'provider/model' refs + env keys still work)")
    if cfg.providers:
        lines.append(f"  {_OK} {len(cfg.providers)} custom provider(s): "
                     + ", ".join(sorted(cfg.providers)))
    else:
        lines.append(f"  {_NO} no custom providers in models.json")
    try:
        m = resolve_model(None)
        lines.append(f"  {_OK} default model resolves: {m.provider}/{m.model_id}")
        return True
    except Exception as exc:  # noqa: BLE001, report instead of crash
        lines.append(f"  {_NO} no default model ({exc})")
        return False


def _check_credentials(lines: list[str]) -> None:
    lines.append("credentials (env)")
    found = False
    for _, (_, env_var, _) in sorted(_BUILTIN_PROVIDERS.items()):
        if os.environ.get(env_var, "").strip():
            lines.append(f"  {_OK} {env_var} set")
            found = True
    if not found:
        lines.append(f"  {_NO} no provider API keys in the environment "
                     "(fine if your Pi config carries them)")


def _check_backends(lines: list[str]) -> None:
    lines.append("backends")
    lines.append(f"  {_OK} supervisor: host CPython (default; no extra runtime)")
    deno = shutil.which("deno")
    lines.append(f"  {_OK} jspi: deno at {deno}" if deno
                 else f"  {_NO} jspi: deno not on PATH (needed only for --backend jspi)")
    docker = shutil.which("docker")
    sbx = shutil.which("sbx")
    if docker and sbx:
        lines.append(f"  {_OK} sbx: docker + sbx CLI present")
    else:
        missing = " and ".join(n for n, p in (("docker", docker), ("sbx CLI", sbx)) if not p)
        lines.append(f"  {_NO} sbx: {missing} not on PATH (needed only for --backend sbx)")


def _check_extras(lines: list[str]) -> None:
    lines.append("extras")
    from rrlm.webtools import web_available

    lines.append(f"  {_OK} web: ddgs + trafilatura installed (--web ready)" if web_available()
                 else f"  {_NO} web: install with `uv sync --extra web` to enable --web")
    try:
        import rlm_gepa  # noqa: F401

        lines.append(f"  {_OK} gepa: rlm_gepa installed (rrlm-gepa ready)")
    except ImportError:
        lines.append(f"  {_NO} gepa: install with `uv sync --extra gepa` to enable rrlm-gepa")


def _check_local_servers(lines: list[str]) -> None:
    """Ping every locally-hosted provider's /models endpoint (fast timeout)."""
    import httpx

    cfg = load_pi_config()
    local = {
        name: pdef["baseUrl"]
        for name, pdef in cfg.providers.items()
        if isinstance(pdef, dict) and isinstance(pdef.get("baseUrl"), str)
        and any(h in pdef["baseUrl"] for h in ("localhost", "127.0.0.1", "0.0.0.0", "::1"))
    }
    if not local:
        return
    lines.append("local model servers")
    for name, base in sorted(local.items()):
        url = base.rstrip("/") + "/models"
        try:
            resp = httpx.get(url, timeout=2.0)
            resp.raise_for_status()
            lines.append(f"  {_OK} {name}: up at {base}")
        except Exception:  # noqa: BLE001, down/refused/timeout all read the same
            lines.append(f"  {_NO} {name}: not responding at {base} "
                         "(start it, or ignore if unused)")


def report() -> str:
    lines: list[str] = []
    _check_versions(lines)
    resolvable = _check_pi(lines)
    _check_credentials(lines)
    _check_backends(lines)
    _check_extras(lines)
    _check_local_servers(lines)
    if resolvable:
        lines.append("verdict: ready. Try: rrlm-solve -i 'Say ok.' -d ''")
    else:
        lines.append(
            "verdict: no default model. Pass one explicitly, e.g. "
            "`rrlm-solve --main openrouter/qwen/qwen3.6-27b ...` with "
            "OPENROUTER_API_KEY set, or configure Pi (~/.pi/agent/)."
        )
    return "\n".join(lines)


def main() -> None:
    print(report())


if __name__ == "__main__":
    main()
