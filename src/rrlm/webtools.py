"""Host-side web-retrieval tools for the RLM agent (opt-in).

These let the orchestrator answer from the live web instead of from pretraining:
it writes REPL code that calls ``web_search()`` / ``fetch()``, reads the returned
text, and verifies before answering.

The functions run HOST-SIDE. predict-rlm bridges tool calls back to the host (the
same path that lets the built-in ``predict()`` reach the host LM from inside a
sandbox), so these work identically on the ``supervisor``, ``sbx``, and ``jspi``
backends. A useful consequence: the model's own sandboxed code stays
network-free; the web is reachable only through these two vetted functions, not
through arbitrary ``httpx`` the model might write.

Keyless by design: DuckDuckGo via ``ddgs`` for search, ``trafilatura`` for
main-text extraction. Install the optional extra:

    uv sync --extra web              # from a checkout
    uv tool install 'rrlm[web]'      # as an installed tool
"""

from __future__ import annotations

import asyncio
import ipaddress
import os
import re
import socket
from collections.abc import Callable
from urllib.parse import urljoin, urlparse

_MISSING = (
    "rrlm web tools need the optional 'web' extra. Install with "
    "`uv sync --extra web` (checkout) or `uv tool install 'rrlm[web]'`."
)
# A polite, identifiable UA. Some sites block the default httpx UA outright.
_UA = "Mozilla/5.0 (compatible; rrlm-web/1.0; +https://github.com/RamXX/rrlm)"

_MAX_REDIRECTS = 5


def _private_fetch_allowed() -> bool:
    """RRLM_WEB_ALLOW_PRIVATE=1 disables the SSRF guard (trusted intranet use)."""
    return os.environ.get("RRLM_WEB_ALLOW_PRIVATE", "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _url_blocked(url: str) -> str | None:
    """SSRF guard: reject URLs whose host resolves to a non-public address.

    ``fetch`` runs HOST-side, outside the sandbox, so without this check the
    model's code could reach localhost services, RFC-1918 intranet hosts, or
    cloud metadata endpoints (169.254.169.254) through it. Every redirect hop
    is re-checked. Returns a reason string when blocked, None when allowed.
    Resolution failures are not blocked here; the fetch itself will fail with
    a clearer connection error.
    """
    if _private_fetch_allowed():
        return None
    host = urlparse(url).hostname
    if not host:
        return f"no hostname in url {url!r}"
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError:
        return None  # let the request fail naturally with a connect error
    for info in infos:
        try:
            ip = ipaddress.ip_address(info[4][0])
        except ValueError:
            continue
        if not ip.is_global:
            return (
                f"host {host!r} resolves to the non-public address {ip}; "
                "fetch() only reaches the public web (set RRLM_WEB_ALLOW_PRIVATE=1 "
                "to allow private addresses in a trusted environment)"
            )
    return None


def _import_ddgs():
    """Return the DDGS class from whichever package name is installed, or None."""
    try:
        from ddgs import DDGS  # current package name

        return DDGS
    except ImportError:
        try:
            from duckduckgo_search import DDGS  # legacy package name

            return DDGS
        except ImportError:
            return None


async def web_search(query: str, max_results: int = 5) -> str:
    """Search the web and return ranked results as text.

    Use this to DISCOVER sources for a factual or current-events question before
    answering. Returns up to ``max_results`` hits, each formatted as
    ``"N. title\\n   url\\n   snippet"``. Follow up with ``fetch(url)`` to read a
    promising result, then verify the fact across two sources before you SUBMIT.
    Do not answer general-knowledge or time-sensitive questions from memory:
    search, read, verify. This is awaitable: ``await web_search("...")``.
    """
    query = (query or "").strip()
    if not query:
        return "web_search: empty query"
    ddgs_cls = _import_ddgs()
    if ddgs_cls is None:
        return _MISSING

    n = max(1, min(int(max_results or 5), 10))

    def _run() -> list[dict]:
        with ddgs_cls() as ddgs:
            return list(ddgs.text(query, max_results=n))

    try:
        hits = await asyncio.to_thread(_run)
    except Exception as exc:  # noqa: BLE001 (return the failure into the REPL)
        return f"web_search error: {type(exc).__name__}: {exc}"
    if not hits:
        return f"web_search: no results for {query!r}"
    lines = []
    for i, h in enumerate(hits, 1):
        title = (h.get("title") or "").strip()
        url = (h.get("href") or h.get("url") or "").strip()
        body = " ".join((h.get("body") or "").split())
        lines.append(f"{i}. {title}\n   {url}\n   {body[:300]}")
    return "\n".join(lines)


async def fetch(url: str, max_chars: int = 8000) -> str:
    """Fetch a URL and return its main text content, cleaned of boilerplate.

    Use after ``web_search`` to read a specific result. Returns the extracted
    article text (via ``trafilatura``), truncated to ``max_chars``; navigation,
    ads, and markup are stripped. On any failure it returns an error string, so
    do not assume success: check the result. Only public web addresses are
    reachable (private/loopback/metadata hosts are refused). This is awaitable,
    and independent fetches can be batched with ``asyncio.gather``.
    """
    url = (url or "").strip()
    if not (url.startswith("http://") or url.startswith("https://")):
        return f"fetch: invalid url {url!r} (must start with http:// or https://)"
    try:
        import httpx
    except ImportError:  # pragma: no cover (httpx ships with the base deps)
        return _MISSING

    try:
        # Redirects are followed manually so EVERY hop passes the SSRF guard
        # (a public page redirecting to an internal address must not work).
        async with httpx.AsyncClient(
            follow_redirects=False, timeout=20.0, headers={"User-Agent": _UA}
        ) as client:
            current = url
            for _ in range(_MAX_REDIRECTS + 1):
                blocked = await asyncio.to_thread(_url_blocked, current)
                if blocked:
                    return f"fetch blocked: {blocked}"
                resp = await client.get(current)
                if resp.is_redirect and resp.headers.get("location"):
                    current = urljoin(current, resp.headers["location"])
                    continue
                resp.raise_for_status()
                html = resp.text
                break
            else:
                return f"fetch error: more than {_MAX_REDIRECTS} redirects from {url!r}"
    except Exception as exc:  # noqa: BLE001
        return f"fetch error: {type(exc).__name__}: {exc}"

    text = None
    try:
        import trafilatura

        text = trafilatura.extract(html, include_comments=False, include_tables=True)
    except ImportError:
        return _MISSING
    except Exception:  # noqa: BLE001 (extraction is best-effort; fall back to raw)
        text = None

    if not text:
        # Crude fallback: strip tags so the caller still gets readable content.
        text = " ".join(re.sub(r"<[^>]+>", " ", html).split())

    text = text.strip()
    max_chars = max(500, int(max_chars or 8000))
    if len(text) > max_chars:
        text = text[:max_chars] + f"\n... [truncated at {max_chars} chars]"
    return text or "fetch: no extractable text"


def web_tools() -> list[Callable]:
    """The host-side web tools to pass to ``PredictRLM(tools=...)``."""
    return [web_search, fetch]


def web_available() -> bool:
    """True if the optional web dependencies (ddgs + trafilatura) are importable.

    httpx is part of the base dependency set, so only the two extras are checked.
    """
    if _import_ddgs() is None:
        return False
    try:
        import trafilatura  # noqa: F401

        return True
    except ImportError:
        return False
