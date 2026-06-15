"""OpenRouter accounting client.

Authoritative per-call cost and timing come from the generation metadata
endpoint (GET /api/v1/generation?id=gen-...). The record is written
asynchronously on OpenRouter's side, so we retry briefly on 404.

Fields of interest in the returned data object (per OpenRouter docs):
  total_cost                USD actually charged
  native_tokens_prompt      billed prompt tokens (native tokenizer)
  native_tokens_completion  billed completion tokens
  latency                   total request latency, ms (TTFT for streamed)
  generation_time           token generation duration, ms
  provider_name, model, finish_reason, created_at
"""

from __future__ import annotations

import logging
import time

import httpx

from rrlm.config import OPENROUTER_BASE_URL

logger = logging.getLogger(__name__)

GENERATION_URL = f"{OPENROUTER_BASE_URL}/generation"

# ~10s worst case; the record is usually queryable within a couple of seconds
RETRY_DELAYS_S = (0.4, 0.8, 1.6, 3.2, 4.0)


def fetch_generation(
    gen_id: str,
    api_key: str,
    *,
    client: httpx.Client | None = None,
    retry_delays: tuple[float, ...] = RETRY_DELAYS_S,
) -> dict | None:
    """Fetch the generation metadata record for one completion.

    Returns the `data` dict, or None if the record never became available.
    """
    own_client = client is None
    client = client or httpx.Client(timeout=15.0)
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        for attempt, delay in enumerate((0.0, *retry_delays)):
            if delay:
                time.sleep(delay)
            try:
                resp = client.get(GENERATION_URL, params={"id": gen_id}, headers=headers)
            except httpx.HTTPError as exc:
                logger.warning("generation fetch %s attempt %d failed: %s", gen_id, attempt, exc)
                continue
            if resp.status_code == 200:
                return resp.json().get("data")
            if resp.status_code == 404:
                continue  # record not written yet
            logger.warning(
                "generation fetch %s returned HTTP %d: %s",
                gen_id,
                resp.status_code,
                resp.text[:200],
            )
            if resp.status_code in (401, 403):
                return None  # not transient; do not burn retries
        logger.warning("generation record %s never became available", gen_id)
        return None
    finally:
        if own_client:
            client.close()
