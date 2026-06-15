"""Per-run metrics: harvest LM call records, reconcile with OpenRouter, log JSONL.

Three cost figures are kept per call, in decreasing order of authority:
  cost_usd          from the OpenRouter generation endpoint (what was charged)
  cost_inline_usd   from the completion response usage.cost (also OpenRouter)
  cost_litellm_usd  LiteLLM's price-table estimate (least trusted)
`best_cost()` picks the most authoritative one available.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx

from rrlm.openrouter import fetch_generation


@dataclass
class CallRecord:
    role: str  # "main" | "sub" | "baseline"
    gen_id: str | None = None
    model: str | None = None
    timestamp: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    cost_inline_usd: float | None = None
    cost_litellm_usd: float | None = None
    # Filled by reconcile():
    cost_usd: float | None = None
    native_tokens_prompt: int | None = None
    native_tokens_completion: int | None = None
    latency_ms: float | None = None
    generation_time_ms: float | None = None
    provider: str | None = None
    finish_reason: str | None = None

    def best_cost(self) -> float | None:
        for value in (self.cost_usd, self.cost_inline_usd, self.cost_litellm_usd):
            if value is not None:
                return float(value)
        return None


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _inline_cost(usage: Any) -> float | None:
    """OpenRouter puts usage.cost on every response; LiteLLM may stash it in model_extra."""
    cost = _get(usage, "cost")
    if cost is None:
        extra = _get(usage, "model_extra") or {}
        cost = extra.get("cost") if isinstance(extra, dict) else None
    return cost


def harvest_lm_history(lm: Any, role: str, start: int = 0) -> list[CallRecord]:
    """Turn dspy.LM.history entries (from index `start`) into CallRecords."""
    records: list[CallRecord] = []
    for entry in lm.history[start:]:
        resp = _get(entry, "response")
        usage = _get(resp, "usage")
        records.append(
            CallRecord(
                role=role,
                gen_id=_get(resp, "id"),
                model=_get(entry, "model") or _get(resp, "model"),
                timestamp=str(_get(entry, "timestamp")),
                prompt_tokens=_get(usage, "prompt_tokens"),
                completion_tokens=_get(usage, "completion_tokens"),
                total_tokens=_get(usage, "total_tokens"),
                cost_inline_usd=_inline_cost(usage),
                cost_litellm_usd=_get(entry, "cost"),
            )
        )
    return records


def reconcile(
    records: list[CallRecord], api_key: str, *, second_pass_delay_s: float = 15.0
) -> int:
    """Fill authoritative cost/timing from the generation endpoint. Returns count filled.

    Long generations can take longer than the per-call retry window to appear in
    OpenRouter's ledger, so records that miss on the first pass get one delayed
    second attempt before we fall back to inline cost figures.
    """

    def _attempt(rec: CallRecord, client: httpx.Client) -> bool:
        data = fetch_generation(rec.gen_id, api_key, client=client)
        if not data:
            return False
        rec.cost_usd = data.get("total_cost")
        rec.native_tokens_prompt = data.get("native_tokens_prompt")
        rec.native_tokens_completion = data.get("native_tokens_completion")
        rec.latency_ms = data.get("latency")
        rec.generation_time_ms = data.get("generation_time")
        rec.provider = data.get("provider_name")
        rec.finish_reason = data.get("finish_reason")
        return True

    filled = 0
    with httpx.Client(timeout=15.0) as client:
        missed = []
        for rec in records:
            # only OpenRouter generations are reconcilable; local endpoints
            # produce foreign ids and would burn the whole retry window each
            if not rec.gen_id or not rec.gen_id.startswith("gen-"):
                continue
            if _attempt(rec, client):
                filled += 1
            else:
                missed.append(rec)
        if missed and second_pass_delay_s > 0:
            time.sleep(second_pass_delay_s)
            for rec in missed:
                if _attempt(rec, client):
                    filled += 1
    return filled


def summarize(records: list[CallRecord]) -> dict:
    """Aggregate totals for result.json, split by role."""
    summary: dict = {"calls": len(records), "by_role": {}}
    total_cost = 0.0
    cost_known = True
    for rec in records:
        role = summary["by_role"].setdefault(
            rec.role,
            {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0},
        )
        role["calls"] += 1
        role["prompt_tokens"] += rec.prompt_tokens or 0
        role["completion_tokens"] += rec.completion_tokens or 0
        cost = rec.best_cost()
        if cost is None:
            cost_known = False
        else:
            role["cost_usd"] += cost
            total_cost += cost
    summary["prompt_tokens"] = sum(r["prompt_tokens"] for r in summary["by_role"].values())
    summary["completion_tokens"] = sum(
        r["completion_tokens"] for r in summary["by_role"].values()
    )
    summary["cost_usd"] = total_cost
    summary["cost_complete"] = cost_known  # False if any call had no cost figure at all
    summary["generation_time_ms"] = sum(r.generation_time_ms or 0 for r in records)
    summary["latency_ms"] = sum(r.latency_ms or 0 for r in records)
    return summary


class RunLogger:
    """Writes one run's artifacts under runs/<run_id>/."""

    def __init__(self, runs_root: Path, run_id: str):
        self.run_id = run_id
        self.run_dir = runs_root / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def write_meta(self, meta: dict) -> None:
        self._write_json("run.json", meta)

    def log_calls(self, records: list[CallRecord]) -> None:
        path = self.run_dir / "events.jsonl"
        with path.open("a") as f:
            for rec in records:
                f.write(json.dumps(asdict(rec), default=str) + "\n")

    def write_result(self, result: dict) -> None:
        self._write_json("result.json", result)

    def write_trace(self, trace: Any) -> None:
        self._write_json("trace.json", trace)

    def _write_json(self, name: str, payload: Any) -> None:
        (self.run_dir / name).write_text(json.dumps(payload, indent=2, default=str))
