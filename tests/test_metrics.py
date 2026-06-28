"""Unit tests for metrics harvesting, reconciliation, and run logging.

Mocks are confined to these unit tests per project testing policy: the
OpenRouter HTTP API is mocked with respx; dspy.LM history entries are
simulated with plain stand-in objects.
"""

import json
from types import SimpleNamespace

import httpx
import respx

from rrlm.metrics import (
    CallRecord,
    RunLogger,
    harvest_lm_history,
    reconcile,
    summarize,
)
from rrlm.openrouter import GENERATION_URL, fetch_generation


def _fake_lm(entries):
    return SimpleNamespace(history=entries)


def _history_entry(gen_id="gen-abc", cost_inline=0.0123, litellm_cost=0.0119):
    usage = SimpleNamespace(
        prompt_tokens=1000,
        completion_tokens=200,
        total_tokens=1200,
        cost=cost_inline,
        model_extra={},
    )
    response = SimpleNamespace(id=gen_id, usage=usage, model="qwen/qwen3.7-max")
    return {
        "response": response,
        "model": "openrouter/qwen/qwen3.7-max",
        "cost": litellm_cost,
        "timestamp": "2026-06-11T00:00:00",
    }


class TestHarvest:
    def test_harvest_extracts_fields(self):
        lm = _fake_lm([_history_entry()])
        records = harvest_lm_history(lm, "main")
        assert len(records) == 1
        rec = records[0]
        assert rec.role == "main"
        assert rec.gen_id == "gen-abc"
        assert rec.prompt_tokens == 1000
        assert rec.cost_inline_usd == 0.0123
        assert rec.cost_litellm_usd == 0.0119

    def test_harvest_respects_start_index(self):
        lm = _fake_lm([_history_entry("gen-1"), _history_entry("gen-2")])
        records = harvest_lm_history(lm, "sub", start=1)
        assert [r.gen_id for r in records] == ["gen-2"]

    def test_harvest_inline_cost_from_model_extra(self):
        entry = _history_entry()
        entry["response"].usage.cost = None
        entry["response"].usage.model_extra = {"cost": 0.5}
        records = harvest_lm_history(_fake_lm([entry]), "main")
        assert records[0].cost_inline_usd == 0.5


class TestBestCost:
    def test_prefers_authoritative(self):
        rec = CallRecord(role="main", cost_usd=1.0, cost_inline_usd=2.0, cost_litellm_usd=3.0)
        assert rec.best_cost() == 1.0

    def test_falls_back_in_order(self):
        assert CallRecord(role="m", cost_inline_usd=2.0, cost_litellm_usd=3.0).best_cost() == 2.0
        assert CallRecord(role="m", cost_litellm_usd=3.0).best_cost() == 3.0
        assert CallRecord(role="m").best_cost() is None


class TestFetchGeneration:
    @respx.mock
    def test_retries_on_404_then_succeeds(self):
        route = respx.get(GENERATION_URL).mock(
            side_effect=[
                httpx.Response(404),
                httpx.Response(200, json={"data": {"total_cost": 0.01, "latency": 900}}),
            ]
        )
        data = fetch_generation("gen-x", "key", retry_delays=(0.0, 0.0))
        assert data == {"total_cost": 0.01, "latency": 900}
        assert route.call_count == 2

    @respx.mock
    def test_gives_up_after_retries(self):
        respx.get(GENERATION_URL).mock(return_value=httpx.Response(404))
        assert fetch_generation("gen-x", "key", retry_delays=(0.0,)) is None

    @respx.mock
    def test_auth_error_stops_immediately(self):
        route = respx.get(GENERATION_URL).mock(return_value=httpx.Response(401))
        assert fetch_generation("gen-x", "key", retry_delays=(0.0, 0.0)) is None
        assert route.call_count == 1


class TestGetHelper:
    def test_get_on_none_returns_default(self):
        from rrlm.metrics import _get

        assert _get(None, "x", "fallback") == "fallback"

    def test_get_on_dict_and_object(self):
        from rrlm.metrics import _get

        assert _get({"a": 1}, "a") == 1
        assert _get(SimpleNamespace(b=2), "b") == 2


class TestFetchGenerationErrors:
    @respx.mock
    def test_http_error_is_swallowed_and_returns_none(self):
        # A transport error on every attempt exhausts retries and yields None
        # (covers the httpx.HTTPError branch and the nonzero-delay sleep path).
        respx.get(GENERATION_URL).mock(side_effect=httpx.ConnectError("boom"))
        assert fetch_generation("gen-x", "key", retry_delays=(0.01,)) is None


class TestReconcile:
    def test_second_pass_fills_records_missed_on_first_pass(self, monkeypatch):
        # First fetch attempt misses (record not yet written), the delayed second
        # pass succeeds -- the exact path real long generations exercise.
        calls = {"n": 0}

        def fake_fetch(gen_id, api_key, *, client=None):
            calls["n"] += 1
            return None if calls["n"] == 1 else {"total_cost": 0.02}

        monkeypatch.setattr("rrlm.metrics.fetch_generation", fake_fetch)
        records = [CallRecord(role="main", gen_id="gen-late")]
        filled = reconcile(records, "key", second_pass_delay_s=0.01)
        assert filled == 1
        assert records[0].cost_usd == 0.02

    @respx.mock
    def test_fills_authoritative_fields(self):
        respx.get(GENERATION_URL).mock(
            return_value=httpx.Response(
                200,
                json={
                    "data": {
                        "total_cost": 0.0125,
                        "native_tokens_prompt": 1100,
                        "native_tokens_completion": 210,
                        "latency": 850,
                        "generation_time": 4200,
                        "provider_name": "Alibaba",
                        "finish_reason": "stop",
                    }
                },
            )
        )
        records = [CallRecord(role="main", gen_id="gen-abc")]
        filled = reconcile(records, "key")
        assert filled == 1
        rec = records[0]
        assert rec.cost_usd == 0.0125
        assert rec.native_tokens_prompt == 1100
        assert rec.latency_ms == 850
        assert rec.generation_time_ms == 4200
        assert rec.provider == "Alibaba"

    def test_skips_records_without_gen_id(self):
        records = [CallRecord(role="main")]
        assert reconcile(records, "key") == 0

    def test_skips_non_openrouter_gen_ids(self):
        # local endpoints (ollama/vllm) return foreign ids; never query OpenRouter
        records = [CallRecord(role="main", gen_id="chatcmpl-12345")]
        assert reconcile(records, "key") == 0


class TestSummarize:
    def test_aggregates_by_role(self):
        records = [
            CallRecord(role="main", prompt_tokens=100, completion_tokens=10, cost_usd=0.01,
                       generation_time_ms=1000, latency_ms=200),
            CallRecord(role="sub", prompt_tokens=50, completion_tokens=5, cost_usd=0.002,
                       generation_time_ms=500, latency_ms=100),
            CallRecord(role="sub", prompt_tokens=50, completion_tokens=5, cost_inline_usd=0.003),
        ]
        s = summarize(records)
        assert s["calls"] == 3
        assert s["by_role"]["main"]["calls"] == 1
        assert s["by_role"]["sub"]["calls"] == 2
        assert s["prompt_tokens"] == 200
        assert s["completion_tokens"] == 20
        assert abs(s["cost_usd"] - 0.015) < 1e-9
        assert s["cost_complete"] is True
        assert s["generation_time_ms"] == 1500

    def test_flags_incomplete_cost(self):
        s = summarize([CallRecord(role="main")])
        assert s["cost_complete"] is False


class TestRunLogger:
    def test_writes_all_artifacts(self, tmp_path):
        logger = RunLogger(tmp_path, "run-1")
        logger.write_meta({"run_id": "run-1", "model": "m"})
        logger.log_calls([CallRecord(role="main", gen_id="gen-1")])
        logger.write_result({"passed": True})
        logger.write_trace({"steps": []})

        run_dir = tmp_path / "run-1"
        assert json.loads((run_dir / "run.json").read_text())["model"] == "m"
        events = [json.loads(line) for line in (run_dir / "events.jsonl").read_text().splitlines()]
        assert events[0]["gen_id"] == "gen-1"
        assert json.loads((run_dir / "result.json").read_text())["passed"] is True
        assert (run_dir / "trace.json").exists()
