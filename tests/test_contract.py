"""The typed solve contract: requests, results, errors, options, injection.

The legacy-shape guarantee is pinned by the entire existing suite (every
integration test asserts the historic dict); these tests cover what is new:
the typed path (arun/run), the stable error categories, engine_options
passthrough, the engine protocol version check, and dspy.LM injection with
budget-preserving adoption.
"""

from __future__ import annotations

import asyncio
import importlib

import dspy
import pytest

from rrlm import solve
from rrlm.conformance import check_engine_sync
from rrlm.contract import (
    RunError,
    SolvePolicy,
    SolveRequest,
    SolveResult,
    Usage,
)
from rrlm.engines import (
    ENGINE_PROTOCOL,
    EngineResult,
    ReferenceEngine,
    register_engine,
)
from rrlm.harness import SharedLM, ensure_shared_lm
from rrlm.solve import run

S = importlib.import_module("rrlm.solve")

DATA = "alpha\nbeta\ngamma\n"


@pytest.fixture(autouse=True)
def _clean_registrations():
    from rrlm import engines as E

    E._REGISTERED.clear()
    yield
    E._REGISTERED.clear()


# --- request validation ------------------------------------------------------


def test_request_rejects_unknown_inputs():
    with pytest.raises(ValueError, match="only the 'data' input"):
        SolveRequest(instruction="x", inputs={"data": "", "image": b".."})


def test_request_rejects_non_string_data():
    with pytest.raises(ValueError, match="to be a str"):
        SolveRequest(instruction="x", inputs={"data": {"rows": []}})


def test_request_rejects_engine_options_without_engine():
    with pytest.raises(ValueError, match="requires engine="):
        SolveRequest(instruction="x", engine_options={"k": "v"})


def test_run_error_rejects_unknown_category():
    with pytest.raises(ValueError, match="unknown error category"):
        RunError("mystery", "boom")


def test_usage_round_trips_sparse_dicts_exactly():
    sparse = {"llm_calls": 0, "cost_usd": 0.0}
    assert Usage.from_dict(sparse).to_dict() == sparse
    assert Usage.from_dict(sparse).calls == 0  # typed accessor default


def test_legacy_dict_shape_and_optional_fields():
    result = SolveResult(
        answer="42",
        error=RunError("engine", "nope"),
        wall_clock_s=1.5,
        usage=Usage.from_dict({"calls": 1}),
        config={"engine": "x"},
        vendor={"ledger": [1]},
        trace=object(),
    )
    legacy = result.to_legacy_dict()
    assert legacy["error"] == "nope" and "trace" not in legacy
    assert legacy["vendor"] == {"ledger": [1]}
    with_trace = result.to_legacy_dict(include_trace=True)
    assert with_trace["trace"] is result.trace


# --- the typed path over the reference engine --------------------------------


def _request(**kwargs) -> SolveRequest:
    return SolveRequest(instruction="count-lines", inputs={"data": DATA}, **kwargs)


def test_run_returns_typed_result():
    result = run(_request(engine="reference"))
    assert isinstance(result, SolveResult)
    assert result.answer == 3 and result.error is None
    assert result.usage.calls == 0  # model-free engine
    assert result.config == {"engine": "reference"}


def test_engine_failure_is_category_engine():
    result = run(
        SolveRequest(instruction="unsupported-op", inputs={"data": DATA}, engine="reference")
    )
    assert result.error is not None
    assert result.error.category == "engine"
    assert "cannot solve" in str(result.error)


def test_engine_timeout_is_category_timeout():
    class Slow(ReferenceEngine):
        name = "slow-typed"

        async def solve(self, request):
            await asyncio.sleep(30)
            return EngineResult(answer="late")

    register_engine("slow-typed", Slow)
    result = run(
        SolveRequest(
            instruction="x", inputs={"data": DATA}, engine="slow-typed",
            policy=SolvePolicy(timeout_s=0.2),
        )
    )
    assert result.error is not None and result.error.category == "timeout"


# --- engine_options passthrough ----------------------------------------------


def test_engine_options_reach_the_engine():
    seen: dict = {}

    class Capture(ReferenceEngine):
        name = "capture-options"

        async def solve(self, request):
            seen.update(request.options)
            return EngineResult(answer="ok")

    register_engine("capture-options", Capture)
    result = solve(
        "x", DATA, engine="capture-options",
        engine_options={"profile": "audit", "depth": 3},
    )
    assert result["answer"] == "ok"
    assert seen == {"profile": "audit", "depth": 3}


def test_cli_engine_option_parses_json_values(monkeypatch, capsys):
    seen: dict = {}

    class Capture(ReferenceEngine):
        name = "cli-options"

        async def solve(self, request):
            seen.update(request.options)
            return EngineResult(answer="ok")

    register_engine("cli-options", Capture)
    monkeypatch.setattr(
        "sys.argv",
        ["rrlm-solve", "--engine", "cli-options",
         "--engine-option", "profile=audit",
         "--engine-option", "depth=3",
         "--engine-option", "flags=[1, 2]",
         "-i", "x", "-d", DATA],
    )
    S.main()
    assert capsys.readouterr().out.strip() == "ok"
    assert seen == {"profile": "audit", "depth": 3, "flags": [1, 2]}


# --- engine protocol version -------------------------------------------------


def test_conformance_accepts_matching_or_absent_protocol():
    class Declared(ReferenceEngine):
        protocol = ENGINE_PROTOCOL

    assert check_engine_sync(Declared()) == []
    assert check_engine_sync(ReferenceEngine()) == []  # absent = accepted


def test_conformance_rejects_stale_protocol():
    class Stale(ReferenceEngine):
        protocol = "rrlm.engine/0"

    failures = check_engine_sync(Stale())
    assert any("rebuild the engine" in f for f in failures)


# --- dspy.LM injection -------------------------------------------------------


def test_ensure_shared_lm_wraps_plain_lm():
    plain = dspy.LM("openai/some-model", api_key="k", max_tokens=512, cache=True)
    shared = ensure_shared_lm(plain)
    assert isinstance(shared, SharedLM)
    assert shared.model == "openai/some-model"
    assert shared.cache is False  # caching would falsify usage accounting
    assert shared.copy() is shared  # identity copy = accounting works


def test_ensure_shared_lm_passthrough_and_copy():
    shared = SharedLM("openai/m", api_key="k", cache=False)
    assert ensure_shared_lm(shared) is shared
    sibling = ensure_shared_lm(shared, copy=True)
    assert isinstance(sibling, SharedLM) and sibling is not shared


def test_injected_lm_plus_reasoning_raises():
    lm = dspy.LM("openai/m", api_key="k")
    with pytest.raises(ValueError, match="injected dspy.LM"):
        solve("x", DATA, main_model=lm, reasoning="high")


@pytest.mark.integration
def test_injected_plain_lm_solves_with_accounting(stub_base_url):
    """A foreign dspy.LM works end to end AND its calls are accounted.

    No Pi config exists in this test: the model reaches rrlm only as an
    injected instance. usage.calls > 0 is the SharedLM-adoption proof - a
    plain LM's private copies would leave the harvested history empty.
    """
    lm = dspy.LM(
        "openai/stub-model",
        api_key="stub-key",
        api_base=f"{stub_base_url}/submit/v1",
        max_tokens=4096,
        cache=False,
    )
    result = solve(
        "compute from data", DATA,
        main_model=lm, backend="supervisor", max_iterations=5,
    )
    assert result["error"] is None
    assert result["answer"] == str(len(DATA))
    assert result["usage"]["calls"] >= 1
    assert result["config"]["main_model"] == "openai/stub-model"
