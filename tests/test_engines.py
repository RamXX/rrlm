"""Engine plugin tests: registry, reference engine, conformance, solve routing.

Unit tests cover the registry and the reference engine directly; the solve()
routing tests run the real engine path in-process (no model resolution, no
network - that is the point of the engine path); the e2e test drives the
installed ``rrlm-solve`` console script as a subprocess with ``--engine``,
fully offline, exactly the way an engine user would.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from rrlm import engines as E
from rrlm.conformance import Probe, check_engine_sync
from rrlm.engines import (
    BudgetLease,
    EngineCapabilities,
    EngineRequest,
    EngineResult,
    ReferenceEngine,
    available_engines,
    get_engine,
    register_engine,
    unregister_engine,
)

S = importlib.import_module("rrlm.solve")

DATA = "alpha\nbeta error\ngamma\n"


@pytest.fixture(autouse=True)
def _clean_registrations():
    """Every test starts and ends with an empty in-process registry."""
    E._REGISTERED.clear()
    yield
    E._REGISTERED.clear()


# --- registry ---------------------------------------------------------------


def test_reference_engine_is_always_available():
    assert "reference" in available_engines()
    eng = get_engine("reference")
    assert eng.name == "reference"
    assert isinstance(eng.capabilities, EngineCapabilities)


def test_unknown_engine_error_lists_available_names():
    with pytest.raises(KeyError, match="unknown engine 'nope'.*reference"):
        get_engine("nope")


def test_register_get_unregister_roundtrip():
    class Custom(ReferenceEngine):
        name = "custom"

    register_engine("custom", Custom)
    assert get_engine("custom").name == "custom"
    unregister_engine("custom")
    with pytest.raises(KeyError):
        get_engine("custom")


def test_duplicate_registration_requires_replace():
    register_engine("dup", ReferenceEngine)
    with pytest.raises(ValueError, match="already registered"):
        register_engine("dup", ReferenceEngine)
    register_engine("dup", ReferenceEngine, replace=True)  # explicit override is fine


def test_builtin_names_cannot_be_replaced():
    with pytest.raises(ValueError, match="built-in"):
        register_engine("reference", ReferenceEngine)


# --- reference engine -------------------------------------------------------


def _solve_ref(instruction: str, **kwargs) -> EngineResult:
    request = EngineRequest(instruction=instruction, data=DATA, **kwargs)
    return asyncio.run(ReferenceEngine().solve(request))


def test_reference_count_lines():
    assert _solve_ref("count-lines").answer == 3


def test_reference_count_chars():
    assert _solve_ref("count-chars").answer == len(DATA)


def test_reference_search_returns_matching_lines():
    assert _solve_ref("search:error").answer == ["beta error"]


def test_reference_echo_returns_data():
    assert _solve_ref("echo").answer == DATA


def test_reference_coerces_str_answer_type():
    result = _solve_ref("count-lines", answer_type=str)
    assert result.answer == "3"


def test_reference_reports_unsupported_via_error_not_raise():
    result = _solve_ref("summarize the vibes")
    assert result.error is not None and "cannot solve" in result.error
    assert result.usage == {"llm_calls": 0, "cost_usd": 0.0}


# --- conformance ------------------------------------------------------------

REF_PROBES = [
    Probe("count-lines", data=DATA, expected=3),
    Probe("search:error", data=DATA, expected=["beta error"]),
]


def test_reference_engine_passes_conformance():
    assert check_engine_sync(ReferenceEngine(), REF_PROBES) == []


def test_conformance_flags_raising_engine():
    class Raising:
        name = "raising"
        capabilities = EngineCapabilities()

        async def solve(self, request):
            raise RuntimeError("boom")

    failures = check_engine_sync(Raising(), [Probe("anything")])
    assert any("must report failures via EngineResult.error" in f for f in failures)
    assert any("solve() raised RuntimeError" in f for f in failures)


def test_conformance_flags_shape_violations():
    class Shapeless:
        name = ""
        capabilities = None

        def solve(self, request):  # not async: the contract requires a coroutine
            return EngineResult(answer="x")

    failures = check_engine_sync(Shapeless())
    assert any("non-empty str" in f for f in failures)
    assert any("EngineCapabilities" in f for f in failures)
    assert any("async function" in f for f in failures)


def test_conformance_flags_wrong_answer():
    failures = check_engine_sync(ReferenceEngine(), [Probe("count-lines", data=DATA, expected=99)])
    assert any("!= expected 99" in f for f in failures)


# --- solve() routing --------------------------------------------------------


def test_asolve_routes_to_engine():
    result = asyncio.run(S.asolve("count-lines", DATA, engine="reference"))
    assert result["answer"] == 3
    assert result["error"] is None
    assert result["config"] == {"engine": "reference"}
    assert result["spawn_stats"] == {}
    assert result["usage"] == {"llm_calls": 0, "cost_usd": 0.0}


def test_asolve_engine_error_is_returned_not_raised():
    result = asyncio.run(S.asolve("no such op", DATA, engine="reference"))
    assert result["answer"] == ""
    assert "cannot solve" in result["error"]


def test_asolve_unknown_engine_raises_loudly():
    # A typo'd engine name is a caller bug, not a run failure: it must raise
    # before anything executes, never degrade into the native path.
    with pytest.raises(KeyError, match="unknown engine"):
        asyncio.run(S.asolve("count-lines", DATA, engine="typo"))


def test_asolve_enforces_timeout_on_slow_engine():
    class Slow(ReferenceEngine):
        name = "slow"

        async def solve(self, request):
            await asyncio.sleep(30)
            return EngineResult(answer="late")

    register_engine("slow", Slow)
    result = asyncio.run(S.asolve("anything", DATA, engine="slow", timeout_s=0.2))
    assert "TimeoutError" in result["error"]
    assert result["wall_clock_s"] < 5


def test_asolve_engine_lease_carries_budgets():
    seen: dict = {}

    class Capture(ReferenceEngine):
        name = "capture"

        async def solve(self, request):
            seen["budget"] = request.budget
            seen["files"] = request.files
            return EngineResult(answer="ok")

    register_engine("capture", Capture)
    result = asyncio.run(
        S.asolve("x", DATA, engine="capture", timeout_s=90.0, max_llm_calls=7, max_cost_usd=1.5)
    )
    assert result["answer"] == "ok"
    assert seen["budget"] == BudgetLease(timeout_s=90.0, max_llm_calls=7, max_cost_usd=1.5)
    assert seen["files"] == ()


def test_asolve_engine_vendor_blob_passes_through_opaquely():
    class Vendored(ReferenceEngine):
        name = "vendored"

        async def solve(self, request):
            return EngineResult(answer="ok", vendor={"ledger": [1, 2, 3]})

    register_engine("vendored", Vendored)
    result = asyncio.run(S.asolve("x", DATA, engine="vendored"))
    assert result["vendor"] == {"ledger": [1, 2, 3]}


def test_asolve_engine_missing_file_fails_fast(tmp_path):
    with pytest.raises(FileNotFoundError):
        asyncio.run(
            S.asolve("count-lines", DATA, engine="reference", files=[tmp_path / "absent.pdf"])
        )


def test_asolve_engine_run_lands_in_trace_index(tmp_path, monkeypatch):
    monkeypatch.setenv("RRLM_TRACE_DIR", str(tmp_path))
    asyncio.run(S.asolve("count-lines", DATA, engine="reference"))
    index = tmp_path / "index.jsonl"
    assert index.is_file()
    rec = json.loads(index.read_text().splitlines()[-1])
    assert rec["engine"] == "reference"
    assert rec["config"] == {"engine": "reference"}
    assert rec["answer"] == "3"
    assert rec["error"] is None


def test_doctor_lists_engines():
    from rrlm.doctor import _check_engines

    lines: list[str] = []
    _check_engines(lines)
    assert lines[0] == "engines"
    assert any("reference" in line for line in lines[1:])


# --- CLI + e2e --------------------------------------------------------------


def _cli() -> list[str]:
    exe = shutil.which("rrlm-solve") or str(Path(sys.executable).parent / "rrlm-solve")
    return [exe]


def _subprocess_env() -> dict:
    """A clean env: no RRLM_* leakage, no credentials needed by the engine path."""
    env = os.environ.copy()
    for var in list(env):
        if var.startswith("RRLM_"):
            del env[var]
    return env


def test_main_engine_env_var_routes(monkeypatch, capsys):
    monkeypatch.setenv("RRLM_ENGINE", "reference")
    monkeypatch.setattr("sys.argv", ["rrlm-solve", "-i", "count-lines", "-d", DATA])
    S.main()
    assert capsys.readouterr().out.strip() == "3"


@pytest.mark.e2e
def test_cli_engine_flag_end_to_end(tmp_path):
    payload = tmp_path / "payload.txt"
    payload.write_text(DATA)
    proc = subprocess.run(
        _cli() + ["--engine", "reference", "-i", "search:error", "-d", f"@{payload}", "--json"],
        capture_output=True, text=True, timeout=60, env=_subprocess_env(),
    )
    assert proc.returncode == 0, proc.stderr
    result = json.loads(proc.stdout)
    assert result["answer"] == ["beta error"]
    assert result["error"] is None
    assert result["config"] == {"engine": "reference"}


@pytest.mark.e2e
def test_cli_engine_error_exits_nonzero(tmp_path):
    proc = subprocess.run(
        _cli() + ["--engine", "reference", "-i", "unsupported-op", "-d", "x"],
        capture_output=True, text=True, timeout=60, env=_subprocess_env(),
    )
    assert proc.returncode == 1
    assert "ERROR:" in proc.stderr and "cannot solve" in proc.stderr
