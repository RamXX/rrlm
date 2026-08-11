"""Protocol conformance checks for rrlm engine implementations.

Engine packages run this from their own test suites so the protocol contract
is verified where the engine lives, without the engine ever appearing in
rrlm's tree, docs, or CI::

    from rrlm.conformance import Probe, check_engine_sync

    def test_conformance():
        failures = check_engine_sync(
            MyEngine(),
            probes=[Probe("count the rows", data="a\\nb", expected=2, answer_type=int)],
        )
        assert not failures, "\\n".join(failures)

A *probe* is a task the engine author knows their engine can solve, with the
expected answer; the suite cannot invent tasks for an arbitrary engine, so
semantic coverage is the author's contribution and protocol-shape coverage is
this module's. Checks are collected, not raised, so one report lists every
violation at once.
"""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass, field
from typing import Any

from rrlm.engines import BudgetLease, Engine, EngineCapabilities, EngineRequest, EngineResult


@dataclass(frozen=True)
class Probe:
    """One task the engine under test is known to solve, with its answer."""

    instruction: str
    data: str = ""
    expected: Any = None
    answer_type: type | None = None
    options: dict = field(default_factory=dict)


async def check_engine(
    engine: Engine,
    probes: tuple[Probe, ...] | list[Probe] = (),
    *,
    timeout_s: float = 60.0,
) -> list[str]:
    """Run every conformance check; return failure descriptions (empty = pass).

    ``timeout_s`` bounds each probe the same way rrlm bounds a real run, so an
    engine that hangs fails conformance instead of hanging the suite.
    """
    failures: list[str] = []

    # -- protocol shape ------------------------------------------------------
    name = getattr(engine, "name", None)
    if not isinstance(name, str) or not name:
        failures.append(f"engine.name must be a non-empty str, got {name!r}")
    caps = getattr(engine, "capabilities", None)
    if not isinstance(caps, EngineCapabilities):
        failures.append(f"engine.capabilities must be an EngineCapabilities, got {type(caps)!r}")
    solve = getattr(engine, "solve", None)
    if not callable(solve) or not inspect.iscoroutinefunction(solve):
        failures.append("engine.solve must be an async function (async def solve(request))")
        return failures  # nothing below can run

    # -- the error contract: unsolvable input is reported, never raised ------
    nonsense = EngineRequest(
        instruction="rrlm-conformance: intentionally unsupported instruction 4f1b",
        budget=BudgetLease(timeout_s=timeout_s),
    )
    try:
        result = await asyncio.wait_for(engine.solve(nonsense), timeout=timeout_s)
    except Exception as exc:  # noqa: BLE001, the whole point is that this must not happen
        failures.append(
            f"solve() raised {type(exc).__name__} on an unsupported instruction; "
            "engines must report failures via EngineResult.error"
        )
    else:
        if not isinstance(result, EngineResult):
            failures.append(f"solve() must return an EngineResult, got {type(result)!r}")
        elif result.error is None and result.answer in ("", None):
            failures.append(
                "unsupported instruction produced neither an answer nor an error; "
                "set EngineResult.error when a task cannot be solved"
            )

    # -- author-supplied probes: semantics + result field types --------------
    for probe in probes:
        request = EngineRequest(
            instruction=probe.instruction,
            data=probe.data,
            answer_type=probe.answer_type,
            budget=BudgetLease(timeout_s=timeout_s),
            options=probe.options,
        )
        label = f"probe {probe.instruction!r}"
        try:
            result = await asyncio.wait_for(engine.solve(request), timeout=timeout_s)
        except TimeoutError:
            failures.append(f"{label}: exceeded the {timeout_s}s lease")
            continue
        except Exception as exc:  # noqa: BLE001, collected into the report
            failures.append(f"{label}: solve() raised {type(exc).__name__}: {exc}")
            continue
        if not isinstance(result, EngineResult):
            failures.append(f"{label}: expected an EngineResult, got {type(result)!r}")
            continue
        if result.error is not None:
            failures.append(f"{label}: unexpected error: {result.error}")
            continue
        if probe.expected is not None and result.answer != probe.expected:
            failures.append(f"{label}: answer {result.answer!r} != expected {probe.expected!r}")
        if not isinstance(result.usage, dict):
            failures.append(f"{label}: usage must be a dict, got {type(result.usage)!r}")
        if not isinstance(result.vendor, dict):
            failures.append(f"{label}: vendor must be a dict, got {type(result.vendor)!r}")

    return failures


def check_engine_sync(
    engine: Engine,
    probes: tuple[Probe, ...] | list[Probe] = (),
    *,
    timeout_s: float = 60.0,
) -> list[str]:
    """Synchronous wrapper around :func:`check_engine` for plain test functions."""
    return asyncio.run(check_engine(engine, probes, timeout_s=timeout_s))
