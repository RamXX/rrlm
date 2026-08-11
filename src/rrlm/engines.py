"""Engine plugins: alternate solvers behind the rrlm solve() contract.

An *engine* is any solver that accepts (instruction, data, files, budget) and
returns a typed answer plus usage - the same contract the built-in predict-rlm
harness fills natively. Engines let a deployment route specific runs (sealed
interpreters, audited symbolic solvers, anything) to another backend without
rrlm knowing what that backend is: rrlm documents only the protocol here, and
engine packages register themselves at install time through the
``rrlm.engines`` entry-point group.

Selection is always explicit: ``solve(..., engine="<name>")``,
``rrlm-solve --engine <name>``, or the ``RRLM_ENGINE`` environment variable.
No heuristic ever picks an engine. Routing between trust levels (host Python
vs a sealed interpreter) is a policy decision that belongs to the caller, not
to inference over the payload.

Writing an engine package::

    # myengine/engine.py
    from rrlm.engines import Engine, EngineCapabilities, EngineRequest, EngineResult

    class MyEngine:
        name = "myengine"
        capabilities = EngineCapabilities(description="...", sealed=True)

        async def solve(self, request: EngineRequest) -> EngineResult:
            ...

    # pyproject.toml
    [project.entry-points."rrlm.engines"]
    myengine = "myengine.engine:MyEngine"

The entry point value is a zero-argument callable returning an Engine (a class
works). Engine construction must be cheap; expensive setup (subprocesses,
connections) belongs in the first ``solve()`` call. Validate an implementation
against :mod:`rrlm.conformance` from the engine package's own test suite.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from importlib.metadata import entry_points
from typing import Any, Protocol, runtime_checkable

ENTRY_POINT_GROUP = "rrlm.engines"

# The engine contract version. Engine packages freeze on this protocol, so it
# only changes with the contract itself: a breaking change to EngineRequest,
# EngineResult, or the solve() semantics bumps the suffix, and an engine MAY
# declare the protocol it was built against via a `protocol` class attribute -
# the conformance suite then fails fast on a mismatch instead of letting a
# stale adapter mis-parse requests at runtime.
ENGINE_PROTOCOL = "rrlm.engine/1"


@dataclass(frozen=True)
class BudgetLease:
    """The slice of the run's budget an engine may spend.

    ``timeout_s`` is enforced host-side (rrlm cancels the run on overrun), so
    it holds even against a misbehaving engine. The call and cost ceilings are
    the engine's obligation: rrlm cannot meter an out-of-process solver, so it
    debits whatever the engine reports in ``EngineResult.usage``. ``None``
    means unlimited.
    """

    timeout_s: float | None = None
    max_llm_calls: int | None = None
    max_cost_usd: float | None = None


@dataclass(frozen=True)
class EngineRequest:
    """One solve request, engine-agnostic.

    ``answer_type`` is the Python type the caller wants the answer parsed into
    (the same values ``solve(answer_type=...)`` accepts); adapters translate it
    to whatever their engine speaks (e.g. a JSON Schema). ``options`` is an
    opaque engine-specific passthrough - rrlm never inspects it.
    """

    instruction: str
    data: str = ""
    files: tuple[str, ...] = ()
    answer_type: type | None = None
    budget: BudgetLease = field(default_factory=BudgetLease)
    trace_dir: str | None = None
    options: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class EngineResult:
    """What an engine returns. Failures go in ``error``, not exceptions:
    an engine that raises is treated as broken, not as unable to answer.

    ``usage`` is engine-defined; the recommended keys are ``llm_calls``,
    ``prompt_tokens``, ``completion_tokens``, and ``cost_usd`` so budget
    debiting and reporting stay meaningful. ``vendor`` carries engine-specific
    artifacts (audit ledgers, proof trees, ...) opaquely - rrlm passes it
    through to the caller untyped and never documents its schema.
    """

    answer: Any = ""
    error: str | None = None
    usage: dict = field(default_factory=dict)
    trace_file: str | None = None
    vendor: dict = field(default_factory=dict)


@dataclass(frozen=True)
class EngineCapabilities:
    """Self-description an engine exposes for doctor/listing purposes.

    ``sealed`` means generated logic executes in a closed interpreter with no
    host filesystem/network access (safe for untrusted payloads). ``audited``
    means the engine produces an audit trail beyond a raw execution trace.
    Both are claims by the engine, surfaced verbatim; rrlm does not verify.
    """

    description: str = ""
    sealed: bool = False
    audited: bool = False


@runtime_checkable
class Engine(Protocol):
    """The protocol every engine implements. See the module docstring."""

    name: str
    capabilities: EngineCapabilities

    async def solve(self, request: EngineRequest) -> EngineResult: ...


class ReferenceEngine:
    """The in-tree reference engine: deterministic, offline, model-free.

    It exists to make the protocol testable and documentable without naming
    any real engine: docs and the conformance suite run against it, and its
    tiny instruction set exercises every field of the contract (typed answers,
    list answers, the error path) with zero dependencies and zero network.

    Instructions: ``count-lines``, ``count-chars``, ``search:<term>`` (lines
    containing term, as a list), ``echo`` (the data back). Anything else is
    reported through ``EngineResult.error``.
    """

    name = "reference"
    capabilities = EngineCapabilities(
        description="deterministic offline reference engine (protocol demo, no LLM)",
        sealed=False,
        audited=False,
    )

    async def solve(self, request: EngineRequest) -> EngineResult:
        instruction = request.instruction.strip()
        data = request.data
        answer: Any
        if instruction == "count-lines":
            answer = len(data.splitlines())
        elif instruction == "count-chars":
            answer = len(data)
        elif instruction.startswith("search:"):
            term = instruction.removeprefix("search:").strip()
            answer = [line for line in data.splitlines() if term in line]
        elif instruction == "echo":
            answer = data
        else:
            return EngineResult(
                error=f"reference engine cannot solve: {instruction!r} "
                      "(supported: count-lines, count-chars, search:<term>, echo)",
                usage={"llm_calls": 0, "cost_usd": 0.0},
            )
        if request.answer_type is str and not isinstance(answer, str):
            answer = str(answer)
        return EngineResult(answer=answer, usage={"llm_calls": 0, "cost_usd": 0.0})


# The registry. Built-ins ship with rrlm; entry points add installed engine
# packages; register_engine() adds in-process engines (embedded apps, tests).
# Entry points may not shadow an existing name - a silent override of, say,
# "reference" would make every environment mean something different by the
# same name, which violates least surprise. Collisions fail loudly instead.
_BUILTIN: dict[str, Callable[[], Engine]] = {ReferenceEngine.name: ReferenceEngine}
_REGISTERED: dict[str, Callable[[], Engine]] = {}


def register_engine(name: str, factory: Callable[[], Engine], *, replace: bool = False) -> None:
    """Register an engine factory in-process (no packaging needed).

    For embedded applications and tests. ``replace=True`` allows overwriting a
    previous in-process registration; built-ins can never be replaced.
    """
    if name in _BUILTIN:
        raise ValueError(f"cannot replace built-in engine {name!r}")
    if name in _REGISTERED and not replace:
        raise ValueError(f"engine {name!r} is already registered (pass replace=True to override)")
    _REGISTERED[name] = factory


def unregister_engine(name: str) -> None:
    """Remove an in-process registration made with :func:`register_engine`."""
    _REGISTERED.pop(name, None)


def _entry_point_factories() -> dict[str, Callable[[], Engine]]:
    """Load engine factories from installed packages (the ``rrlm.engines`` group).

    A collision with a built-in or another entry point is a broken install and
    raises; a single unloadable entry point must not take down every engine,
    so load errors surface per-name when that engine is actually requested.
    """
    factories: dict[str, Callable[[], Engine]] = {}
    for ep in entry_points(group=ENTRY_POINT_GROUP):
        if ep.name in _BUILTIN and ep.value != f"{ReferenceEngine.__module__}:ReferenceEngine":
            raise ValueError(
                f"entry point {ep.value!r} tries to shadow built-in engine {ep.name!r}"
            )
        if ep.name in factories:
            raise ValueError(f"two installed packages both provide engine {ep.name!r}")
        if ep.name in _BUILTIN:
            continue  # rrlm's own dist metadata re-lists a built-in; harmless
        factories[ep.name] = _make_lazy_loader(ep)
    return factories


def _make_lazy_loader(ep) -> Callable[[], Engine]:
    def load() -> Engine:
        factory = ep.load()
        return factory()

    return load


def available_engines() -> dict[str, Callable[[], Engine]]:
    """All resolvable engine factories by name: built-ins, entry points, and
    in-process registrations (which win over an entry point of the same name,
    because an explicit in-process choice is more deliberate than an install).
    """
    merged: dict[str, Callable[[], Engine]] = dict(_BUILTIN)
    merged.update(_entry_point_factories())
    merged.update(_REGISTERED)
    return merged


def get_engine(name: str) -> Engine:
    """Instantiate the engine registered under ``name``.

    Raises ``KeyError`` with the available names when unknown, and lets any
    construction error propagate untouched - a broken engine package should
    fail loudly at selection time, not degrade into another engine.
    """
    engines = available_engines()
    if name not in engines:
        known = ", ".join(sorted(engines)) or "none"
        raise KeyError(f"unknown engine {name!r} (available: {known})")
    return engines[name]()
