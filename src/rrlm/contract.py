"""The typed solve contract: the spine underneath the kwargs facade.

``rrlm.solve()`` / ``asolve()`` remain the friendly public API; internally
every call compiles into a :class:`SolveRequest`, runs through
:func:`rrlm.solve.arun`, and produces a :class:`SolveResult` that
``to_legacy_dict()`` renders into the exact dict shape solve() has always
returned. Servers, workflow engines, and agents that want machine-stable
types call the typed path directly::

    from rrlm import SolveRequest, arun

    request = SolveRequest(
        instruction="Total the amounts per vendor.",
        inputs={"data": text},
        policy=SolvePolicy(timeout_s=300, max_llm_calls=20),
    )
    result = await arun(request)
    if result.error and result.error.category == "budget":
        ...

Versioning: ``PROTOCOL`` names this contract (``rrlm.solve/1``). The v1 input
model is deliberately narrow - ``inputs`` accepts exactly one key, ``data``,
holding a string - so that richer input kinds (structured records, images,
remote resources) can be added in a later revision without ambiguity about
what v1 consumers were promised. Unknown input keys fail loudly today instead
of being silently stringified.

Error model: a failed *run* is a value, a bad *call* is an exception.
``SolveResult.error`` is a :class:`RunError` with a stable ``category``
(``timeout``, ``budget``, ``execution``, ``engine``); ``str(error)`` is the
same message string the legacy dict carries. Caller mistakes (missing files,
unknown engine, invalid parameter combinations) raise ``ValueError`` /
``FileNotFoundError`` / ``KeyError`` from the call itself and never appear in
a result.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

PROTOCOL = "rrlm.solve/1"

# The stable failure categories. "timeout": the wall-clock ceiling cancelled
# the run. "budget": a global call/cost ceiling was exhausted. "engine": an
# engine plugin reported the task unsolvable. "execution": everything else
# that failed inside the run.
ERROR_CATEGORIES = ("timeout", "budget", "execution", "engine")


@dataclass(frozen=True)
class RunError:
    """One failed run, typed. ``str(err)`` is the legacy error string."""

    category: str
    message: str

    def __post_init__(self) -> None:
        if self.category not in ERROR_CATEGORIES:
            raise ValueError(
                f"unknown error category {self.category!r}: one of {ERROR_CATEGORIES}"
            )

    def __str__(self) -> str:
        return self.message


@dataclass(frozen=True)
class Usage:
    """Aggregated run usage: typed accessors over a faithful raw mapping.

    The raw mapping round-trips exactly - the native harness reports the full
    ``summarize()`` shape while engines report sparse, engine-defined keys,
    and inventing zeros for absent keys would misreport "not measured" as
    "zero". The accessors give the common fields stable names and defaults.
    """

    raw: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> Usage:
        return cls(raw=dict(raw or {}))

    def to_dict(self) -> dict:
        return dict(self.raw)

    @property
    def calls(self) -> int:
        return int(self.raw.get("calls", 0))

    @property
    def prompt_tokens(self) -> int:
        return int(self.raw.get("prompt_tokens", 0))

    @property
    def completion_tokens(self) -> int:
        return int(self.raw.get("completion_tokens", 0))

    @property
    def cost_usd(self) -> float | None:
        value = self.raw.get("cost_usd")
        return None if value is None else float(value)

    @property
    def by_role(self) -> Mapping[str, Any]:
        return self.raw.get("by_role", {})


@dataclass(frozen=True)
class SolvePolicy:
    """Budgets and execution knobs for one run; all defaults match solve()."""

    backend: str | None = None
    max_depth: int = 2
    max_iterations: int = 30
    max_llm_calls: int = 50
    max_spawns: int = 16
    max_cost_usd: float | None = None
    max_action_retries: int = 2
    timeout_s: float | None = None
    reasoning: str | None = None
    temperature: float | None = None
    web: bool = False


@dataclass(frozen=True)
class ModelSelection:
    """The orchestrator and leaf models for one run.

    Each slot takes a Pi model reference string (``provider/model`` or a bare
    id), a ``dspy.LM`` instance (the injection seam for non-Pi embedders;
    budgets and accounting still apply - rrlm wraps foreign instances), or
    None (main: Pi's current default; sub: same as main). With an injected
    LM instance, ``SolvePolicy.reasoning`` must stay None: reasoning belongs
    on the LM you configured.
    """

    main: Any = None
    sub: Any = None


@dataclass(frozen=True)
class SolveRequest:
    """One solve, fully specified. See the module docstring for versioning."""

    instruction: str
    inputs: Mapping[str, Any] = field(default_factory=dict)
    files: tuple[str, ...] = ()
    answer_type: type | None = None
    engine: str | None = None
    engine_options: Mapping[str, Any] = field(default_factory=dict)
    policy: SolvePolicy = field(default_factory=SolvePolicy)
    models: ModelSelection = field(default_factory=ModelSelection)
    tools: tuple[Callable, ...] = ()
    skills: tuple = ()
    doctrine: str | None = None
    mcp: tuple = ()
    on_event: Callable[[dict], None] | None = None
    # The continuation slot: a caller-owned interpreter whose REPL namespace
    # outlives this run (what rrlm.Session passes). None = fresh per run.
    session: Any = None
    reconcile_cost: bool = True
    return_trace: bool = False

    def __post_init__(self) -> None:
        unknown = set(self.inputs) - {"data"}
        if unknown:
            raise ValueError(
                f"{PROTOCOL} supports only the 'data' input; got extra keys "
                f"{sorted(unknown)}. Richer input kinds arrive in a later "
                "contract revision; do not stringify them into 'data' silently."
            )
        data = self.inputs.get("data", "")
        if not isinstance(data, str):
            raise ValueError(
                f"{PROTOCOL} requires inputs['data'] to be a str, got {type(data).__name__}"
            )
        if self.engine:
            if self.session is not None:
                raise ValueError(
                    "engine= and a session are mutually exclusive: engines are "
                    "stateless by contract"
                )
            if self.mcp:
                raise ValueError("engine= and mcp= are mutually exclusive: engines own their tools")
        elif self.engine_options:
            raise ValueError("engine_options= requires engine=")

    @property
    def data(self) -> str:
        return self.inputs.get("data", "")


@dataclass(frozen=True)
class SolveResult:
    """One finished run, typed. ``to_legacy_dict()`` is the historic shape."""

    answer: Any = ""
    error: RunError | None = None
    wall_clock_s: float = 0.0
    trace_file: str | None = None
    spawn_stats: Mapping[str, Any] = field(default_factory=dict)
    usage: Usage = field(default_factory=Usage)
    config: Mapping[str, Any] = field(default_factory=dict)
    vendor: Mapping[str, Any] | None = None
    trace: Any = None

    def to_legacy_dict(self, *, include_trace: bool = False) -> dict:
        result = {
            "answer": self.answer,
            "error": str(self.error) if self.error is not None else None,
            "wall_clock_s": self.wall_clock_s,
            "trace_file": self.trace_file,
            "spawn_stats": dict(self.spawn_stats),
            "usage": self.usage.to_dict(),
            "config": dict(self.config),
        }
        if self.vendor:
            result["vendor"] = dict(self.vendor)
        if include_trace:
            result["trace"] = self.trace
        return result
