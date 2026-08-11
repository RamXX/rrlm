"""Multi-turn sessions: one persistent REPL namespace across solve() calls.

A one-shot :func:`rrlm.solve` pays the data load and the REPL scaffold every
call and forgets everything it computed. A :class:`Session` keeps the
predict-rlm supervisor interpreter - and therefore the whole REPL namespace:
variables, parsed structures, defined helpers - alive across calls, so
follow-up questions build on completed work instead of restarting from zero::

    from rrlm import Session

    with Session(main_model="openrouter/qwen/qwen3.6-27b") as session:
        session.solve("Parse the ledger into `entries` and report the row count.", data=text)
        session.solve("Using `entries` from before, total the amounts per vendor.")
        session.solve("Which vendor total changed the most vs `entries[0]`?")

Scope and honesty about what persists:

* What persists is the **interpreter namespace** (the supervisor runner
  process). Each call is still its own agent run: fresh conversation, fresh
  budgets, fresh trace. Cross-call knowledge lives in variables, not in chat
  history, so instructions should name the variables earlier calls created.
* Sessions run on the ``supervisor`` backend only (host CPython - trusted
  data). The sandboxed backends create their environment per run; for warm
  cross-run ``sbx`` containers (filesystem persistence, not namespace
  persistence) use ``RRLM_SBX_NAME`` instead.
* ``rlm_spawn`` children keep getting their own throwaway interpreters; child
  runs never read or pollute the session namespace.

The session owns the interpreter's lifetime: nothing is torn down between
calls, and :meth:`Session.close` (or the context manager) releases the runner
process. :meth:`Session.reset` clears the namespace without ending the
session.
"""

from __future__ import annotations

import asyncio
from typing import Any

from rrlm.solve import asolve, asolve_many

# Per-REPL-turn execution cap for session interpreters. Sessions default to the
# generous local-serving value (not the 300s cloud default) because a session's
# accumulated namespace makes later turns cheap but the first parse/load turn
# is often the heavy one, and killing it loses the whole point of the session.
DEFAULT_SESSION_EXEC_TIMEOUT = 3600.0


def _preserve_kernel_across_runs(interpreter) -> None:
    """Make tool re-registration kernel-preserving on this interpreter.

    The session namespace physically lives in the supervisor's *kernel* child
    process, and predict-rlm's ``register_tools`` request discards that kernel
    (a new one forks with the updated tool wrappers). dspy re-injects tools on
    every run of an injected interpreter (predict_rlm resets
    ``_tools_registered``), so without this patch every session turn would
    start by killing the namespace it is supposed to keep.

    Skipping the re-registration is behaviorally safe when the tool *name set*
    is unchanged: the kernel-side wrappers call host tools by name, and the
    host resolves ``interpreter.tools[name]`` at call time (runner.py), which
    dspy has already updated with the run's fresh closures. When the name set
    does change (say, web tools toggled on), registration proceeds and that
    turn starts a fresh namespace - correctness over persistence.

    Instance-level on purpose: one-shot interpreters keep stock behavior, and
    the patch rides along with whatever DirectPythonBackend subclass is
    installed. Raises when the predict-rlm internals it relies on are missing,
    because degrading silently would turn every session into an expensive
    one-shot.
    """
    required = ("_register_runtime", "_tools_registered", "tools")
    if not all(hasattr(interpreter, name) for name in required):
        raise RuntimeError(
            "installed predict-rlm is incompatible with rrlm sessions: the "
            f"supervisor backend no longer exposes {required!r}"
        )
    base_register = interpreter._register_runtime
    registered_names: set[str] | None = None

    def kernel_preserving_register() -> None:
        nonlocal registered_names
        names = set(interpreter.tools)
        if not interpreter._tools_registered and registered_names == names:
            interpreter._tools_registered = True  # same names: keep the kernel
        base_register()
        if interpreter._tools_registered:
            registered_names = names

    interpreter._register_runtime = kernel_preserving_register


class Session:
    """A multi-turn solve context over one persistent REPL namespace.

    Keyword arguments become per-session defaults forwarded to every call
    (``main_model``, ``sub_model``, budgets, ...); individual calls may
    override them. ``backend`` other than ``supervisor`` and ``engine`` are
    rejected: engines are stateless by contract and sandboxed backends do not
    persist a namespace across runs.
    """

    def __init__(
        self,
        *,
        exec_timeout: float = DEFAULT_SESSION_EXEC_TIMEOUT,
        **defaults: Any,
    ) -> None:
        backend = defaults.pop("backend", None)
        if backend not in (None, "supervisor"):
            raise NotImplementedError(
                f"sessions require backend='supervisor', got {backend!r}. Namespace "
                "persistence needs a live interpreter process; for warm sbx containers "
                "across one-shot runs use RRLM_SBX_NAME."
            )
        if defaults.get("engine"):
            raise ValueError("sessions cannot route to an engine: engines are stateless")
        if "interpreter" in defaults:
            raise ValueError("the session owns its interpreter; do not pass interpreter=")
        # Construction is cheap by design: the supervisor starts its runner
        # process lazily on the first executed turn, not here. Resolve the
        # class by attribute so tests can observe construction.
        import predict_rlm

        self._interpreter = predict_rlm.DirectPythonBackend(exec_timeout=exec_timeout)
        _preserve_kernel_across_runs(self._interpreter)
        self._defaults = defaults
        self._closed = False

    # -- solving -------------------------------------------------------------

    async def asolve(self, instruction: str, data: str = "", **kwargs: Any) -> dict:
        """One turn in this session (async). Same result dict as :func:`rrlm.asolve`."""
        return await asolve(
            instruction, data, **self._call_kwargs(kwargs)
        )

    def solve(self, instruction: str, data: str = "", **kwargs: Any) -> dict:
        """One turn in this session. Same result dict as :func:`rrlm.solve`."""
        return asyncio.run(self.asolve(instruction, data, **kwargs))

    async def asolve_many(self, instructions: list[str], data: str = "", **kwargs: Any) -> dict:
        """Several questions in one amortized turn (async); see :func:`rrlm.asolve_many`."""
        return await asolve_many(instructions, data, **self._call_kwargs(kwargs))

    def solve_many(self, instructions: list[str], data: str = "", **kwargs: Any) -> dict:
        """Several questions in one amortized turn; see :func:`rrlm.solve_many`."""
        return asyncio.run(self.asolve_many(instructions, data, **kwargs))

    def _call_kwargs(self, overrides: dict) -> dict:
        if self._closed:
            raise RuntimeError("session is closed")
        merged = {**self._defaults, **overrides}
        if merged.get("engine"):
            raise ValueError("sessions cannot route to an engine: engines are stateless")
        merged["backend"] = "supervisor"
        merged["interpreter"] = self._interpreter
        return merged

    # -- lifecycle -----------------------------------------------------------

    def reset(self) -> None:
        """Clear the REPL namespace; the session (and its process) stays usable."""
        if self._closed:
            raise RuntimeError("session is closed")
        self._interpreter.reset()

    def close(self) -> None:
        """Release the interpreter (terminates the runner process). Idempotent.

        Bounded: a stuck polite shutdown falls back to killing the runner
        (see :func:`rrlm.harness.shutdown_interpreter`).
        """
        if self._closed:
            return
        self._closed = True
        from rrlm.harness import shutdown_interpreter

        shutdown_interpreter(self._interpreter)

    async def aclose(self) -> None:
        """Async :meth:`close` (shutdown blocks briefly on process exit)."""
        if self._closed:
            return
        await asyncio.to_thread(self.close)

    def __enter__(self) -> Session:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    async def __aenter__(self) -> Session:
        return self

    async def __aexit__(self, *exc_info) -> None:
        await self.aclose()
