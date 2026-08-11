"""General-purpose RLM-first solve entry point, the backend Pi delegates to.

Given an instruction and a (possibly large) data payload, run the RLM-first
agent: the data lands in the REPL, the orchestrator writes code to probe it,
fans out cheap sub-LM classification only when the data is irreducible, and
returns a verified answer. The data never enters the orchestrator's context.

Models are resolved from your Pi config (``rrlm.pi_config``): pass a Pi model
reference (``provider/model`` or a bare model id), or omit ``--main`` to use the
model Pi is currently set to. ``--sub`` defaults to the same model as ``--main``;
point it at a cheaper non-thinking model to make the fan-out path inexpensive.

CLI:
    rrlm-solve --instruction "..." --data @path/to/file
    echo "<data>" | rrlm-solve --instruction "..." --data -
    rrlm-solve -i "..." -d "inline text" --main openrouter/qwen/qwen3.6-27b --json
    rrlm-solve -i "total?" -i "how many rows?" -d @orders.csv        # multi-question
    rrlm-solve -i "sum the amounts" --file invoices.pdf --answer-type float

Library:
    from rrlm import solve, asolve, solve_many
    result = solve("Which product has the most negative reviews?", data=text)
    print(result["answer"])
    result = await asolve(...)          # same signature, for async callers
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from collections.abc import Callable
from contextlib import AsyncExitStack
from pathlib import Path

from rrlm.config import BACKENDS, HarnessConfig, load_env, resolve_backend
from rrlm.contract import (
    ModelSelection,
    RunError,
    SolvePolicy,
    SolveRequest,
    SolveResult,
    Usage,
)
from rrlm.events import EventEmitter
from rrlm.harness import (
    BudgetExceededError,
    RunBudget,
    build_lm,
    build_rlm,
    document_skills_for,
    ensure_shared_lm,
    lm_is_local,
    make_signature,
    shutdown_interpreter,
)
from rrlm.metrics import harvest_lm_history, reconcile, summarize
from rrlm.pi_config import resolve_model

# CLI names for --answer-type -> the Python annotation the SUBMIT value is
# parsed into. Library callers can pass any type (incl. Pydantic models).
ANSWER_TYPES: dict[str, type] = {
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "json": dict,
    "list": list[str],
}


async def asolve(
    instruction: str,
    data: str = "",
    *,
    main_model: str | None = None,
    sub_model: str | None = None,
    reasoning: str | None = None,
    temperature: float | None = None,
    backend: str | None = None,
    engine: str | None = None,
    max_depth: int = 2,
    max_iterations: int = 30,
    max_llm_calls: int = 50,
    max_spawns: int = 16,
    max_cost_usd: float | None = None,
    max_action_retries: int = 2,
    timeout_s: float | None = None,
    web: bool = False,
    files: list[str | Path] | None = None,
    answer_type: type | None = None,
    tools: list[Callable] | None = None,
    skills: list | None = None,
    doctrine: str | None = None,
    reconcile_cost: bool = True,
    return_trace: bool = False,
    interpreter=None,
    mcp: list | None = None,
    on_event=None,
    engine_options: dict | None = None,
) -> dict:
    """Run the RLM-first agent over (instruction, data); return answer + metrics.

    The async twin of :func:`solve` (same parameters, same result); use this
    from servers, notebooks with a running loop, or other agents.

    ``main_model``/``sub_model`` are Pi model references (``provider/model`` or a
    bare id); ``None`` for ``main_model`` uses Pi's current default and ``None``
    for ``sub_model`` reuses ``main_model``. Returns a dict: answer, error,
    wall_clock_s, trace_file, spawn_stats, usage, config.

    ``reasoning`` defaults to ``off`` for thinking-capable orchestrators (the
    settled finding: orchestrator thinking adds latency/variance without
    accuracy) and ``default`` otherwise. ``backend`` defaults to ``RRLM_BACKEND``
    or ``supervisor``. ``max_action_retries`` is applied only when the installed
    predict-rlm supports per-turn action retries.

    Budgets are global to the run (shared across the whole rlm_spawn tree):
    ``max_llm_calls`` caps sub-LM (predict) calls, ``max_spawns`` caps child
    agents, ``max_cost_usd`` is a soft USD ceiling checked before each call
    (needs a cost-reporting provider; local models are $0 and never trip it),
    and ``timeout_s`` cancels the whole run on overrun.

    ``files`` mounts real files (PDF, XLSX, DOCX, CSV, anything) into the
    sandbox and auto-attaches the matching predict-rlm document skills; on the
    jspi backend their packages auto-install, on supervisor/sbx the environment
    must provide them. ``answer_type`` types the final answer (int, float,
    bool, dict, list[...], a Pydantic model). ``tools`` adds host-side callables
    the agent can await from the REPL; ``skills`` adds predict-rlm Skill
    bundles. ``doctrine`` overrides the built-in doctrine text (see rrlm.gepa).

    ``web=True`` gives the agent host-side ``web_search`` / ``fetch`` tools plus a
    doctrine to retrieve-and-verify instead of answering from memory (needs the
    optional ``web`` extra: ddgs + trafilatura). It works on every backend.

    ``engine`` routes the run to an installed engine plugin instead of the
    built-in predict-rlm harness (see :mod:`rrlm.engines`). Selection is always
    explicit - this argument, ``--engine``, or ``RRLM_ENGINE``; nothing routes
    by inference. With an engine, the model/backend/doctrine parameters are
    ignored (the engine owns its own models and execution), while ``timeout_s``
    stays enforced host-side and the call/cost ceilings are passed down as the
    engine's budget lease.

    ``interpreter`` (advanced; supervisor backend only) is a caller-owned
    predict-rlm DirectPythonBackend reused across calls so the REPL namespace
    persists - the caller manages its shutdown. Use :class:`rrlm.Session`
    rather than passing this directly. Without it, the run's interpreter is
    created and shut down here, per call.

    ``mcp`` mounts stdio MCP servers' tools as awaitable host tools for this
    run (a list of :class:`rrlm.mcptools.MCPServerSpec`; needs the ``mcp``
    extra). Connections open before the run and close when it ends. MCP tools
    execute host-side on every backend - see the security note in
    :mod:`rrlm.mcptools`.

    ``on_event`` receives structured progress dicts as the run advances
    (``run_started``, ``llm_call``, ``spawn_started``/``spawn_finished``,
    ``run_finished`` - see :mod:`rrlm.events`). The callback is synchronous,
    must be fast, and can never break the run (its exceptions are swallowed).

    ``engine_options`` is an opaque engine-specific mapping passed through to
    the selected engine untouched (requires ``engine=``).

    This function is the kwargs facade over the typed contract: it compiles
    into a :class:`rrlm.contract.SolveRequest`, runs :func:`arun`, and renders
    the result back to this historic dict shape. Typed callers use
    :func:`arun` / :func:`run` directly.
    """
    request = SolveRequest(
        instruction=instruction,
        inputs={"data": data},
        files=tuple(str(p) for p in files or ()),
        answer_type=answer_type,
        engine=engine or None,
        engine_options=dict(engine_options or {}),
        policy=SolvePolicy(
            backend=backend,
            max_depth=max_depth,
            max_iterations=max_iterations,
            max_llm_calls=max_llm_calls,
            max_spawns=max_spawns,
            max_cost_usd=max_cost_usd,
            max_action_retries=max_action_retries,
            timeout_s=timeout_s,
            reasoning=reasoning,
            temperature=temperature,
            web=web,
        ),
        models=ModelSelection(main=main_model, sub=sub_model),
        tools=tuple(tools or ()),
        skills=tuple(skills or ()),
        doctrine=doctrine,
        mcp=tuple(mcp or ()),
        on_event=on_event,
        session=interpreter,
        reconcile_cost=reconcile_cost,
        return_trace=return_trace,
    )
    result = await arun(request)
    return result.to_legacy_dict(include_trace=return_trace)


async def arun(request: SolveRequest) -> SolveResult:
    """Run one typed :class:`SolveRequest` to a typed :class:`SolveResult`.

    The typed twin of :func:`asolve`: run failures come back as
    ``result.error`` (a :class:`rrlm.contract.RunError` with a stable
    category), while caller mistakes raise from the call itself. See
    :mod:`rrlm.contract` for the contract and its versioning.
    """
    # Validate the cheap, local things first: files fail fast, before any
    # model resolution or key loading. (Cross-field validation lives on the
    # request itself.)
    if request.files:
        missing = [p for p in request.files if not Path(p).is_file()]
        if missing:
            raise FileNotFoundError(f"input file(s) not found: {', '.join(missing)}")

    emitter = EventEmitter(request.on_event) if request.on_event is not None else None
    if request.engine:
        if emitter:
            emitter.emit(
                "run_started", engine=request.engine,
                instruction_chars=len(request.instruction), data_chars=len(request.data),
            )
        result = await _solve_with_engine(request)
        if emitter:
            emitter.emit(
                "run_finished",
                error=str(result.error) if result.error else None,
                wall_clock_s=result.wall_clock_s,
            )
        return result
    return await _solve_native(request, emitter)


def run(request: SolveRequest) -> SolveResult:
    """Synchronous wrapper around :func:`arun` (same contract)."""
    return asyncio.run(arun(request))


def _resolve_lms(main_sel, sub_sel, *, reasoning, temperature):
    """Build the (main, sub) LMs from model references and/or injected instances.

    Each selector is a Pi model reference string (resolved through
    ``rrlm.pi_config`` and built with rrlm's reasoning/temperature/limit
    handling) or an injected ``dspy.LM`` (adopted via
    :func:`rrlm.harness.ensure_shared_lm` so budgets, events, and usage
    accounting keep working - a plain LM would silently disable every
    ceiling). With any injected instance, ``reasoning`` must be None: that
    configuration belongs on the LM the caller built. Returns
    ``(main_lm, sub_lm, main_ref, sub_ref, reasoning, local)``.
    """
    import dspy

    injected = isinstance(main_sel, dspy.LM) or isinstance(sub_sel, dspy.LM)
    if injected and reasoning is not None:
        raise ValueError(
            "reasoning= cannot be combined with an injected dspy.LM: configure "
            "reasoning on the LM instance you inject"
        )
    defaults = HarnessConfig()

    if isinstance(main_sel, dspy.LM):
        main_lm = ensure_shared_lm(main_sel)
        main_ref = str(getattr(main_sel, "model", "injected"))
        main_local = lm_is_local(main_sel)
        main_resolved = None
    else:
        main_resolved = resolve_model(main_sel)
        if reasoning is None and not injected:
            reasoning = "off" if main_resolved.supports_reasoning else "default"
        main_lm = build_lm(
            main_resolved,
            min(defaults.main_max_tokens, main_resolved.max_tokens),
            temperature,
            reasoning=reasoning or "default",
        )
        main_ref, main_local = main_resolved.ref, main_resolved.is_local

    if sub_sel is None:
        # The sub role reuses the main model but needs its OWN instance:
        # budget roles and per-role history harvesting attach per object.
        if main_resolved is not None:
            sub_lm = build_lm(
                main_resolved,
                min(defaults.sub_max_tokens, main_resolved.max_tokens),
                temperature,
                reasoning=reasoning or "default",
            )
        else:
            sub_lm = ensure_shared_lm(main_sel, copy=True)
        sub_ref, sub_local = main_ref, main_local
    elif isinstance(sub_sel, dspy.LM):
        sub_lm = ensure_shared_lm(sub_sel)
        sub_ref = str(getattr(sub_sel, "model", "injected"))
        sub_local = lm_is_local(sub_sel)
    else:
        sub_resolved = resolve_model(sub_sel)
        sub_lm = build_lm(
            sub_resolved,
            min(defaults.sub_max_tokens, sub_resolved.max_tokens),
            temperature,
            reasoning=reasoning or "default",
        )
        sub_ref, sub_local = sub_resolved.ref, sub_resolved.is_local

    return main_lm, sub_lm, main_ref, sub_ref, reasoning, main_local or sub_local


async def _solve_native(request: SolveRequest, emitter) -> SolveResult:
    """The predict-rlm harness path for one request."""
    # Re-bind request fields to the local names the (long-lived, well-tested)
    # body below has always used; the body itself is the same run loop.
    policy = request.policy
    instruction, data = request.instruction, request.data
    files = list(request.files)
    answer_type = request.answer_type
    tools = list(request.tools)
    skills = list(request.skills)
    doctrine = request.doctrine
    mcp = list(request.mcp)
    interpreter = request.session
    reconcile_cost = request.reconcile_cost
    main_model, sub_model = request.models.main, request.models.sub
    backend = policy.backend
    reasoning, temperature, web = policy.reasoning, policy.temperature, policy.web
    max_depth, max_iterations = policy.max_depth, policy.max_iterations
    max_llm_calls, max_spawns = policy.max_llm_calls, policy.max_spawns
    max_cost_usd, max_action_retries = policy.max_cost_usd, policy.max_action_retries
    timeout_s = policy.timeout_s

    file_objs = None
    extra_skills = list(skills or [])
    if files:
        from predict_rlm import File

        file_objs = [File(path=str(p)) for p in files]
        extra_skills = document_skills_for(files) + extra_skills
    backend = resolve_backend(backend)
    if interpreter is not None and backend != "supervisor":
        # A config error, not a run failure: raise before anything executes.
        raise ValueError(f"interpreter= requires backend='supervisor', got {backend!r}")

    # Load .env first so OPENROUTER_API_KEY (the no-Pi path) is visible to model
    # resolution and to cost reconciliation.
    or_key = load_env()
    if temperature is None:
        temperature = 0.2
    main_lm, sub_lm, main_ref, sub_ref, reasoning, local = _resolve_lms(
        main_model, sub_model, reasoning=reasoning, temperature=temperature
    )

    cfg = HarnessConfig(
        main_model=main_ref,
        sub_model=sub_ref,
        reasoning=reasoning or "default",
        temperature=temperature,
        backend=backend,
        max_depth=max_depth,
        max_iterations=max_iterations,
        max_llm_calls=max_llm_calls,
        max_spawns=max_spawns,
        max_cost_usd=max_cost_usd,
        sandbox_exec_timeout=3600.0 if local else 300.0,
        max_action_retries=max_action_retries,
        web=web,
    )
    main_start, sub_start = len(main_lm.history), len(sub_lm.history)

    # One budget object shared by both LMs and the whole spawn tree: this is
    # what makes max_llm_calls / max_cost_usd / max_spawns real per-run
    # ceilings instead of per-agent allowances.
    budget = RunBudget(
        max_sub_calls=max_llm_calls, max_cost_usd=max_cost_usd, max_spawns=max_spawns
    )
    for lm, role in ((main_lm, "main"), (sub_lm, "sub")):
        attach = getattr(lm, "attach_budget", None)
        if attach is not None:
            attach(budget, role)
        if emitter is not None:
            attach_events = getattr(lm, "attach_events", None)
            if attach_events is not None:
                attach_events(emitter, role)

    if emitter:
        emitter.emit(
            "run_started", backend=backend,
            instruction_chars=len(instruction), data_chars=len(data or ""),
        )
    answer, error, error_category, spawn_stats = "", None, "execution", {}
    prediction = None
    run_trace = None
    rlm = None
    t0 = time.monotonic()
    try:
        async with AsyncExitStack() as stack:
            if mcp:
                # MCP connections live exactly as long as the run: opened here,
                # closed when the stack exits (success, error, or timeout).
                from rrlm.mcptools import connect_mcp_servers, mcp_tools_note

                mounted = await connect_mcp_servers(stack, mcp)
                tools = list(tools or []) + [t.call for t in mounted]
                from predict_rlm import Skill

                extra_skills = extra_skills + [
                    Skill(name="mounted-mcp-tools", instructions=mcp_tools_note(mounted))
                ]
            rlm = build_rlm(
                cfg, main_lm, sub_lm,
                signature=make_signature(answer_type, with_files=bool(file_objs)),
                budget=budget, extra_tools=tools, extra_skills=extra_skills,
                doctrine=doctrine, interpreter=interpreter, emitter=emitter,
            )
            call_kwargs: dict = {"task": instruction, "data": data}
            if file_objs:
                call_kwargs["files"] = file_objs
            coro = rlm.acall(**call_kwargs)
            if timeout_s and timeout_s > 0:
                # Hard total wall-clock ceiling: cancel the whole run on overrun.
                prediction = await asyncio.wait_for(coro, timeout=timeout_s)
            else:
                prediction = await coro
            answer = prediction.answer
            run_trace = getattr(prediction, "trace", None)
            spawn_stats = dict(rlm.spawn_stats)
    except (asyncio.TimeoutError, TimeoutError):
        error = f"TimeoutError: run exceeded timeout_s={timeout_s}s"
        error_category = "timeout"
    except Exception as exc:  # noqa: BLE001, return the failure to the caller
        # anyio-based components (MCP stdio clients) wrap a single real failure
        # in an ExceptionGroup during cleanup; unwrap so the error is readable.
        cause: BaseException = exc
        while isinstance(cause, BaseExceptionGroup) and len(cause.exceptions) == 1:
            cause = cause.exceptions[0]
        error = f"{type(cause).__name__}: {cause}"
        error_category = "budget" if isinstance(cause, BudgetExceededError) else "execution"
        # predict-rlm attaches the RunTrace to the exception; failure traces
        # are the most valuable GEPA signal, so capture them too.
        run_trace = getattr(exc, "trace", None)
    finally:
        # predict-rlm treats interpreter= as caller-owned and never shuts it
        # down; when this run created its own (supervisor backend, no Session),
        # release the runner process here or every solve() leaks one until the
        # host process exits. Session-injected interpreters stay alive by design.
        owned = getattr(rlm, "rrlm_owned_interpreter", None)
        if owned is not None:
            await asyncio.to_thread(shutdown_interpreter, owned)
    wall_clock_s = time.monotonic() - t0
    if emitter:
        emitter.emit("run_finished", error=error, wall_clock_s=round(wall_clock_s, 2))

    # Capture the predict-rlm RunTrace for later RLM-GEPA, if RRLM_TRACE_DIR is set.
    trace_file = None
    trace_dir = os.environ.get("RRLM_TRACE_DIR")
    if trace_dir and run_trace is not None:
        trace_file = export_trace(
            run_trace, trace_dir=trace_dir, instruction=instruction,
            answer=answer if isinstance(answer, str) else repr(answer),
            data_chars=len(data or ""), wall_clock_s=round(wall_clock_s, 2),
            error=error,
            config={"main_model": main_ref, "sub_model": sub_ref, "reasoning": reasoning},
        )

    records = harvest_lm_history(main_lm, "main", main_start) + harvest_lm_history(
        sub_lm, "sub", sub_start
    )
    # Only hosted OpenRouter calls are reconcilable; local/foreign gen ids skip.
    if reconcile_cost and any(r.gen_id and r.gen_id.startswith("gen-") for r in records):
        # reconcile blocks on HTTP + sleeps; keep the caller's loop responsive.
        await asyncio.to_thread(reconcile, records, or_key)

    return SolveResult(
        answer=answer,
        error=RunError(error_category, error) if error else None,
        wall_clock_s=round(wall_clock_s, 2),
        trace_file=trace_file,
        spawn_stats=spawn_stats,
        usage=Usage.from_dict(summarize(records)),
        config={
            "main_model": main_ref,
            "sub_model": sub_ref,
            "reasoning": reasoning,
            "backend": backend,
            "web": web,
        },
        # The live RunTrace object, for programmatic consumers (rrlm.gepa reads
        # it per evaluation). Not JSON-serializable; the legacy dict carries it
        # only when return_trace was asked for.
        trace=run_trace,
    )


def solve(instruction: str, data: str = "", **kwargs) -> dict:
    """Synchronous wrapper around :func:`asolve` (same parameters, same result).

    Cannot be called while an asyncio event loop is running; use ``asolve``
    from async code.
    """
    return asyncio.run(asolve(instruction, data, **kwargs))


async def _solve_with_engine(request: SolveRequest) -> SolveResult:
    """Run one request through its engine plugin; return a typed result.

    Rendered legacy, the result has the same shape as the native path so
    callers and the CLI need not care which path ran; ``config`` carries the
    engine name and ``vendor`` appears only when the engine returned
    engine-specific artifacts. The wall-clock ceiling is enforced here with
    the same cancel-the-run semantics as the native path, so it binds even a
    misbehaving engine.
    """
    from rrlm.engines import BudgetLease, EngineRequest, get_engine

    name = request.engine
    timeout_s = request.policy.timeout_s
    eng = get_engine(name)
    trace_dir = os.environ.get("RRLM_TRACE_DIR")
    engine_request = EngineRequest(
        instruction=request.instruction,
        data=request.data,
        files=request.files,
        answer_type=request.answer_type,
        budget=BudgetLease(
            timeout_s=timeout_s,
            max_llm_calls=request.policy.max_llm_calls,
            max_cost_usd=request.policy.max_cost_usd,
        ),
        trace_dir=trace_dir,
        options=request.engine_options,
    )
    answer, error, error_category, engine_result = "", None, "engine", None
    t0 = time.monotonic()
    try:
        coro = eng.solve(engine_request)
        if timeout_s and timeout_s > 0:
            engine_result = await asyncio.wait_for(coro, timeout=timeout_s)
        else:
            engine_result = await coro
        answer, error = engine_result.answer, engine_result.error
    except (asyncio.TimeoutError, TimeoutError):
        error = f"TimeoutError: run exceeded timeout_s={timeout_s}s"
        error_category = "timeout"
    except Exception as exc:  # noqa: BLE001, return the failure to the caller
        error = f"{type(exc).__name__}: {exc}"
        error_category = "execution"
    wall_clock_s = time.monotonic() - t0

    trace_file = engine_result.trace_file if engine_result else None
    if trace_dir:
        # Engine runs land in the same index.jsonl the native path feeds, so
        # `rrlm-traces list` shows one unified history; the `engine` key is
        # what distinguishes them (a free string - no engine enum anywhere).
        _append_index(
            trace_dir,
            {
                "trace_file": os.path.basename(trace_file) if trace_file else None,
                "instruction": (request.instruction or "")[:500],
                "answer": (answer if isinstance(answer, str) else repr(answer))[:500],
                "error": error,
                "data_chars": len(request.data),
                "wall_clock_s": round(wall_clock_s, 2),
                "config": {"engine": name},
                "engine": name,
            },
        )

    return SolveResult(
        answer=answer,
        error=RunError(error_category, error) if error else None,
        wall_clock_s=round(wall_clock_s, 2),
        trace_file=trace_file,
        spawn_stats={},
        usage=Usage.from_dict(engine_result.usage if engine_result else {}),
        config={"engine": name},
        vendor=(engine_result.vendor or None) if engine_result else None,
    )


def _many_task(instructions: list[str]) -> str:
    numbered = "\n".join(f"{i}. {q}" for i, q in enumerate(instructions, 1))
    return (
        "Answer EACH of the following numbered questions about the same `data`, "
        f"independently and in order:\n{numbered}\n\n"
        "Verify every answer, then SUBMIT `answer` as a list of strings, one "
        "answer per question, in the same order as the questions."
    )


async def asolve_many(instructions: list[str], data: str = "", **kwargs) -> dict:
    """Answer several questions over the same data in ONE run.

    Amortizes the data load and the REPL scaffold across all questions (calling
    solve() per question re-pays both every time). Accepts the same keyword
    arguments as :func:`asolve` except ``answer_type`` (forced to list[str]).
    The result gains an ``answers`` list aligned with ``instructions`` (None if
    the run failed or returned a malformed value).
    """
    if not instructions:
        raise ValueError("instructions must be a non-empty list")
    kwargs.pop("answer_type", None)
    result = await asolve(_many_task(list(instructions)), data, answer_type=list[str], **kwargs)
    answer = result.get("answer")
    result["answers"] = list(answer) if isinstance(answer, (list, tuple)) else None
    return result


def solve_many(instructions: list[str], data: str = "", **kwargs) -> dict:
    """Synchronous wrapper around :func:`asolve_many`."""
    return asyncio.run(asolve_many(instructions, data, **kwargs))


def export_trace(
    trace,
    *,
    trace_dir: str,
    instruction: str = "",
    answer: str = "",
    data_chars: int = 0,
    config: dict | None = None,
    wall_clock_s: float | None = None,
    error: str | None = None,
) -> str | None:
    """Best-effort: write a predict-rlm RunTrace to a UNIQUE file under
    ``trace_dir`` (one per process, so concurrent/repeated rrlm-solve calls
    accumulate) plus an ``index.jsonl`` line pairing instruction->answer->trace.
    These are the traces consumed by RLM-GEPA later (failure traces included:
    they carry the strongest optimization signal). Returns the path or None.

    No-op (returns None) when ``trace_dir`` is falsy, the trace is missing or
    unexportable, or anything goes wrong, trace capture must never break solve().
    """
    if not trace_dir or trace is None or not hasattr(trace, "to_exportable_json"):
        return None
    try:
        os.makedirs(trace_dir, exist_ok=True)
        stamp = f"{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}-{os.getpid()}"
        path = os.path.join(trace_dir, f"trace-{stamp}.json")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(trace.to_exportable_json())
        _append_index(
            trace_dir,
            {
                "trace_file": os.path.basename(path),
                "instruction": (instruction or "")[:500],
                "answer": (answer or "")[:500],
                "error": error,
                "data_chars": data_chars,
                "wall_clock_s": wall_clock_s,
                "config": config or {},
            },
        )
        return path
    except Exception:  # noqa: BLE001, trace capture is best-effort, never fatal
        return None


def _append_index(trace_dir: str, record: dict) -> None:
    """Best-effort append of one record to ``trace_dir``/index.jsonl.

    Shared by the native and engine solve paths so both feed one history;
    never raises, index upkeep must not break a solve that already succeeded.
    """
    try:
        os.makedirs(trace_dir, exist_ok=True)
        index = os.path.join(trace_dir, "index.jsonl")
        with open(index, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")
    except Exception:  # noqa: BLE001, best-effort by contract
        pass


def _read_data(arg: str | None) -> str:
    """Resolve the --data argument: '-' = stdin, '@path' = file, else literal."""
    if arg is None:
        return ""
    if arg == "-":
        return sys.stdin.read()
    if arg.startswith("@"):
        with open(arg[1:], encoding="utf-8") as f:
            return f.read()
    return arg


def _json_default(obj):
    """JSON-encode Pydantic models (typed answers) and anything else as str."""
    dump = getattr(obj, "model_dump", None)
    if callable(dump):
        return dump()
    return str(obj)


def _env_float(name: str) -> float | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="rrlm-solve",
        description="RLM-first solve: instruction + data -> answer (models from Pi config)",
    )
    parser.add_argument(
        "--instruction", "-i", action="append", required=True, dest="instructions",
        help="what to accomplish; repeat the flag to answer several questions "
             "over the same data in one run",
    )
    parser.add_argument(
        "--data", "-d", default=None, help="data payload: literal, @file, or - for stdin"
    )
    parser.add_argument(
        "--file", "-f", action="append", dest="files", default=None, metavar="PATH",
        help="mount a real file (PDF/XLSX/DOCX/CSV/...) into the sandbox; repeatable. "
             "Matching document skills attach automatically.",
    )
    parser.add_argument(
        "--main", "--main-model", dest="main_model",
        default=os.environ.get("RRLM_MAIN") or None,
        help="orchestrator model (Pi 'provider/model' or bare id); "
             "default: env RRLM_MAIN, else Pi's current model",
    )
    parser.add_argument(
        "--sub", "--sub-model", dest="sub_model",
        default=os.environ.get("RRLM_SUB") or None,
        help="leaf model for predict() fan-out; default: env RRLM_SUB, else same as --main",
    )
    parser.add_argument(
        "--reasoning", default=None, choices=["default", "off", "low", "medium", "high"],
        help="default: off for thinking-capable orchestrators, else default",
    )
    parser.add_argument("--temperature", type=float, default=None, help="sampling temperature (default 0.2)")
    parser.add_argument(
        "--backend", default=None, choices=list(BACKENDS),
        help="execution sandbox; default: env RRLM_BACKEND, else supervisor "
             "(host CPython, fastest; use jspi/sbx to isolate untrusted work)",
    )
    parser.add_argument(
        "--engine", default=os.environ.get("RRLM_ENGINE") or None, metavar="NAME",
        help="route the run to an installed engine plugin instead of the built-in "
             "harness (env RRLM_ENGINE); see rrlm.engines. Model/backend/doctrine "
             "flags are ignored with an engine; budget flags still apply.",
    )
    parser.add_argument(
        "--engine-option", action="append", dest="engine_options", default=None,
        metavar="KEY=VALUE",
        help="engine-specific option passed through to the selected engine untouched "
             "(repeatable; requires --engine). VALUE is parsed as JSON when it is "
             "valid JSON, else kept as a string.",
    )
    parser.add_argument(
        "--answer-type", default=None, choices=sorted(ANSWER_TYPES),
        help="type the final answer is parsed into (single-question runs); "
             "'json' = a JSON object, 'list' = a list of strings",
    )
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--action-retries", type=int, default=2, help="per-turn re-asks on parse failure")
    # Guardrails (hard ceilings; enforced globally across the whole spawn tree).
    parser.add_argument("--timeout", type=float, default=None,
                        help="hard total wall-clock ceiling in seconds (env RRLM_TIMEOUT); cancels the run on overrun")
    parser.add_argument("--max-llm-calls", type=int, default=50,
                        help="global cap on sub-LM (predict) calls across all depths")
    parser.add_argument("--max-iterations", type=int, default=30, help="cap on REPL turns per agent")
    parser.add_argument("--max-spawns", type=int, default=16,
                        help="global cap on rlm_spawn child agents across the whole run")
    parser.add_argument("--max-cost", type=float, default=None, dest="max_cost_usd",
                        help="soft USD ceiling for the run (env RRLM_MAX_COST); needs a "
                             "cost-reporting provider (e.g. OpenRouter); local models are $0")
    parser.add_argument(
        "--doctrine", default=None, metavar="PATH",
        help="override the built-in doctrine with the text in PATH (e.g. an RLM-GEPA winner)",
    )
    parser.add_argument(
        "--mcp", action="append", dest="mcp", default=None, metavar="SPEC",
        help="mount an MCP server's tools for this run (repeatable; needs the 'mcp' "
             "extra). SPEC is an http(s):// URL (streamable HTTP, preferred), "
             "sse+http(s):// for legacy SSE servers, or a shell-quoted command line "
             "for a local stdio server. Tools run host-side.",
    )
    parser.add_argument(
        "--events", action="store_true",
        help="stream structured progress events as JSONL on stderr "
             "(run_started, llm_call, spawn_*, run_finished)",
    )
    parser.add_argument(
        "--web", action="store_true", default=None,
        help="give the agent live web retrieval (web_search/fetch); env RRLM_WEB; needs the 'web' extra",
    )
    parser.add_argument("--json", action="store_true", help="emit full result JSON, not just the answer")
    args = parser.parse_args()

    web = args.web
    if web is None:
        web = os.environ.get("RRLM_WEB", "").strip().lower() in ("1", "true", "yes", "on")

    timeout_s = args.timeout if args.timeout is not None else _env_float("RRLM_TIMEOUT")
    max_cost = args.max_cost_usd if args.max_cost_usd is not None else _env_float("RRLM_MAX_COST")

    doctrine = None
    if args.doctrine:
        doctrine = Path(args.doctrine).read_text(encoding="utf-8")

    mcp = None
    if args.mcp:
        from rrlm.mcptools import MCPServerSpec

        mcp = [MCPServerSpec.parse(spec) for spec in args.mcp]

    engine_options = None
    if args.engine_options:
        engine_options = {}
        for item in args.engine_options:
            key, sep, value = item.partition("=")
            if not sep or not key:
                parser.error(f"--engine-option needs KEY=VALUE, got {item!r}")
            try:
                engine_options[key] = json.loads(value)
            except json.JSONDecodeError:
                engine_options[key] = value

    on_event = None
    if args.events:
        # stdout carries the answer/result; events stream on stderr as JSONL.
        def on_event(event: dict) -> None:
            print(json.dumps(event, default=str), file=sys.stderr, flush=True)

    common = dict(
        main_model=args.main_model,
        sub_model=args.sub_model,
        reasoning=args.reasoning,
        temperature=args.temperature,
        backend=args.backend,
        engine=args.engine,
        max_depth=args.max_depth,
        max_iterations=args.max_iterations,
        max_llm_calls=args.max_llm_calls,
        max_spawns=args.max_spawns,
        max_cost_usd=max_cost,
        max_action_retries=args.action_retries,
        timeout_s=timeout_s,
        web=web,
        files=args.files,
        doctrine=doctrine,
        mcp=mcp,
        on_event=on_event,
        engine_options=engine_options,
    )

    data = _read_data(args.data)
    if len(args.instructions) > 1:
        result = solve_many(args.instructions, data, **common)
    else:
        answer_type = ANSWER_TYPES[args.answer_type] if args.answer_type else None
        result = solve(args.instructions[0], data, answer_type=answer_type, **common)

    if args.json:
        print(json.dumps(result, indent=2, default=_json_default))
    elif result["error"]:
        print(f"ERROR: {result['error']}", file=sys.stderr)
        sys.exit(1)
    elif "answers" in result:
        answers = result["answers"]
        if answers is None:
            print(f"ERROR: expected a list of answers, got: {result['answer']!r}", file=sys.stderr)
            sys.exit(1)
        for ans in answers:
            print(ans)
    else:
        answer = result["answer"]
        print(answer if isinstance(answer, str) else json.dumps(answer, default=_json_default))


if __name__ == "__main__":
    main()
