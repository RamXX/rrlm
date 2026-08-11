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
from pathlib import Path

from rrlm.config import BACKENDS, HarnessConfig, load_env, resolve_backend
from rrlm.harness import RunBudget, build_lm, build_rlm, document_skills_for, make_signature
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
    """
    # Validate the cheap, local things first: files and backend fail fast,
    # before any model resolution or key loading.
    if engine and interpreter is not None:
        raise ValueError(
            "engine= and interpreter= are mutually exclusive: engines are "
            "stateless by contract and own their execution"
        )
    if files:
        missing = [str(p) for p in files if not Path(p).is_file()]
        if missing:
            raise FileNotFoundError(f"input file(s) not found: {', '.join(missing)}")

    if engine:
        return await _solve_with_engine(
            engine, instruction, data, files=files, answer_type=answer_type,
            timeout_s=timeout_s, max_llm_calls=max_llm_calls, max_cost_usd=max_cost_usd,
        )

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
    main = resolve_model(main_model)
    sub = resolve_model(sub_model) if sub_model else main

    if reasoning is None:
        reasoning = "off" if main.supports_reasoning else "default"
    if temperature is None:
        temperature = 0.2
    local = main.is_local or sub.is_local

    cfg = HarnessConfig(
        main_model=main.ref,
        sub_model=sub.ref,
        reasoning=reasoning,
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

    # Clamp per-turn caps to each model's real output limit so smaller models in
    # someone else's Pi config don't trip a ValueError.
    main_max = min(cfg.main_max_tokens, main.max_tokens)
    sub_max = min(cfg.sub_max_tokens, sub.max_tokens)
    main_lm = build_lm(main, main_max, temperature, reasoning=reasoning)
    sub_lm = build_lm(sub, sub_max, temperature, reasoning=reasoning)
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

    answer, error, spawn_stats = "", None, {}
    prediction = None
    run_trace = None
    rlm = None
    t0 = time.monotonic()
    try:
        rlm = build_rlm(
            cfg, main_lm, sub_lm,
            signature=make_signature(answer_type, with_files=bool(file_objs)),
            budget=budget, extra_tools=tools, extra_skills=extra_skills,
            doctrine=doctrine, interpreter=interpreter,
        )
        call_kwargs: dict = {"task": instruction, "data": data}
        if file_objs:
            call_kwargs["files"] = file_objs
        coro = rlm.acall(**call_kwargs)
        if timeout_s and timeout_s > 0:
            # Hard total wall-clock ceiling: cancel the whole run if it overruns.
            prediction = await asyncio.wait_for(coro, timeout=timeout_s)
        else:
            prediction = await coro
        answer = prediction.answer
        run_trace = getattr(prediction, "trace", None)
        spawn_stats = dict(rlm.spawn_stats)
    except (asyncio.TimeoutError, TimeoutError):
        error = f"TimeoutError: run exceeded timeout_s={timeout_s}s"
    except Exception as exc:  # noqa: BLE001, return the failure to the caller
        error = f"{type(exc).__name__}: {exc}"
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
            await asyncio.to_thread(owned.shutdown)
    wall_clock_s = time.monotonic() - t0

    # Capture the predict-rlm RunTrace for later RLM-GEPA, if RRLM_TRACE_DIR is set.
    trace_file = None
    trace_dir = os.environ.get("RRLM_TRACE_DIR")
    if trace_dir and run_trace is not None:
        trace_file = export_trace(
            run_trace, trace_dir=trace_dir, instruction=instruction,
            answer=answer if isinstance(answer, str) else repr(answer),
            data_chars=len(data or ""), wall_clock_s=round(wall_clock_s, 2),
            error=error,
            config={"main_model": main.ref, "sub_model": sub.ref, "reasoning": reasoning},
        )

    records = harvest_lm_history(main_lm, "main", main_start) + harvest_lm_history(
        sub_lm, "sub", sub_start
    )
    # Only hosted OpenRouter calls are reconcilable; local/foreign gen ids skip.
    if reconcile_cost and any(r.gen_id and r.gen_id.startswith("gen-") for r in records):
        # reconcile blocks on HTTP + sleeps; keep the caller's loop responsive.
        await asyncio.to_thread(reconcile, records, or_key)

    result = {
        "answer": answer,
        "error": error,
        "wall_clock_s": round(wall_clock_s, 2),
        "trace_file": trace_file,
        "spawn_stats": spawn_stats,
        "usage": summarize(records),
        "config": {
            "main_model": main.ref,
            "sub_model": sub.ref,
            "reasoning": reasoning,
            "backend": backend,
            "web": web,
        },
    }
    if return_trace:
        # The live RunTrace object, for programmatic consumers (rrlm.gepa needs
        # it per evaluation). Not JSON-serializable; never set by the CLI.
        result["trace"] = run_trace
    return result


def solve(instruction: str, data: str = "", **kwargs) -> dict:
    """Synchronous wrapper around :func:`asolve` (same parameters, same result).

    Cannot be called while an asyncio event loop is running; use ``asolve``
    from async code.
    """
    return asyncio.run(asolve(instruction, data, **kwargs))


async def _solve_with_engine(
    name: str,
    instruction: str,
    data: str,
    *,
    files: list[str | Path] | None,
    answer_type: type | None,
    timeout_s: float | None,
    max_llm_calls: int,
    max_cost_usd: float | None,
) -> dict:
    """Run one solve through an engine plugin; return the standard result dict.

    The result has the same shape as the native path (answer, error,
    wall_clock_s, trace_file, spawn_stats, usage, config) so callers and the
    CLI need not care which path ran; ``config`` carries the engine name and
    ``vendor`` appears only when the engine returned engine-specific artifacts.
    The wall-clock ceiling is enforced here with the same cancel-the-run
    semantics as the native path, so it binds even a misbehaving engine.
    """
    from rrlm.engines import BudgetLease, EngineRequest, get_engine

    eng = get_engine(name)
    trace_dir = os.environ.get("RRLM_TRACE_DIR")
    request = EngineRequest(
        instruction=instruction,
        data=data,
        files=tuple(str(p) for p in files or ()),
        answer_type=answer_type,
        budget=BudgetLease(
            timeout_s=timeout_s, max_llm_calls=max_llm_calls, max_cost_usd=max_cost_usd
        ),
        trace_dir=trace_dir,
    )
    answer, error, engine_result = "", None, None
    t0 = time.monotonic()
    try:
        coro = eng.solve(request)
        if timeout_s and timeout_s > 0:
            engine_result = await asyncio.wait_for(coro, timeout=timeout_s)
        else:
            engine_result = await coro
        answer, error = engine_result.answer, engine_result.error
    except (asyncio.TimeoutError, TimeoutError):
        error = f"TimeoutError: run exceeded timeout_s={timeout_s}s"
    except Exception as exc:  # noqa: BLE001, return the failure to the caller
        error = f"{type(exc).__name__}: {exc}"
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
                "instruction": (instruction or "")[:500],
                "answer": (answer if isinstance(answer, str) else repr(answer))[:500],
                "error": error,
                "data_chars": len(data or ""),
                "wall_clock_s": round(wall_clock_s, 2),
                "config": {"engine": name},
                "engine": name,
            },
        )

    result = {
        "answer": answer,
        "error": error,
        "wall_clock_s": round(wall_clock_s, 2),
        "trace_file": trace_file,
        "spawn_stats": {},
        "usage": dict(engine_result.usage) if engine_result else {},
        "config": {"engine": name},
    }
    if engine_result and engine_result.vendor:
        result["vendor"] = engine_result.vendor
    return result


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
