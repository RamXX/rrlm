"""Run one task under one condition and record comparable metrics.

Usage:
    python -m rrlm.bench.runner --task ledger --model openrouter/qwen/qwen3.7-max \
        --condition rlm --size 2000

Models are Pi model references (provider/model or a bare id; see rrlm.pi_config).
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import datetime as dt
import importlib.metadata
import json
import time
import traceback

import dspy

from rrlm.bench.tasks import TASK_BUILDERS, Task
from rrlm.config import RUNS_DIR, HarnessConfig, load_env
from rrlm.harness import BaselineTask, build_lm, build_rlm
from rrlm.metrics import RunLogger, harvest_lm_history, reconcile, summarize
from rrlm.pi_config import ResolvedModel, resolve_model


def _redacted(model: ResolvedModel) -> dict:
    return {**dataclasses.asdict(model), "api_key": "<redacted>"}


def _versions() -> dict:
    out = {}
    for pkg in ("dspy", "predict-rlm", "litellm"):
        try:
            out[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            out[pkg] = "unknown"
    return out


def run_task(
    task: Task, main: ResolvedModel, sub: ResolvedModel, condition: str, cfg: HarnessConfig
) -> dict:
    api_key = load_env()  # OpenRouter key for cost reconciliation (may be empty)
    started_at = dt.datetime.now(dt.timezone.utc)
    variant = condition
    if cfg.reasoning != "default":
        variant += f"-r{cfg.reasoning}"
    if sub.ref != main.ref:
        variant += f"-sub:{sub.ref}"
    safe_model = main.ref.replace("/", "-")
    run_id = f"{started_at.strftime('%Y%m%d-%H%M%S')}_{task.task_id}_{safe_model}_{variant}"
    logger = RunLogger(RUNS_DIR, run_id)

    main_lm = build_lm(
        main, min(cfg.main_max_tokens, main.max_tokens), cfg.temperature, reasoning=cfg.reasoning
    )
    sub_lm = build_lm(
        sub, min(cfg.sub_max_tokens, sub.max_tokens), cfg.temperature, reasoning=cfg.reasoning
    )
    main_start, sub_start = len(main_lm.history), len(sub_lm.history)

    logger.write_meta(
        {
            "run_id": run_id,
            "started_at": started_at.isoformat(),
            "task_id": task.task_id,
            "task_kind": task.kind,
            "task_meta": task.meta,
            "model": main.ref,
            "model_spec": {"main": _redacted(main), "sub": _redacted(sub)},
            "condition": condition,
            "harness": cfg.as_dict(),
            "versions": _versions(),
        }
    )

    answer, status, error, trace, spawn_stats = "", "completed", None, None, {}
    t0 = time.monotonic()
    try:
        if condition == "rlm":
            rlm = build_rlm(cfg, main_lm, sub_lm)
            prediction = asyncio.run(rlm.acall(task=task.instruction, data=task.data))
            answer = prediction.answer
            trace = getattr(prediction, "trace", None)
            spawn_stats = dict(rlm.spawn_stats)
        elif condition == "baseline":
            with dspy.context(lm=main_lm):
                prediction = dspy.Predict(BaselineTask)(task=task.instruction, data=task.data)
            answer = prediction.answer
        else:
            raise ValueError(f"unknown condition: {condition}")
    except Exception as exc:  # noqa: BLE001 -- record the failure, then continue accounting
        status, error = "error", f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    wall_clock_s = time.monotonic() - t0

    records = harvest_lm_history(main_lm, "main", main_start) + harvest_lm_history(
        sub_lm, "sub", sub_start
    )
    if condition == "baseline":
        for rec in records:
            rec.role = "baseline"
    print(f"reconciling {len(records)} calls with OpenRouter ...")
    filled = reconcile(records, api_key)
    logger.log_calls(records)

    passed, detail = (False, "no answer") if not answer else task.check(answer)
    result = {
        "run_id": run_id,
        "status": status,
        "error": error,
        "passed": passed,
        "check_detail": detail,
        "answer": answer[:2000],
        "wall_clock_s": round(wall_clock_s, 2),
        "spawn_stats": spawn_stats,
        "reconciled_calls": filled,
        "usage": summarize(records),
    }
    logger.write_result(result)

    if trace is not None and hasattr(trace, "to_exportable_json"):
        try:
            exported = trace.to_exportable_json()
            if isinstance(exported, str):  # avoid double-encoding JSON strings
                exported = json.loads(exported)
            logger.write_trace(exported)
        except Exception as exc:  # noqa: BLE001
            print(f"trace export failed: {exc}")

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one rrlm experiment condition")
    parser.add_argument("--task", default="ledger", choices=sorted(TASK_BUILDERS))
    parser.add_argument(
        "--model", default=None,
        help="orchestrator model (Pi 'provider/model' or bare id); default: Pi's current model",
    )
    parser.add_argument("--condition", default="rlm", choices=["rlm", "baseline"])
    parser.add_argument("--size", type=int, default=2000, help="task size (e.g. ledger lines)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--max-iterations", type=int, default=30)
    parser.add_argument("--backend", default="jspi", choices=["jspi", "sbx", "supervisor"])
    parser.add_argument(
        "--reasoning", default="default", choices=["default", "off", "low", "medium", "high"]
    )
    parser.add_argument(
        "--sub-model", default=None, help="sub-LM for predict() (Pi 'provider/model' or bare id)"
    )
    parser.add_argument(
        "--sandbox-exec-timeout",
        type=float,
        default=0.0,
        help="per-turn sandbox wall-clock cap; 0 = auto (3600 local, 300 hosted)",
    )
    parser.add_argument(
        "--action-retries",
        type=int,
        default=0,
        help="per-turn action-generation re-asks on parse failure (0 = off)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="sampling temperature (default: harness 0.2; set per model recommendation)",
    )
    args = parser.parse_args()

    load_env()  # load .env so OPENROUTER_API_KEY is visible to model resolution
    task = TASK_BUILDERS[args.task](size=args.size, seed=args.seed)
    # local endpoints serve leaves serially; a wide fan-out runs as one REPL turn
    # and overruns the 300s sandbox cap, so give local runs a generous window.
    main = resolve_model(args.model)
    sub = resolve_model(args.sub_model) if args.sub_model else main
    local = main.is_local or sub.is_local
    cfg = HarnessConfig(
        main_model=main.ref,
        sub_model=sub.ref,
        max_depth=args.max_depth,
        max_iterations=args.max_iterations,
        backend=args.backend,
        reasoning=args.reasoning,
        sandbox_exec_timeout=args.sandbox_exec_timeout
        if args.sandbox_exec_timeout
        else (3600.0 if local else 300.0),
        max_action_retries=args.action_retries,
        **({"temperature": args.temperature} if args.temperature is not None else {}),
    )
    result = run_task(task, main, sub, args.condition, cfg)

    usage = result["usage"]
    print(json.dumps(result, indent=2, default=str))
    print(
        f"\n[{result['run_id']}] passed={result['passed']} "
        f"wall={result['wall_clock_s']}s calls={usage['calls']} "
        f"tokens={usage['prompt_tokens']}+{usage['completion_tokens']} "
        f"cost=${usage['cost_usd']:.4f} (complete={usage['cost_complete']})"
    )


if __name__ == "__main__":
    main()
