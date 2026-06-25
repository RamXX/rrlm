"""General-purpose RLM-first solve entry point -- the backend Pi delegates to.

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

Library:
    from rrlm import solve
    result = solve("Which product has the most negative reviews?", data=text)
    print(result["answer"])
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time

from rrlm.config import HarnessConfig, load_env
from rrlm.harness import build_lm, build_rlm
from rrlm.metrics import harvest_lm_history, reconcile, summarize
from rrlm.pi_config import resolve_model


def solve(
    instruction: str,
    data: str = "",
    *,
    main_model: str | None = None,
    sub_model: str | None = None,
    reasoning: str | None = None,
    temperature: float | None = None,
    backend: str = "jspi",
    max_depth: int = 2,
    max_iterations: int = 30,
    max_action_retries: int = 2,
    reconcile_cost: bool = True,
) -> dict:
    """Run the RLM-first agent over (instruction, data); return answer + metrics.

    ``main_model``/``sub_model`` are Pi model references (``provider/model`` or a
    bare id); ``None`` for ``main_model`` uses Pi's current default and ``None``
    for ``sub_model`` reuses ``main_model``. Returns a dict: answer, error,
    wall_clock_s, spawn_stats, usage, config.

    ``reasoning`` defaults to ``off`` for thinking-capable orchestrators (the
    settled finding: orchestrator thinking adds latency/variance without
    accuracy) and ``default`` otherwise. ``max_action_retries`` is applied only
    when the installed predict-rlm supports per-turn action retries.
    """
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
        sandbox_exec_timeout=3600.0 if local else 300.0,
        max_action_retries=max_action_retries,
    )

    # Clamp per-turn caps to each model's real output limit so smaller models in
    # someone else's Pi config don't trip a ValueError.
    main_max = min(cfg.main_max_tokens, main.max_tokens)
    sub_max = min(cfg.sub_max_tokens, sub.max_tokens)
    main_lm = build_lm(main, main_max, temperature, reasoning=reasoning)
    sub_lm = build_lm(sub, sub_max, temperature, reasoning=reasoning)
    main_start, sub_start = len(main_lm.history), len(sub_lm.history)

    answer, error, spawn_stats = "", None, {}
    t0 = time.monotonic()
    try:
        rlm = build_rlm(cfg, main_lm, sub_lm)
        prediction = asyncio.run(rlm.acall(task=instruction, data=data))
        answer = prediction.answer
        spawn_stats = dict(rlm.spawn_stats)
    except Exception as exc:  # noqa: BLE001 -- return the failure to the caller
        error = f"{type(exc).__name__}: {exc}"
    wall_clock_s = time.monotonic() - t0

    records = harvest_lm_history(main_lm, "main", main_start) + harvest_lm_history(
        sub_lm, "sub", sub_start
    )
    # Only hosted OpenRouter calls are reconcilable; local/foreign gen ids skip.
    if reconcile_cost and any(r.gen_id and r.gen_id.startswith("gen-") for r in records):
        reconcile(records, or_key)

    return {
        "answer": answer,
        "error": error,
        "wall_clock_s": round(wall_clock_s, 2),
        "spawn_stats": spawn_stats,
        "usage": summarize(records),
        "config": {
            "main_model": main.ref,
            "sub_model": sub.ref,
            "reasoning": reasoning,
            "backend": backend,
        },
    }


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


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="rrlm-solve",
        description="RLM-first solve: instruction + data -> answer (models from Pi config)",
    )
    parser.add_argument("--instruction", "-i", required=True, help="what to accomplish")
    parser.add_argument(
        "--data", "-d", default=None, help="data payload: literal, @file, or - for stdin"
    )
    parser.add_argument(
        "--main", "--main-model", dest="main_model", default=None,
        help="orchestrator model (Pi 'provider/model' or bare id); default: Pi's current model",
    )
    parser.add_argument(
        "--sub", "--sub-model", dest="sub_model", default=None,
        help="leaf model for predict() fan-out; default: same as --main",
    )
    parser.add_argument(
        "--reasoning", default=None, choices=["default", "off", "low", "medium", "high"],
        help="default: off for thinking-capable orchestrators, else default",
    )
    parser.add_argument("--temperature", type=float, default=None, help="sampling temperature (default 0.2)")
    parser.add_argument("--backend", default="jspi", choices=["jspi", "sbx", "supervisor"])
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--action-retries", type=int, default=2, help="per-turn re-asks on parse failure")
    parser.add_argument("--json", action="store_true", help="emit full result JSON, not just the answer")
    args = parser.parse_args()

    result = solve(
        args.instruction,
        _read_data(args.data),
        main_model=args.main_model,
        sub_model=args.sub_model,
        reasoning=args.reasoning,
        temperature=args.temperature,
        backend=args.backend,
        max_depth=args.max_depth,
        max_action_retries=args.action_retries,
    )

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    elif result["error"]:
        print(f"ERROR: {result['error']}", file=sys.stderr)
        sys.exit(1)
    else:
        print(result["answer"])


if __name__ == "__main__":
    main()
