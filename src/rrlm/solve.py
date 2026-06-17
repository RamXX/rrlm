"""General-purpose RLM-first solve entry point -- the backend Pi delegates to.

Given an instruction and a (possibly large) data payload, run the RLM-first
agent: the data lands in the REPL, the orchestrator writes code to probe it,
fans out cheap sub-LM classification only when the data is irreducible, and
returns a verified answer. The data never enters the orchestrator's context.

CLI:
    python -m rrlm.solve --instruction "..." --data @path/to/file
    echo "<data>" | python -m rrlm.solve --instruction "..." --data -
    python -m rrlm.solve --instruction "..." --data "inline text" --json

Library:
    from rrlm.solve import solve
    result = solve("Which product has the most negative reviews?", data=text)
    print(result["answer"])
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time

from rrlm.config import MODELS, HarnessConfig, load_env
from rrlm.harness import build_lm, build_rlm
from rrlm.metrics import harvest_lm_history, reconcile, summarize

# Settled orchestrator (won the local bake-off): the pi-tune Q6_K model served
# by llama-server (no-thinking native, temp 0.7) + supergemma leaf. reasoning and
# temperature are resolved per-model from the registry when not overridden.
DEFAULT_MAIN = "qwen3.6-27b-pitune-local"
DEFAULT_SUB = "supergemma-26b-local"


def solve(
    instruction: str,
    data: str = "",
    *,
    main_model: str = DEFAULT_MAIN,
    sub_model: str = DEFAULT_SUB,
    reasoning: str | None = None,
    temperature: float | None = None,
    backend: str = "jspi",
    max_depth: int = 2,
    max_iterations: int = 30,
    max_action_retries: int = 2,
    reconcile_cost: bool = True,
) -> dict:
    """Run the RLM-first agent over (instruction, data); return answer + metrics.

    Returns a dict: answer, wall_clock_s, spawn_stats, usage, error.

    `reasoning`/`temperature` default to the main model's registry recommendation
    (e.g. pi-tune: no-thinking native, temp 0.7). `max_action_retries` defaults
    to 2: it absorbs intermittent malformed/empty action turns.
    """
    api_key = load_env()
    spec = MODELS[main_model]
    reasoning = reasoning if reasoning is not None else spec.default_reasoning
    temperature = temperature if temperature is not None else spec.default_temperature
    local = spec.api_base is not None or MODELS[sub_model].api_base is not None
    cfg = HarnessConfig(
        main_model=main_model,
        sub_model=sub_model,
        reasoning=reasoning,
        temperature=temperature,
        backend=backend,
        max_depth=max_depth,
        max_iterations=max_iterations,
        sandbox_exec_timeout=3600.0 if local else 300.0,
        max_action_retries=max_action_retries,
    )

    main_lm = build_lm(main_model, api_key, cfg.main_max_tokens, cfg.temperature, reasoning=reasoning)
    sub_lm = build_lm(sub_model, api_key, cfg.sub_max_tokens, cfg.temperature, reasoning=reasoning)
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
    # Only hosted (OpenRouter) calls are reconcilable; local gen ids are skipped.
    if reconcile_cost and any(r.gen_id and r.gen_id.startswith("gen-") for r in records):
        reconcile(records, api_key)

    return {
        "answer": answer,
        "error": error,
        "wall_clock_s": round(wall_clock_s, 2),
        "spawn_stats": spawn_stats,
        "usage": summarize(records),
        "config": {
            "main_model": main_model,
            "sub_model": sub_model,
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
    parser = argparse.ArgumentParser(description="RLM-first solve: instruction + data -> answer")
    parser.add_argument("--instruction", "-i", required=True, help="what to accomplish")
    parser.add_argument(
        "--data", "-d", default=None, help="data payload: literal, @file, or - for stdin"
    )
    parser.add_argument("--main-model", default=DEFAULT_MAIN, choices=sorted(MODELS))
    parser.add_argument("--sub-model", default=DEFAULT_SUB, choices=sorted(MODELS))
    parser.add_argument(
        "--reasoning", default=None, choices=["default", "off", "low", "medium", "high"],
        help="default: per-model registry recommendation",
    )
    parser.add_argument("--temperature", type=float, default=None, help="default: per-model recommendation")
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
