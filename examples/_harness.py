"""Shared helper for the rrlm real-use-case evals.

Each eval builds (instruction, data), runs ``rrlm.solve`` against the model from
your Pi config (override with ``RRLM_MAIN`` / ``RRLM_SUB``), and checks the answer.
These exercise paths that synthetic template data never did: exact tabular
aggregation, cross-file code reasoning, and the end-to-end Pi delegation UX.

Run one directly, e.g.::

    RRLM_MAIN=openrouter/qwen/qwen3.6-27b python examples/eval_tabular.py
"""

from __future__ import annotations

import os
from collections.abc import Callable

from rrlm import solve


def run_eval(
    name: str,
    instruction: str,
    data: str,
    check: Callable[[str], tuple[bool, str]],
    *,
    backend: str | None = None,
) -> bool:
    """Run one eval; print a compact report; return True on pass."""
    main = os.environ.get("RRLM_MAIN") or None
    sub = os.environ.get("RRLM_SUB") or None
    backend = backend or os.environ.get("RRLM_BACKEND", "jspi")
    print(f"[{name}] main={main or 'pi-default'} sub={sub or 'same-as-main'} backend={backend}")
    print(f"[{name}] data={len(data):,} chars; running ...")

    result = solve(instruction, data, main_model=main, sub_model=sub, backend=backend)
    if result["error"]:
        print(f"[{name}] ERROR: {result['error']}")
        return False

    answer = (result["answer"] or "").strip()
    usage = result["usage"]
    ok, detail = check(answer)
    print(f"[{name}] answer: {answer[:200]!r}")
    print(
        f"[{name}] {'PASS' if ok else 'FAIL'}, {detail} | "
        f"model={result['config']['main_model']} calls={usage['calls']} "
        f"tokens={usage['prompt_tokens']}+{usage['completion_tokens']} "
        f"cost=${usage['cost_usd']:.4f} wall={result['wall_clock_s']}s "
        f"subs={result['spawn_stats']}"
    )
    return ok
