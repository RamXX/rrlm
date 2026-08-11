"""``rrlm-session``: one persistent Session behind a line-delimited protocol.

This is the bridge that lets a long-lived host (the Pi extension, any agent)
keep one :class:`rrlm.Session` - and therefore one REPL namespace - alive
across many solve calls without linking Python. The host spawns
``rrlm-session`` once, writes one JSON request per line on stdin, and reads
one JSON response per line on stdout. Stderr is free for logs and warnings;
stdout carries protocol lines only.

Requests (``id`` is any JSON value; it is echoed back on the response)::

    {"id": 1, "op": "solve", "instruction": "...", "data": "..."}         # inline data
    {"id": 2, "op": "solve", "instruction": "...", "data_file": "/path"}  # staged file
    {"id": 3, "op": "solve", "instruction": "...", "options": {"answer_type": "int"}}
    {"id": 4, "op": "reset"}   # clear the REPL namespace, keep the session
    {"id": 5, "op": "ping"}    # liveness + version
    {"id": 6, "op": "close"}   # release the interpreter and exit 0

Responses: ``{"id": ..., "result": {...}}`` on success (for ``solve``, the
result is the standard solve dict: answer, error, usage, ...), or
``{"id": ..., "error": "..."}`` for a request that could not be served. A
malformed line answers ``{"id": null, "error": ...}``. Requests are served
sequentially - a session is one interpreter, so there is nothing to gain
from concurrency, and ordering stays deterministic.

Lifecycle: stdin EOF is treated as ``close``. That makes orphan cleanup
automatic - if the host process dies, the pipe closes, the session releases
its interpreter, and this process exits; nothing leaks.

``options`` accepts the per-call keyword arguments of
:meth:`rrlm.Session.asolve` (budgets, ``timeout_s``, ...) plus
``answer_type`` as a string from the CLI vocabulary (``str``, ``int``,
``float``, ``bool``, ``json``, ``list``).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from rrlm.session import Session
from rrlm.solve import ANSWER_TYPES, _env_float, _json_default

PROTOCOL = "rrlm-session/1"


def handle_request(session: Session, request: dict) -> dict:
    """Serve one parsed request; always returns a response dict.

    Raises only for ``close`` (via SystemExit after responding is the
    caller's job) - everything else, including a failed solve, comes back as
    a response so the host's read loop never desynchronizes.
    """
    rid = request.get("id")
    op = request.get("op", "solve")
    try:
        if op == "ping":
            from rrlm import __version__

            return {"id": rid, "result": {"ok": True, "protocol": PROTOCOL,
                                          "rrlm": __version__}}
        if op == "reset":
            session.reset()
            return {"id": rid, "result": {"ok": True}}
        if op == "solve":
            instruction = request.get("instruction")
            if not isinstance(instruction, str) or not instruction:
                return {"id": rid, "error": "solve needs a non-empty 'instruction' string"}
            if "data" in request and "data_file" in request:
                return {"id": rid, "error": "pass 'data' or 'data_file', not both"}
            data = request.get("data", "")
            if "data_file" in request:
                data = Path(request["data_file"]).read_text(encoding="utf-8")
            options = dict(request.get("options") or {})
            if "answer_type" in options:
                name = options["answer_type"]
                if name not in ANSWER_TYPES:
                    known = ", ".join(sorted(ANSWER_TYPES))
                    return {"id": rid, "error": f"unknown answer_type {name!r} (one of: {known})"}
                options["answer_type"] = ANSWER_TYPES[name]
            result = session.solve(instruction, data, **options)
            return {"id": rid, "result": result}
        return {"id": rid, "error": f"unknown op {op!r} (solve, reset, ping, close)"}
    except Exception as exc:  # noqa: BLE001, a bad request must not kill the session
        return {"id": rid, "error": f"{type(exc).__name__}: {exc}"}


def serve(session: Session, stdin, stdout) -> None:
    """The request loop: one JSON line in, one JSON line out, until EOF/close."""
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                raise ValueError("request must be a JSON object")
        except Exception as exc:  # noqa: BLE001, answer instead of desynchronizing
            _respond(stdout, {"id": None, "error": f"bad request line: {exc}"})
            continue
        if request.get("op") == "close":
            _respond(stdout, {"id": request.get("id"), "result": {"ok": True}})
            return
        _respond(stdout, handle_request(session, request))


def _respond(stdout, response: dict) -> None:
    stdout.write(json.dumps(response, default=_json_default) + "\n")
    stdout.flush()


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="rrlm-session",
        description="Serve one persistent rrlm Session over line-delimited JSON "
                    "on stdin/stdout (see rrlm.session_server).",
    )
    parser.add_argument(
        "--main", "--main-model", dest="main_model",
        default=os.environ.get("RRLM_MAIN") or None,
        help="orchestrator model for the whole session (Pi ref); "
             "default: env RRLM_MAIN, else Pi's current model",
    )
    parser.add_argument(
        "--sub", "--sub-model", dest="sub_model",
        default=os.environ.get("RRLM_SUB") or None,
        help="leaf model for predict() fan-out; default: env RRLM_SUB, else same as --main",
    )
    parser.add_argument("--timeout", type=float, default=None,
                        help="per-call wall-clock ceiling in seconds (env RRLM_TIMEOUT)")
    parser.add_argument("--max-llm-calls", type=int, default=50,
                        help="per-call global cap on sub-LM (predict) calls")
    parser.add_argument("--max-cost", type=float, default=None, dest="max_cost_usd",
                        help="per-call soft USD ceiling (env RRLM_MAX_COST)")
    args = parser.parse_args()

    timeout_s = args.timeout if args.timeout is not None else _env_float("RRLM_TIMEOUT")
    max_cost = args.max_cost_usd if args.max_cost_usd is not None else _env_float("RRLM_MAX_COST")

    defaults: dict = {"max_llm_calls": args.max_llm_calls}
    if args.main_model:
        defaults["main_model"] = args.main_model
    if args.sub_model:
        defaults["sub_model"] = args.sub_model
    if timeout_s is not None:
        defaults["timeout_s"] = timeout_s
    if max_cost is not None:
        defaults["max_cost_usd"] = max_cost

    session = Session(**defaults)
    try:
        serve(session, sys.stdin, sys.stdout)
    finally:
        session.close()


if __name__ == "__main__":
    main()
