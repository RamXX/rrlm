"""A real, offline OpenAI-compatible stub server for rrlm integration / e2e tests.

This is NOT an in-process mock. It is a standalone HTTP service started as a real
subprocess (real sockets, real request/response bodies). Tests point a ``dspy.LM``
(via Pi config) at its base URL, so the whole stack runs for real: litellm builds
and sends an HTTP POST to ``/v1/chat/completions``, this process answers with
canned but well-formed OpenAI chat-completion JSON, dspy parses it, and the
generated REPL code executes in predict-rlm's real local-CPython backend over the
real ``data`` variable. Determinism comes from the canned answers, not from
patching the call path, which is what lets the same tests serve as integration and
end-to-end coverage with no network and no credentials.

Behaviour is selected by a path prefix so a single process can serve every
scenario a test needs (the LM's ``api_base`` carries the prefix, e.g.
``http://127.0.0.1:PORT/submit/v1``):

  submit   one action turn returns REPL code that computes the answer from the
           real ``data`` variable and calls SUBMIT (the happy path).
  predict  one action turn returns REPL code that fans out an ``await predict()``
           leaf call and then SUBMITs; the leaf call comes back here too.
  never    every action turn returns harmless non-submitting code, so the run
           exhausts ``max_iterations`` and falls through to the extract signature.
  slow     the server sleeps before answering an action call, so a wall-clock
           timeout on the caller's side fires against a genuinely slow endpoint.
  spawn    one action turn spawns a child agent via rlm_spawn, then SUBMITs.
  typed    SUBMITs a real int (exercises typed `answer: int` signatures).
  filesread  reads the first mounted file (the `files` variable) and SUBMITs
           its content (exercises File input mounting).
  listsubmit  SUBMITs a list[str] (exercises solve_many / list answers).

The action vs extract vs leaf-predict call is told apart by the dspy ChatAdapter
output-field markers present in the request body (``[[ ## code ## ]]`` for an
action turn, ``[[ ## label ## ]]`` for our leaf predict signature, otherwise the
final ``answer`` extract). Responses are emitted in the ChatAdapter wire format
the caller's adapter expects.

Run directly:  python stub_server.py [--host 127.0.0.1] [--port 0] [--slow-seconds N]
On startup it prints a line ``STUB_READY <host> <port>`` to stdout so a parent
process can learn the bound (possibly OS-assigned) port.
"""

from __future__ import annotations

import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def _chat_completion(content: str, model: str = "stub-model") -> dict:
    """A minimal but complete OpenAI chat-completion response body."""
    return {
        "id": "chatcmpl-stub-0001",
        "object": "chat.completion",
        "created": 1700000000,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
    }


def _action(reasoning: str, code: str) -> str:
    """Format an action turn (reasoning + code) in dspy ChatAdapter wire form."""
    return (
        f"[[ ## reasoning ## ]]\n{reasoning}\n\n"
        f"[[ ## code ## ]]\n```repl\n{code}\n```\n\n"
        f"[[ ## completed ## ]]\n"
    )


def _field(name: str, value: str) -> str:
    """Format a single-output-field response (extract / leaf predict)."""
    return f"[[ ## {name} ## ]]\n{value}\n\n[[ ## completed ## ]]\n"


# REPL programs returned for each action scenario. They run for real in the
# predict-rlm backend over the real `data` variable.
_SUBMIT_CODE = "answer = str(len(data))\nSUBMIT(answer=answer)"
_NEVER_CODE = "probe = len(data)  # deliberately never calls SUBMIT"
_PREDICT_CODE = (
    'res = await predict("chunk: str -> label: str", chunk=data[:50])\n'
    "SUBMIT(answer=str(res.label))"
)
# Capacity-driven recursion: spawn one child agent over a slice, then submit.
# The child (a deeper RLM with no spawn tool of its own under max_depth=1) runs
# its own turns and falls through to extract, whose answer the parent wraps.
_SPAWN_CODE = (
    'res = await rlm_spawn("leaf subtask", data[:20])\n'
    'SUBMIT(answer="spawned:" + str(res))'
)
# Typed answers: SUBMIT a real int, parsed back into an `answer: int` signature.
_TYPED_CODE = "SUBMIT(answer=len(data))"
# File inputs: the `files` REPL variable lists the mounted sandbox paths; read
# the first file's real content and submit it.
_FILES_CODE = (
    "with open(files[0], encoding='utf-8') as fh:\n"
    "    content = fh.read()\n"
    "SUBMIT(answer=content.strip())"
)
# Multi-question runs: SUBMIT a list[str], one answer per question.
_LIST_CODE = 'SUBMIT(answer=[str(len(data)), "second-answer"])'


def _select_content(mode: str, body: str, slow_seconds: float) -> str:
    """Decide the response content from the scenario and the request markers."""
    is_action = "## code ##" in body
    is_leaf = "## label ##" in body

    if is_action:
        if mode == "slow":
            time.sleep(slow_seconds)
            return _action("slow probe", _SUBMIT_CODE)
        if mode == "never":
            return _action("looping without submit", _NEVER_CODE)
        if mode == "predict":
            return _action("fan out one leaf predict", _PREDICT_CODE)
        if mode == "spawn":
            return _action("spawn a child over a slice", _SPAWN_CODE)
        if mode == "typed":
            return _action("submit a typed (int) answer", _TYPED_CODE)
        if mode == "filesread":
            return _action("read the mounted file and submit its content", _FILES_CODE)
        if mode == "listsubmit":
            return _action("submit one answer per question", _LIST_CODE)
        return _action("compute the answer from data", _SUBMIT_CODE)
    if is_leaf:
        # Leaf predict() sub-call: return the single declared output field.
        return _field("label", "negative")
    # Extract signature: the run ended without SUBMIT (e.g. max_iterations).
    return _field("answer", "extracted-fallback-answer")


class _Handler(BaseHTTPRequestHandler):
    # Set by the server factory below.
    slow_seconds = 0.5

    def log_message(self, *args):  # noqa: D401, silence default stderr logging
        return

    def _write_json(self, status: int, payload: dict) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):  # noqa: N802, BaseHTTPRequestHandler API
        # Some clients probe /v1/models; answer benignly.
        if self.path.endswith("/models"):
            self._write_json(200, {"object": "list", "data": [{"id": "stub-model"}]})
        else:
            self._write_json(404, {"error": "not found"})

    def do_POST(self):  # noqa: N802, BaseHTTPRequestHandler API
        if not self.path.endswith("/chat/completions"):
            self._write_json(404, {"error": f"unknown path {self.path}"})
            return
        # First path segment is the scenario selector (submit/never/predict/slow).
        mode = self.path.lstrip("/").split("/", 1)[0]
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length).decode("utf-8") if length else ""
        try:
            parsed = json.loads(raw) if raw else {}
            body = json.dumps(parsed)
        except json.JSONDecodeError:
            body = raw
        content = _select_content(mode, body, self.slow_seconds)
        self._write_json(200, _chat_completion(content))


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenAI-compatible stub server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--slow-seconds", type=float, default=0.5)
    args = parser.parse_args()

    _Handler.slow_seconds = args.slow_seconds
    server = ThreadingHTTPServer((args.host, args.port), _Handler)
    host, port = server.server_address[0], server.server_address[1]
    print(f"STUB_READY {host} {port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
