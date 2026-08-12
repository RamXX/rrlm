"""A stand-in for ``pi --mode rpc`` used by the ACP bridge tests.

Speaks the documented pi RPC protocol shape over stdin/stdout (strict JSONL:
command in, response + events out) with deterministic canned behavior, so the
bridge is exercised against a real subprocess and real pipes, no mocks:

* Each ``prompt`` streams ``text_delta`` chunks saying ``turn-N`` (N counts
  prompts served by this process, proving one process carries the whole
  conversation) plus one bash tool execution.
* A prompt containing ``HANG`` starts but never settles until ``abort``
  arrives, for the cancellation path.
* A prompt containing ``DIALOG`` emits an ``extension_ui_request`` select
  dialog and waits for the response; the answer is reported in the streamed
  text, proving the bridge's headless auto-cancel.
* A prompt whose message names an attached data file reads it and reports
  ``data-len=N``, proving resource staging.

Ignores its argv (the bridge appends ``--mode rpc``); exits 0 on stdin EOF.
"""

from __future__ import annotations

import json
import sys


def emit(obj: dict) -> None:
    print(json.dumps(obj), flush=True)


def delta(text: str) -> None:
    emit({"type": "message_update", "assistantMessageEvent": {
        "type": "text_delta", "contentIndex": 0, "delta": text}})


def settle() -> None:
    emit({"type": "agent_end", "messages": [], "willRetry": False})
    emit({"type": "agent_settled"})


def serve_prompt(message: str, prompts: int, stdin) -> bool:
    """Serve one accepted prompt; returns True when left hanging."""
    emit({"type": "agent_start"})
    if "HANG" in message:
        return True
    if "DIALOG" in message:
        emit({"type": "extension_ui_request", "id": "ui-1", "method": "select",
              "title": "pick one", "options": ["a", "b"], "timeout": 10000})
        for line in stdin:
            response = json.loads(line)
            if response.get("type") == "extension_ui_response" and response.get("id") == "ui-1":
                delta("cancelled" if response.get("cancelled") else "answered")
                break
        settle()
        return False
    emit({"type": "tool_execution_start", "toolCallId": "call_1",
          "toolName": "bash", "args": {"command": "ls"}})
    emit({"type": "tool_execution_end", "toolCallId": "call_1", "toolName": "bash",
          "isError": False,
          "result": {"content": [{"type": "text", "text": "ok"}],
                     "details": {"summary": "model=stub calls=3 tokens=10+5"}}})
    delta(f"turn-{prompts}")
    emit({"type": "message_end", "message": {"role": "assistant", "usage": {
        "input": 10, "output": 5, "cacheRead": 2, "cacheWrite": 0,
        "cost": {"total": 0.01}}}})
    if "Attached data file: " in message:
        path = message.rsplit("Attached data file: ", 1)[1].strip()
        with open(path, encoding="utf-8") as handle:
            delta(f" data-len={len(handle.read())}")
    settle()
    return False


MODELS = [
    {"id": "model-a", "name": "Stub A", "provider": "stub"},
    {"id": "model-b", "name": "Stub B", "provider": "stub"},
]


def main() -> None:
    prompts = 0
    hanging = False
    current = MODELS[0]
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        command = json.loads(line)
        kind = command.get("type")
        rid = command.get("id")
        if kind == "prompt":
            prompts += 1
            emit({"type": "response", "command": "prompt", "success": True, "id": rid})
            hanging = serve_prompt(command.get("message", ""), prompts, sys.stdin)
        elif kind == "abort":
            emit({"type": "response", "command": "abort", "success": True, "id": rid})
            if hanging:
                hanging = False
                settle()
        elif kind == "get_available_models":
            emit({"type": "response", "command": kind, "success": True, "id": rid,
                  "data": {"models": MODELS}})
        elif kind == "get_state":
            emit({"type": "response", "command": kind, "success": True, "id": rid,
                  "data": {"model": current, "isStreaming": False}})
        elif kind == "set_model":
            wanted = [m for m in MODELS if m["provider"] == command.get("provider")
                      and m["id"] == command.get("modelId")]
            if wanted:
                current = wanted[0]
                emit({"type": "response", "command": kind, "success": True, "id": rid,
                      "data": current})
            else:
                emit({"type": "response", "command": kind, "success": False, "id": rid,
                      "error": "Model not found"})
        else:
            emit({"type": "response", "command": kind, "success": True, "id": rid})


if __name__ == "__main__":
    main()
