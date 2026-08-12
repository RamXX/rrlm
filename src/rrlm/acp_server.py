"""``rrlm-acp``: rrlm (or a full Pi agent) behind the Agent Client Protocol.

ACP (https://agentclientprotocol.com) is the JSON-RPC 2.0 protocol clients
like Zed and Block's Buzz use to drive agents: the client spawns the agent as
a subprocess and speaks newline-delimited JSON over stdin/stdout. Any command
that speaks ACP over stdio works as an agent, which is what this module
provides in two flavors behind one console script (also ``python -m
rrlm.acp_server``):

* ``rrlm-acp`` (default): the **harness agent**. One ACP session = one
  persistent :class:`rrlm.Session`; each prompt is one solve. Namespace
  persists across turns, conversation does not. A precise, budgeted data
  oracle: right when the client brings the conversation and just needs
  (instruction, data) -> answer.
* ``rrlm-acp --pi``: the **full agent**, Pi in the middle. One ACP session =
  one long-lived ``pi --mode rpc`` subprocess, so the ongoing conversation,
  memory, tools, skills, and extensions (including the rlm-backend extension
  that delegates data-heavy work to rrlm) all live where they already work.
  This is the mode a persistent conversational host like Buzz wants.

Protocol surface (v1, deliberately minimal):

* ``initialize``: protocol version 1, no auth methods, no session loading.
  ``embeddedContext`` is advertised because embedded resources are how a
  client hands over data.
* ``session/new``: creates the backend agent. In harness mode, stdio
  ``mcpServers`` entries are mounted via :mod:`rrlm.mcptools` when the
  ``mcp`` extra is installed; in pi mode they are ignored with a stderr
  warning (Pi manages its own tool surface). The response advertises the
  selectable models both ways clients read them: the stable session config
  option (``configOptions`` entry with ``category: "model"``, switched via
  ``session/set_config_option``) and the legacy ``models`` state (switched
  via ``session/set_model``), which is what Buzz's model picker probes. Pi
  mode lists pi's own model catalog; harness mode lists the models in the
  user's Pi config.
* ``session/prompt``: text blocks are the message; embedded ``resource``
  blocks (and readable ``file://`` ``resource_link`` blocks) are the data.
  Progress streams back as ``session/update`` notifications, then the
  response ends the turn with a ``stopReason``.
* ``session/cancel``: cancels the in-flight turn (pi mode: ``abort``); the
  pending prompt answers ``stopReason: "cancelled"`` as the spec requires.

Pi's RPC events map onto ACP updates: ``text_delta`` streams as
``agent_message_chunk``, ``thinking_delta`` as ``agent_thought_chunk``,
``tool_execution_start/end`` as ``tool_call``/``tool_call_update``, and
``agent_settled`` ends the turn. Extension UI dialogs are auto-cancelled
(headless host; the extension sees its documented default), fire-and-forget
UI requests are ignored. Lifecycle mirrors ``rrlm-session``: stdin EOF tears
everything down, so a dead host leaks nothing. Stdout carries protocol lines
only; stderr (ours and pi's) is free for logs.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import shlex
import shutil
import sys
import tempfile
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import unquote, urlparse

from rrlm.session import Session
from rrlm.solve import _env_float, _json_default

# Highest protocol version answered; initialize echoes min(requested, this).
# 2 matches Buzz's pin (and Block's own buzz-agent): it unlocks the
# systemPrompt field on session/new. True ACP v2 semantics are still a draft;
# like buzz-agent, we speak v1 shapes behind the number.
PROTOCOL_VERSION = 2

# Emitted as a session/update while a turn runs; any stdout line resets the
# host's idle clock (Buzz kills silent agents after 900s by default).
KEEPALIVE_INTERVAL_S = 30.0

# JSON-RPC 2.0 error codes.
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603


class RequestError(Exception):
    """A request that cannot be served; carries its JSON-RPC error code."""

    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = code


# --- prompt content ----------------------------------------------------------


def split_prompt(blocks) -> tuple[str, str]:
    """Split ACP content blocks into an (instruction, data) pair.

    ``text`` blocks are the instruction (joined in order). Embedded
    ``resource`` blocks carry data the client already holds in memory;
    ``resource_link`` blocks are read from disk when they point at a local
    file (the agent runs on the client's machine, and the client sent the
    link precisely so the agent would use it). Each data piece is labeled
    with its URI so documents stay distinguishable. Unreadable or non-local
    links degrade to a note appended to the instruction rather than failing
    the turn. Unknown block types are ignored: the capabilities we advertise
    mean a conforming client never sends image/audio.
    """
    instruction_parts: list[str] = []
    data_parts: list[str] = []
    notes: list[str] = []
    for block in blocks or ():
        if not isinstance(block, dict):
            continue
        kind = block.get("type")
        if kind == "text":
            text = block.get("text") or ""
            if text.strip():
                instruction_parts.append(text)
        elif kind == "resource":
            resource = block.get("resource") or {}
            if isinstance(resource.get("text"), str):
                data_parts.append(_labeled(resource.get("uri"), resource["text"]))
        elif kind == "resource_link":
            uri = block.get("uri") or ""
            path = _local_path(uri)
            if path is None:
                notes.append(f"(linked resource, not fetched: {uri})")
                continue
            try:
                data_parts.append(_labeled(uri, path.read_text(encoding="utf-8")))
            except OSError as exc:
                notes.append(f"(linked resource {uri} could not be read: {exc})")
    return "\n\n".join(instruction_parts + notes), "\n\n".join(data_parts)


def _labeled(uri, text: str) -> str:
    return f"[{uri}]\n{text}" if uri else text


def _local_path(uri: str) -> Path | None:
    """A filesystem path for a ``file://`` URI or absolute path; else None."""
    if uri.startswith("file://"):
        return Path(unquote(urlparse(uri).path))
    if uri.startswith("/"):
        return Path(uri)
    return None


def _mcp_specs(servers) -> tuple:
    """ACP ``mcpServers`` entries -> :class:`rrlm.mcptools.MCPServerSpec` tuple.

    Only stdio entries (those with a ``command``) are mounted; HTTP/SSE MCP
    is not advertised in our capabilities. When more than one server is
    mounted, each one's tools are prefixed with its name to keep tool names
    from clashing. Without the ``mcp`` extra the servers are skipped with a
    stderr warning instead of failing the session: a missing optional feature
    must not break the integration.
    """
    stdio = [s for s in servers or () if isinstance(s, dict) and s.get("command")]
    if not stdio:
        return ()
    try:
        from rrlm.mcptools import MCPServerSpec
    except ImportError:
        print(
            "rrlm-acp: ignoring session mcpServers (install rrlm[mcp] to mount them)",
            file=sys.stderr,
        )
        return ()
    specs = []
    for server in stdio:
        env = server.get("env") or ()
        if isinstance(env, dict):  # tolerate both map and ACP's EnvVariable[] shape
            env_map = dict(env)
        else:
            env_map = {e["name"]: e["value"] for e in env if isinstance(e, dict)}
        name = server.get("name") or ""
        specs.append(
            MCPServerSpec(
                command=server["command"],
                args=tuple(server.get("args") or ()),
                env=env_map or None,
                prefix=f"{name}_" if name and len(stdio) > 1 else "",
            )
        )
    return tuple(specs)


# --- agent backends ----------------------------------------------------------
#
# A backend serves one ACP session. ``run_turn(turn_id, instruction, data,
# emit)`` runs one prompt turn, streaming ACP ``update`` dicts through
# ``emit`` (which is thread-safe), and returns the final message text, or
# None when the text was already streamed. It raises for turn failures and
# lets asyncio.CancelledError propagate after its own cleanup.
# ``list_models`` returns ``([{"ref", "display"}, ...], current_ref)`` for
# the session/new advertisement; ``set_model(ref)`` switches and returns
# whether it took. ``usage_snapshot`` returns session-cumulative token
# totals (or None before any usage exists) in the goose ``usage_update``
# vocabulary, which is the only channel Buzz's metrics read. ``aclose``
# releases whatever the backend holds.


class _UsageTotals:
    """Session-cumulative token/cost totals, in usage_update field names."""

    def __init__(self):
        self.input = 0
        self.output = 0
        self.cache_read = 0
        self.cache_write = 0
        self.cost = 0.0

    def add(self, input_tokens, output_tokens, *, cache_read=None, cache_write=None,
            cost=None) -> None:
        self.input += int(input_tokens or 0)
        self.output += int(output_tokens or 0)
        self.cache_read += int(cache_read or 0)
        self.cache_write += int(cache_write or 0)
        self.cost += float(cost or 0.0)

    def snapshot(self) -> dict | None:
        if not (self.input or self.output):
            return None
        usage = {
            "accumulatedInputTokens": self.input,
            "accumulatedOutputTokens": self.output,
            "accumulatedCachedInputTokens": self.cache_read,
            "accumulatedCacheWriteTokens": self.cache_write,
            "accumulatedTotalTokens": self.input + self.output,
        }
        if self.cost:
            usage["accumulatedCost"] = round(self.cost, 6)
        return usage


class HarnessAgent:
    """One persistent :class:`rrlm.Session` behind the backend interface."""

    def __init__(self, *, defaults: dict, mcp: tuple = (), session_factory=Session):
        self._session = session_factory(**defaults)
        self._defaults = dict(defaults)
        self._mcp = mcp
        self._main_override: str | None = None
        self._usage = _UsageTotals()

    async def run_turn(self, turn_id: str, instruction: str, data: str, emit) -> str:
        progress = _HarnessProgress(emit, turn_id)
        options: dict = {"on_event": progress.on_event}
        if self._main_override:
            options["main_model"] = self._main_override
        if self._mcp:
            options["mcp"] = list(self._mcp)
        result = await self._session.asolve(instruction, data, **options)
        usage = result.get("usage") or {}
        self._usage.add(usage.get("prompt_tokens"), usage.get("completion_tokens"),
                        cost=usage.get("cost_usd"))
        if result.get("error") is None:
            return str(result.get("answer"))
        return f"The run failed: {result['error']}"

    def usage_snapshot(self) -> dict | None:
        return self._usage.snapshot()

    async def list_models(self) -> tuple[list[dict], str | None]:
        """The models in the user's Pi config, as Pi refs (provider/id)."""
        from rrlm.pi_config import load_pi_config

        cfg = load_pi_config()
        models = []
        for provider, pdef in (cfg.providers or {}).items():
            for entry in (pdef or {}).get("models") or ():
                if isinstance(entry, dict) and entry.get("id"):
                    ref = f"{provider}/{entry['id']}"
                    models.append({"ref": ref, "display": entry.get("name") or ref})
        current = self._main_override or self._defaults.get("main_model")
        if current and all(m["ref"] != current for m in models):
            models.insert(0, {"ref": current, "display": current})
        return models, current

    async def set_model(self, ref: str) -> bool:
        # Pi refs resolve at call time (rrlm.pi_config), so switching is
        # just overriding the orchestrator for subsequent turns.
        self._main_override = ref
        return True

    async def aclose(self) -> None:
        await asyncio.to_thread(self._session.close)


class _HarnessProgress:
    """Maps one turn's rrlm progress events onto one ACP ``tool_call``.

    The callback runs on the solve's own path, possibly off the event loop
    thread, so it only formats and hands dicts to the thread-safe ``emit``.
    It must stay fast and must never raise (rrlm swallows callback
    exceptions, but relying on that would hide bugs).
    """

    def __init__(self, emit, tool_call_id: str):
        self._emit = emit
        self._tool_call_id = tool_call_id
        self._started = False
        self._calls = 0
        self._spawns = 0
        self._cost = 0.0

    def on_event(self, event: dict) -> None:
        name = event.get("event")
        if not self._started:
            self._started = True
            self._emit({"sessionUpdate": "tool_call", "toolCallId": self._tool_call_id,
                        "title": "rlm solve", "kind": "execute", "status": "in_progress"})
            if name == "run_started":
                return
        if name == "llm_call":
            self._calls += 1
            self._cost += event.get("cost_usd") or 0.0
            self._progress("in_progress")
        elif name == "spawn_started":
            self._spawns += 1
            self._progress("in_progress")
        elif name == "run_finished":
            status = "completed" if event.get("error") is None else "failed"
            self._progress(status, wall_clock_s=event.get("wall_clock_s"))

    def _progress(self, status: str, wall_clock_s=None) -> None:
        title = f"rlm solve: {self._calls} LLM calls"
        if self._spawns:
            title += f", {self._spawns} spawns"
        if self._cost:
            title += f", ${self._cost:.4f}"
        if wall_clock_s is not None:
            title += f", {wall_clock_s:.1f}s"
        self._emit({"sessionUpdate": "tool_call_update", "toolCallId": self._tool_call_id,
                    "status": status, "title": title})


# How a pi tool name renders as an ACP tool-call kind.
_PI_TOOL_KINDS = {
    "read": "read", "bash": "execute", "edit": "edit", "write": "edit",
    "grep": "search", "find": "search", "glob": "search", "ls": "search",
    "rlm_solve": "execute", "web_search": "fetch", "fetch": "fetch",
}

# How long a cancelled or closing turn waits for pi to settle after abort.
PI_ABORT_GRACE_S = 10.0
# Per-line buffer for pi's stdout. One RPC response is one line, and
# get_available_models alone exceeds asyncio's 64KB default many times over.
PI_STREAM_LIMIT = 32 * 1024 * 1024
# Tool output beyond this many characters is elided in tool_call_update
# content; hosts show progress, not transcripts.
PI_TOOL_OUTPUT_LIMIT = 2000


class PiAgent:
    """One long-lived ``pi --mode rpc`` subprocess behind the backend interface.

    Pi owns the conversation: every prompt turn lands in the same pi session,
    so history, memory, extensions, and skills behave exactly as they do in
    the terminal. The bridge stages embedded ACP resources to files and
    points pi at them instead of inlining them, which keeps large data out of
    pi's context and lets the rlm-backend extension delegate it properly.
    """

    def __init__(self, command: list[str], env: dict | None = None,
                 system_prompt: str | None = None):
        self._argv = list(command)
        self._env = dict(env) if env is not None else None
        self._proc: asyncio.subprocess.Process | None = None
        self._reader_task: asyncio.Task | None = None
        self._drain_task: asyncio.Task | None = None
        self._pending: dict[str, asyncio.Future] = {}
        self._next_id = 0
        self._turn: _PiTurn | None = None
        self._staging: str | None = None
        self._usage = _UsageTotals()
        # ref ("provider/id") -> (provider, id); model ids may themselves
        # contain slashes (openrouter/qwen/...), so never re-split a ref.
        self._catalog: dict[str, tuple[str, str]] = {}
        if system_prompt:
            # The host's persona/system prompt (Buzz sends it on session/new
            # once we answer protocol 2). Staged to a file: pi's
            # --append-system-prompt reads file contents, and argv has size
            # limits the 512KB Buzz allows would blow through.
            path = Path(self._staging_dir()) / "system-prompt.md"
            path.write_text(system_prompt, encoding="utf-8")
            self._argv += ["--append-system-prompt", str(path)]

    # -- subprocess plumbing --------------------------------------------------

    async def _ensure_proc(self) -> None:
        if self._proc is not None and self._proc.returncode is None:
            return
        if self._proc is not None:
            raise RuntimeError(
                f"pi exited with code {self._proc.returncode}; check stderr for its logs"
            )
        self._proc = await asyncio.create_subprocess_exec(
            *self._argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=None,  # pi's stderr joins ours; stdout must stay protocol-only
            env={**os.environ, **self._env} if self._env else None,
            limit=PI_STREAM_LIMIT,
        )
        self._reader_task = asyncio.get_running_loop().create_task(self._read_events())

    async def _read_events(self) -> None:
        proc = self._proc
        while True:
            try:
                line = await proc.stdout.readline()
            except (ValueError, asyncio.LimitOverrunError) as exc:
                # A line even the raised limit cannot hold; the stream can no
                # longer be trusted to frame records. Fail loudly and let the
                # EOF cleanup below fail whatever was in flight.
                print(f"rrlm-acp: pi stdout overran the line buffer: {exc}",
                      file=sys.stderr)
                break
            if not line:
                break
            # Strict JSONL per pi's framing rules: split on \n, strip \r.
            raw = line.decode("utf-8", errors="replace").rstrip("\r\n")
            if not raw:
                continue
            try:
                event = json.loads(raw)
            except ValueError:
                print(f"rrlm-acp: dropping non-JSON pi output: {raw[:200]}", file=sys.stderr)
                continue
            self._on_pi_event(event)
        # EOF: pi is gone. Fail whatever was waiting on it.
        for future in self._pending.values():
            if not future.done():
                future.set_exception(RuntimeError("pi exited mid-command"))
        self._pending.clear()
        if self._turn is not None:
            self._turn.fail("pi exited mid-turn; check stderr for its logs")

    def _on_pi_event(self, event: dict) -> None:
        kind = event.get("type")
        if kind == "response":
            future = self._pending.pop(event.get("id"), None)
            if future is not None and not future.done():
                future.set_result(event)
        elif kind == "extension_ui_request":
            self._answer_ui_request(event)
        elif self._turn is not None:
            self._turn.on_event(event)

    def _answer_ui_request(self, event: dict) -> None:
        # Headless host: dialogs are auto-cancelled so no extension can ever
        # block the bridge; the extension receives its documented default.
        # Fire-and-forget methods (notify, setStatus, ...) need no answer.
        if event.get("method") in ("select", "confirm", "input", "editor"):
            print(
                f"rrlm-acp: auto-cancelling pi UI dialog "
                f"{event.get('method')!r}: {event.get('title')!r}",
                file=sys.stderr,
            )
            self._write({"type": "extension_ui_response", "id": event.get("id"),
                         "cancelled": True})

    def _write(self, command: dict) -> None:
        self._proc.stdin.write((json.dumps(command) + "\n").encode("utf-8"))

    async def _command(self, command: dict) -> dict:
        self._next_id += 1
        rid = f"acp-{self._next_id}"
        future = asyncio.get_running_loop().create_future()
        self._pending[rid] = future
        self._write({**command, "id": rid})
        await self._proc.stdin.drain()
        return await future

    # -- the backend interface ------------------------------------------------

    async def run_turn(self, turn_id: str, instruction: str, data: str, emit) -> None:
        await self._ensure_proc()
        if self._drain_task is not None and not self._drain_task.done():
            # A cancelled turn may still be winding down inside pi; sending a
            # new prompt now would be rejected as mid-stream. The drain is
            # bounded, so this wait is too.
            with contextlib.suppress(Exception):
                await self._drain_task
        message = instruction
        if data:
            message += f"\n\nAttached data file: {self._stage(data)}"
        turn = _PiTurn(turn_id, emit)
        self._turn = turn
        try:
            response = await self._command({"type": "prompt", "message": message})
            if not response.get("success"):
                raise RuntimeError(f"pi rejected the prompt: {response.get('error')}")
            try:
                await turn.settled.wait()
            except asyncio.CancelledError:
                # The host's cancel grace is short (Buzz: 5s before it deems
                # the process poisoned), so the abort/drain happens in the
                # background and the caller answers "cancelled" immediately.
                self._drain_task = asyncio.get_running_loop().create_task(
                    self._abort_and_drain(turn))
                raise
            if turn.error is not None:
                raise RuntimeError(turn.error)
        finally:
            turn.fail_open_tools()
            self._usage.add(*turn.usage_tokens, cache_read=turn.usage_cache_read,
                            cache_write=turn.usage_cache_write, cost=turn.usage_cost)
            # On the cancel path the drain task still needs pi's events (it
            # waits for agent_settled), so the turn stays routed and the
            # drain clears it; every other path clears it here.
            if self._drain_task is None or self._drain_task.done():
                self._turn = None
        return None  # the turn's text was streamed as it was generated

    async def steer(self, text: str) -> bool:
        """Deliver a mid-run message into the current turn via pi's steer."""
        if self._turn is None or self._turn.settled.is_set():
            return False
        response = await self._command({"type": "steer", "message": text})
        return bool(response.get("success"))

    def usage_snapshot(self) -> dict | None:
        return self._usage.snapshot()

    async def _abort_and_drain(self, turn: _PiTurn) -> None:
        """Best-effort abort; bounded, so a wedged pi cannot hold the session."""
        try:
            with contextlib.suppress(Exception):
                await self._command({"type": "abort"})
            with contextlib.suppress(asyncio.TimeoutError, Exception):
                await asyncio.wait_for(turn.settled.wait(), PI_ABORT_GRACE_S)
        finally:
            if self._turn is turn:
                self._turn = None

    async def list_models(self) -> tuple[list[dict], str | None]:
        """Pi's own model catalog and current model, over RPC."""
        await self._ensure_proc()
        listed = await self._command({"type": "get_available_models"})
        state = await self._command({"type": "get_state"})
        models = []
        for entry in (listed.get("data") or {}).get("models") or ():
            if not (isinstance(entry, dict) and entry.get("id") and entry.get("provider")):
                continue
            ref = f"{entry['provider']}/{entry['id']}"
            self._catalog[ref] = (entry["provider"], entry["id"])
            models.append({"ref": ref, "display": entry.get("name") or ref})
        current_model = (state.get("data") or {}).get("model") or {}
        current = None
        if current_model.get("id") and current_model.get("provider"):
            current = f"{current_model['provider']}/{current_model['id']}"
        return models, current

    async def set_model(self, ref: str) -> bool:
        provider_and_id = self._catalog.get(ref)
        if provider_and_id is None:
            return False
        await self._ensure_proc()
        response = await self._command({"type": "set_model",
                                        "provider": provider_and_id[0],
                                        "modelId": provider_and_id[1]})
        return bool(response.get("success"))

    def _staging_dir(self) -> str:
        if self._staging is None:
            self._staging = tempfile.mkdtemp(prefix="rrlm-acp-")
        return self._staging

    def _stage(self, data: str) -> str:
        """Write one turn's data payload to a file pi (and rlm_solve) can read."""
        path = Path(self._staging_dir()) / f"data-{uuid.uuid4().hex[:8]}.txt"
        path.write_text(data, encoding="utf-8")
        return str(path)

    async def aclose(self) -> None:
        if self._drain_task is not None and not self._drain_task.done():
            with contextlib.suppress(Exception):
                await self._drain_task
        if self._proc is not None and self._proc.returncode is None:
            with contextlib.suppress(Exception):
                self._proc.stdin.close()
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self._proc.wait(), PI_ABORT_GRACE_S)
            if self._proc.returncode is None:
                self._proc.kill()
        if self._reader_task is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await self._reader_task
        if self._staging is not None:
            shutil.rmtree(self._staging, ignore_errors=True)
            self._staging = None


class _PiTurn:
    """Translates one prompt turn's pi events into ACP session updates."""

    def __init__(self, turn_id: str, emit):
        self._turn_id = turn_id
        self._emit = emit
        self.settled = asyncio.Event()
        self.error: str | None = None
        self._message_index = 0
        self._open_tools: set[str] = set()
        self.usage_tokens = (0, 0)
        self.usage_cache_read = 0
        self.usage_cache_write = 0
        self.usage_cost = 0.0

    def fail(self, message: str) -> None:
        self.error = message
        self.settled.set()

    def fail_open_tools(self) -> None:
        """Give every unfinished tool call a terminal state (cancel path).

        Hosts treat a tool_call without a terminal update as still running
        forever; Buzz's transcript is explicit about wanting the failed state.
        Idempotent: the set empties on first use.
        """
        for tool_call_id in sorted(self._open_tools):
            self._emit({"sessionUpdate": "tool_call_update",
                        "toolCallId": tool_call_id, "status": "failed"})
        self._open_tools.clear()

    def on_event(self, event: dict) -> None:
        kind = event.get("type")
        if kind == "message_update":
            delta = event.get("assistantMessageEvent") or {}
            if delta.get("type") == "text_delta" and delta.get("delta"):
                self._chunk("agent_message_chunk", delta["delta"])
            elif delta.get("type") == "thinking_delta" and delta.get("delta"):
                self._chunk("agent_thought_chunk", delta["delta"])
        elif kind == "message_start":
            self._message_index += 1
        elif kind == "message_end":
            usage = (event.get("message") or {}).get("usage") or {}
            if isinstance(usage, dict):
                self.usage_tokens = (self.usage_tokens[0] + int(usage.get("input") or 0),
                                     self.usage_tokens[1] + int(usage.get("output") or 0))
                self.usage_cache_read += int(usage.get("cacheRead") or 0)
                self.usage_cache_write += int(usage.get("cacheWrite") or 0)
                cost = usage.get("cost")
                if isinstance(cost, dict):
                    cost = cost.get("total")
                if isinstance(cost, (int, float)):
                    self.usage_cost += cost
        elif kind == "tool_execution_start":
            tool_call_id = self._tool_call_id(event)
            self._open_tools.add(tool_call_id)
            self._emit({
                "sessionUpdate": "tool_call",
                "toolCallId": tool_call_id,
                "title": _pi_tool_title(event),
                "kind": _PI_TOOL_KINDS.get(event.get("toolName"), "other"),
                "status": "in_progress",
                "rawInput": event.get("args") or {},
            })
        elif kind == "tool_execution_end":
            tool_call_id = self._tool_call_id(event)
            self._open_tools.discard(tool_call_id)
            update = {
                "sessionUpdate": "tool_call_update",
                "toolCallId": tool_call_id,
                "status": "failed" if event.get("isError") else "completed",
            }
            output = _pi_tool_output(event.get("result"))
            if output:
                update["content"] = [{"type": "content",
                                      "content": {"type": "text", "text": output}}]
            self._emit(update)
        elif kind == "agent_settled":
            self.settled.set()

    def _chunk(self, update: str, text: str) -> None:
        # A shared messageId per logical pi message lets transcript UIs
        # coalesce streamed chunks into one bubble.
        self._emit({"sessionUpdate": update,
                    "messageId": f"{self._turn_id}-m{self._message_index}",
                    "content": {"type": "text", "text": text}})

    def _tool_call_id(self, event: dict) -> str:
        return f"{self._turn_id}-{event.get('toolCallId')}"


def _pi_tool_title(event: dict) -> str:
    name = event.get("toolName") or "tool"
    args = event.get("args") or {}
    detail = args.get("command") or args.get("path") or args.get("instruction") or ""
    detail = str(detail).split("\n", 1)[0]
    if len(detail) > 80:
        detail = detail[:77] + "..."
    return f"{name}: {detail}" if detail else name


def _pi_tool_output(result) -> str:
    if not isinstance(result, dict):
        return ""
    texts = [c.get("text", "") for c in result.get("content") or ()
             if isinstance(c, dict) and c.get("type") == "text"]
    output = "\n".join(t for t in texts if t)
    if len(output) > PI_TOOL_OUTPUT_LIMIT:
        output = output[:PI_TOOL_OUTPUT_LIMIT] + "\n[output elided]"
    # Tools that report metrics (rlm_solve's model/calls/tokens/wall line
    # among them) put a summary in details; surface it after truncation so
    # usage always survives even when the payload is elided.
    details = result.get("details")
    summary = details.get("summary") if isinstance(details, dict) else None
    if summary:
        output = f"{output}\n[{summary}]" if output else f"[{summary}]"
    return output


# --- the ACP server ----------------------------------------------------------


@dataclass
class _AcpSession:
    """Per-session state: the backend agent plus turn bookkeeping."""

    agent: object
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    prompt_task: asyncio.Task | None = None
    turns: int = 0
    models: list[dict] = field(default_factory=list)
    current_model: str | None = None


def _model_config_option(state: _AcpSession) -> dict:
    """The stable session config option advertising this session's models.

    ``displayName`` on the option values is not in the ACP schema (which uses
    ``name``), but Buzz's picker reads it for the label; sending both keeps
    spec clients and Buzz happy.
    """
    return {
        "id": "model",
        "name": "Model",
        "category": "model",
        "type": "select",
        "currentValue": state.current_model or state.models[0]["ref"],
        "options": [{"value": m["ref"], "name": m["display"], "displayName": m["display"]}
                    for m in state.models],
    }


class AcpServer:
    """One ACP v1 agent over an injected line writer.

    ``agent_factory(params)`` builds the backend for one ``session/new``
    (its raw params dict included ``cwd`` and ``mcpServers``); the two
    production factories are :func:`harness_agent_factory` and
    :func:`pi_agent_factory`. ``send`` exists for tests; production writes
    to stdout.
    """

    def __init__(self, *, agent_factory, send=None, steering: bool = False):
        self._agent_factory = agent_factory
        self._sessions: dict[str, _AcpSession] = {}
        self._write_lock = threading.Lock()
        self._send_raw = send if send is not None else self._write_stdout
        self._steering = steering

    # -- wire -----------------------------------------------------------------

    @staticmethod
    def _write_stdout(payload: dict) -> None:
        try:
            sys.stdout.write(json.dumps(payload, default=_json_default) + "\n")
            sys.stdout.flush()
        except (BrokenPipeError, ValueError):
            pass  # host is gone (pipe closed); shutdown is already under way

    def _send(self, payload: dict) -> None:
        # The lock makes the writer safe from event callbacks running off the
        # event loop thread; each payload is exactly one stdout line.
        with self._write_lock:
            self._send_raw(payload)

    def _result(self, rid, result: dict) -> None:
        self._send({"jsonrpc": "2.0", "id": rid, "result": result})

    def _error(self, rid, code: int, message: str) -> None:
        self._send({"jsonrpc": "2.0", "id": rid, "error": {"code": code, "message": message}})

    def send_update(self, session_id: str, update: dict) -> None:
        self._send({"jsonrpc": "2.0", "method": "session/update",
                    "params": {"sessionId": session_id, "update": update}})

    # -- dispatch -------------------------------------------------------------

    async def handle_line(self, line: str) -> None:
        """Parse and dispatch one wire line; never raises."""
        try:
            message = json.loads(line)
            if not isinstance(message, dict):
                raise ValueError("message must be a JSON object")
        except Exception as exc:  # noqa: BLE001, answer instead of desynchronizing
            self._error(None, PARSE_ERROR, f"parse error: {exc}")
            return
        await self.handle_message(message)

    async def handle_message(self, message: dict) -> None:
        rid = message.get("id")
        method = message.get("method")
        params = message.get("params") or {}
        if not isinstance(method, str):
            # A response to a request we never sent, or junk. Only answer
            # things that expect an answer.
            if rid is not None:
                self._error(rid, INVALID_REQUEST, "message has no method")
            return
        try:
            if method == "initialize":
                self._result(rid, self._initialize(params))
            elif method == "session/new":
                self._result(rid, await self._new_session(params))
            elif method == "session/set_config_option":
                self._result(rid, await self._set_config_option(params))
            elif method == "session/set_model":
                self._result(rid, await self._set_model_legacy(params))
            elif method == "session/prompt":
                self._start_prompt(rid, params)
            elif method == "_session/steering":
                self._result(rid, await self._steer(params))
            elif method == "session/cancel":
                # Valid both as a notification and as a request; a request
                # gets an explicit null ack.
                self._cancel(params)
                if rid is not None:
                    self._result(rid, None)
            elif rid is not None:
                self._error(rid, METHOD_NOT_FOUND, f"method not supported: {method}")
        except RequestError as exc:
            if rid is not None:
                self._error(rid, exc.code, str(exc))
        except Exception as exc:  # noqa: BLE001, one bad request must not kill the agent
            if rid is not None:
                self._error(rid, INTERNAL_ERROR, f"{type(exc).__name__}: {exc}")

    # -- methods --------------------------------------------------------------

    def _initialize(self, params: dict) -> dict:
        from rrlm import __version__

        requested = params.get("protocolVersion")
        version = (min(requested, PROTOCOL_VERSION)
                   if isinstance(requested, int) and requested >= 1 else PROTOCOL_VERSION)
        info = {"name": "rrlm", "version": __version__}
        result = {
            "protocolVersion": version,
            "agentCapabilities": {
                "loadSession": False,
                "promptCapabilities": {
                    "image": False,
                    "audio": False,
                    "embeddedContext": True,
                },
            },
            # agentInfo is the v1 name; serverInfo is what newer clients
            # (Buzz among them) read first. Same content, both keys.
            "agentInfo": info,
            "serverInfo": info,
            "authMethods": [],
        }
        if self._steering:
            # Advertises the _session/steering method (mid-run message
            # injection); without it, hosts fall back to cancel-and-reprompt
            # for messages that arrive while a turn is running.
            result["_meta"] = {"steering": {"supported": True}}
        return result

    async def _new_session(self, params: dict) -> dict:
        session_id = f"rrlm-{uuid.uuid4().hex}"
        state = _AcpSession(agent=self._agent_factory(params))
        self._sessions[session_id] = state
        result: dict = {"sessionId": session_id}
        # Model discovery is advisory: a session without a model list is
        # still a working session, and clients probe this response under a
        # tight deadline (Buzz allows ~10s including our spawn), so listing
        # is bounded and failures degrade to "no models advertised".
        try:
            state.models, state.current_model = await asyncio.wait_for(
                state.agent.list_models(), timeout=8.0)
        except Exception as exc:  # noqa: BLE001
            print(f"rrlm-acp: model listing failed: {type(exc).__name__}: {exc}",
                  file=sys.stderr)
        if state.models:
            result["configOptions"] = [_model_config_option(state)]
            legacy: dict = {"availableModels": [{"modelId": m["ref"], "name": m["display"]}
                                                for m in state.models]}
            if state.current_model:
                legacy["currentModelId"] = state.current_model
            result["models"] = legacy
        return result

    async def _set_config_option(self, params: dict) -> dict:
        state = self._state(params)
        if params.get("configId") != "model":
            raise RequestError(INVALID_PARAMS,
                               f"unknown configId: {params.get('configId')!r}")
        await self._apply_model(state, params.get("value"))
        return {"configOptions": [_model_config_option(state)]}

    async def _set_model_legacy(self, params: dict) -> dict:
        state = self._state(params)
        await self._apply_model(state, params.get("modelId"))
        return {"sessionId": params.get("sessionId"), "modelId": params.get("modelId")}

    async def _apply_model(self, state: _AcpSession, ref) -> None:
        if not isinstance(ref, str) or all(m["ref"] != ref for m in state.models):
            raise RequestError(INVALID_PARAMS, f"unknown model: {ref!r}")
        if not await state.agent.set_model(ref):
            raise RequestError(INTERNAL_ERROR, f"switching to {ref!r} did not take")
        state.current_model = ref

    def _start_prompt(self, rid, params: dict) -> None:
        state = self._state(params)
        instruction, data = split_prompt(params.get("prompt"))
        if not instruction:
            raise RequestError(INVALID_PARAMS, "prompt has no text content")
        state.prompt_task = asyncio.get_running_loop().create_task(
            self._run_prompt(rid, params.get("sessionId"), state, instruction, data)
        )

    async def _run_prompt(self, rid, session_id: str, state: _AcpSession,
                          instruction: str, data: str) -> None:
        keepalive = asyncio.get_running_loop().create_task(self._keepalive(session_id))
        try:
            # One turn at a time per session: one interpreter, one pi process.
            async with state.lock:
                state.turns += 1
                text = await state.agent.run_turn(
                    f"{session_id}-turn-{state.turns}", instruction, data,
                    lambda update: self.send_update(session_id, update),
                )
        except asyncio.CancelledError:
            # session/cancel: the spec wants the pending prompt answered with
            # a cancelled stop reason, not a JSON-RPC error or a dead task,
            # and hosts enforce a short grace (Buzz: 5s) before declaring the
            # process poisoned; backends defer their cleanup accordingly.
            asyncio.current_task().uncancel()
            self._result(rid, {"stopReason": "cancelled"})
            return
        except Exception as exc:  # noqa: BLE001, the turn failed; the agent lives on
            self._error(rid, INTERNAL_ERROR, f"{type(exc).__name__}: {exc}")
            return
        finally:
            keepalive.cancel()
        if text is not None:
            self.send_update(session_id, {
                "sessionUpdate": "agent_message_chunk",
                "content": {"type": "text", "text": text},
            })
        self._send_usage(session_id, state)
        self._result(rid, {"stopReason": "end_turn"})

    async def _keepalive(self, session_id: str) -> None:
        while True:
            await asyncio.sleep(KEEPALIVE_INTERVAL_S)
            self.send_update(session_id, {"sessionUpdate": "keepalive"})

    def _send_usage(self, session_id: str, state: _AcpSession) -> None:
        """Session-cumulative usage on the goose notification channel.

        Buzz's token/cost metrics read exactly this (the standard
        ``usage_update`` session/update variant is transcript-only there),
        and the ordering matters: usage rides out before the prompt response.
        """
        snapshot = getattr(state.agent, "usage_snapshot", lambda: None)()
        if snapshot:
            self._send({"jsonrpc": "2.0", "method": "_goose/unstable/session/update",
                        "params": {"sessionId": session_id,
                                   "update": {"sessionUpdate": "usage_update",
                                              **snapshot}}})

    async def _steer(self, params: dict) -> dict:
        state = self._state(params)
        text = "\n\n".join(
            block.get("text", "") for block in params.get("prompt") or ()
            if isinstance(block, dict) and block.get("type") == "text").strip()
        steer = getattr(state.agent, "steer", None)
        if not text or steer is None or not await steer(text):
            # No run to inject into (or a backend that cannot steer): an
            # application error tells the host to dispatch the message as a
            # normal prompt instead.
            raise RequestError(INVALID_PARAMS, "no prompt in flight to steer")
        return {"outcome": "injected"}

    def _cancel(self, params: dict) -> None:
        state = self._sessions.get(params.get("sessionId"))
        if state and state.prompt_task and not state.prompt_task.done():
            state.prompt_task.cancel()

    def _state(self, params: dict) -> _AcpSession:
        session_id = params.get("sessionId")
        state = self._sessions.get(session_id)
        if state is None:
            raise RequestError(INVALID_PARAMS, f"unknown sessionId: {session_id!r}")
        return state

    # -- lifecycle ------------------------------------------------------------

    async def aclose(self) -> None:
        """Cancel in-flight turns and release every session's backend."""
        for state in self._sessions.values():
            if state.prompt_task and not state.prompt_task.done():
                state.prompt_task.cancel()
        for state in self._sessions.values():
            if state.prompt_task:
                with contextlib.suppress(asyncio.CancelledError):
                    await state.prompt_task
            await state.agent.aclose()
        self._sessions.clear()


# --- factories and entry point -----------------------------------------------


def harness_agent_factory(defaults: dict, session_factory=Session):
    """Sessions solve directly on the rrlm harness (the data-oracle agent)."""

    def factory(params: dict) -> HarnessAgent:
        return HarnessAgent(defaults=defaults, mcp=_mcp_specs(params.get("mcpServers")),
                            session_factory=session_factory)

    return factory


def pi_agent_factory(command: list[str], env: dict | None = None):
    """Sessions each hold one ``pi --mode rpc`` subprocess (the full agent)."""

    def factory(params: dict) -> PiAgent:
        if params.get("mcpServers"):
            print("rrlm-acp --pi: ignoring session mcpServers "
                  "(configure MCP in pi instead; pi's shell inherits the "
                  "host env, so host CLIs keep working)", file=sys.stderr)
        system_prompt = params.get("systemPrompt")
        return PiAgent(command, env=env,
                       system_prompt=system_prompt if isinstance(system_prompt, str)
                       else None)

    return factory


async def aserve(server: AcpServer, stdin=None) -> None:
    """The read loop: one wire line per iteration, until stdin EOF.

    Prompts run as tasks so the loop keeps reading (that is what makes
    ``session/cancel`` reachable mid-turn). EOF means the host is done or
    dead; either way everything is torn down and nothing leaks.
    """
    stdin = stdin if stdin is not None else sys.stdin
    try:
        while True:
            line = await asyncio.to_thread(stdin.readline)
            if not line:
                break
            if line.strip():
                await server.handle_line(line.strip())
    finally:
        await server.aclose()


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="rrlm-acp",
        description="Serve an Agent Client Protocol (ACP) v1 agent over stdio: "
                    "the rrlm harness directly (default), or a full Pi agent "
                    "with rrlm as its delegation backend (--pi). Works with "
                    "Block's Buzz, Zed, and any ACP client.",
    )
    parser.add_argument(
        "--pi", action="store_true",
        help="put Pi in the middle: each ACP session runs one persistent "
             "'pi --mode rpc' subprocess (conversation, memory, tools, and "
             "the rlm-backend extension), instead of solving directly",
    )
    parser.add_argument(
        "--pi-command", default=os.environ.get("RRLM_ACP_PI") or "pi",
        help="the pi executable for --pi mode (env RRLM_ACP_PI); "
             "shell-split, so a wrapper like 'uv run pi' works",
    )
    parser.add_argument(
        "--main", "--main-model", dest="main_model",
        default=os.environ.get("RRLM_MAIN") or None,
        help="orchestrator model (Pi ref) for the harness; in --pi mode, "
             "exported as RRLM_MAIN for the rlm-backend extension; "
             "default: env RRLM_MAIN, else Pi's current model",
    )
    parser.add_argument(
        "--sub", "--sub-model", dest="sub_model",
        default=os.environ.get("RRLM_SUB") or None,
        help="leaf model for predict() fan-out; default: env RRLM_SUB, else same as --main",
    )
    parser.add_argument("--timeout", type=float, default=None,
                        help="per-turn wall-clock ceiling in seconds for the harness "
                             "(env RRLM_TIMEOUT); exported in --pi mode")
    parser.add_argument("--max-llm-calls", type=int, default=50,
                        help="per-turn global cap on sub-LM (predict) calls (harness mode)")
    parser.add_argument("--max-cost", type=float, default=None, dest="max_cost_usd",
                        help="per-turn soft USD ceiling (env RRLM_MAX_COST); "
                             "exported in --pi mode")
    parser.add_argument(
        "pi_args", nargs=argparse.REMAINDER,
        help="after '--': extra arguments for the pi subprocess in --pi mode, "
             "e.g. -- --provider anthropic -e ~/.pi/agent/extensions/rlm-backend/index.ts",
    )
    args = parser.parse_args()

    timeout_s = args.timeout if args.timeout is not None else _env_float("RRLM_TIMEOUT")
    max_cost = args.max_cost_usd if args.max_cost_usd is not None else _env_float("RRLM_MAX_COST")

    if args.pi:
        extra = args.pi_args[1:] if args.pi_args[:1] == ["--"] else args.pi_args
        command = [*shlex.split(args.pi_command), "--mode", "rpc", *extra]
        # The rlm-backend extension's child processes read these; passing
        # them into pi's env keeps one ACP command line the single source of
        # configuration.
        env = {var: str(value) for var, value in
               (("RRLM_MAIN", args.main_model), ("RRLM_SUB", args.sub_model),
                ("RRLM_TIMEOUT", timeout_s), ("RRLM_MAX_COST", max_cost))
               if value is not None}
        factory = pi_agent_factory(command, env=env or None)
    else:
        defaults: dict = {"max_llm_calls": args.max_llm_calls}
        if args.main_model:
            defaults["main_model"] = args.main_model
        if args.sub_model:
            defaults["sub_model"] = args.sub_model
        if timeout_s is not None:
            defaults["timeout_s"] = timeout_s
        if max_cost is not None:
            defaults["max_cost_usd"] = max_cost
        factory = harness_agent_factory(defaults)

    # Steering (mid-run message injection) is a pi capability: pi natively
    # queues steer messages between assistant turns. The harness solve is
    # a single uninterruptible run, so it does not advertise it.
    asyncio.run(aserve(AcpServer(agent_factory=factory, steering=args.pi)))


if __name__ == "__main__":
    main()
