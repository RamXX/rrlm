# rrlm as a Pi backend

> rrlm docs: [README](../README.md) | [Design](../docs/DESIGN.md) | [CI](../docs/CI.md) | [Benchmarks and findings](../docs/FINDINGS.md) | [Local serving](../docs/LOCAL_SERVING.md) | Pi integration (this page)

This wires the RLM-first harness into [pi](https://github.com/earendil-works/pi)
as a delegatable tool, so a pi agent can offload data-heavy subtasks to the
recursive-language-model harness instead of pulling large data into its own
context.

## Pieces

- `extensions/rlm-backend/index.ts` registers the `rlm_solve` tool. It stages
  the data to a temp file and, per call, either shells out to `rrlm-solve
  --json` (one-shot, the default) or routes the call into a **persistent
  session** (`session: true`): one long-lived `rrlm-session` subprocess whose
  REPL namespace survives across calls, so variables and parsed data from one
  delegation are available to the next. Pi carries the conversation; the
  session carries the computed state. By default the harness orchestrates with
  the **same model Pi is currently using** (read from the tool's execution
  context) and resolves endpoints/credentials from your Pi config; switching
  Pi's model restarts the session with a fresh namespace.
- `skills/rlm-first/SKILL.md` teaches the agent *when* to delegate: large
  data, exact aggregation/search over many items, or per-item semantic judgment
  at scale; when to hold a session (`session: true` for follow-up questions
  over the same data, naming the variables to keep); and when not to delegate
  at all (small data it can just read).

## The session bridge

`rrlm-session` (also `python -m rrlm.session_server`) serves one persistent
[`Session`](../README.md#multi-turn-sessions-a-persistent-repl-namespace)
over line-delimited JSON on stdin/stdout - the protocol the extension speaks,
usable by any long-lived host. One request per line
(`{"id": 1, "op": "solve", "instruction": "...", "data_file": "/path"}`;
also `reset`, `ping`, `close`), one response per line. stdin EOF closes the
session and releases the interpreter, so if the host dies nothing lingers.
See `src/rrlm/session_server.py` for the full protocol.

## The backend entry point

`rrlm-solve` (also `python -m rrlm.solve`) is a general (instruction, data) ->
answer entry point, independent of the benchmark runner:

```bash
# inline / file / stdin
rrlm-solve -i "Which product id has the most negative reviews?" -d @reviews.txt
echo "<data>" | rrlm-solve -i "..." -d -
rrlm-solve -i "..." -d @data.txt --json   # full result incl. metrics
```

Models are **Pi model references** (`provider/model`, or a bare model id) resolved
from `~/.pi/agent/`: local, OpenRouter, OpenAI, Anthropic, z.ai, etc. Omit
`--main` to use the model Pi is set to; `--sub` defaults to the same model (point
it at a cheaper non-thinking model to make the fan-out path inexpensive).

## Install into pi

1. Install rrlm so `rrlm-solve` is on your PATH (Deno is needed only if you opt
   into the jspi sandbox). rrlm is not on a package index; install from source:

   ```bash
   curl -fsSL https://raw.githubusercontent.com/RamXX/rrlm/main/install.sh | bash
   # or from a checkout: git clone https://github.com/RamXX/rrlm && cd rrlm && uv tool install .
   ```

   For development from a checkout, set `RRLM_DIR` instead (the extension then
   runs `uv run rrlm-solve` inside that project).

2. Point pi at the extension and skill. Easiest is per-invocation:

   ```bash
   pi -e $HOME/.pi/agent/extensions/rlm-backend/index.ts \
      --skill /path/to/rrlm/pi/skills/rlm-first
   ```

   Or make it permanent by adding the extension path to `settings.json`
   `extensions` and the skill dir to `skills`.

## Environment knobs

`rrlm-solve` and `rrlm-session` read these themselves, and the extension's
child processes inherit them, so setting them in Pi's environment is all it
takes. The extension only passes `--main`/`--sub` explicitly (Pi's current
model is not in the child env).

| Var | Default | Meaning |
|-----|---------|---------|
| `RRLM_MAIN` | Pi's current model | orchestrator model reference (`provider/model`) |
| `RRLM_SUB` | same as `RRLM_MAIN` | leaf model reference for `predict()` fan-out |
| `RRLM_BACKEND` | `supervisor` | sandbox backend (`supervisor`, `jspi`, or `sbx`) |
| `RRLM_WEB` | unset | `1` to give the agent live web retrieval (`web_search`/`fetch`); needs the rrlm `web` extra |
| `RRLM_TIMEOUT` | unset | hard wall-clock ceiling in seconds per `rlm_solve` call |
| `RRLM_MAX_COST` | unset | soft USD ceiling per call (cost-reporting providers only) |
| `RRLM_TRACE_DIR` | unset | capture RunTraces (+ index.jsonl) for RLM-GEPA |
| `RRLM_DIR` | unset (use installed `rrlm-solve`) | project checkout to run via `uv run` |

## Verified

End-to-end: pi calls `rlm_solve`, the harness runs against the model Pi is using,
and the answer returns, confirmed via json-mode `tool_execution_start` /
`tool_execution_end` events (see `examples/eval_pi.py`). The extension's `execute`
signature is version-robust (it detects the AbortSignal and the execution context
by shape) across recent pi releases.
