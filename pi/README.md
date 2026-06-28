# rrlm as a Pi backend

This wires the RLM-first harness into [pi](https://github.com/earendil-works/pi)
as a delegatable tool, so a pi agent can offload data-heavy subtasks to the
recursive-language-model harness instead of pulling large data into its own
context.

## Pieces

- `extensions/rlm-backend/index.ts` registers the `rlm_solve` tool. It stages
  the data to a temp file and shells out to `rrlm-solve --json`, returning the
  verified answer plus usage metrics. By default it orchestrates with the **same
  model Pi is currently using** (read from the tool's execution context) and
  resolves endpoints/credentials from your Pi config.
- `skills/rlm-first/SKILL.md` teaches the agent *when* to delegate: large
  data, exact aggregation/search over many items, or per-item semantic judgment
  at scale; and when not to (small data it can just read).

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

1. Install rrlm so `rrlm-solve` is on your PATH (and Deno is available for the
   default jspi sandbox). rrlm is not on a package index; install from source:

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

## Environment knobs read by the extension

| Var | Default | Meaning |
|-----|---------|---------|
| `RRLM_MAIN` | Pi's current model | orchestrator model reference (`provider/model`) |
| `RRLM_SUB` | same as `RRLM_MAIN` | leaf model reference for `predict()` fan-out |
| `RRLM_BACKEND` | `jspi` | sandbox backend (`jspi`, `sbx`, or `supervisor`) |
| `RRLM_WEB` | unset | `1` to give the agent live web retrieval (`web_search`/`fetch`); needs the rrlm `web` extra |
| `RRLM_DIR` | unset (use installed `rrlm-solve`) | project checkout to run via `uv run` |

## Verified

End-to-end: pi calls `rlm_solve`, the harness runs against the model Pi is using,
and the answer returns, confirmed via json-mode `tool_execution_start` /
`tool_execution_end` events (see `examples/eval_pi.py`). The extension's `execute`
signature is version-robust (it detects the AbortSignal and the execution context
by shape) across recent pi releases.
