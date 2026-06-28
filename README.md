# rrlm

An **RLM-first backend for the [Pi coding agent](https://github.com/earendil-works/pi)**.

`rrlm` lets a coding agent handle data far larger than its context window. Instead
of stuffing a big file, log, or codebase into the conversation, the agent delegates
to a Recursive Language Model (RLM) harness: the data lands in a sandboxed Python
REPL, the model acts only by writing code to probe it, fans out cheap sub-model
calls only for irreducible semantic judgment, verifies, and returns an answer. The
agent keeps a *map* of state in context, not the state itself.

It ships as a Pi tool (`rlm_solve`) plus a routing skill, and as a standalone CLI
(`rrlm-solve`) and library (`from rrlm import solve`). Built on
[`predict-rlm`](https://pypi.org/project/predict-rlm/).

**Why it works** (see [docs/FINDINGS.md](docs/FINDINGS.md)): on exact aggregation
over many items, context-stuffing silently miscounts before it hits context limits,
while the RLM's cost and accuracy stay flat in data size. For 262K-context local
models, stuffing a 1M-token task is impossible -- the RLM is the only way to do it
at all, for under a cent.

## Models come from Pi

rrlm does **not** keep its own model registry. It resolves models from your Pi
config (`~/.pi/agent/models.json`, `settings.json`, `auth.json`, and
`~/.pi/config.json`) -- local servers, OpenRouter, OpenAI, Anthropic, z.ai,
whatever you have configured. A model reference is `provider/model` (e.g.
`openrouter/qwen/qwen3.6-27b`, `lmstudio/qwen/qwen3.6-27b`) or a bare model id; omit
it to use the model Pi is currently set to.

## Install

```bash
uv tool install rrlm          # or: pipx install rrlm
```

You also need [Deno](https://deno.land) for the default Pyodide sandbox (`jspi`
backend), and either a model configured in Pi or an `OPENROUTER_API_KEY`.

From a checkout for development:

```bash
git clone https://github.com/RamXX/rrlm && cd rrlm
uv sync
```

## Use it

### As a CLI / library

```bash
# inline / file / stdin; models default to your Pi config
rrlm-solve -i "Total revenue for completed EMEA orders." -d @orders.csv
echo "<data>" | rrlm-solve -i "..." -d -
rrlm-solve -i "..." -d @data.txt --main openrouter/qwen/qwen3.6-27b --json
```

```python
from rrlm import solve

result = solve("Which product id has the most negative reviews?", data=text)
print(result["answer"], result["usage"]["cost_usd"])
```

### As a Pi backend (the main event)

Wire the extension + skill into Pi so the agent delegates data-heavy subtasks
automatically. See [pi/README.md](pi/README.md). In short:

```bash
pi -e /path/to/rrlm/pi/extensions/rlm-backend/index.ts \
   --skill /path/to/rrlm/pi/skills/rlm-first
```

The agent gets an `rlm_solve` tool and a skill telling it *when* to use it (large
data, exact aggregation/search over many items, per-item judgment at scale) and
when not to (small data it can just read). By default `rlm_solve` orchestrates with
the same model Pi is currently using.

## How the harness decides

The orchestrator follows a fixed doctrine (`src/rrlm/playbooks.py`): probe the data
cheaply, prefer deterministic Python over LM calls, use `predict()` only for genuine
semantic judgment (batched with `asyncio.gather`), recurse only when a sub-problem's
working set is too large, and verify before answering. Orchestrator thinking
defaults to off (it adds latency and variance without accuracy here); point the leaf
(`--sub`) at a cheap non-thinking model to make the fan-out path inexpensive.

## Guardrails, traces, and sandbox isolation

All of these live at the rrlm layer (within predict-rlm's constructs -- nothing patches
predict-rlm):

- **Guardrails** (hard ceilings): `--timeout` (env `RRLM_TIMEOUT`) caps total
  wall-clock and cancels an overrunning run; `--max-llm-calls` caps sub-LM calls (the
  de-facto spend ceiling); `--max-iterations` caps REPL turns.
- **Traces for optimization**: set `RRLM_TRACE_DIR` and every `rlm_solve` call writes
  its predict-rlm RunTrace plus an `index.jsonl` (instruction -> answer -> config) --
  the dataset for [RLM-GEPA](https://pypi.org/project/predict-rlm/). Inspect and curate
  with `rrlm-traces list`, `rrlm-traces read --last`, `rrlm-traces grep <pattern>`.
- **Execution sandbox** (`RRLM_BACKEND`): `supervisor` (host CPython, default, fastest),
  `jspi` (Deno/Pyodide WASM, local, $0), or `sbx` (Docker Linux container, strongest
  isolation; auto-reuses a warm container to keep per-call overhead low). See
  [docs/LOCAL_SERVING.md](docs/LOCAL_SERVING.md).

## Reproduce the benchmarks

The research side lives in `src/rrlm/bench/` and writes per-run artifacts under
`runs/`. With an `OPENROUTER_API_KEY`:

```bash
make compare SIZE=5000       # RLM vs context-stuffed baseline + comparison table
make report                  # table across all recorded runs
```

Full results and methodology: [docs/FINDINGS.md](docs/FINDINGS.md).

## Real-use-case evals

```bash
make eval-tabular            # exact aggregation over a large CSV (verifiable truth)
make eval-bugfind            # code reasoning over a real repository
make eval-pi                 # end-to-end Pi session that delegates to rlm_solve
```

## Local, offline, $0 inference

You can run everything against on-device models (no API keys, fully private). See
[docs/LOCAL_SERVING.md](docs/LOCAL_SERVING.md) and the `make serve-orch` /
`make serve-leaf` targets.

## Development

```bash
make test                    # unit tests (no network, no Deno)
make lint                    # ruff
```

See [CONTRIBUTING.md](CONTRIBUTING.md). MIT licensed.
