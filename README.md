# rrlm

An **RLM-first backend for the [Pi coding agent](https://github.com/earendil-works/pi)** --
and a demonstration that the Recursive Language Model (RLM) is a **posture for quality
work, not just a trick for huge context.**

Most RLM work fixates on cramming data bigger than the context window into a REPL.
That's real, but it misses the point. RLM is **code-first and verify** -- write code,
run it, read the result, fix, iterate -- instead of emitting an answer in one stochastic
shot. That posture makes even a **small local model generate quality software.**

We prove it two ways:

1. **Code generation (the headline).** A 35B MoE model on a laptop builds a complete
   graph CRM in Go -- file by file, compiling as it goes, *fixing its own bugs* -- in
   ~12 minutes, at $0, with **minimal data/context**. The capability is the
   code-first / run-it / verify / iterate loop, not a big prompt. See
   **[examples/crm](examples/crm)**.
2. **Data beyond context (the usual RLM story).** Exact computation over data far larger
   than the window: the agent writes code to probe a sandboxed REPL, fans out cheap
   sub-model calls only for irreducible semantic judgment, and verifies -- keeping a
   *map* of state in context, not the state itself. Context-stuffing silently miscounts
   long before it hits the limit; the RLM stays exact and cheap as data grows (a
   1M-token task on a 262K-context model is impossible to stuff -- the RLM does it for
   under a cent). See [docs/FINDINGS.md](docs/FINDINGS.md) and
   [experiments/superpowers](experiments/superpowers).

It ships as a Pi tool (`rlm_solve`) plus a routing skill, a CLI (`rrlm-solve`), and a
library (`from rrlm import solve`). Built on [`predict-rlm`](https://pypi.org/project/predict-rlm/).

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

### Generate code with it (the headline use)

The same posture drives *building software*: a local model writes code one file at a
time, compiles, reads the error, fixes, and iterates -- delegating to `rlm_solve` only
for the rare data-heavy subtask. [**examples/crm**](examples/crm) is the worked proof:
a 35B model on a laptop builds a complete graph CRM in Go in ~12 minutes, self-fixing
its own bugs, with minimal data/context. That is the capability this repo exists to
demonstrate.

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
