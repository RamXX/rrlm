# rrlm

[![CI: Dagger](https://img.shields.io/badge/CI-Dagger-131226?logo=dagger&logoColor=white)](docs/CI.md)
[![not GitHub Actions](https://img.shields.io/badge/not-GitHub%20Actions-lightgrey)](docs/CI.md)
[![coverage 97%](https://img.shields.io/badge/coverage-97%25-brightgreen)](docs/CI.md)
[![tests 151 passing](https://img.shields.io/badge/tests-151%20passing-brightgreen)](tests)
[![license MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

> CI runs as a provider-agnostic [Dagger](https://dagger.io) pipeline (`make ci`), not
> GitHub Actions. These are static badges reflecting the gate (`dagger call ci`: ruff +
> the 98% offline suite, 80% floor); wire any provider to `dagger call ci` for live
> status. See [docs/CI.md](docs/CI.md).

An **RLM-first backend for the [Pi coding agent](https://github.com/earendil-works/pi)**,
and a demonstration that the Recursive Language Model (RLM) is a **posture for quality
work, not just a trick for huge context.**

Most RLM work fixates on cramming data bigger than the context window into a REPL.
That's real, but it misses the point. RLM is **code-first and verify** (write code,
run it, read the result, fix, iterate) instead of emitting an answer in one stochastic
shot. That posture makes even a **small local model generate quality software.**

We prove it two ways:

1. **Code generation (the headline).** A small local model (Ornith-1.0-35B, a Qwen3.5
   MoE with 256 experts and only 8 active per token) on a laptop builds a complete graph
   CRM in Go, file by file, compiling as it goes, *fixing its own bugs*, in ~12.5 minutes,
   at $0, with **minimal data/context**. The capability is the code-first, run-it,
   verify, iterate loop, not a big prompt. See **[examples/crm](examples/crm)**.
2. **Data beyond context (the usual RLM story).** Exact computation over data far larger
   than the window: the agent writes code to probe a sandboxed REPL, fans out cheap
   sub-model calls only for irreducible semantic judgment, and verifies, keeping a
   *map* of state in context, not the state itself. Context-stuffing silently miscounts
   long before it hits the limit; the RLM stays exact and cheap as data grows (a
   1M-token task on a 262K-context model is impossible to stuff, the RLM does it for
   under a cent). See [docs/FINDINGS.md](docs/FINDINGS.md) and
   [experiments/superpowers](experiments/superpowers).

It ships as a Pi tool (`rlm_solve`) plus a routing skill, a CLI (`rrlm-solve`), and a
library (`from rrlm import solve`). Built on [`predict-rlm`](https://pypi.org/project/predict-rlm/).

## Models come from Pi

rrlm does **not** keep its own model registry. It resolves models from your Pi
config (`~/.pi/agent/models.json`, `settings.json`, `auth.json`, and
`~/.pi/config.json`): local servers, OpenRouter, OpenAI, Anthropic, z.ai,
whatever you have configured. A model reference is `provider/model` (e.g.
`openrouter/qwen/qwen3.6-27b`, `lmstudio/qwen/qwen3.6-27b`) or a bare model id; omit
it to use the model Pi is currently set to.

## Install

rrlm is not published to a package index. Install it from source.

One-line install (clones into `~/.rrlm`, sets up the virtualenv, and puts the
`rrlm-solve` and `rrlm-traces` commands on your PATH via `uv`):

```bash
curl -fsSL https://raw.githubusercontent.com/RamXX/rrlm/main/install.sh | bash
```

Or do it by hand:

```bash
git clone https://github.com/RamXX/rrlm && cd rrlm
uv sync                  # development: run via `make` / `uv run`
uv tool install .        # optional: install the rrlm-solve / rrlm-traces CLIs
```

### Prerequisites

- [`uv`](https://docs.astral.sh/uv/), the Python package manager this project uses.
  Install with `curl -LsSf https://astral.sh/uv/install.sh | sh` (the one-line
  installer above bootstraps `uv` for you if it is missing).
- [Deno](https://deno.land), only if you use the default Pyodide sandbox (the `jspi`
  backend). Install with `curl -fsSL https://deno.land/install.sh | sh`, or `brew
  install deno`, or follow the [Deno install guide](https://docs.deno.com/runtime/getting_started/installation/).
  The `supervisor` backend (the `rrlm-solve` default) needs no Deno.
- [Docker](https://www.docker.com/) (or another container runtime), only if you want to
  run CI locally (the Dagger pipeline, `make ci`) or use the `sbx` sandbox backend. The
  `sbx` backend additionally needs the `sbx` CLI (`brew install docker/tap/sbx`, then
  `sbx login`).
- The optional `web` extra (`uv tool install 'rrlm[web]'` or `uv sync --extra web`),
  only if you use live web access (`--web`); see [Live web access](#live-web-access-opt-in).
- A model configured in Pi (see below), or an `OPENROUTER_API_KEY`.

## Use it

### Code with Pi, interactively (the main use)

[Pi](https://github.com/earendil-works/pi) is an interactive coding agent in your
terminal, the same shape you already know from Claude Code: you describe what you want
and it reads files, writes code, runs commands, reads the errors, and fixes them. rrlm
plugs in so you can drive that loop with a small **local** model (or any model), keeping
the code-first, run-it, verify posture this repo is about. Launch Pi with the extension
and skill loaded:

```bash
pi -e /path/to/rrlm/pi/extensions/rlm-backend/index.ts \
   --skill /path/to/rrlm/pi/skills/rlm-first
```

Then just talk to it: "add a `--json` flag and a test for it", "find and fix the failing
case", "build a graph CRM from this spec". It works file by file, compiles, and
self-corrects, the exact loop that produced [**examples/crm**](examples/crm), where a
35B local model built a complete Go CRM in about 12.5 minutes, fixing its own bugs, with
minimal data and context. Point Pi at a local orchestrator for $0, fully private coding
(see [docs/LOCAL_SERVING.md](docs/LOCAL_SERVING.md)).

Alongside the coding loop, the agent also gets an `rlm_solve` tool plus a skill that
tells it *when* to reach for it (large data, exact aggregation or search over many items,
per-item judgment at scale) and when not to (small data it can just read), so a
data-heavy subtask stays exact and cheap without derailing the work. By default
`rlm_solve` orchestrates with the model Pi is currently using. Details in
[pi/README.md](pi/README.md).

### As a CLI / library (data tasks, scripting)

When you want the RLM data capability directly, without the agent, use the `rrlm-solve`
CLI or the Python library:

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

## How the harness decides

The orchestrator follows a fixed doctrine (`src/rrlm/playbooks.py`): probe the data
cheaply, prefer deterministic Python over LM calls, use `predict()` only for genuine
semantic judgment (batched with `asyncio.gather`), recurse only when a sub-problem's
working set is too large, and verify before answering. Orchestrator thinking
defaults to off (it adds latency and variance without accuracy here); point the leaf
(`--sub`) at a cheap non-thinking model to make the fan-out path inexpensive.

## Guardrails, traces, and sandbox isolation

All of these live at the rrlm layer (within predict-rlm's constructs; nothing patches
predict-rlm):

- **Guardrails** (hard ceilings): `--timeout` (env `RRLM_TIMEOUT`) caps total
  wall-clock and cancels an overrunning run; `--max-llm-calls` caps sub-LM calls (the
  de-facto spend ceiling); `--max-iterations` caps REPL turns.
- **Traces for optimization**: set `RRLM_TRACE_DIR` and every `rlm_solve` call writes
  its predict-rlm RunTrace plus an `index.jsonl` (instruction -> answer -> config),
  the dataset for [RLM-GEPA](https://pypi.org/project/predict-rlm/). Inspect and curate
  with `rrlm-traces list`, `rrlm-traces read --last`, `rrlm-traces grep <pattern>`.
- **Execution sandbox** (`RRLM_BACKEND`): `supervisor` (host CPython, default, fastest),
  `jspi` (Deno/Pyodide WASM, local, $0), or `sbx` (Docker Linux container, strongest
  isolation; needs Docker and the `sbx` CLI; auto-reuses a warm container to keep
  per-call overhead low). See [docs/LOCAL_SERVING.md](docs/LOCAL_SERVING.md).

## Live web access (opt-in)

By default the agent answers from the data you give it. Add `--web` (or env
`RRLM_WEB=1`) and it gets two host-side tools, `web_search(query)` and
`fetch(url)`, plus a doctrine to *retrieve and verify instead of answering from
memory*. So "What is the capital of France?" is answered by writing code that
searches, fetches the source, extracts the fact, and cross-checks it, not by
recalling pretraining.

```bash
uv tool install 'rrlm[web]'                  # adds the keyless deps (ddgs + trafilatura)
rrlm-solve --web -i "What is the capital of France? Cite your source."
```

The tools run on the host (predict-rlm bridges tool calls back), so they work on
every backend (`supervisor`, `jspi`, `sbx`) with no network opened inside the
sandbox: the model's own code stays isolated and reaches the web *only* through
these two vetted functions. Retrieval is keyless (DuckDuckGo search + main-text
extraction). See [`src/rrlm/webtools.py`](src/rrlm/webtools.py).

## Reproduce the benchmarks

The research side lives in `src/rrlm/bench/` and writes per-run artifacts under
`runs/`. With an `OPENROUTER_API_KEY`:

```bash
make compare SIZE=5000       # RLM vs context-stuffed baseline + comparison table
make report                  # table across all recorded runs
```

Full results and methodology: [docs/FINDINGS.md](docs/FINDINGS.md).

## Real-use-case evals

The first three are DATA evals; they default to a cloud pair (Qwen3.6-27B + gemma-4-26b
via OpenRouter) so they run without a GPU. The fourth is the CODE-GENERATION example,
the headline use, and it runs on the local Ornith orchestrator.

```bash
make eval-tabular            # data: exact aggregation over a large CSV (verifiable truth)
make eval-bugfind            # data: code reasoning over a real repository
make eval-pi                 # data: end-to-end Pi session that delegates to rlm_solve
make eval-crm                # CODE GEN: a local model builds LadyCRM (see examples/crm)
```

`make eval-crm` needs `make serve-orch` + `make serve-leaf` running; override the
orchestrator with `CRM_MODEL=<provider/model>`. Run the data evals locally too with
`RRLM_MAIN=ornith/ornith-1.0-35b RRLM_SUB=supergemma/...`.

## Local, offline, $0 inference

You can run everything against on-device models (no API keys, fully private). The
settled local stack (a MoE orchestrator + a cheap leaf) and the bake-off that chose it,
with the performance numbers, are in [docs/LOCAL_SERVING.md](docs/LOCAL_SERVING.md);
bring it up with the `make serve-orch` / `make serve-leaf` targets.

## Development

```bash
make test                    # offline suite (unit + integration + e2e; no network, no Deno)
make lint                    # ruff
make cov                     # offline suite with the 80% coverage gate
```

The suite runs fully offline and deterministically: a local OpenAI-compatible stub
server stands in for the model, so the integration and e2e tests exercise the real code
path (the `rrlm-solve` CLI, the library, and the predict-rlm REPL loop) with no LLM-call
mocks. Combined coverage of `src/rrlm/` is 97%, gated at 80% by `make cov` and CI.

### CI (Dagger, provider-agnostic)

CI is a [Dagger](https://dagger.io) module (`dagger/`), not a provider workflow.
It runs the same offline suite as `make cov` in a container, so it requires a
container runtime (Docker):

```bash
make ci                      # = dagger call ci : lint, then the 80% coverage gate
```

Install the Dagger CLI once (`curl -fsSL https://dl.dagger.io/dagger/install.sh | sh`,
or `brew install dagger/tap/dagger`; docs at https://docs.dagger.io), then any CI
provider runs the exact same gate with one command: `dagger call ci`. See
[docs/CI.md](docs/CI.md).

See [CONTRIBUTING.md](CONTRIBUTING.md). MIT licensed.
