# rrlm

[![CI](https://github.com/RamXX/rrlm/actions/workflows/ci.yml/badge.svg)](https://github.com/RamXX/rrlm/actions/workflows/ci.yml)
[![CI: Dagger](https://img.shields.io/badge/portable%20gate-Dagger-131226?logo=dagger&logoColor=white)](docs/CI.md)
[![coverage 96%](https://img.shields.io/badge/coverage-96%25-brightgreen)](docs/CI.md)
[![tests 292 passing](https://img.shields.io/badge/tests-292%20passing-brightgreen)](tests)
[![license MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

> The canonical CI gate is a provider-agnostic [Dagger](https://dagger.io) pipeline
> (`make ci` = `dagger call ci`: ruff + the fully offline suite, 80% coverage floor);
> GitHub Actions runs the same contract natively for live status. See
> [docs/CI.md](docs/CI.md).

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
The reasoning behind the design, its tradeoffs, and when this is the wrong tool are in
[docs/DESIGN.md](docs/DESIGN.md).

## Getting started

Three commands to a working install, the first answer needs **no model and no
API key**:

```bash
# 1. Install (clones into ~/.rrlm, sets up the env, puts the CLIs on PATH)
curl -fsSL https://raw.githubusercontent.com/RamXX/rrlm/main/install.sh | bash

# 2. Prove the install end to end with the built-in reference engine
#    (deterministic, offline, model-free): counts the lines of stdin.
printf 'alpha\nbeta\ngamma\n' | rrlm-solve --engine reference -i count-lines -d -
# -> 3

# 3. Check what your environment can do (Pi config, credentials, backends, engines)
rrlm-doctor
```

Then run a first real solve. Any of these work; pick the one that matches what
you have:

```bash
# You use Pi: no flags needed, rrlm uses the model Pi is currently set to.
rrlm-solve -i "Which product id has the most negative reviews?" -d @reviews.csv

# You have an OpenRouter key (no Pi needed):
OPENROUTER_API_KEY=... rrlm-solve --main openrouter/qwen/qwen3.6-27b -i "..." -d @data.txt

# You have any provider's key (OpenAI shown; Anthropic etc. work the same):
OPENAI_API_KEY=... rrlm-solve --main openai/gpt-5.1 -i "..." -d @data.txt
```

Where to next: [use it from Pi, the CLI, or Python](#use-it) for the full
surface (typed answers, files, multi-question runs, sessions),
[when to use it and when not](#when-to-use-it-and-when-not) before you commit,
and the [documentation map](#documentation-map) for everything else.

## When to use it, and when not

Reach for rrlm when the work is **exact computation or per-item judgment over
data too large (or too costly) to put in a prompt**: ledgers, logs, large
CSVs, document sets, codebases; or when you want a **small local model doing
verified, code-first work** at $0.

Skip it when the data is small (under roughly 12k tokens the REPL scaffold
costs more than it saves; just read the data in context) or when you need low
latency (a solve is an agent run, not a completion). Two capabilities live in
specific places rather than in the core: conversation comes from the host
agent (behind Pi, the agent carries the dialogue and delegates data work to
rrlm; a library `Session` persists computed state, not chat), and
byte-identical replays come from deterministic engine plugins (the live LLM
harness is intentionally nondeterministic; a symbolic or compiled-strategy
engine behind `engine=` replays exactly). The full reasoning, with the
tradeoffs behind each default, is in [docs/DESIGN.md](docs/DESIGN.md).

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
`rrlm-solve`, `rrlm-traces`, and `rrlm-doctor` commands on your PATH via `uv`):

```bash
curl -fsSL https://raw.githubusercontent.com/RamXX/rrlm/main/install.sh | bash
```

Or do it by hand:

```bash
git clone https://github.com/RamXX/rrlm && cd rrlm
uv sync                  # development: run via `make` / `uv run`
uv tool install .        # optional: install the rrlm-solve / rrlm-traces / rrlm-doctor CLIs
```

### Prerequisites

- [`uv`](https://docs.astral.sh/uv/), the Python package manager this project uses.
  Install with `curl -LsSf https://astral.sh/uv/install.sh | sh` (the one-line
  installer above bootstraps `uv` for you if it is missing).
- [Deno](https://deno.land), only if you opt into the Pyodide sandbox (`--backend
  jspi`). Install with `curl -fsSL https://deno.land/install.sh | sh`, or `brew
  install deno`, or follow the [Deno install guide](https://docs.deno.com/runtime/getting_started/installation/).
  The default `supervisor` backend needs no Deno.
- [Docker](https://www.docker.com/) (or another container runtime), only if you want to
  run CI locally (the Dagger pipeline, `make ci`) or use the `sbx` sandbox backend. The
  `sbx` backend additionally needs the `sbx` CLI (`brew install docker/tap/sbx`, then
  `sbx login`).
- The optional `web` extra (`uv tool install 'rrlm[web]'` or `uv sync --extra web`),
  only if you use live web access (`--web`); see [Live web access](#live-web-access-opt-in).
- A model configured in Pi (see below), or an `OPENROUTER_API_KEY`. No Pi at all is
  fine too: any built-in provider works with an explicit reference plus its usual env
  key, e.g. `rrlm-solve --main openai/gpt-5.1 ...` with `OPENAI_API_KEY` set.

Run `rrlm-doctor` any time to check all of the above (versions, Pi config,
credentials, backends, extras, and whether your local model servers are up).

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

# typed answers: parse the result into a real type, not prose
rrlm-solve -i "How many distinct customers?" -d @orders.csv --answer-type int

# real files (PDF / XLSX / DOCX / CSV / anything): mounted into the sandbox,
# with the matching document skills attached automatically
rrlm-solve -i "Sum the invoice totals." --file q1.pdf --file q2.pdf --answer-type float

# several questions over the same data in ONE run (data loads once)
rrlm-solve -i "Total revenue?" -i "Top product id?" -i "How many rows?" -d @orders.csv
```

```python
from rrlm import solve, asolve, solve_many

result = solve("Which product id has the most negative reviews?", data=text)
print(result["answer"], result["usage"]["cost_usd"])

# typed answers, file mounting, and your own host-side tools
result = solve("Extract the total.", files=["invoice.pdf"], answer_type=float,
               tools=[my_lookup_fn])

# several questions, one amortized run
result = solve_many(["Total revenue?", "Top product?"], data=text)
print(result["answers"])

# async twin for servers / other agents (same parameters)
result = await asolve("...", data=text)
```

On the `jspi` backend, document-skill packages (pymupdf, openpyxl, python-docx)
install automatically inside the sandbox; on `supervisor`/`sbx`, install them in
the environment rrlm runs in if your task needs them.

## How the harness decides

The orchestrator follows a fixed doctrine (`src/rrlm/playbooks.py`): probe the data
cheaply, prefer deterministic Python over LM calls, use `predict()` only for genuine
semantic judgment (batched with `asyncio.gather`), recurse only when a sub-problem's
working set is too large, and verify before answering. Orchestrator thinking
defaults to off (it adds latency and variance without accuracy here); point the leaf
(`--sub`) at a cheap non-thinking model to make the fan-out path inexpensive.

The doctrine text is swappable: `--doctrine <file>` (or `solve(..., doctrine=...)`)
replaces it for a run, which is how an RLM-GEPA-optimized doctrine is deployed
(see below).

## Guardrails, traces, and sandbox isolation

All of these live at the rrlm layer (within predict-rlm's constructs; nothing patches
predict-rlm). The budgets are **global to the run**: they are shared across the whole
`rlm_spawn` tree, so a recursing agent cannot multiply its allowance.

- **Guardrails**: `--timeout` (env `RRLM_TIMEOUT`) caps total wall-clock and cancels
  an overrunning run; `--max-llm-calls` caps sub-LM (`predict`) calls across all
  depths; `--max-spawns` caps child agents; `--max-cost` (env `RRLM_MAX_COST`) is a
  soft USD ceiling checked before each call (needs a cost-reporting provider such as
  OpenRouter; local models are $0 and never trip it); `--max-iterations` caps REPL
  turns per agent.
- **Traces for optimization**: set `RRLM_TRACE_DIR` and every `rlm_solve` call writes
  its predict-rlm RunTrace plus an `index.jsonl` (instruction -> answer -> config),
  including failed runs, whose traces carry the strongest optimization signal.
  Inspect and curate with `rrlm-traces list`, `rrlm-traces read --last`,
  `rrlm-traces grep <pattern>`.
- **Execution sandbox** (`--backend` / env `RRLM_BACKEND`): `supervisor` (host
  CPython, the default: fastest, no extra runtime, fine for trusted data), `jspi`
  (Deno/Pyodide WASM sandbox, local, $0), or `sbx` (Docker Linux container, strongest
  isolation; needs Docker and the `sbx` CLI; auto-reuses a warm container to keep
  per-call overhead low). Prefer `jspi` or `sbx` when the data or task is untrusted.
  An *implicit* supervisor default (no flag, no env) warns once per process that
  model-generated Python runs on the host; any explicit choice, including
  `RRLM_BACKEND=supervisor`, is respected quietly.
  See [docs/LOCAL_SERVING.md](docs/LOCAL_SERVING.md).

## Multi-turn sessions (a persistent REPL namespace)

One-shot `solve()` calls forget everything they computed. A `Session` keeps the
supervisor interpreter - and with it the whole REPL namespace: variables,
parsed structures, defined helpers - alive across calls, so follow-ups build
on completed work instead of re-paying the data load and the scaffold:

```python
from rrlm import Session

with Session(main_model="openrouter/qwen/qwen3.6-27b") as session:
    session.solve("Parse the ledger into `entries`; report the row count.", data=text)
    session.solve("Using `entries`, total the amounts per vendor.")
    session.solve("Which vendor's total changed most vs last quarter's `entries`?")
```

What persists is the **interpreter namespace**, not conversation history: each
call is still its own agent run with fresh budgets and its own trace, so
instructions should name the variables earlier calls created. `reset()`
clears the namespace without ending the session; `close()` (or the context
manager) releases the interpreter process. Long-lived hosts that cannot link
Python get the same capability over a line-delimited protocol via the
`rrlm-session` CLI (one persistent Session on stdin/stdout; see
`src/rrlm/session_server.py`), which is how the [Pi
extension](pi/README.md) holds a session across a conversation
(`rlm_solve(..., session: true)`), and over the Agent Client Protocol via
`rrlm-acp` (see [ACP](#acp-an-agent-for-buzz-zed-or-any-acp-client)). Sessions run on the `supervisor`
backend (trusted data) only; changing the tool set between calls (e.g.
toggling `web=`) starts a fresh namespace by design. For warm `sbx`
containers across one-shot runs (filesystem persistence, not namespace
persistence) use `RRLM_SBX_NAME`.

## ACP: an agent for Buzz, Zed, or any ACP client

`rrlm-acp` speaks the [Agent Client Protocol](https://agentclientprotocol.com)
(v1, JSON-RPC over stdio), so any ACP client can spawn it as an agent. It
serves two very different agents behind one command:

**`rrlm-acp --pi`: the full agent (what a persistent host like Buzz wants).**
Each ACP session runs one long-lived `pi --mode rpc` subprocess, so the
ongoing conversation, memory, tools, skills, and extensions all live in Pi,
where they already work; with the [rlm-backend extension](pi/README.md)
installed, Pi keeps delegating data-heavy subtasks to rrlm exactly as it does
in the terminal. Pi's streaming (text, thinking, tool executions) is
forwarded live as ACP updates; `session/cancel` maps to Pi's `abort`;
embedded ACP resources are staged to files and referenced by path, so bulk
data stays out of Pi's context and lands in `rlm_solve` instead. Extension
UI dialogs are auto-cancelled (headless host), and Pi's stderr passes
through for diagnostics. Everything after `--` goes to the pi subprocess:

```bash
rrlm-acp --pi -- --provider anthropic   # any pi flags: -e, --skill, --model, ...
```

**`rrlm-acp` (no flags): the data oracle.** One persistent
[`Session`](#multi-turn-sessions-a-persistent-repl-namespace) per ACP
session: the REPL namespace survives across prompts, but each prompt is one
budgeted solve with no conversation history. Text blocks are the
instruction, embedded resources are the data, progress streams as tool-call
updates. Stdio MCP servers passed at session setup are mounted when the
`mcp` extra is installed. Right when the client carries the conversation
itself and needs exact answers over large data.

Both modes configure models and budgets like `rrlm-session`: `--main`,
`--sub`, `--timeout`, `--max-cost` (in `--pi` mode these export as `RRLM_*`
into Pi's environment for the extension), or the `RRLM_*` variables
directly.

For [Block's Buzz](https://github.com/block/buzz), the whole integration
(the zero-configuration [`scripts/rrlm-buzz`](scripts/rrlm-buzz) launcher,
harness registration, the model picker, persona delivery, steering,
cancellation, usage metrics, file handling in channels, a team-ready
persona, and troubleshooting) is documented in
[docs/BUZZ.md](docs/BUZZ.md). The protocol subset served (and what is
deliberately left out) is documented in `src/rrlm/acp_server.py`.

## Engine plugins (route a run to another solver)

The built-in predict-rlm harness is one way to fill the solve contract
(instruction + data + budgets -> typed answer + usage). An *engine plugin* is
another solver behind the same contract: a sealed interpreter, an audited
symbolic engine, anything. rrlm documents only the protocol; engine packages
register themselves at install time and never appear in rrlm's tree or docs.

```bash
rrlm-solve --engine <name> -i "..." -d @file    # or env RRLM_ENGINE
```

```python
result = solve("...", data=text, engine="<name>")   # library form
```

Selection is always **explicit** (an argument or `RRLM_ENGINE`); nothing routes
by inference, because choosing between trust levels is the caller's policy
decision. With an engine, model/backend/doctrine parameters are ignored (the
engine owns its execution); `--timeout` stays enforced host-side, and the
call/cost ceilings pass down as the engine's budget lease. Engine-specific
knobs pass through opaquely with `engine_options={...}` (CLI:
`--engine-option key=value`, repeatable, JSON values recognized); rrlm never
inspects them. Engine runs land in the same `RRLM_TRACE_DIR` `index.jsonl`
history with an `engine` field. The contract is versioned (`rrlm.engine/1`):
an engine may declare the protocol it was built against and the conformance
suite fails fast on a mismatch.

Writing an engine: implement the small protocol in
[`src/rrlm/engines.py`](src/rrlm/engines.py) (an async `solve()` returning an
`EngineResult`; failures go in `result.error`, never exceptions), expose it
through the `rrlm.engines` entry-point group, and validate it from your own
test suite with `rrlm.conformance.check_engine_sync`. The in-tree `reference`
engine (deterministic, offline, model-free) is the protocol demo the docs,
tests, and `rrlm-doctor` use:

```bash
rrlm-solve --engine reference -i count-lines -d @big.log     # protocol smoke test
rrlm-doctor                                                  # lists installed engines
```

## MCP tools and progress events (opt-in)

Mount any MCP server's tools as awaitable host tools for a run (needs the
`mcp` extra: `uv sync --extra mcp`). Remote servers over **streamable HTTP**
are the preferred form; local stdio subprocesses and the legacy HTTP+SSE
transport are also supported:

```python
from rrlm import solve
from rrlm.mcptools import MCPServerSpec

result = solve(
    "Look up the vendor in the CRM and report its status.", data=text,
    mcp=[
        MCPServerSpec(url="https://crm.example.com/mcp",           # streamable HTTP
                      headers={"Authorization": "Bearer ..."},
                      allow=("lookup_vendor",)),
        MCPServerSpec(command="local-tools-server"),               # stdio subprocess
        MCPServerSpec(url="https://old.example.com/sse",           # legacy SSE
                      transport="sse"),
    ],
)
```

```bash
rrlm-solve --mcp https://crm.example.com/mcp -i "..." -d @file    # streamable HTTP
rrlm-solve --mcp "crm-mcp-server --profile prod" ...              # stdio command
rrlm-solve --mcp sse+https://old.example.com/sse ...              # legacy SSE
```

Protocol generations are negotiated by the SDK at `initialize`, so servers on
the current stateless-HTTP revision and servers still on older stateful
revisions both work; the offline suite exercises all three transports and
both server styles against a real MCP server subprocess.

The agent sees each tool's name and description and calls
`await tool_name(...)` from the REPL; connections live exactly as long as the
run. MCP tools execute host-side with this process's permissions on every
backend (the sandbox isolates generated Python, not host tools), so prefer
`allow=` for servers that expose more than the task needs.

For hosts that want live progress, `solve(..., on_event=callback)` streams
structured events (`run_started`, `llm_call`, `spawn_started`/`spawn_finished`,
`run_finished`; see `src/rrlm/events.py`), and `rrlm-solve --events` prints
them as JSONL on stderr while stdout carries the answer.

## Optimize the doctrine with RLM-GEPA (opt-in)

The doctrine is a text component, so it is optimizable. `rrlm-gepa` (the `gepa`
extra) runs [RLM-GEPA](https://pypi.org/project/predict-rlm/) with the real rrlm
harness as the executor and your examples as the score:

```bash
uv sync --extra gepa                       # or: uv tool install 'rrlm[gepa]'
export RRLM_GEPA_DATASET=examples.jsonl    # {"instruction","data"|"data_file","expected","checker"}
export RRLM_MAIN=... RRLM_SUB=...          # models for the executor (Pi refs)
rrlm-gepa optimize --check                 # validate wiring
rrlm-gepa optimize --max-metric-calls 400  # evolve the doctrine
rrlm-gepa stats runs/<run-dir>
rrlm-solve --doctrine <winner.txt> ...     # deploy the winner
```

See `src/rrlm/gepa.py` for the dataset format (checkers: `exact`, `contains`,
`number`) and details. To measure generalization instead of memorization, tag
examples with a `domain` and hold whole domains out of optimization
(`RRLM_GEPA_HOLDOUT_DOMAINS=code,web`; they never enter train or val), then
score the winner on exactly those domains with `rrlm-gepa eval --doctrine
<winner.txt>`. The protocol and the target evaluation matrix are in
[docs/EVALS.md](docs/EVALS.md).

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
these two vetted functions. `fetch` refuses non-public addresses (localhost,
RFC-1918 ranges, cloud metadata endpoints), re-checking every redirect hop, so
the tools cannot be steered at your intranet; set `RRLM_WEB_ALLOW_PRIVATE=1` to
lift that in a trusted environment. Retrieval is keyless (DuckDuckGo search +
main-text extraction). See [`src/rrlm/webtools.py`](src/rrlm/webtools.py).

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

## Embedding: the typed contract and LM injection

`solve()`/`asolve()` are a kwargs facade over a typed, versioned contract
(`rrlm.solve/1`). Servers, workflow engines, and agents that want
machine-stable types use it directly: build a `SolveRequest`, run it, and get
a `SolveResult` whose `error` is a typed `RunError` with a stable category
(`timeout`, `budget`, `execution`, `engine`) instead of a string to parse:

```python
from rrlm import SolveRequest, SolvePolicy, arun

request = SolveRequest(
    instruction="Total the amounts per vendor.",
    inputs={"data": text},
    policy=SolvePolicy(timeout_s=300, max_llm_calls=20),
)
result = await arun(request)          # or rrlm.run(request) from sync code
if result.error and result.error.category == "budget":
    ...
```

Embedders that do not use Pi at all can inject `dspy.LM` instances directly
as `main_model`/`sub_model` (or `ModelSelection`): rrlm adopts foreign
instances into its accounting model, so budgets, usage, and events keep
working, and forces LM caching off (cache hits would falsify cost
accounting). Configure reasoning on the LM you inject; `reasoning=` alongside
an injected instance is rejected. See
[`src/rrlm/contract.py`](src/rrlm/contract.py) for the contract and its
versioning rules.

## Documentation map

| Document | What it covers |
| --- | --- |
| README (this page) | Install, getting started, the full usage surface |
| [docs/DESIGN.md](docs/DESIGN.md) | Design choices and tradeoffs, expected behavior, ideal use cases, when rrlm is the wrong tool |
| [docs/FINDINGS.md](docs/FINDINGS.md) | Benchmark methodology and results: RLM vs context-stuffing |
| [docs/LOCAL_SERVING.md](docs/LOCAL_SERVING.md) | The settled local model stack ($0, offline) and the bake-off that chose it |
| [docs/EVALS.md](docs/EVALS.md) | The evaluation matrix and the GEPA domain-holdout generalization protocol |
| [docs/CONTRACT_V2.md](docs/CONTRACT_V2.md) | Proposal: input polymorphism, resources, and artifacts for the next contract revision |
| [docs/CI.md](docs/CI.md) | The portable Dagger CI gate, the GitHub Actions workflow, and why repeat runs are fast |
| [pi/README.md](pi/README.md) | Wiring rrlm into Pi: the `rlm_solve` tool and the routing skill |
| [docs/BUZZ.md](docs/BUZZ.md) | Running Pi + rrlm as a team agent in Block's Buzz over ACP |
| [examples/crm](examples/crm) | The code-generation showcase: a local 35B model builds a Go CRM |
| [experiments/superpowers](experiments/superpowers) | Beyond-context tasks the RLM solves that stuffing cannot |
| [CONTRIBUTING.md](CONTRIBUTING.md) | How to contribute; test and lint expectations |

## Development

```bash
make test                    # offline suite (unit + integration + e2e; no network, no Deno)
make lint                    # ruff
make cov                     # offline suite with the 80% coverage gate
```

The suite runs fully offline and deterministically: a local OpenAI-compatible stub
server stands in for the model, so the integration and e2e tests exercise the real code
path (the `rrlm-solve` CLI, the library, and the predict-rlm REPL loop) with no LLM-call
mocks. Combined coverage of `src/rrlm/` is 96%, gated at 80% by `make cov` and CI.

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
