# Design: choices, tradeoffs, and expected behavior

> rrlm docs: [README](../README.md) | Design (this page) |
> [CI](CI.md) | [Benchmarks and findings](FINDINGS.md) |
> [Local serving](LOCAL_SERVING.md) | [Pi integration](../pi/README.md)

This page explains what rrlm is optimized for, what it deliberately is not,
and the reasoning behind the design decisions a user will run into. Read it
when you want to know *why* the software behaves the way it does; the
[README](../README.md) covers *how* to use it.

## The posture

rrlm is built on one idea: **write code, run it, read the result, fix,
iterate** beats emitting an answer in one stochastic shot. The data lands as a
variable in a REPL, never in the model's context; the model probes it with
Python, computes deterministically wherever possible, spends cheap sub-model
calls only on irreducible semantic judgment, and verifies before submitting.
Two consequences follow, and both are measured rather than asserted (see
[FINDINGS.md](FINDINGS.md)):

* Cost and accuracy stay flat as data grows. Context-stuffing degrades
  silently long before it hits the window limit; the RLM does exact work on
  data far larger than the window for cents.
* Even a small local model produces quality output, because the loop, not the
  parameter count, carries the quality (see [examples/crm](../examples/crm)).

## When rrlm is the right tool

* **Exact aggregation, search, or reconciliation over large data**: ledgers,
  logs, CSV/Parquet-scale text, codebases; anything where the answer is
  computable or checkable and the data does not fit (or should not go)
  in a prompt.
* **Per-item semantic judgment at scale**: classify or extract across
  thousands of items, where deterministic code shapes the work and batched
  cheap leaf calls handle the judgment.
* **Document questions over real files** (PDF / XLSX / DOCX / CSV), mounted
  into the sandbox with matching document skills.
* **A delegated data tool inside a coding agent**: the original use. Pi (or
  any host) hands `rlm_solve` a data-heavy subtask and gets a verified answer
  back without polluting its own context.
* **Driving small local models through code-first work** at $0 and fully
  private ([LOCAL_SERVING.md](LOCAL_SERVING.md)).

## When it is not

* **Small data.** The REPL scaffold costs roughly 15-25k tokens of overhead.
  Below about 12k tokens of data, just read the data in context; the routing
  skill shipped for Pi encodes exactly this rule.
* **Latency-sensitive paths.** A solve is a multi-turn agent run (tens of
  seconds to minutes), not a single completion.
* **A standalone conversational assistant.** rrlm itself keeps no chat
  history: nothing re-reads a transcript. The conversation layer is
  deliberately someone else's job, and the shipped composition provides it:
  behind [Pi](../pi/README.md), the agent carries the dialogue and delegates
  data work to `rlm_solve`, holding one persistent session across the
  conversation when it passes `session: true` (backed by the `rrlm-session`
  bridge). Library users get multi-turn *computation* through `Session`, which
  persists the REPL namespace (variables, parsed data, helpers) between
  calls, with instructions naming what earlier calls created. What rrlm alone
  cannot give you is dialogue memory.
* **Byte-reproducible pipelines on the live harness.** The LLM-driven
  harness is intentionally nondeterministic (see "Expected behavior");
  reliability there comes from in-run verification, not replayability. When
  you need identical replays, rrlm still serves them, through the engine
  plugin route: an engine that computes deterministically (a symbolic solver,
  a compiled strategy) returns identical output for identical input through
  the same `solve()` contract, budgets, and trace history. The in-tree
  `reference` engine is a working example. Route with `engine=`; just do not
  expect determinism from the live harness itself.
* **Untrusted input on the default backend.** The default `supervisor`
  backend executes model-written Python on your host. For adversarial or
  unvetted data, choose `jspi` or `sbx` explicitly, or route to a sealed
  engine plugin.

## Expected behavior

* **The result contract.** `solve()` returns a dict: `answer`, `error`,
  `wall_clock_s`, `trace_file`, `spawn_stats`, `usage`, `config`. Run
  failures (timeouts, budget exhaustion, execution errors) come back in
  `error` as a string; the call itself raises only for *caller* mistakes
  (missing files, unknown engine, invalid parameter combinations). A raised
  exception means fix your call; an `error` value means the run failed.
  Underneath the dict sits a typed, versioned contract (`rrlm.solve/1`,
  `src/rrlm/contract.py`): `SolveRequest` in, `SolveResult` out, with
  `RunError` categories (`timeout`, `budget`, `execution`, `engine`) for
  machine consumers; `solve()` compiles into it and renders back.
* **Nondeterminism is by design.** Temperature is 0.2 and LM caching is off
  (real calls are measured, never cache hits). Two runs of the same task may
  take different paths and different wall-clock times. Reliability comes from
  in-REPL verification before SUBMIT, not from reproducibility.
* **Budgets are hard edges, globally.** `--max-llm-calls`, `--max-spawns`,
  `--timeout`, and `--max-cost` are shared across the whole recursion tree; a
  spawning agent cannot multiply its allowance. The cost ceiling is soft by
  one in-flight call; the wall clock cancels the run outright.
* **One loud warning, once.** If the supervisor backend is selected purely by
  default (no flag, no env), rrlm warns once per process that model-generated
  Python runs on the host. Any explicit choice, including
  `RRLM_BACKEND=supervisor`, is respected silently.
* **Cleanup is bounded.** Interpreter processes are released after every
  one-shot run and on `Session.close()`; a stuck release escalates from a
  polite shutdown to killing the runner within seconds, never hanging.
* **Cost accounting is best-available.** OpenRouter runs reconcile against
  the provider's reported cost; local models report $0; other providers use
  the best per-call estimate available.

## Design decisions and what they trade away

**Models come from Pi; there is no model registry.** rrlm resolves
`provider/model` references from your existing Pi config (or plain env keys),
so it never maintains its own catalog and never drifts from what you already
run. Embedders outside Pi can also inject `dspy.LM` instances directly;
foreign instances are adopted into rrlm's accounting model (budgets, usage,
events) rather than trusted blindly, and caching is forced off. Tradeoff:
model metadata for references is only as good as the config it reads, and an
injected LM's reasoning configuration is the caller's job.

**One-shot solves by default; sessions are opt-in.** A stateless
`solve(instruction, data)` is the simplest correct contract for a delegated
tool: no hidden state, budgets and traces per call. `Session` exists for the
follow-up-question shape and persists only the interpreter namespace.
Tradeoff: multi-turn work must name its variables across calls, and sessions
are limited to the supervisor backend where a live interpreter can persist.

**The default backend is fast, not sandboxed.** `supervisor` (host CPython)
is the default because the primary use is trusted data on your own machine
and it needs no extra runtime. Isolation is explicit: `jspi` (WASM) or `sbx`
(container). Tradeoff: safety-by-default is sacrificed for speed and zero
setup, mitigated by the one-time warning and by making the sandboxes one flag
away. The reverse default would tax every trusted run with sandbox overhead
and a Deno/Docker dependency.

**The doctrine is swappable text, and therefore optimizable.** The
orchestrator's behavioral contract (probe, compute, batch judgment, recurse
only under capacity pressure, verify) is one text block, replaceable per run
and evolvable with [RLM-GEPA](../README.md#optimize-the-doctrine-with-rlm-gepa-opt-in)
against your own labeled examples. Tradeoff: behavior is prompt-taught, not
enforced; a weak model can deviate from the doctrine.

**Engine plugins route by explicit choice only.** Alternate solvers (sealed
interpreters, audited engines) register through an entry-point group and are
selected by name: an argument, `--engine`, or `RRLM_ENGINE`. Nothing routes by
inference, because choosing between trust levels is caller policy, and a
heuristic that decides whether untrusted data reaches host Python is a
vulnerability, not a convenience. Tradeoff: no automatic "best engine for
this task" magic; callers own the routing rules.

**Tools run host-side; the sandbox stays closed.** Web access and MCP servers
execute on the host and are bridged into the REPL by name. The model's
generated code never gets a network; it reaches the world only through vetted
functions (with SSRF guards on `fetch`, allowlists for MCP servers).
Tradeoff: host tools run with your process's permissions on every backend, so
tool selection, not the sandbox, is the security boundary for them.

**Progress events can never break a run.** `on_event` callbacks are
synchronous, best-effort, and exception-swallowed. Tradeoff: a broken display
layer fails silently rather than loudly, deliberately, because observability
must not become a new failure mode.

**The tests are offline and mock-free; CI is a portable gate.** Integration
and e2e tests drive the real stack against local stub servers (an
OpenAI-compatible LM stub, a real MCP server) as genuine subprocesses; no
LLM-call mocks anywhere. The canonical CI gate is a Dagger module any
provider or laptop runs identically; GitHub Actions runs the same make
contract natively for speed on ephemeral runners ([CI.md](CI.md)).
Tradeoff: stub-driven determinism cannot catch model-behavior regressions;
those live in the `real`-marked tests and the benchmarks.

**predict-rlm internals are coupled by feature detection, pinned by tests.**
rrlm rides on [predict-rlm](https://pypi.org/project/predict-rlm/) and, where
the upstream surface is missing (per-turn retries, sandbox timeouts, budget
sharing, interpreter lifecycle), it feature-detects and compensates rather
than forking. Every such coupling point has a test that fails loudly if the
upstream moves. Tradeoff: version bumps of predict-rlm need attention; the
suite tells you where.

## Where the evidence lives

Claims above about cost, accuracy, and local-model capability are backed by
recorded runs: [FINDINGS.md](FINDINGS.md) (RLM vs context-stuffing
benchmarks), [../experiments/superpowers](../experiments/superpowers)
(beyond-context tasks), [../examples/crm](../examples/crm) (the local-model
code-generation run), and [LOCAL_SERVING.md](LOCAL_SERVING.md) (the serving
bake-off).
