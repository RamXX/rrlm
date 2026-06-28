# RLM for code generation: LadyCRM

**Why this example exists.** RLM (Recursive Language Models) is almost always pitched as
a way to handle data bigger than the context window. That is real, but it badly
undersells the idea. RLM is a posture for quality work: code-first and verify. Write
code, run it, read the result, fix, iterate, instead of emitting an answer in one
stochastic shot. That posture is what lets a small local model generate quality
software, even when the data and context involved are minimal. Most RLM demos miss this;
it is the whole point of this repo.

Here a single small model on a laptop (Ornith-1.0-35B, a Qwen3.5 MoE with 256 experts
and only 8 active per token, served locally at $0 with no internet) builds **LadyCRM**,
a complete graph-native CRM CLI in Go on an embedded property-graph database (LadybugDB),
file by file, compiling after every step, and fixing its own bugs.

## What the model built

A working `crm` binary with `init`, `contact add/list`, `company add`, `link works-at`,
`deal add/stage/list`, `interact`, `timeline`, `path` (graph traversal), `report` (an
RLM-powered natural-language query over the graph), and `import`. `make build` green,
`make test` green.

- [`SPEC.md`](SPEC.md): what the model was told to build (the LadybugDB API plus the spec).
- [`generated/`](generated/): the actual output, about 9 small Go files (a clean
  `internal/store` package, a dispatcher, and an `rlm` helper). This is a snapshot of
  what the model produced; building it needs Go and LadybugDB (CGO).

## The measured result ([`ASSESSMENT.md`](ASSESSMENT.md))

- About 12.5 minutes, 3 passes end to end: Phase 1 in one pass, the entire full spec in
  one 512s pass, plus one targeted fix pass. [`build_metrics.tsv`](build_metrics.tsv) has
  per-pass timing.
- Every command works, including `report` (structured and semantic queries).
- The model self-fixed a bug it introduced (a Cypher `WITH`-aliasing error in `path`) in
  a single pass. A weaker model on an earlier run needed a human to fix the same command.
  The stronger coder driving the RLM loop self-heals. That contrast is the thesis.

The data and context in this whole exercise are tiny. The capability comes entirely from
the code-first, run-it, verify, iterate loop, not from a big prompt.

## How it works

The build is driven by `pi` (the coding agent) plus rrlm's `rlm-backend` extension, one
file per step with `make build` after each (`drive.sh`). `build_loop.sh` runs Phase 1
(`prompts/continue.md`) then the full spec (`prompts/continue2.md`), gated on `make build`
and `make test` (and on the `rlm` helper and `report` command being present), and records
per-pass time and quality. The orchestrator can delegate data-heavy subtasks to
`rlm_solve`, but for building it mostly just writes and compiles code, which is the point.

## Reproduce

Serve a local orchestrator and leaf (see
[`../../docs/LOCAL_SERVING.md`](../../docs/LOCAL_SERVING.md), which also explains the
bake-off that chose Ornith and the performance you can expect):

```bash
make serve-orch      # Ornith on llama.cpp --parallel (own terminal)
make serve-leaf      # supergemma leaf (own terminal)
```

Then, from the repo root:

```bash
make eval-crm                         # builds LadyCRM with the local Ornith orchestrator
make eval-crm CRM_MODEL=lmstudio/qwen/qwen3.6-27b   # or any Pi-configured model
```

`eval-crm` copies `template/` to `examples/crm/runs/build/` (gitignored), then runs
`build_loop.sh`. To drive it by hand: `bash build_loop.sh <provider/model> <run-dir>`
after `cp -R template <run-dir>`. Building the generated CRM needs Go and LadybugDB; the
model infers the API from `SPEC.md`.
