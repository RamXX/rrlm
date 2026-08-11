# Proposal: contract v2 - input polymorphism, resources, artifacts

> rrlm docs: [README](../README.md) | [Design](DESIGN.md) | [CI](CI.md) |
> [Benchmarks and findings](FINDINGS.md) | [Evals](EVALS.md) |
> [Pi integration](../pi/README.md)

**Status: proposal.** Nothing here is implemented; this page fixes the design
direction so that when the first real consumer arrives, v2 grows from a plan
instead of an improvisation. The current contract is `rrlm.solve/1`
([src/rrlm/contract.py](../src/rrlm/contract.py)): one instruction, one
string input named `data`, typed answers, typed errors. That narrowness is
deliberate and enforced (unknown input keys fail loudly), which is exactly
what makes a clean v2 possible.

## Why not now

The RLM pattern already absorbs most "structured input" needs: data lands as
a REPL variable and the model parses it in-sandbox (JSON text becomes parsed
structures in one cheap turn). The cases v1 genuinely cannot express are
multimodal inputs, live handles (databases, object stores), streaming
sources, and runs that *produce* files. None currently has a consumer in
this codebase. Designing container abstractions with zero consumers is how
the wrong abstraction gets frozen, and this contract is the one interface
that both the native harness and every private engine adapter freeze on.
So: direction now, code when the first consumer exists.

## The v2 shape

`inputs` stays a mapping but its values widen from `str` to input kinds:

```python
inputs = {
    "data": TextInput("..."),                      # v1's blob, explicit now
    "records": StructuredInput(rows),              # JSON-able; lands parsed
    "scan": FileInput("invoice.pdf"),              # today's files=, per-name
    "photos": ResourceInput("s3://bucket/prefix"), # remote, engine-fetched
}
```

Rules carried over from v1's discipline:

* **Every input kind states where it materializes**: in the REPL namespace
  (native harness), in the engine's own runtime, or as a mounted path. No
  kind may silently stringify into the prompt.
* **The native harness derives its DSPy signature from `inputs`** (the
  fixed `task, data` signature generalizes to one field per input name), and
  `rlm_spawn` children inherit the same derivation.
* **Engines receive the same request unchanged.** An engine declares which
  input kinds it supports in its capabilities; the conformance suite gains
  per-kind probes. Unsupported kinds fail at selection time, not mid-run.
* **Outputs gain `artifacts`**: a run that produces files returns
  `(name, path, media_type)` triples collected from a sandbox output
  directory, alongside the typed `answer`, never instead of it.

`SolveRequest` and `SolveResult` field names stay; `PROTOCOL` becomes
`rrlm.solve/2`; a v1 request (plain string in `inputs["data"]`) remains valid
forever by treating `str` as shorthand for `TextInput`. The engine contract
bumps to `rrlm.engine/2` at the same time, with the version-declaration
mechanism already in place (engines declaring `rrlm.engine/1` keep working
against v1-shaped requests, which v2 emits whenever only v1 features are
used).

## Admission criteria for starting the implementation

Any one of these unlocks the work, in this order of likelihood:

1. A multimodal task with a measurable eval (images into `predict()` -
   predict-rlm already supports multimodal leaves).
2. A private engine needing a non-text payload it currently smuggles through
   `engine_options`.
3. A production consumer needing produced-file outputs (reports,
   transformed datasets).

Until then, v1 plus `engine_options` (opaque, engine-specific) is the
pressure valve, and anything that starts feeling like a second input kind
inside `engine_options` is the signal that criterion 2 has arrived.
