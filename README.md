# rrlm -- RLM-first coding agent experiment

Tests the hypothesis that an agent which defaults to Recursive Language Model
(RLM) execution -- all data lives in a REPL, the model acts only by writing
Python, decomposition recurses via depth-gated sub-agents -- outperforms
context-stuffing, especially for small-context models that are strong coders.

Built on `predict-rlm` (DSPy `RLM` subclass) over OpenRouter models.

## Setup

```bash
uv sync
cp .env.example .env   # then fill in OPENROUTER_API_KEY
```

Requires Deno (for the Pyodide sandbox) and, later, Docker (sbx backend).

## Run

```bash
make test                      # unit tests, no network
make smoke                     # RLM condition, synthetic ledger task
make baseline                  # context-stuffed baseline, same task
make compare SIZE=5000         # both conditions + comparison table
make report                    # table across all recorded runs
```

## Metrics

Every run writes `runs/<run_id>/`:

- `run.json` -- model, condition, harness config, library versions
- `events.jsonl` -- one record per LM call: generation id, tokens, and
  authoritative cost (USD) + timing (`latency`, `generation_time`, ms) fetched
  from OpenRouter's `GET /api/v1/generation` endpoint
- `result.json` -- pass/fail, wall clock, spawn depth stats, usage totals
- `trace.json` -- predict-rlm `RunTrace` (per-iteration reasoning/code/output)

Cost precedence per call: generation endpoint `total_cost` > inline
`usage.cost` > LiteLLM estimate. `usage.cost_complete` in `result.json` flags
runs where any call lacked an authoritative figure.
