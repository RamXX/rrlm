# Local model serving (offline, $0 inference)

rrlm runs against on-device models with no API keys and full privacy. This is the
optional power-user path; for first use, any model configured in Pi (including a
cloud one) works without any of this.

The settled local configuration (see [FINDINGS.md](FINDINGS.md) and the Track C
bake-off in [../experiments/dflash-vs-mtp/FINDINGS.md](../experiments/dflash-vs-mtp/FINDINGS.md))
is a **fast, high-quality MoE orchestrator** plus a **cheap, quantized, non-thinking
leaf**:

- Orchestrator: **Ornith-1.0-35B** (`deepreinforce-ai/Ornith-1.0-35B-GGUF`), a Qwen3.5
  **MoE with 256 experts and 8 active per token** (35B total, ~3B active; agentic coding,
  SWE-bench Verified 75.6), Q6_K, served by `llama-server` with continuous batching
  (`--parallel`, `-fa on`), thinking off for the RLM role.
- Leaf: `supergemma-26b` (gemma-4-26b uncensored, 4-bit MLX) served via DFlash.

Ornith is the only orchestrator, chosen by the bake-off below.

## How we chose the local model (the bake-off)

This stack is not a guess; it is the result of a measured bake-off. Full record:
[../experiments/dflash-vs-mtp/FINDINGS.md](../experiments/dflash-vs-mtp/FINDINGS.md).
The short version:

1. **Candidates.** A Pi-harness-tuned dense Qwen3.6-27B (pi-tune, Q6_K GGUF) and the
   official Qwen3.6-27B at Q8 (MLX), across three local engines: llama.cpp (GGUF),
   mlx_lm (MLX), and DFlash (MLX speculative decoding), plus llama.cpp MTP
   self-speculative decoding.
2. **Decode speed looked decisive, then wasn't.** On a sustained 512-token generation,
   DFlash hit ~39 tok/s versus ~12.8 (plain MLX) and ~11 (llama.cpp MTP). But run through
   the real RLM loop that advantage mostly vanished: the workload is prefill-bound (a
   large REPL context is re-sent each turn, outputs are short), so decode throughput is
   not the bottleneck. DFlash's apparent edge was largely a prefix-cache artifact.
3. **Concurrency picked the engine.** For parallel agents, the MLX engines (mlx_lm and
   DFlash) are single-stream and collapse at ~8 concurrent requests; llama.cpp
   `--parallel` degrades gracefully. So the engine is llama.cpp.
4. **A MoE picked the model.** Swapping the dense 27B for Ornith-1.0-35B (Qwen3.5 MoE,
   256 experts, 8 active per token) was the real unlock. Because only ~8 experts fire per
   token, prefill (the bottleneck) is cheap and there is GPU headroom to batch. On the
   same llama.cpp engine, Ornith is roughly 5 to 8x faster end-to-end than the dense 27B,
   it passes the full superpowers proof, and it self-fixed a bug the dense model needed a
   human to fix.

## Performance (M3 Max, 128GB; Ornith Q6_K on llama.cpp --parallel)

- Single short RLM solve: a few seconds warm; longer with many REPL turns or heavy leaf
  fan-out.
- Code generation: the CRM example builds the full spec in ~12.5 minutes over 3 passes
  (Phase 1 in ~2 min, the full spec in one ~8.5 min pass, plus a ~2 min fix pass). See
  [../examples/crm](../examples/crm).
- Concurrency: aggregate throughput holds up to ~60 tok/s across 8 parallel clients with
  no collapse; the single-stream MLX paths collapsed at the same load.
- Quant rule: Q6 or higher for the orchestrator, never Q4 (the leaf is intentionally
  4-bit, a cheap high-volume classifier).

## Scripts

All live in `scripts/local-serving/` and are parameterized by environment variables
(no absolute paths baked in). Binaries (`llama-server`, `dflash`) are resolved from
`PATH`; override with `$DFLASH` etc.

| Script | Serves | Default port |
|--------|--------|--------------|
| `serve-ornith.sh` | Ornith-1.0-35B Q6_K orchestrator via `llama-server` (`--parallel`, `-fa on`) | 8774 |
| `serve-models.sh` | supergemma leaf via DFlash | 8771 |
| `purge-dflash-cache.sh` | drop the regenerable DFlash prefix cache (leaf) | n/a |

```bash
# orchestrator (own terminal): Ornith, thinking off, continuous batching
make serve-orch                       # -> serve-ornith.sh (NOTHINK=1)

# leaf (own terminal)
make serve-leaf                       # -> serve-models.sh start

make serve-stop                       # stop everything
```

`serve-ornith.sh` env vars: `PARALLEL` (continuous-batching slots, default 4),
`CTX` (default 65536), `NOTHINK=1` (suppress `<think>` for the RLM role), `NGL`,
`PORT`, `QUANT_FILE`. For parallel agents keep the default `--parallel 4`; for the
superpowers proof (sequential cells needing the full 65536 context wall) use
`PARALLEL=1 CTX=65536`.

Prerequisites: [`llama.cpp`](https://github.com/ggml-org/llama.cpp) (`llama-server`)
for the orchestrator, and DFlash for the leaf. GGUF/MLX weights download from Hugging
Face on first run (some DFlash draft repos are gated, run `hf auth login`).

## Point Pi (and rrlm) at the local servers

Add the local endpoints to `~/.pi/agent/models.json` as OpenAI-compatible providers,
e.g.:

```json
{
  "providers": {
    "ornith": {
      "baseUrl": "http://127.0.0.1:8774/v1",
      "api": "openai-completions",
      "apiKey": "local",
      "models": [{ "id": "ornith-1.0-35b", "contextWindow": 262144, "maxTokens": 32768 }]
    },
    "supergemma": {
      "baseUrl": "http://127.0.0.1:8771/v1",
      "api": "openai-completions",
      "apiKey": "local",
      "models": [{ "id": "supergemma4-26b", "contextWindow": 262100, "maxTokens": 32768 }]
    }
  }
}
```

Then run with those references (this is also the `rrlm-solve` default):

```bash
rrlm-solve -i "..." -d @data.txt \
  --main ornith/ornith-1.0-35b --sub supergemma/supergemma4-26b
```

## Why this shape

Wall-clock for these workloads is dominated by **prefill** of the re-sent REPL context
and by **leaf fan-out**, not orchestrator decode. An MoE orchestrator (Ornith) makes
prefill cheap and, on `llama-server --parallel`, lets concurrent agents batch
instead of collapse (the single-stream MLX paths, mlx_lm and DFlash, serialized and
fell over at ~8 concurrent requests). rrlm also auto-raises the per-turn sandbox
timeout to 3600s for local endpoints, because the local leaf still serves serially and
a wide fan-out runs as one REPL turn (the imdb-1500 cell makes hundreds of sequential
leaf calls). The leaf is the bottleneck for heavy semantic fan-out, but the
supergemma + DFlash leaf is proven and works well, so it is kept as-is (a llama.cpp
`--parallel` leaf was considered and set aside, no need to disturb a working leaf).
Details in [../experiments/dflash-vs-mtp/FINDINGS.md](../experiments/dflash-vs-mtp/FINDINGS.md).


## rlm_solve execution isolation (optional)

`rlm_solve`'s generated Python runs in a sandbox chosen by `--backend` (a predict-rlm
built-in; nothing here patches predict-rlm):

| `RRLM_BACKEND` | isolation | notes |
|---|---|---|
| `supervisor` (default) | none, host CPython | fastest; fine for trusted local use |
| `jspi` | Deno/Pyodide WASM | local, $0, zero-setup (Deno present); slower cold-start |
| `sbx` | real Linux container (Docker) | strongest; needs `predict-rlm[sbx]` + the `sbx` CLI (`brew install docker/tap/sbx`, `sbx login`); ~25s/call ephemeral overhead (use a persistent reused sandbox to amortize) |

`rrlm-solve` itself (and the library) picks the backend from `--backend`, then
`RRLM_BACKEND`, then the `supervisor` default. For an isolated run:

```bash
RRLM_BACKEND=sbx rrlm-solve -i "..." -d @data.txt
```
