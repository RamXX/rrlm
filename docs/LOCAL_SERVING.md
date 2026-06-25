# Local model serving (offline, $0 inference)

rrlm runs against on-device models with no API keys and full privacy. This is the
optional power-user path; for first use, any model configured in Pi (including a
cloud one) works without any of this.

The settled local configuration from the experiments (see
[FINDINGS.md](FINDINGS.md)) is a **high-fidelity orchestrator** plus a **cheap,
quantized, non-thinking leaf**:

- Orchestrator: a Pi-harness-tuned Qwen3.6-27B (`bytkim/Qwen3.6-27B-MTP-pi-tune`),
  Q6_K GGUF, served by `llama-server`, no-thinking, temp 0.7.
- Leaf: `supergemma-26b` (gemma-4-26b uncensored, 4-bit MLX) served via DFlash.

## Scripts

All live in `scripts/local-serving/` and are parameterized by environment variables
(no absolute paths baked in). Binaries (`llama-server`, `dflash`, `mlx_lm.server`)
are resolved from `PATH`; override with `$DFLASH`, `$MLX_LM_SERVER`, etc.

| Script | Serves | Default port |
|--------|--------|--------------|
| `serve-pitune.sh` | pi-tune Q6_K orchestrator via `llama-server` | 8773 |
| `serve-models.sh` | supergemma leaf via DFlash (heretic orchestrator optional) | 8771 |
| `serve-qwen36.sh` | official Qwen3.6-27B 8-bit via `mlx_lm.server` (draftless) | 8772 |
| `dflash-qwen36.sh` | official Qwen3.6-27B 8-bit via DFlash (speculative) | 8770 |

```bash
# orchestrator (own terminal)
make serve-orch                       # -> scripts/local-serving/serve-pitune.sh

# leaf (own terminal)
make serve-leaf                       # -> scripts/local-serving/serve-models.sh start

make serve-stop                       # stop everything
```

Prerequisites depend on the script: [`llama.cpp`](https://github.com/ggml-org/llama.cpp)
(`llama-server`), [`mlx_lm`](https://github.com/ml-explore/mlx-lm), and/or DFlash.
GGUF/MLX weights download from Hugging Face on first run (some DFlash draft repos are
gated -- run `hf auth login`). Override any model/draft/port via the env vars
documented at the top of each script.

## Point Pi (and rrlm) at the local servers

Add the local endpoints to `~/.pi/agent/models.json` as OpenAI-compatible providers,
e.g.:

```json
{
  "providers": {
    "pitune": {
      "baseUrl": "http://127.0.0.1:8773/v1",
      "api": "openai-completions",
      "apiKey": "local",
      "models": [{ "id": "qwen3.6-27b-pi-tune", "contextWindow": 262100, "maxTokens": 65536 }]
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

Then run with those references:

```bash
rrlm-solve -i "..." -d @data.txt \
  --main pitune/qwen3.6-27b-pi-tune --sub supergemma/supergemma4-26b
```

## Why this shape

Wall-clock for these workloads is dominated by leaf fan-out, not orchestrator decode,
so orchestrator *reliability* (clean, well-formed turns) matters more than its speed.
Speculative tricks (DFlash, MTP) and smaller quants traded reliability for speed the
workload cannot use. rrlm also auto-raises the per-turn sandbox timeout to 3600s for
local endpoints, because local leaves serve serially and a wide fan-out runs as one
REPL turn. Details in [FINDINGS.md](FINDINGS.md).
