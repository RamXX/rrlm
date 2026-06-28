#!/usr/bin/env bash
# llama-server for deepreinforce-ai/Ornith-1.0-35B (Qwen3.5-based 35B MoE, agentic
# coding model; SWE-bench Verified 75.6). Served via llama.cpp with CONTINUOUS
# BATCHING (--parallel), which the Track C concurrency bake-off picked as the only
# engine that degrades gracefully under parallel agents (Paivot). Q6_K only (never Q4).
#
#   ./serve-ornith.sh                 # serve on :8774, continuous batching, fa on
#   NOTHINK=1 ./serve-ornith.sh       # default the chat template to enable_thinking=false
#
# Ornith is a REASONING model (emits <think> by default). For the rrlm RLM
# orchestrator role we run reasoning=off; NOTHINK=1 bakes enable_thinking=false as
# the server default (clients can still override per request). Ornith's recommended
# sampling is temp 0.6 / top-p 0.95 / top-k 20.
set -euo pipefail

REPO="${REPO:-deepreinforce-ai/Ornith-1.0-35B-GGUF}"
QUANT_FILE="${QUANT_FILE:-ornith-1.0-35b-Q6_K.gguf}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8774}"
CTX="${CTX:-131072}"          # total ctx; split across slots -> per-slot = CTX/PARALLEL
PARALLEL="${PARALLEL:-4}"     # continuous-batching slots (concurrent agents). With the
                             # defaults each slot gets 32768 tokens. Coding agents send
                             # large prompts: for a SINGLE heavy agent (e.g. the CRM
                             # build) use PARALLEL=1 to give it the full CTX per request.
NGL="${NGL:-999}"            # force all layers onto Metal (this build's auto-fit can leave it on CPU)

MODEL_PATH="$(find "$HOME/.cache/huggingface/hub/models--deepreinforce-ai--Ornith-1.0-35B-GGUF/snapshots" \
  -name "$QUANT_FILE" 2>/dev/null | head -1)"
if [ -z "$MODEL_PATH" ]; then
  echo "GGUF not found in HF cache; run: hf download $REPO $QUANT_FILE" >&2
  exit 1
fi

THINK_ARGS=()
if [ "${NOTHINK:-0}" = "1" ]; then
  THINK_ARGS=(--chat-template-kwargs '{"enable_thinking": false}')
  echo "thinking DISABLED at server default (enable_thinking=false)" >&2
fi

# -fa on: flash attention (big prefill win; required as a value on this build).
# --parallel N + implicit continuous batching: graceful concurrency for parallel agents.
# q8_0 KV cache: halves KV memory at 65536 ctx across N slots.
exec llama-server --model "$MODEL_PATH" \
  --host "$HOST" --port "$PORT" --ctx-size "$CTX" \
  --n-gpu-layers "$NGL" --parallel "$PARALLEL" -fa on \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --jinja "${THINK_ARGS[@]}" \
  --temp 0.6 --top-p 0.95 --top-k 20
