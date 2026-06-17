#!/usr/bin/env bash
# llama-server for bytkim/Qwen3.6-27B-MTP-pi-tune (GGUF, Q6_K) -- a Qwen3.6-27B
# fine-tuned VIA THE PI AGENT HARNESS for no-thinking agentic coding (trained on
# real agent traces: tool calls, code edits, repo work). No-thinking native.
#
# GGUF -> served via llama.cpp's OpenAI-compatible llama-server (not mlx_lm).
# MTP speculative decoding would need a patched llama.cpp (PR #22673); we run
# plain autoregressive here (reliable; spec-decode is marginal for our workload).
#
# Uncommon port (877x convention). Recommended sampling baked in as server
# defaults (temp passed per-request by the harness; the rest default here):
#   temp 0.7, top-p 0.8, top-k 20, min-p 0, presence-penalty 1.5
set -euo pipefail

REPO="${REPO:-bytkim/Qwen3.6-27B-MTP-pi-tune-GGUF}"
QUANT_FILE="${QUANT_FILE:-Qwen3.6-27B-MTP-pi-tune-Q6_K.gguf}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8773}"
CTX="${CTX:-65536}"

MODEL_PATH="$(find "$HOME/.cache/huggingface/hub/models--bytkim--Qwen3.6-27B-MTP-pi-tune-GGUF/snapshots" \
  -name "$QUANT_FILE" 2>/dev/null | head -1)"
if [ -z "$MODEL_PATH" ]; then
  echo "GGUF not found in HF cache; run: hf download $REPO $QUANT_FILE" >&2
  exit 1
fi

# SPEC_MTP=1 enables MTP self-speculative decoding using the GGUF's embedded
# nextn heads (llama.cpp >= build 9670 / PR #22673). No separate draft model.
SPEC_ARGS=()
if [ "${SPEC_MTP:-0}" = "1" ]; then
  SPEC_ARGS=(--spec-type draft-mtp)
  echo "MTP self-speculative decoding ENABLED (--spec-type draft-mtp)" >&2
fi

exec llama-server --model "$MODEL_PATH" \
  --host "$HOST" --port "$PORT" --ctx-size "$CTX" \
  --jinja "${SPEC_ARGS[@]}" \
  --temp 0.7 --top-p 0.8 --top-k 20 --min-p 0 --presence-penalty 1.5
