#!/usr/bin/env bash
# DFlash speculative decoding for the OFFICIAL Qwen3.6-27B at 8-bit (MLX).
#
# Target: mlx-community/Qwen3.6-27B-8bit -- official weights, Q8 (near-bf16
# fidelity), so the orchestrator reliably follows the RLM turn format AND is
# fast. Draft: z-lab/Qwen3.6-27B-DFlash (block diffusion, ~15-16 accepted
# tokens/step, LOSSLESS -- the target validates every draft, so output quality
# equals the target running alone).
#
# This replaces the earlier uncensored "heretic" target, which was unreliable
# as an orchestrator (intermittent malformed turns) -- and the MTPLX server,
# whose engine could spin-wedge. Official Q8 on the proven dflash-mlx server is
# the reliable+fast path.
#
# Prereqs (one-time): authenticate to Hugging Face and get access to the gated
# draft repo z-lab/Qwen3.6-27B-DFlash (hf auth login), then:
#   ./dflash-qwen36.sh bench   # baseline MLX vs DFlash, prints tok/s + speedup
#   ./dflash-qwen36.sh serve   # OpenAI-compatible server on :8770 (thinking OFF)
#
# Ports follow the serve-models.sh convention: uncommon + consecutive (877x) to
# avoid clashes with LM Studio (1234), Ollama (11434), and the usual 8000/8080.
set -euo pipefail

# Override MODEL to point at a local path or a different repo if desired.
MODEL="${MODEL:-mlx-community/Qwen3.6-27B-8bit}"
DRAFT="${DRAFT:-z-lab/Qwen3.6-27B-DFlash}"

# Thinking is disabled by default to match the rrlm reasoning=off orchestrator
# config (thinking adds latency/variance without accuracy gains for this work).
# Clients can still override per request via chat_template_kwargs.
NOTHINK_ARGS=(--chat-template-args '{"enable_thinking": false}')

# dflash resolved from PATH; override with $DFLASH if installed elsewhere.
DFLASH="${DFLASH:-dflash}"
cmd="${1:-bench}"

case "$cmd" in
  bench)
    # Quantize the bf16 draft to 4-bit on the fly to cut memory.
    exec "$DFLASH" benchmark --model "$MODEL" --draft "$DRAFT" \
      --draft-quant w4:gs64 --max-tokens 256 --block-tokens 16
    ;;
  serve)
    HOST="${HOST:-127.0.0.1}"
    PORT="${PORT:-8770}"   # uncommon + consecutive with supergemma (8771)
    exec "$DFLASH" serve --model "$MODEL" --draft "$DRAFT" \
      --draft-quant w4:gs64 --host "$HOST" --port "$PORT" "${NOTHINK_ARGS[@]}"
    ;;
  *)
    echo "usage: $0 {bench|serve}" >&2; exit 2 ;;
esac
