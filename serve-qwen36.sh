#!/usr/bin/env bash
# Headless OpenAI-compatible server for the OFFICIAL Qwen3.6-27B 8-bit MLX model.
#
# Draftless mlx_lm fallback (no speculative decoding) for when DFlash is not
# wanted -- correctness is identical to the DFlash path, just slower. The
# preferred path is ./dflash-qwen36.sh serve (DFlash speculative decoding).
set -euo pipefail

# Uncommon port (877x convention, like serve-models.sh) to avoid conflicts.
MODEL="${MODEL:-mlx-community/Qwen3.6-27B-8bit}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8772}"

exec /Users/ramirosalas/.local/bin/mlx_lm.server \
  --model "$MODEL" \
  --host "$HOST" \
  --port "$PORT"
