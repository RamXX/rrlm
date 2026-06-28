#!/usr/bin/env bash
# Drive ONE Pi build pass against a local orchestrator model, with the rrlm
# rlm-backend extension + rlm-first skill loaded so the agent can delegate
# data-heavy work to rlm_solve. State lives on disk (--no-session), so each pass
# re-reads the current code -- ideal for small-context local models.
#
# Usage: drive.sh <model-ref> <run-dir> <prompt-file>
#   model-ref  e.g. qwen-official/mlx-community/Qwen3.6-27B-8bit  (a Pi model ref)
#   run-dir    the CRM project dir this pass works in
#   prompt-file file whose contents become the agent prompt
#
# Env: RRLM_DIR (default ~/workspace/rrlm), THINKING (default off), PI_TIMEOUT.
set -euo pipefail

MODEL="${1:?model ref}"
# Resolve run dir and prompt file to ABSOLUTE paths before we cd into the run dir.
RUNDIR="$(cd "${2:?run dir}" && pwd)"
PROMPT_FILE="$(cd "$(dirname "${3:?prompt file}")" && pwd)/$(basename "$3")"
RRLM_DIR="${RRLM_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
EXT="$RRLM_DIR/pi/extensions/rlm-backend/index.ts"
SKILL="$RRLM_DIR/pi/skills/rlm-first"
THINKING="${THINKING:-off}"

mkdir -p "$RUNDIR/.drive"
ts=$(date +%Y%m%d-%H%M%S)
log="$RUNDIR/.drive/pass-$ts.jsonl"
prompt="$(cat "$PROMPT_FILE")"

cd "$RUNDIR"
echo "[drive] model=$MODEL thinking=$THINKING dir=$RUNDIR -> $log" >&2
# Drop streaming token-delta events (message_update) before saving: pi re-emits the
# entire growing message on every token, so they balloon the log O(n^2). We keep the
# structural events (turn_*, tool_execution_*, message_start/end) which carry the
# committed tool calls and their file contents.
RRLM_DIR="$RRLM_DIR" pi --mode json -p --no-session \
  --model "$MODEL" --thinking "$THINKING" \
  -e "$EXT" --skill "$SKILL" \
  "$prompt" | grep --line-buffered -v '"type":"message_update"' | tee "$log"
