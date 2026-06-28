#!/usr/bin/env bash
# Run BOTH local models as OpenAI-compatible DFlash servers on consecutive ports.
#
#   ./serve-models.sh start                      # launch both, thinking ON (default)
#   ./serve-models.sh start --no-think heretic   # heretic thinking OFF
#   ./serve-models.sh start --no-think supergemma
#   ./serve-models.sh start --no-think both
#   ./serve-models.sh stop     # stop both
#   ./serve-models.sh status   # show health + last log lines
#   ./serve-models.sh logs     # tail both logs
#
# --no-think sets the server-side chat-template default enable_thinking=false
# for the chosen model(s). Clients can still override per request via
# chat_template_kwargs. Thinking is ON unless --no-think names the model.
#
# Both use DFlash speculative decoding (lossless: output == target model).
# Ports are uncommon + consecutive to avoid clashes with LM Studio (1234),
# Ollama (11434), and the usual 8000/8080.
set -euo pipefail

# dflash resolved from PATH; override with $DFLASH if installed elsewhere.
DFLASH="${DFLASH:-dflash}"
# Draft quantization for DFlash speculative decoding. w8 (8-bit) by default --
# never Q4 (project quant rule). DFlash is lossless regardless of draft precision
# (the target re-validates every drafted token); this only affects draft speed.
# DFlash MLX kernels support 4-bit or 8-bit only (no 6-bit). Override with $DRAFT_QUANT.
DRAFT_QUANT="${DRAFT_QUANT:-w8:gs64}"
RUN="${RUN:-/tmp/llm-servers}"
mkdir -p "$RUN"

# Directory of this script, used to find the cache-purge helper.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# name | port | target (HF repo id or local path) | DFlash draft. All env-overridable.
# heretic is an OPTIONAL legacy orchestrator (a local uncensored build); leave
# HERETIC_MODEL empty to skip it. The supergemma leaf is the settled config and
# downloads from a public HF repo, so it works out of the box.
HERETIC_MODEL="${HERETIC_MODEL:-}"
HERETIC_DRAFT="${HERETIC_DRAFT:-z-lab/Qwen3.6-27B-DFlash}"
HERETIC_PORT="${HERETIC_PORT:-8770}"

SG_MODEL="${SG_MODEL:-Jiunsong/supergemma4-26b-uncensored-mlx-4bit-v2}"
SG_DRAFT="${SG_DRAFT:-z-lab/gemma-4-26B-A4B-it-DFlash}"
SG_PORT="${SG_PORT:-8771}"

HOST=127.0.0.1

_start_one() {
  local name="$1" port="$2" model="$3" draft="$4" nothink="$5"
  local pidf="$RUN/$name.pid" log="$RUN/$name.log"
  if [ -f "$pidf" ] && kill -0 "$(cat "$pidf")" 2>/dev/null; then
    echo "[$name] already running (pid $(cat "$pidf"), port $port)"; return
  fi
  local extra=() think="ON"
  if [ "$nothink" = "1" ]; then
    extra=(--chat-template-args '{"enable_thinking": false}'); think="OFF"
  fi
  echo "[$name] starting on $HOST:$port (thinking $think) ..."
  nohup "$DFLASH" serve --model "$model" --draft "$draft" \
    --draft-quant "$DRAFT_QUANT" --host "$HOST" --port "$port" "${extra[@]}" > "$log" 2>&1 &
  echo $! > "$pidf"
  echo "[$name] pid $(cat "$pidf") | log $log"
}

_stop_one() {
  local name="$1"; local pidf="$RUN/$name.pid"
  if [ -f "$pidf" ]; then
    kill "$(cat "$pidf")" 2>/dev/null && echo "[$name] stopped (pid $(cat "$pidf"))" || echo "[$name] not running"
    rm -f "$pidf"
  else
    echo "[$name] no pidfile"
  fi
}

_status_one() {
  local name="$1" port="$2"
  local code; code=$(curl -s -o /dev/null -w "%{http_code}" "http://$HOST:$port/v1/models" 2>/dev/null || echo 000)
  printf "[%s] port %s -> HTTP %s\n" "$name" "$port" "$code"
}

cmd="${1:-status}"; shift || true

# Parse optional flags (currently: --no-think heretic|supergemma|both)
NOTHINK_HERETIC=0; NOTHINK_SG=0
while [ $# -gt 0 ]; do
  case "$1" in
    --no-think)
      case "${2:-}" in
        heretic)    NOTHINK_HERETIC=1 ;;
        supergemma) NOTHINK_SG=1 ;;
        both)       NOTHINK_HERETIC=1; NOTHINK_SG=1 ;;
        *) echo "--no-think expects: heretic | supergemma | both" >&2; exit 2 ;;
      esac
      shift 2 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

case "$cmd" in
  start)
    if [ -n "$HERETIC_MODEL" ]; then
      _start_one heretic "$HERETIC_PORT" "$HERETIC_MODEL" "$HERETIC_DRAFT" "$NOTHINK_HERETIC"
      echo "  heretic:    http://$HOST:$HERETIC_PORT/v1"
    else
      echo "[heretic] skipped (set HERETIC_MODEL to enable the optional orchestrator)"
    fi
    _start_one supergemma "$SG_PORT" "$SG_MODEL" "$SG_DRAFT" "$NOTHINK_SG"
    echo "Models load lazily; check './serve-models.sh status' in ~30-60s."
    echo "  supergemma: http://$HOST:$SG_PORT/v1"
    ;;
  stop)
    _stop_one heretic; _stop_one supergemma
    # Servers are down, reclaim the (regenerable) DFlash prefix cache.
    "$SCRIPT_DIR/purge-dflash-cache.sh" || true ;;
  restart)
    # Stop in-place WITHOUT purging: a restart keeps the prefix cache on disk so
    # the freshly-started servers can reuse it instead of re-warming cold.
    _stop_one heretic; _stop_one supergemma; sleep 2
    rt=()
    if   [ "$NOTHINK_HERETIC" = 1 ] && [ "$NOTHINK_SG" = 1 ]; then rt=(--no-think both)
    elif [ "$NOTHINK_HERETIC" = 1 ]; then rt=(--no-think heretic)
    elif [ "$NOTHINK_SG" = 1 ];      then rt=(--no-think supergemma); fi
    "$0" start "${rt[@]}" ;;
  status)
    _status_one heretic "$HERETIC_PORT"; _status_one supergemma "$SG_PORT" ;;
  logs)
    tail -n 20 "$RUN/heretic.log" "$RUN/supergemma.log" 2>/dev/null ;;
  *)
    echo "usage: $0 {start|stop|restart|status|logs}" >&2; exit 2 ;;
esac
