#!/usr/bin/env bash
# Track C: fair llama.cpp leg. Measure sustained decode throughput of the pi-tune
# GGUF (Q6_K) under llama-server, on the SAME prompt the DFlash MLX benchmark used
# (the dflash "smoke" math prompt, chat-templated), so DFlash vs MTP is apples-to-
# apples. Uses the OPTIMAL llama.cpp config: flash attention on (-fa on), all
# layers on Metal (-ngl 999). Tests PLAIN vs MTP self-speculative decoding.
#
# For each config we measure SHORT (64-tok) and LONG (512-tok) generations with
# ignore_eos so the run always reaches the requested length. The long/short ratio
# exposes whether MTP collapses on long generations (a documented failure mode).
#
# Runs ONE server at a time (start -> measure -> stop). Q6_K only (never Q4).
#
#   ./bench_llamacpp.sh            # plain-fa then mtp-fa, writes results/llamacpp_q6.json
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$SCRIPT_DIR/results"; mkdir -p "$OUT"
HOST=127.0.0.1; PORT="${PORT:-8773}"
SHORT="${SHORT:-64}"; LONG="${LONG:-512}"
CTX="${CTX:-16384}"
RESULT="$OUT/llamacpp_q6.json"

GGUF="$(find "$HOME/.cache/huggingface/hub/models--bytkim--Qwen3.6-27B-MTP-pi-tune-GGUF/snapshots" \
  -name 'Qwen3.6-27B-MTP-pi-tune-Q6_K.gguf' 2>/dev/null | head -1)"
[ -z "$GGUF" ] && { echo "GGUF not found" >&2; exit 1; }

# Identical user prompt to the dflash smoke-default suite (so DFlash vs llama.cpp
# run the same workload through their respective chat templates).
PROMPT='The function $f$ satisfies the functional equation \[ f(x) + f(y) = f(x + y) - xy - 1 \] for all real numbers $x$ and $y$. If $f(1) = 1$, then find all integers $n$ such that $f(n) = n$. Enter all such integers, separated by commas. Please reason step by step, and put your final answer within \boxed{}.'

_wait_health() {
  local deadline=$(( $(date +%s) + 240 ))
  while [ "$(date +%s)" -lt "$deadline" ]; do
    curl -s "http://$HOST:$PORT/health" 2>/dev/null | grep -q '"status":"ok"' && return 0
    sleep 2
  done; return 1
}

# /v1/chat/completions with max_tokens=$1, ignore_eos. Print TSV:
#   predicted_n  predicted_per_second  source(timings|walltime)
_measure() {
  local n="$1" body resp wall
  body="$(python3 -c 'import json,sys;print(json.dumps({
    "messages":[{"role":"user","content":sys.argv[1]}],
    "max_tokens":int(sys.argv[2]),"ignore_eos":True,"stream":False,
    "temperature":0.7,"top_p":0.8,"top_k":20,"min_p":0,"presence_penalty":1.5,
    "chat_template_kwargs":{"enable_thinking":False}}))' "$PROMPT" "$n")"
  resp="$(curl -s -w '\n__WALL__%{time_total}' "http://$HOST:$PORT/v1/chat/completions" \
            -H 'Content-Type: application/json' -d "$body")"
  wall="${resp##*__WALL__}"; resp="${resp%__WALL__*}"
  python3 - "$resp" "$wall" "$n" <<'PY'
import json, sys
resp, wall, n = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
try:
    d = json.loads(resp)
except Exception as e:
    print(f"PARSE_ERROR\tNA\t{e}"); sys.exit()
t = d.get("timings") or {}
pn = t.get("predicted_n")
pps = t.get("predicted_per_second")
if pps is None:  # fallback: usage tokens / wall clock
    ct = (d.get("usage") or {}).get("completion_tokens")
    if ct and wall > 0:
        pn, pps = ct, ct / wall
        print(f"{pn}\t{pps:.3f}\twalltime"); sys.exit()
    print(f"NA\tNA\tno_timings"); sys.exit()
print(f"{pn}\t{pps:.3f}\ttimings")
PY
}

declare -a ROWS
run_cfg() {
  local label="$1"; shift
  local logf="/tmp/llm-servers/llamacpp-$label.log"
  echo "[$label] starting llama-server: $* ..." >&2
  llama-server --model "$GGUF" --host "$HOST" --port "$PORT" --ctx-size "$CTX" \
    -ngl 999 --parallel 1 --jinja -fa on \
    --temp 0.7 --top-p 0.8 --top-k 20 --min-p 0 --presence-penalty 1.5 \
    "$@" >"$logf" 2>&1 &
  local pid=$!
  if ! _wait_health; then echo "[$label] not healthy; see $logf" >&2; kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null; return 1; fi
  echo "[$label] healthy; warmup ..." >&2; _measure 16 >/dev/null
  for n in "$SHORT" "$LONG"; do
    local line; line="$(_measure "$n")"
    echo "[$label] n=$n -> $line" >&2
    ROWS+=("$label	$n	$line")
  done
  echo "[$label] stopping (pid $pid) ..." >&2; kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
  for _ in $(seq 1 30); do curl -s "http://$HOST:$PORT/health" >/dev/null 2>&1 || break; sleep 1; done
  echo "[$label] cooldown 15s ..." >&2; sleep 15
}

# Plain (no spec) and MTP self-speculative decoding, both with -fa on.
run_cfg plain_fa
run_cfg mtp_fa --spec-type draft-mtp --cache-type-k q8_0 --cache-type-v q8_0

echo "label	max_tokens	predicted_n	predicted_per_second	source"
printf '%s\n' "${ROWS[@]}"
python3 - "$RESULT" "${ROWS[@]}" <<'PY'
import json, sys
out, rows = sys.argv[1], sys.argv[2:]
recs=[]
for r in rows:
    f=r.split("\t")
    recs.append(dict(label=f[0],max_tokens=int(f[1]),predicted_n=f[2],
                     predicted_per_second=f[3],source=f[4] if len(f)>4 else "NA"))
json.dump({"engine":"llama.cpp/llama-server","model":"bytkim/Qwen3.6-27B-MTP-pi-tune-GGUF:Q6_K",
           "config":"fa=on ngl=999 parallel=1 ctx=%s"%__import__("os").environ.get("CTX","16384"),
           "prompt":"dflash smoke-default (math)","records":recs}, open(out,"w"), indent=2)
print("wrote",out,file=sys.stderr)
PY
