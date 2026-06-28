#!/usr/bin/env bash
# Track C: measure sustained decode throughput of the pi-tune GGUF (Q6_K) under
# llama.cpp's llama-server, PLAIN vs MTP self-speculative decoding, on the Metal
# GPU. Runs ONE server at a time (start -> measure -> stop) per the project rule.
#
# For each spec setting we measure a SHORT (64-tok) and a LONG (512-tok)
# generation. The ratio long/short exposes the documented MTP failure mode:
# fast on short bursts, collapsing on long generations.
#
#   ./bench_mtp.sh                 # plain then MTP, writes results/mtp_q6.json
#
# Honors the project quant rule: Q6_K only (never Q4). Forces all layers onto the
# Metal GPU (NGL=999) since this llama.cpp build's auto-fit can leave it on CPU.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVE="$SCRIPT_DIR/../../scripts/local-serving/serve-pitune.sh"
OUT="$SCRIPT_DIR/results"; mkdir -p "$OUT"
HOST=127.0.0.1; PORT="${PORT:-8773}"
SHORT="${SHORT:-64}"; LONG="${LONG:-512}"
RESULT="$OUT/mtp_q6.json"

# A representative agentic-coding prompt that reliably elicits a long answer.
PROMPT="${PROMPT:-Write a complete, well-commented Go implementation of a generic in-memory LRU cache (Get, Put, eviction by capacity), then write table-driven unit tests for it. Explain each design decision step by step as you go.}"

_wait_health() {
  local deadline=$(( $(date +%s) + 240 ))
  while [ "$(date +%s)" -lt "$deadline" ]; do
    if curl -s "http://$HOST:$PORT/health" 2>/dev/null | grep -q '"status":"ok"'; then return 0; fi
    sleep 2
  done
  return 1
}

# POST /completion with n_predict=$1, print one TSV line:
#   n_predict  predicted_n  predicted_ms  predicted_per_second  prompt_per_second
_measure() {
  local n="$1" resp
  resp="$(curl -s "http://$HOST:$PORT/completion" \
    -H 'Content-Type: application/json' \
    -d "{\"prompt\": $(python3 -c 'import json,sys;print(json.dumps(sys.argv[1]))' "$PROMPT"),
         \"n_predict\": $n, \"cache_prompt\": false, \"temperature\": 0.7,
         \"top_p\": 0.8, \"top_k\": 20, \"min_p\": 0, \"presence_penalty\": 1.5}")"
  python3 - "$resp" <<'PY'
import json, sys
try:
    t = json.loads(sys.argv[1]).get("timings", {})
    print("\t".join(str(t.get(k, "NA")) for k in
        ("predicted_n","predicted_ms","predicted_per_second","prompt_per_second")))
except Exception as e:
    print(f"PARSE_ERROR\t{e}\tNA\tNA")
PY
}

declare -a ROWS
run_leg() {
  local label="$1" specmtp="$2" pid
  echo "[$label] starting llama-server (SPEC_MTP=$specmtp, NGL=999, NPAR=1) ..." >&2
  NGL=999 NPAR=1 SPEC_MTP="$specmtp" PORT="$PORT" "$SERVE" >"/tmp/llm-servers/mtp-$label.log" 2>&1 &
  pid=$!
  if ! _wait_health; then
    echo "[$label] server did not become healthy; see /tmp/llm-servers/mtp-$label.log" >&2
    kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null; return 1
  fi
  echo "[$label] healthy; warmup ..." >&2
  _measure 16 >/dev/null
  for n in "$SHORT" "$LONG"; do
    echo "[$label] measuring n_predict=$n ..." >&2
    local line; line="$(_measure "$n")"
    echo "[$label] n=$n -> $line" >&2
    ROWS+=("$label	$n	$line")
  done
  echo "[$label] stopping server (pid $pid) ..." >&2
  kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
  # wait for port to free
  for _ in $(seq 1 30); do curl -s "http://$HOST:$PORT/health" >/dev/null 2>&1 || break; sleep 1; done
  echo "[$label] cooldown 15s ..." >&2; sleep 15
}

run_leg plain 0
run_leg mtp   1

# Emit TSV (stdout) and JSON (file)
echo "label	n_predict	predicted_n	predicted_ms	predicted_per_second	prompt_per_second"
printf '%s\n' "${ROWS[@]}"

python3 - "$RESULT" "${ROWS[@]}" <<'PY'
import json, sys
out = sys.argv[1]; rows = sys.argv[2:]
recs = []
for r in rows:
    f = r.split("\t")
    recs.append(dict(label=f[0], n_predict=int(f[1]),
                     predicted_n=f[2], predicted_ms=f[3],
                     predicted_per_second=f[4], prompt_per_second=f[5]))
json.dump({"engine":"llama.cpp/llama-server","model":"bytkim/Qwen3.6-27B-MTP-pi-tune-GGUF:Q6_K",
           "gpu":"metal(ngl=999)","parallel":1,"records":recs}, open(out,"w"), indent=2)
print(f"\nwrote {out}", file=sys.stderr)
PY
