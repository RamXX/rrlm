#!/usr/bin/env bash
# Track C end-to-end: run the "winner" (official Qwen3.6-27B-8bit served via DFlash)
# through the repo's real RLM eval, on the same cells the experiment already ran with
# the mlx_lm and llama.cpp backends. Confirms accuracy holds and the speed win carries
# into the actual task (not just synthetic decode).
#
# Requires the DFlash orchestrator on the qwen-official port (8772) and the leaf (8771):
#   PORT=8772 ./scripts/local-serving/dflash-qwen36.sh serve   # w8 draft (never Q4)
#   ./scripts/local-serving/serve-models.sh start              # supergemma leaf, w8
set -uo pipefail
cd "$(dirname "$0")/../.."
RUN="uv run -p $(cat .python-version) -- python -m rrlm.bench.runner"
ORCH="qwen-official/mlx-community/Qwen3.6-27B-8bit"
LEAF="supergemma/Jiunsong/supergemma4-26b-uncensored-mlx-4bit-v2"
OUT="experiments/dflash-vs-mtp/results/winner_cells.tsv"
printf "task\tsize\tseed\tcondition\tbackend\tpassed\tstatus\twall_s\tprompt_tok\tcompletion_tok\tdetail\trun_id\n" > "$OUT"

cell() { # size
  local size=$1
  echo "=== ledger rlm size=$size via DFlash orchestrator (qwen-official @ :8772) ===" >&2
  local outp; outp=$(timeout 3600 $RUN --task ledger --condition rlm --size "$size" \
    --model "$ORCH" --sub-model "$LEAF" --reasoning off --seed 42 2>&1)
  local rid; rid=$(echo "$outp" | grep -oE '\[[0-9]{8}-[0-9]{6}_[^]]+\]' | head -1 | tr -d '[]')
  [ -z "$rid" ] && rid=$(ls -t runs/ 2>/dev/null | head -1)
  python3 - "$rid" "$size" "$OUT" <<'PY'
import json,sys
rid,size,out=sys.argv[1:4]
try: r=json.load(open(f"runs/{rid}/result.json"))
except Exception as e: r={"status":f"noresult:{e}"}
u=r.get("usage",{})
row=["ledger",size,"42","rlm","dflash",str(r.get("passed")),r.get("status","?"),
     str(r.get("wall_clock_s","")),str(u.get("prompt_tokens","")),str(u.get("completion_tokens","")),
     str(r.get("check_detail",""))[:50].replace("\t"," ").replace("\n"," "),rid]
open(out,"a").write("\t".join(row)+"\n")
print("  ->", "passed="+row[5], row[6], row[7]+"s", file=sys.stderr)
PY
}

cell 2000
cell 20000
echo "DONE" >&2
column -t -s "$(printf '\t')" "$OUT" >&2
