#!/usr/bin/env bash
# Track C end-to-end A/B/C: run the repo's RLM ledger eval against whichever
# orchestrator is currently serving, and APPEND to a shared comparison table. Run
# it once per backend (DFlash / mlx_lm / llama.cpp), starting the matching server
# first, so all backends are measured back-to-back under identical machine state.
#
#   LABEL=qwen-dflash MODEL=qwen-official/mlx-community/Qwen3.6-27B-8bit bash run_e2e_cell.sh
#   LABEL=qwen-mlxlm  MODEL=qwen-official/mlx-community/Qwen3.6-27B-8bit bash run_e2e_cell.sh
#   LABEL=pitune-mtp  MODEL=pitune/qwen3.6-27b-pi-tune                   bash run_e2e_cell.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
RUN="uv run -p $(cat .python-version) -- python -m rrlm.bench.runner"
MODEL="${MODEL:-qwen-official/mlx-community/Qwen3.6-27B-8bit}"
LABEL="${LABEL:-unknown}"
LEAF="supergemma/Jiunsong/supergemma4-26b-uncensored-mlx-4bit-v2"
SIZES="${SIZES:-2000 20000}"
SEED="${SEED:-42}"
OUT="experiments/dflash-vs-mtp/results/e2e_compare.tsv"
[ -f "$OUT" ] || printf "backend\ttask\tsize\tseed\tcondition\tpassed\tstatus\twall_s\tprompt_tok\tcompletion_tok\tdetail\trun_id\n" > "$OUT"

cell() { # size
  local size=$1
  echo "=== [$LABEL] ledger rlm size=$size ===" >&2
  local outp; outp=$(timeout 3600 $RUN --task ledger --condition rlm --size "$size" \
    --model "$MODEL" --sub-model "$LEAF" --reasoning off --seed "$SEED" 2>&1)
  local rid; rid=$(echo "$outp" | grep -oE '\[[0-9]{8}-[0-9]{6}_[^]]+\]' | head -1 | tr -d '[]')
  [ -z "$rid" ] && rid=$(ls -t runs/ 2>/dev/null | head -1)
  LABEL="$LABEL" SEED="$SEED" python3 - "$rid" "$size" "$OUT" <<'PY'
import json,os,sys
rid,size,out=sys.argv[1:4]; label=os.environ["LABEL"]; seed=os.environ.get("SEED","42")
try: r=json.load(open(f"runs/{rid}/result.json"))
except Exception as e: r={"status":f"noresult:{e}"}
u=r.get("usage",{})
row=[label,"ledger",size,seed,"rlm",str(r.get("passed")),r.get("status","?"),
     str(r.get("wall_clock_s","")),str(u.get("prompt_tokens","")),str(u.get("completion_tokens","")),
     str(r.get("check_detail",""))[:50].replace("\t"," ").replace("\n"," "),rid]
open(out,"a").write("\t".join(row)+"\n")
print("  ->","passed="+row[5],row[6],row[7]+"s",file=sys.stderr)
PY
}

for s in $SIZES; do cell "$s"; done
echo "[$LABEL] DONE" >&2
