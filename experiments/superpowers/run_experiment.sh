#!/usr/bin/env bash
# Single RESUMABLE driver for the whole RLM-superpower experiment. Each cell is
# skipped if its result is already in results.tsv, so a kill/reap just means
# relaunch, completed cells return instantly. Launch DETACHED (nohup &) so the
# driver itself survives background-task reaping (model servers + this driver).
#
#   nohup bash experiments/superpowers/run_experiment.sh > experiments/superpowers/driver.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/../.."   # rrlm root
RUN="uv run -p $(cat .python-version) -- python -m rrlm.bench.runner"
# Orchestrator: Ornith-1.0-35B (Qwen3.5 MoE), the settled top model (Track C).
# Both the old PITUNE and QWEN slots now point here so the whole matrix runs on Ornith.
ORCH="ornith/ornith-1.0-35b"
PITUNE="$ORCH"
QWEN="$ORCH"
LEAF="supergemma/Jiunsong/supergemma4-26b-uncensored-mlx-4bit-v2"
EXP="experiments/superpowers"
OUT="$EXP/results.tsv"
[ -f "$OUT" ] || printf "task\tmodel\tsize\tseed\tcondition\tpassed\tstatus\twall_s\tprompt_tok\tcompletion_tok\tcheck_detail\trun_id\n" > "$OUT"

# already recorded? match (task, provider, size, seed, recorded-condition)
done_cell() {
  awk -F'\t' -v t="$1" -v m="$2" -v s="$3" -v sd="$4" -v c="$5" \
    '$1==t && $2==m && $3==s && $4==sd && $5==c {found=1} END{exit !found}' "$OUT"
}

cell() { # task condition size model seed [sub] [rec_cond]
  local task=$1 cond=$2 size=$3 model=$4 seed=$5 sub=${6:-} rec=${7:-}
  local prov=${model%%/*}
  [ -z "$rec" ] && rec=$cond
  if done_cell "$task" "$prov" "$size" "$seed" "$rec"; then
    echo "skip (done): $task $prov $size $seed $rec" >&2; return
  fi
  echo "=== $task | $rec | size=$size | $prov | seed=$seed ===" >&2
  local args=(--task "$task" --condition "$cond" --size "$size" --model "$model" --seed "$seed" --reasoning off)
  [ -n "$sub" ] && args+=(--sub-model "$sub")
  local outp; outp=$(timeout 3600 $RUN "${args[@]}" 2>&1)
  local rid; rid=$(echo "$outp" | grep -oE '\[[0-9]{8}-[0-9]{6}_[^]]+\]' | head -1 | tr -d '[]')
  [ -z "$rid" ] && rid=$(ls -t runs/ 2>/dev/null | head -1)
  python3 - "$rid" "$task" "$prov" "$size" "$seed" "$rec" "$OUT" <<'PY'
import json,sys
rid,task,prov,size,seed,rec,out=sys.argv[1:8]
try: r=json.load(open(f"runs/{rid}/result.json"))
except Exception: r={}
u=r.get("usage",{})
row=[task,prov,size,seed,rec,str(r.get("passed")),r.get("status","?"),
     str(r.get("wall_clock_s","")),str(u.get("prompt_tokens","")),str(u.get("completion_tokens","")),
     str(r.get("check_detail",""))[:60].replace("\t"," ").replace("\n"," "),rid]
open(out,"a").write("\t".join(row)+"\n")
print(f"  -> passed={r.get('passed')} status={r.get('status')} detail={str(r.get('check_detail',''))[:50]}",file=sys.stderr)
PY
}

# ---- Phase 1: ledger across the context wall (capability + accuracy), pi-tune
for size in 500 2000 5000 20000; do
  cell ledger baseline $size "$PITUNE" 42
  cell ledger rlm      $size "$PITUNE" 42 "$LEAF"
done
# ---- Phase 2: needle (retrieval) + bugfind (code), pi-tune
cell needle  baseline 2000 "$PITUNE" 42
cell needle  rlm      2000 "$PITUNE" 42 "$LEAF"
cell bugfind baseline 60   "$PITUNE" 42
cell bugfind rlm      60   "$PITUNE" 42 "$LEAF"
# ---- Phase 3: imdb semantic, small (fits), baseline ok; rlm via cheap leaf
cell imdb baseline 200 "$PITUNE" 42
cell imdb rlm      200 "$PITUNE" 42 "$LEAF"
# ---- Phase 4: cross-model robustness (official Qwen3.6-27B)
for size in 2000 20000; do
  cell ledger baseline $size "$QWEN" 42
  cell ledger rlm      $size "$QWEN" 42 "$LEAF"
done
# ---- Phase 5: second seed on key cells, pi-tune
for size in 2000 5000; do
  cell ledger baseline $size "$PITUNE" 43
  cell ledger rlm      $size "$PITUNE" 43 "$LEAF"
done
# ---- Phase 6: ACCURACY SWEEP, fitting sizes, 3 seeds (rebut "just context size")
for seed in 42 43 44; do
  for size in 100 300 600 1000 1300; do
    cell ledger baseline $size "$PITUNE" $seed
    cell ledger rlm      $size "$PITUNE" $seed "$LEAF"
  done
done
# ---- Phase 7: leaf ablation, imdb with orchestrator as its own leaf (rlm-self)
cell imdb rlm 200 "$PITUNE" 42 "" rlm-self
# ---- Phase 8: imdb CAPABILITY gap, reviews overflow the window
cell imdb baseline 1500 "$PITUNE" 42
cell imdb rlm      1500 "$PITUNE" 42 "$LEAF"

echo "ALL CELLS DONE; summarizing" >&2
uv run -p "$(cat .python-version)" -- python "$EXP/summarize.py" > "$EXP/PROOF_tables.md" 2>"$EXP/summarize.err" || true
date -u +%Y-%m-%dT%H:%M:%SZ > "$EXP/PIPELINE_DONE"
echo "PIPELINE DONE" >&2
