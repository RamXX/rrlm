#!/usr/bin/env bash
# Generalized, TIMED, TRACE-CAPTURING LadyCRM build loop. Drives a local orchestrator
# model (via `pi` + the rrlm rlm-backend extension) to build LadyCRM file-by-file:
# Phase 1 (continue.md) then the full spec (continue2.md). Each pass re-reads the
# current code (--no-session), so it suits small-context local models.
#
# Measures per-pass + total wall-clock (.build_metrics.tsv) and quality gates
# (make build/test, rlm.go + report present, CLI smoke). Sets RRLM_TRACE_DIR so the
# agent's data-heavy rlm_solve calls accumulate predict-rlm RunTraces for later GEPA.
#
#   bash build_loop.sh <model-ref> <run-dir-rel>
#   bash build_loop.sh ornith/ornith-1.0-35b runs/ornith
set -uo pipefail
CRM="$(cd "$(dirname "$0")" && pwd)"
MODEL="${1:?model ref}"; RUNREL="${2:?run dir (relative to repo)}"; RUN="$CRM/$RUNREL"
cd "$CRM"
export RRLM_DIR="${RRLM_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}" THINKING=off
export RRLM_TRACE_DIR="$RUN/traces"
mkdir -p "$RRLM_TRACE_DIR"
METRICS="$RUN/.build_metrics.tsv"
[ -f "$METRICS" ] || printf "phase\tpass\tstart_epoch\tdur_s\tgo_files\tbuild\ttest\trlm\treport\n" > "$METRICS"

go_files() { find "$RUN" -name '*.go' -not -path '*/.drive/*' | wc -l | tr -d ' '; }
bstat() { ( cd "$RUN" && make build >/dev/null 2>&1 ) && echo GREEN || echo BROKEN; }
tstat() { ( cd "$RUN" && make test >/dev/null 2>&1 ) && echo PASS || echo FAIL; }
has_rlm() { [ -f "$RUN/internal/rlm/rlm.go" ] && echo yes || echo no; }
has_report() { grep -q '"report"' "$RUN/cmd/crm/main.go" 2>/dev/null && echo yes || echo no; }
phase1_done() { [ -f "$RUN/internal/store/store_test.go" ] && [ "$(bstat)" = GREEN ] && [ "$(tstat)" = PASS ]; }
full_done() { [ "$(has_rlm)" = yes ] && [ "$(has_report)" = yes ] && [ "$(bstat)" = GREEN ] && [ "$(tstat)" = PASS ]; }

run_pass() { # phase prompt pass-num
  local phase="$1" prompt="$2" n="$3" t0 dur b t
  t0=$(date +%s)
  echo "[$phase] pass $n start $(date +%H:%M:%S) | go-files=$(go_files)" >&2
  bash drive.sh "$MODEL" "$RUNREL" "prompts/$prompt" > "$RUN/.pass_${phase}_$n.log" 2>&1 || true
  dur=$(( $(date +%s) - t0 )); b=$(bstat); t=$(tstat)
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$phase" "$n" "$t0" "$dur" "$(go_files)" "$b" "$t" "$(has_rlm)" "$(has_report)" >> "$METRICS"
  echo "[$phase] pass $n done | dur=${dur}s files=$(go_files) build=$b test=$t rlm=$(has_rlm) report=$(has_report)" >&2
}

BUILD_T0=$(date +%s)
for n in $(seq 1 "${PHASE1_PASSES:-10}"); do
  phase1_done && { echo "[phase1] done" >&2; break; }
  run_pass phase1 continue.md "$n"
  phase1_done && { echo "[phase1] ACCEPTANCE after pass $n" >&2; break; }
done
for n in $(seq 1 "${FULL_PASSES:-18}"); do
  full_done && { echo "[full] done" >&2; break; }
  run_pass full continue2.md "$n"
  full_done && { echo "[full] FULL BUILD COMPLETE after pass $n" >&2; break; }
done
TOTAL=$(( $(date +%s) - BUILD_T0 ))

{
  echo "=== Ornith LadyCRM build result $(date -u +%FT%TZ) ==="
  echo "model: $MODEL | total_build_s: $TOTAL ($(awk "BEGIN{printf \"%.1f\", $TOTAL/60}") min)"
  cd "$RUN"
  echo "go files ($(go_files)):"; find . -name '*.go' -not -path '*/.drive/*' | sed 's#^./##' | sort
  echo "build: $(bstat) | test: $(tstat) | rlm.go: $(has_rlm) | report cmd: $(has_report)"
  echo "--- make test ---"; make test 2>&1 | tail -3
  echo "--- CLI smoke ---"; rm -f /tmp/ladycrm_ornith.db*
  ./bin/crm --db /tmp/ladycrm_ornith.db init 2>&1 | tail -1
  cid=$(./bin/crm --db /tmp/ladycrm_ornith.db contact add --name Ada --email ada@x.io 2>&1 | tail -1); echo "contact add -> $cid"
  ./bin/crm --db /tmp/ladycrm_ornith.db contact list 2>&1 | tail -3
  echo "predict-rlm traces captured: $(ls "$RRLM_TRACE_DIR"/trace-*.json 2>/dev/null | wc -l | tr -d ' ')"
} > "$RUN/.build_result.txt" 2>&1
date -u +%FT%TZ > "$RUN/.build_done"
echo "[build] COMPLETE in ${TOTAL}s -> $RUN/.build_result.txt" >&2
