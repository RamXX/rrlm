#!/usr/bin/env bash
# Purge the DFlash prefix cache (it is regenerated automatically on next serve).
#
# DFlash writes a speculative-decoding prefix/KV cache under
#   ${XDG_CACHE_HOME:-$HOME/.cache}/dflash      (observed: .../dflash/prefix_l2)
# which grows to tens of GB across long serving sessions. The cache is pure
# derived state, deleting it costs only a cold-start re-warm, so we drop it
# whenever the local LLM servers are brought DOWN, to reclaim disk. It refills
# on its own as soon as the servers come back up and start handling prompts.
#
# SAFETY: refuses to purge while any `dflash serve` process is still alive. A
# live server owns the cache, and yanking it mid-flight risks corruption. Call
# this only after the servers are stopped (serve-models.sh stop does that).
#
#   ./purge-dflash-cache.sh          # purge iff no dflash server is running
#   FORCE=1 ./purge-dflash-cache.sh  # purge even if a server appears to run
set -euo pipefail

CACHE_DIR="${DFLASH_CACHE_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/dflash}"

if [ "${FORCE:-0}" != "1" ] && pgrep -f 'dflash serve' >/dev/null 2>&1; then
  echo "[dflash-cache] a 'dflash serve' process is still running; skipping purge."
  echo "[dflash-cache] stop the servers first, or re-run with FORCE=1 to override."
  exit 0
fi

if [ ! -d "$CACHE_DIR" ]; then
  echo "[dflash-cache] nothing to purge ($CACHE_DIR does not exist)."
  exit 0
fi

before="$(du -sh "$CACHE_DIR" 2>/dev/null | cut -f1 || true)"
# Remove the cache contents but keep the directory so dflash can repopulate it.
find "$CACHE_DIR" -mindepth 1 -maxdepth 1 -exec rm -rf {} + 2>/dev/null || true
echo "[dflash-cache] purged $CACHE_DIR (was ${before:-0})."
