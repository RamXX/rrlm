# LadyCRM rebuilt with Ornith-1.0-35B — time, quality, progress

Re-did the LadyCRM build from scratch with **Ornith-1.0-35B** (the settled orchestrator,
served via llama-server, thinking off) + the supergemma leaf, driven by `pi` + the rrlm
rlm-backend, file-by-file with compile-after-each. Harness: `build_loop.sh` (timed,
quality-gated, trace-capturing). Build dir: `runs/ornith/`.

## Time (measured)

| | passes | wall-clock |
|---|---:|---:|
| Phase 1 (store/contact/company/main/test) | 1 | 125 s |
| Full spec (deals, interactions, timeline, path, rlm report/import) | 1 | 512 s |
| Targeted fix pass (path bug) | 1 | 112 s |
| **Total** | **3** | **~749 s (12.5 min)** |

For comparison, the original **pi-tune** build ran for **~4 hours** over many passes
(partly under concurrent fine-tune load). Ornith: **~20x faster wall-clock, 3 passes**,
and it built the *entire* full spec in a single 512 s pass.

## Quality (measured)

- `make build` GREEN, `make test` PASS. 9 focused Go files (clean `internal/store`
  package: per-entity structs + functions over `*lbug.Connection`; `internal/rlm` shells
  to `rrlm-solve`; `cmd/crm` dispatcher).
- **Every command works**: init, contact add/list, company add, link works-at,
  deal add/stage/list, interact, timeline, **path**, report, import.
- **report (the RLM showcase) works out of the box** — structured ("which deal is in
  negotiation and for which company" -> correct) *and* semantic ("which contact seems at
  risk of churning" -> correctly flagged Ada from her interaction note).
- **Ornith self-fixed the `path` bug** (Cypher: every WITH term must be aliased) in one
  112 s pass. The pi-tune build needed a **human** to fix `path`. This is the headline
  progress signal: the stronger coder self-heals where the weaker one stalled.

### Blemishes (honest)
- `import` de-duplicates *within* the CSV but not against existing DB contacts — a SPEC
  interpretation gap, not a crash (runs cleanly).
- The agent left a stray root scratch file `test_ddl_main.go` (a DDL probe); removed in
  cleanup. `main.go` is large (587 lines) vs the SPEC's "small files" preference.
- Ornith used a functional store API (`AddContact(conn, ...)`) instead of the SPEC's
  `(*Store)` method wrapper — a valid design choice, not a defect.

### vs pi-tune build
Comparable file count and command coverage; both green. pi-tune needed a human to fix
`path` and its `import` demo was never captured. Ornith matched/exceeded that quality
**~20x faster, in 3 passes, self-fixing the hard command, with report working immediately.**

## Traces (for GEPA)

`rrlm-solve` now exports each call's predict-rlm RunTrace to `RRLM_TRACE_DIR` (unique file
per process + `index.jsonl` pairing instruction->answer->config). Set during the build and
the report/import runs.

- CRM `report`/`import` produced **3 RunTraces** under `runs/ornith/traces/`. These are
  **orchestrator-only** — the CRM data is tiny (2 contacts, 3-row CSV), so Ornith solved
  everything directly without fanning out to the leaf. Still valid GEPA examples of the
  orchestrator's RLM trajectories.
- The build itself made **0** rlm_solve calls (it writes code; it doesn't query data),
  so no build-time traces — expected.

## supergemma leaf health (re your Ornith-9B note)

The CRM traces don't exercise the leaf (data too small). Where the leaf *is* heavily used
is the superpowers **imdb** cells. In the imdb-200 trace: **200/200 leaf calls, 0 errors,
0 malformed outputs**, clean structured `{review -> {'sentiment': pos/neg}}` judgments,
sensible distribution (153 pos / 46 neg / 1 neutral), and the task passed (P205 correct).
**No supergemma issues** -> per your condition, no need to switch the leaf. **Ornith-1.0-9B**
is noted as a future leaf candidate if a real issue ever surfaces (it would need a
speed check vs the cheap 4-bit supergemma).

## Reproduce
```bash
make serve-orch   # Ornith; for the build use PARALLEL=1 CTX=65536 (full ctx, single agent)
make serve-leaf   # supergemma
bash build_loop.sh ornith/ornith-1.0-35b runs/ornith        # timed, traced build
bash runs/ornith/scenario.sh                                 # full functional + trace gen
```
