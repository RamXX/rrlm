# LadyCRM built with Ornith-1.0-35B: time, quality, progress

Built LadyCRM from scratch with **Ornith-1.0-35B** (a Qwen3.5 MoE, 256 experts with 8
active per token; served via llama-server, thinking off) plus the supergemma leaf,
driven by `pi` and the rrlm rlm-backend, one file at a time with a compile after each.
Harness: `build_loop.sh` (timed, quality-gated, trace-capturing).

## Time (measured)

| | passes | wall-clock |
|---|---:|---:|
| Phase 1 (store/contact/company/main/test) | 1 | 125 s |
| Full spec (deals, interactions, timeline, path, rlm report/import) | 1 | 512 s |
| Targeted fix pass (path bug) | 1 | 112 s |
| **Total** | **3** | **~749 s (12.5 min)** |

For comparison, an earlier build with a dense Qwen3.6-27B (pi-tune) ran for hours over
many passes. Ornith built the entire full spec in a single 512s pass; the MoE keeps
prefill (the bottleneck of this loop) cheap.

## Quality (measured)

- `make build` GREEN, `make test` PASS. 9 focused Go files (a clean `internal/store`
  package: per-entity structs and functions over `*lbug.Connection`; `internal/rlm`
  shells to `rrlm-solve`; `cmd/crm` dispatcher).
- **Every command works**: init, contact add/list, company add, link works-at,
  deal add/stage/list, interact, timeline, **path**, report, import.
- **report (the RLM showcase) works out of the box**, both structured ("which deal is in
  negotiation and for which company", correct) and semantic ("which contact seems at risk
  of churning", correctly flagged Ada from her interaction note).
- **Ornith self-fixed the `path` bug** (a Cypher error: every `WITH` term must be
  aliased) in one 112s pass. The earlier dense model needed a human to fix `path`. That
  is the headline progress signal: the stronger coder self-heals where the weaker one
  stalled.

### Blemishes (honest)
- `import` de-duplicates *within* the CSV but not against existing DB contacts: a spec
  interpretation gap, not a crash (it runs cleanly).
- The agent left a stray root scratch file `test_ddl_main.go` (a DDL probe), removed in
  cleanup. `main.go` is large (587 lines) versus the spec's "small files" preference.
- Ornith used a functional store API (`AddContact(conn, ...)`) instead of the spec's
  `(*Store)` method wrapper: a valid design choice, not a defect.

## Traces (for GEPA)

`rrlm-solve` exports each call's predict-rlm RunTrace to `RRLM_TRACE_DIR` (a unique file
per process plus an `index.jsonl` pairing instruction, answer, and config). It is set
during the build and the report/import runs.

- The CRM `report` and `import` runs produced RunTraces. They are orchestrator-only here:
  the CRM data is tiny (a couple of contacts, a 3-row CSV), so Ornith solved everything
  directly without fanning out to the leaf. They are still valid GEPA examples of the
  orchestrator's trajectories.
- The build itself made zero `rlm_solve` calls (it writes code, it does not query data),
  so there are no build-time traces, as expected.

The leaf (supergemma) is exercised heavily elsewhere (the superpowers imdb cells): in the
imdb-200 trace it made 200/200 clean leaf calls, 0 errors, 0 malformed outputs, with a
correct result, so the leaf was kept as-is.

## Reproduce
```bash
make serve-orch     # Ornith on llama.cpp --parallel
make serve-leaf     # supergemma leaf
make eval-crm       # builds LadyCRM into examples/crm/runs/build (timed, traced)
```
Override the orchestrator with `make eval-crm CRM_MODEL=<provider/model>`.
