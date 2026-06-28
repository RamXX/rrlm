# Does RLM give a small local model superpowers?

A controlled experiment. **Hold the model fixed; vary only the technique.** Show
that a small model running entirely on a laptop can do tasks with RLM that it
provably cannot do without it.

- **baseline** = the model alone (task + data stuffed into its prompt).
- **rlm** = the same model + the RLM technique (data lives in a sandboxed REPL;
  the model acts only by writing Python to probe it; a cheap leaf model is fanned
  out for irreducible semantic judgment).

Every task has machine-checkable ground truth (the harness computes the true
answer and compares). All inference is local, so cost is $0 and the metrics are
accuracy + wall-clock + tokens.

## What's here

| File | What |
|---|---|
| `PROOF.md` | **the writeup**: tables, analysis, evidence, scoreboard |
| `environment.txt` | exact hardware, model builds, library versions, context size |
| `run_experiment.sh` | the **resumable** driver -- runs every (task x condition x size x seed x model) cell; skips any already in results.tsv |
| `results.tsv` | one row per cell: passed/status/wall/tokens/answer-vs-truth/run_id |
| `summarize.py` | aggregates results.tsv + run artifacts into `PROOF_tables.md` and pulls the actual code the model wrote |
| `PROOF_tables.md` | generated per-task tables + extracted code/errors |
| `evidence/<run_id>/` | preserved per-cell artifacts: `run.json` (config + ground truth), `result.json` (answer + pass/fail), `trace.json` (the RLM's REPL turns) |

## Reproduce from zero

1. Install rrlm (`uv sync` in the repo) and Deno (the RLM sandbox). See the repo
   root README.
2. Serve the local models (see `environment.txt` for exact builds):

   ```bash
   # leaf (DFlash :8771) -- the cheap fan-out model
   ./scripts/local-serving/serve-models.sh start
   # orchestrator: Ornith-1.0-35B Q6_K (llama-server :8774). Single 65536 slot for
   # the proof's context wall (the experiment runs cells sequentially):
   PARALLEL=1 CTX=65536 NOTHINK=1 ./scripts/local-serving/serve-ornith.sh
   ```

   These appear in Pi's `~/.pi/agent/models.json` as providers `ornith` and
   `supergemma` (rrlm resolves models from Pi config). The served context window is
   **65536 tokens** -- the wall the experiment pushes data past. (For production /
   parallel agents, drop `PARALLEL=1` to use continuous batching; see
   docs/LOCAL_SERVING.md.)

3. Run the experiment (resumable -- appends to `results.tsv` as each cell finishes;
   re-run to resume after any interruption, completed cells are skipped). Launch it
   detached so it survives background-task reaping:

   ```bash
   nohup bash experiments/superpowers/run_experiment.sh \
     > experiments/superpowers/driver.log 2>&1 &
   ```

   It auto-runs `summarize.py` and writes `PIPELINE_DONE` when finished. To
   summarize manually at any point:

   ```bash
   python experiments/superpowers/summarize.py > experiments/superpowers/PROOF_tables.md
   ```

## Run a single cell by hand

```bash
# the model alone (baseline) -- fails to ingest large data
python -m rrlm.bench.runner --task ledger --condition baseline --size 20000 \
  --model ornith/ornith-1.0-35b --reasoning off

# the same model + RLM -- solves it
python -m rrlm.bench.runner --task ledger --condition rlm --size 20000 \
  --model ornith/ornith-1.0-35b \
  --sub-model supergemma/Jiunsong/supergemma4-26b-uncensored-mlx-4bit-v2 --reasoning off
```

Each writes `runs/<run_id>/` with `run.json`, `result.json`, `events.jsonl`, and
(rlm) `trace.json`.
