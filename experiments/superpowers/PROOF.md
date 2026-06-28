# Does the RLM technique give a small local model superpowers?

> **Re-run on Ornith-1.0-35B (2026-06-27).** The proof is now generated with Ornith
> (Qwen3.5 35B MoE) as the sole orchestrator + the supergemma leaf. Result holds and
> is faster: **RLM passed 26 / 26 (100%); baseline (model alone) passed 5 / 25 (20%)**
> across **51 cells** -- the 5 baseline passes are again the small/easy cases. Current
> authoritative data: [`results.tsv`](results.tsv) and [`PROOF_tables.md`](PROOF_tables.md)
> (both regenerated on Ornith). The narrative below was written for the original
> dense-model run (pi-tune / official Qwen3.6-27B); the capability cliff and every
> conclusion reproduce on Ornith, but model-specific anecdotes (e.g. the official-Qwen
> GPU crash, exact wall-clock numbers) refer to that original run.

**Yes -- demonstrated, not asserted.** A small model running entirely on a laptop,
with the Recursive Language Model (RLM) technique, performs tasks it **provably
cannot do on its own**: exact computation over data far larger than its context
window, and reliable computation where reading-and-reasoning silently fails. No
bigger model, no cloud, $0 per run.

This is a controlled experiment: **hold the model fixed, vary only the technique.**

- **baseline** -- the model alone: task + data in its prompt.
- **rlm** -- the *same* model + RLM: data lives in a sandboxed Python REPL; the
  model acts only by writing code to probe it; a cheap "leaf" model is fanned out
  for irreducible semantic judgment.

Every task has machine-checkable ground truth (the harness computes the true answer
and compares). All inference is local, so cost is **$0** and the metrics are
accuracy, wall-clock, and tokens. **55 cells** across 4 task types, data sizes
spanning the context wall, 3 seeds, and 2 model builds.

> **Bottom line (Ornith re-run): baseline (model alone) passed 5 / 25 (20%). RLM
> (same model + technique) passed 26 / 26 (100%).** The baseline passes are exactly
> the small, fitting, easy cases; everything large or exact, the model alone got wrong
> or could not attempt -- and RLM solved all of it. (Original dense-model run: baseline
> 4 / 27, RLM 28 / 28 -- same pattern; the cell count differs only because the former
> cross-model phase is now redundant under one model.)

Raw data: [`results.tsv`](results.tsv). Per-cell tables + extracted code:
[`PROOF_tables.md`](PROOF_tables.md). Environment:
[`environment.txt`](environment.txt). Reproduction: [`README.md`](README.md).
Per-run artifacts (config, ground truth, the model's REPL turns): `evidence/`.

> **The headline.** On a model with a 65,536-token window, RLM computed the exact
> filtered sum over a **1.4 MB / ~900K-token** ledger -- correct to the cent
> (`115,797.87`) -- in 200s, making **zero** sub-LM calls (4 turns of plain Python).
> The same model *alone* cannot ingest the data at all (hard context-overflow error
> in 0.9s). On the official Qwen build with a 4x-larger 262K window, the model alone
> *crashed the GPU server* trying. The model did not get smarter. The technique gave
> it a capability and a reliability it does not otherwise possess.

---

## 1. The capability cliff (ledger, pi-tune, 65K window)

Task: "total of status=ok transactions for user X" -- a filter-then-sum over N rows
(~46 tokens/row, so the 65K window fills at ~1,400 rows).

| rows | data | **baseline (model alone)** | **rlm (model + technique)** |
|---|---|---|---|
| 500 | ~23K tok (fits) | **WRONG** 3501.97 vs 2743.79, 237s | **exact** 2743.79, 159s |
| 2,000 | ~92K tok | **IMPOSSIBLE** (context error), 0.7s | **exact** 10143.63, 208s |
| 5,000 | ~230K tok | **IMPOSSIBLE**, 0.7s | **exact** 26453.9, 105s |
| 20,000 | ~900K tok | **IMPOSSIBLE**, 0.9s | **exact** 115797.87, 200s |

Verbatim failure at 2,000 rows: `litellm.BadRequestError: request (91,764 tokens)
exceeds the available context size (65,536 tokens)`.

RLM wall-clock is **flat in data size** (105-208s from 500 to 20,000 rows); baseline
is only "fast" because it fails instantly. Replicated at seed 43 (RLM exact at 2,000
and 5,000; baseline overflow).

## 2. It is NOT "just context size" (accuracy sweep, fitting sizes, 3 seeds)

The obvious objection: "use a bigger window." Rebuttal -- across ledger sizes that
**all fit** the window, the model alone is unreliable while RLM is exact. Baseline
correctness (passes / 3 seeds):

| rows (all fit) | baseline correct | rlm correct |
|---|---|---|
| 100 | **1 / 3** (off by 10, off by 195 on the misses) | 3 / 3 |
| 300 | **0 / 3** | 3 / 3 |
| 600 | **0 / 3** | 3 / 3 |
| 1,000 | **0 / 3** | 3 / 3 |
| 1,300 | **0 / 3** | 3 / 3 |

The model alone sums tens of numbers reliably and is wrong by a few hundred rows --
**well before any context limit.** RLM: **15/15 exact.** So the gap is not about
window size; it is that *the model's in-context arithmetic is unreliable and the
technique replaces it with exact code.* (And on the 262K-window Qwen build, baseline
still miscounts a 2,000-row ledger -- Section 4 -- the accuracy gap survives 4x more
context.)

## 3. Mechanism -- two distinct superpowers

Measured sub-LM calls per run prove the mechanism:

| run | sub-LM calls | mechanism |
|---|---|---|
| ledger-20000 rlm | **0** (4 code turns) | **code execution** over data the model never reads |
| imdb-200 rlm (cheap leaf) | **201** | **cheap fan-out** -- one semantic judgment per review |
| imdb-200 rlm (self leaf) | 256, but 1111s vs 364s | leaf = **efficiency**, not correctness |

The actual code the model wrote for the 900K-token ledger (then *verified* with a
second independent parse before submitting -- the doctrine's verify step):

```python
u754_ok = []
for line in data.splitlines():
    if 'user=u754' not in line or 'status=ok' not in line:
        continue
    amount = re.search(r'amount=(\S+)', line).group(1)
    u754_ok.append(float(amount))
total = round(sum(u754_ok), 2)          # then a second, different parse re-checks it
```

**Leaf ablation** (imdb-200, all *correct*): baseline 249s -- RLM+cheap-leaf 364s --
RLM+self-leaf 1111s. The cheap supergemma leaf does not change correctness (the
orchestrator fanning out to itself is also exact); it makes the fan-out **~3x
faster** by delegating the 200 classifications to a small fast model.

## 4. It generalizes -- across models and task types

**Cross-model (official Qwen3.6-27B, mlx, native 262K window):**

| size | baseline | rlm |
|---|---|---|
| 2,000 | **WRONG** 10666.61 vs 10143.63, **1237s** (fits its 262K window, still miscounts) | **exact** 10143.63, 114s |
| 20,000 | **CRASHED the GPU server** (`[METAL] command buffer execution failed`) after 31 min on the 900K-token prompt | **exact** 115797.87, 206s |

So on Qwen, RLM is exact *and* ~11x faster than the model's own (wrong) reading; and
the model-alone path on 900K tokens is not merely impossible, it is catastrophic.

**Semantic data (imdb -- real movie reviews, "which product has the highest negative
fraction"):**

| reviews | baseline | rlm |
|---|---|---|
| 200 (fits) | correct, 249s | correct, 364s (201 leaf calls) |
| 1,500 (overflow) | **IMPOSSIBLE** (context error) | **exact** P205, 1403s |

The capability gap returns on natural language too: at 1,500 reviews the model alone
cannot ingest them; RLM works over all of them.

## 5. The boundary -- where the model alone is fine (or better)

RLM is a *targeted* superpower, not a universal upgrade. On small inputs that fit and
are easy, the model alone is correct and **faster**, because RLM pays a fixed
scaffolding tax:

| task | size | baseline | rlm |
|---|---|---|---|
| bugfind | 60 fns (14 KB) | correct, **22s** | correct, 719s |
| needle | 2,000 lines (retrieval) | correct, 330s | correct, 207s |
| imdb | 200 reviews | correct, 249s | correct, 364s |

That RLM does not win everywhere is what makes its decisive wins credible. The
routing rule: **use RLM when data is large or exactness over many items is required;
read directly when the input is small and the answer is a direct read.**

## 6. What is proven

For a fixed small local model, the RLM technique converts tasks from **impossible**
(data exceeds the window -- or crashes the server) or **unreliable** (miscounts what
it reads, even within the window) into **exact and cheap**, by moving the data out of
the model's head and into a REPL it drives with code, fanning a cheap leaf out only
for irreducible semantic judgment. That is a capability the model does not possess on
its own -- a superpower granted by the technique, on commodity hardware, at $0.

## Caveats (full disclosure)

- RLM's correctness depends on the orchestrator writing *correct* code. Across this
  experiment **every RLM cell passed** -- 100% (the lone transient 0.27s connection
  error on qwen-20000, caused by the baseline crashing the GPU server, succeeded on
  re-run with the server healthy: exact, 206s).
- "Small" here is a 27B model; the capability wall moves with the window, but the
  accuracy gap (Section 2) is independent of window size.
- Local decode is ~7 tok/s; RLM wall-clock includes real generation. The claim is
  feasibility, correctness, and $0 -- not latency-optimality.
- imdb-1500 RLM used a mostly-code strategy (2 leaf calls) -- a coding-strong
  orchestrator compresses even semantic structure when it can; the clean per-item
  fan-out is the imdb-200 cell.

## Reproduce

```bash
# serve the three local models (see environment.txt for exact builds)
./scripts/local-serving/serve-models.sh start      # leaf  :8771
./scripts/local-serving/serve-pitune.sh            # orch  :8773
./scripts/local-serving/serve-qwen36.sh            # orch  :8772
# run everything (resumable; skips completed cells)
bash experiments/superpowers/run_experiment.sh
# one A/B by hand:
python -m rrlm.bench.runner --task ledger --condition baseline --size 20000 --model pitune/qwen3.6-27b-pi-tune --reasoning off
python -m rrlm.bench.runner --task ledger --condition rlm      --size 20000 --model pitune/qwen3.6-27b-pi-tune --sub-model supergemma/Jiunsong/supergemma4-26b-uncensored-mlx-4bit-v2 --reasoning off
```
