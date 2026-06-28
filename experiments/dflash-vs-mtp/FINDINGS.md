# Track C: DeepSpec / DFlash vs MTP-draft on Apple Silicon

**Question (from CONTINUITY.md):** Does DeepSpec, DeepSeek's speculative-decoding
framework, give *faster, steadier* decode than MTP-draft for our models on Apple
Silicon?

**Short answer:** DeepSpec *itself* cannot run here: it is CUDA-only **training/eval**
code (8-GPU NVIDIA), not an inference engine. But the *method* it trains, **DFlash**
(a block-diffusion drafter from z-lab), has a native **MLX** inference runtime
(`dflash-mlx`) that already serves our orchestrator. On this M3 Max, DFlash gives
**~39 tok/s sustained** on a 512-token generation, about **3.1x** over the plain MLX
baseline (12.8 tok/s) and **~3.5x** over llama.cpp's MTP-draft path (11.3 tok/s) on the
same-size model. So the answer is yes, *but the win comes from the DFlash MLX runtime, not from
DeepSpec the framework.*

**IMPORTANT: the synthetic win does NOT carry into the real RLM task.** When the winner
is run through the repo's actual RLM eval (Section 8), the 3x synthetic decode advantage
*does not* translate: the RLM orchestrator workload is **prefill-dominated** (large REPL
contexts re-sent each turn, short structured-code outputs), so decode throughput is not
the bottleneck. End-to-end, DFlash showed no reliable speedup over plain mlx_lm, and in
matched cases was slower (with measurement confounds). Correctness holds (lossless).
Read Section 8 before acting on Section 6's serving recommendation.

All synthetic numbers below are on a free machine (fine-tune done), one engine at a time,
sustained ≥512-token generations, **Q6/Q8 weights only, never Q4** (per project rule).

---

## 1. What DeepSpec actually is (and why it doesn't run on Apple Silicon)

`deepseek-ai/DeepSpec` is *"a full-stack codebase for training and evaluating draft
models for speculative decoding."* It bundles data-prep utilities, draft-model
implementations, training code, and eval scripts, and implements three drafter
families: **DSpark**, **DFlash**, and **Eagle3**. Its configs and scripts *"assume a
single node with 8 GPUs"* via `CUDA_VISIBLE_DEVICES`, i.e. NVIDIA/CUDA. There is no
Metal/MPS path. It adapts code from SpecForge and targets Qwen3 / Gemma families.

Consequence: you cannot "run DeepSpec" on a Mac to get faster decode. DeepSpec is how
you *train* a drafter; you then need an *inference engine* that can execute that drafter
against a target model. On Apple Silicon that engine is **`dflash-mlx`** (below).

## 2. The Apple-Silicon-viable path: DFlash via `dflash-mlx`

**DFlash** (z-lab; paper arXiv:2602.06036, "DFlash") is a speculative-decoding method
whose drafter is a *lightweight block-diffusion model*: it generates a whole block of
draft tokens in a single parallel forward pass (no autoregressive drafting), conditioned
on hidden features pulled from several layers of the target model. The paper reports
*">6x lossless acceleration"* and *"up to 2.5x higher speedup than EAGLE-3"* on NVIDIA
GPUs (acceptance length ~4.6 tokens/step on Math500). It is **lossless**: the target
re-validates every drafted token, so output equals the target running alone.

The drafter for our orchestrator is `z-lab/Qwen3.6-27B-DFlash` (5-layer Qwen3 mini-model,
`block_size=16`, reads target hidden states from layers [1,16,31,46,61]). It is run by
**`dflash-mlx` 0.1.8** (a `uv` tool: `~/.local/share/uv/tools/dflash-mlx`), which is
MLX/Metal-native (`dflash doctor` confirms Metal backend; older-than-M5 GPUs use steel
fallback kernels). This is the same runtime `serve-models.sh` already uses for the
supergemma leaf. **This is the only DeepSpec-family path that runs on this Mac.**

Quant note: DFlash's MLX verify kernels only support **4-bit or 8-bit** drafts (no
6-bit path). Honoring "never Q4", we quantize the draft to **w8:gs64**. DFlash is lossless
regardless of draft precision (the Q8 target re-validates), so this only costs a little
speed vs. the w4 default; our DFlash numbers are therefore *conservative on speed*.

## 3. The two methods being compared

- **DFlash (MLX):** external block-diffusion drafter + `mlx-community/Qwen3.6-27B-8bit`
  (Q8) target, via `dflash serve`/`benchmark`. Adaptive verify (shrinks the block when
  acceptance drops).
- **MTP-draft (llama.cpp):** *self*-speculative, uses the GGUF's embedded multi-token-
  prediction (`nextn`) heads as the drafter, no separate model. Engaged by
  `llama-server --spec-type draft-mtp` on `bytkim/Qwen3.6-27B-MTP-pi-tune-GGUF:Q6_K`.

Both are lossless. They differ in engine (MLX vs llama.cpp/Metal), target weights
(Q8 official vs Q6_K pi-tune fine-tune), and drafting mechanism (external block diffusion
vs embedded MTP heads); see Caveats.

## 4. Setup

- Hardware: Apple M3 Max, 128 GB unified memory, macOS 25.6. Free machine (LFM
  fine-tune finished; no other model servers).
- Workload: the `dflash` "smoke" math prompt (102 prompt tokens), chat-templated,
  thinking off; sustained generation with EOS suppressed so every run reaches length.
- Measured: short (64-tok) and long (512-tok) generations; tok/s = decode throughput.
  DFlash via `dflash benchmark` (median of 2 repeats, 10s cooldown); llama.cpp via
  `llama-server` timings on `/v1/chat/completions`.

## 5. Results

### DFlash MLX (Qwen3.6-27B-8bit target + w8 draft), 512-tok sustained, median of 2

| metric | MLX baseline (no spec) | DFlash | ratio |
|---|---:|---:|---:|
| decode tok/s | 12.79 | **39.18** | **3.06x** |
| prefill tok/s | 49.0 | 120.4 | 2.46x |
| TTFT | 3132 ms | 865 ms | 3.6x faster |
| peak memory | 28.8 GB | 31.0 GB | +2.2 GB |
| acceptance ratio | n/a | 0.84 | n/a |

Steadiness: accepted-tokens/step falls from **5.45** (first 20) to **2.6** (last 20) as
the sequence grows, but the *adaptive* verifier shrinks the block (16→4) to compensate,
holding net throughput at 38.0 / 40.3 tok/s across the two repeats. Not a burst but a
sustained 512-token rate.

### llama.cpp as-committed (serve-pitune.sh: **no `-fa`**, ctx 65536)

| config | 64-tok tok/s | 512-tok tok/s |
|---|---:|---:|
| plain | 5.69 | 6.11 |
| MTP   | 8.56 | 7.89 |

This is the *current committed* serving config. Two findings: (1) it is badly throttled
because `serve-pitune.sh` omits `-fa on` (flash attention); (2) MTP does **not**
catastrophically collapse on long generations here (7.89 vs 8.56): the "~4 t/s collapse"
in the old script comment was the concurrent fine-tune starving the GPU, not MTP itself.

### llama.cpp optimized (`-fa on`, ctx 16384), same math prompt

| config | 64-tok tok/s | 512-tok tok/s |
|---|---:|---:|
| plain (fa on) | 9.71 | 10.12 |
| MTP (fa on)   | 10.92 | 11.26 |

Flash attention ~1.7x's the as-committed numbers. MTP adds only **~11%** over plain on
Metal (11.26 vs 10.12), and its long generation is *no slower* than its short one. On a
free machine, MTP-draft on Apple Silicon neither collapses nor helps much.

### Head-to-head (sustained 512-tok, math prompt, free machine)

| path | engine | target | spec | tok/s | vs llama.cpp MTP |
|---|---|---|---|---:|---:|
| **DFlash** | MLX | Qwen3.6-27B **Q8** | block-diffusion draft (w8) | **39.18** | **3.48x** |
| MLX baseline | MLX | Qwen3.6-27B Q8 | none | 12.79 | 1.14x |
| llama.cpp MTP | llama.cpp | pi-tune **Q6_K** | MTP nextn heads (fa on) | 11.26 | 1.00x |
| llama.cpp plain | llama.cpp | pi-tune Q6_K | none (fa on) | 10.12 | 0.90x |
| llama.cpp MTP (as-committed) | llama.cpp | pi-tune Q6_K | MTP, **no `-fa`** | 7.89 | 0.70x |

The DFlash MLX path is **~3.5x faster** than the best llama.cpp MTP path, and *steadier*
(adaptive verify holds the rate across the whole 512-token run). Notably, even the **plain
MLX baseline beats llama.cpp MTP**: the MLX engine is faster than llama.cpp/Metal for this
model *before* any speculation, and DFlash then triples it.

## 6. Recommendation

1. **Serve the orchestrator via DFlash MLX**, `mlx-community/Qwen3.6-27B-8bit` +
   `z-lab/Qwen3.6-27B-DFlash` (w8 draft), i.e. the path `dflash-qwen36.sh` already
   settled on. It is ~3.5x faster than the llama.cpp MTP path, *and* higher fidelity
   (Q8 vs Q6_K), at a modest memory cost (31 GB vs ~21 GB, trivial on 128 GB). This is
   the direct answer to Track C: **yes, the DFlash runtime gives faster and steadier
   decode than MTP-draft on Apple Silicon.**
2. **DeepSpec (the framework) buys nothing here directly**: it is CUDA training/eval
   code. Its only future relevance is *training* a better DFlash (or DSpark) drafter for
   our pi-tune weights on a rented CUDA box; the resulting drafter would then run under
   `dflash-mlx`. Not pursued now (no GPU box, and the stock z-lab draft already wins).
3. **If the llama.cpp/GGUF path is kept as a fallback, fix `serve-pitune.sh` to pass
   `-fa on`.** Omitting it costs ~40% throughput (6.1 → 10.1 t/s plain). MTP heads add
   only ~11% on Metal, so they are not worth special handling when DFlash is available.
4. **Retire the "MTP collapses on long generations" worry.** It did not reproduce on a
   free machine in either config; the original collapse was the concurrent fine-tune
   starving the GPU.

### Caveats / confounds

- Not a method-isolated A/B: DFlash uses the Q8 official target on MLX; llama.cpp uses
  the Q6_K pi-tune fine-tune. It *is* the real-world orchestrator choice. The conclusion
  is robust regardless: the MLX baseline (no spec) already beats llama.cpp MTP, so the
  engine + DFlash dominates independent of the quant/weights difference.
- Both paths are lossless, so output quality = the target's quality; the only quality
  lever is target fidelity (Q8 > Q6_K), which also favors DFlash.
- Single math prompt (the `dflash` smoke default). Acceptance on real agentic-code traces
  will differ, likely *higher* for repetitive code, which would favor DFlash further, but
  this is unverified.
- DFlash draft run at **w8** (not the w4 default) to honor "never Q4", so DFlash numbers
  are conservative on speed. The `z-lab/Qwen3.6-27B-DFlash` card also says it is "still
  under training", acceptance may improve with final weights.
- DFlash median of 2 repeats; llama.cpp single measurement per length. Thermal pressure
  reported "unknown" (results may be mildly throttled).

## 7. Reproduce

```bash
# DFlash MLX leg (Q8 target + w8 draft; never Q4):
dflash benchmark --model mlx-community/Qwen3.6-27B-8bit \
  --draft z-lab/Qwen3.6-27B-DFlash --draft-quant w8:gs64 \
  --max-tokens 512 --block-tokens 16 --no-eos --repeat 2 --cooldown 10 \
  --out experiments/dflash-vs-mtp/results/dflash_q8_w8_512.json

# llama.cpp MTP leg (Q6_K; fa on), same prompt:
bash experiments/dflash-vs-mtp/bench_llamacpp.sh

# as-committed serve-pitune.sh config (for contrast):
bash experiments/dflash-vs-mtp/bench_mtp.sh
```

---

## 8. End-to-end validation on the repo's RLM eval (the real test)

The synthetic decode benchmark (Sections 5-6) is a sustained-decode microbenchmark.
"Run the winner through the extensive test and see if it holds" means: serve the
orchestrator via DFlash and run the repo's actual RLM eval (`experiments/superpowers`),
then check whether accuracy and the speed advantage hold.

**Setup.** The official `qwen-official` model (`mlx-community/Qwen3.6-27B-8bit`) served via
DFlash on port 8772 (so the Pi provider transparently uses it), leaf = supergemma (DFlash
:8771), both with **w8** drafts (never Q4). Cells: `ledger` RLM, sizes 2000 / 20000,
several seeds. Compared against the same model served by plain `mlx_lm.server` (the
draftless path). Scripts: `run_winner_cells.sh`, `run_e2e_cell.sh`; data:
`results/e2e_compare.tsv`.

### Accuracy: HOLDS

Every DFlash RLM cell passed with an exact ground-truth match (e.g. ledger-2000 ->
10143.63, ledger-20000 -> 115797.87). DFlash is lossless (the target re-validates every
drafted token), confirmed in practice. (One *mlx_lm* fresh-seed run failed: temp 0.7
makes individual pass/fail stochastic for *either* backend; RLM is robust in aggregate,
not every single sample.)

### Speed: the synthetic win does NOT translate

ledger RLM wall-clock (seconds), DFlash vs the same model on plain mlx_lm:

| cell / condition | DFlash | mlx_lm | note |
|---|---:|---:|---|
| 2000, cold first-run of session | 94.86 | 93.54 | dominated by one-time warmup |
| 2000, warm, **identical** prompt (seed 42) | 23.1 | 51-72 | DFlash's prefix cache skips prefill, *artifact of repeating the same prompt* |
| 2000, **fresh** prompt seed 43 | 113.65 | 86.39 | DFlash slower |
| 2000, **fresh** prompt seed 44 | 207.56 | 65.49 | matched trajectory (both 2 calls, ~12k prompt, ~500-600 compl), DFlash ~3x slower |
| 20000, warm identical (seed 42) | 30.55 | 60.95 | prefix-cache artifact again |

**Why the inversion.** From the DFlash orchestrator log during the fresh runs:
- Each call spends **35-85s on prefill** of a ~6k-token REPL context. The RLM task is
  **prefill-dominated** with short decode (~250 tok/call), the *opposite* of the
  sustained-decode microbenchmark. Decode speedup barely moves the wall-clock.
- In-task DFlash **decode was 7.5-17.7 tok/s, far below the synthetic 39**, at 67-78%
  acceptance on tabular/code content, and it *degraded* call-over-call (17.7 -> 7.5),
  with prefill also falling (153 -> 74 tok/s).
- The eye-catching warm "23 s" was a **prefix-cache artifact**: repeating the *identical*
  prompt let DFlash skip prefill entirely. On fresh prompts that advantage disappears.

### Confounds

- End-to-end RLM wall-clock has **2-4x run-to-run variance** from cold/warm caches,
  prefix-cache reuse, and apparent soft throttling/contention over a long (~40 min)
  back-to-back measurement session (no formal `pmset` thermal warning, but decode
  visibly decayed). The DFlash fresh-seed runs were measured **last (hottest)**, so their
  absolute slowness is partly environmental; I do **not** claim a clean "DFlash is 3x
  slower" verdict.
- temp 0.7 -> the orchestrator's trajectory (turns, code written) varies per run,
  injecting wall-clock variance independent of serving speed.

### What does hold

1. **Correctness holds** (lossless; exact matches).
2. **The synthetic decode win does not produce an end-to-end RLM speedup.** The RLM
   orchestrator is prefill-bound and short-decode; decode speculation optimizes the wrong
   phase. In-task decode never approached the synthetic 39 tok/s.
3. For a *definitive* end-to-end speed verdict, the right instrument is a controlled,
   cooled, alternating A/B with an RLM-shaped synthetic workload (~6k-token prompt + ~300
   decode tokens), one engine at a time with cooldowns, not the trajectory-noisy live eval.

### Revised recommendation

- **Sustained long-form generation** (chat, long code emission): DFlash wins (~3x), use it.
- **The rrlm RLM orchestrator** (prefill-heavy, short structured decode): DFlash gives no
  clear end-to-end benefit. Prefer the simpler, robust path (plain `mlx_lm.server`), and
  spend optimization effort on **prefill / prefix reuse / shrinking the re-sent REPL
  context**, not on decode speculation. This *revises* Section 6, which judged DFlash the
  serving winner on synthetic decode alone.

---

## 9. Concurrency bake-off for Paivot (parallel agents, quality-first)

New deployment context: rrlm will back **Paivot**, where agents are ephemeral from
Pi's view but the **model server persists** (load once), and **multiple agents may run
in parallel**. Quality > speed. This reframes the serving choice around *concurrency*,
not single-stream decode. Harness: `concurrency_bench.py` fires C concurrent chat
requests sharing a long (~5.5k-token) prefix (the shared RLM framing every agent sends)
+ a short unique suffix + 160 decode tokens; one server at a time, cooldowns between.

Aggregate completion throughput (tok/s) and how it scales with concurrency C:

| C | mlx_lm (Q8) agg / scale / p95 | DFlash (Q8+w8) agg / scale | llama.cpp `--parallel 8` (Q6, fa) agg / scale / p95 |
|---|---|---|---|
| 1 | 3.80 / 1.0 / 42s | 3.68 / 1.0 | 6.09 / 1.0 / 26s |
| 2 | 7.61 / 2.0 / 42s | 5.82 / 1.58 | 4.60 / 0.76 / 70s |
| 4 | 6.96 / 1.83 / 92s | 5.26 / 1.43 | 5.95 / 0.98 / 108s |
| 8 | **3.72 / 0.98 / 344s** | (collapsed, >10 min) | **6.44 / 1.06 / 199s** |

**Findings:**
1. **One 27B on one GPU is GPU-bound**: *no* engine gives parallel speedup; aggregate
   throughput plateaus at ~6-7 tok/s regardless of C. The RLM workload is prefill-heavy,
   which saturates the GPU on prompt processing and leaves nothing to batch. More parallel
   agents = the same total throughput split more ways (per-agent latency rises with C).
2. **MLX engines (mlx_lm, DFlash) are single-stream and collapse under load.** mlx_lm
   peaks at C=2 then collapses at C=8 (aggregate back to 1-client level, p95 = 344s).
   DFlash behaves the same (C=8 didn't finish in 10 min). They have no continuous batching.
3. **llama.cpp `--parallel` degrades gracefully**, aggregate stays flat (no collapse),
   C=8 p95 = 199s (~1.7x better than mlx_lm's 344s), and it has the **lowest per-request
   latency** at C=1 (26s vs 42s) because fa-on prefill is fast (this workload is
   prefill-bound, so prefill efficiency dominates).
4. **DFlash's decode edge disappears here**: effective tok/s at C=1 (3.68) ~ mlx_lm
   (3.80), because the ~5.5k-token shared prefix makes prefill, not decode, the cost.

**Serving conclusion for Paivot:** use **`llama-server --parallel` (continuous batching,
`-fa on`)** for the orchestrator. It is the only engine that doesn't collapse under
concurrent agents and has the lowest per-request latency on the prefill-heavy RLM load.
MLX/DFlash are fine for a *single* long-form generation but wrong for parallel agents.
Caveat: a single local 27B caps aggregate throughput (~6 tok/s shared) no matter the
engine; for real parallelism, use a smaller orchestrator or more hardware.

**Model choice (quality-first):** decided by a separate quality eval, not the engine.
Candidates on llama.cpp: pi-tune Q6 (fine-tuned for agentic coding; already 100% on the
superpowers RLM cells; the current rrlm default) and **Ornith-1.0-35B Q6** (reportedly a
top coding model), under evaluation in Section 10.

---

## 10. Ornith-1.0-35B as orchestrator (the winner)

Tested deepreinforce-ai/Ornith-1.0-35B (Qwen3.5-based **35B MoE**, 256 experts with 8
active per token; agentic coding, SWE-bench Verified 75.6) at **Q6_K** on `llama-server --parallel 4 -fa on` (the engine
Section 9 picked), thinking suppressed (enable_thinking=false) for the RLM orchestrator
role, supergemma leaf. It follows the rrlm RLM REPL turn format correctly **out of the
box** despite not being trained on it.

### Quality + end-to-end speed (vs pi-tune Q6 dense, SAME engine, warm)

| RLM cell | Ornith-35B MoE | pi-tune-27B dense | speedup | both pass? |
|---|---:|---:|---:|---|
| ledger 2000 | 16.8 / 17.4 s | 81.9 s | ~4.8x | yes (exact) |
| ledger 20000 | 19.4 s | 151.4 s | ~7.8x | yes (exact) |
| bugfind 60 (code) | 148.5 s | 718.7 s | ~4.8x | yes (found fn_040_sum_to) |
| imdb 200 (semantic+leaf) | 63.9 s | 363.8 s | ~5.7x | yes (P205, 0.55) |

Ornith passes **all three task types** (arithmetic, code, semantic-with-leaf-fan-out)
with exact ground-truth matches, ~5-8x faster end-to-end than the dense 27B. imdb's
`by_role` shows both `main` and `sub` calls -> Ornith correctly drives the leaf fan-out.

### Concurrency (the Paivot factor): Ornith actually scales

| C | Ornith agg tok/s / scale / p95 | dense mlx_lm agg / p95 |
|---|---|---|
| 1 | 43.5 / 1.0 / 3.7 s | 3.8 / 42 s |
| 4 | 42.4 / 0.97 / 15 s | 7.0 / 92 s |
| 8 | **60.2 / 1.38 / 21 s** | 3.7 / **344 s** |

Ornith's MoE keeps active params low, leaving GPU headroom for llama.cpp continuous
batching to **scale up** (60 tok/s aggregate at C=8, 21 s latency), where every dense
27B saturated one stream and collapsed (mlx_lm p95 = 344 s at C=8). ~7-11x the
single-request throughput of the dense models, too.

### Decision

**Orchestrator = Ornith-1.0-35B Q6_K on `llama-server --parallel -fa on`** (thinking off
for the RLM role). It wins on every axis that matters for rrlm/Paivot: quality (correct on
all task types; top coding model), end-to-end speed (~5-8x over dense 27B; MoE prefill is
cheap and RLM is prefill-bound), and concurrency (scales gracefully for parallel agents).
Q6_K honors the never-Q4 rule. pi-tune Q6 remains a solid fallback; mlx_lm/DFlash are for
single-stream long-form generation, not parallel agents.

Open follow-ups: (a) the **leaf** (supergemma via DFlash) still serializes under
concurrency, so consider serving it on llama.cpp `--parallel` too if parallel agents make
heavy semantic-leaf calls; (b) Ornith is a reasoning model run thinking-off (matches the
RLM design, which supplies reasoning externally), thinking-on was not needed for
correctness here.
