# Findings: RLM-first vs context-stuffing (qwen3.7-max, 2026-06-12)

> **Scope note (read first).** This document records the RLM-vs-baseline benchmark
> methodology and results, which stand. Its local-serving conclusion further down
> ("Final orchestrator: pi-tune Q6_K", 2026-06-17) is a point-in-time result that was
> later **superseded** by a dedicated serving bake-off: the settled local orchestrator
> is now **Ornith-1.0-35B** on `llama-server --parallel`, not pi-tune. For the current
> local stack and the bake-off that chose it, see
> [LOCAL_SERVING.md](LOCAL_SERVING.md) and
> [../experiments/dflash-vs-mtp/FINDINGS.md](../experiments/dflash-vs-mtp/FINDINGS.md)
> (Section 10). The pi-tune material below is kept as a dated lab record.

All runs: OpenRouter `qwen/qwen3.7-max`, authoritative cost/timing from the
generation endpoint, artifacts under `runs/`. n=1-2 per cell. Conditions:
`rlm` (predict-rlm REPL agent, doctrine skill, depth-gated rlm_spawn) vs
`baseline` (full data stuffed into one prompt).

## Results by task type

### Mechanical aggregation (ledger: exact sum over N lines)

| size | data | RLM | baseline |
|---|---|---|---|
| 500 | 35KB | pass $0.022 / 26s | pass $0.039 / 50s |
| 2,000 | 140KB | pass $0.013 / 24s | pass $0.153 / 192s |
| 10,000 | 700KB | pass $0.008 / 15s | **wrong answer** $0.699 / 541s |
| 50,000 | 3.5MB | pass $0.013 / 22s | **provider 400** before generating |

RLM cost/latency is flat in input size. Baseline accuracy cracks between 2K
and 10K items; the provider rejects 50K outright despite the advertised 1M ctx.

### Semantic retrieval (needle: paraphrased relocation fact among distractors)

120KB, 2 seeds: RLM pass $0.012 / 18s; baseline pass $0.039 / 28s. RLM ~3x
cheaper, both accurate.

### Semantic aggregation (reviews: highest negative-fraction product)

| size | data | RLM | baseline |
|---|---|---|---|
| 600 | 60KB | pass $0.025 / 51s | pass $0.059 / 187s |
| 5,000 | 637KB | pass $0.038 / 89s | pass $0.277 / 139s |

RLM 2.4-7.4x cheaper. The agent defeated dedup-resistant slot-filled reviews
by *template induction* (normalize variable slots, classify recovered
skeletons, aggregate mechanically).

### Code reasoning (bugfind: one buggy function among N)

| size | data | RLM | baseline |
|---|---|---|---|
| 120 | 14KB | pass $0.024 / 37s | pass **$0.013** / 35s |
| 2,000 | 240KB | pass $0.023 / 44s | pass $0.095 / **23s** |

The only baseline cost win in the study: tiny data, where RLM scaffold
overhead (~15-25K prompt tokens) exceeds the data itself. At 2,000 functions
RLM is 4x cheaper (it groups identical-base functions, diffs within groups,
property-tests the suspect in the REPL before submitting).

Docker `sbx` backend parity: bugfind-120 $0.025 vs $0.024 on jspi, +15-20s
sandbox startup (amortizable with SbxPool).

## Cross-cutting observations

1. **Zero `predict()` calls and zero `rlm_spawn` across all 9+ RLM runs.**
   The agent always found a pure-code strategy, including template induction
   against corpora engineered to resist dedup. Synthetic/template data cannot
   force the semantic fan-out path: a coding-strong model treats hidden
   structure as a compression problem and wins. Exercising predict()/recursion
   requires genuinely irreducible natural text or real repos.
2. **Thinking-model tax on baseline.** qwen3.7-max burns completion tokens
   reasoning through mechanical work in the stuffed condition (33.7K completion
   tokens at ledger-10K, and reasoning is apparently not capped by max_tokens
   on this provider). This eliminates the expected small-task crossover for
   mechanical work: RLM wins even at 500 lines.
3. **Baseline failure modes at scale are distinct:** silent accuracy collapse
   first (wrong-but-confident totals), hard provider rejection later.
4. **RLM scaffold overhead is the routing variable.** Fixed ~15-25K prompt
   tokens + 2-5 serial REPL turns. Below ~10-15K data tokens on read-only
   tasks, stuffing is cheaper and similarly accurate.

## Routing recommendation (draft)

A trivial pre-router suffices; everything else the RLM doctrine already
self-routes internally (code vs predict vs spawn):

```
if task requires exactness over many items (count/sum/exhaustive search):
    -> RLM at any size            # baseline accuracy cracks before ctx limits
elif data_tokens < ~12K and task is read/judgment only:
    -> direct prompt              # RLM scaffold overhead dominates
else:
    -> RLM
```

With latency-sensitive workloads, add: medium read-only tasks (~50-250KB)
favor baseline on wall-clock when the model can answer without heavy
reasoning (bugfind-2000: 23s vs 44s), pay ~4x cost for ~2x speed.

## Cross-model battery (2026-06-12, seed 42)

Models: qwen3.7-max (1M ctx, thinking), qwen3.7-plus (1M ctx, thinking),
gemma-4-26b-a4b-it (262K TOTAL ctx, pinned cloudflare/siliconflow),
qwen3.6-27b (262K ctx, pinned chutes/deepinfra). The last two are the
locally-runnable, low-context arm of the hypothesis.

### ledger-10000 (mechanical, ~460K tokens stuffed)

| model | RLM | baseline |
|---|---|---|
| qwen3.7-plus | pass $0.0067 / 39s | **wrong** $0.754 / 1516s (82K thinking tokens) |
| gemma-4-26b | pass $0.0023 / 18s | **rejected** (exceeds 262K ctx) |
| qwen3.6-27b | pass $0.0077 / 34s | **rejected** (exceeds 262K ctx) |

RLM is not an optimization for the 262K models here: it is the only way the
task can be done at all. Capability grant confirmed: small-context models do
1M-token-class work for under a cent.

### bugfind-2000 (code, 240KB)

| model | RLM | baseline |
|---|---|---|
| qwen3.7-plus | pass $0.065 / 483s (11 turns, struggled) | pass $0.025 / 23s |
| gemma-4-26b | **fail** $0.015 / 276s | pass $0.0081 / 13s (21 output tokens) |
| qwen3.6-27b | pass $0.023 / 80s | pass $0.025 / 74s |

Within-context code reading favors the baseline for weaker orchestrators;
gemma could not drive the REPL to a correct diff-based strategy but read the
answer directly in 13s. Only qwen3.6-27b matched its own baseline via REPL.

### imdb-1000 (natural semantic text, 521KB, template induction impossible)

| model | RLM | baseline |
|---|---|---|
| qwen3.7-max | pass $0.367 / 194s (21 predict calls) | pass $0.227 / 385s |
| qwen3.7-plus | **fail** (no answer, 16 sub-calls) | pass $0.073 / 464s |
| gemma-4-26b | pass $0.057 / 629s (**1020 predict calls**) | pass $0.013 / 29s |
| qwen3.6-27b | pass $0.539 / 625s (52 predict calls) | **fail** (truncated at 32K completion cap) |

The semantic fan-out path finally activated, on real text every model
reached for predict(), at self-chosen granularity (per-review for gemma,
~50-per-batch for max). Doctrine self-routing works. But for strong long-ctx
readers within budget, stuffing is cheaper and faster on pure semantic reads.

### Thinking ablation (ledger-2000 baseline, qwen3.7-max)

reasoning=default: correct, $0.153, 192s. reasoning=off: **wrong in 5s**
($0.115, 19 output tokens). The baseline's thinking burn is not waste: it is
the only thing making stuffed mechanical work correct. Stuffed condition must
choose slow+expensive+right vs fast+wrong; RLM sidesteps the dilemma.

## Updated routing rule (model-aware)

```
if data_tokens > model_context * ~0.8:        -> RLM (only option)
elif task needs exactness over many items:    -> RLM at any size
elif data_tokens < ~12K, read-only:           -> direct prompt
elif task is semantic-read and model is a strong long-ctx reader:
                                              -> direct prompt (cheaper+faster)
else:                                         -> RLM
```

Cost lever for semantic fan-out: sub-LM thinking burn dominates (27b spent
230K completion tokens across 52 sub-calls). Route predict() to a cheap
non-thinking sub model (--sub-model) or reasoning=off for the sub-LM.

## Mixed-pair configuration: qwen3.6-27b orchestrator + gemma-4-26b sub-LM

imdb-1000 (semantic fan-out task), all RLM-condition configs compared:

| config | result | cost | wall |
|---|---|---|---|
| 27b solo (thinking subs) | pass | $0.539 | 625s |
| 27b stuffed baseline | fail (completion truncation) | $0.163 | 1289s |
| gemma solo | pass | $0.057 | 629s |
| gemma stuffed baseline | pass | $0.013 | 29s |
| 27b+gemma, thinking orchestrator (s42) | **fail** (sandbox exec timeout: hung fan-out code) | $0.038 | 529s |
| 27b+gemma, thinking orchestrator (s43) | pass | $0.094 | 414s |
| **27b+gemma, reasoning=off orchestrator** | **pass** | **$0.048** | **87s** |

The no-thinking pair is 11x cheaper and 7x faster than 27b solo. Its trace is
the textbook RLM program: probe -> parse -> per-review predict() via one
asyncio.gather over gemma (1,001 sub-calls) -> aggregate -> SUBMIT, in 4 REPL
turns.

**Orchestrator thinking is a liability, not an asset** (n=3 signal): with
reasoning on, the 27b orchestrator hung its own fan-out code once (300s
sandbox timeout) and was 5x slower when it worked. The doctrine supplies the
plan; orchestration is mechanical code-writing where thinking adds latency
and variance. Leaf-classification needs no thinking either. Recommended
local-class default (now in Makefile): MODEL=qwen3.6-27b SUB=gemma-4-26b
REASONING=off.

Pair on non-fan-out tasks (no regression): ledger-10000 pass $0.0072/46s,
bugfind-2000 pass $0.0131/55s.

## Local replication (2026-06-14/15): same models, on-device

Goal: reproduce the OpenRouter pair result on locally-served models. Orchestrator
= Qwen3.6-27B, leaf = gemma-4-26b, both served on-device (mlx_lm.server / DFlash
/ LM Studio), reasoning off, cost = $0 (metrics: wall-clock + tokens).

### Uncensored + quantized variants: fail at scale

heretic (Qwen3.6-27B uncensored oQ8) orchestrator + supergemma (gemma-4-26b 4bit)
leaf. Replicated at moderate scale, ledger-10000, bugfind-2000, imdb-200 (full
200-call fan-out, correct) all pass. But imdb-1000 failed three times, three
different ways, all orchestrator-side: dithered on batch sizes then guessed
wrong; ran past the LM call timeout; ran away to a 16k-token malformed output.
The leaf classified correctly whenever invoked.

### Official orchestrator + same cheap leaf: passes at full scale

Swapping ONLY the orchestrator to official Qwen3.6-27B (via LM Studio), keeping
the identical quantized supergemma leaf: imdb-1000 passes on both seeds and both
sandbox backends (4/4).

| seed | jspi | sbx (Docker) |
|---|---|---|
| 42 | pass P203, 1943s, 2000 subs | pass P203, 1340s, 1010 subs |
| 43 | pass P205, 2357s, 153 subs | pass P205, 2099s, 2000 subs |

Both decomposition strategies appear (per-review fan-out and batching); both
reach the right answer. sbx is ~30% faster than jspi for wide fan-out.

**Conclusion: orchestrator fidelity is the entire gap.** Same cheap leaf; only
the orchestrator changed (uncensored-oQ8 -> official) and the result flips from
unreliable to 4/4. This is the strongest form of the earlier finding that
orchestrator quality dominates leaf quality, and it validates the cost-optimal
shape: a high-fidelity orchestrator with cheap, quantized, non-thinking leaves.

### Engineering findings for local RLM deployment

- **predict-rlm timeouts assume cloud-speed concurrent leaves.** The whole
  fan-out runs as one REPL turn; the 300s sandbox `exec_timeout` (both
  JspiInterpreter and SbxConfig) is exceeded when local leaves serve serially.
  Fix: scale the per-turn window to fan-out width (`sandbox_exec_timeout`, auto
  3600s for local endpoints). This explained why imdb-200 passed (fit 300s) but
  imdb-500/1000 failed (overran it).
- **LM Studio rejects `response_format: json_object`** (wants `json_schema` or
  `text`); DSPy's JSONAdapter fallback sends json_object. Fix: register the model
  with litellm as `supports_response_schema=True` so DSPy emits json_schema.
- **mlx_lm.server loads the model named per request**, the registry slug must
  equal the server `--model` value; thinking is toggled via
  `chat_template_kwargs.enable_thinking`, not the OpenRouter `reasoning` field.
- DFlash speculative-decoding drafts (z-lab) are gated HF repos; lossless, so
  draftless serving is metric-equivalent (only slower).

## Packaging: rrlm as a Pi backend

The settled harness is wrapped for [pi](https://github.com/earendil-works/pi)
(see `pi/`): a `rlm_solve` tool (extension) delegates data-heavy subtasks to
`rrlm.solve`, and an `rlm-first` skill encodes the routing rule. Verified
end-to-end, pi calls `rlm_solve`, the harness runs, the answer returns
(json-mode `tool_execution_*` events confirm the call). The agent keeps a map of
state in context; the data lives in the harness REPL.

## Local orchestrator selection (2026-06-17): server decode matters

After validating the official orchestrator path, we compared four local
orchestrators for Qwen3.6-27B, all with the supergemma-26b leaf, jspi backend,
reasoning off, on imdb-1000:

| Orchestrator | imdb-1000 | Why |
|---|---|---|
| Official Q8 via **mlx_lm** (autoregressive) | **2/2** | reliable |
| Official Q8 via **DFlash** (speculative) | 1/2 | s43 hard-fails on a DFlash decode artifact |
| **MTPLX** (MTP-optimized server) | unreliable | ~50% parse failures + an engine spin-wedge that hangs the run |
| Uncensored **heretic** (oQ8) | unreliable | intermittent malformed/empty turns |

Key findings:

- **DFlash determinism breaks RLM orchestration on edge cases.** DFlash ignores
  client temperature (temp 0.0 == temp 2.0, byte-identical) and its
  block-diffusion speculative decode produced a malformed turn at one specific
  REPL state (s43 turn ~14) that *no* retry could fix on that server. The
  **same build via mlx_lm** (standard autoregressive decode) passed the
  identical task. So it is the server's decode path, not the model build.
- **DFlash's speedup is marginal for this workload.** Wall-clock is dominated by
  leaf fan-out (hundreds-to-thousands of supergemma calls), not the ~15-19
  orchestrator turns DFlash accelerates. mlx_lm runs were comparable-or-faster.
  Speculative decoding on the orchestrator is not worth its reliability cost
  here.
- **Settled production config:** official Qwen3.6-27B **Q8 via mlx_lm**
  (`serve-qwen36.sh`, :8772) + supergemma leaf (:8771) + **jspi** backend +
  nudge-retry. Matches the LM Studio official 4/4 (also autoregressive serving).
- **Supervisor backend (predict-rlm 0.7 DirectPythonBackend) rejected:** it
  deadlocked after ~11 turns and leaked subprocesses. Use jspi/sbx.
  > Superseded: this was observed on a 0.7.0-alpha build. On stock predict-rlm
  > 0.7.0 from PyPI the supervisor backend runs the full offline suite and daily
  > use cleanly, and it is now the rrlm default (no Deno/Docker required); jspi
  > and sbx remain the isolation options. Kept as a dated lab record.
- **Action-generation retry (local predict-rlm branch, not upstreamed):** the
  retry must change the *prompt* (a nudge naming the parse failure), not just
  sampling, deterministic backends ignore temperature, so plain re-ask and
  temperature escalation are no-ops there. The nudge altered the prompt but
  could not rescue the DFlash s43 turn (the model was genuinely stuck at that
  state); switching the server fixed it. The retry remains sound general
  insurance for stochastic backends.

## Orchestrator as of 2026-06-17: pi-tune Q6_K (later superseded by Ornith)

> Superseded: this was the chosen orchestrator on 2026-06-17. A subsequent serving
> bake-off settled on **Ornith-1.0-35B** on `llama-server --parallel` (see the scope
> note at the top and [LOCAL_SERVING.md](LOCAL_SERVING.md)). The analysis below is kept
> as a dated record; `serve-pitune.sh` is no longer part of the repo.

A Pi-harness-tuned Qwen3.6-27B (`bytkim/Qwen3.6-27B-MTP-pi-tune`, no-thinking
native, trained on real agent traces) replaced the base-Qwen orchestrators.
Served as Q6_K GGUF via llama-server (plain autoregressive), temp 0.7.

| Orchestrator (+ supergemma leaf, jspi) | imdb-1000 | action turns | notes |
|---|---|---|---|
| base Qwen Q8 / mxfp8 (mlx_lm) | 2/2 | 12-19 | needed retry/server-swap insurance |
| base Qwen DFlash | 1/2 | 14 | s43 unrecoverable (deterministic decode) |
| **pi-tune Q6_K (llama.cpp)** | **2/2** | **6-7** | clean turns, zero malformed, zero retries |
| pi-tune Q4_K_M | 2/2 | 7-10 | passed but sloppier (more exec-error churn); see below |

Decisions, with receipts:

- **pi-tune wins the orchestrator role.** It emits clean, well-formed action
  turns and solves in ~half the turns of base Qwen, eliminating the malformed-
  turn failures that drove the retry/server-swap work. Purpose-built (Pi-harness
  SFT, no-thinking).
- **MTP self-speculation rejected.** llama.cpp build 9670 has `--spec-type
  draft-mtp` (PR #22673). It loads and keeps quality (2/2), but on Apple Silicon
  it is **~33% slower** (6.09 vs 9.18 tok/s decode at 60% draft acceptance) --
  the draft+verify overhead exceeds the speculative savings on a bandwidth-bound
  Mac. Kept as an off-by-default `SPEC_MTP` toggle in `serve-pitune.sh`.
- **Q6_K beats Q4_K_M.** Q4 passed 2/2 but was sloppier (more REPL exec-error
  churn: 3/7 and 5/10 steps vs Q6's ~1-2/6) and **not faster** (8.87 vs 9.18
  tok/s, bandwidth-bound; the smaller quant's dequant cost cancels the size
  benefit), saving only ~5GB. Q6_K is the settled quant; Q4 GGUF deleted.
- **General lesson (reaffirmed):** for this leaf-fan-out-dominated workload,
  orchestrator decode speed barely moves wall-clock; orchestrator *reliability*
  (clean turns) is what matters. Speculative tricks (DFlash, MTP) and smaller
  quants traded reliability/cleanliness for speed that the workload can't use.

Local config as of this date (since superseded by Ornith, see the scope note):
`bytkim/Qwen3.6-27B-MTP-pi-tune` **Q6_K** via llama-server (:8773, no-thinking,
temp 0.7) + supergemma-26b leaf (DFlash :8771) + **jspi** backend + nudge-retry
insurance.

## Open items / caveats

- n=1-2 per cell, single model, synthetic template data.
- predict()/rlm_spawn paths unexercised, needs natural corpora and real
  repos (Docker sbx) where code cannot compress the semantics.
- Reasoning-token cap on OpenRouter qwen appears decoupled from max_tokens;
  budget guards should use cost ceilings rather than token caps.
