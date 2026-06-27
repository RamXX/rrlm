# There Is Another Way

## How a 27-billion-parameter model on a laptop did work that "requires" a frontier model and a data center — for nothing

There is an assumption baked into almost every roadmap in AI right now: if you want
more capability, you buy it. A bigger model. A longer context window. More tokens,
more GPUs, more cloud. Capability is a line item.

This is the story of taking that assumption apart — not with an argument, but with a
controlled experiment you can re-run yourself, and a working application built from
nothing to prove the point twice. It runs through a small model on a laptop, with no
internet and at zero cost, doing exact computation over 900,000 tokens of data on a
model whose context window holds 65,000 — work it **cannot do on its own, at any size
of prompt** — and it ends with that same small model building real, working software,
file by file, compiling as it goes. Not because the model got smarter. Because we
changed the *technique*.

The technique is the Recursive Language Model, but the deeper idea is a posture: a
model should try to *do* things — write code, run it, check the answer — and only fall
back to guessing from its weights when nothing checkable will do. The point of this
article is that the technique, not the model size, is the capability — and that has
consequences the whole industry has not priced in yet.

---

## 1. The bet

Conventional agents pour data *into* the model: read the file, paste the logs, stuff
the codebase into the context window, and hope the model reasons over all of it.
Every token of data competes with every token of thought, and the model's
in-context arithmetic and memory degrade quietly as the pile grows.

The Recursive Language Model (Zhang, Kraska, Khattab, 2025) inverts this. The data
never enters the model's head. It lands in a sandboxed Python REPL as a variable.
The model acts *only by writing code* to probe it — `len(data)`, a few slices, a
regex, a sum — and it spawns cheap sub-model calls only for the irreducible judgment
that code can't express. The model's context holds a **map** of the problem, not the
territory.

The bet we set out to prove: this technique doesn't just make big models cheaper. It
gives **small** models capabilities they otherwise do not have. If that's true, the
economics of the whole field tilt — because small models run on hardware you already
own.

We built `rrlm` to test it: an RLM-first harness on top of `predict-rlm`, wired so
that any coding agent can offload data-heavy work to a recursive REPL instead of
drowning in its own context.

---

## 2. From a private experiment to a tool anyone can run

The first job was honesty about state. `rrlm` started as a personal experiment —
hard-wired to one machine, one set of local model paths, one person's config. To
claim it works, it had to work for *anyone*.

So we made it real:

- **It reads your config, not ours.** Models resolve from your
  [Pi](https://github.com/earendil-works/pi) setup — local servers, OpenRouter,
  OpenAI, Anthropic, whatever you already run. No hardcoded registry, no absolute
  paths, no "works on my machine."
- **It installs from zero.** A clean `uv sync` pulls stock `predict-rlm` from PyPI;
  a feature-detection shim means it runs whether or not you have any local patches.
  We verified it in a throwaway virtualenv built from the published wheel.
- **It proved itself on real tasks**, not toy demos: exact aggregation over a 315 KB
  CSV (correct, $0.005, 15s), code reasoning across a real repository, and — the one
  that matters — an **end-to-end Pi session** where the agent itself decided to
  delegate to the RLM backend and got the right answer, on the current Pi release.

That last one is the shape of the future we're arguing for: an agent that knows when
*not* to read, and hands the heavy lifting to a technique instead of a bigger bill.

---

## 3. The detour that taught us the most

To put the idea under load, we pointed a *local* model — through Pi — at a real
build: a graph-native CRM in Go, on an embedded graph database. The thesis we were
chasing here was the application layer: a small model, on-device, building real
software and using RLM for the data-heavy parts.

It went sideways, and the failure was the lesson.

The local model, generating at roughly seven tokens a second, decided to write the
entire data layer — seven hundred lines — in a single file, in one shot, **without
ever compiling once.** Sixteen minutes in, it still hadn't finished that first file,
and it had checked nothing. Left alone it would have produced a wall of untested code
and then spent hours trying to untangle it. We killed it.

It would have been easy to write that up as "small models can't code." That would
have been wrong, and lazy. The model's *code* was fine — graph-aware, well-structured.
Its *method* was doomed, and the method was something we could fix from the outside:
force one small file at a time, compile after every file, never fly blind. That is
how careful engineers work, and it is exactly the discipline a slow local model needs.

The detour clarified the real question. We had been measuring the wrong thing. "Can a
small model code?" is a distraction. The question worth answering is sharper, and the
person funding this said it plainly:

> *We're trying to prove that RLM, the technique, can give a small model superpowers.*

So we set the build aside — we would come back to it, and finish it — and built a
proof first.

---

## 4. The experiment

The design is deliberately boring, because boring is what makes it credible. **Hold
the model fixed. Vary only the technique.**

- **baseline** — the model alone. Task and data go into its prompt. This is how
  everyone uses these models today.
- **rlm** — the *same* model, same weights, same machine, plus the RLM technique:
  data in a REPL, the model writes code, a cheap leaf model is fanned out for
  semantic judgment.

Every task has a ground-truth answer the harness computes independently and checks
against. Everything runs locally, so the cost is exactly zero and the only things
worth measuring are **correctness, wall-clock, and tokens.**

The rig, for the record:

- **The model:** Qwen3.6-27B — two builds, a Pi-tuned Q6_K and the official 8-bit —
  served on an Apple M3 Max with a **65,536-token** context window. A small model by
  2026 standards, on a laptop.
- **The leaf:** a quantized 26B served via speculative decoding, for the fan-out.
- **The tasks:** exact aggregation (a filtered sum over a ledger), retrieval (a fact
  hidden among distractors), code reasoning (one bug among many functions), and real
  natural-language judgment (which product has the worst reviews). Sizes deliberately
  swept *across* the model's context wall. Three random seeds. Two model builds.
- **55 cells in total**, every one with a verifiable answer.

---

## 5. What we found

**The scoreboard, first, because it is not subtle:**

> **The model alone passed 4 of 27 cells (15%). The same model, with RLM, passed 28
> of 28 (100%).**

The four the model got on its own were exactly the small, easy, fits-in-the-window
cases. Everything large or exact, it got **wrong or could not attempt** — and the
technique solved all of it. Three findings carry the proof.

### The capability cliff

The ledger is a filter-then-sum: total the `ok` transactions for one user. Token-dense
rows fill the 65K window at about 1,400 of them.

| rows | data | model alone | model + RLM |
|---|---|---|---|
| 500 | fits | **wrong** — 3501.97 vs 2743.79 | **exact** — 2743.79 |
| 2,000 | ~92K tok | **impossible** — context-overflow error in 0.7s | **exact** — 10,143.63 |
| 20,000 | ~900K tok | **impossible** | **exact — 115,797.87, 200s, zero sub-LM calls** |

Past the wall the model alone is simply dead. RLM keeps the data in the REPL, writes
a dozen lines of Python, and computes the exact answer — and its wall-clock is *flat*
in data size (it does the same small amount of work whether the ledger has 500 rows
or 20,000). The model never reads 900,000 tokens. It never has to. It computes over
them.

### It is not "just use a bigger window"

The obvious objection is that this is all about context size — buy a longer window and
the gap closes. It does not. We swept ledger sizes that **all fit** the window
comfortably, across three seeds:

| rows (all fit) | model alone, correct | model + RLM, correct |
|---|---|---|
| 100 | **1 / 3** | 3 / 3 |
| 300 | **0 / 3** | 3 / 3 |
| 600 | **0 / 3** | 3 / 3 |
| 1,000 | **0 / 3** | 3 / 3 |
| 1,300 | **0 / 3** | 3 / 3 |

The model alone can reliably add up a few dozen numbers. By a few hundred it is
already wrong — off by ten on one run, off by a couple hundred on another — and this
is **well inside** its context window. Its in-context arithmetic is simply not
trustworthy. RLM was exact on all fifteen. The gap is not about how much the model
can *see*. It is about whether it can *compute*, and code computes where reading
guesses.

The point survives a bigger window, too: on the official Qwen build with a window
*four times larger* (262K tokens), the model alone still miscounted a 2,000-row
ledger — and took **twenty minutes** to be wrong. With RLM: exact, in 114 seconds.
And when we pushed that build to 20,000 rows, the model-alone path did not just fail
— the 900,000-token prompt **crashed the GPU server** outright (`[METAL] command
buffer execution failed`). RLM solved the identical task in 206 seconds without
breaking a sweat, because it never put the data in front of the GPU in the first
place.

### What the superpower actually is

We instrumented every run, so the mechanism is not a story, it is a count.

- On the 900K-token ledger, RLM made **zero** sub-model calls. It solved it in four
  turns of plain Python — and, unprompted, it *verified its own answer with a second,
  independent parse* before submitting:

  ```python
  ok = []
  for line in data.splitlines():
      if 'user=u754' not in line or 'status=ok' not in line:
          continue
      ok.append(float(re.search(r'amount=(\S+)', line).group(1)))
  total = round(sum(ok), 2)        # then re-parsed a different way and re-checked
  ```

- On the natural-language task, where code cannot compress genuine judgment, RLM fanned
  the cheap leaf out — **201 calls**, one per review — and aggregated the verdicts
  mechanically. When the reviews overflowed the window (1,500 of them), the model
  alone was again impossible; RLM was exact.

Two distinct superpowers, then, from one technique: **code execution** over data the
model never reads, and **cheap fan-out** for the judgment it can't avoid. An ablation
nailed down what the small leaf buys you — not correctness (the orchestrator can fan
out to itself and still be right) but **roughly 3x the speed**. The cheap model is an
efficiency, not a crutch.

### The honest boundary

This is not magic and we will not pretend it is. RLM is a *targeted* superpower, and
on small, easy, fits-in-the-window tasks the model alone is fine — and faster. It read
60 functions and found the bug in 22 seconds; RLM took twelve minutes to reach the
same answer, because writing and running code carries a fixed overhead that tiny
inputs don't justify. That RLM does *not* win everywhere is precisely what makes the
places it wins — decisively, repeatably, on data that is large or computation that
must be exact — worth believing.

---

## 6. It held up under fire

A proof you can't trust under stress isn't a proof. This one earned it the hard way.

Mid-run, the harness reaped three background processes at once — including a model
server — and silently killed the experiment. Separately, that 900,000-token prompt
crashed the GPU. Neither was a reason to stop; both were data. We diagnosed the
reaping (detached processes survive; tracked ones don't), rebuilt the entire
experiment to be **resumable** — every completed cell skipped on relaunch — and
detached it so it could outlive the next reap. We restarted the crashed server and
re-ran the single casualty cleanly. Not one data point was lost, and the final tally —
28 of 28 for RLM — includes that recovery in full view. The crash itself became
evidence: the model-alone path doesn't merely fail at scale, it self-destructs, while
the technique never even exposes the machine to the load.

Everything is committed: the writeup, the raw results, the resumable driver, the exact
environment, and 58 preserved run snapshots including the actual REPL traces. One
command re-runs it from zero.

---

## 7. But it was never just about data

Here is where we almost fooled ourselves. It is tempting to read the result above as
"RLM is a great trick for big data." That is too small. The real variable in every one
of those 55 cells was not *size* — it was *verifiability*. Reading-and-reasoning is
stochastic and uncheckable, so the model guessed and missed. Writing code is
deterministic and checkable, so it was exact, and it re-checked itself. The principle
was never "use a REPL for large inputs." It is:

> **Wherever a procedure exists whose answer you can check, prefer it to the model's
> unverifiable guess. Fall back to raw generation only for the irreducible part no
> procedure can express.**

That is a posture for *everything*, not a tactic for spreadsheets. "What is the capital
of France" stops being a recall from the model's weights — which can be stale or simply
wrong — and becomes a grounded lookup you can cite and re-run. The model's role shifts
from *oracle* to *orchestrator*: its first instinct is to write the program, call the
tool, decompose the problem, and check the result; its own free-form generation becomes
one instrument in the kit, and the fallback rather than the default.

So we went back to the CRM — the build we'd abandoned — to test the posture on
something that looks nothing like a sum.

The first attempt had failed exactly the way you'd expect a small model to fail: handed
the spec, it tried to write the entire data layer, seven hundred lines, in one shot,
never compiling, and after sixteen minutes had produced nothing that worked. The easy
conclusion was "small models can't build software." It was the wrong conclusion. The
model's *code* was fine. It had abandoned the discipline — it stopped decomposing and
stopped verifying. So we rebuilt the harness to make the posture non-negotiable: one
small file at a time, compile after every file, check the real API instead of guessing
it, write a test.

The same model — same weights, same laptop — then built a working CRM. Six clean,
properly sized files, not a blob. It compiled at every step. It grounded itself against
the real database API with `go doc` instead of hallucinating it (which is exactly what
had broken the first attempt). Its test passed. And the command line *worked*:
`crm init` created a real graph database, `crm contact add` wrote a person into it, and
`crm contact list` read that person back out — data that genuinely round-tripped
through the database, not a printed illusion.

Nothing about the model got better between the failure and the success. The
*discipline* did. Building software, it turns out, is the same superpower as summing a
ledger: act by producing something you can check. The verifier just changes shape — a
recomputed total, a passing test, a database that returns what you put into it.

---

## 8. What this means

Step back from the numbers and look at the shape.

The industry's answer to "I need more capability" is *spend more* — a larger model, a
longer window, more cloud. This experiment is an existence proof that there is another
lever entirely. A 27B model on a laptop, with no network and at no cost, did exact
work over data **fourteen times larger than its context window**, reliably, where the
same model reading the same data was wrong or could not start. The capability did not
come from scale. It came from **architecture** — from giving the model a REPL and the
instinct to write code instead of read.

That has consequences that are easy to state and hard to overstate:

- **Capability decouples from model size.** The thing you were going to buy a frontier
  model for, a small one can do — if you change how it works on the data.
- **It decouples from the cloud.** This ran on-device. Your data never left the
  machine. For anyone with privacy, latency, sovereignty, or cost constraints — which
  is to say, most serious deployments — that is not a footnote, it is the whole game.
- **It decouples from spend.** Zero dollars per run. The marginal cost of being exact
  over a million tokens of your own data, on your own hardware, is electricity.

And underneath the economics sits a prize that matters more than any of them.
**An agent whose default is to act through checkable procedures is an agent whose every
answer arrives with a procedure you can audit and re-run.** Not "trust me" — but "here
is the code, here is the source, run it yourself." Groundedness by construction. For
anything regulated, high-stakes, or simply important, that is the difference between a
plausible guess and a result you can stand behind, and it falls out of the posture for
free.

None of this says big models don't matter. It says the *default* is wrong. The default
should not be "throw a bigger model at it." The default should be: **try to answer with
code you can verify; decompose the problem; check the result; and fall back to the
model's raw, stochastic generation only for the part that nothing checkable can
express.** Hold the data out of the model's head and give it a place to compute. Reach
for scale when you've earned the need, not before. The model is not your oracle. It is
your programmer.

There is another way. We didn't argue it. We measured it — 55 controlled tests with the
answers checked, plus a working application built from scratch — on a laptop, for free,
and then wrote down exactly how, so you can do it too.

The model didn't get smarter. We gave it a better way to think. Start there.

---

*The full proof, raw data, preserved traces, and a one-command reproduction live in
[`experiments/superpowers/`](experiments/superpowers/) — including
[`PROOF.md`](experiments/superpowers/PROOF.md) with every table, the verbatim errors,
the code the model wrote, and the caveats stated plainly. `rrlm` itself is in this
repository: an RLM-first backend you can wire into your own agent today.*
