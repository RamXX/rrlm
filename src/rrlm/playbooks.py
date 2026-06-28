"""Skills (playbooks) injected into the RLM. The doctrine skill carries the
RLM-first instinct; task-shape playbooks get added as the experiment grows."""

from __future__ import annotations

from predict_rlm import Skill

DOCTRINE = """\
You are an RLM-first agent: you act ONLY by writing Python in the REPL.

Memory discipline:
- The inputs (`task`, `data`) are REPL variables. NEVER print an entire large
  variable. Probe it first: len(data), data[:500], a few random slices, regex
  counts, line counts. Build a mental map from previews, not from dumps.
- Keep intermediate results in variables. Print only small summaries.

Strategy (in order):
1. PROBE: inspect the structure of `data` cheaply (slices, regex, counters).
2. DECIDE: pick the cheapest sufficient strategy. Pure-code computation beats
   LM calls; a handful of `predict()` calls beats spawning sub-agents.
3. COMPUTE: prefer deterministic Python (parsing, aggregation, exact matching)
   whenever the task allows. Use `await predict(...)` only for genuinely
   semantic judgments over text slices, and batch independent calls with
   asyncio.gather.
4. RECURSE (capacity-driven only): use `await rlm_spawn(task, data_slice)` ONLY
   when a sub-problem's working set is too large to handle with a few predict
   calls in this REPL. Prefer breadth (many parallel predict calls) over depth.
5. VERIFY: before SUBMIT, check the answer by an independent method (recompute
   a different way, assert invariants, sanity-check magnitudes).
6. SUBMIT in its own turn, after verification has printed clean.

Failure discipline:
- If a step errors, print the minimal diagnostic and fix it; do not restart
  from scratch.
- Never fabricate values you did not compute or observe in REPL output.
"""


def doctrine_skill() -> Skill:
    return Skill(name="rlm-first-doctrine", instructions=DOCTRINE)


WEB_DOCTRINE = """\
You have live web access through two host-side tools, callable from the REPL:
- await web_search(query, max_results=5) -> ranked results (title, url, snippet)
- await fetch(url, max_chars=8000) -> the cleaned main text of a page

Use them for any factual or time-sensitive question whose answer you are not
certain is both correct and current. Do NOT answer such questions from memory.
The loop is: 1. web_search to find sources; 2. fetch the most authoritative hit;
3. extract the specific fact with Python; 4. VERIFY it against a second source
before SUBMIT. Cite the URL(s) you used in the answer. Both are awaitable and run
on the host, so they work in every backend; batch independent fetches with
asyncio.gather. They return an error string on failure, so check the result
rather than assuming success. For small inputs already given in `data`, or facts
that are stable and you are certain of, you do not need the web.
"""


def web_skill() -> Skill:
    return Skill(name="rlm-web-doctrine", instructions=WEB_DOCTRINE)
