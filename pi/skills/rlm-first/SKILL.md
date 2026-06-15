---
name: rlm-first
description: >-
  Decide when to delegate a data-heavy subtask to the rlm_solve tool (the
  RLM-first harness) instead of reading data into your own context. Use when
  a task involves a large data payload, exact aggregation or exhaustive search
  over many items, or per-item semantic judgment at scale -- e.g. "which
  product has the most negative reviews across 5000 reviews", "sum the ok
  transactions for user X in this 50k-line ledger", "find the one buggy
  function in this module". Do NOT use it for small data you can simply read.
---

# RLM-first delegation

You have an `rlm_solve` tool backed by a recursive-language-model harness. It
loads a data payload into a sandboxed REPL -- **not your context** -- writes
code to probe it, fans out cheap sub-model calls only for irreducible semantic
judgment, verifies, and returns an answer. This lets you handle data far larger
than your context window, and stay correct on exact computation that
free-form reading gets wrong.

## When to delegate to `rlm_solve`

Delegate when ANY of these hold:

- **Data exceeds (or strains) your context** -- a file or blob too large to
  read comfortably. The harness's cost and reliability are flat in data size.
- **Exactness over many items** -- counting, summing, exhaustive search, or
  "find the one X among N". Reading-and-reasoning silently miscounts at scale;
  code in the REPL does not.
- **Per-item semantic judgment at scale** -- e.g. classify N free-text items
  then aggregate. The harness fans out cheap sub-model calls and aggregates
  mechanically.

Pass the FULL data (inline via `data`, or `data_path` for a file on disk).
Never pre-summarize or truncate it -- defeating the purpose. Make the
`instruction` specific and answerable from the data alone.

## When NOT to delegate

- **Small data you can just read** -- if it fits comfortably in context and the
  task is a direct read or a single judgment, read it yourself. The harness has
  fixed scaffold overhead (~15-25k tokens, several REPL turns) that is not worth
  paying for small inputs.
- **No data payload** -- pure reasoning, code authoring, or conversation. Handle
  it directly.
- **You need to keep reasoning over the data afterward** -- `rlm_solve` returns
  an answer, not the loaded data. If you need iterative back-and-forth over the
  same large payload, call it once with a precise instruction.

## Routing rule (one line)

If the answer requires touching a lot of data, or being exact over many items,
or judging many items semantically -> `rlm_solve`. Otherwise, do it yourself.

## Example calls

- `rlm_solve(instruction="Which product id has the highest fraction of negative
  reviews? Answer with the id only.", data_path="/abs/reviews.txt")`
- `rlm_solve(instruction="Total amount of status=ok transactions for user u573,
  rounded to 2 decimals.", data_path="/abs/ledger.txt")`
- `rlm_solve(instruction="Name the one function with a bug.", data=<module source>)`
