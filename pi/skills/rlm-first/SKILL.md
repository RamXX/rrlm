---
name: rlm-first
description: >-
  Decide when to delegate a data-heavy subtask to the rlm_solve tool (the
  RLM-first harness) instead of reading data into your own context. Use when
  a task involves a large data payload, exact aggregation or exhaustive search
  over many items, or per-item semantic judgment at scale, e.g. "which
  product has the most negative reviews across 5000 reviews", "sum the ok
  transactions for user X in this 50k-line ledger", "find the one buggy
  function in this module". Do NOT use it for small data you can simply read.
---

# RLM-first delegation

You have an `rlm_solve` tool backed by a recursive-language-model harness. It
loads a data payload into a sandboxed REPL, **not your context**, writes
code to probe it, fans out cheap sub-model calls only for irreducible semantic
judgment, verifies, and returns an answer. This lets you handle data far larger
than your context window, and stay correct on exact computation that
free-form reading gets wrong.

## When to delegate to `rlm_solve`

Delegate when ANY of these hold:

- **Data exceeds (or strains) your context**, a file or blob too large to
  read comfortably. The harness's cost and reliability are flat in data size.
- **Exactness over many items**, counting, summing, exhaustive search, or
  "find the one X among N". Reading-and-reasoning silently miscounts at scale;
  code in the REPL does not.
- **Per-item semantic judgment at scale**, e.g. classify N free-text items
  then aggregate. The harness fans out cheap sub-model calls and aggregates
  mechanically.

Pass the FULL data (inline via `data`, or `data_path` for a file on disk).
Never pre-summarize or truncate it, defeating the purpose. Make the
`instruction` specific and answerable from the data alone.

## When NOT to delegate

- **Small data you can just read**, if it fits comfortably in context and the
  task is a direct read or a single judgment, read it yourself. The harness has
  fixed scaffold overhead (~15-25k tokens, several REPL turns) that is not worth
  paying for small inputs.
- **No data payload**, pure reasoning, code authoring, or conversation. Handle
  it directly. Exception: if the `rlm_solve` tool description says web access is
  enabled, delegate factual or current-events questions you cannot answer with
  certainty (or that need a cited source URL) to it, even with no data, it will
  search the live web, fetch the source, and verify.
- **You need the raw data in YOUR context afterward**, `rlm_solve` returns an
  answer, not the loaded data. But if what you need is *follow-up questions
  over the same data*, use session mode (below) instead of re-sending the
  payload each time.

## Follow-up questions: session mode

When you expect more than one question over the same data, set
`session: true` on every related call. The harness then keeps ONE persistent
REPL across your calls: variables, parsed structures, and helpers from one
call are available to the next, so the data loads once and later questions
are fast and cheap.

Work the session like this:

1. First call: pass the data and tell the harness what to KEEP, by name,
   e.g. "Parse the ledger into `entries`; report the row count."
2. Later calls: omit the data; reference the names, e.g. "Using `entries`,
   total the amounts per vendor."
3. Start a genuinely new topic on the same session with
   `reset_session: true` (clears the persisted state first).

What persists is computed state, not your conversation: each call is still
its own run with fresh budgets, and only the variables you asked to keep
carry over. Switching Pi to a different model restarts the session (state is
lost), so finish a session's work before changing models.

## Routing rule (one line)

If the answer requires touching a lot of data, or being exact over many items,
or judging many items semantically -> `rlm_solve`; if more questions on the
same data will follow -> add `session: true`. Otherwise, do it yourself.

## Example calls

- `rlm_solve(instruction="Which product id has the highest fraction of negative
  reviews? Answer with the id only.", data_path="/abs/reviews.txt")`
- `rlm_solve(instruction="Total amount of status=ok transactions for user u573,
  rounded to 2 decimals.", data_path="/abs/ledger.txt")`
- `rlm_solve(instruction="Name the one function with a bug.", data=<module source>)`
- Session over one payload:
  `rlm_solve(instruction="Parse the ledger into `entries`; report the row count.",
  data_path="/abs/ledger.txt", session=true)` then
  `rlm_solve(instruction="Using `entries`, total the amounts per vendor.",
  session=true)`
