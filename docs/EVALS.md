# Evaluation matrix and the generalization protocol

> rrlm docs: [README](../README.md) | [Design](DESIGN.md) | [CI](CI.md) |
> [Benchmarks and findings](FINDINGS.md) | [Local serving](LOCAL_SERVING.md) |
> Evals (this page) | [Pi integration](../pi/README.md)

[FINDINGS.md](FINDINGS.md) established the core claim (RLM beats
context-stuffing on cost and exactness as data grows) on a narrow base:
synthetic template data, one or two runs per cell, limited exercise of
`predict()` and `rlm_spawn`. This page is the plan for widening that base,
and the protocol that keeps doctrine optimization from silently overfitting
the task shapes it was trained on.

## The overfitting problem, and the protocol that detects it

The doctrine is optimizable text ([RLM-GEPA](../README.md#optimize-the-doctrine-with-rlm-gepa-opt-in)),
and an optimizer improves what it can measure: a doctrine evolved on tabular
aggregation examples may get better at tabular aggregation by getting worse
at everything else, and a same-domain validation split cannot see that.

The protocol (implemented today):

1. Tag every dataset example with a ``domain`` (its task family).
2. Hold whole domains out of optimization: with
   ``RRLM_GEPA_HOLDOUT_DOMAINS=code,web``, those examples never enter train
   *or* val - the doctrine never sees them in any form.
3. After optimization, score the winner on exactly the held-out domains::

       RRLM_GEPA_HOLDOUT_DOMAINS=code,web rrlm-gepa optimize --max-metric-calls 400
       RRLM_GEPA_HOLDOUT_DOMAINS=code,web rrlm-gepa eval --doctrine runs/<run>/winner.txt
       rrlm-gepa eval                       # baseline: the built-in doctrine, same examples

``rrlm-gepa eval`` scores through the same code path and the same machine
checkers the optimizer used (``score_example_with_doctrine``), so the
in-domain and held-out numbers are comparable by construction. The decision
rule: **a winner ships only if it beats the built-in doctrine both in-domain
and held-out.** In-domain gain with held-out regression is overfitting;
reject the winner or broaden the training domains.

Rotate the held-out set across optimization campaigns (hold out ``code,web``
this time, ``docs,tabular`` next) so every domain eventually serves as an
unseen probe.

## The target matrix

Rows are task families (the ``domain`` tags); columns are the execution
conditions worth crossing them with. Cells marked now exist as runnable
evals; the rest are the build-out order.

| Domain tag | Task family | Status |
| --- | --- | --- |
| `tabular` | exact aggregation over large tabular/ledger data | now (`make eval-tabular`, bench `ledger`) |
| `search` | needle-finding / exhaustive search | now (bench `needle`) |
| `semantic` | per-item judgment at scale (classify, extract, aggregate) | now (bench `reviews`) |
| `code` | reasoning over real repositories | now (`make eval-bugfind`) |
| `docs` | natural document collections (PDF / XLSX / DOCX, multi-file) | planned: real document sets, file mounting |
| `web` | live retrieval with citation (`--web`) | planned: pinned-answer questions |
| `mcp` | tool-mediated tasks over MCP servers | planned: recorded/stub servers |
| `adversarial` | malformed, misleading, or trap data | planned: injection-shaped and corrupt inputs |

Crossing conditions, applied per campaign rather than exhaustively: model
family (local Ornith vs hosted), backend (`supervisor` vs `jspi`/`sbx`), and
data scale (within-context vs beyond-context). Fill cells incrementally;
every cell is a JSONL file in the standard dataset format with ``domain``
tags, so the same `optimize`/`eval` tooling covers all of them.

## Costs and discipline

Every cell run spends real model calls; this is why the matrix fills
incrementally instead of landing at once. Rules that keep the numbers
meaningful:

* Held-out domains are stated *before* optimization starts, in the campaign's
  env, and never adjusted afterward.
* `rrlm-gepa eval` runs examples sequentially with the same budgets the
  optimizer used; do not raise budgets only for the report.
* Record both numbers (winner and built-in baseline) for every campaign;
  a single number proves nothing.
* Failed runs score 0.0 and stay in the mean - dropping them would reward
  doctrines that crash on hard examples.
