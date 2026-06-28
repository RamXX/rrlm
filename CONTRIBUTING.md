# Contributing to rrlm

Thanks for your interest. rrlm is small on purpose; keep changes narrow and atomic.

## Setup

```bash
uv sync
make test      # offline suite (no network, no Deno)
make lint      # ruff
```

## Conventions

- Write tests alongside code. Unit tests may mock; the real-use-case evals
  (`examples/`, `tests/test_real.py`) run against live models and are excluded from
  the default `make test`.
- Keep the public surface small: the product is `rrlm.solve` + the Pi backend in
  `pi/`. The benchmark harness lives under `src/rrlm/bench/` and must not leak into
  the product path.
- Models are always resolved from Pi config via `rrlm.pi_config`: never hardcode a
  model registry, endpoint, or absolute path.
- No emojis in code, commits, or docs.

## Running the evals locally

```bash
make eval-tabular            # exact aggregation over a large CSV
make eval-bugfind            # code reasoning over this repository
make eval-pi                 # end-to-end Pi delegation (needs pi + rrlm-solve)
# choose models with: RRLM_MAIN=provider/model RRLM_SUB=provider/model make eval-tabular
```

## Working against a local predict-rlm checkout

rrlm depends on `predict-rlm` from PyPI. If you are hacking on both,
point the dependency at a local checkout **without editing `pyproject.toml`**:

```bash
uv pip install -e ../predict-rlm
```

rrlm feature-detects optional `predict-rlm` constructor params (e.g.
`max_action_generation_retries`), so it runs against both stock and patched builds.

## Pull requests

- Start the description with a short rationale (*why*, not just *what*).
- Ensure `make test` and `make lint` pass. To run the full portable gate the way
  CI does (in a container), use `make ci` (needs the Dagger CLI + Docker; see
  [docs/CI.md](docs/CI.md)).
- One logical change per PR.
