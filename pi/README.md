# rrlm as a Pi backend

This wires the RLM-first harness into [pi](https://github.com/earendil-works/pi)
as a delegatable tool, so a pi agent can offload data-heavy subtasks to the
recursive-language-model harness instead of pulling large data into its own
context.

## Pieces

- `extensions/rlm-backend/index.ts` -- registers the `rlm_solve` tool. It stages
  the data to a temp file and shells out to `python -m rrlm.solve --json` in the
  rrlm project, returning the verified answer plus usage metrics.
- `skills/rlm-first/SKILL.md` -- teaches the agent *when* to delegate: large
  data, exact aggregation/search over many items, or per-item semantic judgment
  at scale; and when not to (small data it can just read).

## The backend entry point

`rrlm.solve` is a general (instruction, data) -> answer entry point, independent
of the benchmark runner:

```bash
# inline / file / stdin
python -m rrlm.solve -i "Which product id has the most negative reviews?" -d @reviews.txt
echo "<data>" | python -m rrlm.solve -i "..." -d -
python -m rrlm.solve -i "..." -d @data.txt --json   # full result incl. metrics
```

Settled defaults (from the experiment): official Qwen3.6-27B orchestrator
(LM Studio) with `reasoning=off`, cheap `supergemma-26b` leaf, sandbox exec
timeout auto-scaled for serial local leaves. Override with `--main-model`,
`--sub-model`, `--backend {jspi,sbx}`.

## Install into pi

Prerequisites: the local models served (`~/workspace/serve-models.sh` +
LM Studio serving `qwen/qwen3.6-27b` on :1234), and the rrlm venv (`uv sync`).

Point pi at the extension and skill. Easiest is per-invocation:

```bash
RRLM_DIR=$HOME/workspace/rrlm OPENAI_API_KEY=lm-studio \
  pi -e $HOME/workspace/rrlm/pi/extensions/rlm-backend/index.ts \
     --skill $HOME/workspace/rrlm/pi/skills/rlm-first \
     --model lmstudio/qwen/qwen3.6-27b
```

Or make it permanent by adding the extension path to `settings.json`
`extensions` and the skill dir to `skills`.

Environment knobs read by the extension:

| Var | Default | Meaning |
|-----|---------|---------|
| `RRLM_DIR` | repo root (resolved from the extension path) | rrlm project dir for `uv run` |
| `RRLM_MAIN` | `qwen3.6-27b-pitune-local` | orchestrator model key |
| `RRLM_SUB` | `supergemma-26b-local` | leaf model key |
| `RRLM_BACKEND` | `jspi` | sandbox backend (`jspi` or `sbx`) |

## Verified

End-to-end: pi (driven by `lmstudio/qwen/qwen3.6-27b`) calls `rlm_solve`, the
harness runs, and the answer returns -- confirmed via json-mode
`tool_execution_start`/`tool_execution_end` events. The extension's `execute`
signature is version-robust (detects the AbortSignal by shape) across pi 0.73-0.77.
