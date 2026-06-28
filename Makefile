PYTHON_VERSION := $(shell cat .python-version)
RUN := uv run -p $(PYTHON_VERSION) --

# --- Models -----------------------------------------------------------------
# rrlm resolves models from your Pi config (see rrlm.pi_config). MAIN/SUB are Pi
# model references (provider/model or a bare id). Two contexts, kept distinct on purpose:
#   * DATA benchmarks/evals (below): default to a CLOUD pair (Qwen3.6-27B + gemma-4-26b
#     via OpenRouter, needs OPENROUTER_API_KEY) so anyone can run them without a GPU.
#   * The LOCAL flagship orchestrator is Ornith-1.0-35B (see serve-orch / CRM_MODEL and
#     docs/LOCAL_SERVING.md). Override MAIN to run the evals locally too, e.g.
#     MAIN=ornith/ornith-1.0-35b SUB=supergemma/... make eval-tabular.
MAIN ?= openrouter/qwen/qwen3.6-27b
SUB  ?= openrouter/google/gemma-4-26b-a4b-it
# Orchestrator for the code-generation example (make eval-crm); the local flagship.
CRM_MODEL ?= ornith/ornith-1.0-35b
REASONING ?= off
SEED ?= 42
SIZE ?= 2000

# Pass-through args for the benchmark runner.
RUNNER_ARGS = --model $(MAIN) --sub-model $(SUB) --reasoning $(REASONING) --seed $(SEED)

# `make solve` entry point: override INSTRUCTION and DATA (literal, @file, or - for stdin).
INSTRUCTION ?= Which product id has the most negative reviews? Answer with the id only.
DATA ?= -

.PHONY: setup test lint smoke baseline compare report clean-runs \
        solve pi-run eval-tabular eval-bugfind eval-pi eval-crm \
        serve-orch serve-leaf serve-stop serve-purge

setup:
	uv sync

test:
	$(RUN) pytest -q -m "not real and not integration"

lint:
	$(RUN) ruff check src/ tests/ examples/

# --- Benchmarks (the research side; see docs/FINDINGS.md) --------------------
# One RLM-condition run on the synthetic ledger task.
smoke:
	$(RUN) rrlm-bench --task ledger --condition rlm --size $(SIZE) $(RUNNER_ARGS)

# Context-stuffed baseline on the same task.
baseline:
	$(RUN) rrlm-bench --task ledger --condition baseline --size $(SIZE) $(RUNNER_ARGS)

# Both conditions back to back, then the comparison table.
compare: smoke baseline report

report:
	$(RUN) rrlm-report

clean-runs:
	rm -rf runs/

# --- The product: RLM-first solve backend Pi delegates to -------------------
solve:
	$(RUN) rrlm-solve -i "$(INSTRUCTION)" -d "$(DATA)" --main "$(MAIN)" --sub "$(SUB)"

# Launch pi with the rlm-backend extension + rlm-first skill (dev mode: the
# extension runs rrlm-solve via `uv run` inside this checkout).
pi-run:
	RRLM_DIR=$(CURDIR) pi \
		-e $(CURDIR)/pi/extensions/rlm-backend/index.ts \
		--skill $(CURDIR)/pi/skills/rlm-first

# --- Real-use-case evals (need a configured model; pi eval needs pi) ---------
# Three DATA evals (default to the cloud MAIN/SUB pair) plus the CODE-GENERATION
# example (eval-crm), the headline use: a local model builds a CRM. See examples/crm.
eval-tabular:
	$(RUN) python examples/eval_tabular.py

eval-bugfind:
	$(RUN) python examples/eval_bugfind.py

eval-pi:
	RRLM_DIR=$(CURDIR) $(RUN) python examples/eval_pi.py

# Code generation: a local model builds LadyCRM file-by-file (Phase 1 + full spec),
# timed + quality-gated + traces. Needs `make serve-orch` + `make serve-leaf` running.
# Override the orchestrator with CRM_MODEL=<provider/model>. Output in examples/crm/runs/.
eval-crm:
	cd examples/crm && rm -rf runs/build && cp -R template runs/build && \
	  RRLM_TRACE_DIR=runs/build/traces bash build_loop.sh $(CRM_MODEL) runs/build

# --- Optional local model servers (offline / $0 inference) ------------------
# See docs/LOCAL_SERVING.md. Run each in its own terminal.
# Orchestrator: Ornith-1.0-35B (Qwen3.5 MoE) on llama.cpp continuous batching -- the
# settled top model (Track C): correct on all RLM task types (full superpowers proof
# re-runs green on Ornith), ~5-8x faster end-to-end than a dense 27B, scales under
# parallel agents. The only orchestrator now; pair it with the supergemma leaf.
serve-orch:
	NOTHINK=1 ./scripts/local-serving/serve-ornith.sh

serve-leaf:
	./scripts/local-serving/serve-models.sh start

serve-stop:
	-./scripts/local-serving/serve-models.sh stop 2>/dev/null; \
	 pkill -f "serve-ornith.sh|llama-server.*8774" 2>/dev/null; \
	 ./scripts/local-serving/purge-dflash-cache.sh 2>/dev/null; \
	 echo "stopped local model servers"

# Manually drop the regenerable DFlash prefix cache (safe: skips if a server is up).
serve-purge:
	./scripts/local-serving/purge-dflash-cache.sh

# (The Track C DFlash-vs-MTP decode bake-off target was removed -- its Qwen/pi-tune
# weights were purged. The investigation record lives in experiments/dflash-vs-mtp/.)
