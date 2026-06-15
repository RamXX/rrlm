PYTHON_VERSION := $(shell cat .python-version)
RUN := uv run -p $(PYTHON_VERSION) --

# Current focus pair: local-class orchestrator + cheap non-thinking classifier.
# Orchestrator thinking OFF: measured 7x faster / equal accuracy on fan-out tasks.
SIZE ?= 2000
MODEL ?= qwen3.6-27b
SUB ?= gemma-4-26b
REASONING ?= off
SEED ?= 42
RUNNER_ARGS = --model $(MODEL) --sub-model $(SUB) --reasoning $(REASONING) --seed $(SEED)

# Pi backend: general (instruction, data) -> answer entry point.
# Override INSTRUCTION and DATA (literal, @file, or - for stdin).
INSTRUCTION ?= Which product id has the most negative reviews? Answer with the id only.
DATA ?= -

.PHONY: setup test smoke baseline compare report clean-runs solve pi-run

setup:
	uv sync

test:
	$(RUN) pytest -q

# One RLM-condition run on the synthetic ledger task
smoke:
	$(RUN) python -m rrlm.runner --task ledger --condition rlm --size $(SIZE) $(RUNNER_ARGS)

# Context-stuffed baseline on the same task
baseline:
	$(RUN) python -m rrlm.runner --task ledger --condition baseline --size $(SIZE) $(RUNNER_ARGS)

# Both conditions back to back, then the comparison table
compare: smoke baseline report

report:
	$(RUN) python -m rrlm.report

clean-runs:
	rm -rf runs/

# RLM-first solve backend (the capability Pi delegates to)
solve:
	$(RUN) python -m rrlm.solve -i "$(INSTRUCTION)" -d "$(DATA)" --main-model qwen3.6-27b-official-local --sub-model supergemma-26b-local

# Launch pi with the rlm-backend extension + rlm-first skill loaded
pi-run:
	RRLM_DIR=$(CURDIR) OPENAI_API_KEY=lm-studio pi \
		-e $(CURDIR)/pi/extensions/rlm-backend/index.ts \
		--skill $(CURDIR)/pi/skills/rlm-first \
		--model lmstudio/qwen/qwen3.6-27b
