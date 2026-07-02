#!/usr/bin/env bash
# rrlm installer: clone the repo, set up the virtualenv, and install the CLIs with uv.
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/RamXX/rrlm/main/install.sh | bash
#
# Environment overrides:
#   RRLM_HOME       install / clone directory (default: ~/.rrlm)
#   RRLM_REF        git ref to check out      (default: main)
#   RRLM_REPO_URL   repository URL            (default: https://github.com/RamXX/rrlm)
set -euo pipefail

REPO_URL="${RRLM_REPO_URL:-https://github.com/RamXX/rrlm}"
RRLM_HOME="${RRLM_HOME:-$HOME/.rrlm}"
RRLM_REF="${RRLM_REF:-main}"

note() { printf '\033[1m[rrlm]\033[0m %s\n' "$*"; }
die()  { printf '\033[1m[rrlm] error:\033[0m %s\n' "$*" >&2; exit 1; }

command -v git >/dev/null 2>&1 || die "git is required; install it and re-run."

# Ensure uv is available (bootstrap it if missing).
if ! command -v uv >/dev/null 2>&1; then
  note "uv not found; installing it from https://astral.sh/uv ..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # uv installs into ~/.local/bin by default; make it visible for this run.
  export PATH="$HOME/.local/bin:$PATH"
  command -v uv >/dev/null 2>&1 || die "uv install failed; see https://docs.astral.sh/uv/"
fi

# Clone fresh, or update an existing checkout.
if [ -d "$RRLM_HOME/.git" ]; then
  note "updating existing checkout at $RRLM_HOME"
  git -C "$RRLM_HOME" fetch --quiet origin "$RRLM_REF"
  git -C "$RRLM_HOME" checkout --quiet "$RRLM_REF"
  git -C "$RRLM_HOME" pull --quiet --ff-only origin "$RRLM_REF" || true
else
  note "cloning $REPO_URL into $RRLM_HOME"
  git clone --quiet --branch "$RRLM_REF" "$REPO_URL" "$RRLM_HOME"
fi

cd "$RRLM_HOME"

note "syncing dependencies (uv sync)"
uv sync --quiet

note "installing the rrlm-solve, rrlm-traces, and rrlm-doctor CLIs on your PATH"
uv tool install --force .

# Deno is needed only for the opt-in jspi (Pyodide) sandbox; warn, do not fail.
if ! command -v deno >/dev/null 2>&1; then
  note "Deno is not installed. It is only needed for the opt-in 'jspi' sandbox backend"
  note "(the default 'supervisor' backend needs no Deno)."
  note "install it later with: curl -fsSL https://deno.land/install.sh | sh"
fi

note "done. Try:  rrlm-doctor   (checks your setup), then:  rrlm-solve --help"
note "checkout lives at: $RRLM_HOME"
