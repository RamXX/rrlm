"""rrlm: an RLM-first backend for the Pi coding agent.

The public entry point is :func:`rrlm.solve.solve`, give it an instruction and
a (possibly large) data payload and it runs the recursive-language-model harness,
returning a verified answer plus usage metrics. Models are resolved from your Pi
config (see :mod:`rrlm.pi_config`).
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from rrlm.solve import solve

try:
    __version__ = version("rrlm")
except PackageNotFoundError:  # running from a source tree without an install
    __version__ = "0.0.0+dev"

__all__ = ["solve", "__version__"]
