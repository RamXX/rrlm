"""rrlm: an RLM-first backend for the Pi coding agent.

The public entry points are :func:`rrlm.solve.solve` (sync) and
:func:`rrlm.solve.asolve` (async): give them an instruction and a (possibly
large) data payload and they run the recursive-language-model harness,
returning a verified answer plus usage metrics. ``solve_many``/``asolve_many``
answer several questions over the same data in one amortized run. Models are
resolved from your Pi config (see :mod:`rrlm.pi_config`).
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from rrlm.solve import asolve, asolve_many, solve, solve_many

try:
    __version__ = version("rrlm")
except PackageNotFoundError:  # running from a source tree without an install
    __version__ = "0.0.0+dev"

__all__ = ["solve", "asolve", "solve_many", "asolve_many", "__version__"]
