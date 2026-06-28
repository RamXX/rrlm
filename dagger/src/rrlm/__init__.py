"""Portable CI for rrlm as a Dagger module.

Exposes lint, test, cov, and ci functions that mirror the repository's make
contract, so CI runs the same offline suite anywhere a container runtime is
available. See main.py for details.
"""

from .main import Rrlm as Rrlm
