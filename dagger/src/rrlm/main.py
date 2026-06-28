"""Portable CI for rrlm, expressed as a Dagger module.

The pipeline mirrors the repository's make contract exactly so `dagger call ci`
runs the same offline suite a developer runs with `make cov`:

  * lint  -> ruff check over src, tests, examples
  * test  -> pytest -m "not real" (offline unit + integration + e2e)
  * cov   -> the same suite with the 80% coverage gate
  * ci    -> the default gate: lint, then cov

It needs only a container runtime (Docker here). Network is used once per fresh
cache to `uv sync`; the test suite itself runs fully offline. Any CI provider
runs this by invoking `dagger call ci`, which is the point of keeping it here
rather than in a provider-specific workflow file.
"""

from typing import Annotated

import dagger
from dagger import DefaultPath, Ignore, dag, function, object_type

# uv-enabled Python 3.13 base. Pinned major.minor matches .python-version.
UV_IMAGE = "ghcr.io/astral-sh/uv:python3.13-bookworm-slim"

# Paths that never affect lint, tests, or coverage and would only bloat the
# upload to the engine. Source, tests, examples, pyproject, and uv.lock stay.
IGNORE = [
    ".git",
    ".venv",
    "dagger/sdk",
    "**/__pycache__",
    "**/*.pyc",
    ".pytest_cache",
    ".ruff_cache",
    "runs",
    "logs",
    "data",
    "dist",
    "htmlcov",
    ".coverage*",
    "*.egg-info",
]


@object_type
class Rrlm:
    """rrlm CI pipeline."""

    source: Annotated[dagger.Directory, DefaultPath("/"), Ignore(IGNORE)]

    def _env(self) -> dagger.Container:
        """uv-enabled Python 3.13 container with the project synced.

        `uv sync` installs the dependency groups and the project itself, so the
        e2e tests can invoke the `rrlm-solve` console script as a subprocess.
        The uv cache lives on a Dagger cache volume so repeat runs skip the
        download work; only a cold cache touches the network.
        """
        uv_cache = dag.cache_volume("rrlm-uv-cache")
        return (
            dag.container()
            .from_(UV_IMAGE)
            .with_env_variable("UV_CACHE_DIR", "/root/.cache/uv")
            .with_mounted_cache("/root/.cache/uv", uv_cache)
            # Copy out of the cache mount instead of hardlinking across it.
            .with_env_variable("UV_LINK_MODE", "copy")
            .with_workdir("/src")
            .with_directory("/src", self.source)
            .with_exec(["uv", "sync", "--frozen"])
        )

    @function
    async def lint(self) -> str:
        """Run ruff over src, tests, and examples (matches `make lint`)."""
        return await (
            self._env()
            .with_exec(["uv", "run", "ruff", "check", "src/", "tests/", "examples/"])
            .stdout()
        )

    @function
    async def test(self) -> str:
        """Run the offline suite, excluding live-model `real` tests (matches `make test`)."""
        return await (
            self._env()
            .with_exec(["uv", "run", "-p", "3.13", "--", "pytest", "-q", "-m", "not real"])
            .stdout()
        )

    @function
    async def cov(self) -> str:
        """Offline suite with coverage, failing under 80% (matches `make cov`)."""
        return await (
            self._env()
            .with_exec(
                [
                    "uv",
                    "run",
                    "-p",
                    "3.13",
                    "--",
                    "pytest",
                    "-q",
                    "-m",
                    "not real",
                    "--cov=src/rrlm",
                    "--cov-report=term-missing",
                    "--cov-fail-under=80",
                ]
            )
            .stdout()
        )

    @function
    async def ci(self) -> str:
        """Default gate: lint, then the coverage-gated offline suite.

        Either step exiting non-zero (lint findings, test failures, or coverage
        under 80%) fails the call.
        """
        lint_out = await self.lint()
        cov_out = await self.cov()
        return f"=== lint ===\n{lint_out}\n=== cov ===\n{cov_out}"
