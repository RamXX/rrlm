"""Real-use-case evals as pytest tests (excluded from the CI unit run).

These need a configured model (and, for the Pi test, `pi` + `rrlm-solve`), so they
are marked and skipped by ``pytest -m "not real and not integration"``.

    pytest -m real          # tabular aggregation + code reasoning
    pytest -m integration   # end-to-end Pi delegation
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
sys.path.insert(0, str(EXAMPLES))


@pytest.mark.real
def test_tabular_exact_aggregation():
    import eval_tabular

    assert eval_tabular.main() == 0


@pytest.mark.real
def test_bugfind_code_reasoning():
    import eval_bugfind

    assert eval_bugfind.main() == 0


@pytest.mark.integration
def test_pi_end_to_end_delegation():
    import eval_pi

    rc = eval_pi.main()
    if rc == 2:
        pytest.skip("pi not installed")
    assert rc == 0
