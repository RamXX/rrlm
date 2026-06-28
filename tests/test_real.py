"""Real-use-case evals as pytest tests (excluded from the default + coverage runs).

These need a live model (and, for the Pi test, `pi` + `rrlm-solve` + Deno), so they
are all marked ``real`` and skipped by ``pytest -m "not real"``. The offline,
stub-backed integration / e2e coverage lives in ``test_integration_solve.py`` and
``test_e2e_cli.py`` (markers ``integration`` / ``e2e``), which DO run by default.

    pytest -m real          # tabular + code reasoning + Pi delegation (needs a model)
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


@pytest.mark.real
def test_pi_end_to_end_delegation():
    import eval_pi

    rc = eval_pi.main()
    if rc == 2:
        pytest.skip("pi not installed")
    assert rc == 0
