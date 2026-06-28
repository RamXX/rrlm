"""Tests for the benchmark runner and report.

The report tests are pure unit tests over fabricated run artifacts. The runner
tests are integration tests: ``run_task`` and the runner ``main`` are exercised
end to end against the real offline stub server (no mocks), with the output
directory redirected to a temp path so nothing touches the repo ``runs/`` tree.
"""

from __future__ import annotations

import json
import sys

import pytest

from rrlm.bench import report, runner
from rrlm.bench.tasks import make_ledger_task
from rrlm.config import HarnessConfig
from rrlm.pi_config import resolve_model


# --------------------------------------------------------------------------- #
# report.py (unit)
# --------------------------------------------------------------------------- #
def _write_run(runs_dir, run_id="run-1", *, passed=True, condition="rlm"):
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(json.dumps({
        "run_id": run_id, "task_id": "ledger-20", "model": "stub/stub-model",
        "condition": condition,
    }))
    (run_dir / "result.json").write_text(json.dumps({
        "run_id": run_id, "passed": passed, "status": "completed", "wall_clock_s": 1.2,
        "usage": {
            "calls": 3, "prompt_tokens": 100, "completion_tokens": 20,
            "cost_usd": 0.01, "generation_time_ms": 3400,
        },
    }))
    return run_dir


def test_report_collect_and_print_table(tmp_path, capsys):
    _write_run(tmp_path, "run-a", passed=True)
    _write_run(tmp_path, "run-b", passed=False, condition="baseline")
    # a stray dir without artifacts must be skipped, not crash collect()
    (tmp_path / "incomplete").mkdir()
    rows = report.collect(tmp_path)
    assert len(rows) == 2
    assert {r["condition"] for r in rows} == {"rlm", "baseline"}
    report.print_table(rows)
    out = capsys.readouterr().out
    assert "run-a" in out and "run-b" in out and "cost_usd" in out


def test_report_print_table_empty(capsys):
    report.print_table([])
    assert "no runs found" in capsys.readouterr().out


def test_report_main_writes_csv(tmp_path, monkeypatch, capsys):
    _write_run(tmp_path, "run-a")
    monkeypatch.setattr(report, "RUNS_DIR", tmp_path)
    csv_path = tmp_path / "out.csv"
    monkeypatch.setattr(sys, "argv", ["rrlm-report", "--csv", str(csv_path)])
    report.main()
    body = csv_path.read_text()
    assert "run_id" in body and "run-a" in body


# --------------------------------------------------------------------------- #
# runner.py helpers (unit)
# --------------------------------------------------------------------------- #
def test_runner_redacts_api_key(stub_model):
    model = resolve_model(stub_model("submit"))
    red = runner._redacted(model)
    assert red["api_key"] == "<redacted>"
    assert red["litellm_id"] == model.litellm_id


def test_runner_versions_reports_known_packages():
    versions = runner._versions()
    assert set(versions) == {"dspy", "predict-rlm", "litellm"}
    assert all(isinstance(v, str) for v in versions.values())


# --------------------------------------------------------------------------- #
# runner.run_task / main (integration, against the stub)
# --------------------------------------------------------------------------- #
@pytest.mark.integration
def test_run_task_rlm_condition_against_stub(stub_model, tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "RUNS_DIR", tmp_path)
    model = resolve_model(stub_model("submit"))
    cfg = HarnessConfig(main_model=model.ref, sub_model=model.ref,
                        backend="supervisor", max_iterations=5, reasoning="off")
    task = make_ledger_task(size=20, seed=1)
    result = runner.run_task(task, model, model, "rlm", cfg)
    # reasoning != default flows into the run_id variant suffix
    assert "-roff" in result["run_id"]
    assert result["status"] == "completed"
    # the stub computes len(data), not the ledger total, so the check fails --
    # what matters here is that the full run completed and wrote its artifacts.
    assert result["answer"] == str(len(task.data))
    run_dir = tmp_path / result["run_id"]
    assert (run_dir / "run.json").exists()
    assert (run_dir / "result.json").exists()
    assert (run_dir / "trace.json").exists()


@pytest.mark.integration
def test_run_task_baseline_condition_against_stub(stub_model, tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "RUNS_DIR", tmp_path)
    model = resolve_model(stub_model("submit"))
    cfg = HarnessConfig(main_model=model.ref, sub_model=model.ref, backend="supervisor")
    task = make_ledger_task(size=20, seed=1)
    result = runner.run_task(task, model, model, "baseline", cfg)
    assert result["status"] == "completed"
    # baseline asks the stub directly for the `answer` field.
    assert result["answer"] == "extracted-fallback-answer"
    events = (tmp_path / result["run_id"] / "events.jsonl").read_text().splitlines()
    rec = json.loads(events[0])
    assert rec["role"] == "baseline"


@pytest.mark.integration
def test_run_task_unknown_condition_records_error(stub_model, tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "RUNS_DIR", tmp_path)
    model = resolve_model(stub_model("submit"))
    cfg = HarnessConfig(main_model=model.ref, sub_model=model.ref, backend="supervisor")
    task = make_ledger_task(size=20, seed=1)
    result = runner.run_task(task, model, model, "nonsense", cfg)
    assert result["status"] == "error"
    assert "unknown condition" in (result["error"] or "")


@pytest.mark.integration
def test_runner_main_against_stub(stub_model, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(runner, "RUNS_DIR", tmp_path)
    model = stub_model("submit")
    monkeypatch.setattr(sys, "argv", [
        "rrlm-bench", "--task", "ledger", "--size", "20", "--condition", "rlm",
        "--model", model, "--backend", "supervisor", "--max-iterations", "5",
    ])
    runner.main()
    out = capsys.readouterr().out
    assert "passed=" in out and "wall=" in out
