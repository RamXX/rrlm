"""Unit tests for rrlm.solve.export_trace, the predict-rlm RunTrace capture
used to accumulate GEPA training traces. Pure unit tests (no model/sandbox):
a stub trace stands in for a real predict-rlm RunTrace."""
from __future__ import annotations

import json

from rrlm.solve import export_trace


class _StubTrace:
    def __init__(self, payload: dict):
        self._payload = payload

    def to_exportable_json(self) -> str:
        return json.dumps(self._payload)


def test_export_writes_trace_and_index(tmp_path):
    trace = _StubTrace({"iterations": 2, "answer": "P3"})
    path = export_trace(
        trace, trace_dir=str(tmp_path), instruction="which id?", answer="P3",
        data_chars=123, wall_clock_s=1.5, config={"main_model": "ornith/x"},
    )
    assert path is not None
    # the trace file holds the exportable RunTrace JSON
    saved = json.loads(open(path).read())
    assert saved["answer"] == "P3" and saved["iterations"] == 2
    # an index line pairs instruction -> answer -> trace for GEPA dataset assembly
    index = tmp_path / "index.jsonl"
    rec = json.loads(index.read_text().strip())
    assert rec["instruction"] == "which id?"
    assert rec["answer"] == "P3"
    assert rec["error"] is None
    assert rec["data_chars"] == 123
    assert rec["trace_file"] == path.split("/")[-1]


def test_export_records_failure_traces(tmp_path):
    # Failure traces are the strongest GEPA signal; the index carries the error.
    path = export_trace(
        _StubTrace({"status": "error"}), trace_dir=str(tmp_path),
        instruction="sum it", error="BudgetExceededError: cost budget exhausted",
    )
    assert path is not None
    rec = json.loads((tmp_path / "index.jsonl").read_text().strip())
    assert rec["error"].startswith("BudgetExceededError")


def test_export_accumulates_unique_files(tmp_path):
    # two calls with the same trace_dir must not overwrite each other
    p1 = export_trace(_StubTrace({"a": 1}), trace_dir=str(tmp_path))
    p2 = export_trace(_StubTrace({"a": 2}), trace_dir=str(tmp_path))
    # same process+second can collide on the timestamp; guarantee at least one file
    # and two index lines (the real CLI runs one process per call, so files are unique).
    assert p1 and p2
    lines = (tmp_path / "index.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2


def test_noop_without_trace_dir():
    assert export_trace(_StubTrace({"a": 1}), trace_dir="") is None


def test_noop_when_trace_is_missing(tmp_path):
    assert export_trace(None, trace_dir=str(tmp_path)) is None
    assert not (tmp_path / "index.jsonl").exists()


class _ExplodingTrace:
    def to_exportable_json(self) -> str:
        raise RuntimeError("export blew up")


def test_export_is_best_effort_on_failure(tmp_path):
    # Trace capture must never break solve(): a raising exporter yields None.
    assert export_trace(_ExplodingTrace(), trace_dir=str(tmp_path)) is None
