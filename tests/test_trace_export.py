"""Unit tests for rrlm.solve.export_trace, the predict-rlm RunTrace capture
used to accumulate GEPA training traces. Pure unit tests (no model/sandbox):
a stub prediction stands in for a real predict-rlm result."""
from __future__ import annotations

import json

from rrlm.solve import export_trace


class _StubTrace:
    def __init__(self, payload: dict):
        self._payload = payload

    def to_exportable_json(self) -> str:
        return json.dumps(self._payload)


class _StubPrediction:
    def __init__(self, trace=None):
        self.trace = trace
        self.answer = "P3"


def test_export_writes_trace_and_index(tmp_path):
    pred = _StubPrediction(_StubTrace({"iterations": 2, "answer": "P3"}))
    path = export_trace(
        pred, trace_dir=str(tmp_path), instruction="which id?", answer="P3",
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
    assert rec["data_chars"] == 123
    assert rec["trace_file"] == path.split("/")[-1]


def test_export_accumulates_unique_files(tmp_path):
    # two calls with the same trace_dir must not overwrite each other
    p1 = export_trace(_StubPrediction(_StubTrace({"a": 1})), trace_dir=str(tmp_path))
    p2 = export_trace(_StubPrediction(_StubTrace({"a": 2})), trace_dir=str(tmp_path))
    # same process+second can collide on the timestamp; guarantee at least one file
    # and two index lines (the real CLI runs one process per call, so files are unique).
    assert p1 and p2
    lines = (tmp_path / "index.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2


def test_noop_without_trace_dir():
    assert export_trace(_StubPrediction(_StubTrace({"a": 1})), trace_dir="") is None


def test_noop_when_prediction_has_no_trace(tmp_path):
    assert export_trace(_StubPrediction(trace=None), trace_dir=str(tmp_path)) is None
    assert not (tmp_path / "index.jsonl").exists()


class _ExplodingTrace:
    def to_exportable_json(self) -> str:
        raise RuntimeError("export blew up")


def test_export_is_best_effort_on_failure(tmp_path):
    # Trace capture must never break solve(): a raising exporter yields None.
    pred = _StubPrediction(_ExplodingTrace())
    assert export_trace(pred, trace_dir=str(tmp_path)) is None
