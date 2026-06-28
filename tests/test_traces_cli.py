"""Unit tests for the rrlm-traces CLI helpers (list/read/grep over captured traces)."""
from __future__ import annotations

import json
from pathlib import Path

from rrlm.traces import grep_traces, list_traces, read_trace


def _make_trace_dir(tmp_path: Path) -> Path:
    d = tmp_path / "traces"
    d.mkdir()
    trace = {
        "status": "completed",
        "model": "ornith/ornith-1.0-35b",
        "sub_model": "supergemma/x",
        "iterations": 2,
        "steps": [
            {"reasoning": "parse the csv", "code": "rows = read_csv(data)", "output": "5 rows"},
            {"reasoning": "find min score", "code": "min(rows, key=score)", "output": "B"},
        ],
    }
    (d / "trace-20260101T000000-1.json").write_text(json.dumps(trace))
    (d / "index.jsonl").write_text(
        json.dumps({
            "trace_file": "trace-20260101T000000-1.json",
            "instruction": "which id is most negative?",
            "answer": "B",
            "wall_clock_s": 1.2,
            "config": {"main_model": "ornith/ornith-1.0-35b"},
        }) + "\n"
    )
    return d


def test_list(tmp_path):
    d = _make_trace_dir(tmp_path)
    out = list_traces(d)
    assert "1 traces" in out
    assert "which id is most negative?" in out
    assert "trace-20260101T000000-1.json" in out


def test_read_last(tmp_path):
    d = _make_trace_dir(tmp_path)
    out = read_trace(d, last=True)
    assert "status=completed" in out
    assert "parse the csv" in out
    assert "min(rows" in out
    assert "step 2" in out


def test_grep_hits_code_and_index(tmp_path):
    d = _make_trace_dir(tmp_path)
    assert "read_csv" in grep_traces(d, "read_csv")       # matches step code
    assert "most negative" in grep_traces(d, "negative")  # matches index instruction


def test_grep_no_match(tmp_path):
    d = _make_trace_dir(tmp_path)
    assert "no matches" in grep_traces(d, "zzz-not-present")


def test_empty_dir(tmp_path):
    (tmp_path / "empty").mkdir()
    assert "no traces" in list_traces(tmp_path / "empty")
