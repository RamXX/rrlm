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


def test_list_trace_files_without_index(tmp_path):
    d = tmp_path / "traces"
    d.mkdir()
    (d / "trace-20260101T000000-9.json").write_text(json.dumps({"steps": []}))
    out = list_traces(d)
    assert "trace-20260101T000000-9.json" in out
    assert "no index.jsonl" in out


def test_read_requires_target(tmp_path):
    d = _make_trace_dir(tmp_path)
    assert "specify a trace file" in read_trace(d)


def test_read_not_found(tmp_path):
    d = _make_trace_dir(tmp_path)
    assert "not found" in read_trace(d, name="missing.json")


def test_read_last_on_empty_dir(tmp_path):
    d = tmp_path / "empty"
    d.mkdir()
    assert "no trace files" in read_trace(d, last=True)


def test_read_handles_double_encoded_trace(tmp_path):
    d = tmp_path / "traces"
    d.mkdir()
    payload = {"status": "completed", "iterations": 1, "steps": [{"code": "x = 1"}]}
    # a double-encoded export: file content is a JSON string, not an object
    (d / "trace-20260101T000000-1.json").write_text(json.dumps(json.dumps(payload)))
    out = read_trace(d, last=True)
    assert "status=completed" in out
    assert "x = 1" in out


def test_grep_bad_pattern(tmp_path):
    d = _make_trace_dir(tmp_path)
    assert "bad pattern" in grep_traces(d, "(unclosed")


def test_grep_ignore_case(tmp_path):
    d = _make_trace_dir(tmp_path)
    assert "read_csv" in grep_traces(d, "READ_CSV", ignore_case=True)


def test_main_list_read_grep(tmp_path, monkeypatch, capsys):
    import sys

    from rrlm import traces

    d = _make_trace_dir(tmp_path)

    monkeypatch.setattr(sys, "argv", ["rrlm-traces", "--dir", str(d), "list"])
    traces.main()
    assert "which id is most negative?" in capsys.readouterr().out

    monkeypatch.setattr(sys, "argv", ["rrlm-traces", "--dir", str(d), "read", "--last"])
    traces.main()
    assert "status=completed" in capsys.readouterr().out

    monkeypatch.setattr(sys, "argv", ["rrlm-traces", "--dir", str(d), "grep", "negative"])
    traces.main()
    assert "negative" in capsys.readouterr().out


def test_resolve_dir_uses_env(monkeypatch, tmp_path):
    from rrlm.traces import _resolve_dir

    monkeypatch.setenv("RRLM_TRACE_DIR", str(tmp_path / "from-env"))
    assert _resolve_dir(None) == tmp_path / "from-env"
    assert _resolve_dir("explicit") == Path("explicit")


def test_list_skips_malformed_index_line(tmp_path):
    d = _make_trace_dir(tmp_path)
    # append a malformed JSON line; it must be skipped, not crash list_traces
    with (d / "index.jsonl").open("a") as fh:
        fh.write("{not valid json\n")
    out = list_traces(d)
    assert "1 traces" in out  # the one good record still parses


def test_grep_skips_unreadable_trace_file(tmp_path):
    d = _make_trace_dir(tmp_path)
    # a trace file with non-JSON content is skipped by grep, not fatal
    (d / "trace-20260101T000001-2.json").write_text("this is not json")
    out = grep_traces(d, "read_csv")
    assert "read_csv" in out  # the valid trace still matches
