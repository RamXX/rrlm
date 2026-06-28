"""rrlm-traces -- inspect the predict-rlm RunTraces captured by rrlm-solve.

When ``RRLM_TRACE_DIR`` is set, each rrlm-solve call writes a RunTrace JSON plus an
``index.jsonl`` line (instruction -> answer -> config). This CLI lists, reads, and
greps those traces -- for debugging and for curating RLM-GEPA training sets.

    rrlm-traces list                      # one row per captured trace
    rrlm-traces read --last               # render the most recent trace
    rrlm-traces read trace-XXode.json     # render a specific trace
    rrlm-traces grep "sentiment"          # search instructions/answers/reasoning/code

The trace dir defaults to ``$RRLM_TRACE_DIR`` (or ``./traces``); override with --dir.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def _resolve_dir(arg: str | None) -> Path:
    d = arg or os.environ.get("RRLM_TRACE_DIR") or "traces"
    return Path(d)


def _load_index(trace_dir: Path) -> list[dict]:
    idx = trace_dir / "index.jsonl"
    if not idx.exists():
        return []
    out = []
    for line in idx.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _trace_files(trace_dir: Path) -> list[Path]:
    return sorted(trace_dir.glob("trace-*.json"), key=lambda p: p.stat().st_mtime)


def list_traces(trace_dir: Path) -> str:
    """One line per trace: instruction -> answer (+ models, wall)."""
    rows = _load_index(trace_dir)
    if not rows:
        files = _trace_files(trace_dir)
        if not files:
            return f"no traces in {trace_dir}"
        return "\n".join(f.name for f in files) + f"\n({len(files)} trace files; no index.jsonl)"
    lines = [f"{len(rows)} traces in {trace_dir}:"]
    for r in rows:
        cfg = r.get("config", {}) or {}
        main = str(cfg.get("main_model", "?")).split("/")[-1]
        instr = (r.get("instruction") or "").replace("\n", " ")[:60]
        ans = (r.get("answer") or "").replace("\n", " ")[:40]
        wall = r.get("wall_clock_s")
        lines.append(f"  {r.get('trace_file','?'):<34} [{main} {wall}s] {instr!r} -> {ans!r}")
    return "\n".join(lines)


def read_trace(trace_dir: Path, name: str | None = None, last: bool = False) -> str:
    """Render one RunTrace as a readable transcript (reasoning / code / output per step)."""
    if last:
        files = _trace_files(trace_dir)
        if not files:
            return f"no trace files in {trace_dir}"
        path = files[-1]
    elif name:
        path = trace_dir / name if not os.path.isabs(name) else Path(name)
    else:
        return "specify a trace file or --last"
    if not path.exists():
        return f"not found: {path}"
    d = json.loads(path.read_text())
    if isinstance(d, str):  # tolerate a double-encoded export
        d = json.loads(d)
    out = [
        f"=== {path.name} ===",
        f"status={d.get('status')} iterations={d.get('iterations')} "
        f"model={str(d.get('model','?')).split('/')[-1]} sub={str(d.get('sub_model','?')).split('/')[-1]}",
    ]
    for i, s in enumerate(d.get("steps", []) or []):
        pcs = s.get("predict_calls") or []
        ncalls = sum(len(g.get("calls", [g])) if isinstance(g, dict) else 1 for g in pcs)
        out.append(f"\n--- step {i + 1} (leaf predict calls: {ncalls}) ---")
        if s.get("reasoning"):
            out.append("reasoning: " + str(s["reasoning"]).replace("\n", " ")[:300])
        if s.get("code"):
            out.append("code:\n    " + str(s["code"]).replace("\n", "\n    ")[:1200])
        outp = s.get("output") or s.get("untruncated_output")
        if outp:
            out.append("output: " + str(outp).replace("\n", " ")[:300])
    return "\n".join(out)


def grep_traces(trace_dir: Path, pattern: str, ignore_case: bool = False) -> str:
    """Search instruction/answer/reasoning/code/output across all traces for a pattern."""
    import re

    flags = re.IGNORECASE if ignore_case else 0
    try:
        rx = re.compile(pattern, flags)
    except re.error as e:
        return f"bad pattern: {e}"
    hits = []
    for path in _trace_files(trace_dir):
        try:
            d = json.loads(path.read_text())
            d = json.loads(d) if isinstance(d, str) else d
        except (json.JSONDecodeError, OSError):
            continue
        for i, s in enumerate(d.get("steps", []) or []):
            for field in ("reasoning", "code", "output"):
                val = s.get(field)
                if val and rx.search(str(val)):
                    snippet = str(val).replace("\n", " ")
                    m = rx.search(snippet)
                    start = max(0, m.start() - 30)
                    hits.append(f"{path.name} step{i + 1} {field}: ...{snippet[start:m.end() + 40]}...")
                    break
    # also grep the index (instruction/answer)
    for r in _load_index(trace_dir):
        for field in ("instruction", "answer"):
            val = r.get(field)
            if val and rx.search(str(val)):
                hits.append(f"{r.get('trace_file','?')} {field}: {str(val).replace(chr(10),' ')[:120]}")
    return "\n".join(hits) if hits else f"no matches for {pattern!r} in {trace_dir}"


def main() -> None:
    p = argparse.ArgumentParser(prog="rrlm-traces", description=__doc__.splitlines()[0])
    p.add_argument("--dir", default=None, help="trace dir (default $RRLM_TRACE_DIR or ./traces)")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="one row per captured trace")
    pr = sub.add_parser("read", help="render a trace transcript")
    pr.add_argument("file", nargs="?", default=None)
    pr.add_argument("--last", action="store_true", help="read the most recent trace")
    pg = sub.add_parser("grep", help="search across traces")
    pg.add_argument("pattern")
    pg.add_argument("-i", "--ignore-case", action="store_true")
    a = p.parse_args()
    d = _resolve_dir(a.dir)
    if a.cmd == "list":
        print(list_traces(d))
    elif a.cmd == "read":
        print(read_trace(d, name=a.file, last=a.last))
    elif a.cmd == "grep":
        print(grep_traces(d, a.pattern, ignore_case=a.ignore_case))


if __name__ == "__main__":
    main()
