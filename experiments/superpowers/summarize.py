#!/usr/bin/env python3
"""Aggregate the RLM-superpower matrix into evidence tables + extracted proof.

Reads results.tsv (one row per cell, with run_id) and the corresponding
runs/<run_id>/ artifacts (run.json = config + ground truth, result.json = answer
+ pass/fail, trace.json = the RLM RunTrace showing the code the model wrote).
Emits a markdown report to stdout and writes evidence/ snapshots so the proof is
reproducible without the gitignored runs/ dir.

Usage: python experiments/superpowers/summarize.py [results.tsv]
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments" / "superpowers"
RUNS = ROOT / "runs"
EVID = EXP / "evidence"


def load(path: Path) -> dict:
    try:
        return json.load(open(path))
    except Exception:
        return {}


def trace_code(trace: dict) -> str:
    """Concatenate the code the RLM wrote, across iterations."""
    steps = trace.get("steps") or trace.get("iterations") or []
    blocks = []
    for s in steps:
        c = s.get("code") or s.get("action") or ""
        if isinstance(c, str) and c.strip():
            blocks.append(c.strip())
    return "\n\n# --- next REPL turn ---\n".join(blocks)


def rows_from_tsv(tsv: Path) -> list[dict]:
    lines = tsv.read_text().splitlines()
    hdr = lines[0].split("\t")
    out = []
    for ln in lines[1:]:
        if not ln.strip():
            continue
        out.append(dict(zip(hdr, ln.split("\t"))))
    return out


def main() -> None:
    tsv = Path(sys.argv[1]) if len(sys.argv) > 1 else EXP / "results.tsv"
    rows = rows_from_tsv(tsv)
    EVID.mkdir(parents=True, exist_ok=True)

    # snapshot each run's small artifacts into evidence/ (preserve from gitignored runs/)
    for r in rows:
        rid = r.get("run_id", "").strip()
        if not rid:
            continue
        src = RUNS / rid
        dst = EVID / rid
        if src.is_dir() and not dst.exists():
            dst.mkdir(parents=True, exist_ok=True)
            for name in ("run.json", "result.json", "trace.json"):
                if (src / name).exists():
                    shutil.copy2(src / name, dst / name)

    # group rows by (task, model)
    print("# RLM-superpower matrix, results\n")
    by_task: dict[tuple, list[dict]] = {}
    for r in rows:
        by_task.setdefault((r["task"], r["model"]), []).append(r)

    for (task, model), cells in sorted(by_task.items()):
        print(f"## {task}, {model}\n")
        print("| size | seed | condition | passed | status | wall_s | prompt_tok | detail |")
        print("|---|---|---|---|---|---|---|---|")
        for c in sorted(cells, key=lambda x: (int(x["size"]), x["seed"], x["condition"])):
            print(
                f"| {c['size']} | {c['seed']} | {c['condition']} | "
                f"{c['passed']} | {c['status']} | {c['wall_s']} | "
                f"{c['prompt_tok']} | {c['check_detail'][:48]} |"
            )
        print()

    # extract representative evidence: one RLM trace's code per task + a baseline error
    print("## Evidence: what each condition actually did\n")
    seen_rlm: set = set()
    seen_base: set = set()
    for r in rows:
        rid = r.get("run_id", "").strip()
        d = EVID / rid
        res = load(d / "result.json")
        if r["condition"] == "rlm" and r["task"] not in seen_rlm and r["passed"] == "True":
            tr = load(d / "trace.json")
            code = trace_code(tr)
            if code:
                seen_rlm.add(r["task"])
                print(f"### RLM solved `{r['task']}` (size {r['size']}) by writing code:\n")
                print("```python")
                print(code[:1600])
                print("```\n")
        if r["condition"] == "baseline" and r["task"] not in seen_base and res.get("error"):
            seen_base.add(r["task"])
            print(f"### Baseline on `{r['task']}` (size {r['size']}) failed:\n")
            print(f"`{res['error'][:240]}`\n")


if __name__ == "__main__":
    main()
