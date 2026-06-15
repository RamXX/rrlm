"""Aggregate runs/ into a comparison table across models and conditions.

Usage:
    python -m rrlm.report [--csv out.csv]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from rrlm.config import RUNS_DIR

COLUMNS = [
    "run_id",
    "task",
    "model",
    "condition",
    "passed",
    "status",
    "wall_s",
    "calls",
    "prompt_tok",
    "completion_tok",
    "cost_usd",
    "gen_time_s",
]


def collect(runs_dir: Path) -> list[dict]:
    rows = []
    for run_dir in sorted(runs_dir.iterdir() if runs_dir.exists() else []):
        meta_path, result_path = run_dir / "run.json", run_dir / "result.json"
        if not (meta_path.exists() and result_path.exists()):
            continue
        meta = json.loads(meta_path.read_text())
        result = json.loads(result_path.read_text())
        usage = result.get("usage", {})
        rows.append(
            {
                "run_id": result.get("run_id", run_dir.name),
                "task": meta.get("task_id"),
                "model": meta.get("model"),
                "condition": meta.get("condition"),
                "passed": result.get("passed"),
                "status": result.get("status"),
                "wall_s": result.get("wall_clock_s"),
                "calls": usage.get("calls"),
                "prompt_tok": usage.get("prompt_tokens"),
                "completion_tok": usage.get("completion_tokens"),
                "cost_usd": round(usage.get("cost_usd") or 0.0, 4),
                "gen_time_s": round((usage.get("generation_time_ms") or 0) / 1000, 1),
            }
        )
    return rows


def print_table(rows: list[dict]) -> None:
    if not rows:
        print("no runs found under", RUNS_DIR)
        return
    widths = {
        col: max(len(col), *(len(str(row[col])) for row in rows)) for col in COLUMNS
    }
    header = "  ".join(col.ljust(widths[col]) for col in COLUMNS)
    print(header)
    print("-" * len(header))
    for row in rows:
        print("  ".join(str(row[col]).ljust(widths[col]) for col in COLUMNS))


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize rrlm runs")
    parser.add_argument("--csv", help="also write the table to this CSV path")
    args = parser.parse_args()

    rows = collect(RUNS_DIR)
    print_table(rows)
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nwrote {len(rows)} rows to {args.csv}", file=sys.stderr)


if __name__ == "__main__":
    main()
