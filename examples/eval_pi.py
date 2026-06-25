"""Real-use-case eval: end-to-end Pi delegation.

Spawns ``pi --mode json`` with the rlm-backend extension and rlm-first skill
loaded, on a prompt whose right move is to delegate to ``rlm_solve``. Then it
confirms from the JSON event stream that the agent actually called ``rlm_solve``
(a ``tool_execution_start``/``tool_execution_end`` for it) and that the answer is
correct. This validates the delegation UX and routing skill, not just the harness
in isolation.

Requires: ``pi`` on PATH, ``rrlm-solve`` installed (or RRLM_DIR set), and a model
configured in Pi. Uses Pi's current model unless RRLM_MAIN overrides it.

    python examples/eval_pi.py
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EXTENSION = REPO_ROOT / "pi" / "extensions" / "rlm-backend" / "index.ts"
SKILL = REPO_ROOT / "pi" / "skills" / "rlm-first"


def main() -> int:
    if not shutil.which("pi"):
        print("[pi] SKIP: `pi` not found on PATH (install pi to run this eval)")
        return 2

    sys.path.insert(0, str(Path(__file__).parent))
    from eval_tabular import build

    _, csv, expected = build(size=4000)
    expected_str = f"{expected:.2f}"

    with tempfile.TemporaryDirectory() as tmp:
        data_path = Path(tmp) / "orders.csv"
        data_path.write_text(csv, encoding="utf-8")
        prompt = (
            f"There is a CSV of thousands of orders at {data_path}. Use the "
            f"rlm_solve tool (pass data_path) to compute the total revenue "
            f"(qty * unit_price, 2 decimals) for orders with region 'EMEA' and "
            f"status 'completed'. Report only the number."
        )
        cmd = [
            "pi", "--mode", "json", "-p", "--no-session",
            "-e", str(EXTENSION), "--skill", str(SKILL),
        ]
        if os.environ.get("PI_MODEL"):
            cmd += ["--model", os.environ["PI_MODEL"]]
        cmd.append(prompt)

        print(f"[pi] expected revenue: {expected_str}")
        print(f"[pi] running: {' '.join(cmd[:8])} ... (timeout 900s)")
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        except subprocess.TimeoutExpired:
            print("[pi] FAIL: timed out")
            return 1

        stdout = proc.stdout
        called_rlm_solve = False
        for line in stdout.splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            blob = json.dumps(ev)
            etype = str(ev.get("type", ""))
            if "rlm_solve" in blob and "tool" in etype:
                called_rlm_solve = True

        answer_ok = expected_str in stdout or str(int(expected)) in stdout
        ok = called_rlm_solve and answer_ok
        print(f"[pi] rlm_solve called: {called_rlm_solve} | answer present: {answer_ok}")
        if not ok and proc.returncode != 0:
            print(f"[pi] pi exit {proc.returncode}; stderr tail:\n{proc.stderr[-1500:]}")
        print(f"[pi] {'PASS' if ok else 'FAIL'}")
        return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
