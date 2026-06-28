"""Real-use-case eval: code reasoning over a real repository.

Loads rrlm's own Python sources (a genuine multi-file codebase, not synthetic
template data) into the harness and asks a "find the one function that does X"
question, the same shape as the bugfind task, over real code. The answer is a
specific, verifiable code fact, so a model cannot guess it without actually
reading across the files in the REPL.

    python examples/eval_bugfind.py
    REPO=/path/to/other/repo python examples/eval_bugfind.py   # point at any repo

For an arbitrary repo, override INSTRUCTION and EXPECT too (env vars), e.g. to ask
about a known bug in that codebase.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(os.environ.get("REPO", Path(__file__).resolve().parents[1] / "src" / "rrlm"))

DEFAULT_INSTRUCTION = (
    "This is the source of a Python package, with each file preceded by a "
    "'# FILE: <path>' header. Exactly one function executes a shell command when "
    "a configured API-key value begins with '!'. Name that function. Answer with "
    "only the function name."
)
DEFAULT_EXPECT = "_resolve_key_string"


def load_repo(root: Path) -> str:
    parts = []
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        parts.append(f"# FILE: {path.relative_to(root)}\n{path.read_text(encoding='utf-8')}")
    return "\n\n".join(parts)


def main() -> int:
    from _harness import run_eval

    instruction = os.environ.get("INSTRUCTION", DEFAULT_INSTRUCTION)
    expect = os.environ.get("EXPECT", DEFAULT_EXPECT)
    data = load_repo(REPO)

    def check(answer: str) -> tuple[bool, str]:
        norm = answer.strip().strip("`").strip("()")
        ok = expect.lower() in norm.lower()
        return ok, f"expected to contain {expect!r}"

    print(f"[bugfind] repo={REPO} files concatenated; expecting {expect!r}")
    return 0 if run_eval("bugfind", instruction, data, check) else 1


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    sys.exit(main())
