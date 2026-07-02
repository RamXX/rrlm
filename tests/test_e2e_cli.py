"""End-to-end tests: the rrlm-solve CLI and the library entry point, no mocks.

These exercise the system the way a user does, against the real offline stub
server. Two surfaces are covered:

  * the ``rrlm-solve`` console script run as a real subprocess (real argv, real
    stdin, real stdout/stderr, real exit codes), and
  * the ``main()`` entry invoked in-process so its argument handling, the
    RRLM_TIMEOUT env path, and the JSON / error exit branches are measured.

Determinism and offline operation come from the stub, not from patching.
"""

from __future__ import annotations

import importlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

# `import rrlm.solve as S` would bind the re-exported solve *function*; force the
# submodule so we can call its main() and reach module globals.
S = importlib.import_module("rrlm.solve")


def _write_dead_endpoint_config(agent_dir: Path) -> str:
    """Write a Pi config whose provider points at a refused port (no server).

    Used to drive the error / non-zero-exit path: model resolution succeeds but
    the real HTTP call fails fast, so solve() returns an error. Returns the ref.
    """
    agent_dir.mkdir(parents=True, exist_ok=True)
    models = {
        "providers": {
            "stub": {
                "baseUrl": "http://127.0.0.1:1/submit/v1",
                "api": "openai-completions",
                "apiKey": "stub-key",
                "models": [{"id": "stub-model", "maxTokens": 4096, "contextWindow": 8192}],
            }
        }
    }
    (agent_dir / "models.json").write_text(json.dumps(models))
    return "stub/stub-model"


def _cli() -> list[str]:
    """The rrlm-solve console-script command (the real installed entry point)."""
    exe = shutil.which("rrlm-solve") or str(Path(sys.executable).parent / "rrlm-solve")
    return [exe]


def _subprocess_env() -> dict:
    """A clean offline env that still points Pi resolution at the stub config."""
    env = os.environ.copy()
    env.pop("OPENROUTER_API_KEY", None)
    return env


# --------------------------------------------------------------------------- #
# Real subprocess invocations of the console script.
# --------------------------------------------------------------------------- #
@pytest.mark.e2e
def test_cli_subprocess_prints_answer(stub_model):
    model = stub_model("submit")
    data = "alpha\nbeta\ngamma"
    proc = subprocess.run(
        [*_cli(), "-i", "compute it", "-d", data, "--main", model,
         "--backend", "supervisor", "--max-iterations", "5"],
        capture_output=True, text=True, env=_subprocess_env(), timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == str(len(data))


@pytest.mark.e2e
def test_cli_subprocess_json_output(stub_model):
    model = stub_model("submit")
    proc = subprocess.run(
        [*_cli(), "-i", "compute it", "-d", "hello", "--main", model,
         "--backend", "supervisor", "--max-iterations", "5", "--json"],
        capture_output=True, text=True, env=_subprocess_env(), timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["answer"] == str(len("hello"))
    assert payload["config"]["backend"] == "supervisor"
    assert payload["error"] is None


@pytest.mark.e2e
def test_cli_subprocess_reads_stdin(stub_model):
    model = stub_model("submit")
    data = "first line\nsecond line\n"
    proc = subprocess.run(
        [*_cli(), "-i", "compute it", "-d", "-", "--main", model,
         "--backend", "supervisor", "--max-iterations", "5"],
        input=data, capture_output=True, text=True, env=_subprocess_env(), timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == str(len(data))


@pytest.mark.e2e
def test_cli_subprocess_error_exit_code(tmp_path):
    """A dead endpoint makes solve() return an error; the CLI exits non-zero."""
    agent_dir = tmp_path / "agent"
    model = _write_dead_endpoint_config(agent_dir)
    env = _subprocess_env()
    env["PI_CODING_AGENT_DIR"] = str(agent_dir)
    proc = subprocess.run(
        [*_cli(), "-i", "will fail", "-d", "x", "--main", model,
         "--backend", "supervisor", "--max-iterations", "2"],
        capture_output=True, text=True, env=env, timeout=60,
    )
    assert proc.returncode == 1
    assert "ERROR" in proc.stderr


# --------------------------------------------------------------------------- #
# In-process main() invocations (so the CLI plumbing is measured by coverage).
# --------------------------------------------------------------------------- #
@pytest.mark.e2e
def test_main_plain_answer(stub_model, monkeypatch, capsys):
    model = stub_model("submit")
    monkeypatch.setattr(
        sys, "argv",
        ["rrlm-solve", "-i", "go", "-d", "hello", "--main", model,
         "--backend", "supervisor", "--max-iterations", "5"],
    )
    S.main()
    assert capsys.readouterr().out.strip() == str(len("hello"))


@pytest.mark.e2e
def test_main_json_branch(stub_model, monkeypatch, capsys):
    model = stub_model("submit")
    monkeypatch.setattr(
        sys, "argv",
        ["rrlm-solve", "-i", "go", "-d", "hello", "--main", model,
         "--backend", "supervisor", "--max-iterations", "5", "--json"],
    )
    S.main()
    payload = json.loads(capsys.readouterr().out)
    assert payload["answer"] == str(len("hello"))


@pytest.mark.e2e
def test_main_error_exits_nonzero(tmp_path, monkeypatch, capsys):
    agent_dir = tmp_path / "agent"
    model = _write_dead_endpoint_config(agent_dir)
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(agent_dir))
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(
        sys, "argv",
        ["rrlm-solve", "-i", "fail", "-d", "x", "--main", model,
         "--backend", "supervisor", "--max-iterations", "2"],
    )
    with pytest.raises(SystemExit) as exc:
        S.main()
    assert exc.value.code == 1
    assert "ERROR" in capsys.readouterr().err


@pytest.mark.e2e
def test_main_timeout_env_var(stub_model, monkeypatch, capsys):
    """RRLM_TIMEOUT (no --timeout flag) supplies the wall-clock ceiling."""
    model = stub_model("slow")
    monkeypatch.setenv("RRLM_TIMEOUT", "0.5")
    monkeypatch.setattr(
        sys, "argv",
        ["rrlm-solve", "-i", "slow", "-d", "x", "--main", model,
         "--backend", "supervisor", "--max-iterations", "2"],
    )
    with pytest.raises(SystemExit) as exc:
        S.main()
    assert exc.value.code == 1
    assert "Timeout" in capsys.readouterr().err


@pytest.mark.e2e
def test_main_invalid_timeout_env_is_ignored(stub_model, monkeypatch, capsys):
    """A non-numeric RRLM_TIMEOUT is ignored (no ceiling), so the run completes."""
    model = stub_model("submit")
    monkeypatch.setenv("RRLM_TIMEOUT", "not-a-number")
    monkeypatch.setattr(
        sys, "argv",
        ["rrlm-solve", "-i", "go", "-d", "hello", "--main", model,
         "--backend", "supervisor", "--max-iterations", "5"],
    )
    S.main()
    assert capsys.readouterr().out.strip() == str(len("hello"))


# --------------------------------------------------------------------------- #
# Library entry point, end to end.
# --------------------------------------------------------------------------- #
@pytest.mark.e2e
def test_library_solve_entrypoint(stub_model):
    from rrlm import solve as solve_fn

    model = stub_model("submit")
    result = solve_fn(
        "compute it", "abcd", main_model=model, backend="supervisor", max_iterations=5
    )
    assert result["error"] is None
    assert result["answer"] == str(len("abcd"))


# --------------------------------------------------------------------------- #
# New CLI surfaces: defaults, env parity, typed answers, files, multi-question.
# --------------------------------------------------------------------------- #
@pytest.mark.e2e
def test_main_default_backend_is_supervisor(stub_model, monkeypatch, capsys):
    """No --backend flag, no RRLM_BACKEND: the default must be supervisor (the
    documented no-Deno default)."""
    model = stub_model("submit")
    monkeypatch.setattr(
        sys, "argv",
        ["rrlm-solve", "-i", "go", "-d", "hello", "--main", model,
         "--max-iterations", "5", "--json"],
    )
    S.main()
    payload = json.loads(capsys.readouterr().out)
    assert payload["config"]["backend"] == "supervisor"
    assert payload["answer"] == str(len("hello"))


@pytest.mark.e2e
def test_main_backend_env_is_honored(stub_model, monkeypatch, capsys):
    model = stub_model("submit")
    monkeypatch.setenv("RRLM_BACKEND", "supervisor")
    monkeypatch.setattr(
        sys, "argv",
        ["rrlm-solve", "-i", "go", "-d", "x", "--main", model,
         "--max-iterations", "5", "--json"],
    )
    S.main()
    assert json.loads(capsys.readouterr().out)["config"]["backend"] == "supervisor"


@pytest.mark.e2e
def test_main_invalid_backend_env_fails_loudly(stub_model, monkeypatch):
    model = stub_model("submit")
    monkeypatch.setenv("RRLM_BACKEND", "firecracker")
    monkeypatch.setattr(
        sys, "argv",
        ["rrlm-solve", "-i", "go", "-d", "x", "--main", model, "--max-iterations", "2"],
    )
    with pytest.raises(ValueError, match="unknown backend"):
        S.main()


@pytest.mark.e2e
def test_main_env_main_model(stub_model, monkeypatch, capsys):
    """RRLM_MAIN supplies the orchestrator when --main is omitted (extension parity)."""
    model = stub_model("submit")
    monkeypatch.setenv("RRLM_MAIN", model)
    monkeypatch.setattr(
        sys, "argv",
        ["rrlm-solve", "-i", "go", "-d", "hey", "--max-iterations", "5"],
    )
    S.main()
    assert capsys.readouterr().out.strip() == str(len("hey"))


@pytest.mark.e2e
def test_main_answer_type_int(stub_model, monkeypatch, capsys):
    model = stub_model("typed")
    monkeypatch.setattr(
        sys, "argv",
        ["rrlm-solve", "-i", "count chars", "-d", "hello", "--main", model,
         "--max-iterations", "5", "--answer-type", "int", "--json"],
    )
    S.main()
    payload = json.loads(capsys.readouterr().out)
    assert payload["answer"] == 5  # a real JSON number, not "5"


@pytest.mark.e2e
def test_main_file_flag_mounts_file(stub_model, monkeypatch, capsys, tmp_path):
    payload_file = tmp_path / "doc.txt"
    payload_file.write_text("mounted-content\n", encoding="utf-8")
    model = stub_model("filesread")
    monkeypatch.setattr(
        sys, "argv",
        ["rrlm-solve", "-i", "read the file", "--file", str(payload_file),
         "--main", model, "--max-iterations", "5"],
    )
    S.main()
    assert capsys.readouterr().out.strip() == "mounted-content"


@pytest.mark.e2e
def test_main_multi_instruction_prints_one_answer_per_line(stub_model, monkeypatch, capsys):
    model = stub_model("listsubmit")
    monkeypatch.setattr(
        sys, "argv",
        ["rrlm-solve", "-i", "how long?", "-i", "second?", "-d", "hello",
         "--main", model, "--max-iterations", "5"],
    )
    S.main()
    assert capsys.readouterr().out.strip().splitlines() == ["5", "second-answer"]


@pytest.mark.e2e
def test_cli_subprocess_multi_instruction(stub_model):
    """The multi-question path through the real console script."""
    model = stub_model("listsubmit")
    proc = subprocess.run(
        [*_cli(), "-i", "q1", "-i", "q2", "-d", "hello", "--main", model,
         "--max-iterations", "5"],
        capture_output=True, text=True, env=_subprocess_env(), timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip().splitlines() == ["5", "second-answer"]
