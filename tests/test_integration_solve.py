"""Integration tests for the rrlm solve harness, end to end, with no mocks.

Every test here drives :func:`rrlm.solve.solve` against the real offline stub
server (a genuine subprocess, real sockets) wired in through a temp Pi config.
The full stack runs: model resolution, ``dspy.LM`` build, litellm HTTP request,
the stub's canned reply, and real code execution in predict-rlm's local-CPython
``supervisor`` backend over the real ``data`` variable. No part of the LLM call
path is patched, which is what makes these integration tests rather than unit
tests, while staying offline and deterministic.
"""

from __future__ import annotations

import json

import pytest

from rrlm import solve


@pytest.mark.integration
def test_supervisor_happy_path_computes_from_real_data(stub_model):
    """submit scenario: the REPL computes len(data) for real and SUBMITs it."""
    model = stub_model("submit")
    data = "alpha\nbeta\ngamma\n"  # the answer is len(data), computed in the REPL
    result = solve(
        "compute something from the data",
        data,
        main_model=model,
        backend="supervisor",
        max_iterations=5,
    )
    assert result["error"] is None
    assert result["answer"] == str(len(data))
    assert result["config"]["backend"] == "supervisor"
    assert result["config"]["main_model"] == model
    # One real action turn, accounted on the main role.
    assert result["usage"]["calls"] == 1
    assert "main" in result["usage"]["by_role"]


@pytest.mark.integration
def test_predict_fanout_uses_sub_lm(stub_model):
    """predict scenario: the REPL fans out a real await predict() leaf call.

    The leaf hits the stub too and is accounted on the sub role, proving the
    sub-LM path runs end to end.
    """
    model = stub_model("predict")
    result = solve(
        "classify the data",
        "this product was a disappointment",
        main_model=model,
        backend="supervisor",
        max_iterations=5,
    )
    assert result["error"] is None
    assert result["answer"] == "negative"  # the stub's canned leaf label
    assert "sub" in result["usage"]["by_role"]
    assert result["usage"]["by_role"]["sub"]["calls"] >= 1


@pytest.mark.integration
def test_max_iterations_falls_through_to_extract(stub_model):
    """never scenario: code that never SUBMITs must exhaust max_iterations.

    The run then uses the extract signature for its final answer, which proves
    the max_iterations cap is honored by the real harness.
    """
    model = stub_model("never")
    result = solve(
        "this can never submit",
        "payload",
        main_model=model,
        backend="supervisor",
        max_iterations=3,
    )
    assert result["error"] is None
    assert result["answer"] == "extracted-fallback-answer"
    # 3 action turns that never submit, then 1 extract call.
    assert result["usage"]["calls"] == 4


@pytest.mark.integration
def test_hard_wall_clock_timeout_aborts_real_run(stub_model):
    """slow scenario: a genuinely slow endpoint must trip the wall-clock ceiling."""
    model = stub_model("slow")
    result = solve(
        "this endpoint is slow",
        "payload",
        main_model=model,
        backend="supervisor",
        max_iterations=3,
        timeout_s=0.5,
    )
    assert result["answer"] == ""
    assert result["error"] and "Timeout" in result["error"]
    assert result["wall_clock_s"] < 2.0  # aborted well before the stub would reply


@pytest.mark.integration
def test_trace_export_to_rrlm_trace_dir(stub_model, tmp_path, monkeypatch):
    """A completed run writes a RunTrace file plus an index line under RRLM_TRACE_DIR."""
    trace_dir = tmp_path / "traces"
    monkeypatch.setenv("RRLM_TRACE_DIR", str(trace_dir))
    model = stub_model("submit")
    result = solve(
        "write me a trace",
        "abcdef",
        main_model=model,
        backend="supervisor",
        max_iterations=5,
    )
    assert result["error"] is None
    assert result["trace_file"] is not None
    saved = json.loads((trace_dir / result["trace_file"].split("/")[-1]).read_text())
    assert saved.get("status") in {"completed", "in_progress"}
    index_lines = (trace_dir / "index.jsonl").read_text().strip().splitlines()
    assert len(index_lines) == 1
    rec = json.loads(index_lines[0])
    assert rec["answer"] == result["answer"]
    assert rec["config"]["main_model"] == model


@pytest.mark.integration
def test_capacity_driven_recursion_via_rlm_spawn(stub_model):
    """spawn scenario: the orchestrator calls rlm_spawn, running a real child agent.

    Exercises the depth-gated rlm_spawn tool end to end. With max_depth=1 the child
    has no spawn tool of its own; it runs its own turns and the parent wraps the
    child's answer. spawn_stats records the one child spawned at depth 1.
    """
    model = stub_model("spawn")
    result = solve(
        "delegate a slice to a child",
        "abcdefghijklmnopqrstuvwxyz0123456789",
        main_model=model,
        backend="supervisor",
        max_iterations=3,
        max_depth=1,
    )
    assert result["error"] is None
    assert result["answer"].startswith("spawned:")
    assert result["spawn_stats"].get(1) == 1


@pytest.mark.integration
def test_separate_sub_model_resolves_and_runs(stub_model):
    """An explicit sub_model is resolved independently and still runs the leaf path."""
    model = stub_model("predict")
    result = solve(
        "classify",
        "bad experience overall",
        main_model=model,
        sub_model=model,  # same stub provider, resolved as its own ResolvedModel
        backend="supervisor",
        max_iterations=5,
    )
    assert result["error"] is None
    assert result["answer"] == "negative"
    assert result["config"]["sub_model"] == model


@pytest.mark.integration
def test_asolve_runs_inside_an_existing_event_loop(stub_model):
    """The async entry point works from async callers (servers, other agents)."""
    import asyncio

    from rrlm import asolve

    model = stub_model("submit")

    async def caller():
        return await asolve(
            "compute it", "xyz", main_model=model, backend="supervisor", max_iterations=5
        )

    result = asyncio.run(caller())
    assert result["error"] is None
    assert result["answer"] == "3"


@pytest.mark.integration
def test_typed_answer_parses_to_int(stub_model):
    """typed scenario: SUBMIT(answer=len(data)) with answer_type=int returns a
    real int, not a string."""
    model = stub_model("typed")
    result = solve(
        "how many characters?", "hello", main_model=model,
        backend="supervisor", max_iterations=5, answer_type=int,
    )
    assert result["error"] is None
    assert result["answer"] == 5
    assert isinstance(result["answer"], int)


@pytest.mark.integration
def test_file_input_is_mounted_and_readable(stub_model, tmp_path):
    """filesread scenario: a real host file is mounted into the sandbox; the
    REPL opens the mounted path and submits its actual content."""
    payload = tmp_path / "payload.txt"
    payload.write_text("file-content-42\n", encoding="utf-8")
    model = stub_model("filesread")
    result = solve(
        "read the mounted file", "", main_model=model,
        backend="supervisor", max_iterations=5, files=[payload],
    )
    assert result["error"] is None
    assert result["answer"] == "file-content-42"


@pytest.mark.integration
def test_missing_input_file_fails_fast(stub_model, tmp_path):
    model = stub_model("submit")
    with pytest.raises(FileNotFoundError, match="not found"):
        solve("read it", "", main_model=model, files=[tmp_path / "nope.pdf"])


@pytest.mark.integration
def test_solve_many_answers_in_one_run(stub_model):
    """listsubmit scenario: several questions, ONE run, answers list aligned."""
    from rrlm import solve_many

    model = stub_model("listsubmit")
    result = solve_many(
        ["how long is the data?", "what comes second?"], "hello",
        main_model=model, backend="supervisor", max_iterations=5,
    )
    assert result["error"] is None
    assert result["answers"] == ["5", "second-answer"]
    # one run: a single action turn served both questions
    assert result["usage"]["by_role"]["main"]["calls"] == 1


@pytest.mark.integration
def test_global_sub_call_budget_blocks_fanout(stub_model):
    """predict scenario with a zero sub-call budget: the fan-out is refused
    before any leaf HTTP call, so the sub role never appears in usage."""
    model = stub_model("predict")
    result = solve(
        "classify", "some text", main_model=model,
        backend="supervisor", max_iterations=2, max_llm_calls=0,
    )
    # The leaf label can never be produced; whether the run ends in the extract
    # fallback or an error, no sub-LM call may have been made.
    assert result["answer"] != "negative"
    assert "sub" not in result["usage"]["by_role"]


@pytest.mark.integration
def test_spawn_budget_refusal_reaches_the_model(stub_model):
    """spawn scenario with max_spawns=0: rlm_spawn returns a refusal string the
    orchestrator can act on (here it submits it, proving the code path)."""
    model = stub_model("spawn")
    result = solve(
        "delegate a slice", "abcdefghijklmnopqrstuvwxyz", main_model=model,
        backend="supervisor", max_iterations=3, max_depth=1, max_spawns=0,
    )
    assert result["error"] is None
    assert result["answer"].startswith("spawned:")
    assert "rlm_spawn refused" in result["answer"]
    assert result["spawn_stats"] == {}  # nothing actually spawned
