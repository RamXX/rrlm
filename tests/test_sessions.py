"""Session tests: namespace persistence, interpreter ownership, lifecycle.

The integration tests run the real stack against the offline stub server's
``session`` scenario, whose canned REPL code increments a counter living in
the interpreter namespace. The counter is the proof: a :class:`rrlm.Session`
answers 1, 2, ... across calls (one persistent namespace), while one-shot
``solve()`` calls answer 1 every time (fresh namespace, and the fresh
interpreter is shut down after each run - the leak fix these tests pin down).

Interpreter creation is observed through a recording subclass installed at
``predict_rlm.DirectPythonBackend``: ``rrlm.harness.build_rlm`` and
``rrlm.Session`` both import it by attribute at call time, so every backend
the code under test creates is captured without patching any rrlm internals.
"""

from __future__ import annotations

import asyncio

import predict_rlm
import pytest

from rrlm import Session, solve
from rrlm.solve import asolve

DATA = "alpha\nbeta\ngamma\n"


@pytest.fixture
def created_backends(monkeypatch):
    """Record every DirectPythonBackend the code under test constructs."""
    created: list = []
    real = predict_rlm.DirectPythonBackend

    class Recording(real):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            created.append(self)

    monkeypatch.setattr(predict_rlm, "DirectPythonBackend", Recording)
    return created


# --- construction / lifecycle (no model, no stub) ---------------------------


def test_session_rejects_sandboxed_backends():
    with pytest.raises(NotImplementedError, match="supervisor"):
        Session(backend="jspi")
    with pytest.raises(NotImplementedError, match="RRLM_SBX_NAME"):
        Session(backend="sbx")


def test_session_rejects_engine_routing():
    with pytest.raises(ValueError, match="stateless"):
        Session(engine="reference")
    session = Session()
    try:
        with pytest.raises(ValueError, match="stateless"):
            session.solve("x", engine="reference")
    finally:
        session.close()


def test_session_owns_its_interpreter():
    with pytest.raises(ValueError, match="owns its interpreter"):
        Session(interpreter=object())


def test_session_close_is_idempotent_and_final(created_backends):
    session = Session()
    session.close()
    session.close()  # second close is a no-op, not an error
    with pytest.raises(RuntimeError, match="closed"):
        session.solve("anything")
    with pytest.raises(RuntimeError, match="closed"):
        session.reset()
    assert len(created_backends) == 1
    assert created_backends[0]._shutdown is True


def test_session_context_manager_closes(created_backends):
    with Session() as session:
        assert session is not None
    assert created_backends[0]._shutdown is True


def test_async_context_manager_closes(created_backends):
    async def use():
        async with Session():
            pass

    asyncio.run(use())
    assert created_backends[0]._shutdown is True


def test_asolve_rejects_interpreter_with_sandboxed_backend():
    with pytest.raises(ValueError, match="requires backend='supervisor'"):
        asyncio.run(asolve("x", DATA, backend="jspi", interpreter=object()))


def test_asolve_rejects_engine_plus_interpreter():
    with pytest.raises(ValueError, match="mutually exclusive"):
        asyncio.run(asolve("x", DATA, engine="reference", interpreter=object()))


# --- namespace persistence and ownership (offline stub, real stack) ---------


@pytest.mark.integration
def test_session_namespace_persists_across_solves(stub_model, created_backends):
    model = stub_model("session")
    with Session(main_model=model, max_iterations=5) as session:
        first = session.solve("start a counter", DATA)
        second = session.solve("increment the counter")
        third = session.solve("increment it again")
    assert (first["error"], second["error"], third["error"]) == (None, None, None)
    # The counter lives in the REPL namespace: only persistence explains 1,2,3.
    assert [r["answer"] for r in (first, second, third)] == ["1", "2", "3"]
    # One session = one interpreter, reused for every turn, released on close.
    assert len(created_backends) == 1
    assert created_backends[0]._shutdown is True


@pytest.mark.integration
def test_one_shot_solves_stay_isolated_and_release_their_interpreter(
    stub_model, created_backends
):
    model = stub_model("session")
    first = solve("count", DATA, main_model=model, max_iterations=5)
    second = solve("count", DATA, main_model=model, max_iterations=5)
    # No bleed between one-shot runs: each starts a fresh namespace...
    assert (first["answer"], second["answer"]) == ("1", "1")
    # ...and each run shut down the interpreter it created (the leak fix).
    assert len(created_backends) == 2
    assert all(backend._shutdown for backend in created_backends)


@pytest.mark.integration
def test_session_reset_clears_the_namespace(stub_model):
    model = stub_model("session")
    with Session(main_model=model, max_iterations=5) as session:
        assert session.solve("start", DATA)["answer"] == "1"
        assert session.solve("continue")["answer"] == "2"
        session.reset()
        assert session.solve("start over")["answer"] == "1"


@pytest.mark.integration
def test_session_async_turns_share_the_namespace(stub_model):
    model = stub_model("session")

    async def run() -> list[str]:
        async with Session(main_model=model, max_iterations=5) as session:
            first = await session.asolve("start", DATA)
            second = await session.asolve("continue")
            return [first["answer"], second["answer"]]

    assert asyncio.run(run()) == ["1", "2"]


@pytest.mark.integration
def test_spawned_children_release_their_interpreters(stub_model, created_backends):
    model = stub_model("spawn")
    result = solve(
        "delegate to a child", DATA,
        main_model=model, backend="supervisor", max_depth=1, max_iterations=5,
    )
    assert result["error"] is None
    assert result["spawn_stats"] == {1: 1}
    # Parent + one child, each with its own interpreter, both released.
    assert len(created_backends) == 2
    assert all(backend._shutdown for backend in created_backends)


def test_shutdown_interpreter_kills_a_stuck_polite_path():
    """A shutdown() blocked on a protocol read must not hang the release.

    Reproduces the CI hang shape: shutdown() blocks until the runner process
    dies (here: an Event released by kill()). The bounded release must return
    promptly and must have escalated to killing the process.
    """
    import threading
    import time

    from rrlm.harness import shutdown_interpreter

    released = threading.Event()
    killed = threading.Event()

    class StuckProcess:
        def kill(self):
            killed.set()
            released.set()  # killing the runner closes the pipe -> read unblocks

        def poll(self):
            return None

    class StuckInterpreter:
        _process = StuckProcess()

        def shutdown(self):
            released.wait(30)  # blocked "readline": only process death frees it

    t0 = time.monotonic()
    shutdown_interpreter(StuckInterpreter(), timeout_s=0.5)
    assert killed.is_set()
    assert time.monotonic() - t0 < 5


def test_shutdown_interpreter_polite_path_never_kills():
    from rrlm.harness import shutdown_interpreter

    killed = []

    class CleanProcess:
        def kill(self):
            killed.append(True)

    class CleanInterpreter:
        _process = CleanProcess()
        was_shutdown = False

        def shutdown(self):
            self.was_shutdown = True

    interp = CleanInterpreter()
    shutdown_interpreter(interp, timeout_s=5.0)
    assert interp.was_shutdown and not killed
