"""Structured progress events for hosts embedding rrlm.

A solve is opaque while it runs; hosts (the Pi extension, servers, TUIs) want
progress they can display. ``solve(..., on_event=cb)`` delivers plain dicts to
the callback as the run advances; ``rrlm-solve --events`` prints them as JSONL
on stderr (stdout stays reserved for the answer/result).

Events carry an ``event`` name plus event-specific fields:

* ``run_started``   - instruction_chars, data_chars, backend or engine
* ``llm_call``      - role (main/sub), model, prompt_tokens, completion_tokens,
                      cost_usd (None when the provider reports none)
* ``spawn_started`` / ``spawn_finished`` - depth, task_chars
* ``run_finished``  - error (None on success), wall_clock_s

The callback is synchronous and must be fast; it is invoked on the run's own
path. Exceptions it raises are swallowed: progress display must never be able
to break a run. Emission order within one run is deterministic, but field
values (token counts, costs) are whatever the provider reported.
"""

from __future__ import annotations

import time
from collections.abc import Callable

OnEvent = Callable[[dict], None]


class EventEmitter:
    """Delivers event dicts to one callback, isolating the run from it."""

    def __init__(self, callback: OnEvent):
        self._callback = callback

    def emit(self, event: str, **fields) -> None:
        try:
            self._callback({"event": event, "ts": time.time(), **fields})
        except Exception:  # noqa: BLE001, display code must never break a run
            pass
