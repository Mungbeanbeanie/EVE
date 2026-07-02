"""ActivityMonitor — knows when the user is (not) interacting with EVE.

The self-improvement loop may only use the machine while EVE is idle;
conversation always wins. The Agent wraps every user turn in `interaction()`,
and the loop asks "has the user been away long enough?" via `wait_for_idle()`.

A monotonic clock plus one lock make this safe to share between the agent's
event loop and the improvement daemon thread (each side runs its own asyncio
loop, so no cross-loop primitives — just polling).
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
import time
from typing import Iterator


class ActivityMonitor:
    """Thread-safe idle clock driven by user-interaction marks."""

    def __init__(self) -> None:
        """Initialize the idle-clock state.

        ``_lock`` guards both counters against concurrent reads/writes from the
        agent's event loop and the improvement daemon (which run separate asyncio
        loops, so we use a threading lock — never an asyncio primitive).
        ``_last_activity`` is the monotonic timestamp of the most recent user
        touch; ``_active_turns`` tracks open conversational turns (>0 means "in
        progress", regardless of how long ago the last touch was).
        """
        self._lock = threading.Lock()
        self._last_activity = time.monotonic()
        self._active_turns = 0  # >0 while a conversational turn is in flight

    # ── Writes (called by the Agent) ──────────────────────────────────────────
    def touch(self) -> None:
        """Record user activity "now" — resets the idle clock."""
        with self._lock:
            self._last_activity = time.monotonic()

    @contextlib.contextmanager
    def interaction(self) -> Iterator[None]:
        """Mark a span of active interaction (one conversational turn).

        While any interaction span is open, `idle_seconds()` reports 0 — a turn
        in progress (however long the model or TTS takes) is never "idle".
        """
        with self._lock:
            self._active_turns += 1
            self._last_activity = time.monotonic()
        try:
            yield
        finally:
            with self._lock:
                self._active_turns -= 1
                self._last_activity = time.monotonic()

    # ── Reads (called by the improvement loop) ────────────────────────────────
    def idle_seconds(self) -> float:
        """Seconds since the last user activity (0 while a turn is in flight)."""
        with self._lock:
            if self._active_turns > 0:
                return 0.0
            return time.monotonic() - self._last_activity

    def is_idle(self, threshold: float) -> bool:
        return self.idle_seconds() >= threshold

    async def wait_for_idle(self, threshold: float, poll: float = 1.0) -> None:
        """Sleep (async) until the user has been idle for `threshold` seconds."""
        while not self.is_idle(threshold):
            await asyncio.sleep(poll)
