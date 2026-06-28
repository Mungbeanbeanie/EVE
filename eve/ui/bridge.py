"""InputBridge — carries user turns from the web UI to the agent.

The visualizer server (:mod:`eve.ui.server`) runs its HTTP handlers on daemon
threads, while the agent runs its loop on a separate thread with its own asyncio
event loop. This bridge is the thread-safe seam between them: the HTTP ``POST``
handlers *submit* events from their thread, and the agent *awaits* them on its.

A plain :class:`queue.Queue` is the whole mechanism — it is already thread-safe,
and the agent consumes it without blocking its event loop by waiting on the
``get`` in a worker thread (:meth:`next_event`). One-way (browser → agent); orb
state flows back the other way over SSE, so the two directions never cross here.
"""

from __future__ import annotations

import asyncio
import queue
from typing import Any

# Event shapes placed on the queue (kept as plain dicts so the HTTP layer can
# build them without importing anything heavy):
#   {"type": "text", "text": "<message>"}      — a typed prompt
#   {"type": "control", "action": "listen"}    — capture one spoken utterance
#   {"type": "stop"}                            — unblock the agent so it can exit
Event = dict[str, Any]


class InputBridge:
    """Thread-safe channel of user turns from the UI to the agent loop."""

    def __init__(self) -> None:
        self._q: "queue.Queue[Event]" = queue.Queue()

    # ── producer side (called from the HTTP handler thread) ──────────────────
    def submit_text(self, text: str) -> None:
        """Queue a typed message for the agent to answer."""
        self._q.put({"type": "text", "text": text})

    def submit_control(self, action: str) -> None:
        """Queue a control action (e.g. ``"listen"`` to capture a spoken turn)."""
        self._q.put({"type": "control", "action": action})

    def stop(self) -> None:
        """Wake a blocked :meth:`next_event` with a sentinel so the loop can exit."""
        self._q.put({"type": "stop"})

    # ── consumer side (awaited from the agent's event loop) ──────────────────
    async def next_event(self) -> Event:
        """Wait for the next UI event without blocking the event loop.

        ``queue.Queue.get`` is a blocking call, so it runs on a worker thread via
        :func:`asyncio.to_thread`; the agent coroutine simply ``await``s the result.
        """
        return await asyncio.to_thread(self._q.get)
