"""Episodic memory — durable "what happened when".

An append-only, timestamped log of interactions and events ("on Tue you asked me to
draft an email to Sam"). Recall is by semantic similarity AND time, so EVE can
answer "what did we talk about yesterday?" and ground replies in past events.

Namespaced inside mem0 under EPISODIC_NS, separate from procedural memory.
"""

from __future__ import annotations

from typing import Any

from eve.memory.base import MemoryRecord, MemoryStore
from eve.memory.mem0_backend import Mem0Backend

EPISODIC_NS = "episodic"


class EpisodicMemory(MemoryStore):
    """Durable, time + vector searchable event log via mem0."""

    def __init__(self, backend: Mem0Backend) -> None:
        self.backend = backend

    async def add(self, content: str, metadata: dict[str, Any] | None = None) -> None:
        """Append a timestamped event/interaction."""
        # TODO(eve): 1. mem = self.backend.client()
        # TODO(eve): 2. Ensure metadata carries a timestamp (created_at) so you can
        #               do time-based queries later; store under user_id=EPISODIC_NS.
        raise NotImplementedError(
            "Implement episodic add — see eve/memory/episodic.py:add"
        )

    async def search(self, query: str, k: int = 5) -> list[MemoryRecord]:
        """Return past events most relevant to `query`."""
        # TODO(eve): mem.search(query, user_id=EPISODIC_NS, limit=k) -> MemoryRecords
        #            (kind="episodic"). Optionally blend recency into the ranking.
        raise NotImplementedError(
            "Implement episodic search — see eve/memory/episodic.py:search"
        )

    async def recent(self, n: int = 10) -> list[MemoryRecord]:
        """Return the most recent events, newest first."""
        # TODO(eve): get_all(user_id=EPISODIC_NS), sort by timestamp desc, take n.
        raise NotImplementedError(
            "Implement episodic recent — see eve/memory/episodic.py:recent"
        )
