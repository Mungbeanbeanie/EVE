"""Procedural memory — durable "how I do things".

Stores learned preferences, standing instructions, and skills the user has taught
EVE ("always summarize emails in 3 bullets", "my work calendar is the default").
These are retrieved by semantic similarity and injected into context so behavior
persists across sessions.

Namespaced inside mem0 under PROCEDURAL_NS so it stays separate from episodic
memories that share the same backend.
"""

from __future__ import annotations

from typing import Any

from eve.memory.base import MemoryRecord, MemoryStore
from eve.memory.mem0_backend import Mem0Backend

PROCEDURAL_NS = "procedural"


class ProceduralMemory(MemoryStore):
    """Durable, vector-searchable store of preferences/skills via mem0."""

    def __init__(self, backend: Mem0Backend) -> None:
        self.backend = backend

    async def add(self, content: str, metadata: dict[str, Any] | None = None) -> None:
        """Persist a learned preference/skill."""
        # TODO(eve): 1. mem = self.backend.client()
        # TODO(eve): 2. Add to mem0 under user_id=PROCEDURAL_NS with your metadata,
        #               e.g. mem.add(content, user_id=PROCEDURAL_NS, metadata=...).
        #               Run blocking calls via asyncio.to_thread.
        raise NotImplementedError(
            "Implement procedural add — see eve/memory/procedural.py:add"
        )

    async def search(self, query: str, k: int = 5) -> list[MemoryRecord]:
        """Return preferences/skills relevant to `query`."""
        # TODO(eve): 1. results = mem.search(query, user_id=PROCEDURAL_NS, limit=k)
        # TODO(eve): 2. Map each hit into MemoryRecord(kind="procedural", score=...).
        raise NotImplementedError(
            "Implement procedural search — see eve/memory/procedural.py:search"
        )

    async def recent(self, n: int = 10) -> list[MemoryRecord]:
        """Return recently-learned preferences/skills."""
        # TODO(eve): use mem0's get_all(user_id=PROCEDURAL_NS) and take the newest n.
        raise NotImplementedError(
            "Implement procedural recent — see eve/memory/procedural.py:recent"
        )
