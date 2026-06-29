"""Procedural memory — durable "how I do things".

Stores learned preferences, standing instructions, and skills the user has taught
EVE ("always summarize emails in 3 bullets", "my work calendar is the default").
These are retrieved by semantic similarity and injected into context so behavior
persists across sessions.

Namespaced inside mem0 under PROCEDURAL_NS so it stays separate from episodic
memories that share the same backend.
"""

from __future__ import annotations

import asyncio
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
        mem = self.backend.client()
        await asyncio.to_thread(
            mem.add, content, user_id=PROCEDURAL_NS, metadata=metadata or {}
        )

    async def search(self, query: str, k: int = 5) -> list[MemoryRecord]:
        """Return preferences/skills relevant to `query`."""
        mem = self.backend.client()
        raw = await asyncio.to_thread(
            mem.search, query, filters={"user_id": PROCEDURAL_NS}, top_k=k
        )
        hits = raw.get("results", raw) if isinstance(raw, dict) else raw
        return [
            MemoryRecord(
                content=hit["memory"],
                kind="procedural",
                metadata=hit.get("metadata") or {},
                score=hit.get("score"),
            )
            for hit in hits
        ]

    async def recent(self, n: int = 10) -> list[MemoryRecord]:
        """Return recently-learned preferences/skills."""
        mem = self.backend.client()
        raw = await asyncio.to_thread(
            mem.get_all, filters={"user_id": PROCEDURAL_NS}, top_k=n
        )
        hits = raw.get("results", raw) if isinstance(raw, dict) else raw
        return [
            MemoryRecord(
                content=hit["memory"],
                kind="procedural",
                metadata=hit.get("metadata") or {},
            )
            for hit in hits[:n]
        ]
