"""Episodic memory — durable "what happened when".

An append-only, timestamped log of interactions and events ("on Tue you asked me to
draft an email to Sam"). Recall is by semantic similarity AND time, so EVE can
answer "what did we talk about yesterday?" and ground replies in past events.

Namespaced inside mem0 under EPISODIC_NS, separate from procedural memory.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
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
        meta = dict(metadata or {})
        meta.setdefault("created_at", datetime.now(timezone.utc).isoformat())
        mem = self.backend.client()
        await asyncio.to_thread(mem.add, content, user_id=EPISODIC_NS, metadata=meta)

    async def search(self, query: str, k: int = 5) -> list[MemoryRecord]:
        """Return past events most relevant to `query`."""
        mem = self.backend.client()
        raw = await asyncio.to_thread(
            mem.search, query, filters={"user_id": EPISODIC_NS}, top_k=k
        )
        hits = raw.get("results", raw) if isinstance(raw, dict) else raw
        return [
            MemoryRecord(
                content=hit["memory"],
                kind="episodic",
                metadata=hit.get("metadata") or {},
                score=hit.get("score"),
            )
            for hit in hits
        ]

    async def recent(self, n: int = 10) -> list[MemoryRecord]:
        """Return the most recent events, newest first."""
        mem = self.backend.client()
        raw = await asyncio.to_thread(
            mem.get_all, filters={"user_id": EPISODIC_NS}, top_k=max(n, 100)
        )
        hits = raw.get("results", raw) if isinstance(raw, dict) else raw
        hits = sorted(hits, key=lambda h: h.get("created_at") or "", reverse=True)
        return [
            MemoryRecord(
                content=hit["memory"],
                kind="episodic",
                metadata=hit.get("metadata") or {},
            )
            for hit in hits[:n]
        ]
