"""Shared memory types and the persistent-store interface.

`MemoryStore` is implemented by the durable layers (procedural, episodic).
WorkingMemory does NOT implement it — it's a simple volatile buffer with its own
small API. Keeping a common record type means the manager can merge results from
every layer uniformly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class MemoryRecord:
    """One unit of remembered information, regardless of layer."""

    content: str
    kind: str  # "procedural" | "episodic" | "message" — where it came from
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    score: float | None = None  # similarity score when returned from a search


class MemoryStore(ABC):
    """A durable, vector-searchable store (procedural or episodic)."""

    @abstractmethod
    async def add(self, content: str, metadata: dict[str, Any] | None = None) -> None:
        """Persist a new memory."""

    @abstractmethod
    async def search(self, query: str, k: int = 5) -> list[MemoryRecord]:
        """Return up to `k` memories most relevant to `query` (vector search)."""

    @abstractmethod
    async def recent(self, n: int = 10) -> list[MemoryRecord]:
        """Return the `n` most recent memories (chronological)."""
