"""MemoryManager — the single memory façade the Agent uses.

It owns all three layers and exposes two verbs:

    recall(query)    -> assemble the message list to send the LLM, blending the
                        working window with relevant procedural + episodic memories.
    remember(...)    -> write a turn to the right layers (always episodic/working;
                        procedural only when something durable was learned).

Construction is real wiring; the read/write *policies* are the learning exercise.
"""

from __future__ import annotations
import asyncio

from eve.config import Config
from eve.llm.base import Message
from eve.memory.episodic import EpisodicMemory
from eve.memory.mem0_backend import Mem0Backend
from eve.memory.procedural import ProceduralMemory
from eve.memory.working import WorkingMemory


class MemoryManager:
    """Composes working + procedural + episodic memory."""

    def __init__(
        self,
        working: WorkingMemory,
        procedural: ProceduralMemory,
        episodic: EpisodicMemory,
    ) -> None:
        self.working = working
        self.procedural = procedural
        self.episodic = episodic

    @classmethod
    def from_config(cls, config: Config) -> "MemoryManager":
        """Build the manager and all three layers from config (real wiring)."""
        backend = Mem0Backend(config)  # shared mem0/pgvector client (lazy)
        return cls(
            working=WorkingMemory(),
            procedural=ProceduralMemory(backend),
            episodic=EpisodicMemory(backend),
        )

    # ── Read ─────────────────────────────────────────────────────────────────
    async def recall(self, query: str) -> list[Message]:
        """Return the full message list for the LLM, with long-term memory blended in."""
        proc_records, epis_records = await asyncio.gather(
            self.procedural.search(query, k=5),
            self.episodic.search(query, k=5),
        )
        retrieved: list[Message] = [
            {"role": "system", "content": r.content}
            for r in (*proc_records, *epis_records)
        ]
        return self.working.render(retrieved=retrieved or None)

    # ── Write ────────────────────────────────────────────────────────────────
    async def remember(self, *, user: str, assistant: str) -> None:
        """Persist a completed turn across the appropriate layers."""
        self.working.add_user(user)
        self.working.add_assistant(assistant)

        await self.episodic.add(f"User: {user}\nEVE: {assistant}")

        if "from now on" in user.lower() or "always" in user.lower():
            await self.procedural.add(f"User: {user}\nEVE: {assistant}")
