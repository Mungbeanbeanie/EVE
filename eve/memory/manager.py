"""MemoryManager — the single memory façade the Agent uses.

It owns all three layers and exposes two verbs:

    recall(query)    -> assemble the message list to send the LLM, blending the
                        working window with relevant procedural + episodic memories.
    remember(...)    -> write a turn to the right layers (always episodic/working;
                        procedural only when something durable was learned).

Construction is real wiring; the read/write *policies* are the learning exercise.
"""

from __future__ import annotations

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
        # TODO(eve): 1. Retrieve in parallel (asyncio.gather):
        #               procedural = self.procedural.search(query, k=...)
        #               episodic   = self.episodic.search(query, k=...)
        # TODO(eve): 2. Convert the MemoryRecords into context messages (or a single
        #               system note) — see WorkingMemory.render.
        # TODO(eve): 3. return self.working.render(retrieved=...).
        raise NotImplementedError(
            "Implement memory recall/blend — see eve/memory/manager.py:recall"
        )

    # ── Write ────────────────────────────────────────────────────────────────
    async def remember(self, *, user: str, assistant: str) -> None:
        """Persist a completed turn across the appropriate layers."""
        # Working memory is volatile plumbing — safe to record immediately:
        self.working.add_assistant(assistant)

        # TODO(eve): 1. Always append the turn to episodic memory (what happened).
        #               e.g. await self.episodic.add(f"User: {user}\nEVE: {assistant}")
        # TODO(eve): 2. Decide whether this turn taught a durable preference/skill
        #               (e.g. "from now on, ..."). If so, await self.procedural.add(...).
        #               You might ask the LLM to extract that, or use simple heuristics.
        raise NotImplementedError(
            "Implement memory write policy — see eve/memory/manager.py:remember"
        )
