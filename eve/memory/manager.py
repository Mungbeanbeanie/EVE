"""MemoryManager — the single memory façade the Agent uses.

It owns all three layers and exposes two verbs:

    recall(query)    -> assemble the message list to send the LLM, blending the
                        working window with relevant procedural + episodic memories.
    remember(...)    -> write a turn to the right layers (always episodic/working;
                        procedural only when something durable was learned).
"""

from __future__ import annotations
import asyncio
import logging

from eve.config import Config
from eve.llm.base import Message
from eve.memory.episodic import EpisodicMemory
from eve.memory.mem0_backend import Mem0Backend
from eve.memory.procedural import ProceduralMemory
from eve.memory.working import WorkingMemory

log = logging.getLogger(__name__)


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
        # In-flight background persistence tasks. asyncio only keeps weak refs to
        # tasks, so we hold strong refs here to stop them being GC'd mid-write.
        self._pending: set[asyncio.Task] = set()

    @classmethod
    def from_config(cls, config: Config) -> "MemoryManager":
        """Build the manager and all three layers from config."""
        backend = Mem0Backend(config)  # shared mem0 client: FAISS index + FastEmbed (lazy)
        return cls(
            working=WorkingMemory(),
            procedural=ProceduralMemory(backend),
            episodic=EpisodicMemory(backend),
        )

    # ── Read ─────────────────────────────────────────────────────────────────
    async def recall(self, query: str) -> list[Message]:
        """Return the full message list for the LLM, with long-term memory blended in.

        Long-term recall is best-effort: if the mem0 backend (embedder/index) is
        unavailable, we still return the live working window so the conversation
        keeps working instead of crashing the turn.
        """
        retrieved: list[Message] | None = None
        try:
            proc_records, epis_records = await asyncio.gather(
                self.procedural.search(query, k=5),
                self.episodic.search(query, k=5),
            )
            retrieved = [
                {"role": "system", "content": r.content}
                for r in (*proc_records, *epis_records)
            ]
        except Exception as exc:  # backend down / not configured — degrade, don't die
            log.warning("Long-term recall unavailable, using working memory only: %s", exc)

        return self.working.render(retrieved=retrieved or None)

    # ── Write ────────────────────────────────────────────────────────────────
    async def remember(self, *, user: str, assistant: str) -> None:
        """Record a completed turn.

        Working memory is updated synchronously (the next turn's recall needs it
        immediately). The slow long-term persistence — mem0 runs LLM fact
        extraction + embedding, which can take tens of seconds — is fired off in
        the background so it never blocks the conversation. Long-term memory is
        therefore eventually consistent: a fact written this turn may not be
        vector-searchable until its background write finishes. Call `flush()` to
        await outstanding writes (the Agent does this on shutdown).
        """
        self.working.add_assistant(assistant)

        task = asyncio.create_task(self._persist_longterm(user=user, assistant=assistant))
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)

    async def _persist_longterm(self, *, user: str, assistant: str) -> None:
        """Best-effort durable write of one turn (runs in the background)."""
        try:
            await self.episodic.add(f"User: {user}\nEVE: {assistant}")

            if "from now on" in user.lower() or "always" in user.lower():
                await self.procedural.add(f"User: {user}\nEVE: {assistant}")
        except Exception as exc:  # backend down / not configured — degrade, don't die
            log.warning("Long-term persistence unavailable, skipping: %s", exc)

    async def flush(self) -> None:
        """Wait for any in-flight background persistence to finish.

        Call before shutdown so the last turn(s) aren't lost when the event loop
        closes. Safe to call when nothing is pending.

        Snapshot into a local list first — ``self._pending`` is mutated by each
        task's done-callback (which calls ``discard()``), and unpacking the set
        directly in ``gather(*self._pending, ...)`` would iterate over it while
        callbacks fire, risking a mutation-during-iteration error or silently
        dropping writes that race in after this flush started. New writes arriving
        mid-flush are intentionally deferred to the next call — they were created
        *after* we decided what "last" means.
        """
        if self._pending:
            tasks = list(self._pending)
            await asyncio.gather(*tasks, return_exceptions=True)
