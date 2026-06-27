"""Working memory — the live, volatile conversation window.

Think of this as the agent's short-term "RAM": the recent turns that are always in
context, no database involved. The basic append/snapshot plumbing is implemented so
the loop runs; the *interesting* parts (token-budgeting, summarizing overflow) are
left for you.
"""

from __future__ import annotations

from collections import deque

from eve.llm.base import Message


class WorkingMemory:
    """A bounded rolling buffer of recent chat messages."""

    def __init__(self, max_turns: int = 20, system_prompt: str | None = None) -> None:
        # Each entry is a chat Message dict: {"role": ..., "content": ...}
        self._buffer: deque[Message] = deque(maxlen=max_turns)
        self.system_prompt = system_prompt or "You are EVE, a helpful personal assistant."

    # ── Writes (plumbing — implemented) ──────────────────────────────────────
    def add_user(self, text: str) -> None:
        self._buffer.append({"role": "user", "content": text})

    def add_assistant(self, text: str) -> None:
        self._buffer.append({"role": "assistant", "content": text})

    def snapshot(self) -> list[Message]:
        """Return current messages as a list (system prompt first)."""
        return [{"role": "system", "content": self.system_prompt}, *self._buffer]

    # ── Context shaping (the learning part) ──────────────────────────────────
    def render(self, retrieved: list[Message] | None = None) -> list[Message]:
        """Build the final message list sent to the LLM.

        This is where short-term (working) and long-term (retrieved) memory meet.
        """
        # TODO(eve): 1. Start from self.snapshot().
        # TODO(eve): 2. Weave in `retrieved` long-term memories — e.g. as a system
        #               note ("Relevant things you remember: ...") placed before the
        #               latest user turn so the model uses them.
        # TODO(eve): 3. Enforce a token budget: if too long, summarize or drop the
        #               oldest turns (this is why max_turns alone isn't enough).
        # TODO(eve): 4. Return the assembled list[Message].
        raise NotImplementedError(
            "Implement context assembly — see eve/memory/working.py:render"
        )
