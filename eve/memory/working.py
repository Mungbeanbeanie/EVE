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
        # Each entry is a chat Message dict: {"role": ..., "content": ...}. A "turn"
        # is a user message + an assistant reply, so the buffer holds 2× max_turns
        # messages to actually retain the last `max_turns` exchanges.
        self._buffer: deque[Message] = deque(maxlen=max_turns * 2)
        self.system_prompt = system_prompt or "You are EVE, a friend and informal personal assistant."

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
        Token budgeting / overflow summarisation is not yet implemented.
        """
        curr_messages = self.snapshot()
        if retrieved:
            retrieved_content = "\n".join(f"- {msg['content']}" for msg in retrieved)
            memory_note: Message = {
                "role": "system",
                "content": f"Relevant things you remember:\n{retrieved_content}",
            }
            if curr_messages and curr_messages[-1]["role"] == "user":
                curr_messages.insert(-1, memory_note)
            else:
                curr_messages.append(memory_note)
        return curr_messages
