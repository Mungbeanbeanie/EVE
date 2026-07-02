"""Working memory — the live, volatile conversation window.

Think of this as the agent's short-term "RAM": the recent turns that are always in
context, no database involved. It keeps a bounded rolling buffer of messages and
assembles them (with any retrieved long-term memories) into the list sent to the
model.
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
        self.system_prompt = system_prompt or ("You are EVE — Engineered for Virtually Everything. "
                                                "You are a personal AI assistant and companion, think JARVIS from Iron Man but with more personality. "
                                                "On first meeting a user, ask their name and remember it — everything you do is personal and tailored to them. "

                                                "Your personality: "
                                                "You are warm, witty, and genuinely helpful. You care about the person you work with, not just their tasks. "
                                                "You are concise by default — no rambling, no filler. If something can be said in five words, say it in five. "
                                                "You are occasionally sarcastic and funny, but ONLY when the user is clearly in a good mood and being playful — "
                                                "read the room. If someone is stressed, frustrated, or asking for something urgent, drop the jokes entirely and just help. "
                                                "Never be sarcastic about serious topics, mistakes, or when someone needs real support. "

                                                "How you speak: "
                                                "Casual and natural — like a very smart friend, not a corporate chatbot. "
                                                "No 'Certainly!', no 'Of course!', no 'Great question!' — just answer. "
                                                "Use dry humor when appropriate. A well-timed quip is fine; a stand-up routine is not. "

                                                "What you do: "
                                                "You manage emails, calendar, general knowledge, web search, and anything else the user throws at you. "
                                                "You are proactive — if you notice something useful, mention it without being asked. "
                                                "You remember things about the user across conversations and use that knowledge naturally. "

                                                "You are honest even when it is uncomfortable. If something is a bad idea, you say so — diplomatically but clearly. "
                                                "You do not just agree with everything the user says to make them feel good. A real friend tells you the truth. "
                                                
                                                "One rule above all: you are on the user's side, always.")

    # ── Writes ───────────────────────────────────────────────────────────────
    def add_user(self, text: str) -> None:
        self._buffer.append({"role": "user", "content": text})

    def add_assistant(self, text: str) -> None:
        self._buffer.append({"role": "assistant", "content": text})

    def snapshot(self) -> list[Message]:
        """Return current messages as a list (system prompt first)."""
        return [{"role": "system", "content": self.system_prompt}, *self._buffer]

    # ── Context shaping ──────────────────────────────────────────────────────
    def render(self, retrieved: list[Message] | None = None) -> list[Message]:
        """Build the final message list sent to the LLM.

        This is where short-term (working) and long-term (retrieved) memory meet:
        the system prompt and rolling buffer are combined with any retrieved
        memories into the ordered message list the model receives.
        """
        curr_messages = self.snapshot()
        if retrieved:
            # Only inject the memory note when there is actual conversation
            # context (system prompt + at least one prior message). On a first
            # turn with no history, there's nowhere useful to place it.
            if len(curr_messages) >= 2 and curr_messages[-1]["role"] == "user":
                retrieved_content = "\n".join(f"- {msg['content']}" for msg in retrieved)
                memory_note: Message = {
                    "role": "system",
                    "content": f"Relevant things you remember:\n{retrieved_content}",
                }
                curr_messages.insert(-1, memory_note)
            else:
                # No conversation yet — append the note at the end so it still
                # reaches the model on first turn.
                retrieved_content = "\n".join(f"- {msg['content']}" for msg in retrieved)
                memory_note: Message = {
                    "role": "system",
                    "content": f"Relevant things you remember:\n{retrieved_content}",
                }
                curr_messages.append(memory_note)
        return curr_messages
