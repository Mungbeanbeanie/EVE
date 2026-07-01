"""Tests for `eve/memory/working.py` — WorkingMemory rendering & buffer semantics.

These tests cover the subtle placement logic in ``WorkingMemory.render()``: where
retrieved long-term memories land relative to short-term (working) context, and
how the deque maxlen boundary discards oldest turns. This directly affects what
the LLM actually sees each turn — wrong ordering means the system prompt gets
buried or retrieved facts get lost.
"""

from __future__ import annotations

import pytest

from eve.memory.working import WorkingMemory


# ── render(): memory-injection ordering ───────────────────────────────────────

class TestRenderInsertionBeforeUserMessage:
    """When the last turn is a user message, retrieved memories go *before* it."""

    def test_inserts_before_last_user_turn(self) -> None:
        wm = WorkingMemory()
        wm.add_user("What's the weather?")
        wm.add_assistant("Sunny.")
        wm.add_user("How about tomorrow?")  # last turn is user

        msgs = wm.render(retrieved=[{"role": "user", "content": "User lives in Seattle."}])

        # Find indices of the memory note and the last user message.
        note_idx = next(i for i, m in enumerate(msgs) if m["role"] == "system" and "Relevant things you remember:" in m["content"])
        last_user_idx = next(i for i, m in enumerate(msgs) if m["role"] == "user" and m["content"] == "How about tomorrow?")

        assert note_idx < last_user_idx, (
            f"Memory note at index {note_idx} must come before the last user message at index {last_user_idx}"
        )

    def test_memory_note_contains_retrieved_content(self) -> None:
        wm = WorkingMemory()
        wm.add_user("Hi")
        msgs = wm.render(retrieved=[{"role": "user", "content": "Alice likes jazz."}])

        note = next(m for m in msgs if m["role"] == "system" and "Relevant things you remember:" in m["content"])
        assert "Alice likes jazz." in note["content"]


class TestRenderAppendWhenNoUserMessage:
    """When the last turn is NOT a user message, retrieved memories are *appended*."""

    def test_appends_after_assistant_turn(self) -> None:
        wm = WorkingMemory()
        wm.add_user("What's 2+2?")
        wm.add_assistant("4.")  # last turn is assistant — append path

        msgs = wm.render(retrieved=[{"role": "user", "content": "User is a math teacher."}])

        note_idx = next(i for i, m in enumerate(msgs) if m["role"] == "system" and "Relevant things you remember:" in m["content"])
        last_msg_idx = len(msgs) - 1

        assert note_idx == last_msg_idx, (
            f"With assistant as last turn, memory note should be appended at index {last_msg_idx}, got {note_idx}"
        )

    def test_appends_when_buffer_empty(self) -> None:
        """No turns in buffer → just system prompt + memory note."""
        wm = WorkingMemory()
        msgs = wm.render(retrieved=[{"role": "user", "content": "Prefers Python."}])

        system_msgs = [m for m in msgs if m["role"] == "system"]
        note = next(m for m in system_msgs if "Relevant things you remember:" in m["content"])

        assert len(system_msgs) == 2, f"Expected original system prompt + memory note (2 total), got {len(system_msgs)}"
        # Note should be the only extra message after system prompt.
        system_count = sum(1 for m in msgs if m["role"] == "system")
        assert system_count == 2, f"Expected system prompt + memory note (2 systems), got {system_count}"


class TestRenderNoInjectionForEmptyRetrieved:
    """Empty or None retrieved lists must leave context unchanged."""

    def test_none_retrieved_unchanged(self) -> None:
        wm = WorkingMemory()
        wm.add_user("Hello")
        msgs_with = wm.render(retrieved=None)
        msgs_without = wm.snapshot()  # no retrieval at all

        assert msgs_with == msgs_without, "None retrieved should produce identical output to snapshot()"

    def test_empty_list_retrieved_unchanged(self) -> None:
        wm = WorkingMemory()
        wm.add_user("Hello")
        msgs_with = wm.render(retrieved=[])
        msgs_without = wm.snapshot()

        assert msgs_with == msgs_without, "Empty retrieved list should produce identical output to snapshot()"

    def test_no_memory_note_in_output_when_empty(self) -> None:
        """Confirm the 'Relevant things you remember:' block is absent."""
        wm = WorkingMemory()
        wm.add_user("Hi")
        msgs = wm.render(retrieved=[])

        for m in msgs:
            assert "Relevant things you remember:" not in m.get("content", ""), (
                f"Unexpected memory note found: {m}"
            )


# ── deque maxlen overflow ─────────────────────────────────────────────────────

class TestDequeOverflow:
    """The rolling buffer discards oldest turns when it exceeds maxlen."""

    def test_oldest_turn_discarded_on_overflow(self) -> None:
        """With max_turns=2 (maxlen=4), adding 5 user+assistant pairs drops the first pair."""
        wm = WorkingMemory(max_turns=2)
        for i in range(5):
            wm.add_user(f"user {i}")
            wm.add_assistant(f"assist {i}")

        buffer_msgs = [m["content"] for m in wm.snapshot() if m["role"] != "system"]
        # Should retain only the last 2 turns (4 messages).
        assert len(buffer_msgs) == 4, f"Expected 4 messages after overflow, got {len(buffer_msgs)}: {buffer_msgs}"
        # Oldest turn ("user 0") must be gone.
        contents = " ".join(buffer_msgs)
        assert "user 0" not in contents, "Oldest user turn should have been discarded"
        assert "user 4" in contents, "Newest user turn should be retained"

    def test_max_turns_one(self) -> None:
        """Edge case: max_turns=1 keeps exactly one exchange."""
        wm = WorkingMemory(max_turns=1)
        for i in range(3):
            wm.add_user(f"user {i}")
            wm.add_assistant(f"assist {i}")

        buffer_msgs = [m["content"] for m in wm.snapshot() if m["role"] != "system"]
        assert len(buffer_msgs) == 2, f"max_turns=1 should keep exactly 2 messages, got {len(buffer_msgs)}"


# ── snapshot / system prompt basics ───────────────────────────────────────────

class TestSnapshotBasics:
    """Sanity checks for the simpler snapshot() path."""

    def test_system_prompt_first(self) -> None:
        wm = WorkingMemory(system_prompt="Be helpful.")
        wm.add_user("Hi")
        msgs = wm.snapshot()

        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"] == "Be helpful."

    def test_default_system_prompt_when_none(self) -> None:
        """Passing system_prompt=None uses the built-in default."""
        wm = WorkingMemory(system_prompt=None)
        assert wm.system_prompt  # should not be empty
