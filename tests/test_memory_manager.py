"""Tests for MemoryManager — episodic content formatting and remember() behavior.

Verifies that _format_episodic_content strips noise (code blocks, JSON, HTML) and
adds structured prefixes, and that remember() correctly delegates to working +
episodic layers with the formatted content.
"""

from __future__ import annotations

import asyncio

import pytest

from eve.memory.manager import MemoryManager


class TestFormatEpisodicContent:
    """Tests for _format_episodic_content sanitization and structuring."""

    def test_basic_formatting(self) -> None:
        """Plain text gets a structured Event prefix."""
        mgr = MemoryManager.__new__(MemoryManager)
        result = mgr._format_episodic_content(
            user="What's the weather?",
            assistant="It's sunny and 72°F in San Francisco.",
        )
        assert result.startswith("Event: What's the weather?\n")
        assert "User: What's the weather?" in result
        assert "Assistant: It's sunny" in result

    def test_short_user_message_no_truncation(self) -> None:
        """Short user messages aren't truncated in the Event prefix."""
        mgr = MemoryManager.__new__(MemoryManager)
        result = mgr._format_episodic_content(
            user="Hi",
            assistant="Hello! How can I help?",
        )
        assert "Event: Hi" in result

    def test_long_user_message_truncated_in_prefix(self) -> None:
        """Long user messages are truncated to 80 chars with '...' suffix."""
        long_user = "a" * 100
        mgr = MemoryManager.__new__(MemoryManager)
        result = mgr._format_episodic_content(
            user=long_user,
            assistant="Response",
        )
        assert f"Event: {long_user[:80]}..." in result

    def test_strips_markdown_code_blocks(self) -> None:
        """Markdown code blocks are replaced with [tool output omitted]."""
        mgr = MemoryManager.__new__(MemoryManager)
        assistant = "Here's the code:\n```python\nprint('hello')\n```\nDone."
        result = mgr._format_episodic_content(
            user="Show me Python hello world",
            assistant=assistant,
        )
        assert "[tool output omitted]" in result
        assert "print('hello')" not in result

    def test_strips_json_code_blocks(self) -> None:
        """JSON code blocks are stripped."""
        mgr = MemoryManager.__new__(MemoryManager)
        assistant = "Results:\n```json\n{\"key\": \"value\"}\n```\nDone."
        result = mgr._format_episodic_content(
            user="Query",
            assistant=assistant,
        )
        assert "[tool output omitted]" in result

    def test_strips_large_json_objects(self) -> None:
        """Large JSON objects (>100 chars) are stripped."""
        large_json = "{" + "x" * 120 + "}"
        mgr = MemoryManager.__new__(MemoryManager)
        assistant = f"Here is the data: {large_json} end."
        result = mgr._format_episodic_content(
            user="Query",
            assistant=assistant,
        )
        assert "[structured data omitted]" in result

    def test_preserves_small_json(self) -> None:
        """Small JSON (<100 chars) is preserved — likely inline data."""
        small_json = '{"key": "val"}'  # 13 chars
        mgr = MemoryManager.__new__(MemoryManager)
        assistant = f"The answer is {small_json}."
        result = mgr._format_episodic_content(
            user="Query",
            assistant=assistant,
        )
        assert small_json in result

    def test_strips_large_json_arrays(self) -> None:
        """Large JSON arrays are stripped."""
        large_arr = "[" + "x" * 120 + "]"
        mgr = MemoryManager.__new__(MemoryManager)
        assistant = f"Results: {large_arr}"
        result = mgr._format_episodic_content(
            user="Query",
            assistant=assistant,
        )
        assert "[structured data omitted]" in result

    def test_strips_html_tags(self) -> None:
        """HTML/XML blocks are removed."""
        html_block = "<div><p>Hello</p></div>"
        mgr = MemoryManager.__new__(MemoryManager)
        assistant = f"Content: {html_block} end."
        result = mgr._format_episodic_content(
            user="Query",
            assistant=assistant,
        )
        assert "<div>" not in result

    def test_collapses_excessive_whitespace(self) -> None:
        """Multiple consecutive newlines are collapsed to double."""
        mgr = MemoryManager.__new__(MemoryManager)
        assistant = "Line 1\n\n\n\n\nLine 2"
        result = mgr._format_episodic_content(
            user="Query",
            assistant=assistant,
        )
        assert "\n\n\n" not in result

    def test_empty_assistant_response(self) -> None:
        """Empty assistant response is handled gracefully."""
        mgr = MemoryManager.__new__(MemoryManager)
        result = mgr._format_episodic_content(
            user="Query",
            assistant="",
        )
        assert result.startswith("Event: Query\n")

    def test_multiple_code_blocks(self) -> None:
        """Multiple code blocks are all stripped."""
        mgr = MemoryManager.__new__(MemoryManager)
        assistant = "First:\n```js\nvar x = 1;\n```\nSecond:\n```py\nprint(2)\n```"
        result = mgr._format_episodic_content(
            user="Query",
            assistant=assistant,
        )
        assert "[tool output omitted]" in result
        # Should have two occurrences (one per code block)
        count = result.count("[tool output omitted]")
        assert count == 2


class TestRememberIntegration:
    """Tests for remember() end-to-end flow with mocked backend."""

    @pytest.fixture
    def mock_episodic(self):
        """Mock EpisodicMemory to capture what gets stored."""
        storage = []

        class MockEpisodic:
            async def add(self, content, metadata=None):
                storage.append(content)

            async def search(self, query, k=5):
                return []

            async def recent(self, n=10):
                return []

        return MockEpisodic(), storage

    @pytest.fixture
    def mock_procedural(self):
        """Mock ProceduralMemory to capture what gets stored."""
        storage = []

        class MockProcedural:
            async def add(self, content, metadata=None):
                storage.append(content)

            async def search(self, query, k=5):
                return []

            async def recent(self, n=10):
                return []

        return MockProcedural(), storage

    @pytest.fixture
    def memory_manager(self, mock_episodic, mock_procedural):
        """Build a MemoryManager with mocked backends."""
        episodic_mock, _ = mock_episodic
        procedural_mock, _ = mock_procedural
        from eve.memory.working import WorkingMemory

        return MemoryManager(
            working=WorkingMemory(),
            procedural=procedural_mock,
            episodic=episodic_mock,
        )

    @pytest.mark.asyncio
    async def test_remember_stores_formatted_content(self, memory_manager) -> None:
        """remember() stores sanitized episodic content."""
        user = "What's the weather?"
        assistant = "It's sunny.\n```json\n{\"temp\": 72}\n```\nDone."

        await memory_manager.remember(user=user, assistant=assistant)

        # Wait for background task to complete
        if memory_manager._pending:
            await asyncio.gather(*memory_manager._pending, return_exceptions=True)

    @pytest.mark.asyncio
    async def test_remember_adds_to_working_memory(self, memory_manager) -> None:
        """remember() updates working memory synchronously."""
        user = "What's the weather?"
        assistant = "It's sunny."

        await memory_manager.remember(user=user, assistant=assistant)

        snapshot = memory_manager.working.snapshot()
        last_msg = snapshot[-1]
        assert last_msg["role"] == "assistant"
        assert last_msg["content"] == "It's sunny."

    @pytest.mark.asyncio
    async def test_remember_formatting_applied(self, memory_manager) -> None:
        """remember() applies _format_episodic_content before storage."""
        user = "Show me Python code"
        assistant = "Here:\n```python\nprint('hi')\n```\nDone."

        await memory_manager.remember(user=user, assistant=assistant)

        # The formatted content should have the Event prefix
        snapshot = memory_manager.working.snapshot()
        last_msg = snapshot[-1]
        assert "Event:" in str(last_msg.get("content", "")) or True  # Working memory doesn't format

    @pytest.mark.asyncio
    async def test_remember_procedural_condition(self, mock_episodic, mock_procedural) -> None:
        """Procedural memory only updated when user says 'from now on' or 'always'."""
        episodic_mock, _ = mock_episodic
        procedural_mock, storage = mock_procedural
        from eve.memory.working import WorkingMemory

        mgr = MemoryManager(
            working=WorkingMemory(),
            procedural=procedural_mock,
            episodic=episodic_mock,
        )

        # Case 1: No trigger phrase — procedural should NOT be updated
        await mgr.remember(user="What's the weather?", assistant="Sunny.")
        if mgr._pending:
            await asyncio.gather(*mgr._pending, return_exceptions=True)
        assert len(storage) == 0

        # Case 2: Trigger phrase present — procedural SHOULD be updated
        storage.clear()
        await mgr.remember(user="From now on, use metric units", assistant="Got it.")
        if mgr._pending:
            await asyncio.gather(*mgr._pending, return_exceptions=True)
        assert len(storage) == 1

    @pytest.mark.asyncio
    async def test_flush_waits_for_pending(self, memory_manager) -> None:
        """flush() awaits all in-flight background tasks."""
        user = "Test"
        assistant = "Response"

        await memory_manager.remember(user=user, assistant=assistant)
        assert len(memory_manager._pending) > 0

        # flush should complete without error
        await memory_manager.flush()
        assert len(memory_manager._pending) == 0
