"""Shared pytest fixtures.

Enables asyncio tests and provides a Config that doesn't depend on a real `.env`,
real keys, or a running database — the smoke tests only check that the skeleton
wires together, not that any feature works.
"""

from __future__ import annotations

import pytest

from eve.config import Config


@pytest.fixture
def config() -> Config:
    """A minimal in-memory Config for tests (no .env / secrets required)."""
    return Config(
        llm_provider="anthropic",
        llm_model="anthropic/claude-opus-4-8",
        database_url="postgresql://eve:eve@localhost:5432/eve",
    )
