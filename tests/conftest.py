"""Shared pytest fixtures.

Provides a Config that doesn't depend on a real `.env` or real keys, so tests run
without any secrets or network access.
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
    )
