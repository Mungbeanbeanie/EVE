"""Shared pytest fixtures.

Provides a Config that doesn't depend on a real `.env` or real keys, so tests run
without any secrets or network access.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from eve.config import Config


@pytest.fixture
def config() -> Config:
    """A minimal in-memory Config for tests (no .env / secrets required)."""
    return Config(
        llm_provider="anthropic",
        llm_model="anthropic/claude-opus-4-8",
        self_improve=False,  # never let a developer's .env spin up the daemon in tests
    )


@pytest.fixture
def tmp_repo(tmp_path: Path) -> Path:
    """A throwaway git repo with one commit and a passing test suite.

    Used by the self-improvement tests: small enough that a real `pytest`
    subprocess inside it finishes in about a second.
    """
    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args: str) -> None:
        subprocess.run(
            ["git", "-c", "user.name=test", "-c", "user.email=test@test", *args],
            cwd=repo, check=True, capture_output=True,
        )

    git("init", "-b", "main")
    (repo / "hello.py").write_text('GREETING = "hi"\n', encoding="utf-8")
    (repo / "tests").mkdir()
    (repo / "tests" / "test_ok.py").write_text(
        "def test_ok():\n    assert True\n", encoding="utf-8"
    )
    git("add", "-A")
    git("commit", "-m", "initial")
    return repo
