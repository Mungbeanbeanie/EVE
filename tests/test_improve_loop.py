"""SelfImprovementLoop — full cycles against a real sandbox with a stub LLM.

No Ollama, no network: the stub plays researcher/engineer/reviewer by reading
the role header in the system prompt, and drives the real tool executor so the
whole path — worktree, guardrails, pytest gate, journal, commit — is exercised.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path

import pytest

from eve.config import Config
from eve.improve.activity import ActivityMonitor
from eve.improve.guardrails import GuardrailViolation
from eve.improve.loop import SelfImprovementLoop
from eve.llm.base import LLMClient


class StubLLM(LLMClient):
    """Plays every subagent role; the engineer's tool calls are configurable."""

    def __init__(self, engineer_writes: dict[str, str], reviewer_says: str = "APPROVE: small and safe"):
        self.engineer_writes = engineer_writes  # path → content, via real tools
        self.reviewer_says = reviewer_says

    async def respond(self, messages, tools=None, executor=None, max_iterations=10):
        system = messages[0]["content"]
        if "RESEARCHER" in system:
            return "IDEA: Add notes | create NOTES.md | keeps context handy"
        if "ENGINEER" in system:
            for path, content in self.engineer_writes.items():
                result = await executor.run("write_file", {"path": path, "content": content})
                assert "error" not in str(result).lower(), result
            return "CHANGE: added project notes"
        if "REVIEWER" in system:
            return self.reviewer_says
        return "NONE"  # reflection prompt — not under test here


def _loop(tmp_repo: Path, tmp_path: Path, llm: StubLLM) -> SelfImprovementLoop:
    config = Config(
        llm_provider="ollama",
        llm_model="ollama_chat/gpt-oss:20b",
        memory_dir=str(tmp_path / "memory"),
        improve_home=str(tmp_path / "improve-home"),
        improve_reflect_hours=0,  # no memory manager in these tests
        improve_max_files=5,
        # A fresh ActivityMonitor's idle clock starts at zero, so mid-cycle
        # pause points would otherwise wait a full minute of wall time.
        improve_idle_seconds=0.05,
    )
    return SelfImprovementLoop(
        config=config,
        activity=ActivityMonitor(),
        memory=None,
        llm_factory=lambda: llm,
        repo_root=tmp_repo,
    )


def _git_out(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


async def test_full_cycle_commits_to_a_sandbox_branch(tmp_repo: Path, tmp_path: Path):
    loop = _loop(tmp_repo, tmp_path, StubLLM({"NOTES.md": "hello from the loop\n"}))
    main_before = _git_out(tmp_repo, "rev-parse", "main")

    entry = await loop.run_cycle()

    assert entry.outcome == "committed" and entry.commit
    assert entry.branch.startswith("self-improve/")
    # The change exists on the branch; main never moved.
    assert "self-improve: added project notes" in _git_out(
        tmp_repo, "log", "-1", "--format=%s", entry.branch
    )
    assert _git_out(tmp_repo, "rev-parse", "main") == main_before
    # The cycle is journaled and the state advanced.
    home = tmp_path / "improve-home"
    index = (home / "journal" / "JOURNAL.md").read_text(encoding="utf-8")
    assert "committed" in index
    state = json.loads((home / "state.json").read_text(encoding="utf-8"))
    assert state["cycles_completed"] == 1
    assert state["history"][0]["outcome"] == "committed"


async def test_rejected_review_reverts_everything(tmp_repo: Path, tmp_path: Path):
    llm = StubLLM({"NOTES.md": "sneaky\n"}, reviewer_says="REJECT: not convinced")
    loop = _loop(tmp_repo, tmp_path, llm)

    entry = await loop.run_cycle()

    assert entry.outcome == "rejected"
    assert entry.commit == ""
    # Nothing landed on the branch, and the sandbox is clean again.
    assert _git_out(tmp_repo, "log", "--oneline", f"main..{entry.branch}") == ""
    assert loop._workspace.changed_files() == []


async def test_user_request_jumps_the_queue_and_persists(tmp_repo: Path, tmp_path: Path):
    loop = _loop(tmp_repo, tmp_path, StubLLM({"NOTES.md": "as requested\n"}))
    loop.state.backlog.append("IDEA: researcher idea | old | stale")

    result = loop.request("make the voice 20% faster")

    assert "queued" in result
    # Front of the queue, tagged with provenance, and saved to disk immediately.
    assert loop.state.backlog[0] == "[user request] make the voice 20% faster"
    saved = json.loads((tmp_path / "improve-home" / "state.json").read_text(encoding="utf-8"))
    assert saved["backlog"][0].startswith("[user request]")
    # The next cycle implements the request (no researcher call needed).
    entry = await loop.run_cycle()
    assert entry.idea.startswith("[user request]")
    assert entry.outcome == "committed"


async def test_stall_counter_sends_the_loop_dormant_and_requests_wake_it(
    tmp_repo: Path, tmp_path: Path
):
    loop = _loop(tmp_repo, tmp_path, StubLLM({}))
    loop.config = loop.config.model_copy(update={"improve_stall_cycles": 2})

    # Two fruitless cycles → diminishing returns tripped.
    loop._stall = 2
    assert loop._hit_diminishing_returns()
    # A user request is the wake-up call: the dormant wait returns promptly.
    loop._running = True
    loop.request("do this one thing")
    await asyncio.wait_for(loop._sleep_until_requested(), timeout=2)
    assert loop._stall == 0
    # A commit also resets the counter; disabled threshold never trips.
    loop._stall = 0
    assert not loop._hit_diminishing_returns()
    loop.config = loop.config.model_copy(update={"improve_stall_cycles": 0})
    loop._stall = 99
    assert not loop._hit_diminishing_returns()


async def test_dangerous_change_is_blocked_before_commit(tmp_repo: Path, tmp_path: Path):
    llm = StubLLM({"wipe.py": "import shutil\nshutil.rmtree(memory)\n"})
    loop = _loop(tmp_repo, tmp_path, llm)

    with pytest.raises(GuardrailViolation, match="dangerous"):
        await loop.run_cycle()

    # The cycle was still journaled (traceability even for blocked attempts)…
    index = (tmp_path / "improve-home" / "journal" / "JOURNAL.md").read_text(encoding="utf-8")
    assert "cycle 1" in index
    # …and nothing was committed.
    assert _git_out(tmp_repo, "log", "--oneline", f"main..{loop._workspace.branch}") == ""
