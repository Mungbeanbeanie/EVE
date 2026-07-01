"""Workspace — the git-worktree sandbox: isolation, guardrails, commit path."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from eve.improve.guardrails import GuardrailViolation
from eve.improve.workspace import Workspace


def _git_out(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


@pytest.fixture
def workspace(tmp_repo: Path, tmp_path: Path) -> Workspace:
    return Workspace.create(tmp_repo, tmp_path / "improve-home", str(tmp_path / "memory"))


def test_create_makes_a_self_improve_branch_off_head(tmp_repo: Path, workspace: Workspace):
    assert workspace.branch.startswith("self-improve/")
    assert workspace.path.exists()
    # The original checkout is untouched: still on main, same tip.
    assert _git_out(tmp_repo, "rev-parse", "--abbrev-ref", "HEAD") == "main"
    assert _git_out(workspace.path, "rev-parse", "HEAD") == _git_out(tmp_repo, "rev-parse", "main")


def test_memory_inside_the_sandbox_is_refused(tmp_repo: Path, tmp_path: Path):
    home = tmp_path / "home"
    ws = Workspace.create(tmp_repo, home, str(tmp_path / "memory"))
    with pytest.raises(GuardrailViolation):
        Workspace(ws.path, ws.branch, memory_dir=str(ws.path / "mem"))


def test_write_and_commit_land_on_the_branch_not_main(tmp_repo: Path, workspace: Workspace):
    main_before = _git_out(tmp_repo, "rev-parse", "main")
    workspace.write_file("NOTES.md", "sandbox says hi\n")
    sha = workspace.commit("self-improve: add notes", max_files=5)
    assert sha
    # Exactly one commit ahead of main, authored by the self-improve identity…
    log = _git_out(tmp_repo, "log", "--oneline", f"main..{workspace.branch}")
    assert len(log.splitlines()) == 1 and "add notes" in log
    # …and main did not move.
    assert _git_out(tmp_repo, "rev-parse", "main") == main_before


def test_protected_paths_are_rejected(workspace: Workspace):
    with pytest.raises(GuardrailViolation):
        workspace.write_file(".env", "LLM_API_KEY=stolen")
    with pytest.raises(GuardrailViolation):
        workspace.read_file("../../outside.txt")


def test_dangerous_diffs_never_commit(workspace: Workspace):
    workspace.write_file("cleanup.py", "import shutil\nshutil.rmtree(memory_dir)\n")
    with pytest.raises(GuardrailViolation, match="dangerous"):
        workspace.commit("self-improve: cleanup", max_files=5)


def test_file_budget_blocks_oversized_commits(workspace: Workspace):
    for i in range(3):
        workspace.write_file(f"f{i}.txt", "x\n")
    with pytest.raises(GuardrailViolation, match="budget"):
        workspace.commit("self-improve: too much", max_files=2)


def test_replace_in_file_is_surgical(workspace: Workspace):
    assert "replaced 1" in workspace.replace_in_file("hello.py", '"hi"', '"hello"')
    assert 'GREETING = "hello"' in workspace.read_file("hello.py")
    with pytest.raises(GuardrailViolation, match="not found"):
        workspace.replace_in_file("hello.py", "no such text", "x")


def test_reset_discards_everything_uncommitted(workspace: Workspace):
    workspace.write_file("junk.txt", "scratch\n")
    workspace.replace_in_file("hello.py", '"hi"', '"junk"')
    workspace.reset()
    assert workspace.changed_files() == []
    assert '"hi"' in workspace.read_file("hello.py")


def test_run_tests_passes_in_a_healthy_sandbox(workspace: Workspace):
    result = workspace.run_tests()
    assert result.ok, result.output
