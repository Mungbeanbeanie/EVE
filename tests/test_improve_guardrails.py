"""Guardrails — the hard safety lines must hold regardless of model output."""

from __future__ import annotations

from pathlib import Path

import pytest

from eve.improve import guardrails
from eve.improve.guardrails import GuardrailViolation


# ── Path confinement ───────────────────────────────────────────────────────────
def test_normal_paths_resolve_inside_the_worktree(tmp_path: Path):
    target = guardrails.safe_worktree_path(tmp_path, "eve/agent.py")
    assert target == tmp_path.resolve() / "eve" / "agent.py"


@pytest.mark.parametrize(
    "bad",
    [
        "../outside.txt",          # relative escape
        "/etc/passwd",             # absolute path
        "",                        # empty
        ".env",                    # secrets
        ".secrets/google.json",    # secrets dir
        ".git/config",             # git internals
        "eve/improve/guardrails.py",  # the guard can't disarm the guard
    ],
)
def test_escapes_and_protected_paths_are_rejected(tmp_path: Path, bad: str):
    with pytest.raises(GuardrailViolation):
        guardrails.safe_worktree_path(tmp_path, bad)


# ── Branch discipline ──────────────────────────────────────────────────────────
def test_only_self_improve_branches_are_allowed():
    guardrails.ensure_improve_branch("self-improve/20260701-120000")
    for forbidden in ("main", "master", "feature/x", "self-improve"):
        with pytest.raises(GuardrailViolation):
            guardrails.ensure_improve_branch(forbidden)


# ── Memory is untouchable ──────────────────────────────────────────────────────
def test_memory_inside_the_sandbox_refuses_to_run(tmp_path: Path):
    with pytest.raises(GuardrailViolation):
        guardrails.ensure_memory_outside(tmp_path, str(tmp_path / "memory"))


def test_memory_outside_the_sandbox_is_fine(tmp_path: Path):
    guardrails.ensure_memory_outside(tmp_path / "worktree", str(tmp_path / "memory"))


# ── Dangerous-diff scan ────────────────────────────────────────────────────────
def test_added_deletion_code_is_flagged():
    diff = (
        "--- a/eve/memory/manager.py\n+++ b/eve/memory/manager.py\n"
        "+    shutil.rmtree(self.memory_dir)\n"
        "+    os.remove(index_path)\n"
    )
    violations = guardrails.scan_diff(diff)
    assert len(violations) == 2


def test_removed_lines_and_context_do_not_count():
    diff = (
        "--- a/x.py\n+++ b/x.py\n"
        "-    shutil.rmtree(old)\n"      # deleting a deletion is fine
        "     rmtree_mentioned_in_context()\n"
        "+    path.write_text(data)\n"   # benign addition
    )
    assert guardrails.scan_diff(diff) == []


# ── File budget ────────────────────────────────────────────────────────────────
def test_file_budget_caps_the_blast_radius():
    guardrails.ensure_file_budget(["a.py", "b.py"], max_files=2)
    with pytest.raises(GuardrailViolation):
        guardrails.ensure_file_budget(["a.py", "b.py", "c.py"], max_files=2)
