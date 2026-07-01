"""Workspace — the git-worktree sandbox where self-improvement edits happen.

The running checkout is never edited. Each session gets a dedicated worktree
on a fresh `self-improve/<timestamp>` branch created from HEAD; all file tools
operate inside it (via guardrails), tests run inside it, and commits land on
its branch. The human reviews and merges (or deletes) the branch later — the
loop itself never pushes and never touches main.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from eve.improve import guardrails
from eve.improve.guardrails import GuardrailViolation

log = logging.getLogger(__name__)

# Commits are attributable at a glance: this identity marks machine-made changes.
_GIT_IDENTITY = ("-c", "user.name=EVE (self-improve)", "-c", "user.email=eve@selfimprove.local")

_TEST_TIMEOUT = 900  # seconds; the suite runs in ~3s today, this is headroom
# Tool-output caps exist to keep a whole subagent conversation inside a local
# model's context window (~32k tokens on Ollama): overflow makes the server
# silently drop the oldest tokens — including the system prompt — and the model
# degenerates into empty replies. 20k chars still covers every file in the
# repo today (largest ≈ 17.5k).
_READ_CAP = 20_000   # chars per read_file call
_LIST_CAP = 6_000    # chars per list_files / search_code call
_TAIL_CAP = 4_000    # chars of test output kept for prompts/journal


@dataclass
class TestResult:
    ok: bool
    output: str  # tail of combined stdout/stderr


class Workspace:
    """One sandbox worktree + its `self-improve/` branch."""

    def __init__(self, path: Path, branch: str, memory_dir: str) -> None:
        self.path = path
        self.branch = branch
        guardrails.ensure_improve_branch(branch)
        guardrails.ensure_memory_outside(path, memory_dir)

    # ── Lifecycle ─────────────────────────────────────────────────────────────
    @classmethod
    def create(cls, repo_root: Path, home: Path, memory_dir: str) -> "Workspace":
        """Add a new worktree + branch off HEAD. One per session."""
        _git(repo_root, "worktree", "prune")  # clear leftovers from deleted trees
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        branch = f"{guardrails.BRANCH_PREFIX}{stamp}"
        path = home / "worktrees" / stamp
        path.parent.mkdir(parents=True, exist_ok=True)
        _git(repo_root, "worktree", "add", "-b", branch, str(path), "HEAD")
        log.info("[improve] sandbox ready: %s on %s", path, branch)
        return cls(path, branch, memory_dir)

    # ── File tools (all guardrailed) ──────────────────────────────────────────
    def read_file(self, path: str) -> str:
        target = guardrails.safe_worktree_path(self.path, path)
        try:
            text = target.read_text(encoding="utf-8", errors="replace")
        except FileNotFoundError:
            raise GuardrailViolation(f"no such file in the sandbox: {path!r}") from None
        if len(text) > _READ_CAP:
            text = text[:_READ_CAP] + f"\n… (truncated at {_READ_CAP} chars)"
        return text

    def write_file(self, path: str, content: str) -> str:
        target = guardrails.safe_worktree_path(self.path, path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"wrote {len(content)} chars to {path}"

    def replace_in_file(self, path: str, old_text: str, new_text: str) -> str:
        """Surgical exact-match edit; errors guide the model to self-correct."""
        target = guardrails.safe_worktree_path(self.path, path)
        text = self.read_file(path)
        count = text.count(old_text)
        if count == 0:
            raise GuardrailViolation(f"old_text not found in {path!r} (must match exactly)")
        if count > 1:
            raise GuardrailViolation(
                f"old_text appears {count}× in {path!r}; include more context to make it unique"
            )
        target.write_text(text.replace(old_text, new_text, 1), encoding="utf-8")
        return f"replaced 1 occurrence in {path}"

    def list_files(self, subdir: str = "") -> str:
        base = guardrails.safe_worktree_path(self.path, subdir) if subdir else self.path
        out = _git(self.path, "ls-files", "--", str(base))
        return _clip(out, _LIST_CAP) or "(no tracked files)"

    def search_code(self, pattern: str) -> str:
        """`git grep -n` over the sandbox (fixed argv — no shell)."""
        try:
            out = _git(self.path, "grep", "-n", "-I", "--", pattern)
        except RuntimeError:
            return "(no matches)"  # git grep exits 1 on zero hits
        return _clip(out, _LIST_CAP) or "(no matches)"

    # ── Verification ──────────────────────────────────────────────────────────
    def run_tests(self) -> TestResult:
        """Run the sandbox's own test suite with the running interpreter."""
        env = dict(os.environ)
        env["PYTHONPATH"] = str(self.path) + os.pathsep + env.get("PYTHONPATH", "")
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "pytest", "-q", "--color=no"],
                cwd=self.path,
                env=env,
                capture_output=True,
                text=True,
                timeout=_TEST_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            return TestResult(ok=False, output=f"test run timed out after {_TEST_TIMEOUT}s")
        output = (proc.stdout + "\n" + proc.stderr).strip()[-_TAIL_CAP:]
        return TestResult(ok=proc.returncode == 0, output=output)

    # ── Git plumbing ──────────────────────────────────────────────────────────
    def stage(self) -> None:
        _git(self.path, "add", "-A")

    def staged_diff(self) -> str:
        return _git(self.path, "diff", "--cached")

    def diff_stat(self) -> str:
        return _git(self.path, "diff", "--cached", "--stat")

    def changed_files(self) -> list[str]:
        """Paths with staged or unstaged changes (rename → new name)."""
        out = _git(self.path, "status", "--porcelain")
        files = []
        for line in out.splitlines():
            path = line[3:].strip()
            if " -> " in path:
                path = path.split(" -> ", 1)[1]
            files.append(path.strip('"'))
        return files

    def reset(self) -> None:
        """Throw away everything uncommitted in the sandbox (never the memory)."""
        _git(self.path, "reset", "--hard", "HEAD")
        _git(self.path, "clean", "-fd")

    def commit(self, message: str, max_files: int) -> str:
        """Guard-check and commit staged+unstaged work; returns the short sha.

        This is the single choke point for persistence: branch discipline, the
        file budget, and the dangerous-diff scan all re-run here regardless of
        what the loop already checked.
        """
        current = _git(self.path, "rev-parse", "--abbrev-ref", "HEAD")
        guardrails.ensure_improve_branch(current)

        self.stage()
        files = self.changed_files()
        if not files:
            raise GuardrailViolation("nothing to commit")
        guardrails.ensure_file_budget(files, max_files)
        violations = guardrails.scan_diff(self.staged_diff())
        if violations:
            raise GuardrailViolation("dangerous diff lines: " + "; ".join(violations[:3]))

        _git(self.path, *_GIT_IDENTITY, "commit", "-m", message)
        return _git(self.path, "rev-parse", "--short", "HEAD")


def _clip(text: str, cap: int) -> str:
    return text if len(text) <= cap else text[:cap] + f"\n… (truncated at {cap} chars)"


def _git(cwd: Path, *args: str) -> str:
    """Run one git command with fixed argv (no shell) and return stdout."""
    proc = subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, timeout=120
    )
    if proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args[:2])} failed: {proc.stderr.strip()}")
    return proc.stdout.strip()
