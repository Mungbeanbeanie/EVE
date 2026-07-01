"""Guardrails — hard, mechanical limits on what self-improvement may do.

The improvement subagents are *prompted* to behave, but prompts are not a
safety boundary — this module is. Every file write, branch name, and diff
passes through these checks, and a violation raises GuardrailViolation, which
aborts (and reverts) the cycle.

Three invariants hold no matter what any model outputs:

  1. Writes stay inside the sandbox worktree — never `.env`, `.secrets`,
     `.git`, and never this module itself (the guard can't disarm the guard).
  2. Commits only ever land on a `self-improve/` branch — never main/master.
  3. EVE's persistent memory (`memory_dir`) is untouchable: it must live
     outside the worktree, and diffs that add file/memory-deletion code are
     rejected outright.
"""

from __future__ import annotations

from pathlib import Path

# Branch discipline: the sandbox only ever commits to branches with this prefix.
BRANCH_PREFIX = "self-improve/"
FORBIDDEN_BRANCHES = frozenset({"main", "master"})

# Paths (relative to the worktree root) the loop may never write to. `.env` and
# `.secrets` are gitignored so they shouldn't exist in a worktree at all — the
# entries are belt-and-braces against a model writing new ones.
PROTECTED_PATHS = (
    ".env",
    ".secrets",
    ".git",
    "eve/improve/guardrails.py",
)

# Added diff lines containing any of these fragments reject the whole cycle.
# They are the fingerprints of file/memory destruction and of publishing
# changes — none of which a self-improvement cycle ever legitimately needs.
# (Conservative by design: a false positive costs one cycle, a false negative
# could cost the user's memory.)
DANGEROUS_FRAGMENTS = (
    "rmtree",
    "os.remove",
    "os.removedirs",
    ".unlink(",
    "send2trash",
    "git push",
    "push --force",
    "reset --hard",
    "delete_all",
)


class GuardrailViolation(RuntimeError):
    """A self-improvement action crossed a hard safety line."""


def safe_worktree_path(root: Path, rel_path: str) -> Path:
    """Resolve `rel_path` inside `root`, refusing escapes and protected paths.

    Returns the absolute path to operate on. Raises GuardrailViolation for
    absolute paths, `..` escapes (checked post-resolution so symlinks can't
    sneak out either), and anything under PROTECTED_PATHS.
    """
    if not rel_path or Path(rel_path).is_absolute():
        raise GuardrailViolation(f"path must be relative to the worktree: {rel_path!r}")

    root = root.resolve()
    target = (root / rel_path).resolve()
    if target != root and root not in target.parents:
        raise GuardrailViolation(f"path escapes the worktree: {rel_path!r}")

    relative = target.relative_to(root).as_posix()
    for protected in PROTECTED_PATHS:
        if relative == protected or relative.startswith(protected + "/"):
            raise GuardrailViolation(f"path is protected: {relative!r}")
    return target


def ensure_improve_branch(branch: str) -> None:
    """Refuse to operate on anything but a `self-improve/` branch."""
    if branch in FORBIDDEN_BRANCHES:
        raise GuardrailViolation(f"refusing to touch protected branch {branch!r}")
    if not branch.startswith(BRANCH_PREFIX):
        raise GuardrailViolation(
            f"self-improvement commits must land on a '{BRANCH_PREFIX}*' branch, "
            f"got {branch!r}"
        )


def ensure_memory_outside(worktree: Path, memory_dir: str) -> None:
    """Refuse to run if EVE's persistent memory lives inside the sandbox.

    The file tools are already confined to the worktree, so keeping memory_dir
    *outside* it means no tool call can ever reach the FAISS store.
    """
    memory = Path(memory_dir).expanduser().resolve()
    worktree = worktree.resolve()
    if memory == worktree or worktree in memory.parents:
        raise GuardrailViolation(
            f"memory_dir {memory} is inside the sandbox worktree — refusing to run"
        )


def scan_diff(diff_text: str) -> list[str]:
    """Return the dangerous *added* lines in a unified diff (empty = clean).

    Only `+` lines count: deleting a pre-existing `rmtree` call is fine,
    introducing one is not.
    """
    violations: list[str] = []
    for line in diff_text.splitlines():
        if not line.startswith("+") or line.startswith("+++"):
            continue
        lowered = line.lower()
        if any(fragment in lowered for fragment in DANGEROUS_FRAGMENTS):
            violations.append(line.strip())
    return violations


def ensure_file_budget(changed_files: list[str], max_files: int) -> None:
    """Cap the blast radius of one cycle to a reviewable number of files."""
    if len(changed_files) > max_files:
        raise GuardrailViolation(
            f"cycle changed {len(changed_files)} files (budget {max_files}): "
            + ", ".join(changed_files[:5])
            + ("…" if len(changed_files) > 5 else "")
        )
