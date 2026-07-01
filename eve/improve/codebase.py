"""Codebase awareness — a compact map of EVE's own source for prompts.

Subagents can't read every file into a 35B model's context each cycle, so we
give them an oriented map: every tracked file with its size and (for Python)
the first line of its module docstring. That is enough for the model to decide
what to `read_file` next.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

# Keep the map prompt-sized even if the repo grows.
_MAX_FILES = 400
_MAX_CHARS = 8000


def repo_root(start: Path | None = None) -> Path | None:
    """Return the git repo root containing `start` (default: this package)."""
    origin = start or Path(__file__).resolve().parent
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=origin,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except OSError:
        return None
    if out.returncode != 0:
        return None
    return Path(out.stdout.strip())


def describe(root: Path) -> str:
    """Render the tracked file tree with one-line summaries, prompt-sized."""
    try:
        out = subprocess.run(
            ["git", "ls-files"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=10,
        )
        files = sorted(out.stdout.split()) if out.returncode == 0 else []
    except OSError:
        files = []
    if not files:
        return "(codebase map unavailable)"

    lines: list[str] = []
    for name in files[:_MAX_FILES]:
        path = root / name
        try:
            size = path.stat().st_size
        except OSError:
            continue  # deleted in the worktree but still tracked
        summary = _first_docstring_line(path) if name.endswith(".py") else ""
        lines.append(f"{name} ({size}B)" + (f" — {summary}" if summary else ""))

    text = "\n".join(lines)
    if len(text) > _MAX_CHARS:
        text = text[:_MAX_CHARS] + "\n… (map truncated)"
    return text


def _first_docstring_line(path: Path) -> str:
    """First line of a module docstring, or "" if there isn't an obvious one."""
    try:
        head = path.read_text(encoding="utf-8", errors="replace")[:2000]
    except OSError:
        return ""
    for quote in ('"""', "'''"):
        start = head.find(quote)
        if start != -1:
            first_line = head[start + 3 :].lstrip().splitlines()
            return first_line[0].strip() if first_line else ""
    return ""
