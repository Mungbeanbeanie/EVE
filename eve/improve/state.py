"""Persistent loop state — backlog, counters, and history across sessions.

A tiny JSON file under `<improve_home>` (NOT inside the repo, NOT inside EVE's
conversational memory store). It lets the loop rotate focus areas, avoid
re-doing past work, and reuse research across restarts. Corruption or absence
degrades to a fresh state — the loop must never crash over bookkeeping.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

_HISTORY_CAP = 50  # enough context to avoid repeats without unbounded growth


@dataclass
class ImproveState:
    """What the loop remembers between cycles and sessions."""

    cycles_completed: int = 0
    backlog: list[str] = field(default_factory=list)  # improvement ideas, FIFO
    history: list[dict] = field(default_factory=list)  # {cycle, focus, outcome, change}
    last_reflection: str = ""  # ISO timestamp of the last memory reflection

    def record(self, *, cycle: int, focus: str, outcome: str, change: str) -> None:
        self.history.append(
            {"cycle": cycle, "focus": focus, "outcome": outcome, "change": change}
        )
        self.history = self.history[-_HISTORY_CAP:]
        self.cycles_completed = cycle

    def history_digest(self, n: int = 8) -> str:
        """Recent outcomes as prompt context, so subagents don't repeat work."""
        if not self.history:
            return "(no previous cycles)"
        return "\n".join(
            f"- cycle {h['cycle']} ({h['focus']}): {h['outcome']} — {h['change']}"
            for h in self.history[-n:]
        )


def load_state(path: Path) -> ImproveState:
    """Read state from disk; any problem yields a clean slate (never raises)."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return ImproveState(**data)
    except FileNotFoundError:
        return ImproveState()
    except Exception as exc:  # corrupt/incompatible file — start over, keep going
        log.warning("[improve] unreadable state at %s (%s); starting fresh", path, exc)
        return ImproveState()


def save_state(path: Path, state: ImproveState) -> None:
    """Atomically persist state (write-then-rename survives a crash mid-write)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(asdict(state), indent=2), encoding="utf-8")
    tmp.replace(path)
