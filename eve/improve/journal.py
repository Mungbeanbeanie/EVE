"""Improvement journal — the human-readable audit trail of every cycle.

Full traceability is one of the safety pillars (borrowed from the Darwin Gödel
Machine): every cycle — committed, skipped, rejected, or failed — gets its own
markdown file under `<improve_home>/journal/`, and one summary line in the
`JOURNAL.md` index. The journal lives *outside* the repo so it survives branch
deletion and never dirties the working tree.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

log = logging.getLogger(__name__)

INDEX_NAME = "JOURNAL.md"


@dataclass
class CycleEntry:
    """Everything worth remembering about one improvement cycle."""

    number: int
    focus: str
    branch: str
    started: datetime = field(default_factory=datetime.now)
    reflection: str = ""  # memory insights distilled this cycle (if any)
    research: str = ""    # researcher findings / sources
    idea: str = ""        # the one idea this cycle implemented
    change: str = ""      # engineer's CHANGE/SKIP line
    diff_stat: str = ""
    files: list[str] = field(default_factory=list)
    tests: str = ""       # tail of the test run
    review: str = ""      # reviewer verdict line
    commit: str = ""      # short sha when committed
    outcome: str = "incomplete"  # committed | skipped | rejected | blocked | reverted | error

    def summary_line(self) -> str:
        """One line for the index / status tool."""
        stamp = self.started.strftime("%Y-%m-%d %H:%M")
        tail = f" — {self.change}" if self.change else ""
        sha = f" @ {self.commit}" if self.commit else ""
        return f"- {stamp} · cycle {self.number} · {self.focus} · **{self.outcome}**{sha}{tail}"

    def render(self) -> str:
        """Full markdown record of the cycle."""
        sections = [
            f"# Cycle {self.number} — {self.focus} ({self.outcome})",
            f"- started: {self.started.isoformat(timespec='seconds')}",
            f"- branch: `{self.branch}`",
            f"- commit: `{self.commit or '(none)'}`",
        ]
        for title, body in (
            ("Memory reflection", self.reflection),
            ("Research", self.research),
            ("Idea", self.idea),
            ("Change", self.change),
            ("Files", "\n".join(f"- `{f}`" for f in self.files)),
            ("Diff stat", f"```\n{self.diff_stat}\n```" if self.diff_stat else ""),
            ("Tests", f"```\n{self.tests}\n```" if self.tests else ""),
            ("Review", self.review),
        ):
            if body:
                sections.append(f"\n## {title}\n\n{body}")
        return "\n".join(sections) + "\n"


class Journal:
    """Writes cycle entries and loop-level notes under `<improve_home>/journal/`."""

    def __init__(self, home: Path) -> None:
        self.dir = home / "journal"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.index = self.dir / INDEX_NAME

    def note(self, text: str) -> None:
        """Append a loop-level event (start/stop/errors) to the index."""
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        self._append_index(f"- {stamp} · {text}")
        log.info("[improve] %s", text)

    def write_cycle(self, entry: CycleEntry) -> Path:
        """Persist one cycle's full record and index it. Returns the file path."""
        name = f"{entry.started:%Y%m%d-%H%M%S}-cycle{entry.number:03d}.md"
        path = self.dir / name
        path.write_text(entry.render(), encoding="utf-8")
        self._append_index(entry.summary_line() + f" ([details]({name}))")
        log.info("[improve] cycle %d → %s (%s)", entry.number, entry.outcome, path.name)
        return path

    def tail(self, n: int = 5) -> list[str]:
        """Last `n` index lines — the loop's recent history at a glance."""
        try:
            lines = self.index.read_text(encoding="utf-8").strip().splitlines()
        except OSError:
            return []
        return [line for line in lines if line.startswith("- ")][-n:]

    def _append_index(self, line: str) -> None:
        header = "" if self.index.exists() else "# EVE self-improvement journal\n\n"
        with self.index.open("a", encoding="utf-8") as fh:
            fh.write(header + line + "\n")
