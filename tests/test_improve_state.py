"""Tests for ``eve.improve.state`` — persistent loop state serialization.

The improvement daemon survives restarts by persisting backlog, counters and
history to a JSON file under ``improve_home``. These tests guarantee that:

* A missing file yields clean defaults (the loop never crashes on first boot).
* Saving + reloading round-trips every field faithfully.
* History is truncated to the configured cap so it can't grow unbounded.
* Corrupted or incompatible JSON degrades gracefully — no exceptions leak.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eve.improve.state import ImproveState, _HISTORY_CAP, load_state, save_state


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def state_file(tmp_path: Path) -> Path:
    """A fresh state file path in a throwaway directory."""
    return tmp_path / "state.json"


def _make_state(
    cycles_completed: int = 5,
    backlog: list[str] | None = None,
    history: list[dict] | None = None,
    last_reflection: str = "2024-06-15T10:30:00",
) -> ImproveState:
    """Build an ImproveState with sensible defaults for round-trip tests."""
    return ImproveState(
        cycles_completed=cycles_completed,
        backlog=backlog or ["fix-tts-latency", "audit-guardrails"],
        history=history or [
            {"cycle": 1, "focus": "memory", "outcome": "OK", "change": "added working"},
            {"cycle": 2, "focus": "pipeline", "outcome": "SKIP", "change": "no-op"},
        ],
        last_reflection=last_reflection,
    )


# ---------------------------------------------------------------------------
# (a) Loading from missing file returns defaults
# ---------------------------------------------------------------------------

def test_load_missing_file_returns_defaults(state_file: Path):
    """A non-existent state file must yield a fresh ImproveState — no crash."""
    assert not state_file.exists()
    state = load_state(state_file)
    assert isinstance(state, ImproveState)
    assert state.cycles_completed == 0
    assert state.backlog == []
    assert state.history == []
    assert state.last_reflection == ""


def test_load_missing_dir_returns_defaults(tmp_path: Path):
    """Even a non-existent parent directory must not raise."""
    missing = tmp_path / "no_such_dir" / "state.json"
    state = load_state(missing)
    assert isinstance(state, ImproveState)
    assert state.cycles_completed == 0


# ---------------------------------------------------------------------------
# (b) Save + reload round-trips faithfully
# ---------------------------------------------------------------------------

def test_save_then_load_round_trip(state_file: Path):
    """Every field must survive a save → load cycle unchanged."""
    original = _make_state()
    save_state(state_file, original)

    loaded = load_state(state_file)
    assert loaded.cycles_completed == original.cycles_completed
    assert loaded.backlog == original.backlog
    assert loaded.history == original.history
    assert loaded.last_reflection == original.last_reflection


def test_save_creates_intermediate_dirs(state_file: Path):
    """Parent directories that don't exist should be created, not error."""
    nested = state_file.parent / "a" / "b" / "c" / "state.json"
    save_state(nested, _make_state())
    assert nested.exists()
    loaded = load_state(nested)
    assert loaded.cycles_completed == 5


def test_save_uses_atomic_rename(state_file: Path):
    """save_state writes to a .tmp then renames — the final file is complete."""
    save_state(state_file, _make_state())
    # The .tmp should have been renamed away.
    assert not state_file.with_suffix(".tmp").exists()
    # And the JSON must be parseable.
    data = json.loads(state_file.read_text(encoding="utf-8"))
    assert data["cycles_completed"] == 5


def test_load_corrupt_tmp_ignored(tmp_path: Path):
    """A leftover .tmp file should not interfere with a valid state.json."""
    sf = tmp_path / "state.json"
    save_state(sf, _make_state())
    # Simulate a crash mid-write by leaving a .tmp that shadows nothing.
    (sf.with_suffix(".tmp")).write_text("garbage", encoding="utf-8")
    loaded = load_state(sf)
    assert loaded.cycles_completed == 5


# ---------------------------------------------------------------------------
# (c) History truncation keeps most recent N entries
# ---------------------------------------------------------------------------

def test_history_truncation_keeps_most_recent(state_file: Path):
    """After many record() calls, history should contain at most _HISTORY_CAP."""
    state = ImproveState()
    for i in range(_HISTORY_CAP + 20):
        state.record(cycle=i, focus=f"f{i}", outcome="ok", change=f"c{i}")

    # History must be truncated to exactly _HISTORY_CAP.
    assert len(state.history) == _HISTORY_CAP

    # The oldest entries (cycle < 20) should be gone; newest (>= 20) remain.
    cycles = [h["cycle"] for h in state.history]
    assert min(cycles) == 20
    assert max(cycles) == _HISTORY_CAP + 19


def test_history_truncation_keeps_latest_entries(state_file: Path):
    """Truncation should keep the *most recent* entries, not oldest."""
    state = ImproveState()
    for i in range(100):
        state.record(cycle=i, focus="x", outcome="ok", change="y")

    # First entry should be cycle 50 (100 - 50), not 0.
    assert state.history[0]["cycle"] == 50
    # Last entry should be cycle 99.
    assert state.history[-1]["cycle"] == 99


def test_cycles_completed_tracks_latest_record(state_file: Path):
    """cycles_completed must reflect the most recent record() call."""
    state = ImproveState()
    state.record(cycle=7, focus="a", outcome="ok", change="b")
    state.record(cycle=12, focus="c", outcome="fail", change="d")
    assert state.cycles_completed == 12


def test_history_digest_returns_string(state_file: Path):
    """history_digest() must always return a string (for prompt injection)."""
    state = ImproveState()
    assert state.history_digest() == "(no previous cycles)"

    state.record(cycle=1, focus="test", outcome="OK", change="x")
    digest = state.history_digest(n=8)
    assert isinstance(digest, str)
    assert "cycle 1" in digest
    assert "test" in digest


def test_history_digest_respects_n_param(state_file: Path):
    """history_digest(n=N) should include at most N entries."""
    state = ImproveState()
    for i in range(20):
        state.record(cycle=i, focus="f", outcome="o", change="c")

    # Request only 3 → digest should have exactly 3 lines.
    digest = state.history_digest(n=3)
    lines = [line for line in digest.splitlines() if line.strip()]
    assert len(lines) == 3


# ---------------------------------------------------------------------------
# (d) Corrupted JSON falls back to defaults without crashing
# ---------------------------------------------------------------------------

def test_corrupt_json_returns_defaults(state_file: Path):
    """Garbage JSON must not raise — should yield a fresh ImproveState."""
    state_file.write_text("{this is not json!!!", encoding="utf-8")
    state = load_state(state_file)
    assert isinstance(state, ImproveState)
    assert state.cycles_completed == 0
    assert state.backlog == []


def test_empty_json_returns_defaults(state_file: Path):
    """An empty JSON file ({} ) should yield defaults."""
    state_file.write_text("{}", encoding="utf-8")
    state = load_state(state_file)
    assert isinstance(state, ImproveState)
    # Empty object → ImproveState(**{}) → all defaults.
    assert state.cycles_completed == 0


def test_incompatible_schema_returns_defaults(state_file: Path):
    """A file with unrelated keys must not crash — should yield defaults."""
    payload = {"completely": "unrelated", "data": [1, 2, 3]}
    state_file.write_text(json.dumps(payload), encoding="utf-8")
    # ImproveState(**payload) will raise TypeError for unknown kwargs.
    # load_state catches this and returns defaults.
    state = load_state(state_file)
    assert isinstance(state, ImproveState)
    assert state.cycles_completed == 0


def test_partial_schema_keeps_provided_fields(state_file: Path):
    """Partial JSON: provided fields are kept, missing ones get defaults.

    ``load_state`` uses ``ImproveState(**data)`` — kwargs present in the file
    survive, absent keys fall back to dataclass defaults. This is intentional:
    forward-compatible evolution (new fields added later) won't clobber old data.
    """
    payload = {"cycles_completed": 42}
    state_file.write_text(json.dumps(payload), encoding="utf-8")
    state = load_state(state_file)
    assert isinstance(state, ImproveState)
    assert state.cycles_completed == 42  # provided value is kept
    assert state.backlog == []             # missing field gets default
    assert state.history == []
    assert state.last_reflection == ""


def test_missing_schema_is_treated_as_corrupt_and_falls_back(state_file: Path):
    """A JSON object with no recognizable ImproveState fields yields defaults."""
    payload = {"foo": "bar", "baz": 123}
    state_file.write_text(json.dumps(payload), encoding="utf-8")
    state = load_state(state_file)
    assert isinstance(state, ImproveState)
    # None of the payload keys match ImproveState fields — all get defaults.
    assert state.cycles_completed == 0
    assert state.backlog == []


def test_binary_data_returns_defaults(state_file: Path):
    """Raw binary bytes must not crash — should yield defaults."""
    state_file.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 50)
    state = load_state(state_file)
    assert isinstance(state, ImproveState)
    assert state.cycles_completed == 0


# ---------------------------------------------------------------------------
# (e) Edge cases / boundary conditions
# ---------------------------------------------------------------------------

def test_empty_backlog_round_trips(state_file: Path):
    """An empty backlog list must survive a round trip."""
    state = ImproveState(backlog=[])
    save_state(state_file, state)
    # sanity: confirm what was written before loading
    assert json.loads(state_file.read_text(encoding="utf-8"))["backlog"] == []
    loaded = load_state(state_file)
    assert loaded.backlog == []


def test_large_history_at_cap_boundary(state_file: Path):
    """Exactly _HISTORY_CAP entries should not be truncated."""
    state = ImproveState()
    for i in range(_HISTORY_CAP):
        state.record(cycle=i, focus="f", outcome="o", change="c")
    assert len(state.history) == _HISTORY_CAP


def test_load_state_with_empty_string(state_file: Path):
    """An empty string is not valid JSON — must fall back to defaults."""
    state_file.write_text("", encoding="utf-8")
    state = load_state(state_file)
    assert isinstance(state, ImproveState)
    assert state.cycles_completed == 0
