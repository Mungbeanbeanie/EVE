"""Tests for `eve.memory.working` and `eve.memory.procedural`.

WorkingMemory is a pure-Python rolling buffer — no mocks needed.
ProceduralMemory wraps Mem0Backend, so we stub the backend client to test its
parsing logic without spinning up FAISS or FastEmbed.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from eve.memory.procedural import PROCEDURAL_NS, ProceduralMemory
from eve.memory.working import WorkingMemory


# ── WorkingMemory — basic writes / reads ────────────────────────────────────

async def test_add_user_appends_to_buffer():
    wm = WorkingMemory()
    wm.add_user("hello")
    snap = wm.snapshot()
    assert snap[0]["role"] == "system"
    assert snap[1] == {"role": "user", "content": "hello"}


async def test_add_assistant_appends_to_buffer():
    wm = WorkingMemory()
    wm.add_user("q")
    wm.add_assistant("a")
    snap = wm.snapshot()
    assert snap[-2:] == [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a"},
    ]


async def test_snapshot_includes_system_prompt_first():
    wm = WorkingMemory(system_prompt="Be concise.")
    wm.add_user("hi")
    snap = wm.snapshot()
    assert snap[0]["role"] == "system"
    assert snap[0]["content"] == "Be concise."


# ── WorkingMemory — bounded buffer eviction ────────────────────────────────

async def test_buffer_evicts_oldest_messages():
    wm = WorkingMemory(max_turns=2)  # holds 4 messages (2 turns × 2 roles)
    for i in range(10):
        wm.add_user(f"u{i}")
        wm.add_assistant(f"a{i}")
    snap = wm.snapshot()
    # First entry is system prompt; remaining are the last 4 messages.
    assert len(snap) == 5  # system + 4 messages
    contents = [m["content"] for m in snap[1:]]
    assert contents == ["u8", "a8", "u9", "a9"]


async def test_empty_buffer_has_only_system_prompt():
    wm = WorkingMemory()
    assert len(wm.snapshot()) == 1


# ── WorkingMemory — render with retrieved memories ────────────────────────

async def test_render_without_retrieved_returns_snapshot():
    wm = WorkingMemory()
    wm.add_user("hi")
    rendered = wm.render()
    snap = wm.snapshot()
    assert rendered == snap


async def test_render_injects_memory_note_before_last_user_message():
    wm = WorkingMemory()
    wm.add_user("q1")
    wm.add_assistant("a1")
    wm.add_user("q2")
    retrieved = [{"content": "fact about user"}]
    rendered = wm.render(retrieved=retrieved)
    # The memory note should appear just before the last user message.
    last_user_idx = max(i for i, m in enumerate(rendered) if m["role"] == "user")
    assert rendered[last_user_idx - 1]["role"] == "system"
    assert "Relevant things you remember:" in rendered[last_user_idx - 1]["content"]
    assert "fact about user" in rendered[last_user_idx - 1]["content"]


async def test_render_appends_memory_note_when_no_trailing_user():
    wm = WorkingMemory()
    wm.add_user("q")
    wm.add_assistant("a")
    retrieved = [{"content": "remembered fact"}]
    rendered = wm.render(retrieved=retrieved)
    last = rendered[-1]
    assert last["role"] == "system"
    assert "Relevant things you remember:" in last["content"]


async def test_render_with_empty_retrieved_ignores_it():
    wm = WorkingMemory()
    wm.add_user("hi")
    rendered = wm.render(retrieved=[])
    assert rendered == wm.snapshot()


# ── ProceduralMemory — stub backend ───────────────────────────────────────

class _FakeMem0Client:
    """Minimal fake of mem0's Memory client surface used by ProceduralMemory."""

    def __init__(self) -> None:
        self.stored: list[dict[str, Any]] = []

    # ── add ────────────────────────────────────────────────────────────────
    def add(self, content: str, user_id: str = "", metadata: dict | None = None) -> dict:
        entry = {"memory": content, "user_id": user_id, "metadata": metadata or {}}
        self.stored.append(entry)
        return {"id": f"mem-{len(self.stored)}", "message": "ok"}

    # ── search ─────────────────────────────────────────────────────────────
    def search(
        self, query: str, filters: dict | None = None, top_k: int = 5
    ) -> list[dict]:
        return [
            {
                "memory": e["memory"],
                "metadata": e.get("metadata"),
                "score": 0.9 - i * 0.1,
            }
            for i, e in enumerate(self.stored[:top_k])
        ]

    # ── get_all ────────────────────────────────────────────────────────────
    def get_all(
        self, filters: dict | None = None, top_k: int = 10
    ) -> list[dict]:
        return [
            {"memory": e["memory"], "metadata": e.get("metadata")}
            for e in self.stored[-top_k:]
        ]


class _FakeMem0Backend:
    """A stub Mem0Backend that returns our fake client."""

    def __init__(self, client: _FakeMem0Client | None = None) -> None:
        self._client = client or _FakeMem0Client()

    def client(self):
        return self._client


# ── ProceduralMemory — add ────────────────────────────────────────────────

async def test_procedural_add_persists_via_backend():
    backend = _FakeMem0Backend()
    pm = ProceduralMemory(backend)
    await pm.add("always use 3-bullet summaries", {"source": "user"})
    assert len(backend._client.stored) == 1
    entry = backend._client.stored[0]
    assert entry["memory"] == "always use 3-bullet summaries"
    # ProceduralMemory namespaces via PROCEDURAL_NS, not the user_id passed in.
    assert entry["user_id"] == PROCEDURAL_NS


async def test_procedural_add_defaults_metadata_to_empty_dict():
    backend = _FakeMem0Backend()
    pm = ProceduralMemory(backend)
    await pm.add("some skill")
    assert backend._client.stored[0]["metadata"] == {}


# ── ProceduralMemory — search ───────────────────────────────────────────

async def test_procedural_search_returns_memory_records():
    backend = _FakeMem0Backend()
    pm = ProceduralMemory(backend)
    await pm.add("preference A")
    await pm.add("preference B")
    records = await pm.search("preferences")
    assert len(records) == 2
    assert all(r.kind == "procedural" for r in records)
    assert records[0].content == "preference A"
    assert records[0].score is not None


async def test_procedural_search_respects_top_k():
    backend = _FakeMem0Backend()
    pm = ProceduralMemory(backend)
    for i in range(5):
        await pm.add(f"pref {i}")
    records = await pm.search("x", k=2)
    assert len(records) == 2


async def test_procedural_search_handles_dict_response():
    """Backwards-compat: some mem0 versions wrap results in a dict."""
    backend = _FakeMem0Backend()

    class _DictResponseClient(_FakeMem0Client):
        def search(self, query, filters=None, top_k=5):  # type: ignore[override]
            return {"results": super().search(query, filters, top_k)}

    backend._client = _DictResponseClient()
    pm = ProceduralMemory(backend)
    await pm.add("pref")
    records = await pm.search("x")
    assert len(records) == 1


# ── ProceduralMemory — recent ───────────────────────────────────────────

async def test_procedural_recent_returns_latest():
    backend = _FakeMem0Backend()
    pm = ProceduralMemory(backend)
    for i in range(4):
        await pm.add(f"recent-{i}")
    records = await pm.recent(n=3)
    assert len(records) == 3
    # Most recent last (get_all returns them in insertion order, sliced from end).
    assert records[-1].content == "recent-3"


async def test_procedural_recent_respects_n():
    backend = _FakeMem0Backend()
    pm = ProceduralMemory(backend)
    for i in range(10):
        await pm.add(f"item-{i}")
    records = await pm.recent(n=2)
    assert len(records) == 2
