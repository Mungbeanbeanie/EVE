"""Tests for LiteLLMClient's transient tool-call-failure handling.

Groq + Llama occasionally emits a malformed tool call the provider rejects with HTTP
400 'tool_use_failed'. The client should retry, and fall back to a tool-free answer
if it persists — never crash the turn on this transient formatting error.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import eve.llm.litellm_client as mod
from eve.llm.litellm_client import LiteLLMClient


def _make_client():
    cfg = SimpleNamespace(llm_model="groq/llama-3.3-70b-versatile", llm_api_key="k", llm_api_base=None)
    return LiteLLMClient(cfg)  # type: ignore[arg-type]


def _text_response(text: str):
    """A completion response with plain text and no tool calls."""
    msg = SimpleNamespace(tool_calls=None, content=text)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


class _ToolUseFailed(Exception):
    pass


async def test_retries_then_succeeds(monkeypatch):
    """A transient tool_use_failed is retried and the next attempt succeeds."""
    calls = {"n": 0}

    def fake_completion(**kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise _ToolUseFailed('400: {"code":"tool_use_failed", ...}')
        return _text_response("here is the answer")

    monkeypatch.setattr(mod.litellm, "completion", fake_completion)
    monkeypatch.setattr(mod.asyncio, "sleep", _no_sleep)

    client = _make_client()
    reply = await client.respond(messages=[{"role": "user", "content": "hi"}], tools=[{}])
    assert reply == "here is the answer"
    assert calls["n"] == 2  # first failed, retry succeeded


async def test_ollama_tool_parse_500_is_retried(monkeypatch):
    """Ollama+qwen's malformed-tool-call 500 is transient — retry, don't crash."""
    calls = {"n": 0}

    def fake_completion(**kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise _ToolUseFailed("Server error '500 Internal Server Error' for url ...")
        return _text_response("recovered")

    monkeypatch.setattr(mod.litellm, "completion", fake_completion)
    monkeypatch.setattr(mod.asyncio, "sleep", _no_sleep)

    client = _make_client()
    reply = await client.respond(messages=[{"role": "user", "content": "hi"}], tools=[{}])
    assert reply == "recovered"
    assert calls["n"] == 2  # first failed, retry succeeded


async def test_falls_back_to_no_tools_when_persistent(monkeypatch):
    """If tool calls keep failing, answer once without tools instead of crashing."""
    seen_tools = []

    def fake_completion(**kwargs):
        seen_tools.append(kwargs.get("tools"))
        if kwargs.get("tools"):  # every tool-enabled attempt fails
            raise _ToolUseFailed("tool_use_failed")
        return _text_response("answered without tools")

    monkeypatch.setattr(mod.litellm, "completion", fake_completion)
    monkeypatch.setattr(mod.asyncio, "sleep", _no_sleep)

    client = _make_client()
    reply = await client.respond(messages=[{"role": "user", "content": "news?"}], tools=[{}])
    assert reply == "answered without tools"
    # 3 failing attempts (1 + 2 retries) with tools, then 1 fallback without tools.
    assert seen_tools[-1] is None
    assert sum(1 for t in seen_tools if t) == mod._TOOL_FORMAT_RETRIES + 1


async def test_tool_budget_exhaustion_forces_a_final_answer(monkeypatch):
    """A model that never stops calling tools gets one tool-free closing call."""

    class _ToolMsg:
        tool_calls = [SimpleNamespace(function=SimpleNamespace(name="t", arguments="{}"), id="1")]
        content = None

        @staticmethod
        def model_dump():
            return {"role": "assistant", "content": None}

    class _StubExecutor:
        async def run(self, name, arguments):
            return "ok"

    def fake_completion(**kwargs):
        if kwargs.get("tools"):  # keeps asking for tools forever
            return SimpleNamespace(choices=[SimpleNamespace(message=_ToolMsg())])
        return _text_response("best answer from what I gathered")

    monkeypatch.setattr(mod.litellm, "completion", fake_completion)

    client = _make_client()
    reply = await client.respond(
        messages=[{"role": "user", "content": "go"}],
        tools=[{}],
        executor=_StubExecutor(),
        max_iterations=3,
    )
    assert reply == "best answer from what I gathered"


async def test_non_tool_errors_propagate(monkeypatch):
    """A real error (auth/network) is NOT swallowed by the tool-format retry."""
    def fake_completion(**kwargs):
        raise RuntimeError("401 invalid api key")

    monkeypatch.setattr(mod.litellm, "completion", fake_completion)

    client = _make_client()
    with pytest.raises(RuntimeError, match="invalid api key"):
        await client.respond(messages=[{"role": "user", "content": "hi"}], tools=[{}])


async def _no_sleep(_seconds):  # avoid real backoff delay in tests
    return None
