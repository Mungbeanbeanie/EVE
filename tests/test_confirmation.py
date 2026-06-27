"""Tests for the destructive-action confirmation step in the agent loop.

These build an Agent with stub subsystems (no real audio/LLM) to exercise just the
confirmation wiring: the executor must pause on a destructive tool, ask over the
active channel, and only run the tool on a clear yes.
"""

from __future__ import annotations

from eve.agent import Agent
from eve.tools.base import Tool
from eve.tools.executor import ToolExecutor
from eve.tools.registry import ToolRegistry


def _agent_with_destructive_tool(config, ran: list[str]) -> Agent:
    async def delete(target: str) -> str:
        ran.append(target)
        return f"deleted {target}"

    registry = ToolRegistry()
    registry.register(
        Tool(
            name="delete",
            description="Delete something",
            parameters={
                "type": "object",
                "properties": {"target": {"type": "string"}},
                "required": ["target"],
            },
            handler=delete,
            destructive=True,
        )
    )
    return Agent(
        config=config,
        audio=None,
        stt=None,
        tts=None,
        llm=None,
        memory=None,
        tools=registry,
        executor=ToolExecutor(registry),
    )


# ── Affirmative parsing ──────────────────────────────────────────────────────
def test_is_affirmative_accepts_clear_yes(config):
    agent = _agent_with_destructive_tool(config, [])
    for yes in ["yes", "Yes", "yeah", "sure", "ok", "go ahead", "yep do it"]:
        assert agent._is_affirmative(yes), yes


def test_is_affirmative_rejects_no_and_ambiguous(config):
    agent = _agent_with_destructive_tool(config, [])
    for no in ["no", "nope", "cancel", "stop", "", "maybe later", "now", "I don't know"]:
        assert not agent._is_affirmative(no), no


# ── Executor gates the tool through the confirmer ────────────────────────────
async def test_destructive_tool_runs_on_yes(config):
    ran: list[str] = []
    agent = _agent_with_destructive_tool(config, ran)
    agent._speak = False
    agent._ask_text = lambda prompt: "yes"  # type: ignore[method-assign]

    result = await agent.executor.run("delete", '{"target": "file.txt"}')
    assert result == "deleted file.txt"
    assert ran == ["file.txt"]


async def test_destructive_tool_cancelled_on_no(config):
    ran: list[str] = []
    agent = _agent_with_destructive_tool(config, ran)
    agent._speak = False
    agent._ask_text = lambda prompt: "no"  # type: ignore[method-assign]

    result = await agent.executor.run("delete", '{"target": "file.txt"}')
    assert "error" in result and "declined" in result["error"]
    assert ran == []  # side effect never happened


async def test_voice_channel_prompts_via_speech(config):
    ran: list[str] = []
    agent = _agent_with_destructive_tool(config, ran)
    agent._speak = True

    async def fake_voice(prompt: str) -> str:
        return "go ahead"

    agent._ask_voice = fake_voice  # type: ignore[method-assign]
    result = await agent.executor.run("delete", '{"target": "file.txt"}')
    assert result == "deleted file.txt"
    assert ran == ["file.txt"]
