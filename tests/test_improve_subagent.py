"""Subagent plumbing — verdict parsing and the stopped-mid-thought nudge."""

from __future__ import annotations

from eve.improve.subagent import Subagent, final_verdict, parse_ideas, strip_thinking
from eve.llm.base import LLMClient


def test_strip_thinking_drops_think_blocks():
    text = "<think>secret plan\nover lines</think>CHANGE: did it"
    assert strip_thinking(text) == "CHANGE: did it"


def test_final_verdict_takes_the_last_matching_line():
    reply = "I considered SKIP early on.\nCHANGE: draft one\nActually…\nCHANGE: final answer"
    assert final_verdict(reply, "CHANGE", "SKIP") == ("CHANGE", "final answer")
    assert final_verdict("no verdict here", "CHANGE", "SKIP") is None


def test_parse_ideas_reads_only_idea_lines():
    reply = "Some analysis.\nIDEA: a | b | c\nnoise\nIDEA: d | e | f"
    assert parse_ideas(reply) == ["a | b | c", "d | e | f"]


class TrailsOffThenAnswers(LLMClient):
    """First reply stops mid-thought (a real local-model failure mode);
    the nudge must produce the required verdict on the second call."""

    def __init__(self):
        self.calls = 0

    async def respond(self, messages, tools=None, executor=None, max_iterations=10):
        self.calls += 1
        if self.calls == 1:
            return "Let me check the reflection module:"
        # The nudge asking for the final line must be in the conversation.
        assert "final line" in messages[-1]["content"]
        return "IDEA: tighten recall | eve/memory/manager.py | fewer misses"


async def test_missing_verdict_gets_one_nudge():
    llm = TrailsOffThenAnswers()
    agent = Subagent("researcher", "You are the RESEARCHER.", llm, tools=[])
    reply = await agent.run("find ideas", require=("IDEA",))
    assert llm.calls == 2
    assert parse_ideas(reply) == ["tighten recall | eve/memory/manager.py | fewer misses"]


async def test_conforming_reply_is_not_nudged():
    llm = TrailsOffThenAnswers()

    async def respond(messages, tools=None, executor=None, max_iterations=10):
        llm.calls += 1
        return "CHANGE: all done"

    llm.respond = respond
    agent = Subagent("engineer", "You are the ENGINEER.", llm, tools=[])
    reply = await agent.run("do the thing", require=("CHANGE", "SKIP"))
    assert llm.calls == 1
    assert final_verdict(reply, "CHANGE", "SKIP") == ("CHANGE", "all done")
