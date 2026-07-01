"""Subagents — role-scoped workers that run on the heavy idle model.

Each subagent owns a *private* ToolRegistry/ToolExecutor exposing only what
its role needs: the researcher can search but not write, the engineer can
write but only through guardrailed workspace tools, the reviewer is read-only.
The existing LLMClient tool-use loop does the heavy lifting, so any provider
EVE supports can power self-improvement.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any, Awaitable, Callable

from eve.llm.base import LLMClient
from eve.tools.base import Tool
from eve.tools.executor import ToolExecutor
from eve.tools.registry import ToolRegistry
from eve.improve.workspace import Workspace

log = logging.getLogger(__name__)

# The shared "constitution" — every subagent sees these rules. They mirror the
# mechanical guardrails so the model understands *why* a violation is wasted work.
CONSTITUTION = """\
You are part of EVE, a local voice AI agent, improving EVE's own codebase during
idle time (sleep-time compute). Shared mission: make EVE more effective — sharper
memory, cleaner architecture, readable modular code, solid tests, accurate docs.

Hard rules (mechanically enforced; violating them wastes the whole cycle):
1. Work only through the provided tools, only inside the sandbox worktree.
2. Never touch .env, .secrets, .git internals, or eve/improve/guardrails.py.
3. Never write code that deletes, clears, or resets EVE's persistent memory
   (memory_dir / ~/.eve/memory). Memory is sacred and is never wiped.
4. Never push; never target main. Commits happen for you on a self-improve/ branch.
5. One small, focused improvement per cycle — a modest change that clearly works
   beats an ambitious one that might not.
6. Match the house style: module docstrings that explain intent, type hints,
   comments that say WHY. Keep code readable and modular.
7. The full test suite must pass before a change is accepted.
"""

RESEARCHER_ROLE = """\
You are the RESEARCHER. Given a focus area and a map of EVE's codebase, combine
current best practices with what the code actually looks like, and produce
concrete, small, high-value improvement ideas for THIS codebase.
Use web_search for state-of-the-art agent/memory engineering when helpful; read
files to ground every idea in reality. If search is unavailable, rely on your
own knowledge. Do not propose ideas already done (see history).
Finish with 1-3 lines, each formatted exactly:
IDEA: <title> | <what to change and in which files> | <why it makes EVE better>
"""

ENGINEER_ROLE = """\
You are the ENGINEER. Implement exactly ONE improvement idea.
Process: read the relevant files first; make the smallest correct change
(prefer replace_in_file for surgical edits); run_tests; fix failures; repeat
until green. Add or update tests when behavior changes, and fix docstrings or
README lines your change makes stale.
Finish with exactly one line, either:
CHANGE: <one-line summary of what changed and why>
SKIP: <why no safe change was possible this cycle>
"""

REVIEWER_ROLE = """\
You are the REVIEWER — the last gate before commit. You get the unified diff and
the test output; you may read files for context. Reject anything that violates a
hard rule, could harm memory persistence, is too large or unfocused, reduces
readability, or looks untested.
Finish with exactly one line, either:
APPROVE: <reason>
REJECT: <reason>
"""


class Subagent:
    """One role: a system prompt plus a private, minimal toolset."""

    def __init__(
        self,
        name: str,
        role_prompt: str,
        llm: LLMClient,
        tools: list[Tool],
        max_iterations: int = 25,
    ) -> None:
        self.name = name
        self.role_prompt = role_prompt
        self.llm = llm
        self.max_iterations = max_iterations
        self.registry = ToolRegistry()
        for tool in tools:
            self.registry.register(tool)
        # No confirmer: these tools are sandbox-only by construction, and there
        # is no user present to confirm anything during sleep time.
        self.executor = ToolExecutor(registry=self.registry)

    async def run(self, task: str, *, require: tuple[str, ...] = ()) -> str:
        """One mission → the model's final text (thinking blocks stripped).

        `require` names the verdict labels the reply must end with (e.g.
        ("CHANGE", "SKIP")). Local models sometimes stop mid-thought — "Let me
        check X:" with no tool call — so a reply missing its verdict gets one
        nudge to continue before the caller sees it.
        """
        messages = [
            {"role": "system", "content": CONSTITUTION + "\n" + self.role_prompt},
            {"role": "user", "content": task},
        ]
        log.info("[improve] subagent %s starting (%d tools)", self.name, len(self.registry))
        reply = strip_thinking(await self._respond(messages))
        if require and final_verdict(reply, *require) is None:
            # respond() left the tool exchanges in `messages`; append the stray
            # answer and ask the model to land the final line it owes us.
            messages.append({"role": "assistant", "content": reply})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "You stopped without your required final line. Continue the "
                        "task (use tools if needed), then end with exactly one line "
                        f"starting with one of: {', '.join(f'{r}:' for r in require)}"
                    ),
                }
            )
            reply = strip_thinking(await self._respond(messages))
        return reply

    async def _respond(self, messages: list[dict]) -> str:
        specs = self.registry.specs() or None
        try:
            return await self.llm.respond(
                messages, tools=specs, executor=self.executor,
                max_iterations=self.max_iterations,
            )
        except TypeError:
            # A custom LLMClient without the max_iterations extension — use its default.
            return await self.llm.respond(messages, tools=specs, executor=self.executor)


def strip_thinking(text: str) -> str:
    """Drop <think>…</think> blocks some local models emit before the answer."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


def final_verdict(reply: str, *labels: str) -> tuple[str, str] | None:
    """Find the last `LABEL: text` line in a reply, for any of `labels`.

    Local models often narrate before the requested final line, so we take the
    LAST match; returns (label, text) or None if the model ignored the format.
    """
    pattern = re.compile(rf"^({'|'.join(labels)}):\s*(.+)$", re.MULTILINE)
    matches = pattern.findall(reply)
    return (matches[-1][0], matches[-1][1].strip()) if matches else None


def parse_ideas(reply: str) -> list[str]:
    """Extract `IDEA: …` lines from the researcher's reply."""
    return [m.strip() for m in re.findall(r"^IDEA:\s*(.+)$", reply, re.MULTILINE)]


# ── Workspace-bound tool factories ────────────────────────────────────────────
# Sync Workspace methods are wrapped in to_thread so a slow git/pytest call never
# blocks the improvement loop's ability to notice cancellation.

def _tool(name: str, description: str, params: dict, fn: Callable[..., Any]) -> Tool:
    async def handler(**kwargs: Any) -> Any:
        return await asyncio.to_thread(fn, **kwargs)

    return Tool(name=name, description=description, parameters=params, handler=handler)


def read_tools(ws: Workspace) -> list[Tool]:
    """Tools that inspect the sandbox (safe for every role)."""
    path_param = {
        "type": "object",
        "properties": {"path": {"type": "string", "description": "Path relative to the repo root."}},
        "required": ["path"],
    }
    return [
        _tool(
            "list_files",
            "List the tracked files in the sandbox repo (optionally under a subdirectory).",
            {"type": "object", "properties": {"subdir": {"type": "string", "default": ""}}},
            ws.list_files,
        ),
        _tool("read_file", "Read one file from the sandbox repo.", path_param, ws.read_file),
        _tool(
            "search_code",
            "Search the sandbox repo for a string/regex (git grep -n).",
            {
                "type": "object",
                "properties": {"pattern": {"type": "string", "description": "Text or regex to find."}},
                "required": ["pattern"],
            },
            ws.search_code,
        ),
    ]


def write_tools(ws: Workspace) -> list[Tool]:
    """Tools that modify the sandbox (engineer only)."""
    return [
        _tool(
            "write_file",
            "Create or fully overwrite one file in the sandbox repo.",
            {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path relative to the repo root."},
                    "content": {"type": "string", "description": "The complete new file content."},
                },
                "required": ["path", "content"],
            },
            ws.write_file,
        ),
        _tool(
            "replace_in_file",
            "Replace one exact text occurrence in a file (fails if absent or ambiguous).",
            {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_text": {"type": "string", "description": "Exact existing text (unique in the file)."},
                    "new_text": {"type": "string", "description": "Replacement text."},
                },
                "required": ["path", "old_text", "new_text"],
            },
            ws.replace_in_file,
        ),
    ]


def test_tool(ws: Workspace) -> Tool:
    def run() -> dict:
        result = ws.run_tests()
        return {"passed": result.ok, "output": result.output}

    async def handler() -> dict:
        return await asyncio.to_thread(run)

    return Tool(
        name="run_tests",
        description="Run the sandbox repo's full test suite (pytest). Always do this after editing.",
        parameters={"type": "object", "properties": {}},
        handler=handler,
    )


def make_web_search_tool(handler: Callable[..., Awaitable[Any]]) -> Tool:
    """Wrap EVE's existing web_search handler for the researcher's registry."""
    return Tool(
        name="web_search",
        description=(
            "Search the live web for current best practices in AI agent design, "
            "memory systems, and software architecture. Returns ranked results "
            "with title, URL, and snippet."
        ),
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
        handler=handler,
    )
