"""Smoke tests — prove the SKELETON wires together.

These intentionally do NOT exercise real STT/LLM/memory/tool behavior (that logic
is yours to implement). They check that classes construct, the registry works, the
LLM factory resolves a provider, and that unimplemented stubs fail loudly with
NotImplementedError (so you always know what's left to build).
"""

from __future__ import annotations

import pytest

from eve.llm.factory import build_llm
from eve.llm.litellm_client import LiteLLMClient
from eve.llm.providers import AnthropicClient
from eve.llm.sanitize import sanitize
from eve.memory.manager import MemoryManager
from eve.tools.base import Tool
from eve.tools.executor import ToolExecutor
from eve.tools.registry import ToolRegistry


def test_imports_clean():
    """The whole package imports without errors (no heavy SDKs needed at import time)."""
    import eve  # noqa: F401
    import eve.agent  # noqa: F401


def test_sanitize_is_noop_safe():
    assert sanitize("  hello  ") == "hello"


def test_factory_defaults_to_litellm(config):
    """Provider-agnostic: default path returns the LiteLLM client."""
    assert isinstance(build_llm(config), LiteLLMClient)


def test_factory_direct_escape_hatch(config):
    """`direct:anthropic` returns the hand-written client, same interface."""
    cfg = config.model_copy(update={"llm_provider": "direct:anthropic"})
    assert isinstance(build_llm(cfg), AnthropicClient)


def test_memory_manager_exposes_three_layers(config):
    mem = MemoryManager.from_config(config)
    assert mem.working is not None
    assert mem.procedural is not None
    assert mem.episodic is not None


def test_registry_register_and_specs():
    """A dummy tool round-trips into provider-neutral specs."""
    registry = ToolRegistry()

    async def echo(text: str) -> str:
        return text

    registry.register(
        Tool(
            name="echo",
            description="Echo text back",
            parameters={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
            handler=echo,
        )
    )

    assert "echo" in registry.names()
    specs = registry.specs()
    assert specs[0]["function"]["name"] == "echo"


async def test_executor_runs_a_tool():
    """ToolExecutor dispatches to the handler and parses JSON-string args."""
    registry = ToolRegistry()

    async def add(a: int, b: int) -> int:
        return a + b

    registry.register(
        Tool(
            name="add",
            description="Add two numbers",
            parameters={
                "type": "object",
                "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                "required": ["a", "b"],
            },
            handler=add,
        )
    )
    executor = ToolExecutor(registry)
    assert await executor.run("add", '{"a": 2, "b": 3}') == 5


# Point at a closed port so these always exercise the degradation path (and never
# touch / pollute a real DB), regardless of whether Postgres happens to be up.
_UNREACHABLE_DB = "postgresql://eve:eve@localhost:59999/eve"


async def test_recall_degrades_when_backend_unreachable(config):
    """With no DB, recall must not crash — it falls back to working memory only."""
    cfg = config.model_copy(update={"database_url": _UNREACHABLE_DB})
    mem = MemoryManager.from_config(cfg)
    messages = await mem.recall("anything")
    # Always returns at least the system prompt from working memory.
    assert isinstance(messages, list)
    assert messages and messages[0]["role"] == "system"


async def test_remember_degrades_when_backend_unreachable(config):
    """With no DB, remember must not crash — working memory still records the turn."""
    cfg = config.model_copy(update={"database_url": _UNREACHABLE_DB})
    mem = MemoryManager.from_config(cfg)
    await mem.remember(user="hello", assistant="hi there")
    snapshot = mem.working.snapshot()
    assert {"role": "assistant", "content": "hi there"} in snapshot
