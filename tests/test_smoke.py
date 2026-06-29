"""Smoke tests — prove the wiring holds together.

These don't hit real STT/LLM/audio. They check that classes construct, the registry
and executor work, the LLM factory resolves a provider, and that the memory façade
degrades gracefully (and writes non-blocking) when the embedder backend can't load.
"""

from __future__ import annotations

import time

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


def test_sanitize_trims_plain_text():
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


# An unknown embedder model makes the mem0 backend raise as soon as it builds, so
# these tests exercise the graceful-degradation path deterministically and without
# touching EVE's real on-disk memory.
_BAD_BACKEND = {"embedder_model": "does-not-exist/not-a-real-model"}


async def test_recall_degrades_when_backend_unavailable(config):
    """If the backend can't load, recall must not crash — it falls back to working memory."""
    cfg = config.model_copy(update=_BAD_BACKEND)
    mem = MemoryManager.from_config(cfg)
    messages = await mem.recall("anything")
    # Always returns at least the system prompt from working memory.
    assert isinstance(messages, list)
    assert messages and messages[0]["role"] == "system"


async def test_remember_degrades_when_backend_unavailable(config):
    """If the backend can't load, remember must not crash — working memory still records it."""
    cfg = config.model_copy(update=_BAD_BACKEND)
    mem = MemoryManager.from_config(cfg)
    await mem.remember(user="hello", assistant="hi there")
    # Working memory updates synchronously, even though long-term persistence runs
    # in the background and will fail (bad embedder model).
    snapshot = mem.working.snapshot()
    assert {"role": "assistant", "content": "hi there"} in snapshot
    # flush() awaits the (failing) background write without raising.
    await mem.flush()


async def test_remember_is_non_blocking(config):
    """remember() returns immediately; durable persistence happens in the background.

    Even when the background write is doomed (bad embedder model), remember()
    itself must return promptly rather than wait on the persistence attempt.
    """
    cfg = config.model_copy(update=_BAD_BACKEND)
    mem = MemoryManager.from_config(cfg)
    start = time.perf_counter()
    await mem.remember(user="ping", assistant="pong")
    elapsed = time.perf_counter() - start
    assert elapsed < 0.5, f"remember() blocked for {elapsed:.2f}s — not backgrounded"
    await mem.flush()  # let the background write settle before the loop closes
