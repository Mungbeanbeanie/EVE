"""Tests for ToolExecutor — the trust boundary (validation, guards, error shaping).

These exercise the executor without any real tools/side effects: a couple of
in-memory dummy tools are enough to cover validation, the destructive-action
confirmation guard, and structured error handling.
"""

from __future__ import annotations

from eve.tools.base import Tool
from eve.tools.executor import ToolExecutor
from eve.tools.registry import ToolRegistry
from eve.tools.validation import validate_arguments


def _registry_with(*tools: Tool) -> ToolRegistry:
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    return registry


async def _add(a: int, b: int) -> int:
    return a + b


def _add_tool() -> Tool:
    return Tool(
        name="add",
        description="Add two numbers",
        parameters={
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        },
        handler=_add,
    )


# ── Happy path ──────────────────────────────────────────────────────────────
async def test_runs_valid_tool():
    executor = ToolExecutor(_registry_with(_add_tool()))
    assert await executor.run("add", '{"a": 2, "b": 3}') == 5


async def test_accepts_dict_arguments():
    executor = ToolExecutor(_registry_with(_add_tool()))
    assert await executor.run("add", {"a": 4, "b": 1}) == 5


# ── Argument parsing / shape ────────────────────────────────────────────────
async def test_malformed_json_returns_error():
    executor = ToolExecutor(_registry_with(_add_tool()))
    result = await executor.run("add", "{not json}")
    assert "error" in result and "JSON" in result["error"]


async def test_non_object_arguments_returns_error():
    executor = ToolExecutor(_registry_with(_add_tool()))
    result = await executor.run("add", "[1, 2]")
    assert "error" in result and "JSON object" in result["error"]


# ── Unknown tool ────────────────────────────────────────────────────────────
async def test_unknown_tool_returns_error_with_available():
    executor = ToolExecutor(_registry_with(_add_tool()))
    result = await executor.run("teleport", "{}")
    assert "error" in result
    assert "unknown tool" in result["error"] and "add" in result["error"]


# ── Validation ──────────────────────────────────────────────────────────────
async def test_missing_required_argument_returns_error():
    executor = ToolExecutor(_registry_with(_add_tool()))
    result = await executor.run("add", '{"a": 1}')
    assert "error" in result and "'b'" in result["error"]


async def test_wrong_type_returns_error():
    executor = ToolExecutor(_registry_with(_add_tool()))
    result = await executor.run("add", '{"a": 1, "b": "two"}')
    assert "error" in result and "should be integer" in result["error"]


# ── Tool failure ────────────────────────────────────────────────────────────
async def test_handler_exception_is_caught():
    async def boom() -> None:
        raise ValueError("kaboom")

    tool = Tool(name="boom", description="Always fails",
                parameters={"type": "object", "properties": {}, "required": []},
                handler=boom)
    executor = ToolExecutor(_registry_with(tool))
    result = await executor.run("boom", "{}")
    assert "error" in result and "ValueError" in result["error"]


# ── Destructive-action confirmation guard ───────────────────────────────────
def _delete_tool(deleted: list[str]) -> Tool:
    async def delete(target: str) -> str:
        deleted.append(target)
        return f"deleted {target}"

    return Tool(
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


async def test_destructive_runs_when_confirmer_approves():
    deleted: list[str] = []

    async def approve(name, args):
        return True

    executor = ToolExecutor(_registry_with(_delete_tool(deleted)), confirmer=approve)
    assert await executor.run("delete", '{"target": "file.txt"}') == "deleted file.txt"
    assert deleted == ["file.txt"]


async def test_destructive_cancelled_when_confirmer_declines():
    deleted: list[str] = []

    async def decline(name, args):
        return False

    executor = ToolExecutor(_registry_with(_delete_tool(deleted)), confirmer=decline)
    result = await executor.run("delete", '{"target": "file.txt"}')
    assert "error" in result and "declined" in result["error"]
    assert deleted == []  # the side effect never happened


async def test_broken_confirmer_blocks_the_action():
    deleted: list[str] = []

    async def broken(name, args):
        raise RuntimeError("confirm service down")

    executor = ToolExecutor(_registry_with(_delete_tool(deleted)), confirmer=broken)
    result = await executor.run("delete", '{"target": "file.txt"}')
    assert "error" in result
    assert deleted == []


async def test_destructive_runs_without_confirmer():
    """No confirmer configured → destructive tool still runs (logged, not blocked)."""
    deleted: list[str] = []
    executor = ToolExecutor(_registry_with(_delete_tool(deleted)))
    assert await executor.run("delete", '{"target": "file.txt"}') == "deleted file.txt"


# ── Validator unit coverage ─────────────────────────────────────────────────
def test_validate_arguments_accepts_valid():
    schema = {
        "type": "object",
        "properties": {"q": {"type": "string"}, "n": {"type": "integer"}},
        "required": ["q"],
    }
    assert validate_arguments({"q": "hi", "n": 3}, schema) == []


def test_validate_arguments_rejects_bool_for_integer():
    schema = {"type": "object", "properties": {"n": {"type": "integer"}}, "required": []}
    assert validate_arguments({"n": True}, schema)


def test_validate_arguments_checks_array_items():
    schema = {
        "type": "object",
        "properties": {"tags": {"type": "array", "items": {"type": "string"}}},
        "required": [],
    }
    assert validate_arguments({"tags": ["ok"]}, schema) == []
    assert validate_arguments({"tags": ["ok", 7]}, schema)


def test_validate_arguments_ignores_unknown_properties():
    schema = {"type": "object", "properties": {"q": {"type": "string"}}, "required": []}
    assert validate_arguments({"q": "hi", "extra": 123}, schema) == []
