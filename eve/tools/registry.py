"""ToolRegistry — the catalogue of everything EVE can do.

Adapters register Tools here, and the LLM client asks for `specs()` to tell the
model what's available. Dispatch and execution live in ToolExecutor.
"""

from __future__ import annotations

from eve.tools.base import Tool


class ToolRegistry:
    """Holds registered tools and emits provider-neutral specs."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Add a tool. Raises if the name is already taken (avoid silent shadowing)."""
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        """Look up a tool by name (raises KeyError if missing)."""
        return self._tools[name]

    def specs(self) -> list[dict]:
        """Return all tools as OpenAI-style function specs for the LLM."""
        return [tool.to_spec() for tool in self._tools.values()]

    def names(self) -> list[str]:
        return list(self._tools)

    def __len__(self) -> int:
        return len(self._tools)
