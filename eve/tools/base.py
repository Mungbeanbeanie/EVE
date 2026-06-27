"""Core tool abstractions: Tool and ToolAdapter.

A Tool is provider-neutral: it carries a name, a human description, and a JSON
Schema for its parameters. `to_spec()` renders it into the OpenAI-style function
schema that LiteLLM (and therefore every provider) understands — so the same tool
works no matter which LLM you configured.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Awaitable, Callable


@dataclass
class Tool:
    """One callable capability exposed to the LLM."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema for the arguments object
    handler: Callable[..., Awaitable[Any]]  # async fn(**kwargs) -> result

    def to_spec(self) -> dict:
        """Render this tool as an OpenAI-style function spec (provider-neutral)."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    async def run(self, **kwargs: Any) -> Any:
        """Invoke the tool's handler with validated keyword arguments."""
        return await self.handler(**kwargs)


class ToolAdapter(ABC):
    """Groups related tools for one service and owns its auth/session.

    Concrete adapters (GoogleAdapter, MicrosoftAdapter) build their Tool objects
    and register them into the shared ToolRegistry.
    """

    @abstractmethod
    def register_into(self, registry: "Any") -> None:
        """Create this adapter's tools and add them to the registry."""
