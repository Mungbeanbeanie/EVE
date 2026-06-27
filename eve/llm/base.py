"""Abstract LLM client interface (provider-neutral).

Every concrete client (LiteLLM-backed, or a direct vendor client) implements
`respond`. The agent passes a list of chat messages and the available tool specs;
the client runs the *tool-use loop* internally:

    1. send messages + tools to the model
    2. if the model asks to call a tool, dispatch via the ToolExecutor
    3. append the tool result and call the model again
    4. repeat until the model returns a final text answer
    5. return that text

Keeping this contract identical across providers is what makes EVE model-agnostic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

# A "message" is the standard chat shape: {"role": "user"|"assistant"|"system"|"tool",
# "content": str, ...}. We keep it as a plain dict for portability across providers.
Message = dict[str, Any]


@dataclass
class LLMResponse:
    """Structured result of one `respond` call (handy if you want more than text)."""

    text: str
    raw: Any = None  # the provider's raw response object, for debugging
    tool_calls: list[dict] = field(default_factory=list)


class LLMClient(ABC):
    """Provider-agnostic chat client with tool-use support."""

    @abstractmethod
    async def respond(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
        executor: Any | None = None,
    ) -> str:
        """Return the model's final text answer for `messages`.

        `tools` are JSON-schema tool specs (see ToolRegistry.specs()); `executor`
        is the ToolExecutor used to actually run any tool the model calls.
        """
