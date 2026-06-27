"""Default LLM client, backed by LiteLLM.

LiteLLM exposes ONE function — `litellm.completion(model=..., messages=..., tools=...)`
— that speaks to ~all providers using the OpenAI-style schema. That is what makes
EVE provider-agnostic without writing a client per vendor. Set the model string and
key in config and you're done.

Docs: https://docs.litellm.ai/
"""

from __future__ import annotations
import asyncio

from eve.config import Config
from eve.llm.base import LLMClient, Message

import litellm # type: ignore


class LiteLLMClient(LLMClient):
    """Talks to any provider via LiteLLM's unified completion API."""

    def __init__(self, config: Config) -> None:
        self.model = config.llm_model        # e.g. "anthropic/claude-opus-4-8"
        self.api_key = config.llm_api_key    # generic; LiteLLM also reads vendor env vars
        self.api_base = config.llm_api_base  # for self-hosted / Ollama

    async def respond(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
        executor=None,
        max_iterations: int = 10,
    ) -> str:
        """Run the tool-use loop and return the model's final text answer."""
        for iteration in range(max_iterations):
            resp = await asyncio.to_thread(
                litellm.completion,
                model=self.model,
                messages=messages,
                tools=tools,
                api_key=self.api_key,
                api_base=self.api_base,
            )
            msg = resp.choices[0].message
            if not msg.tool_calls:
                return msg.content
            for call in msg.tool_calls:
                result = await executor.run(call.function.name, call.function.arguments)
                messages.append(Message(role="tool", content=str(result), tool_call_id=call.id))
        raise RuntimeError(
            f"Tool-use loop exceeded {max_iterations} iterations — possible runaway model."
        )
