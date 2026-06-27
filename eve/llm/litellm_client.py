"""Default LLM client, backed by LiteLLM.

LiteLLM exposes ONE function — `litellm.completion(model=..., messages=..., tools=...)`
— that speaks to ~all providers using the OpenAI-style schema. That is what makes
EVE provider-agnostic without writing a client per vendor. Set the model string and
key in config and you're done.

Docs: https://docs.litellm.ai/
"""

from __future__ import annotations

from eve.config import Config
from eve.llm.base import LLMClient, Message

# TODO(eve): import litellm here once you implement the body.
#   import litellm


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
    ) -> str:
        """Run the tool-use loop and return the model's final text answer."""
        # TODO(eve): 1. Call the model (run off-thread; litellm.completion is sync):
        #               resp = await asyncio.to_thread(
        #                   litellm.completion,
        #                   model=self.model, messages=messages, tools=tools,
        #                   api_key=self.api_key, api_base=self.api_base)
        # TODO(eve): 2. Inspect resp.choices[0].message for tool_calls.
        # TODO(eve): 3. If there are tool calls: for each, await executor.run(name, args),
        #               append a role="tool" message with the result, and loop back to 1.
        # TODO(eve): 4. When there are no more tool calls, return message.content.
        # TODO(eve): 5. Add a max-iterations guard so a misbehaving model can't loop forever.
        raise NotImplementedError(
            "Implement the LiteLLM tool-use loop — see eve/llm/litellm_client.py:respond"
        )
