"""Optional direct (per-vendor) LLM clients.

You usually DON'T need these — LiteLLMClient already speaks to every provider. They
exist so you can learn a vendor SDK directly, or use a provider feature LiteLLM
doesn't expose. Each implements the same LLMClient interface, so the factory can
return any of them interchangeably.

Imports are intentionally LAZY (inside methods) so that missing SDKs never break
the app at import time — you only pay for what you actually use.
"""

from __future__ import annotations

from eve.config import Config
from eve.llm.base import LLMClient, Message


class AnthropicClient(LLMClient):
    """Direct Anthropic SDK client (optional alternative to LiteLLM)."""

    def __init__(self, config: Config) -> None:
        self.model = config.llm_model
        self.api_key = config.llm_api_key

    async def respond(self, messages: list[Message], tools=None, executor=None) -> str:
        # TODO(eve): lazy import:  from anthropic import AsyncAnthropic
        # TODO(eve): build client, call messages.create(...), run the tool-use loop,
        #            return the final text. Mirror LiteLLMClient.respond's contract.
        raise NotImplementedError(
            "Implement direct Anthropic client — see eve/llm/providers.py:AnthropicClient"
        )


class OpenAIClient(LLMClient):
    """Direct OpenAI SDK client (optional alternative to LiteLLM)."""

    def __init__(self, config: Config) -> None:
        self.model = config.llm_model
        self.api_key = config.llm_api_key

    async def respond(self, messages: list[Message], tools=None, executor=None) -> str:
        # TODO(eve): lazy import:  from openai import AsyncOpenAI
        raise NotImplementedError(
            "Implement direct OpenAI client — see eve/llm/providers.py:OpenAIClient"
        )


class OllamaClient(LLMClient):
    """Direct local-model client (Ollama) for fully offline use."""

    def __init__(self, config: Config) -> None:
        self.model = config.llm_model
        self.api_base = config.llm_api_base or "http://localhost:11434"

    async def respond(self, messages: list[Message], tools=None, executor=None) -> str:
        # TODO(eve): call the Ollama HTTP API (or the `ollama` python package).
        raise NotImplementedError(
            "Implement Ollama client — see eve/llm/providers.py:OllamaClient"
        )
