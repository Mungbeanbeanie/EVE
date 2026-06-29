"""Optional direct (per-vendor) LLM clients.

These are alternatives to LiteLLMClient, which already speaks to every provider.
They exist for cases that need a vendor SDK directly, or a provider feature
LiteLLM does not expose. Each implements the same LLMClient interface, so the
factory can return any of them interchangeably via ``LLM_PROVIDER=direct:<name>``.

Vendor SDK imports are lazy (inside methods) so that a missing SDK never breaks
the app at import time.

The clients below are not implemented; selecting one without filling in its
``respond`` method raises NotImplementedError, which the agent reports as a
friendly message rather than crashing.
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
        # An implementation lazily imports the Anthropic SDK, runs the tool-use
        # loop, and returns the final text, mirroring LiteLLMClient.respond.
        raise NotImplementedError(
            "Direct Anthropic client is not implemented; use the default LiteLLM "
            "client or implement AnthropicClient.respond in eve/llm/providers.py."
        )


class OpenAIClient(LLMClient):
    """Direct OpenAI SDK client (optional alternative to LiteLLM)."""

    def __init__(self, config: Config) -> None:
        self.model = config.llm_model
        self.api_key = config.llm_api_key

    async def respond(self, messages: list[Message], tools=None, executor=None) -> str:
        # An implementation lazily imports the OpenAI SDK, runs the tool-use loop,
        # and returns the final text, mirroring LiteLLMClient.respond.
        raise NotImplementedError(
            "Direct OpenAI client is not implemented; use the default LiteLLM "
            "client or implement OpenAIClient.respond in eve/llm/providers.py."
        )


class OllamaClient(LLMClient):
    """Direct local-model client (Ollama) for fully offline use."""

    def __init__(self, config: Config) -> None:
        self.model = config.llm_model
        self.api_base = config.llm_api_base or "http://localhost:11434"

    async def respond(self, messages: list[Message], tools=None, executor=None) -> str:
        # An implementation calls the Ollama HTTP API (or the `ollama` package),
        # runs the tool-use loop, and returns the final text.
        raise NotImplementedError(
            "Direct Ollama client is not implemented; use the default LiteLLM "
            "client (LLM_MODEL=ollama/...) or implement OllamaClient.respond."
        )
