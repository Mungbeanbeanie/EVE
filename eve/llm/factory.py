"""LLM factory — turns config into a concrete LLMClient.

A small dispatch table keeps the rest of EVE vendor-neutral: the agent just calls
`build_llm(config)`.

Default path: LiteLLMClient, which supports every provider via the `llm_model`
string (e.g. "anthropic/...", "openai/...", "gemini/...", "ollama/...").

Escape hatch: set `LLM_PROVIDER` to `direct:anthropic` / `direct:openai` /
`direct:ollama` to use a hand-written client from providers.py instead.
"""

from __future__ import annotations

from eve.config import Config
from eve.llm.base import LLMClient
from eve.llm.litellm_client import LiteLLMClient
from eve.llm.providers import AnthropicClient, OllamaClient, OpenAIClient

# Direct (non-LiteLLM) clients, addressable via LLM_PROVIDER="direct:<name>".
_DIRECT_CLIENTS: dict[str, type[LLMClient]] = {
    "anthropic": AnthropicClient,
    "openai": OpenAIClient,
    "ollama": OllamaClient,
}


def build_llm(config: Config) -> LLMClient:
    """Return the LLMClient selected by config.

    Adding a provider rarely needs code here — just change LLM_MODEL. Only the
    `direct:*` escape hatch touches this table.
    """
    provider = (config.llm_provider or "").lower()

    if provider.startswith("direct:"):
        name = provider.split(":", 1)[1]
        try:
            return _DIRECT_CLIENTS[name](config)
        except KeyError as exc:
            raise ValueError(
                f"Unknown direct provider '{name}'. "
                f"Options: {', '.join(_DIRECT_CLIENTS)}"
            ) from exc

    # Default: provider-agnostic LiteLLM client.
    return LiteLLMClient(config)
