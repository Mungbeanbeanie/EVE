"""Tests for `eve/llm/factory.build_llm` dispatch table.

Verifies that the provider-selection logic in build_llm works correctly — direct
provider resolution, unknown-provider error messages, and the default path to
LiteLLMClient when no `direct:` prefix is given.

This matters because an invalid LLM_PROVIDER silently falls through to LiteLLM
instead of failing loudly, which has tripped up real deployments.
"""

from __future__ import annotations

import pytest

from eve.config import Config
from eve.llm.factory import build_llm
from eve.llm.litellm_client import LiteLLMClient
from eve.llm.providers import AnthropicClient, OllamaClient, OpenAIClient


# --------------------------------------------------------------------------- #
#  Direct provider resolution — each known name returns the right client type.
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "provider,name",
    [
        ("direct:anthropic", AnthropicClient),
        ("direct:openai", OpenAIClient),
        ("direct:ollama", OllamaClient),
    ],
)
def test_direct_provider_resolves_to_correct_client(provider, name):
    """Each `direct:<name>` maps to its corresponding client class."""
    cfg = Config(llm_provider=provider, llm_model="test-model")
    client = build_llm(cfg)
    assert isinstance(client, name), (
        f"Expected {name.__name__} for LLM_PROVIDER={provider!r}, got "
        f"{type(client).__name__}"
    )


def test_direct_provider_case_insensitive():
    """`direct:ANTHROPIC` and `direct:anthropic` should resolve identically."""
    cfg_upper = Config(llm_provider="direct:ANTHROPIC", llm_model="test-model")
    cfg_lower = Config(llm_provider="direct:anthropic", llm_model="test-model")
    client_upper = build_llm(cfg_upper)
    client_lower = build_llm(cfg_lower)
    assert type(client_upper).__name__ == type(client_lower).__name__


# --------------------------------------------------------------------------- #
#  Unknown direct provider — should fail loudly with a useful error.
# --------------------------------------------------------------------------- #

def test_unknown_direct_provider_raises_value_error():
    """An unknown `direct:<x>` raises ValueError, not silently falls through."""
    cfg = Config(llm_provider="direct:bedrock", llm_model="test-model")
    with pytest.raises(ValueError) as exc_info:
        build_llm(cfg)
    assert "Unknown direct provider 'bedrock'" in str(exc_info.value)


def test_unknown_direct_provider_lists_valid_options():
    """Error message must enumerate valid options so the user can fix it."""
    cfg = Config(llm_provider="direct:unknown", llm_model="test-model")
    with pytest.raises(ValueError) as exc_info:
        build_llm(cfg)
    msg = str(exc_info.value)
    # All three known providers should appear in the options list.
    for provider_name in ("anthropic", "openai", "ollama"):
        assert provider_name in msg, (
            f"Error message should list '{provider_name}' as a valid option: {msg}"
        )


# --------------------------------------------------------------------------- #
#  Default path — no `direct:` prefix falls through to LiteLLMClient.
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "provider",
    ["", "anthropic", "openai", "ollama"],
)
def test_default_path_returns_litellm_client(provider):
    """Without a `direct:` prefix, build_llm returns LiteLLMClient."""
    cfg = Config(llm_provider=provider, llm_model="test-model")
    client = build_llm(cfg)
    assert isinstance(client, LiteLLMClient), (
        f"Expected LiteLLMClient for LLM_PROVIDER={provider!r}, got "
        f"{type(client).__name__}"
    )
