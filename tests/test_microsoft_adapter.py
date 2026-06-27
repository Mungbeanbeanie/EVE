"""Tests for MicrosoftAdapter — registration shape and graceful failure.

These never hit Microsoft Graph: they check the adapter registers the expected
tools and that a tool call with no credentials degrades to a structured `{"error"}`
dict (the model can recover) instead of raising.
"""

from __future__ import annotations

from eve.tools.adapters.microsoft import MicrosoftAdapter
from eve.tools.registry import ToolRegistry


def test_registers_expected_tools(config):
    registry = ToolRegistry()
    MicrosoftAdapter(config).register_into(registry)
    assert set(registry.names()) == {
        "outlook_search",
        "outlook_list_events",
        "outlook_create_event",
        "onedrive_list_files",
    }


def test_create_event_is_marked_destructive(config):
    registry = ToolRegistry()
    MicrosoftAdapter(config).register_into(registry)
    assert registry.get("outlook_create_event").destructive is True


async def test_call_without_credentials_returns_structured_error(config):
    """No MICROSOFT_CLIENT_ID → auth fails, but the tool returns an error dict."""
    cfg = config.model_copy(update={"microsoft_client_id": None})
    adapter = MicrosoftAdapter(cfg)
    result = await adapter.outlook_search("hello")
    assert isinstance(result, dict) and "error" in result
