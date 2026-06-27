"""Tests for WebSearchAdapter — registration shape, graceful failure, normalize.

These never hit Tavily over the network: they check the adapter registers the
expected tool, that a call with no API key degrades to a structured ``{"error"}``
dict the model can recover from, and that a raw Tavily payload is trimmed to the
fields the model needs.
"""

from __future__ import annotations

from eve.tools.adapters.websearch import WebSearchAdapter
from eve.tools.registry import ToolRegistry


def test_registers_web_search_tool(config):
    registry = ToolRegistry()
    WebSearchAdapter(config).register_into(registry)
    assert registry.names() == ["web_search"]


def test_web_search_is_not_destructive(config):
    """Read-only lookup → no confirmation gate."""
    registry = ToolRegistry()
    WebSearchAdapter(config).register_into(registry)
    assert registry.get("web_search").destructive is False


async def test_call_without_api_key_returns_structured_error(config):
    """No TAVILY_API_KEY → tool returns an error dict instead of raising."""
    cfg = config.model_copy(update={"tavily_api_key": None})
    result = await WebSearchAdapter(cfg).web_search("what is the weather in paris")
    assert isinstance(result, dict) and "error" in result
    assert "not configured" in result["error"].lower()


def test_normalize_trims_payload_and_keeps_answer():
    raw = {
        "answer": "Paris is sunny.",
        "results": [
            {
                "title": "Weather",
                "url": "https://example.com",
                "content": "Sunny, 24C.",
                "score": 0.9,
                "raw_content": "lots of noise we drop",
            }
        ],
    }
    out = WebSearchAdapter._normalize("paris weather", raw)
    assert out["query"] == "paris weather"
    assert out["answer"] == "Paris is sunny."
    assert out["results"] == [
        {"title": "Weather", "url": "https://example.com", "content": "Sunny, 24C."}
    ]


def test_normalize_omits_answer_when_absent():
    out = WebSearchAdapter._normalize("q", {"results": []})
    assert "answer" not in out
    assert out["results"] == []
