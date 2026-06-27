"""Web search adapter — live internet search via the Tavily API.

Gives EVE a single read-only tool, ``web_search``, so the model can look up
current facts (news, prices, docs, anything past its training cutoff) instead of
guessing. Tavily is purpose-built for LLM agents: one HTTP call returns ranked,
de-duplicated results whose ``content`` is already trimmed to the relevant passage,
plus an optional synthesized ``answer``.

Auth is a single API key (no OAuth, no per-user consent) — get one at
https://app.tavily.com and set ``TAVILY_API_KEY`` in ``.env``. The key is read
lazily, so importing this module without a key configured is fine; the tool only
errors (recoverably) if it is actually called without one.

``requests`` is imported lazily inside the blocking call, matching the other
adapters, so importing this module stays cheap.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable

from eve.config import Config
from eve.tools.base import Tool, ToolAdapter
from eve.tools.registry import ToolRegistry

log = logging.getLogger(__name__)

_TAVILY_ENDPOINT = "https://api.tavily.com/search"

# Tavily caps max_results at 20; keep our default small so the model gets a tight,
# readable context window rather than 20 noisy hits.
_DEFAULT_MAX_RESULTS = 5
_MAX_RESULTS_CAP = 20


class WebSearchAdapter(ToolAdapter):
    """Exposes a single ``web_search`` Tool backed by the Tavily API."""

    def __init__(self, config: Config) -> None:
        self.api_key = config.tavily_api_key

    # ── HTTP ───────────────────────────────────────────────────────────────────
    def _search_blocking(
        self, query: str, max_results: int, topic: str, include_answer: bool
    ) -> dict:
        """Blocking Tavily request; raises on missing key or HTTP error."""
        if not self.api_key:
            raise RuntimeError(
                "Web search is not configured. Set TAVILY_API_KEY in .env "
                "(get a free key at https://app.tavily.com)."
            )
        import requests

        # Clamp to Tavily's accepted range so a model asking for 1000 results
        # doesn't trip a 4xx; topic falls back to "general" for unknown values.
        bounded = max(1, min(int(max_results), _MAX_RESULTS_CAP))
        safe_topic = topic if topic in ("general", "news") else "general"

        resp = requests.post(
            _TAVILY_ENDPOINT,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                # api_key in the body too: harmless with the Bearer header and keeps
                # this working against older Tavily deployments that read it there.
                "api_key": self.api_key,
                "query": query,
                "max_results": bounded,
                "topic": safe_topic,
                "search_depth": "basic",
                "include_answer": include_answer,
            },
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json() if resp.content else {}

    async def _call(self, fn: Callable[..., Any], *args: Any) -> Any:
        """Run a blocking search off-thread, returning a structured error on failure.

        Tools must not crash the turn: the model gets an ``{"error": ...}`` dict it
        can read and recover from (e.g. tell the user search is unavailable).
        """
        try:
            return await asyncio.to_thread(fn, *args)
        except Exception as exc:  # network / auth / API errors → readable for the model
            log.warning("Web search call failed: %s", exc)
            return {"error": f"{type(exc).__name__}: {exc}"}

    # ── Registration ───────────────────────────────────────────────────────────
    def register_into(self, registry: ToolRegistry) -> None:
        """Create the web_search tool and add it to the registry."""
        registry.register(
            Tool(
                name="web_search",
                description=(
                    "Search the live web for current information (news, facts, prices, "
                    "documentation) that may be newer than your training data. Returns "
                    "ranked results with a title, URL, and a relevant text snippet, plus "
                    "a short synthesized answer when available."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query, phrased as you would type it into a search engine.",
                        },
                        "max_results": {
                            "type": "integer",
                            "description": f"How many results to return (1-{_MAX_RESULTS_CAP}).",
                            "default": _DEFAULT_MAX_RESULTS,
                        },
                        "topic": {
                            "type": "string",
                            "description": "'general' for most queries, or 'news' for recent/breaking events.",
                            "enum": ["general", "news"],
                            "default": "general",
                        },
                    },
                    "required": ["query"],
                },
                handler=self.web_search,
                # Read-only lookup, no real-world side effect → no confirmation needed.
                destructive=False,
            )
        )

    # ── Tool handler ───────────────────────────────────────────────────────────
    async def web_search(
        self,
        query: str,
        max_results: int = _DEFAULT_MAX_RESULTS,
        topic: str = "general",
    ) -> dict:
        """Search the web and return a compact, model-readable result set."""
        raw = await self._call(
            self._search_blocking, query, max_results, topic, True
        )
        if isinstance(raw, dict) and "error" in raw:
            return raw  # propagate the structured error unchanged
        return self._normalize(query, raw)

    @staticmethod
    def _normalize(query: str, raw: dict) -> dict:
        """Trim Tavily's payload to the fields the model actually needs."""
        results = [
            {
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "content": item.get("content", ""),
            }
            for item in raw.get("results", [])
        ]
        normalized: dict[str, Any] = {"query": query, "results": results}
        answer = raw.get("answer")
        if answer:
            normalized["answer"] = answer
        return normalized
