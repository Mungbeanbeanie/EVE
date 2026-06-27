"""Microsoft 365 adapter — Outlook Mail, Calendar, OneDrive (via Microsoft Graph).

Stubbed for later: it implements the SAME ToolAdapter interface as GoogleAdapter,
so when you're ready you fill these in exactly like the Google one and register it
in Agent.from_config — no other code changes.

Auth uses MSAL (already in requirements.txt) to get a Graph access token; calls go
to https://graph.microsoft.com/v1.0/...
"""

from __future__ import annotations

from eve.config import Config
from eve.tools.base import Tool, ToolAdapter
from eve.tools.registry import ToolRegistry

SCOPES = [
    # TODO(eve): e.g. "Mail.Read", "Calendars.ReadWrite", "Files.ReadWrite"
]


class MicrosoftAdapter(ToolAdapter):
    """Exposes Outlook/Calendar/OneDrive capabilities as Tools (stub)."""

    def __init__(self, config: Config) -> None:
        self.client_id = config.microsoft_client_id
        self._token = None

    def _access_token(self) -> str:
        """Acquire a Microsoft Graph access token via MSAL."""
        # TODO(eve): use msal.PublicClientApplication (device-code or interactive flow),
        #            cache the token, refresh as needed.
        raise NotImplementedError(
            "Implement Microsoft auth — see eve/tools/adapters/microsoft.py:_access_token"
        )

    def register_into(self, registry: ToolRegistry) -> None:
        """Create Microsoft tools and add them to the registry."""
        registry.register(
            Tool(
                name="outlook_search",
                description="Search the user's Outlook mail and return message summaries.",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search text"},
                        "max_results": {"type": "integer", "default": 10},
                    },
                    "required": ["query"],
                },
                handler=self.outlook_search,
            )
        )
        # TODO(eve): register calendar/onedrive tools, mirroring the Google adapter.

    async def outlook_search(self, query: str, max_results: int = 10):
        """Search Outlook mail via Microsoft Graph."""
        # TODO(eve): GET https://graph.microsoft.com/v1.0/me/messages?$search="{query}"
        #            with the bearer token from self._access_token(); normalize results.
        raise NotImplementedError(
            "Implement outlook_search — see eve/tools/adapters/microsoft.py:outlook_search"
        )
