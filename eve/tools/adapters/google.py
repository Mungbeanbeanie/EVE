"""Google Workspace adapter — Gmail, Calendar, Drive.

Wraps Google APIs as EVE Tools. Auth is OAuth 2.0 (user consent): you exchange a
client id/secret for a token the first time, then reuse the cached token. Scopes
determine what EVE may touch — request the minimum you need.

Useful libraries (already in requirements.txt):
    google-auth-oauthlib   — the OAuth consent flow
    google-api-python-client — the Gmail/Calendar/Drive service clients

This adapter is scaffolded first; fill the TODOs to bring tools online one at a time.
"""

from __future__ import annotations

from eve.config import Config
from eve.tools.base import Tool, ToolAdapter
from eve.tools.registry import ToolRegistry

# Request the narrowest scopes that cover your features.
SCOPES = [
    # TODO(eve): e.g. "https://www.googleapis.com/auth/gmail.readonly",
    #                 "https://www.googleapis.com/auth/calendar.events",
    #                 "https://www.googleapis.com/auth/drive.file",
]


class GoogleAdapter(ToolAdapter):
    """Exposes Gmail/Calendar/Drive capabilities as Tools."""

    def __init__(self, config: Config) -> None:
        self.client_id = config.google_client_id
        self.client_secret = config.google_client_secret
        self.token_path = config.google_token_path
        self._creds = None  # cached OAuth credentials

    # ── Auth ─────────────────────────────────────────────────────────────────
    def _credentials(self):
        """Return valid OAuth credentials, running/refreshing the flow as needed."""
        # TODO(eve): 1. If a token file exists at self.token_path, load it.
        # TODO(eve): 2. If missing/expired, run InstalledAppFlow with SCOPES and
        #               your client id/secret, then save the token to self.token_path.
        # TODO(eve): 3. Refresh expired-but-refreshable creds. Cache on self._creds.
        raise NotImplementedError(
            "Implement Google OAuth — see eve/tools/adapters/google.py:_credentials"
        )

    # ── Registration ─────────────────────────────────────────────────────────
    def register_into(self, registry: ToolRegistry) -> None:
        """Create Google tools and add them to the registry."""
        registry.register(
            Tool(
                name="gmail_search",
                description="Search the user's Gmail and return matching message summaries.",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Gmail search query"},
                        "max_results": {"type": "integer", "default": 10},
                    },
                    "required": ["query"],
                },
                handler=self.gmail_search,
            )
        )
        # TODO(eve): register more tools — calendar_create_event, drive_list_files, etc.
        #            Follow the same Tool(...) shape, each pointing at a handler below.

    # ── Tool handlers ────────────────────────────────────────────────────────
    async def gmail_search(self, query: str, max_results: int = 10):
        """Search Gmail and return lightweight message summaries."""
        # TODO(eve): 1. service = build("gmail", "v1", credentials=self._credentials())
        # TODO(eve): 2. List message ids for `query`, fetch each, extract
        #               from/subject/snippet. Run blocking calls via asyncio.to_thread.
        # TODO(eve): 3. Return a compact list[dict] the model can read.
        raise NotImplementedError(
            "Implement gmail_search — see eve/tools/adapters/google.py:gmail_search"
        )
