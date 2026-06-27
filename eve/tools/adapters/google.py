"""Google Workspace adapter — Gmail, Calendar, Drive.

Wraps Google APIs as EVE Tools. Auth is OAuth 2.0 (user consent): you exchange a
client id/secret for a token the first time, then reuse the cached token. Scopes
determine what EVE may touch — request the minimum you need.

Useful libraries (already in requirements.txt):
    google-auth-oauthlib   — the OAuth consent flow
    google-api-python-client — the Gmail/Calendar/Drive service clients

Heavy Google SDKs are imported lazily inside the methods that need them, so simply
importing this module (and therefore the whole package) stays cheap.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from eve.config import Config
from eve.tools.base import Tool, ToolAdapter
from eve.tools.registry import ToolRegistry

log = logging.getLogger(__name__)

# Request the narrowest scopes that cover the tools below. Changing this list
# invalidates an existing token — delete the token file to re-consent.
SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",       # gmail_search
    "https://www.googleapis.com/auth/calendar.events",      # calendar read + create
    "https://www.googleapis.com/auth/drive.metadata.readonly",  # drive_list_files
]

# OAuth endpoints — constant for all Google "Desktop app" clients.
_AUTH_URI = "https://accounts.google.com/o/oauth2/auth"
_TOKEN_URI = "https://oauth2.googleapis.com/token"


class GoogleAdapter(ToolAdapter):
    """Exposes Gmail/Calendar/Drive capabilities as Tools."""

    def __init__(self, config: Config) -> None:
        self.client_id = config.google_client_id
        self.client_secret = config.google_client_secret
        self.token_path = config.google_token_path
        self._creds = None  # cached OAuth credentials
        self._services: dict[tuple[str, str], Any] = {}  # cached API clients

    # ── Auth ─────────────────────────────────────────────────────────────────
    def _client_config(self) -> dict:
        """Build the in-memory client-secrets dict InstalledAppFlow expects."""
        return {
            "installed": {
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "auth_uri": _AUTH_URI,
                "token_uri": _TOKEN_URI,
                "redirect_uris": ["http://localhost"],
            }
        }

    def _credentials(self):
        """Return valid OAuth credentials, running/refreshing the flow as needed."""
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow

        if self._creds and self._creds.valid:
            return self._creds

        creds = None
        token_path = Path(self.token_path)

        # 1. Load a previously-saved token if present.
        if token_path.exists():
            creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
            if creds and creds.valid:
                self._creds = creds
                return creds

        # 2. Refresh an expired-but-refreshable token; otherwise run consent flow.
        if creds and creds.expired and creds.refresh_token:
            log.info("Refreshing expired Google credentials")
            creds.refresh(Request())
        else:
            if not (self.client_id and self.client_secret):
                raise RuntimeError(
                    "Google OAuth is not configured. Set GOOGLE_CLIENT_ID and "
                    "GOOGLE_CLIENT_SECRET in .env (see README → Google tools)."
                )
            log.info("Starting Google OAuth consent flow (a browser window will open)")
            flow = InstalledAppFlow.from_client_config(self._client_config(), SCOPES)
            creds = flow.run_local_server(port=0)

        # 3. Persist the token for next time and cache in memory.
        token_path.parent.mkdir(parents=True, exist_ok=True)
        token_path.write_text(creds.to_json())
        self._creds = creds
        return creds

    def _service(self, api: str, version: str):
        """Return a cached Google API client for (api, version)."""
        key = (api, version)
        if key not in self._services:
            from googleapiclient.discovery import build

            self._services[key] = build(
                api, version, credentials=self._credentials(), cache_discovery=False
            )
        return self._services[key]

    async def _call(self, fn: Callable[..., Any], *args: Any) -> Any:
        """Run a blocking Google call off-thread, returning a structured error on failure.

        Tools must not crash the turn: the model gets an `{"error": ...}` dict it can
        read and recover from (e.g. ask the user to authenticate or rephrase).
        """
        try:
            return await asyncio.to_thread(fn, *args)
        except Exception as exc:  # network / auth / API errors → readable for the model
            log.warning("Google tool call failed: %s", exc)
            return {"error": f"{type(exc).__name__}: {exc}"}

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
                        "query": {
                            "type": "string",
                            "description": "Gmail search query, e.g. 'from:sam is:unread newer_than:7d'",
                        },
                        "max_results": {"type": "integer", "default": 10},
                    },
                    "required": ["query"],
                },
                handler=self.gmail_search,
            )
        )
        registry.register(
            Tool(
                name="calendar_list_events",
                description="List upcoming events from the user's primary Google Calendar.",
                parameters={
                    "type": "object",
                    "properties": {
                        "max_results": {"type": "integer", "default": 10},
                        "time_min": {
                            "type": "string",
                            "description": "RFC3339 lower bound (e.g. 2026-07-01T00:00:00Z). "
                            "Defaults to now.",
                        },
                    },
                    "required": [],
                },
                handler=self.calendar_list_events,
            )
        )
        registry.register(
            Tool(
                name="calendar_create_event",
                description="Create an event on the user's primary Google Calendar.",
                parameters={
                    "type": "object",
                    "properties": {
                        "summary": {"type": "string", "description": "Event title"},
                        "start": {
                            "type": "string",
                            "description": "Start time, RFC3339 with offset, e.g. 2026-07-01T15:00:00-04:00",
                        },
                        "end": {
                            "type": "string",
                            "description": "End time, RFC3339 with offset, e.g. 2026-07-01T16:00:00-04:00",
                        },
                        "description": {"type": "string"},
                        "attendees": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Attendee email addresses",
                        },
                    },
                    "required": ["summary", "start", "end"],
                },
                handler=self.calendar_create_event,
                destructive=True,  # writes to the user's calendar → confirm before run
            )
        )
        registry.register(
            Tool(
                name="drive_list_files",
                description="List files in the user's Google Drive, optionally filtered by a query.",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Drive query, e.g. \"name contains 'budget'\" or "
                            "\"mimeType='application/pdf'\"",
                        },
                        "max_results": {"type": "integer", "default": 10},
                    },
                    "required": [],
                },
                handler=self.drive_list_files,
            )
        )

    # ── Tool handlers ────────────────────────────────────────────────────────
    async def gmail_search(self, query: str, max_results: int = 10):
        """Search Gmail and return lightweight message summaries."""
        return await self._call(self._gmail_search_blocking, query, max_results)

    def _gmail_search_blocking(self, query: str, max_results: int) -> list[dict]:
        service = self._service("gmail", "v1")
        listed = (
            service.users()
            .messages()
            .list(userId="me", q=query, maxResults=max_results)
            .execute()
        )
        summaries: list[dict] = []
        for ref in listed.get("messages", []):
            msg = (
                service.users()
                .messages()
                .get(
                    userId="me",
                    id=ref["id"],
                    format="metadata",
                    metadataHeaders=["From", "Subject", "Date"],
                )
                .execute()
            )
            headers = {
                h["name"]: h["value"] for h in msg.get("payload", {}).get("headers", [])
            }
            summaries.append(
                {
                    "id": ref["id"],
                    "from": headers.get("From", ""),
                    "subject": headers.get("Subject", ""),
                    "date": headers.get("Date", ""),
                    "snippet": msg.get("snippet", ""),
                }
            )
        return summaries

    async def calendar_list_events(self, max_results: int = 10, time_min: str | None = None):
        """List upcoming primary-calendar events."""
        return await self._call(self._calendar_list_events_blocking, max_results, time_min)

    def _calendar_list_events_blocking(
        self, max_results: int, time_min: str | None
    ) -> list[dict]:
        service = self._service("calendar", "v3")
        lower_bound = time_min or datetime.now(timezone.utc).isoformat()
        listed = (
            service.events()
            .list(
                calendarId="primary",
                timeMin=lower_bound,
                maxResults=max_results,
                singleEvents=True,
                orderBy="startTime",
            )
            .execute()
        )
        return [
            {
                "id": event.get("id"),
                "summary": event.get("summary", "(no title)"),
                "start": event.get("start", {}).get("dateTime")
                or event.get("start", {}).get("date"),
                "end": event.get("end", {}).get("dateTime")
                or event.get("end", {}).get("date"),
                "location": event.get("location", ""),
            }
            for event in listed.get("items", [])
        ]

    async def calendar_create_event(
        self,
        summary: str,
        start: str,
        end: str,
        description: str | None = None,
        attendees: list[str] | None = None,
    ):
        """Create a primary-calendar event."""
        return await self._call(
            self._calendar_create_event_blocking,
            summary,
            start,
            end,
            description,
            attendees,
        )

    def _calendar_create_event_blocking(
        self,
        summary: str,
        start: str,
        end: str,
        description: str | None,
        attendees: list[str] | None,
    ) -> dict:
        service = self._service("calendar", "v3")
        body: dict[str, Any] = {
            "summary": summary,
            "start": {"dateTime": start},
            "end": {"dateTime": end},
        }
        if description:
            body["description"] = description
        if attendees:
            body["attendees"] = [{"email": email} for email in attendees]
        event = service.events().insert(calendarId="primary", body=body).execute()
        return {
            "id": event.get("id"),
            "summary": event.get("summary"),
            "htmlLink": event.get("htmlLink"),
            "status": event.get("status"),
        }

    async def drive_list_files(self, query: str | None = None, max_results: int = 10):
        """List Drive files (optionally filtered)."""
        return await self._call(self._drive_list_files_blocking, query, max_results)

    def _drive_list_files_blocking(
        self, query: str | None, max_results: int
    ) -> list[dict]:
        service = self._service("drive", "v3")
        params: dict[str, Any] = {
            "pageSize": max_results,
            "fields": "files(id, name, mimeType, modifiedTime, webViewLink)",
            "orderBy": "modifiedTime desc",
        }
        if query:
            params["q"] = query
        listed = service.files().list(**params).execute()
        return [
            {
                "id": f.get("id"),
                "name": f.get("name"),
                "mimeType": f.get("mimeType"),
                "modifiedTime": f.get("modifiedTime"),
                "link": f.get("webViewLink"),
            }
            for f in listed.get("files", [])
        ]
