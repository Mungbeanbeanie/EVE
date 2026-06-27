"""Microsoft 365 adapter — Outlook Mail, Calendar, OneDrive (via Microsoft Graph).

Implements the SAME ToolAdapter interface as GoogleAdapter: build Graph-backed
Tools and register them. Auth uses MSAL's device-code flow (no client secret, no
redirect server needed) to get a delegated access token, cached on disk between
runs. Calls go to https://graph.microsoft.com/v1.0/...

Heavy SDKs (msal, requests) are imported lazily inside the methods that need them,
so importing this module — and the whole package — stays cheap.
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

# Delegated Graph permissions — the narrowest that cover the tools below. MSAL adds
# the reserved scopes (openid/profile/offline_access) itself, so don't list them.
SCOPES = [
    "Mail.Read",            # outlook_search
    "Calendars.ReadWrite",  # calendar read + create
    "Files.Read",           # onedrive_list_files
]

_GRAPH_BASE = "https://graph.microsoft.com/v1.0"
_AUTHORITY_HOST = "https://login.microsoftonline.com"


class MicrosoftAdapter(ToolAdapter):
    """Exposes Outlook/Calendar/OneDrive capabilities as Tools."""

    def __init__(self, config: Config) -> None:
        self.client_id = config.microsoft_client_id
        self.authority = f"{_AUTHORITY_HOST}/{config.microsoft_tenant_id}"
        self.cache_path = config.microsoft_token_cache_path
        self._app = None  # cached MSAL PublicClientApplication

    # ── Auth ───────────────────────────────────────────────────────────────────
    def _load_cache(self):
        """Return an MSAL token cache, seeded from disk if a prior token was saved."""
        from msal import SerializableTokenCache

        cache = SerializableTokenCache()
        cache_path = Path(self.cache_path)
        if cache_path.exists():
            cache.deserialize(cache_path.read_text())
        return cache

    def _save_cache(self, cache) -> None:
        """Persist the token cache to disk if MSAL changed it (new/refreshed token)."""
        if not cache.has_state_changed:
            return
        cache_path = Path(self.cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(cache.serialize())

    def _application(self, cache):
        """Return a cached MSAL PublicClientApplication bound to `cache`."""
        from msal import PublicClientApplication

        if self._app is None:
            if not self.client_id:
                raise RuntimeError(
                    "Microsoft auth is not configured. Set MICROSOFT_CLIENT_ID in .env "
                    "(see README → Microsoft tools)."
                )
            self._app = PublicClientApplication(
                self.client_id, authority=self.authority, token_cache=cache
            )
        return self._app

    def _access_token(self) -> str:
        """Acquire a Microsoft Graph access token, refreshing or prompting as needed.

        Tries the cached token first (silent refresh); if that fails it runs the
        device-code flow, printing a short code for the user to enter at
        microsoft.com/devicelogin. The token is cached on disk for next time.
        """
        cache = self._load_cache()
        app = self._application(cache)

        # 1. Reuse a cached/refreshable token without prompting.
        accounts = app.get_accounts()
        result = (
            app.acquire_token_silent(SCOPES, account=accounts[0]) if accounts else None
        )

        # 2. Otherwise fall back to the interactive device-code flow.
        if not result:
            flow = app.initiate_device_flow(scopes=SCOPES)
            if "user_code" not in flow:
                raise RuntimeError(f"Failed to start device flow: {flow.get('error_description')}")
            log.info("Microsoft sign-in required: %s", flow["message"])
            print(flow["message"])  # e.g. "go to microsoft.com/devicelogin and enter CODE"
            result = app.acquire_token_by_device_flow(flow)  # blocks until the user finishes

        self._save_cache(cache)

        if "access_token" not in result:
            raise RuntimeError(
                f"Microsoft auth failed: {result.get('error_description', result.get('error'))}"
            )
        return result["access_token"]

    # ── HTTP ───────────────────────────────────────────────────────────────────
    def _graph(self, method: str, path: str, *, params: dict | None = None,
               json_body: dict | None = None) -> dict:
        """Blocking Graph request with a bearer token; raises on HTTP error."""
        import requests

        resp = requests.request(
            method,
            f"{_GRAPH_BASE}{path}",
            headers={"Authorization": f"Bearer {self._access_token()}"},
            params=params,
            json=json_body,
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json() if resp.content else {}

    async def _call(self, fn: Callable[..., Any], *args: Any) -> Any:
        """Run a blocking Graph call off-thread, returning a structured error on failure.

        Tools must not crash the turn: the model gets an `{"error": ...}` dict it can
        read and recover from (e.g. ask the user to authenticate or rephrase).
        """
        try:
            return await asyncio.to_thread(fn, *args)
        except Exception as exc:  # network / auth / API errors → readable for the model
            log.warning("Microsoft tool call failed: %s", exc)
            return {"error": f"{type(exc).__name__}: {exc}"}

    # ── Registration ───────────────────────────────────────────────────────────
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
        registry.register(
            Tool(
                name="outlook_list_events",
                description="List upcoming events from the user's Outlook calendar.",
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
                handler=self.outlook_list_events,
            )
        )
        registry.register(
            Tool(
                name="outlook_create_event",
                description="Create an event on the user's Outlook calendar.",
                parameters={
                    "type": "object",
                    "properties": {
                        "summary": {"type": "string", "description": "Event title"},
                        "start": {
                            "type": "string",
                            "description": "Start time, e.g. 2026-07-01T15:00:00",
                        },
                        "end": {
                            "type": "string",
                            "description": "End time, e.g. 2026-07-01T16:00:00",
                        },
                        "timezone": {
                            "type": "string",
                            "description": "IANA/Windows time zone for start & end. Defaults to UTC.",
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
                handler=self.outlook_create_event,
                destructive=True,  # writes to the user's calendar → confirm before run
            )
        )
        registry.register(
            Tool(
                name="onedrive_list_files",
                description="List files in the user's OneDrive, optionally filtered by a search term.",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search term to match file names/content",
                        },
                        "max_results": {"type": "integer", "default": 10},
                    },
                    "required": [],
                },
                handler=self.onedrive_list_files,
            )
        )

    # ── Tool handlers ──────────────────────────────────────────────────────────
    async def outlook_search(self, query: str, max_results: int = 10):
        """Search Outlook mail and return lightweight message summaries."""
        return await self._call(self._outlook_search_blocking, query, max_results)

    def _outlook_search_blocking(self, query: str, max_results: int) -> list[dict]:
        data = self._graph(
            "GET",
            "/me/messages",
            params={
                "$search": f'"{query}"',
                "$top": max_results,
                "$select": "id,subject,from,receivedDateTime,bodyPreview",
            },
        )
        return [
            {
                "id": msg.get("id"),
                "from": msg.get("from", {}).get("emailAddress", {}).get("address", ""),
                "subject": msg.get("subject", ""),
                "date": msg.get("receivedDateTime", ""),
                "snippet": msg.get("bodyPreview", ""),
            }
            for msg in data.get("value", [])
        ]

    async def outlook_list_events(self, max_results: int = 10, time_min: str | None = None):
        """List upcoming calendar events."""
        return await self._call(self._outlook_list_events_blocking, max_results, time_min)

    def _outlook_list_events_blocking(
        self, max_results: int, time_min: str | None
    ) -> list[dict]:
        lower_bound = time_min or datetime.now(timezone.utc).isoformat()
        data = self._graph(
            "GET",
            "/me/events",
            params={
                "$filter": f"start/dateTime ge '{lower_bound}'",
                "$orderby": "start/dateTime",
                "$top": max_results,
                "$select": "id,subject,start,end,location",
            },
        )
        return [
            {
                "id": event.get("id"),
                "summary": event.get("subject", "(no title)"),
                "start": event.get("start", {}).get("dateTime"),
                "end": event.get("end", {}).get("dateTime"),
                "location": event.get("location", {}).get("displayName", ""),
            }
            for event in data.get("value", [])
        ]

    async def outlook_create_event(
        self,
        summary: str,
        start: str,
        end: str,
        timezone: str = "UTC",
        description: str | None = None,
        attendees: list[str] | None = None,
    ):
        """Create a calendar event."""
        return await self._call(
            self._outlook_create_event_blocking,
            summary,
            start,
            end,
            timezone,
            description,
            attendees,
        )

    def _outlook_create_event_blocking(
        self,
        summary: str,
        start: str,
        end: str,
        tz: str,
        description: str | None,
        attendees: list[str] | None,
    ) -> dict:
        body: dict[str, Any] = {
            "subject": summary,
            "start": {"dateTime": start, "timeZone": tz},
            "end": {"dateTime": end, "timeZone": tz},
        }
        if description:
            body["body"] = {"contentType": "text", "content": description}
        if attendees:
            body["attendees"] = [
                {"emailAddress": {"address": email}, "type": "required"}
                for email in attendees
            ]
        event = self._graph("POST", "/me/events", json_body=body)
        return {
            "id": event.get("id"),
            "summary": event.get("subject"),
            "link": event.get("webLink"),
            "status": event.get("responseStatus", {}).get("response"),
        }

    async def onedrive_list_files(self, query: str | None = None, max_results: int = 10):
        """List OneDrive files (optionally filtered by a search term)."""
        return await self._call(self._onedrive_list_files_blocking, query, max_results)

    def _onedrive_list_files_blocking(
        self, query: str | None, max_results: int
    ) -> list[dict]:
        # Search when a term is given, otherwise list the drive root's children.
        path = f"/me/drive/root/search(q='{query}')" if query else "/me/drive/root/children"
        data = self._graph(
            "GET",
            path,
            params={
                "$top": max_results,
                "$select": "id,name,file,folder,lastModifiedDateTime,webUrl",
            },
        )
        return [
            {
                "id": f.get("id"),
                "name": f.get("name"),
                "mimeType": f.get("file", {}).get("mimeType")
                or ("folder" if "folder" in f else ""),
                "modifiedTime": f.get("lastModifiedDateTime"),
                "link": f.get("webUrl"),
            }
            for f in data.get("value", [])
        ]
