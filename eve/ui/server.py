"""VizServer — serves the EVE window and pushes live agent state to it.

Zero third-party dependencies: a stdlib ``ThreadingHTTPServer`` serves the
static window (``eve/ui/web``) and exposes a Server-Sent Events stream at
``/events``. The Python agent calls :meth:`VizServer.set_state` whenever it
moves between ``idle / listening / thinking / speaking``; every connected window
receives the update and drives the orb to match.

State flows one way (agent → browser), which is exactly what SSE is for, so we
avoid pulling in a WebSocket library. The server runs in a daemon thread and
``set_state`` is safe to call from the asyncio agent thread.

Every queued SSE frame carries its own event name: persistent orb state rides the
``state`` event (replayed to new connections), while transient reply captions ride
the ``reply`` event (one-shot, never replayed). See :meth:`VizServer._broadcast`.
"""

from __future__ import annotations

import json
import logging
import mimetypes
import queue
import threading
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from eve.ui.bridge import InputBridge

log = logging.getLogger(__name__)

WEB_DIR = Path(__file__).parent / "web"

# A frame is at most this many seconds away from a heartbeat comment, which lets
# the server notice a window that has gone away and reclaim its thread.
_HEARTBEAT_SECONDS = 15.0

_VALID_STATES = frozenset({"idle", "listening", "thinking", "speaking"})


class VizServer:
    """Hosts the EVE window and broadcasts agent state to connected browsers."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8765,
        accent: str = "amber",
        bridge: "InputBridge | None" = None,
        on_stop_speech: "Callable[[], None] | None" = None,
        on_stop_listen: "Callable[[], None] | None" = None,
    ) -> None:
        self.host = host
        self.port = port
        # Optional browser → agent channel. When attached, the window's text box
        # and mic button POST here and the agent consumes the events; when absent
        # those POSTs are simply rejected (the orb still works receive-only).
        self.bridge = bridge
        # Called directly from the HTTP thread when the user hits Stop — bypasses
        # the bridge so it fires even while the agent is blocked inside speak().
        self.on_stop_speech = on_stop_speech
        # Called directly from the HTTP thread to end an in-progress mic capture
        # (push-to-talk's second tap). Like on_stop_speech it bypasses the bridge:
        # the agent is blocked inside record_utterance on the worker thread, so a
        # bridge event would not be read until recording already ended.
        self.on_stop_listen = on_stop_listen
        self._state: dict[str, str] = {"state": "idle", "accent": accent}
        # Queued frames carry their own SSE event name: {"event": <name>, "data": ...}.
        self._subscribers: set[queue.Queue[dict]] = set()
        self._lock = threading.Lock()
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    # ── lifecycle ────────────────────────────────────────────────────────────
    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/"

    def start(self, *, open_browser: bool = False) -> "VizServer":
        """Start the HTTP server in a background daemon thread."""
        handler = _make_handler(self)
        self._httpd = ThreadingHTTPServer((self.host, self.port), handler)
        # Reflect the bound port back (useful when port=0 picks a free one).
        self.port = self._httpd.server_address[1]
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        log.info("EVE window available at %s", self.url)
        if open_browser:
            import webbrowser

            webbrowser.open(self.url)
        return self

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None

    # ── broadcast API (thread-safe) ──────────────────────────────────────────
    def set_state(self, name: str) -> None:
        """Push a new orb state (``idle / listening / thinking / speaking``)."""
        if name not in _VALID_STATES:
            log.debug("Ignoring unknown viz state %r", name)
            return
        self._update(state=name)

    def set_accent(self, key: str) -> None:
        """Push a new accent palette key (``amber / cyan / violet / mono``)."""
        self._update(accent=key)

    def push_reply(self, role: str, text: str) -> None:
        """Broadcast a one-way reply caption (``role`` is "you" or "eve").

        TRANSIENT by design: unlike orb state this is *not* merged into
        ``self._state``, so a browser that connects later does not replay a stale
        line — it only ever sees captions that arrive while it is connected.
        """
        self._broadcast({"event": "reply", "data": {"role": role, "text": text}})

    def _update(self, **changes: str) -> None:
        """Merge ``changes`` into the persistent state and broadcast a snapshot."""
        with self._lock:
            self._state = {**self._state, **changes}
            snapshot = dict(self._state)
        self._broadcast({"event": "state", "data": snapshot})

    def _broadcast(self, frame: dict) -> None:
        """Fan one ``{"event", "data"}`` frame out to every subscriber queue."""
        with self._lock:
            subscribers = list(self._subscribers)
        for q in subscribers:
            try:
                q.put_nowait(frame)
            except queue.Full:  # pragma: no cover - unbounded queue
                pass

    # ── subscriber registry (used by the SSE handler) ────────────────────────
    def _subscribe(self) -> "tuple[queue.Queue[dict], dict]":
        """Register a new SSE client; return its queue and the initial state frame."""
        q: queue.Queue[dict] = queue.Queue()
        with self._lock:
            self._subscribers.add(q)
            snapshot = dict(self._state)
        return q, {"event": "state", "data": snapshot}

    def _unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            self._subscribers.discard(q)


def _make_handler(viz: VizServer) -> type[BaseHTTPRequestHandler]:
    """Build a request handler bound to a specific VizServer instance."""

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_args) -> None:  # silence default stderr logging
            pass

        def do_GET(self) -> None:  # noqa: N802 - stdlib naming
            path = self.path.split("?", 1)[0]
            if path == "/events":
                self._serve_events()
            else:
                self._serve_static(path)

        # ---- browser → agent input (text box + mic button) ----
        def do_POST(self) -> None:  # noqa: N802 - stdlib naming
            path = self.path.split("?", 1)[0]
            if path == "/input":
                self._handle_input()
            elif path == "/control":
                self._handle_control()
            else:
                self.send_error(404, "Not Found")

        def _read_json(self) -> dict:
            """Parse the request body as JSON, or return {} on any problem."""
            try:
                length = int(self.headers.get("Content-Length", 0))
            except (TypeError, ValueError):
                return {}
            if length <= 0:
                return {}
            try:
                return json.loads(self.rfile.read(length).decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                return {}

        def _handle_input(self) -> None:
            """POST /input {"text": "..."} → queue a typed prompt for the agent."""
            if viz.bridge is None:
                self.send_error(503, "No agent attached")
                return
            text = str(self._read_json().get("text", "")).strip()
            if not text:
                self.send_error(400, "Empty text")
                return
            viz.bridge.submit_text(text)
            self._send_no_content()

        def _handle_control(self) -> None:
            """POST /control {"action": "..."} → send a control signal to the agent."""
            action = str(self._read_json().get("action", "")).strip()
            # Stop actions are direct callbacks: they must fire from the HTTP thread
            # even while the agent is blocked inside speak()/record_utterance(), so
            # they bypass the bridge entirely (which the agent only drains at idle).
            if action == "stop_speech":
                if viz.on_stop_speech is not None:
                    viz.on_stop_speech()
                self._send_no_content()
                return
            if action == "stop_listen":
                if viz.on_stop_listen is not None:
                    viz.on_stop_listen()
                self._send_no_content()
                return
            if viz.bridge is None:
                self.send_error(503, "No agent attached")
                return
            if action != "listen":
                self.send_error(400, "Unknown action")
                return
            viz.bridge.submit_control(action)
            self._send_no_content()

        def _send_no_content(self) -> None:
            self.send_response(204)
            self.send_header("Content-Length", "0")
            self.end_headers()

        # ---- static window ----
        def _serve_static(self, path: str) -> None:
            rel = "index.html" if path in ("/", "") else path.lstrip("/")
            target = (WEB_DIR / rel).resolve()
            # Path-traversal guard: never serve outside the web directory.
            if WEB_DIR.resolve() not in target.parents and target != WEB_DIR.resolve():
                self.send_error(403, "Forbidden")
                return
            if not target.is_file():
                self.send_error(404, "Not Found")
                return
            body = target.read_bytes()
            ctype = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
            if target.suffix == ".js":
                ctype = "text/javascript"
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        # ---- SSE state stream ----
        def _serve_events(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()

            q, initial = viz._subscribe()
            try:
                self._send_event(initial)
                while True:
                    try:
                        frame = q.get(timeout=_HEARTBEAT_SECONDS)
                    except queue.Empty:
                        self.wfile.write(b": ping\n\n")  # keep-alive comment
                        self.wfile.flush()
                        continue
                    self._send_event(frame)
            except (BrokenPipeError, ConnectionResetError):
                pass  # window closed
            finally:
                viz._unsubscribe(q)

        def _send_event(self, frame: dict) -> None:
            """Write one SSE frame, using the event name it carries."""
            data = json.dumps(frame["data"])
            self.wfile.write(f"event: {frame['event']}\ndata: {data}\n\n".encode("utf-8"))
            self.wfile.flush()

    return Handler
