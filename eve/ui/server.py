"""VizServer — serves the EVE window and pushes live agent state to it.

Zero third-party dependencies: a stdlib ``ThreadingHTTPServer`` serves the
static window (``eve/ui/web``) and exposes a Server-Sent Events stream at
``/events``. The Python agent calls :meth:`VizServer.set_state` whenever it
moves between ``idle / listening / thinking / speaking``; every connected window
receives the update and drives the orb to match.

State flows one way (agent → browser), which is exactly what SSE is for, so we
avoid pulling in a WebSocket library. The server runs in a daemon thread and
``set_state`` is safe to call from the asyncio agent thread.
"""

from __future__ import annotations

import json
import logging
import mimetypes
import queue
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

log = logging.getLogger(__name__)

WEB_DIR = Path(__file__).parent / "web"

# A frame is at most this many seconds away from a heartbeat comment, which lets
# the server notice a window that has gone away and reclaim its thread.
_HEARTBEAT_SECONDS = 15.0

_VALID_STATES = frozenset({"idle", "listening", "thinking", "speaking"})


class VizServer:
    """Hosts the EVE window and broadcasts agent state to connected browsers."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8765, accent: str = "amber") -> None:
        self.host = host
        self.port = port
        self._state: dict[str, str] = {"state": "idle", "accent": accent}
        self._subscribers: set[queue.Queue[dict[str, str]]] = set()
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

    def _update(self, **changes: str) -> None:
        with self._lock:
            self._state = {**self._state, **changes}
            snapshot = dict(self._state)
            subscribers = list(self._subscribers)
        for q in subscribers:
            try:
                q.put_nowait(snapshot)
            except queue.Full:  # pragma: no cover - unbounded queue
                pass

    # ── subscriber registry (used by the SSE handler) ────────────────────────
    def _subscribe(self) -> "tuple[queue.Queue[dict[str, str]], dict[str, str]]":
        q: queue.Queue[dict[str, str]] = queue.Queue()
        with self._lock:
            self._subscribers.add(q)
            snapshot = dict(self._state)
        return q, snapshot

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

            q, snapshot = viz._subscribe()
            try:
                self._send_event(snapshot)
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

        def _send_event(self, payload: dict[str, str]) -> None:
            data = json.dumps(payload)
            self.wfile.write(f"event: state\ndata: {data}\n\n".encode("utf-8"))
            self.wfile.flush()

    return Handler
