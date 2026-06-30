"""Tests for the UI → agent InputBridge and the VizServer broadcast layer.

The bridge is a thin thread-safe queue; these check that submitted events come
back out, in order, in the shape the agent's window loop expects.

The VizServer tests below cover the SSE-side contracts the window depends on:
every queued frame carries its own ``{"event", "data"}`` shape, ``push_reply``
fans a transient ``reply`` caption out without polluting replayed state, and the
direct ``stop_listen`` / ``stop_speech`` control callbacks fire from the HTTP
thread (bypassing the bridge). These stay hardware-free — no real audio is ever
imported; stop callbacks are plain spies.
"""

from __future__ import annotations

import io
import json

from eve.ui.bridge import InputBridge
from eve.ui.server import VizServer, _make_handler


async def test_submit_text_round_trips():
    bridge = InputBridge()
    bridge.submit_text("hello eve")
    assert await bridge.next_event() == {"type": "text", "text": "hello eve"}


async def test_submit_control_round_trips():
    bridge = InputBridge()
    bridge.submit_control("listen")
    assert await bridge.next_event() == {"type": "control", "action": "listen"}


async def test_stop_emits_sentinel():
    bridge = InputBridge()
    bridge.stop()
    assert await bridge.next_event() == {"type": "stop"}


async def test_events_preserve_order():
    bridge = InputBridge()
    bridge.submit_text("first")
    bridge.submit_control("listen")
    bridge.stop()
    assert (await bridge.next_event())["text"] == "first"
    assert (await bridge.next_event())["action"] == "listen"
    assert (await bridge.next_event())["type"] == "stop"


# ── VizServer broadcast / SSE frame contract ─────────────────────────────────

def _drain(q):
    """Pop every frame currently on a subscriber queue (non-blocking)."""
    frames = []
    while not q.empty():
        frames.append(q.get_nowait())
    return frames


def test_subscribe_returns_initial_state_frame():
    """A new SSE client gets the current state as an ``{"event","data"}`` frame."""
    viz = VizServer(accent="cyan")
    q, initial = viz._subscribe()
    assert initial == {"event": "state", "data": {"state": "idle", "accent": "cyan"}}
    assert _drain(q) == []  # nothing queued yet beyond the initial snapshot


def test_set_state_broadcasts_named_state_frame():
    """set_state fans a ``state`` frame (not a bare dict) to every subscriber."""
    viz = VizServer()
    q, _initial = viz._subscribe()
    viz.set_state("thinking")
    assert _drain(q) == [
        {"event": "state", "data": {"state": "thinking", "accent": "amber"}}
    ]


def test_push_reply_broadcasts_reply_frame():
    """push_reply emits the fixed ``reply`` contract: data == {role, text}."""
    viz = VizServer()
    q, _initial = viz._subscribe()
    viz.push_reply("eve", "Hi there!")
    assert _drain(q) == [
        {"event": "reply", "data": {"role": "eve", "text": "Hi there!"}}
    ]


def test_push_reply_is_transient_not_replayed():
    """A reply must NOT merge into state, so later subscribers don't replay it."""
    viz = VizServer()
    viz.push_reply("eve", "old line")
    # A client that connects afterwards only ever sees the (reply-free) state.
    _q, initial = viz._subscribe()
    assert initial == {"event": "state", "data": {"state": "idle", "accent": "amber"}}
    assert "old line" not in json.dumps(viz._state)


def _make_request_handler(viz: VizServer):
    """Build a VizServer Handler wired to a fake socket so HTTP branches run.

    Lets us exercise the real ``do_POST`` / ``_handle_control`` path (the
    bridge-bypassing stop callbacks) without binding a port or opening a mic.
    """
    handler_cls = _make_handler(viz)

    class _FakeHandler(handler_cls):
        def __init__(self, body: bytes) -> None:
            # Skip BaseHTTPRequestHandler.__init__ (it wants a live socket); wire
            # just the attributes the POST path reads from / writes to.
            self.rfile = io.BytesIO(body)
            self.wfile = io.BytesIO()
            self.headers = {"Content-Length": str(len(body))}
            self.responses: list[int] = []

        def send_response(self, code, *_a):  # capture status without a socket
            self.responses.append(code)

        def send_header(self, *_a):
            pass

        def end_headers(self):
            pass

    def post(action: str) -> _FakeHandler:
        h = _FakeHandler(json.dumps({"action": action}).encode("utf-8"))
        h.path = "/control"
        h.do_POST()
        return h

    return post


def test_stop_listen_action_fires_callback():
    """control 'stop_listen' calls on_stop_listen directly and 204s."""
    calls = []
    viz = VizServer(on_stop_listen=lambda: calls.append("stop_listen"))
    post = _make_request_handler(viz)
    handler = post("stop_listen")
    assert calls == ["stop_listen"]
    assert handler.responses == [204]  # no-content, never touched the bridge


def test_stop_speech_action_fires_callback():
    """control 'stop_speech' still calls on_stop_speech directly (regression)."""
    calls = []
    viz = VizServer(on_stop_speech=lambda: calls.append("stop_speech"))
    post = _make_request_handler(viz)
    handler = post("stop_speech")
    assert calls == ["stop_speech"]
    assert handler.responses == [204]


def test_stop_actions_work_without_a_bridge():
    """Stop callbacks bypass the bridge, so they 204 even with no bridge attached."""
    seen = []
    viz = VizServer(  # bridge is None
        on_stop_speech=lambda: seen.append("speech"),
        on_stop_listen=lambda: seen.append("listen"),
    )
    post = _make_request_handler(viz)
    assert post("stop_speech").responses == [204]
    assert post("stop_listen").responses == [204]
    assert seen == ["speech", "listen"]


def test_on_stop_listen_defaults_to_none():
    """The new constructor param is optional and defaults to None."""
    assert VizServer().on_stop_listen is None
