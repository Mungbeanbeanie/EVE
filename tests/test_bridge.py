"""Tests for the UI → agent InputBridge.

The bridge is a thin thread-safe queue; these check that submitted events come
back out, in order, in the shape the agent's window loop expects.
"""

from __future__ import annotations

from eve.ui.bridge import InputBridge


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
