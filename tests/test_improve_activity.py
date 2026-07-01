"""ActivityMonitor — idle-clock semantics the improvement loop depends on."""

from __future__ import annotations

import asyncio

from eve.improve.activity import ActivityMonitor


def test_idle_clock_starts_at_zero_and_grows():
    monitor = ActivityMonitor()
    assert monitor.idle_seconds() >= 0
    assert not monitor.is_idle(threshold=3600)  # brand new → not "away for an hour"


def test_touch_resets_the_clock():
    monitor = ActivityMonitor()
    monitor._last_activity -= 100  # simulate 100s of silence
    assert monitor.is_idle(threshold=50)
    monitor.touch()
    assert not monitor.is_idle(threshold=50)


def test_interaction_span_pins_idle_to_zero():
    monitor = ActivityMonitor()
    monitor._last_activity -= 100
    with monitor.interaction():
        # However long ago the last activity was, an in-flight turn is never idle.
        assert monitor.idle_seconds() == 0.0
    # Exiting the span restarts the clock from "now".
    assert monitor.idle_seconds() < 1.0


def test_nested_interactions_stay_active_until_all_exit():
    monitor = ActivityMonitor()
    with monitor.interaction():
        with monitor.interaction():
            assert monitor.idle_seconds() == 0.0
        assert monitor.idle_seconds() == 0.0  # one span still open


async def test_wait_for_idle_returns_once_threshold_passes():
    monitor = ActivityMonitor()
    monitor._last_activity -= 10
    # Already idle past the threshold → must return promptly, not hang.
    await asyncio.wait_for(monitor.wait_for_idle(threshold=5, poll=0.01), timeout=1)
