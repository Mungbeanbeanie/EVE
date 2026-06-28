"""EVE visualizer window.

A faithful recreation of the design handoff's glass-panel voice visualizer,
hosted by a zero-dependency stdlib server. Import :class:`VizServer` to drive the
orb from the agent, or run ``python -m eve.ui`` to preview the window standalone.
"""

from __future__ import annotations

from eve.ui.server import VizServer
from eve.ui.window import launch_window

__all__ = ["VizServer", "launch_window"]
