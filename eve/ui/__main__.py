"""Preview the EVE window standalone: ``python -m eve.ui``.

Serves the glass panel with no agent attached, so every control drives the orb
locally with the synthetic envelope (exactly like the design prototype). Use this
to iterate on the visuals without booting the full pipeline.
"""

from __future__ import annotations

import argparse
import time

from eve.ui.server import VizServer


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview the EVE visualizer window")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--accent", default="amber", choices=["amber", "cyan", "violet", "mono"])
    parser.add_argument("--no-browser", action="store_true", help="don't auto-open a browser")
    args = parser.parse_args()

    server = VizServer(host=args.host, port=args.port, accent=args.accent).start(
        open_browser=not args.no_browser
    )
    print(f"EVE window → {server.url}  (Ctrl+C to quit)")
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        print("\nClosing EVE window.")
        server.stop()


if __name__ == "__main__":
    main()
