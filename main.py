"""EVE entrypoint.

Boots configuration, builds the Agent with all subsystems wired together, and
runs it in either VOICE mode (mic -> STT -> LLM -> TTS -> speaker) or TEXT mode
(stdin -> LLM -> stdout). Text mode is the fastest way to develop the
LLM + memory + tools loop before touching the audio/GPU stack.

Run:
    python main.py --mode text              # no audio hardware needed
    python main.py --mode voice             # needs mic/speaker, ffmpeg, Whisper
    python main.py --mode voice --window    # + the on-screen visualizer orb
"""

from __future__ import annotations

import argparse
import asyncio

from eve.agent import Agent
from eve.config import load_config
from eve.utils.logging import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EVE — personal AI agent")
    parser.add_argument(
        "--mode",
        choices=["voice", "text"],
        default="text",
        help="voice = mic/speaker pipeline; text = terminal REPL (default)",
    )
    parser.add_argument(
        "--window",
        action="store_true",
        help="open the EVE visualizer window; the orb mirrors the live pipeline",
    )
    parser.add_argument(
        "--window-port",
        type=int,
        default=8765,
        help="port for the visualizer window server (default: 8765)",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    config = load_config()
    setup_logging(config.log_level)

    # Agent.from_config wires the pipeline, memory layers, tools, and LLM client
    # together using only the abstract interfaces — see eve/agent.py.
    agent = Agent.from_config(config)

    # Optional visualizer window: a local server hosts the glass-panel orb and
    # the agent pushes its live state (listening/thinking/speaking) to it.
    viz = None
    if args.window:
        from eve.ui import VizServer

        viz = VizServer(port=args.window_port).start(open_browser=True)
        agent.set_viz(viz)

    try:
        await agent.start(mode=args.mode)
    finally:
        if viz is not None:
            viz.stop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nEVE shutting down. Bye!")
