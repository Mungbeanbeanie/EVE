"""EVE entrypoint.

Boots configuration, builds the Agent with all subsystems wired together, and
runs it in either VOICE mode (mic -> STT -> LLM -> TTS -> speaker) or TEXT mode
(stdin -> LLM -> stdout). Text mode is the fastest way to develop the
LLM + memory + tools loop before touching the audio/GPU stack.

Run:
    python main.py --mode text              # no audio hardware needed
    python main.py --mode voice             # needs mic/speaker, ffmpeg, Whisper
    python main.py --mode voice --window    # + the native menu-bar window

Threading note: a native window (macOS) must own the process's MAIN thread —
Cocoa's hard rule. So in ``--window`` mode the agent's asyncio loop runs on a
worker thread and the window owns the main thread. Headless mode keeps the agent
on the main thread.
"""

from __future__ import annotations

import argparse
import asyncio
import threading

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
    parser.add_argument(
        "--dock",
        action="store_true",
        help="show a Dock icon (default: menu-bar only, no Dock icon)",
    )
    parser.add_argument(
        "--improve",
        action="store_true",
        help="enable the sleep-time self-improvement loop (same as SELF_IMPROVE=true)",
    )
    return parser.parse_args()


def _run_agent_thread(agent: Agent) -> None:
    """Run the agent's asyncio loop on a worker thread (window mode).

    Window mode always uses the UI-driven ``"window"`` loop: input arrives from
    the window's text box / mic button via the bridge, not from stdin or the
    terminal. The ``--mode`` flag only matters for headless runs.
    """
    try:
        asyncio.run(agent.start(mode="window"))
    except KeyboardInterrupt:  # pragma: no cover - worker thread rarely sees this
        pass


def run_with_window(agent: Agent, args: argparse.Namespace) -> None:
    """Window mode: native UI on the main thread, agent on a worker thread."""
    from eve.ui import VizServer, launch_window
    from eve.ui.bridge import InputBridge

    # The bridge is the browser → agent channel; the server feeds it from POSTs
    # and the agent consumes it in run_window().
    bridge = InputBridge()
    viz = VizServer(
        port=args.window_port,
        bridge=bridge,
        on_stop_speech=agent.tts.stop_speaking,
        # Second push-to-talk tap → end the in-progress mic capture directly from
        # the HTTP thread (the agent is blocked inside record_utterance and can't
        # read a bridge event until recording already ended).
        on_stop_listen=agent.audio.stop_recording,
    ).start(open_browser=False)
    agent.set_viz(viz)
    agent.set_bridge(bridge)

    # The agent runs in the background; the window owns the main thread below.
    worker = threading.Thread(target=_run_agent_thread, args=(agent,), daemon=True)
    worker.start()

    def on_quit() -> None:
        agent.stop()
        bridge.stop()  # unblock the agent's next_event() so the loop exits
        viz.stop()

    # The native window loads the orb in "embedded" mode so the page drops its
    # fake macOS chrome — the real OS window provides the frame and titlebar.
    ran_native = launch_window(
        viz.url + "?embedded=1",
        title="EVE",
        hide_dock=not args.dock,
        menu_bar=True,
        on_quit=on_quit,
    )

    # No native backend → the orb opened in a browser tab and launch_window
    # returned immediately. Keep the process alive on the agent worker instead.
    if not ran_native:
        try:
            worker.join()
        except KeyboardInterrupt:
            on_quit()


def main() -> None:
    args = parse_args()
    config = load_config()
    if args.improve:
        # CLI wins over .env for this one flag — handy for one-off improve runs.
        config = config.model_copy(update={"self_improve": True})
    setup_logging(config.log_level)

    # Agent.from_config wires the pipeline, memory layers, tools, and LLM client
    # together using only the abstract interfaces — see eve/agent.py.
    agent = Agent.from_config(config)

    if args.window:
        run_with_window(agent, args)
    else:
        # Headless: the agent owns the main thread.
        asyncio.run(agent.start(mode=args.mode))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nEVE shutting down. Bye!")
