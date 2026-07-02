"""Agent — the orchestrator.

This is the spine of EVE. It owns one instance of each subsystem (audio I/O, STT,
TTS, the LLM client, the memory manager, and the tool registry/executor) and runs
the main loop. It depends only on the abstract interfaces, so any implementation
can be swapped without touching this file.
"""

from __future__ import annotations

import logging
import re

from eve.config import Config
from eve.improve.activity import ActivityMonitor
from eve.llm.factory import build_llm
from eve.llm.sanitize import sanitize
from eve.memory.manager import MemoryManager
from eve.pipeline.audio_io import PyAudioIO
from eve.pipeline.base import TTSEngine
from eve.pipeline.stt import WhisperSTT
from eve.pipeline.tts import build_tts
from eve.tools.base import Tool
from eve.tools.executor import ToolExecutor
from eve.tools.registry import ToolRegistry
from eve.tools.adapters.google import GoogleAdapter
from eve.tools.adapters.websearch import WebSearchAdapter

log = logging.getLogger(__name__)


class Agent:
    """Wires subsystems together and runs the voice or text loop."""

    def __init__(
        self,
        *,
        config: Config,
        audio: PyAudioIO,
        stt: WhisperSTT,
        tts: TTSEngine,
        llm,  # LLMClient — provider-agnostic, built by the factory
        memory: MemoryManager,
        tools: ToolRegistry,
        executor: ToolExecutor,
        viz=None,  # optional eve.ui.VizServer — drives the visualizer window
    ) -> None:
        self.config = config
        self.audio = audio
        self.stt = stt
        self.tts = tts
        self.llm = llm
        self.memory = memory
        self.tools = tools
        self.executor = executor
        # Optional on-screen orb. Anything with a ``set_state(name)`` method works;
        # when absent every call is a no-op, so the headless loop is unchanged.
        self.viz = viz
        # Gate destructive tools (e.g. creating a calendar event) behind a yes/no
        # prompt over the active channel. The executor calls this back before running
        # any tool flagged destructive=True; returning False cancels the call.
        self.executor.confirmer = self._confirm_destructive
        self._running = False
        self._speak = False  # tracks the active channel (voice vs text) for prompts
        # Optional browser → agent input channel (the window's text box + mic
        # button). Set in window mode; drives run_window() instead of stdin/mic.
        self.bridge = None
        # Idle clock + (optional) sleep-time self-improvement daemon. The monitor
        # always exists (it's just a timestamp); the loop only runs when enabled.
        self.activity = ActivityMonitor()
        self.improver = None  # eve.improve.SelfImprovementLoop when self_improve is on
        if config.self_improve:
            self.tools.register(
                Tool(
                    name="self_improvement_status",
                    description=(
                        "Report on EVE's own self-improvement loop: what it is doing "
                        "right now, which sandbox branch it is working on, and the "
                        "most recent improvement cycles from its journal. Use this "
                        "when the user asks what you've been improving or learning, "
                        "or what happened while they were away."
                    ),
                    parameters={"type": "object", "properties": {}},
                    handler=self._improvement_status,
                )
            )
            self.tools.register(
                Tool(
                    name="request_improvement",
                    description=(
                        "Queue a user-directed improvement to EVE's own code. The "
                        "sleep-time self-improvement loop will attempt it, ahead of "
                        "its own ideas, next time the user is away. Use this whenever "
                        "the user asks you to improve, change, add, or optimize one "
                        "of your own features or behaviors (e.g. 'make your voice "
                        "faster', 'add a feature that…')."
                    ),
                    parameters={
                        "type": "object",
                        "properties": {
                            "idea": {
                                "type": "string",
                                "description": (
                                    "The improvement as 1-2 concrete sentences: what "
                                    "should change, and what success looks like."
                                ),
                            },
                        },
                        "required": ["idea"],
                    },
                    handler=self._request_improvement,
                )
            )

    # ── Visualizer ────────────────────────────────────────────────────────────
    def set_viz(self, viz) -> None:
        """Attach (or replace) the visualizer window driven by agent state."""
        self.viz = viz

    def set_bridge(self, bridge) -> None:
        """Attach the UI input channel that feeds the window-mode loop."""
        self.bridge = bridge

    def _set_state(self, name: str) -> None:
        """Drive the orb to a pipeline state; a no-op when no window is attached."""
        if self.viz is not None:
            self.viz.set_state(name)

    def _push_reply(self, role: str, text: str) -> None:
        """Show a turn caption in the window; a no-op when no window is attached.

        Mirrors :meth:`_set_state`: the window renders these one-way captions
        ("you"/"eve") so the conversation is legible on screen even though the
        reply is also spoken. Headless runs (no viz) are unaffected.
        """
        if self.viz is not None:
            self.viz.push_reply(role, text)

    # ── Construction ─────────────────────────────────────────────────────────
    @classmethod
    def from_config(cls, config: Config) -> "Agent":
        """Build a fully-wired Agent from configuration.

        Nothing here names a specific LLM vendor — the factory resolves the client
        from config, so the agent stays provider-agnostic.
        """
        memory = MemoryManager.from_config(config)
        tools = ToolRegistry()
        GoogleAdapter(config).register_into(tools)
        WebSearchAdapter(config).register_into(tools)
        executor = ToolExecutor(registry=tools)

        return cls(
            config=config,
            audio=PyAudioIO(config),
            stt=WhisperSTT(config),
            tts=build_tts(config),
            llm=build_llm(config),
            memory=memory,
            tools=tools,
            executor=executor,
        )

    # ── Entrypoints ──────────────────────────────────────────────────────────
    def stop(self) -> None:
        """Ask the run loop to exit after its current turn.

        Safe to call from another thread (e.g. the window's Quit handler). The
        loop checks ``self._running`` between turns; a turn already blocked on the
        mic/stdin finishes or is abandoned when the process exits.
        """
        self._running = False

    async def start(self, mode: str) -> None:
        """Run the agent in the requested mode until interrupted."""
        self._running = True
        log.info("EVE starting in %s mode (model=%s)", mode, self.config.llm_model)
        if self.config.self_improve:
            # Sleep-time compute: a heavier local model improves EVE's codebase
            # while the user is away. Runs on a daemon thread with its own event
            # loop so long generations never block conversation or shutdown.
            from eve.improve.loop import SelfImprovementLoop

            self.improver = SelfImprovementLoop(
                config=self.config,
                activity=self.activity,
                memory=self.memory,
                web_search=self._find_web_search(),
            )
            self.improver.start_in_thread()
            log.info(
                "Self-improvement on: %s after %.0fs idle",
                self.config.improve_model, self.config.improve_idle_seconds,
            )
        try:
            if mode == "window":
                await self.run_window()
            elif mode == "voice":
                await self.run_voice()
            else:
                await self.run_text()
        finally:
            if self.improver is not None:
                self.improver.stop()
            # remember() persists long-term memory in the background; make sure any
            # in-flight writes finish before the event loop tears down.
            await self.memory.flush()

    def _find_web_search(self):
        """Hand the improvement loop EVE's existing web_search tool, if registered."""
        try:
            return self.tools.get("web_search").handler
        except KeyError:
            return None

    async def _improvement_status(self) -> dict:
        """Tool handler: let the chat model narrate the self-improvement loop."""
        if self.improver is None:
            return {"enabled": False, "note": "the self-improvement loop is not running"}
        return self.improver.status()

    async def _request_improvement(self, idea: str) -> dict:
        """Tool handler: queue a user-directed improvement for idle time."""
        if self.improver is None:
            return {"error": "the self-improvement loop is not running"}
        return self.improver.request(idea)

    async def run_voice(self) -> None:
        """Voice loop: Mic -> STT -> sanitize -> LLM -> TTS -> Speaker.

        Loops forever (until Ctrl+C): after each reply it returns to listening. The
        "Listening…" cue makes that obvious, and blank/noise transcripts are skipped
        so a stray sound doesn't trigger an empty turn.
        """
        mode = self.config.voice_input.lower()
        if mode == "ptt":
            print("EVE voice mode — push-to-talk. Ctrl+C to quit.")
        elif mode == "wake":
            print(f"EVE voice mode — say '{self.config.wake_word}' to talk. Ctrl+C to quit.")
        while self._running:
            self._set_state("listening")  # orb reflects that EVE is capturing audio
            if mode == "ptt":
                audio = await self.audio.record_push_to_talk()
            elif mode == "wake":
                audio = await self.audio.record_with_wake_word()
            else:
                print("🎤 Listening… (speak, then pause; Ctrl+C to quit)")
                audio = await self.audio.record_utterance()  # VAD-segmented buffer
            text = (await self.stt.transcribe(audio)).strip()
            if not text:
                self._set_state("idle")
                continue  # silence / background noise → keep listening
            print(f"you > {text}")
            await self._handle_turn(text, speak=True)

    async def run_text(self) -> None:
        """Text loop: same brain as voice, but stdin/stdout instead of audio."""
        print("EVE text mode. Type a message (Ctrl+C to quit).")
        while self._running:
            try:
                text = input("you > ").strip()
            except EOFError:
                break
            if not text:
                continue
            await self._handle_turn(text, speak=False)

    async def run_window(self) -> None:
        """UI-driven loop: input comes from the window, replies are spoken.

        The window's text box and mic button POST to the visualizer server, which
        hands events to us through the :class:`~eve.ui.bridge.InputBridge`. This
        replaces stdin/terminal control entirely: a ``text`` event answers a typed
        prompt, a ``control: listen`` event captures one spoken utterance (VAD —
        no terminal Enter). Either way EVE speaks the reply so the window behaves
        like a voice assistant.
        """
        if self.bridge is None:
            raise RuntimeError("window mode requires a bridge; call set_bridge() first")
        log.info("EVE window mode — type or tap to talk.")
        self._set_state("idle")
        while self._running:
            event = await self.bridge.next_event()
            kind = event.get("type")
            if kind == "stop":
                break
            if kind == "text":
                text = str(event.get("text", "")).strip()
                if text:
                    print(f"you > {text}")
                    await self._handle_turn(text, speak=True)
            elif kind == "control" and event.get("action") == "listen":
                await self._listen_once()

    async def _listen_once(self) -> None:
        """Capture one VAD-segmented utterance from the mic and handle it."""
        self._set_state("listening")
        audio = await self.audio.record_utterance()
        text = (await self.stt.transcribe(audio)).strip()
        if not text:
            self._set_state("idle")  # silence / noise → back to rest
            return
        print(f"you > {text}")
        await self._handle_turn(text, speak=True)

    # ── One conversational turn ──────────────────────────────────────────────
    async def _handle_turn(self, text: str, *, speak: bool) -> None:
        """Process a single user utterance end-to-end.

        Wraps the pipeline so an unconfigured component (e.g. a direct provider
        client that raises NotImplementedError) surfaces a friendly hint rather
        than crashing the session.
        """
        # Remember which channel this turn uses so a mid-turn confirmation prompt
        # (triggered by the LLM calling a destructive tool) reaches the user the same
        # way they're talking to EVE.
        self._speak = speak
        # The interaction span keeps the idle clock at zero for the whole turn, so
        # the self-improvement loop can never wake up mid-conversation.
        with self.activity.interaction():
            try:
                text = sanitize(text)
                self._push_reply("you", text)  # echo the user's turn as an on-screen caption
                self.memory.working.add_user(text)

                # Pull relevant procedural + episodic memories and merge with the
                # live working window into the message list the LLM will see.
                context = await self.memory.recall(text)

                # The LLM client runs the tool-use loop internally, calling back into
                # self.executor when the model requests a tool.
                self._set_state("thinking")  # orb pulses while the model works
                reply = await self.llm.respond(
                    messages=context,
                    tools=self.tools.specs(),
                    executor=self.executor,
                )

                await self.memory.remember(user=text, assistant=reply)

                self._set_state("speaking")  # orb swells while EVE replies
                self._push_reply("eve", reply)  # caption the reply even though it's spoken
                if speak:
                    await self.tts.speak(reply)
                else:
                    print(f"eve > {reply}")

            except NotImplementedError as exc:
                log.warning("Component not available → %s", exc)
                print(f"[EVE] {exc}")
                # Surface a short, user-facing line in the window (the f-string above is
                # developer detail for the terminal; the caption stays friendly).
                self._push_reply("eve", "That feature isn't set up yet.")
            except Exception as exc:
                # One bad turn (network blip, tool error, runaway loop) must not kill the
                # session — log it, tell the user, and return to listening.
                log.exception("Turn failed")
                message = "Sorry, something went wrong handling that. Please try again."
                self._push_reply("eve", message)  # show the same friendly line on screen
                if speak:
                    await self.tts.speak(message)
                else:
                    print(f"eve > {message}  ({type(exc).__name__}: {exc})")
            finally:
                # Turn over → orb rests. The voice loop flips back to "listening" on
                # its next iteration; text mode simply waits at idle for the next line.
                self._set_state("idle")

    # ── Destructive-action confirmation ───────────────────────────────────────
    # Word-level matches (not substrings) so "now"/"know" don't read as "no".
    _AFFIRMATIVE = frozenset(
        {"yes", "yeah", "yep", "yup", "sure", "ok", "okay", "confirm",
         "affirmative", "correct", "proceed", "y"}
    )
    _NEGATIVE = frozenset(
        {"no", "nope", "nah", "cancel", "stop", "abort", "negative", "dont", "n"}
    )

    async def _confirm_destructive(self, name: str, arguments: dict) -> bool:
        """Ask the user to approve a destructive tool call before it runs.

        Wired into the ToolExecutor: any tool flagged ``destructive=True`` pauses
        here for an explicit yes/no, delivered over whichever channel (voice or text)
        the current turn is using. Anything that isn't clearly affirmative is treated
        as a decline — the safe default for an action with real-world side effects.
        """
        prompt = f"You asked me to run '{name}' with {arguments}. Shall I go ahead?"
        answer = await self._ask_voice(prompt) if self._speak else self._ask_text(prompt)
        approved = self._is_affirmative(answer)
        log.info("Destructive tool '%s' %s", name, "approved" if approved else "declined")
        return approved

    def _ask_text(self, prompt: str) -> str:
        """Prompt for confirmation over stdin and return the raw answer."""
        try:
            return input(f"eve > {prompt} [y/N] ").strip()
        except EOFError:
            return ""  # no input stream → treat as decline

    async def _ask_voice(self, prompt: str) -> str:
        """Speak the confirmation prompt, then capture and transcribe the reply."""
        await self.tts.speak(prompt)
        audio = await self.audio.record_utterance()
        return await self.stt.transcribe(audio)

    def _is_affirmative(self, answer: str) -> bool:
        """Decide if a free-form answer is a clear yes (default: no)."""
        words = set(re.findall(r"[a-z']+", answer.lower()))
        if words & self._NEGATIVE:
            return False  # any explicit "no" word wins
        if "go" in words and "ahead" in words:
            return True
        return bool(words & self._AFFIRMATIVE)
