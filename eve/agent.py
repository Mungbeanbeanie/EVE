"""Agent — the orchestrator.

This is the spine of EVE. It owns one instance of each subsystem (audio I/O, STT,
TTS, the LLM client, the memory manager, and the tool registry/executor) and runs
the main loop. It depends ONLY on the abstract interfaces, so any implementation
can be swapped without touching this file.

This file is written "for real" (the wiring) on purpose: the loop calls into the
subsystem methods, which are the stubs you will implement. When a stub raises
NotImplementedError, the loop catches it and tells you exactly which file/TODO to
go fill in — so you can build the agent one piece at a time and always run it.
"""

from __future__ import annotations

import logging

from eve.config import Config
from eve.llm.factory import build_llm
from eve.llm.sanitize import sanitize
from eve.memory.manager import MemoryManager
from eve.pipeline.audio_io import PyAudioIO
from eve.pipeline.stt import WhisperSTT
from eve.pipeline.tts import Pyttsx3TTS
from eve.tools.executor import ToolExecutor
from eve.tools.registry import ToolRegistry
from eve.tools.adapters.google import GoogleAdapter
from eve.tools.adapters.microsoft import MicrosoftAdapter

log = logging.getLogger(__name__)


class Agent:
    """Wires subsystems together and runs the voice or text loop."""

    def __init__(
        self,
        *,
        config: Config,
        audio: PyAudioIO,
        stt: WhisperSTT,
        tts: Pyttsx3TTS,
        llm,  # LLMClient — provider-agnostic, built by the factory
        memory: MemoryManager,
        tools: ToolRegistry,
        executor: ToolExecutor,
    ) -> None:
        self.config = config
        self.audio = audio
        self.stt = stt
        self.tts = tts
        self.llm = llm
        self.memory = memory
        self.tools = tools
        self.executor = executor
        self._running = False

    # ── Construction ─────────────────────────────────────────────────────────
    @classmethod
    def from_config(cls, config: Config) -> "Agent":
        """Build a fully-wired Agent from configuration.

        Construction is real plumbing; the constructed objects' *behavior* is what
        you implement. Note how nothing here mentions a specific LLM vendor — the
        factory resolves that from config.
        """
        memory = MemoryManager.from_config(config)
        tools = ToolRegistry()
        GoogleAdapter(config).register_into(tools)
        MicrosoftAdapter(config).register_into(tools)
        executor = ToolExecutor(registry=tools)

        return cls(
            config=config,
            audio=PyAudioIO(config),
            stt=WhisperSTT(config),
            tts=Pyttsx3TTS(config),
            llm=build_llm(config),
            memory=memory,
            tools=tools,
            executor=executor,
        )

    # ── Entrypoints ──────────────────────────────────────────────────────────
    async def start(self, mode: str) -> None:
        """Run the agent in the requested mode until interrupted."""
        self._running = True
        log.info("EVE starting in %s mode (model=%s)", mode, self.config.llm_model)
        if mode == "voice":
            await self.run_voice()
        else:
            await self.run_text()

    async def run_voice(self) -> None:
        """Voice loop: Mic -> STT -> sanitize -> LLM -> TTS -> Speaker."""
        while self._running:
            audio = await self.audio.record_utterance()  # VAD-segmented buffer
            text = await self.stt.transcribe(audio)
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

    # ── One conversational turn ──────────────────────────────────────────────
    async def _handle_turn(self, text: str, *, speak: bool) -> None:
        """Process a single user utterance end-to-end.

        Wraps the pipeline so an unimplemented stub becomes a friendly hint rather
        than a crash — letting you build EVE incrementally.
        """
        try:
            text = sanitize(text)
            self.memory.working.add_user(text)

            # Pull relevant procedural + episodic memories and merge with the
            # live working window into the message list the LLM will see.
            context = await self.memory.recall(text)

            # The LLM client runs the tool-use loop internally, calling back into
            # self.executor when the model requests a tool.
            reply = await self.llm.respond(
                messages=context,
                tools=self.tools.specs(),
                executor=self.executor,
            )

            await self.memory.remember(user=text, assistant=reply)

            if speak:
                await self.tts.speak(reply)
            else:
                print(f"eve > {reply}")

        except NotImplementedError as exc:
            log.warning("Not implemented yet → %s", exc)
            print(f"[EVE stub] {exc}")
