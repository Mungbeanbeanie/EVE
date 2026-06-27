"""Local text-to-speech with pyttsx3.

pyttsx3 runs fully offline (no network latency) using the OS speech engine
(NSSpeechSynthesizer on macOS, SAPI5 on Windows, espeak on Linux). It is
synchronous and not thread-safe, so drive it carefully off the event loop.
"""

from __future__ import annotations

from eve.config import Config
from eve.pipeline.base import TTSEngine

import asyncio

import pyttsx3


class Pyttsx3TTS(TTSEngine):
    """Concrete TTS using the local pyttsx3 engine."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self._engine = None  # lazy init; pyttsx3.init() can be slow / picky

    def _ensure_engine(self):
        """Initialize the pyttsx3 engine once and tune voice/rate."""
        if self._engine is not None:
            return 
        self._engine = pyttsx3.init()
        self._engine.setProperty("rate", 185)

    async def speak(self, text: str) -> None:
        """Synthesize `text` and play it through the speaker."""
        self._ensure_engine()
        def _speak() -> None:
            self._engine.say(text)
            self._engine.runAndWait()
        await asyncio.to_thread(_speak)