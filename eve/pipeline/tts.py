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
        self._select_english_voice()

    # Reliable, broadly-installed English voices, in preference order. The platform
    # *default* voice can synthesize empty audio in headless/sandboxed contexts
    # (macOS NSSpeechSynthesizer), so we steer toward a known-good named voice first.
    _PREFERRED_VOICES = ("samantha", "alex", "daniel", "karen", "moira")

    def _select_english_voice(self) -> None:
        """Pick a concrete English voice, falling back gracefully to the default."""
        try:
            voices = self._engine.getProperty("voices") or []
        except Exception:  # some drivers don't expose a voice list
            return

        def name_of(voice) -> str:
            return f"{getattr(voice, 'id', '')} {getattr(voice, 'name', '')}".lower()

        def is_english(voice) -> bool:
            langs = [lang.decode() if isinstance(lang, bytes) else str(lang)
                     for lang in (getattr(voice, "languages", None) or [])]
            blob = f"{name_of(voice)} {' '.join(langs)}".lower()
            return "en_" in blob or "en-" in blob or "english" in blob

        # 1. A preferred named voice, if installed.
        chosen = next(
            (v for pref in self._PREFERRED_VOICES for v in voices if pref in name_of(v)),
            None,
        )
        # 2. Otherwise any English voice.
        if chosen is None:
            chosen = next((v for v in voices if is_english(v)), None)
        if chosen is not None:
            self._engine.setProperty("voice", chosen.id)

    async def speak(self, text: str) -> None:
        """Synthesize `text` and play it through the speaker."""
        self._ensure_engine()
        def _speak() -> None:
            self._engine.say(text)
            self._engine.runAndWait()
        await asyncio.to_thread(_speak)