"""Local text-to-speech with pyttsx3.

pyttsx3 runs fully offline (no network latency) using the OS speech engine
(NSSpeechSynthesizer on macOS, SAPI5 on Windows, espeak on Linux). It is
synchronous and not thread-safe, so drive it carefully off the event loop.
"""

from __future__ import annotations

import asyncio
import logging

import pyttsx3

from eve.config import Config
from eve.pipeline.base import TTSEngine

log = logging.getLogger(__name__)


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
        """Choose the voice: explicit config first, then a preferred/English fallback."""
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

        # 1. A voice the user named in config (e.g. TTS_VOICE=Zoe).
        wanted = (self.config.tts_voice or "").strip().lower()
        chosen = next((v for v in voices if wanted and wanted in name_of(v)), None)
        if wanted and chosen is None:
            log.warning(
                "Configured TTS_VOICE=%r not found; falling back. Run "
                "`python -m eve.pipeline.tts` to list installed voices.",
                self.config.tts_voice,
            )
        # 2. A reliable preferred named voice, if installed.
        if chosen is None:
            chosen = next(
                (v for pref in self._PREFERRED_VOICES for v in voices if pref in name_of(v)),
                None,
            )
        # 3. Otherwise any English voice.
        if chosen is None:
            chosen = next((v for v in voices if is_english(v)), None)
        if chosen is not None:
            self._engine.setProperty("voice", chosen.id)
            log.info("TTS voice: %s", getattr(chosen, "name", chosen.id))

    async def speak(self, text: str) -> None:
        """Synthesize `text` and play it through the speaker."""
        self._ensure_engine()
        def _speak() -> None:
            self._engine.say(text)
            self._engine.runAndWait()
        await asyncio.to_thread(_speak)


def _list_voices() -> None:
    """Print installed TTS voices so you can pick one for TTS_VOICE in .env."""
    engine = pyttsx3.init()
    print("Installed voices (set TTS_VOICE to part of a name):\n")
    for v in engine.getProperty("voices") or []:
        langs = ", ".join(
            lang.decode() if isinstance(lang, bytes) else str(lang)
            for lang in (getattr(v, "languages", None) or [])
        )
        print(f"  {getattr(v, 'name', '?'):<24} {langs:<12} {v.id}")


if __name__ == "__main__":  # `python -m eve.pipeline.tts`
    _list_voices()