"""Local text-to-speech with pyttsx3.

pyttsx3 runs fully offline (no network latency) using the OS speech engine
(NSSpeechSynthesizer on macOS, SAPI5 on Windows, espeak on Linux). It is
synchronous and not thread-safe, so drive it carefully off the event loop.
"""

from __future__ import annotations

from eve.config import Config
from eve.pipeline.base import TTSEngine

# TODO(eve): import pyttsx3 here once you implement the body.
#   import pyttsx3


class Pyttsx3TTS(TTSEngine):
    """Concrete TTS using the local pyttsx3 engine."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self._engine = None  # lazy init; pyttsx3.init() can be slow / picky

    def _ensure_engine(self):
        """Initialize the pyttsx3 engine once and tune voice/rate."""
        # TODO(eve): 1. self._engine = pyttsx3.init()
        # TODO(eve): 2. Optionally set rate/volume/voice:
        #               self._engine.setProperty("rate", 185)
        # TODO(eve): 3. Return/cache the engine.
        raise NotImplementedError(
            "Implement pyttsx3 init — see eve/pipeline/tts.py:_ensure_engine"
        )

    async def speak(self, text: str) -> None:
        """Synthesize `text` and play it through the speaker."""
        # TODO(eve): 1. self._ensure_engine().
        # TODO(eve): 2. engine.say(text); engine.runAndWait()  — these BLOCK, so run
        #               them via asyncio.to_thread to keep the loop responsive.
        # TODO(eve): 3. (Optional) for lower perceived latency, stream sentence by
        #               sentence instead of waiting for the full reply.
        raise NotImplementedError(
            "Implement TTS playback — see eve/pipeline/tts.py:speak"
        )
