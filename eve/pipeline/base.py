"""Abstract interfaces for the voice pipeline.

The Agent talks to these ABCs only. Swap in any implementation (cloud STT, a
different TTS, etc.) by subclassing — no change to the orchestrator.

Audio convention used across EVE: raw 16-bit PCM, mono, 16 kHz, as `bytes`.
Whisper and webrtcvad both like 16 kHz; keep the whole pipeline consistent.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class AudioIO(ABC):
    """Captures microphone audio and plays audio to the speaker."""

    @abstractmethod
    async def record_utterance(self) -> bytes:
        """Block until a complete spoken utterance is captured, then return it.

        Implementations should use VAD to detect when the user starts and stops
        speaking, and return the PCM buffer for that single utterance.
        """

    @abstractmethod
    async def play(self, audio: bytes) -> None:
        """Play a PCM audio buffer through the speaker."""


class STTEngine(ABC):
    """Converts an audio buffer into text."""

    @abstractmethod
    async def transcribe(self, audio: bytes) -> str:
        """Return the recognized text for a PCM audio buffer."""


class TTSEngine(ABC):
    """Converts text into speech and plays it (or returns audio)."""

    @abstractmethod
    async def speak(self, text: str) -> None:
        """Synthesize `text` and play it through the speaker."""

    def stop_speaking(self) -> None:
        """Interrupt speech in progress. No-op if nothing is playing."""
