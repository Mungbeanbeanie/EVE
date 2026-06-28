"""Tests for TTS engine selection and the ElevenLabs → local fallback.

These never hit the network or the audio hardware: they check that
``build_tts`` picks the right engine from config and that ``ElevenLabsTTS``
delegates to its fallback when a synthesis attempt fails.
"""

from __future__ import annotations

import sys

from eve.pipeline.base import TTSEngine
from eve.pipeline.tts import ElevenLabsTTS, MacSayTTS, Pyttsx3TTS, build_tts

# The local engine is platform-aware: macOS uses the thread-safe `say` binary,
# everything else uses pyttsx3. Tests assert against whichever applies here.
_LOCAL_ENGINE = MacSayTTS if sys.platform == "darwin" else Pyttsx3TTS


class _RecordingTTS(TTSEngine):
    """A fake TTS that records what it was asked to say (no audio)."""

    def __init__(self) -> None:
        self.spoken: list[str] = []

    async def speak(self, text: str) -> None:
        self.spoken.append(text)


def test_build_tts_uses_local_when_no_key(config):
    """No ELEVENLABS_API_KEY → the offline local engine for this platform."""
    cfg = config.model_copy(update={"elevenlabs_api_key": None})
    assert isinstance(build_tts(cfg), _LOCAL_ENGINE)


def test_build_tts_ignores_blank_key(config):
    """A whitespace-only key is treated as unset."""
    cfg = config.model_copy(update={"elevenlabs_api_key": "   "})
    assert isinstance(build_tts(cfg), _LOCAL_ENGINE)


def test_build_tts_uses_elevenlabs_when_key_set(config):
    """A key present → ElevenLabs, with the local engine wired in as fallback."""
    cfg = config.model_copy(update={"elevenlabs_api_key": "sk-test"})
    tts = build_tts(cfg)
    assert isinstance(tts, ElevenLabsTTS)
    assert isinstance(tts.fallback, _LOCAL_ENGINE)


async def test_elevenlabs_falls_back_on_error(config):
    """If synthesis raises, EVE speaks via the fallback instead of going silent."""
    cfg = config.model_copy(update={"elevenlabs_api_key": "sk-test"})
    fallback = _RecordingTTS()
    tts = ElevenLabsTTS(cfg, fallback=fallback)

    def _boom(_text: str) -> None:
        raise RuntimeError("network down")

    tts._stream_blocking = _boom  # type: ignore[method-assign]

    await tts.speak("hello there")
    assert fallback.spoken == ["hello there"]


async def test_elevenlabs_error_without_fallback_is_swallowed(config):
    """A failure with no fallback must not raise out of the turn."""
    cfg = config.model_copy(update={"elevenlabs_api_key": "sk-test"})
    tts = ElevenLabsTTS(cfg, fallback=None)

    def _boom(_text: str) -> None:
        raise RuntimeError("network down")

    tts._stream_blocking = _boom  # type: ignore[method-assign]

    await tts.speak("hello there")  # should simply return
