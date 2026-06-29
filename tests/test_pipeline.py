"""Hardware-free tests for the voice pipeline.

These exercise the voice path without touching a real mic or speaker: the VAD must
return a boolean decision, and the STT must satisfy the STTEngine interface (be
instantiable and transcribe real PCM).
"""

from __future__ import annotations

import wave

import pytest

from eve.config import Config
from eve.pipeline.base import STTEngine
from eve.pipeline.stt import WhisperSTT
from eve.pipeline.vad import VoiceActivityDetector


# ── VAD ──────────────────────────────────────────────────────────────────────
def test_is_speech_returns_bool_for_silence():
    """Silence must return a real boolean; None would stall the record loop."""
    vad = VoiceActivityDetector(sample_rate=16_000)
    decision = vad.is_speech(b"\x00" * vad.frame_bytes())
    assert isinstance(decision, bool)
    assert decision is False  # pure silence is not speech


def test_is_speech_rejects_wrong_frame_size():
    vad = VoiceActivityDetector(sample_rate=16_000)
    with pytest.raises(ValueError):
        vad.is_speech(b"\x00" * 10)


def test_frame_bytes_matches_16khz_30ms():
    vad = VoiceActivityDetector(sample_rate=16_000)
    assert vad.frame_bytes() == int(16_000 * 0.030) * 2  # 960 bytes


# ── STT ──────────────────────────────────────────────────────────────────────
def test_whisper_stt_satisfies_interface():
    """WhisperSTT must implement the abstract `transcribe` method."""
    stt = WhisperSTT(Config())
    assert isinstance(stt, STTEngine)
    assert hasattr(stt, "transcribe")


@pytest.mark.slow
async def test_whisper_transcribes_real_speech(tmp_path):
    """End-to-end STT on synthesized speech. Skipped if macOS `say`/`afconvert` absent.

    Downloads the faster-whisper 'base' model on first run, so it's marked slow.
    """
    import shutil
    import subprocess

    if not (shutil.which("say") and shutil.which("afconvert")):
        pytest.skip("requires macOS `say` + `afconvert` to synthesize input audio")

    aiff = tmp_path / "in.aiff"
    wav = tmp_path / "in.wav"
    subprocess.run(["say", "-v", "Samantha", "-o", str(aiff), "hello world"], check=True)
    subprocess.run(
        ["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", str(aiff), str(wav)],
        check=True,
    )
    with wave.open(str(wav)) as w:
        pcm = w.readframes(w.getnframes())

    text = (await WhisperSTT(Config()).transcribe(pcm)).lower()
    assert "hello" in text
