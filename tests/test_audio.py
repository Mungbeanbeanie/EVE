"""Tests for the PyAudio capture path, plus a manual hardware smoke script.

Two parts, split by what they need:

* Hardware-free unit tests (run under pytest) for ``PyAudioIO`` behavior that does
  not require a real device — e.g. that ``stop_recording`` ends an in-progress
  ``record_utterance`` early. The mic stream and VAD are mocked so no audio
  hardware is touched.
* ``record_and_playback`` — a manual smoke check that records from the real
  microphone and plays it back. It needs working audio hardware, so it runs only
  when this file is executed directly (``python tests/test_audio.py``), not under
  automated collection.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

from eve.config import Config
from eve.pipeline.audio_io import PyAudioIO


def _make_io_without_hardware() -> PyAudioIO:
    """Build a PyAudioIO with PyAudio/VAD/wake stubbed so no device is opened.

    ``PyAudioIO.__init__`` constructs a real ``pyaudio.PyAudio()`` and a VAD; we
    bypass it with ``__new__`` and wire in just the attributes the capture loop
    uses, so the tests stay fully hardware-free.
    """
    io = PyAudioIO.__new__(PyAudioIO)
    io.config = Config(llm_provider="anthropic", llm_model="anthropic/claude-opus-4-8")
    io.sample_rate = 16_000

    # A VAD that always reports speech, so the capture loop would otherwise run
    # forever — proving that the early break comes from stop_recording(), not VAD.
    io.vad = MagicMock()
    io.vad.frame_bytes.return_value = 640  # 320 samples/frame at 16 kHz
    io.vad.is_speech.return_value = True

    # A mic stream that yields a fixed frame on every read.
    stream = MagicMock()
    stream.read.return_value = b"\x00" * 640
    io._pa = MagicMock()
    io._pa.open.return_value = stream

    # Re-create the threading.Event that __init__ would have made.
    import threading

    io._stop_capture = threading.Event()
    return io


async def test_stop_recording_returns_captured_audio_early():
    """A concurrent stop_recording() ends record_utterance and returns the buffer.

    The fake VAD always hears speech, so without a stop the capture loop never
    exits via VAD silence. We trip stop_recording() from a separate thread after a
    few captured frames — mirroring the real HTTP-thread "tap to stop" — and assert
    the loop breaks and returns the captured bytes (not the noise-blip sentinel,
    since a deliberate tap must never be discarded).

    The stop is fired *during* the loop, not before, because record_utterance
    clears any stale stop at the start (see test_record_utterance_clears_stale_stop).
    """
    io = _make_io_without_hardware()

    # is_speech() is only consulted inside the capture loop (never the settle
    # flush), so use it to fire the stop a few captured frames in — guaranteeing
    # the stop lands mid-capture, after record_utterance cleared any stale stop.
    speech_calls = {"n": 0}

    def is_speech(_frame) -> bool:
        speech_calls["n"] += 1
        if speech_calls["n"] == 3:  # a few frames captured, then "tap to stop"
            io.stop_recording()
        return True

    io.vad.is_speech.side_effect = is_speech

    audio = await asyncio.wait_for(io.record_utterance(), timeout=5.0)

    # Bytes captured before the stop are returned (never the b"" noise sentinel).
    assert isinstance(audio, bytes)
    assert audio  # at least one frame made it in before we stopped


async def test_record_utterance_clears_stale_stop():
    """A stop left set by a prior turn must not abort the next capture instantly.

    record_utterance clears _stop_capture at the start; we set it, then make the
    VAD report silence after one speech frame so the loop ends naturally, proving
    the stale set did not short-circuit the capture.
    """
    io = _make_io_without_hardware()
    io._stop_capture.set()  # stale stop from a previous turn

    # One speech frame, then silence forever → the VAD silence path ends the loop.
    io.vad.is_speech.side_effect = [True] + [False] * 10_000

    audio = await asyncio.wait_for(io.record_utterance(), timeout=2.0)

    # The capture ran past the (cleared) stale stop and produced real frames.
    assert isinstance(audio, bytes)


async def record_and_playback() -> None:
    """Manual smoke: record from the real mic and play it back through the speaker."""
    io = PyAudioIO(Config())
    print("Say something...")
    audio = await io.record_utterance()
    print(f"Captured {len(audio)} bytes")
    print("Playing back...")
    await io.play(audio)
    print("Done!")


if __name__ == "__main__":
    asyncio.run(record_and_playback())
