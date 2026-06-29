"""Manual hardware smoke check for the audio pipeline.

Records a short utterance from the real microphone and plays it back through the
speaker, so it needs working audio hardware and cannot run in automated CI (it is
excluded from pytest collection in ``pytest.ini``).

Run it directly:

    python tests/test_audio.py
"""

from __future__ import annotations

import asyncio

from eve.config import Config
from eve.pipeline.audio_io import PyAudioIO


async def record_and_playback() -> None:
    io = PyAudioIO(Config())
    print("Say something...")
    audio = await io.record_utterance()
    print(f"Captured {len(audio)} bytes")
    print("Playing back...")
    await io.play(audio)
    print("Done!")


if __name__ == "__main__":
    asyncio.run(record_and_playback())
