"""Microphone capture + speaker playback via PyAudio, segmented with VAD.

This is the bridge between hardware and the rest of EVE. `record_utterance`
should open the mic, stream frames through the VAD, and return the PCM buffer for
one utterance (from first speech to a trailing run of silence).
"""

from __future__ import annotations

from eve.config import Config
from eve.pipeline.base import AudioIO
from eve.pipeline.vad import VoiceActivityDetector

# TODO(eve): import pyaudio here once you implement the body.
#   import pyaudio

SAMPLE_RATE = 16_000  # keep consistent with VAD + Whisper


class PyAudioIO(AudioIO):
    """Concrete AudioIO backed by PyAudio + webrtcvad."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.sample_rate = SAMPLE_RATE
        self.vad = VoiceActivityDetector(sample_rate=SAMPLE_RATE)
        # TODO(eve): self._pa = pyaudio.PyAudio()  (open lazily / close on shutdown)

    async def record_utterance(self) -> bytes:
        """Capture a single spoken utterance and return it as PCM bytes.

        Suggested approach:
        """
        # TODO(eve): 1. Open an input stream: format=paInt16, channels=1,
        #               rate=self.sample_rate, frames_per_buffer=vad.frame_bytes()//2.
        # TODO(eve): 2. Read frames in a loop; use self.vad.is_speech(frame) to detect
        #               the start of speech, then keep buffering.
        # TODO(eve): 3. Stop after N consecutive silent frames (e.g. ~800 ms of silence).
        # TODO(eve): 4. Run blocking PyAudio reads off the event loop
        #               (asyncio.to_thread) so you don't block other coroutines.
        # TODO(eve): 5. Return the concatenated speech frames as bytes.
        raise NotImplementedError(
            "Implement mic capture — see eve/pipeline/audio_io.py:record_utterance"
        )

    async def play(self, audio: bytes) -> None:
        """Play PCM audio through the default output device."""
        # TODO(eve): 1. Open an output stream (paInt16, mono, self.sample_rate).
        # TODO(eve): 2. Write `audio` to it (again off-thread via asyncio.to_thread).
        # TODO(eve): 3. Close/cleanup the stream.
        raise NotImplementedError(
            "Implement speaker playback — see eve/pipeline/audio_io.py:play"
        )
