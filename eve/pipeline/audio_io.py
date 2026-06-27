"""Microphone capture + speaker playback via PyAudio, segmented with VAD.

This is the bridge between hardware and the rest of EVE. `record_utterance`
should open the mic, stream frames through the VAD, and return the PCM buffer for
one utterance (from first speech to a trailing run of silence).
"""

from __future__ import annotations

import asyncio
import threading
import time

import pyaudio

from eve.config import Config
from eve.pipeline.base import AudioIO
from eve.pipeline.vad import VoiceActivityDetector


SAMPLE_RATE = 16_000  # 16 kHz: the rate webrtcvad and Whisper both expect (see base.py)
# Discard buffered mic input for this long after opening the stream. This drops the
# tail of EVE's own speech (and any backlog) so it doesn't transcribe itself.
SETTLE_SECONDS = 0.4
# Ignore "utterances" with less than this much actual speech — usually an echo/noise
# blip rather than a real spoken request.
MIN_SPEECH_SECONDS = 0.4


class PyAudioIO(AudioIO):
    """Concrete AudioIO backed by PyAudio + webrtcvad."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.sample_rate = SAMPLE_RATE
        self.vad = VoiceActivityDetector(sample_rate=SAMPLE_RATE)
        self._pa = pyaudio.PyAudio()

    async def record_utterance(self) -> bytes:
        """Capture a single spoken utterance and return it as PCM bytes.

        Suggested approach:
        """
        def record_blocking() -> bytes:
            chunk = self.vad.frame_bytes() // 2  # samples per VAD frame
            stream = self._pa.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,  # default input device (don't hardcode an index)
                frames_per_buffer=chunk,
            )

            # Flush echo: throw away whatever was buffered before/just as we started
            # listening (the tail of EVE's own TTS) so we don't transcribe ourselves.
            settle_deadline = time.monotonic() + SETTLE_SECONDS
            while time.monotonic() < settle_deadline:
                stream.read(chunk, exception_on_overflow=False)

            silence_threshold = int(0.8 * self.sample_rate / chunk)
            min_speech_frames = int(MIN_SPEECH_SECONDS * self.sample_rate / chunk)
            speech_frames: list[bytes] = []
            speech_started = False
            silent_frames = 0
            voiced_frames = 0

            while True:
                # exception_on_overflow=False: if the input buffer overruns while
                # we were busy (transcribing/speaking the previous turn), drop the
                # late frames instead of crashing with OSError [-9981].
                frame = stream.read(chunk, exception_on_overflow=False)
                if self.vad.is_speech(frame):
                    speech_started = True
                    silent_frames = 0
                    voiced_frames += 1
                    speech_frames.append(frame)
                elif speech_started:
                    silent_frames += 1
                    speech_frames.append(frame)
                    if silent_frames >= silence_threshold:
                        break

            stream.stop_stream()
            stream.close()

            # Too little real speech → likely an echo/noise blip; report nothing so
            # the loop keeps listening instead of "hearing" a phantom utterance.
            if voiced_frames < min_speech_frames:
                return b""
            return b"".join(speech_frames)

        return await asyncio.to_thread(record_blocking)

    async def record_push_to_talk(self) -> bytes:
        """Record one utterance gated by the keyboard (push-to-talk).

        The mic is opened only between two Enter presses, so EVE can never capture
        its own speech — the robust alternative to silence-detected listening.
        """
        await asyncio.to_thread(input, "⏎  Press Enter, speak, then press Enter to send… ")
        print("🔴 Recording… press Enter to stop.")

        chunk = self.vad.frame_bytes() // 2
        frames: list[bytes] = []
        stop = threading.Event()

        def capture() -> None:
            stream = self._pa.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=chunk,
            )
            while not stop.is_set():
                frames.append(stream.read(chunk, exception_on_overflow=False))
            stream.stop_stream()
            stream.close()

        worker = threading.Thread(target=capture, daemon=True)
        worker.start()
        await asyncio.to_thread(input)  # second Enter ends the recording
        stop.set()
        await asyncio.to_thread(worker.join)
        return b"".join(frames)

    async def play(self, audio: bytes) -> None:
        """Play PCM audio through the default output device."""
        def _play_blocking() -> None:
            stream = self._pa.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                output=True,
            )
            stream.write(audio)
            stream.stop_stream()
            stream.close()

        await asyncio.to_thread(_play_blocking)
