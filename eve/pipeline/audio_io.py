"""Microphone capture + speaker playback via PyAudio, segmented with VAD.

This is the bridge between hardware and the rest of EVE. `record_utterance`
should open the mic, stream frames through the VAD, and return the PCM buffer for
one utterance (from first speech to a trailing run of silence).
"""

from __future__ import annotations

import asyncio

import pyaudio

from eve.config import Config
from eve.pipeline.base import AudioIO
from eve.pipeline.vad import VoiceActivityDetector


SAMPLE_RATE = 44_100  


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
            stream = self._pa.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=44100,
                input=True,
                input_device_index = 2,
                frames_per_buffer=self.vad.frame_bytes() // 2,
            )

            silence_threshold = int(0.8 *self.sample_rate / (self.vad.frame_bytes() //2))
            speech_frames = []
            speech_started = False
            silent_frames = 0

            while True:
                frame = stream.read(self.vad.frame_bytes() // 2)
                if self.vad.is_speech(frame):
                    speech_started = True
                    silent_frames = 0
                    speech_frames.append(frame)
                elif speech_started:
                    silent_frames += 1
                    speech_frames.append(frame)
                    if silent_frames >= silence_threshold:
                        break
            
            stream.stop_stream()
            stream.close()

            return b"".join(speech_frames)
        
        return await asyncio.to_thread(record_blocking)

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
