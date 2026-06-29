"""Voice-activity detection (silence detection) via webrtcvad.

Used by AudioIO to decide when an utterance starts and ends, so we only send
*speech* to Whisper (faster, cleaner transcripts). webrtcvad works on short
frames (10/20/30 ms) of 16-bit mono PCM at 8/16/32/48 kHz.
"""

from __future__ import annotations

import webrtcvad


class VoiceActivityDetector:
    """Wraps webrtcvad with simple start/stop utterance segmentation."""

    # Common frame durations webrtcvad accepts (ms). 30ms is a good default.
    FRAME_MS = 30

    def __init__(self, sample_rate: int = 16_000, aggressiveness: int = 3) -> None:
        """Create a VAD.

        aggressiveness: 0 (most permissive) .. 3 (most aggressive at filtering
        non-speech).
        """
        self.sample_rate = sample_rate
        self.aggressiveness = aggressiveness
        self._vad = webrtcvad.Vad(aggressiveness)

    def is_speech(self, frame: bytes) -> bool:
        """Return True if a single PCM frame contains speech.

        `frame` must be exactly FRAME_MS worth of samples (see frame_bytes()).
        """
        expected = self.frame_bytes()
        if len(frame) != expected:
            raise ValueError(f"Frame must be {expected} bytes, got {len(frame)}")
        return self._vad.is_speech(frame, self.sample_rate)

    def frame_bytes(self) -> int:
        """Number of PCM bytes in one VAD frame (16-bit mono => 2 bytes/sample)."""
        return int(self.sample_rate * (self.FRAME_MS / 1000.0)) * 2
