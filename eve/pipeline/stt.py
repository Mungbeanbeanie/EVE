"""Local speech-to-text with faster-whisper (CTranslate2 backend, no torch).

faster-whisper runs the Whisper models through CTranslate2, so it needs no torch —
which is why it works on Intel macOS, where PyTorch no longer ships wheels. It is
also CPU-friendly and typically faster than openai-whisper, keeping transcription
latency low.

Key API (https://github.com/SYSTRAN/faster-whisper):
    from faster_whisper import WhisperModel
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, info = model.transcribe(audio)   # audio = path | numpy float32 array
    text = " ".join(seg.text for seg in segments)

Model size is the main latency lever (tiny/base are fast; medium/large are slow but
more accurate). On CPU, compute_type="int8" gives the best speed.
"""

from __future__ import annotations

import asyncio
import logging
import time
import numpy as np

from faster_whisper import WhisperModel
from eve.config import Config
from eve.pipeline.base import STTEngine

log = logging.getLogger(__name__)


class WhisperSTT(STTEngine):
    """Concrete STT using a locally-loaded faster-whisper model."""

    def __init__(self, config: Config) -> None:
        # Eagerly load the model here (not lazily in `transcribe`) so the multi-second
        # cold start happens during agent startup — before the user speaks — rather than
        # on the first utterance where it would be felt as a noticeable latency spike.
        self.model_name = config.whisper_model   # tiny | base | small | medium | large
        self.device = config.whisper_device      # auto | cpu | cuda (cpu on Intel mac)
        device = "cpu" if self.device == "auto" else self.device
        compute_type = "int8" if device == "cpu" else "float16"
        log.info("Loading Whisper model %s (%s/%s)", self.model_name, device, compute_type)
        self._model = WhisperModel(self.model_name, device=device, compute_type=compute_type)

    async def transcribe(self, audio: bytes) -> str:
        """Convert a PCM audio buffer to text and return the recognized utterance."""
        samples = np.frombuffer(audio, np.int16).astype(np.float32) / 32768.0

        def transcribe() -> str:
            start = time.perf_counter()
            segments, _info = self._model.transcribe(samples, language="en")
            text = " ".join(seg.text for seg in segments).strip()
            log.debug("Transcription took %.2fs", time.perf_counter() - start)
            return text

        return await asyncio.to_thread(transcribe)

