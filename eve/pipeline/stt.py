"""Local speech-to-text with faster-whisper (CTranslate2 backend, no torch).

faster-whisper runs the Whisper models through CTranslate2, so it needs NO torch —
which is exactly why it works on Intel macOS where PyTorch no longer ships wheels.
It's also CPU-friendly and typically faster than openai-whisper, helping the MVP
goal of transcribing "not super delayed".

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
import time
import numpy as np

from faster_whisper import WhisperModel
from eve.config import Config
from eve.pipeline.base import STTEngine


class WhisperSTT(STTEngine):
    """Concrete STT using a locally-loaded faster-whisper model."""

    def __init__(self, config: Config) -> None:
        self.model_name = config.whisper_model   # tiny | base | small | medium | large
        self.device = config.whisper_device      # auto | cpu | cuda (cpu on Intel mac)
        self._model = None                       # lazy-loaded on first use (slow to load)

    def _ensure_model(self) -> None:
        """Load the faster-whisper model once, choosing device + compute type."""
        if self._model is not None:
            return
        device = "cpu" if self.device == "auto" else self.device
        compute_type = "int8" if device == "cpu" else "float16"
        self._model = WhisperModel(self.model_name, device=device, compute_type=compute_type)

    async def _transcribe(self, audio: bytes) -> str:
        """Convert a PCM audio buffer to text.

        Part of: Mic -> STT -> sanitize -> LLM. Returns the recognized utterance.
        """
        self._ensure_model()
        samples = np.frombuffer(audio, np.int16).astype(np.float32) / 32768.0

        def transcribe() -> str:
            start = time.perf_counter()
            segments, info = self._model.transcribe(samples, language="en")
            text = " ".join(seg.text for seg in segments).strip()
            elapsed = time.perf_counter() - start
            print(f"Transcription took {elapsed:.2f}s")
            return text
    
        return await asyncio.to_thread(transcribe)
    
