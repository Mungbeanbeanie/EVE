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

from eve.config import Config
from eve.pipeline.base import STTEngine

# TODO(eve): import faster-whisper here once you implement the body.
#   from faster_whisper import WhisperModel


class WhisperSTT(STTEngine):
    """Concrete STT using a locally-loaded faster-whisper model."""

    def __init__(self, config: Config) -> None:
        self.model_name = config.whisper_model   # tiny | base | small | medium | large
        self.device = config.whisper_device      # auto | cpu | cuda (cpu on Intel mac)
        self._model = None                       # lazy-loaded on first use (slow to load)

    def _ensure_model(self) -> None:
        """Load the faster-whisper model once, choosing device + compute type."""
        # TODO(eve): 1. Resolve device: on this Intel Mac use "cpu". If self.device
        #               is "auto", default to "cpu" (no CUDA on Intel macOS).
        # TODO(eve): 2. Pick compute_type: "int8" for CPU (fast), "float16" for CUDA.
        # TODO(eve): 3. self._model = WhisperModel(self.model_name, device=..., compute_type=...)
        #               The first run downloads the model weights and caches them.
        raise NotImplementedError(
            "Load faster-whisper model — see eve/pipeline/stt.py:_ensure_model"
        )

    async def transcribe(self, audio: bytes) -> str:
        """Convert a PCM audio buffer to text.

        Part of: Mic -> STT -> sanitize -> LLM. Returns the recognized utterance.
        """
        # TODO(eve): 1. self._ensure_model().
        # TODO(eve): 2. Convert PCM int16 bytes -> float32 numpy array normalized to
        #               [-1, 1] at 16 kHz mono (faster-whisper accepts a numpy array):
        #                 import numpy as np
        #                 samples = np.frombuffer(audio, np.int16).astype(np.float32) / 32768.0
        # TODO(eve): 3. Run off-thread (asyncio.to_thread) to avoid blocking the loop:
        #                 segments, info = self._model.transcribe(samples, language="en")
        # TODO(eve): 4. Join segment texts, strip, and return; log elapsed time for
        #               the latency goal. (segments is a generator — iterating it is
        #               what actually does the work.)
        raise NotImplementedError(
            "Implement faster-whisper transcription — see eve/pipeline/stt.py:transcribe"
        )
