"""Wake-word ("hot word") detection via openWakeWord — local, no torch.

This lets EVE idle silently until she hears her wake word (e.g. "Hey Jarvis"),
then capture the spoken command — the Alexa/"Hey Google" interaction model. It
sits in front of the normal VAD capture: AudioIO streams short mic frames through
``detect()`` and only wakes up the rest of the pipeline once a frame scores above
threshold.

openWakeWord runs pretrained ONNX/tflite models on CPU (no torch), matching EVE's
"works on Intel macOS" constraint. It expects 80 ms frames (1280 samples) of
16-bit mono PCM at 16 kHz and keeps its own rolling buffer internally.

    from openwakeword.model import Model
    model = Model(wakeword_models=["hey_jarvis"], inference_framework="onnx")
    scores = model.predict(np.frombuffer(frame, np.int16))  # {name: 0..1}

``wake_word`` may be a built-in name (alexa, hey_jarvis, hey_mycroft, …) or a path
to a custom ``.onnx`` / ``.tflite`` model you trained for "Hey EVE".
"""

from __future__ import annotations

import logging
import os

import numpy as np

log = logging.getLogger(__name__)

# openWakeWord's native frame size: 80 ms @ 16 kHz. Feed it exactly this much per
# predict() call so its internal melspectrogram buffering stays aligned.
FRAME_SAMPLES = 1280


class WakeWordDetector:
    """Wraps openWakeWord with a simple boolean ``detect(frame)`` interface."""

    def __init__(
        self,
        wake_word: str = "hey_jarvis",
        threshold: float = 0.5,
        sample_rate: int = 16_000,
    ) -> None:
        """Store config; the (slow) model load is deferred to first use.

        wake_word: a built-in model name or a path to a custom .onnx/.tflite model.
        threshold: score in 0..1 above which a frame counts as the wake word.
        """
        self.wake_word = wake_word
        self.threshold = threshold
        self.sample_rate = sample_rate
        self._model = None  # lazy-loaded (downloads/loads ONNX on first use)

    @property
    def frame_samples(self) -> int:
        """Samples per frame the detector expects (read this many from the mic)."""
        return FRAME_SAMPLES

    def _ensure_model(self) -> None:
        """Load the openWakeWord model once, downloading built-ins if needed."""
        if self._model is not None:
            return
        try:
            import openwakeword
            from openwakeword.model import Model
        except ImportError as exc:  # optional dependency — only needed for wake mode
            raise RuntimeError(
                "Wake-word mode needs openWakeWord. Install it with "
                "`pip install openwakeword`, or use VOICE_INPUT=vad / ptt instead."
            ) from exc

        is_custom_path = os.path.exists(self.wake_word) or self.wake_word.endswith(
            (".onnx", ".tflite")
        )
        if is_custom_path:
            wakeword_models = [self.wake_word]
        else:
            # Built-in keyword: ensure the shared feature models + this keyword's
            # weights are present (no-op once cached under the openwakeword pkg).
            openwakeword.utils.download_models()
            wakeword_models = [self.wake_word]

        log.info("Loading wake-word model: %s (threshold=%.2f)", self.wake_word, self.threshold)
        self._model = Model(wakeword_models=wakeword_models, inference_framework="onnx")

    def detect(self, frame: bytes) -> bool:
        """Return True if `frame` (exactly frame_samples of 16-bit PCM) is the wake word."""
        self._ensure_model()
        samples = np.frombuffer(frame, np.int16)
        scores = self._model.predict(samples)
        return any(score >= self.threshold for score in scores.values())

    def reset(self) -> None:
        """Clear the model's rolling buffer (call before each fresh listen)."""
        if self._model is not None:
            self._model.reset()
