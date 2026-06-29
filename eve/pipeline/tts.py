"""Text-to-speech engines and the factory that selects one.

EVE has two interchangeable TTS backends behind the ``TTSEngine`` interface:

* ``Pyttsx3TTS`` — fully offline (no network latency) via the OS speech engine
  (NSSpeechSynthesizer on macOS, SAPI5 on Windows, espeak on Linux). It is
  synchronous and not thread-safe, so drive it carefully off the event loop.
* ``ElevenLabsTTS`` — cloud TTS for higher-quality / custom (cloned) voices,
  streamed to the speaker as raw PCM for low time-to-first-audio.

``build_tts(config)`` picks ElevenLabs when an API key is configured and
otherwise uses the local engine — which also serves as the runtime fallback so a
missing SDK, a bad key, or a failed request degrades to offline speech instead of
a silent turn.
"""

from __future__ import annotations

import asyncio
import logging
import sys
import threading

import pyttsx3

from eve.config import Config
from eve.pipeline.base import TTSEngine

log = logging.getLogger(__name__)


class MacSayTTS(TTSEngine):
    """Local TTS on macOS via the built-in ``say`` command.

    pyttsx3's macOS driver (``NSSpeechSynthesizer``) is unreliable off the main
    thread — and EVE's window mode runs the whole agent loop on a worker thread —
    so a reused engine can hang or synthesize silence. The ``say`` binary is a
    separate process with no run-loop constraints, so it produces audio from any
    thread. We stream it via :func:`asyncio.create_subprocess_exec`, which keeps
    the event loop free while the OS speaks.

    Honors ``TTS_VOICE`` as the ``say -v`` voice name; an unknown name makes
    ``say`` fall back to the system default rather than failing the turn.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self._proc: asyncio.subprocess.Process | None = None

    async def speak(self, text: str) -> None:
        """Synthesize and play `text` by piping it to the macOS `say` binary."""
        if not text.strip():
            return
        args = ["say"]
        voice = (self.config.tts_voice or "").strip()
        if voice:
            args += ["-v", voice]
        try:
            self._proc = await asyncio.create_subprocess_exec(
                *args,
                text,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await self._proc.communicate()
            if self._proc.returncode not in (0, -15):  # -15 = SIGTERM from stop_speaking
                log.warning(
                    "macOS `say` exited %s: %s",
                    self._proc.returncode,
                    stderr.decode("utf-8", "replace").strip(),
                )
        except FileNotFoundError:  # `say` missing (non-macOS) — should not happen
            log.warning("macOS `say` not found; no speech produced for this turn.")
        finally:
            self._proc = None

    def stop_speaking(self) -> None:
        proc = self._proc
        if proc is not None and proc.returncode is None:
            proc.terminate()


class Pyttsx3TTS(TTSEngine):
    """Concrete TTS using the local pyttsx3 engine."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self._engine = None  # lazy init; pyttsx3.init() can be slow / picky

    def _ensure_engine(self):
        """Initialize the pyttsx3 engine once and tune voice/rate."""
        if self._engine is not None:
            return
        self._engine = pyttsx3.init()
        self._engine.setProperty("rate", 185)
        self._select_english_voice()

    # Reliable, broadly-installed English voices, in preference order. The platform
    # *default* voice can synthesize empty audio in headless/sandboxed contexts
    # (macOS NSSpeechSynthesizer), so we steer toward a known-good named voice first.
    _PREFERRED_VOICES = ("samantha", "alex", "daniel", "karen", "moira")

    def _select_english_voice(self) -> None:
        """Choose the voice: explicit config first, then a preferred/English fallback."""
        try:
            voices = self._engine.getProperty("voices") or []
        except Exception:  # some drivers don't expose a voice list
            return

        def name_of(voice) -> str:
            return f"{getattr(voice, 'id', '')} {getattr(voice, 'name', '')}".lower()

        def is_english(voice) -> bool:
            langs = [lang.decode() if isinstance(lang, bytes) else str(lang)
                     for lang in (getattr(voice, "languages", None) or [])]
            blob = f"{name_of(voice)} {' '.join(langs)}".lower()
            return "en_" in blob or "en-" in blob or "english" in blob

        # 1. A voice the user named in config (e.g. TTS_VOICE=Zoe).
        wanted = (self.config.tts_voice or "").strip().lower()
        chosen = next((v for v in voices if wanted and wanted in name_of(v)), None)
        if wanted and chosen is None:
            log.warning(
                "Configured TTS_VOICE=%r not found; falling back. Run "
                "`python -m eve.pipeline.tts` to list installed voices.",
                self.config.tts_voice,
            )
        # 2. A reliable preferred named voice, if installed.
        if chosen is None:
            chosen = next(
                (v for pref in self._PREFERRED_VOICES for v in voices if pref in name_of(v)),
                None,
            )
        # 3. Otherwise any English voice.
        if chosen is None:
            chosen = next((v for v in voices if is_english(v)), None)
        if chosen is not None:
            self._engine.setProperty("voice", chosen.id)
            log.info("TTS voice: %s", getattr(chosen, "name", chosen.id))

    async def speak(self, text: str) -> None:
        """Synthesize `text` and play it through the speaker."""
        self._ensure_engine()
        def _speak() -> None:
            self._engine.say(text)
            self._engine.runAndWait()
        await asyncio.to_thread(_speak)


class ElevenLabsTTS(TTSEngine):
    """Cloud TTS via ElevenLabs, streamed to the speaker as raw PCM.

    Requests ``output_format="pcm_16000"`` — signed 16-bit little-endian mono at
    16 kHz, which is exactly EVE's audio convention (see base.py) — and writes the
    streamed chunks straight to PyAudio as they arrive, so the first audio plays
    before the whole clip is synthesized (low time-to-first-audio).

    Resilient by design: any failure (SDK not installed, network, quota, bad key)
    is logged and delegated to ``fallback`` (the local pyttsx3 engine) so a turn is
    never left silent.
    """

    # Matches output_format="pcm_16000" and the rest of the pipeline.
    SAMPLE_RATE = 16_000

    def __init__(self, config: Config, fallback: TTSEngine | None = None) -> None:
        self.config = config
        self.fallback = fallback
        self._client = None  # lazy: only build the client / PyAudio on first use
        self._pa = None
        self._stop_event = threading.Event()

    def _ensure_client(self) -> None:
        """Build the ElevenLabs client and PyAudio output once, on first use."""
        if self._client is not None:
            return
        from elevenlabs.client import ElevenLabs  # optional dep — import lazily
        import pyaudio

        self._client = ElevenLabs(api_key=self.config.elevenlabs_api_key)
        self._pa = pyaudio.PyAudio()

    async def speak(self, text: str) -> None:
        """Stream `text` from ElevenLabs to the speaker; fall back on any error."""
        self._stop_event.clear()
        try:
            await asyncio.to_thread(self._stream_blocking, text)
        except Exception as exc:  # network, quota, bad key, SDK missing, …
            log.warning(
                "ElevenLabs TTS failed (%s: %s); falling back to the local voice.",
                type(exc).__name__,
                exc,
            )
            if self.fallback is not None:
                await self.fallback.speak(text)

    def stop_speaking(self) -> None:
        self._stop_event.set()
        if self.fallback is not None:
            self.fallback.stop_speaking()

    def _stream_blocking(self, text: str) -> None:
        """Synthesize and play synchronously (run off the event loop in a thread)."""
        import pyaudio

        self._ensure_client()
        audio_stream = self._client.text_to_speech.convert(
            self.config.elevenlabs_voice_id,  # voice_id is positional-first
            text=text,
            model_id=self.config.elevenlabs_model,
            output_format="pcm_16000",
        )
        stream = self._pa.open(
            format=pyaudio.paInt16, channels=1, rate=self.SAMPLE_RATE, output=True
        )
        try:
            for chunk in audio_stream:
                if self._stop_event.is_set():
                    break
                if chunk:
                    stream.write(chunk)
        finally:
            stream.stop_stream()
            stream.close()


def _build_local_tts(config: Config) -> TTSEngine:
    """Pick the most reliable offline engine for the platform.

    macOS gets :class:`MacSayTTS` (subprocess ``say``) because it works from any
    thread — critical for window mode, where the agent loop is off the main
    thread and pyttsx3's NSSpeechSynthesizer goes silent. Everywhere else uses
    pyttsx3 (SAPI5 / espeak), which is fine on its native drivers.
    """
    if sys.platform == "darwin":
        return MacSayTTS(config)
    return Pyttsx3TTS(config)


def build_tts(config: Config) -> TTSEngine:
    """Select the TTS engine from config: ElevenLabs if a key is set, else local.

    The local engine is always constructed — both as the default and as the
    runtime fallback for ElevenLabs — so EVE still talks if the optional SDK is
    missing or a cloud request fails.
    """
    local = _build_local_tts(config)
    if not (config.elevenlabs_api_key or "").strip():
        return local
    try:
        import elevenlabs  # noqa: F401 — verify the optional dependency is present
    except ImportError:
        log.warning(
            "ELEVENLABS_API_KEY is set but the `elevenlabs` package is not "
            "installed (pip install elevenlabs); using the local voice instead."
        )
        return local
    log.info(
        "TTS: ElevenLabs (voice_id=%s, model=%s)",
        config.elevenlabs_voice_id,
        config.elevenlabs_model,
    )
    return ElevenLabsTTS(config, fallback=local)


def _list_elevenlabs_voices(config: Config) -> None:
    """Print ElevenLabs voices (id + name) so you can pick ELEVENLABS_VOICE_ID."""
    from elevenlabs.client import ElevenLabs

    client = ElevenLabs(api_key=config.elevenlabs_api_key)
    print("ElevenLabs voices (set ELEVENLABS_VOICE_ID to an id):\n")
    for v in client.voices.get_all().voices:
        print(f"  {getattr(v, 'name', '?'):<24} {v.voice_id}")


def _list_voices() -> None:
    """Print installed TTS voices so you can pick one for TTS_VOICE in .env."""
    engine = pyttsx3.init()
    print("Installed voices (set TTS_VOICE to part of a name):\n")
    for v in engine.getProperty("voices") or []:
        langs = ", ".join(
            lang.decode() if isinstance(lang, bytes) else str(lang)
            for lang in (getattr(v, "languages", None) or [])
        )
        print(f"  {getattr(v, 'name', '?'):<24} {langs:<12} {v.id}")


if __name__ == "__main__":  # `python -m eve.pipeline.tts`
    # List the voices for whichever backend is configured: ElevenLabs when a key
    # is present (pick an ELEVENLABS_VOICE_ID), otherwise the local pyttsx3 voices.
    from eve.config import load_config

    cfg = load_config()
    if (cfg.elevenlabs_api_key or "").strip():
        _list_elevenlabs_voices(cfg)
    else:
        _list_voices()