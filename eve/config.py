"""Typed application configuration.

All runtime settings live here so the rest of the codebase never reads os.environ
directly. Values come from environment variables / the `.env` file (see
`.env.example`).
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict # type: ignore


class Config(BaseSettings):
    """Strongly-typed settings, populated from the environment / `.env`."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # ignore unrelated env vars instead of erroring
    )

    # ── LLM (provider-agnostic) ──────────────────────────────────────────────
    llm_provider: str = "anthropic"
    llm_model: str = "anthropic/claude-opus-4-8"  # LiteLLM model string
    llm_api_key: str | None = None
    llm_api_base: str | None = None  # for self-hosted / Ollama

    # ── Embedder (vectorizes memories; separate from the LLM) ────────────────
    # FastEmbed (ONNX) runs in-process; the model is downloaded and cached on first
    # run. `embedding_dims` must match the model's output size. List models + dims:
    #   python -c "from fastembed import TextEmbedding as T; [print(m['model'], m['dim']) for m in T.list_supported_models()]"
    embedder_provider: str = "fastembed"
    embedder_model: str = "nomic-ai/nomic-embed-text-v1.5"
    embedding_dims: int = 768  # output size of embedder_model; sizes the FAISS index

    # ── Memory store (on-disk FAISS index) ───────────────────────────────────
    # Where long-term memory persists (the FAISS index + metadata). "~" is
    # expanded and the directory is created on first use; delete its contents to
    # wipe EVE's long-term memory.
    memory_dir: str = "~/.eve/memory"

    # ── Speech-to-text ───────────────────────────────────────────────────────
    whisper_model: str = "base"
    whisper_device: str = "auto"  # auto | cpu | cuda

    # ── Text-to-speech ───────────────────────────────────────────────────────
    # Substring of the local (pyttsx3) voice name to use (case-insensitive), e.g.
    # "Samantha", "Daniel", "Zoe". Leave blank to auto-pick an English voice.
    tts_voice: str | None = None

    # ElevenLabs cloud TTS (optional). When `elevenlabs_api_key` is set, EVE uses
    # ElevenLabs for higher-quality / custom (cloned) voices and falls back to the
    # local pyttsx3 voice automatically if the key is absent or a request fails.
    elevenlabs_api_key: str | None = None
    # Voice id from your ElevenLabs library (Voices → ⋯ → "Copy voice ID"). The
    # default is the stock "Rachel" voice; set this to your custom/cloned voice id.
    elevenlabs_voice_id: str = "21m00Tcm4TlvDq8ikWAM"
    # Low-latency models keep voice replies snappy: eleven_flash_v2_5 (fastest) or
    # eleven_turbo_v2_5. Use eleven_multilingual_v2 only if you need top quality.
    elevenlabs_model: str = "eleven_flash_v2_5"

    # ── Voice input mode ─────────────────────────────────────────────────────
    # "vad"  = always-listening, auto-segmented by silence detection.
    # "ptt"  = push-to-talk: press Enter to start/stop each utterance (no echo).
    # "wake" = idle until the wake word is heard, then capture the command.
    voice_input: str = "vad"

    # ── Wake word (only used when voice_input="wake") ────────────────────────
    # A built-in openWakeWord name (alexa | hey_jarvis | hey_mycroft | hey_rhasspy)
    # or a path to a custom .onnx/.tflite model trained for "Hey EVE".
    wake_word: str = "hey_jarvis"
    wake_threshold: float = 0.5  # detection score (0..1); raise to reduce false wakes

    # ── Tools ────────────────────────────────────────────────────────────────
    google_client_id: str | None = None
    google_client_secret: str | None = None
    google_token_path: str = "./.secrets/google_token.json"
    # Web search (Tavily). Get a free key at https://app.tavily.com. The web_search
    # tool is registered unconditionally; it only errors if called without a key.
    tavily_api_key: str | None = None

    # ── Self-improvement loop (sleep-time compute) ───────────────────────────
    # When enabled, EVE uses idle time (no conversation for improve_idle_seconds)
    # to run a heavier local model that researches, implements, and test-gates
    # small improvements to EVE's own codebase — always inside a sandbox worktree
    # on a `self-improve/*` branch (never main), never touching memory_dir.
    # Cycles are journaled under improve_home/journal. See eve/improve/.
    self_improve: bool = False
    improve_model: str = "ollama_chat/ornith:35b"  # heavy model for idle work
    improve_idle_seconds: float = 180.0  # user must be away this long to start
    improve_max_files: int = 10          # per-cycle changed-file budget
    improve_max_cycles: int = 0          # per-session cycle cap (0 = unlimited)
    improve_home: str = "~/.eve/improve" # journal, state, sandbox worktrees
    improve_reflect_hours: float = 6.0   # min gap between memory reflections (0 = off)

    # ── Misc ─────────────────────────────────────────────────────────────────
    log_level: str = "INFO"


@lru_cache
def load_config() -> Config:
    """Return a cached Config instance (read the environment only once)."""
    return Config()
