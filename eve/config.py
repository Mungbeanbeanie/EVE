"""Typed application configuration.

All runtime settings live here so the rest of the codebase never reads os.environ
directly. Values come from environment variables / the `.env` file (see
`.env.example`). This file IS fully implemented — config loading is plumbing, not
the learning exercise.
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

    # ── Embedder (separate from LLM — Anthropic has no embedding API) ────────
    embedder_provider: str = "ollama"
    embedder_model: str = "nomic-embed-text"
    embedder_base_url: str = "http://localhost:11434"  # Ollama server URL

    # ── Database (memory backend) ────────────────────────────────────────────
    database_url: str = "postgresql://eve:eve@localhost:5432/eve"

    # ── Speech-to-text ───────────────────────────────────────────────────────
    whisper_model: str = "base"
    whisper_device: str = "auto"  # auto | cpu | cuda

    # ── Text-to-speech ───────────────────────────────────────────────────────
    # Substring of the macOS/Windows voice name to use (case-insensitive), e.g.
    # "Samantha", "Daniel", "Zoe". Leave blank to auto-pick an English voice.
    tts_voice: str | None = None

    # ── Voice input mode ─────────────────────────────────────────────────────
    # "vad" = always-listening, auto-segmented by silence detection.
    # "ptt" = push-to-talk: press Enter to start/stop each utterance (no echo).
    voice_input: str = "vad"

    # ── Tools ────────────────────────────────────────────────────────────────
    google_client_id: str | None = None
    google_client_secret: str | None = None
    google_token_path: str = "./.secrets/google_token.json"
    # Web search (Tavily). Get a free key at https://app.tavily.com. The web_search
    # tool is registered unconditionally; it only errors if called without a key.
    tavily_api_key: str | None = None

    # ── Misc ─────────────────────────────────────────────────────────────────
    log_level: str = "INFO"


@lru_cache
def load_config() -> Config:
    """Return a cached Config instance (read the environment only once)."""
    return Config()
