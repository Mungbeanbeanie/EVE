"""Shared mem0 client wired to Postgres + pgvector.

mem0 (https://docs.mem0.ai/) is the memory engine: it handles embedding, storing,
and semantic search of memories. Both the procedural and episodic layers talk to
mem0 through this single configured client, distinguishing themselves via mem0's
`user_id` / metadata namespacing.

We point mem0's vector store at the same Postgres/pgvector database EVE already
runs (see DATABASE_URL and schema.sql).
"""

from __future__ import annotations

import logging

import psycopg  # type: ignore

from eve.config import Config

from mem0 import Memory # type: ignore

log = logging.getLogger(__name__)

# Fast reachability probe so a missing DB fails in ~seconds instead of the
# pgvector pool's default 30s, and so we never spawn the pool's background
# connection threads (which otherwise stall process exit) when it's down.
_PREFLIGHT_TIMEOUT_S = 2


class Mem0Backend:
    """Lazily-constructed, shared mem0 Memory instance."""

    def __init__(self, config: Config) -> None:
        self._config = config
        self._memory = None  # lazy: don't connect to the DB until first use
        self._unavailable = False  # cached: skip retries once we know the DB is down

    def _preflight(self) -> None:
        """Fail fast if the Postgres backend isn't reachable.

        Without this, building the mem0 pgvector client succeeds even when the DB
        is down (the pool connects in background threads), and every later query
        blocks ~30s before timing out — and the lingering pool threads slow exit.
        """
        try:
            with psycopg.connect(
                self._config.database_url, connect_timeout=_PREFLIGHT_TIMEOUT_S
            ):
                pass
        except Exception as exc:
            self._unavailable = True
            raise RuntimeError(
                f"memory backend (Postgres) unreachable at "
                f"{self._config.database_url!r}: {exc}"
            ) from exc

    def client(self) -> Memory:
        """Return the shared mem0 Memory client, building it on first call."""
        if self._memory is not None:
            return self._memory
        if self._unavailable:
            raise RuntimeError("memory backend previously unreachable; skipping")

        self._preflight()

        # mem0's native provider clients (groq, anthropic, openai, ...) want the
        # *bare* model name — the "provider/model" form (e.g. "groq/llama-3.3...")
        # is a LiteLLM-only convention. Strip the leading "<provider>/" segment from
        # the model string itself (not from llm_provider, which can disagree with the
        # actual prefix and would then leave it un-stripped → mem0 404s).
        model = self._config.llm_model.split("/", 1)[-1]

        llm_cfg: dict = {
            "model": model,
            "api_key": self._config.llm_api_key,
        }
        if self._config.llm_api_base:
            llm_cfg["api_base"] = self._config.llm_api_base

        cfg = {
            "vector_store": {
                "provider": "pgvector",
                "config": {
                    "connection_string": self._config.database_url,
                    "embedding_model_dims": 768,
                },
            },
            "llm": {
                "provider": self._config.llm_provider,
                "config": llm_cfg,
            },
            "embedder": {
                "provider": self._config.embedder_provider,
                "config": {
                    "model": self._config.embedder_model,
                    "ollama_base_url": self._config.embedder_base_url,
                },
            },
        }
        self._memory = Memory.from_config(cfg)
        return self._memory