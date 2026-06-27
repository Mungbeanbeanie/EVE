"""Shared mem0 client wired to Postgres + pgvector.

mem0 (https://docs.mem0.ai/) is the memory engine: it handles embedding, storing,
and semantic search of memories. Both the procedural and episodic layers talk to
mem0 through this single configured client, distinguishing themselves via mem0's
`user_id` / metadata namespacing.

We point mem0's vector store at the same Postgres/pgvector database EVE already
runs (see DATABASE_URL and schema.sql).
"""

from __future__ import annotations

from eve.config import Config

from mem0 import Memory # type: ignore


class Mem0Backend:
    """Lazily-constructed, shared mem0 Memory instance."""

    def __init__(self, config: Config) -> None:
        self._config = config
        self._memory = None  # lazy: don't connect to the DB until first use

def client(self) -> Memory:
    """Return the shared mem0 Memory client, building it on first call."""
    if self._memory is not None:
        return self._memory

    llm_cfg: dict = {
        "model": self._config.llm_model,
        "api_key": self._config.llm_api_key,
    }
    if self._config.llm_api_base:
        llm_cfg["api_base"] = self._config.llm_api_base

    cfg = {
        "vector_store": {
            "provider": "pgvector",
            "config": {
                "url": self._config.database_url,
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
