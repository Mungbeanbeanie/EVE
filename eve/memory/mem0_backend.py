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

# TODO(eve): import mem0 here once you implement the body.
#   from mem0 import Memory


class Mem0Backend:
    """Lazily-constructed, shared mem0 Memory instance."""

    def __init__(self, config: Config) -> None:
        self.database_url = config.database_url
        self._memory = None  # lazy: don't connect to the DB until first use

    def client(self):
        """Return the shared mem0 Memory client, building it on first call."""
        # TODO(eve): 1. Build a mem0 config dict pointing the vector_store at pgvector
        #               using self.database_url (host/port/db/user/password). See
        #               mem0 docs for the "pgvector" provider config shape.
        # TODO(eve): 2. Optionally configure the embedder + LLM mem0 uses internally
        #               (mem0 can reuse your provider key).
        # TODO(eve): 3. self._memory = Memory.from_config(cfg); cache + return it.
        raise NotImplementedError(
            "Wire mem0 to pgvector — see eve/memory/mem0_backend.py:client"
        )
