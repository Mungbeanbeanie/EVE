"""Shared mem0 client backed by an on-disk FAISS index.

mem0 (https://docs.mem0.ai/) is the memory engine: it embeds, stores, and
semantically searches memories. Both the procedural and episodic layers talk to
mem0 through this one configured client, distinguishing themselves via mem0's
`user_id` namespacing.

  - vector store: FAISS, persisted to `config.memory_dir` on disk.
  - embedder:     FastEmbed (ONNX), running in-process.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

# Opt out of mem0's anonymous usage telemetry (PostHog) before importing it.
# Set MEM0_TELEMETRY=true to re-enable it.
os.environ.setdefault("MEM0_TELEMETRY", "false")

from mem0 import Memory  # type: ignore  # noqa: E402  (import after the env var above)

from eve.config import Config

log = logging.getLogger(__name__)


class _DropFaissKeywordWarning(logging.Filter):
    """Drop mem0's "faiss has no keyword search" warning.

    FAISS has no keyword index, so mem0 warns that hybrid (BM25) search is
    disabled. EVE uses semantic search only, so this line is not actionable.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        return "does not support keyword search" not in record.getMessage()


logging.getLogger("mem0.memory.main").addFilter(_DropFaissKeywordWarning())

COLLECTION = "eve"


class Mem0Backend:
    """Lazily-constructed, shared mem0 Memory instance (FAISS + FastEmbed)."""

    def __init__(self, config: Config) -> None:
        self._config = config
        self._memory: Memory | None = None  # lazy: don't load the model until first use
        self._failed = False  # cached: skip retries once we know init failed

    def client(self) -> Memory:
        """Return the shared mem0 Memory client, building it on first call.

        The first ever call loads the FastEmbed model (downloaded and cached under
        ~/.cache the very first time) and opens/creates the FAISS index on disk.
        """
        if self._memory is not None:
            return self._memory
        if self._failed:
            raise RuntimeError("memory backend failed to initialize earlier; skipping")

        try:
            self._memory = self._build()
        except Exception:
            self._failed = True
            raise
        return self._memory

    def _build(self) -> Memory:
        path = Path(self._config.memory_dir).expanduser()
        path.mkdir(parents=True, exist_ok=True)

        # mem0's native LLM clients want the *bare* model name; the "provider/model"
        # form is a LiteLLM convention. Strip a leading "<provider>/" if present.
        model = self._config.llm_model.split("/", 1)[-1]
        llm_cfg: dict = {"model": model, "api_key": self._config.llm_api_key}
        if self._config.llm_api_base:
            llm_cfg["api_base"] = self._config.llm_api_base

        log.info("Loading memory: FAISS at %s, embedder %s", path, self._config.embedder_model)
        return Memory.from_config(
            {
                "vector_store": {
                    "provider": "faiss",
                    "config": {
                        "collection_name": COLLECTION,
                        "path": str(path),
                        "embedding_model_dims": self._config.embedding_dims,
                        # mem0's FAISS only L2-normalizes vectors under the
                        # "euclidean" strategy, so euclidean + normalize_L2 is how
                        # we get correct cosine-equivalent ranking even when the
                        # embedder's output isn't unit length (nomic isn't; some
                        # bge models are).
                        "distance_strategy": "euclidean",
                        "normalize_L2": True,
                    },
                },
                "embedder": {
                    "provider": self._config.embedder_provider,
                    "config": {
                        "model": self._config.embedder_model,
                        "embedding_dims": self._config.embedding_dims,
                    },
                },
                "llm": {
                    "provider": self._config.llm_provider,
                    "config": llm_cfg,
                },
            }
        )
