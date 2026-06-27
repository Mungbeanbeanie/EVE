-- EVE memory schema (Postgres + pgvector).
--
-- docker-compose mounts this into the db container's init dir, so it runs ONCE on
-- first startup. mem0 can also create/manage its own tables; treat this as a
-- reference + a guaranteed home for the `vector` extension. Adjust the embedding
-- dimension to match the embedder you configure in mem0 (1536 for OpenAI
-- text-embedding-3-small, 768 for many local models, etc.).

CREATE EXTENSION IF NOT EXISTS vector;

-- ── Procedural memory: durable "how I do things" ─────────────────────────────
-- TODO(eve): confirm column shape against what your mem0 pgvector config expects;
--            mem0 may manage its own table — keep or drop this accordingly.
CREATE TABLE IF NOT EXISTS procedural_memory (
    id          BIGSERIAL PRIMARY KEY,
    content     TEXT        NOT NULL,
    embedding   vector(768),            
    metadata    JSONB       NOT NULL DEFAULT '{}'::jsonb,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ── Episodic memory: durable "what happened when" ────────────────────────────
CREATE TABLE IF NOT EXISTS episodic_memory (
    id          BIGSERIAL PRIMARY KEY,
    content     TEXT        NOT NULL,
    embedding   vector(768),              
    metadata    JSONB       NOT NULL DEFAULT '{}'::jsonb,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ── Indexes for fast vector + time recall ────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_episodic_embedding   ON episodic_memory   USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_procedural_embedding ON procedural_memory USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_episodic_created_at  ON episodic_memory   (created_at DESC);
