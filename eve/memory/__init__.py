"""Memory subsystem — three complementary layers behind one MemoryManager.

    WorkingMemory     — volatile, in-process rolling window of the live conversation
                        (what's in "RAM" right now). No database.
    ProceduralMemory  — durable "how I do things": learned preferences, skills,
                        standing instructions. Vector-indexed for recall.
    EpisodicMemory    — durable "what happened when": timestamped events/interactions.
                        Vector + time recall.

Procedural + episodic are persisted in Postgres/pgvector via mem0. The manager is
the single façade the Agent uses: `recall(query)` to read, `remember(...)` to write.
"""
