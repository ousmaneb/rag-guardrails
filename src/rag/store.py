"""Vector store abstraction.

``pgvector`` (Postgres) is the production store — one container does DB +
vectors. ``memory`` is a numpy cosine store for local dev, tests and CI, where
standing up Postgres is overkill. Both expose the same interface so nothing
downstream cares which one is active.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from .config import Settings, get_settings
from .models import Chunk, RetrievedChunk


class VectorStore(Protocol):
    def reset(self, dim: int) -> None: ...
    def upsert(self, chunks: list[Chunk], vectors: np.ndarray) -> None: ...
    def search(self, query_vector: np.ndarray, top_k: int) -> list[RetrievedChunk]: ...
    def count(self) -> int: ...


class MemoryStore:
    """In-process cosine-similarity store (vectors are L2-normalised)."""

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._matrix: np.ndarray | None = None

    def reset(self, dim: int) -> None:
        self._chunks = []
        self._matrix = None

    def upsert(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        self._chunks.extend(chunks)
        self._matrix = (
            vectors if self._matrix is None else np.vstack([self._matrix, vectors])
        )

    def search(self, query_vector: np.ndarray, top_k: int) -> list[RetrievedChunk]:
        if self._matrix is None or not self._chunks:
            return []
        scores = self._matrix @ query_vector
        idx = np.argsort(-scores)[:top_k]
        return [RetrievedChunk(self._chunks[i], float(scores[i])) for i in idx]

    def count(self) -> int:
        return len(self._chunks)


class PgVectorStore:
    """Postgres + pgvector. Cosine distance (`<=>`); similarity = 1 - distance."""

    def __init__(self, database_url: str) -> None:
        self._url = database_url

    def _connect(self):
        import psycopg
        from pgvector.psycopg import register_vector

        conn = psycopg.connect(self._url, autocommit=True)
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(conn)
        return conn

    def reset(self, dim: int) -> None:
        with self._connect() as conn:
            conn.execute("DROP TABLE IF EXISTS chunks")
            conn.execute(
                f"""
                CREATE TABLE chunks (
                    chunk_id  TEXT PRIMARY KEY,
                    text      TEXT NOT NULL,
                    source    TEXT NOT NULL,
                    title     TEXT NOT NULL,
                    embedding vector({dim})
                )
                """
            )
            conn.execute(
                "CREATE INDEX ON chunks USING hnsw (embedding vector_cosine_ops)"
            )

    def upsert(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        with self._connect() as conn, conn.cursor() as cur:
            for chunk, vec in zip(chunks, vectors, strict=True):
                cur.execute(
                    """
                    INSERT INTO chunks (chunk_id, text, source, title, embedding)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (chunk_id) DO UPDATE
                      SET text = EXCLUDED.text, embedding = EXCLUDED.embedding
                    """,
                    (chunk.chunk_id, chunk.text, chunk.source, chunk.title, vec),
                )

    def search(self, query_vector: np.ndarray, top_k: int) -> list[RetrievedChunk]:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT chunk_id, text, source, title,
                       1 - (embedding <=> %s) AS score
                FROM chunks
                ORDER BY embedding <=> %s
                LIMIT %s
                """,
                (query_vector, query_vector, top_k),
            )
            rows = cur.fetchall()
        return [
            RetrievedChunk(
                Chunk(chunk_id=r[0], text=r[1], source=r[2], title=r[3]),
                float(r[4]),
            )
            for r in rows
        ]

    def count(self) -> int:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM chunks")
            return int(cur.fetchone()[0])


# The in-memory store lives in process memory, so ingestion and serving must
# share the same instance when they run in one process (eval, tests, notebooks).
# pgvector is backed by the DB, so a fresh handle each call is fine.
_MEMORY_SINGLETON: MemoryStore | None = None


def get_store(settings: Settings | None = None) -> VectorStore:
    global _MEMORY_SINGLETON
    settings = settings or get_settings()
    if settings.vector_store == "memory":
        if _MEMORY_SINGLETON is None:
            _MEMORY_SINGLETON = MemoryStore()
        return _MEMORY_SINGLETON
    if settings.vector_store == "pgvector":
        return PgVectorStore(settings.database_url)
    raise ValueError(f"Unknown vector store: {settings.vector_store}")
