"""Retrieval: embed the query, search the vector store, return scored chunks."""

from __future__ import annotations

from .config import Settings, get_settings
from .embeddings import Embedder, get_embedder
from .models import RetrievedChunk
from .store import VectorStore, get_store


class Retriever:
    def __init__(
        self,
        embedder: Embedder | None = None,
        store: VectorStore | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.embedder = embedder or get_embedder(self.settings)
        self.store = store or get_store(self.settings)

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        k = top_k or self.settings.top_k
        query_vec = self.embedder.embed([query])[0]
        return self.store.search(query_vec, k)

    def top_score(self, results: list[RetrievedChunk]) -> float:
        return results[0].score if results else 0.0
