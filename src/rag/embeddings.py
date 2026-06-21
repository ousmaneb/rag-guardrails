"""Embedding models behind a tiny interface.

Default is an open sentence-transformers model (bge-small) — no per-call cost,
shows the stack can swap providers. A deterministic hashing embedder is the
offline fallback so CI and tests never need to download a model.
"""

from __future__ import annotations

import hashlib
from typing import Protocol

import numpy as np

from .config import Settings, get_settings


class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray: ...


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class HashEmbedder:
    """Hashing trick → fixed-dim L2-normalised vectors. Deterministic, offline.

    Not as good as a real model, but enough to exercise retrieval end-to-end
    and keep CI hermetic.
    """

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for token in text.lower().split():
                h = int.from_bytes(
                    hashlib.md5(token.encode()).digest()[:4], "little"
                )
                out[i, h % self.dim] += 1.0
        return _l2_normalize(out)


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        # Method was renamed across sentence-transformers versions.
        get_dim = getattr(
            self._model, "get_embedding_dimension", None
        ) or self._model.get_sentence_embedding_dimension
        self.dim = get_dim()

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = self._model.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True
        )
        return vecs.astype(np.float32)


def get_embedder(settings: Settings | None = None) -> Embedder:
    settings = settings or get_settings()
    if settings.embeddings_provider == "hash":
        return HashEmbedder()
    try:
        return SentenceTransformerEmbedder(settings.embeddings_model)
    except Exception as exc:  # pragma: no cover - environment dependent
        # Never hard-fail the pipeline because a model couldn't be downloaded.
        print(f"[embeddings] falling back to hash embedder: {exc}")
        return HashEmbedder()
