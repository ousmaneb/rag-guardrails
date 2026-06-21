"""Offline ingestion pipeline: load -> chunk -> embed -> upsert.

Run as a module:  python -m rag.ingest
or via the console:  python -m rag.ingest --corpus data/corpus
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .chunking import chunk_text
from .config import get_settings
from .embeddings import get_embedder
from .models import Chunk
from .store import get_store


def load_documents(corpus_dir: Path) -> list[tuple[str, str, str]]:
    """Return (source, title, text) for every .md / .txt file in the corpus."""
    docs: list[tuple[str, str, str]] = []
    for path in sorted(corpus_dir.rglob("*")):
        if path.suffix.lower() not in {".md", ".txt"}:
            continue
        text = path.read_text(encoding="utf-8")
        title = text.lstrip().splitlines()[0].lstrip("# ").strip() if text.strip() else path.stem
        docs.append((path.name, title or path.stem, text))
    return docs


def build_chunks(docs: list[tuple[str, str, str]], chunk_tokens: int, overlap: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    for source, title, text in docs:
        for i, piece in enumerate(chunk_text(text, chunk_tokens, overlap)):
            chunks.append(
                Chunk(
                    chunk_id=f"{source}::{i}",
                    text=piece,
                    source=source,
                    title=title,
                    metadata={"position": i},
                )
            )
    return chunks


def ingest(corpus_dir: str | Path = "data/corpus") -> int:
    settings = get_settings()
    embedder = get_embedder(settings)
    store = get_store(settings)

    docs = load_documents(Path(corpus_dir))
    chunks = build_chunks(docs, settings.chunk_tokens, settings.chunk_overlap)
    if not chunks:
        raise SystemExit(f"No documents found in {corpus_dir}")

    vectors = embedder.embed([c.text for c in chunks])

    store.reset(embedder.dim)
    store.upsert(chunks, vectors)
    print(
        f"Ingested {len(chunks)} chunks from {len(docs)} documents "
        f"into {settings.vector_store} (dim={embedder.dim})."
    )
    return len(chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest a corpus into the vector store.")
    parser.add_argument("--corpus", default="data/corpus", help="Path to the corpus directory")
    args = parser.parse_args()
    ingest(args.corpus)


if __name__ == "__main__":
    main()
