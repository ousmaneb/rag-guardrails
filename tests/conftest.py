"""Hermetic test config: in-memory store, hashing embedder, echo LLM.

Set before any `rag` module reads settings (get_settings is cached).
"""

import os

os.environ.setdefault("VECTOR_STORE", "memory")
os.environ.setdefault("EMBEDDINGS_PROVIDER", "hash")
os.environ.setdefault("LLM_PROVIDER", "echo")
# The hashing embedder produces lower cosine scores than bge-small; relax the
# retrieval-confidence gate for the offline path so it doesn't abstain on
# everything.
os.environ.setdefault("MIN_RETRIEVAL_SCORE", "0.05")
os.environ.pop("ANTHROPIC_API_KEY", None)
