"""Central configuration, loaded from environment / .env.

Everything tunable in one place so eval experiments (chunk size, k, model,
thresholds) are reproducible from a single config object.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM
    llm_provider: str = "anthropic"  # "anthropic" | "echo"
    anthropic_api_key: str | None = None
    llm_model: str = "claude-opus-4-8"
    llm_effort: str = "medium"  # low | medium | high | max
    llm_max_tokens: int = 1024

    # Embeddings
    embeddings_provider: str = "sentence-transformers"  # or "hash"
    embeddings_model: str = "BAAI/bge-small-en-v1.5"

    # Vector store
    vector_store: str = "pgvector"  # "pgvector" | "memory"
    database_url: str = "postgresql://rag:rag@localhost:5432/rag"

    # Retrieval
    top_k: int = 5
    chunk_tokens: int = 600
    chunk_overlap: int = 80
    min_retrieval_score: float = 0.25

    # Guardrails
    injection_score_threshold: float = 0.8
    grounding_min_support: float = 0.35

    # API
    rate_limit_per_minute: int = 30
    cors_allow_origins: str = "*"

    @property
    def llm_provider_effective(self) -> str:
        """Fall back to the offline echo provider when no API key is present."""
        if self.llm_provider == "anthropic" and not self.anthropic_api_key:
            return "echo"
        return self.llm_provider


@lru_cache
def get_settings() -> Settings:
    return Settings()
