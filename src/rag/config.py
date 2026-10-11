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
    # Max distinct clients tracked by the in-process rate limiter. Oldest
    # entries are evicted first so memory stays bounded.
    rate_limit_max_clients: int = 10_000
    # Number of reverse proxies in front of the API that append to
    # X-Forwarded-For (e.g. 1 behind Railway/Render/nginx). 0 = use the TCP
    # peer address and ignore X-Forwarded-For, which clients can spoof.
    trust_proxy_hops: int = 0
    # Comma-separated browser origins allowed to call the API. Empty = no
    # cross-origin access (the Streamlit UI calls the API server-side, so it
    # doesn't need CORS). Avoid "*" outside of throwaway demos.
    cors_allow_origins: str = ""
    # Optional shared secret. When set, POST /chat requires it (X-API-Key header
    # or "Authorization: Bearer <key>") and /health only returns details to
    # callers that present it. When unset, /chat is open (public demo mode).
    api_key: str | None = None
    # Return model/provider/config details from /health without the API key.
    # Off by default so the public endpoint doesn't fingerprint the deployment.
    health_details_public: bool = False

    @property
    def cors_origins(self) -> list[str]:
        return [o.strip() for o in self.cors_allow_origins.split(",") if o.strip()]

    @property
    def llm_provider_effective(self) -> str:
        """Fall back to the offline echo provider when no API key is present."""
        if self.llm_provider == "anthropic" and not self.anthropic_api_key:
            return "echo"
        return self.llm_provider


@lru_cache
def get_settings() -> Settings:
    return Settings()
