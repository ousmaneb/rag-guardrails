"""FastAPI app: POST /chat, GET /health.

A simple in-process rate limiter caps the public endpoint so a demo link can't
run up a bill (a small nod to OWASP LLM10, Unbounded Consumption).
"""

from __future__ import annotations

import time
from collections import deque
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .config import get_settings
from .ingest import ingest
from .pipeline import RagPipeline

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build the index on startup if it's empty.

    Makes `uvicorn rag.api:app` self-sufficient: with the in-memory store the
    index lives in this process, so it must be built here (a separate `ingest`
    command would populate a different process). With pgvector this only ingests
    when the table is empty, so it's cheap on restarts.
    """
    pipe = get_pipeline()
    try:
        if pipe.retriever.store.count() == 0:
            ingest()
    except Exception as exc:  # pragma: no cover - startup best-effort
        print(f"[startup] ingestion skipped: {exc}")
    yield


app = FastAPI(
    title="RAG Assistant with Guardrails",
    version="1.0.0",
    description="Retrieval-augmented assistant hardened against the OWASP LLM Top 10.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.cors_allow_origins.split(",")],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pipeline is built lazily on first request so the app imports cheaply (and so
# tests can monkeypatch the store before the model loads).
_pipeline: RagPipeline | None = None


def get_pipeline() -> RagPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RagPipeline(settings=settings)
    return _pipeline


# --- minimal sliding-window rate limiter (per client IP) --------------------
_hits: dict[str, deque] = {}


def _rate_limited(client: str) -> bool:
    now = time.time()
    window = _hits.setdefault(client, deque())
    while window and now - window[0] > 60:
        window.popleft()
    if len(window) >= settings.rate_limit_per_minute:
        return True
    window.append(now)
    return False


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)


class ChatResponse(BaseModel):
    answer: str
    citations: list[dict]
    latency_ms: float
    guardrail_flags: list[str]
    blocked: bool
    abstained: bool
    injection_score: float
    grounding: float
    top_retrieval_score: float


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "vector_store": settings.vector_store,
        "llm_provider": settings.llm_provider_effective,
        "model": settings.llm_model,
    }


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, request: Request) -> ChatResponse:
    client = request.client.host if request.client else "unknown"
    if _rate_limited(client):
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again shortly.")
    result = get_pipeline().answer(req.question)
    return ChatResponse(**result.to_dict())
