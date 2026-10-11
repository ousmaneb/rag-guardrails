"""FastAPI app: POST /chat, GET /health.

A simple in-process rate limiter caps the public endpoint so a demo link can't
run up a bill (a small nod to OWASP LLM10, Unbounded Consumption). Setting
API_KEY locks /chat to callers that present the key.
"""

from __future__ import annotations

import hmac
import threading
import time
from collections import OrderedDict, deque
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
if settings.cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "Authorization", "X-API-Key"],
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
# Bounded: idle clients are pruned and, past `rate_limit_max_clients`, the
# least-recently-seen client is evicted, so many distinct IPs can't grow memory
# without limit.
_WINDOW_SECONDS = 60.0
_hits: OrderedDict[str, deque] = OrderedDict()
_hits_lock = threading.Lock()


def _rate_limited(client: str, now: float | None = None) -> bool:
    now = time.time() if now is None else now
    with _hits_lock:
        # Drop clients whose newest hit is outside the window (oldest first).
        while _hits:
            oldest_key, oldest = next(iter(_hits.items()))
            if oldest and now - oldest[-1] <= _WINDOW_SECONDS:
                break
            del _hits[oldest_key]

        window = _hits.get(client)
        if window is None:
            window = deque()
            _hits[client] = window
            while len(_hits) > max(1, settings.rate_limit_max_clients):
                _hits.popitem(last=False)
        else:
            _hits.move_to_end(client)
        while window and now - window[0] > _WINDOW_SECONDS:
            window.popleft()
        if len(window) >= settings.rate_limit_per_minute:
            return True
        window.append(now)
        return False


def client_ip(request: Request) -> str:
    """Client address for rate limiting.

    With TRUST_PROXY_HOPS=N, take the Nth entry from the right of
    X-Forwarded-For (the address our own outermost trusted proxy saw). Entries
    further left are client-controlled and are never used.
    """
    peer = request.client.host if request.client else "unknown"
    hops = settings.trust_proxy_hops
    if hops <= 0:
        return peer
    header = ",".join(request.headers.getlist("x-forwarded-for"))
    chain = [part.strip() for part in header.split(",") if part.strip()]
    if len(chain) < hops:
        return peer
    return chain[-hops]


def _presented_key(request: Request) -> str | None:
    key = request.headers.get("x-api-key")
    if key:
        return key
    auth = request.headers.get("authorization", "")
    scheme, _, token = auth.partition(" ")
    if scheme.lower() == "bearer" and token:
        return token.strip()
    return None


def _has_valid_key(request: Request) -> bool:
    if not settings.api_key:
        return False
    presented = _presented_key(request)
    if presented is None:
        return False
    return hmac.compare_digest(presented.encode(), settings.api_key.encode())


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
def health(request: Request) -> dict:
    # Public liveness probe stays minimal; config details only for callers with
    # the API key (or when HEALTH_DETAILS_PUBLIC=true).
    if not (_has_valid_key(request) or settings.health_details_public):
        return {"status": "ok"}
    pipe = get_pipeline()
    try:
        doc_count = pipe.retriever.store.count()
    except Exception:
        doc_count = None
    return {
        "status": "ok",
        "vector_store": settings.vector_store,
        "llm_provider": settings.llm_provider_effective,
        "model": settings.llm_model,
        "embeddings": settings.embeddings_model
        if settings.embeddings_provider != "hash"
        else "hash",
        "top_k": settings.top_k,
        "chunk_tokens": settings.chunk_tokens,
        "rate_limit_per_minute": settings.rate_limit_per_minute,
        "indexed_chunks": doc_count,
    }


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, request: Request) -> ChatResponse:
    if _rate_limited(client_ip(request)):
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again shortly.")
    if settings.api_key and not _has_valid_key(request):
        raise HTTPException(
            status_code=401,
            detail="Missing or invalid API key.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    result = get_pipeline().answer(req.question)
    return ChatResponse(**result.to_dict())
