"""API hardening: optional API key, minimal /health, bounded rate limiter, proxy hops."""

import pytest
from fastapi.testclient import TestClient
from test_pipeline import build_pipeline

from rag import api


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(api, "_pipeline", build_pipeline())
    monkeypatch.setattr(api, "_hits", type(api._hits)())
    monkeypatch.setattr(api.settings, "api_key", None)
    monkeypatch.setattr(api.settings, "health_details_public", False)
    monkeypatch.setattr(api.settings, "trust_proxy_hops", 0)
    monkeypatch.setattr(api.settings, "rate_limit_per_minute", 30)
    return TestClient(api.app)


def test_health_is_minimal_without_key(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_health_details_with_key(client, monkeypatch):
    monkeypatch.setattr(api.settings, "api_key", "s3cret")
    assert client.get("/health", headers={"X-API-Key": "wrong"}).json() == {"status": "ok"}
    body = client.get("/health", headers={"X-API-Key": "s3cret"}).json()
    assert body["status"] == "ok"
    assert "model" in body and "indexed_chunks" in body


def test_chat_open_when_no_key_configured(client):
    r = client.post("/chat", json={"question": "What is prompt injection?"})
    assert r.status_code == 200


def test_chat_requires_key_when_configured(client, monkeypatch):
    monkeypatch.setattr(api.settings, "api_key", "s3cret")
    q = {"question": "What is prompt injection?"}
    assert client.post("/chat", json=q).status_code == 401
    assert client.post("/chat", json=q, headers={"X-API-Key": "nope"}).status_code == 401
    assert client.post("/chat", json=q, headers={"X-API-Key": "s3cret"}).status_code == 200
    r = client.post("/chat", json=q, headers={"Authorization": "Bearer s3cret"})
    assert r.status_code == 200


def test_no_cors_by_default(client):
    r = client.get("/health", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in r.headers


def test_rate_limiter_is_bounded(monkeypatch):
    monkeypatch.setattr(api, "_hits", type(api._hits)())
    monkeypatch.setattr(api.settings, "rate_limit_max_clients", 5)
    for i in range(50):
        api._rate_limited(f"10.0.0.{i}", now=1000.0)
    assert len(api._hits) == 5
    # Entries that have aged out of the window are pruned.
    api._rate_limited("10.9.9.9", now=2000.0)
    assert list(api._hits) == ["10.9.9.9"]


def test_rate_limit_enforced(monkeypatch):
    monkeypatch.setattr(api, "_hits", type(api._hits)())
    monkeypatch.setattr(api.settings, "rate_limit_per_minute", 2)
    assert not api._rate_limited("1.1.1.1", now=0.0)
    assert not api._rate_limited("1.1.1.1", now=1.0)
    assert api._rate_limited("1.1.1.1", now=2.0)
    assert not api._rate_limited("1.1.1.1", now=62.0)


def test_spoofed_forwarded_for_ignored_by_default(client, monkeypatch):
    monkeypatch.setattr(api.settings, "rate_limit_per_minute", 1)
    q = {"question": "What is prompt injection?"}
    assert client.post("/chat", json=q, headers={"X-Forwarded-For": "1.1.1.1"}).status_code == 200
    # A different spoofed XFF must not reset the limit.
    assert client.post("/chat", json=q, headers={"X-Forwarded-For": "2.2.2.2"}).status_code == 429


def test_trusted_proxy_hops_uses_rightmost_entry(client, monkeypatch):
    monkeypatch.setattr(api.settings, "trust_proxy_hops", 1)
    monkeypatch.setattr(api.settings, "rate_limit_per_minute", 1)
    q = {"question": "What is prompt injection?"}
    h1 = {"X-Forwarded-For": "6.6.6.6, 203.0.113.7"}
    h2 = {"X-Forwarded-For": "7.7.7.7, 203.0.113.7"}  # same real client, spoofed prefix
    h3 = {"X-Forwarded-For": "203.0.113.8"}  # different real client
    assert client.post("/chat", json=q, headers=h1).status_code == 200
    assert client.post("/chat", json=q, headers=h2).status_code == 429
    assert client.post("/chat", json=q, headers=h3).status_code == 200
