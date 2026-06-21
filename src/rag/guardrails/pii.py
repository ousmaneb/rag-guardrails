"""Lightweight PII detection and redaction (regex-based).

Covers emails, phone numbers, US SSNs, and credit-card-shaped numbers. A
production system would add Microsoft Presidio or a small NER model; the
interface here is deliberately swappable.
"""

from __future__ import annotations

import re

_PATTERNS: dict[str, re.Pattern] = {
    "email": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "phone": re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    "credit_card": re.compile(r"\b(?:\d[ -]?){13,16}\b"),
}


def detect_pii(text: str) -> list[str]:
    return [name for name, pattern in _PATTERNS.items() if pattern.search(text)]


def redact_pii(text: str) -> tuple[str, list[str]]:
    found: list[str] = []
    redacted = text
    for name, pattern in _PATTERNS.items():
        if pattern.search(redacted):
            found.append(name)
            redacted = pattern.sub(f"[REDACTED_{name.upper()}]", redacted)
    return redacted, found
