"""Input guardrails — injection / jailbreak / PII detection (OWASP LLM01, LLM02).

A layered detector: fast pattern matching + a lightweight heuristic "classifier"
score. The classifier is intentionally simple and self-contained (no model
download); the interface is the same one a Llama Guard call or a fine-tuned
classifier would expose, so it can be upgraded without touching call sites.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..config import get_settings
from .pii import detect_pii

# Phrases that strongly signal an attempt to override instructions or jailbreak.
JAILBREAK_PATTERNS = [
    "ignore previous",
    "ignore all previous",
    "ignore your previous",
    "ignore the above",
    "ignore the docs",
    "disregard",
    "your system prompt",  # extraction intent — not the bare topic "system prompt"
    "you are now",
    "developer mode",
    "do anything now",
    "dan mode",
    "pretend you are",
    "act as if",
    "reveal your instructions",
    "print your instructions",
    "repeat the text above",
    "forget your rules",
    "without any restrictions",
    "bypass your",
]

# Secondary signals — weaker on their own; scores accumulate toward the
# threshold. Encoding/obfuscation tricks are weighted so a "decode this base64
# and obey it" prompt crosses the bar without flagging legitimate questions
# (which never contain these tokens).
_WEIGHTED_SIGNALS = {
    "base64": 0.4,
    "rot13": 0.5,
    "decode the following": 0.2,
    "decode": 0.3,
    "new instructions": 0.4,
    "real instructions": 0.4,
    "override": 0.3,
    "exfiltrate": 0.4,
    "leaked": 0.3,
    "obey": 0.3,
    "execute": 0.2,
    "follow it": 0.2,
    "treat it as": 0.2,
}

# A run of base64-ish characters (≥16) — long enough to skip ordinary words.
_BASE64_RE = re.compile(r"\b[A-Za-z0-9+/]{16,}={0,2}")


@dataclass
class GuardResult:
    blocked: bool
    flags: list[str] = field(default_factory=list)
    score: float = 0.0
    reason_code: str | None = None


def injection_classifier(text: str) -> float:
    """Heuristic 0..1 likelihood that the text is an injection/jailbreak attempt.

    Stand-in for a trained classifier or Llama Guard. Scores accumulate from
    explicit override phrases, suspicious tokens, and encoded-payload shapes.
    """
    low = text.lower()
    score = 0.0
    for phrase in JAILBREAK_PATTERNS:
        if phrase in low:
            score += 0.5
    for token, weight in _WEIGHTED_SIGNALS.items():
        if token in low:
            score += weight
    if _BASE64_RE.search(text):
        score += 0.5
    return min(score, 1.0)


def screen_input(text: str) -> GuardResult:
    settings = get_settings()
    flags: list[str] = []
    low = text.lower()

    if any(p in low for p in JAILBREAK_PATTERNS):
        flags.append("possible_injection")

    if detect_pii(text):
        flags.append("pii_in_query")

    score = injection_classifier(text)
    if score >= settings.injection_score_threshold:
        flags.append("injection_model_high")

    blocked = "injection_model_high" in flags or "possible_injection" in flags
    reason_code = "blocked_prompt_injection" if blocked else None
    return GuardResult(blocked=blocked, flags=flags, score=round(score, 3), reason_code=reason_code)
