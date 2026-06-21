"""Output guardrails — PII redaction, grounding check, policy filter.

Maps to OWASP LLM02 (sensitive info), LLM05 (improper output handling), and
LLM09 (misinformation). The grounding check is a lightweight lexical-overlap
NLI stand-in: it measures how much of the answer's content words are supported
by the retrieved context, and abstains (cite-or-refuse) when support is low.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..config import get_settings
from ..models import RetrievedChunk
from .pii import redact_pii

REFUSAL = "I don't have enough grounded information in the provided sources to answer that."

# Detects the model echoing OUR system prompt back (OWASP LLM07). Matches
# distinctive verbatim fragments of the prompt — NOT the generic phrase
# "system prompt", which legitimately appears when explaining prompt injection.
_PROMPT_LEAK_RE = re.compile(
    r"(you answer only from the <context>"
    r"|text inside <context>"
    r"|never reveal (this|the)( system)? prompt"
    r"|reference data, never instructions"
    r"|cite sources inline as)",
    re.IGNORECASE,
)

_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "be", "to",
    "of", "in", "on", "for", "with", "as", "by", "that", "this", "it", "from",
    "at", "your", "you", "i", "we", "they", "can", "will", "not", "no",
    "based", "context", "answer", "provided", "above",
}


def _content_words(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]{3,}", text.lower())
    return {w for w in words if w not in _STOPWORDS}


def grounding_support(answer: str, results: list[RetrievedChunk]) -> float:
    """Fraction of the answer's content words that appear in retrieved context."""
    answer_words = _content_words(answer)
    if not answer_words:
        return 1.0
    context_words: set[str] = set()
    for r in results:
        context_words |= _content_words(r.chunk.text)
    supported = answer_words & context_words
    return len(supported) / len(answer_words)


@dataclass
class OutputGuardResult:
    answer: str
    flags: list[str] = field(default_factory=list)
    grounding: float = 1.0
    abstained: bool = False


def screen_output(answer: str, results: list[RetrievedChunk]) -> OutputGuardResult:
    settings = get_settings()
    flags: list[str] = []

    # 1. Policy filter — never echo the system prompt back (LLM07).
    if _PROMPT_LEAK_RE.search(answer):
        flags.append("system_prompt_leak")
        return OutputGuardResult(answer=REFUSAL, flags=flags, grounding=0.0, abstained=True)

    # 2. PII redaction (LLM02).
    redacted, pii_found = redact_pii(answer)
    if pii_found:
        flags.append("pii_redacted")
    answer = redacted

    # 3. Grounding / cite-or-refuse gate (LLM05, LLM09).
    grounding = grounding_support(answer, results)
    abstained = False
    looks_like_idk = "don't know" in answer.lower() or "do not know" in answer.lower()
    if grounding < settings.grounding_min_support and not looks_like_idk:
        flags.append("low_grounding_abstained")
        answer = REFUSAL
        abstained = True

    return OutputGuardResult(
        answer=answer, flags=flags, grounding=round(grounding, 3), abstained=abstained
    )
