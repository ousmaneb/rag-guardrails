"""Document chunking.

Splits on paragraph boundaries and packs paragraphs into ~chunk_tokens-sized
windows with a small overlap, so a chunk rarely cuts a sentence in half. Tokens
are approximated as words (~0.75 word/token); good enough for chunk sizing and
keeps the pipeline dependency-light.
"""

from __future__ import annotations

import re


def _approx_tokens(text: str) -> int:
    return max(1, int(len(text.split()) / 0.75))


def chunk_text(text: str, chunk_tokens: int, overlap: int) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = _approx_tokens(para)
        if current and current_tokens + para_tokens > chunk_tokens:
            chunks.append("\n\n".join(current))
            # Carry the tail paragraphs forward to create overlap.
            carry: list[str] = []
            carry_tokens = 0
            for prev in reversed(current):
                pt = _approx_tokens(prev)
                if carry_tokens + pt > overlap:
                    break
                carry.insert(0, prev)
                carry_tokens += pt
            current = carry
            current_tokens = carry_tokens

        current.append(para)
        current_tokens += para_tokens

    if current:
        chunks.append("\n\n".join(current))
    return chunks
