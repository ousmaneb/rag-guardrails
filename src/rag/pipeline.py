"""The online query pipeline: input guardrails -> retrieve -> prompt -> LLM ->
output guardrails. Returns a structured result the API and eval harness share.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

from .config import Settings, get_settings
from .guardrails import screen_input, screen_output
from .llm import LLMProvider, get_llm
from .prompt import SYSTEM, build_prompt, citations
from .retriever import Retriever


@dataclass
class ChatResult:
    answer: str
    citations: list[dict] = field(default_factory=list)
    latency_ms: float = 0.0
    guardrail_flags: list[str] = field(default_factory=list)
    blocked: bool = False
    abstained: bool = False
    injection_score: float = 0.0
    grounding: float = 1.0
    top_retrieval_score: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


class RagPipeline:
    def __init__(
        self,
        retriever: Retriever | None = None,
        llm: LLMProvider | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.retriever = retriever or Retriever(settings=self.settings)
        self.llm = llm or get_llm(self.settings)

    def answer(self, question: str) -> ChatResult:
        start = time.perf_counter()

        # 1. Input guardrails (OWASP LLM01/LLM02).
        guard_in = screen_input(question)
        if guard_in.blocked:
            return ChatResult(
                answer="This request was blocked by the input guardrails "
                "(possible prompt injection or jailbreak attempt).",
                latency_ms=round((time.perf_counter() - start) * 1000, 1),
                guardrail_flags=guard_in.flags,
                blocked=True,
                injection_score=guard_in.score,
            )

        # 2. Retrieval.
        results = self.retriever.retrieve(question)
        top_score = self.retriever.top_score(results)

        # Abstain early when retrieval confidence is too low (LLM09).
        if not results or top_score < self.settings.min_retrieval_score:
            out = screen_output("", results)
            return ChatResult(
                answer="I don't have enough relevant information in the knowledge "
                "base to answer that.",
                latency_ms=round((time.perf_counter() - start) * 1000, 1),
                guardrail_flags=[*guard_in.flags, "low_retrieval_confidence"],
                abstained=True,
                injection_score=guard_in.score,
                grounding=out.grounding,
                top_retrieval_score=round(top_score, 4),
            )

        # 3. Prompt assembly with instruction isolation + LLM generation.
        prompt = build_prompt(question, results)
        draft = self.llm.generate(SYSTEM, prompt)

        # 4. Output guardrails (LLM02/LLM05/LLM09).
        guard_out = screen_output(draft, results)

        return ChatResult(
            answer=guard_out.answer,
            citations=citations(results),
            latency_ms=round((time.perf_counter() - start) * 1000, 1),
            guardrail_flags=[*guard_in.flags, *guard_out.flags],
            blocked=False,
            abstained=guard_out.abstained,
            injection_score=guard_in.score,
            grounding=guard_out.grounding,
            top_retrieval_score=round(top_score, 4),
        )
