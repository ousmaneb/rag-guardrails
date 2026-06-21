"""Provider-agnostic LLM interface.

The rest of the app depends only on ``LLMProvider.generate``. Swap Claude for a
local Ollama model (or anything else) by adding a provider and a config flag —
no call sites change. Default provider is Anthropic Claude.
"""

from __future__ import annotations

from typing import Protocol

from .config import Settings, get_settings


class LLMProvider(Protocol):
    def generate(self, system: str, prompt: str) -> str: ...


class EchoProvider:
    """Deterministic, dependency-free provider for CI / offline dev.

    It does not call any API. It returns a grounded-looking answer assembled
    from the context so the pipeline (and tests) can run end-to-end without a
    key. Real evals require a real provider.
    """

    def generate(self, system: str, prompt: str) -> str:
        marker = "<context>"
        if marker in prompt and "</context>" in prompt:
            ctx = prompt.split(marker, 1)[1].split("</context>", 1)[0].strip()
            first = next((ln.strip() for ln in ctx.splitlines() if ln.strip()), "")
            if first:
                return f"Based on the retrieved context: {first} [1]"
        return "I don't know based on the provided context."


# Model families that support adaptive thinking + the effort parameter. Haiku
# and older Sonnet tiers reject both with a 400, so we only send them when the
# configured model is known to support them.
_ADAPTIVE_THINKING_MODELS = (
    "opus-4-6", "opus-4-7", "opus-4-8", "sonnet-4-6", "fable-5", "mythos-5",
)


class AnthropicProvider:
    """Claude via the official Anthropic SDK.

    Uses adaptive thinking + the effort parameter on models that support them
    (Opus 4.6+/Sonnet 4.6+/Fable); falls back to a plain request otherwise.
    Streams to stay under HTTP timeouts on larger ``max_tokens``.
    """

    def __init__(self, settings: Settings) -> None:
        import anthropic

        self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self._model = settings.llm_model
        self._effort = settings.llm_effort
        self._max_tokens = settings.llm_max_tokens

    def _supports_adaptive(self) -> bool:
        return any(tag in self._model for tag in _ADAPTIVE_THINKING_MODELS)

    def generate(self, system: str, prompt: str) -> str:
        kwargs: dict = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": prompt}],
        }
        if self._supports_adaptive():
            kwargs["thinking"] = {"type": "adaptive"}
            kwargs["output_config"] = {"effort": self._effort}

        with self._client.messages.stream(**kwargs) as stream:
            message = stream.get_final_message()

        return "".join(b.text for b in message.content if b.type == "text").strip()


def get_llm(settings: Settings | None = None) -> LLMProvider:
    settings = settings or get_settings()
    provider = settings.llm_provider_effective
    if provider == "anthropic":
        return AnthropicProvider(settings)
    if provider == "echo":
        return EchoProvider()
    raise ValueError(f"Unknown LLM provider: {provider}")
