"""The LLM router: three interchangeable providers in front of a guarantee.

    from lead_engine.llm import LLMRouter, build_clients
    from lead_engine.copy import build_fallback_outreach

    router = LLMRouter(build_clients(groq_key=key))
    result = router.generate(prompt, fallback=build_fallback_outreach(lead, score))

Two rules hold this package together, and every design decision in it follows from one of
them:

  1. The LLM only ever writes prose over rows that already exist. Nothing here scores, and
     nothing here decides what is true. `lead_engine/scoring.py` does that in deterministic
     Python, and `lead_engine/outreach` validates that generated prose cites only numbers
     it was handed.
  2. The run completes with every provider down. `generate()` takes the deterministic
     fallback as an argument and validates it before opening a socket, so the last link of
     the chain cannot fail. "All providers unavailable" is a supported configuration.

`cache` is imported eagerly -- it is pure Python over a `%s` string and imports no driver --
and so is `router`. `providers` needs httpx, so it is imported lazily: a caller that only
wants `prompt_hash` should not need an HTTP client installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .cache import (
    STATUS_ERROR,
    STATUS_OK,
    Cache,
    CachedResponse,
    NullCache,
    ResponseCache,
    prompt_hash,
)
from .router import (
    CLOSED,
    COOLDOWN_SECONDS,
    FAILURE_THRESHOLD,
    HALF_OPEN,
    OPEN,
    TEMPLATE,
    Attempt,
    BucketSpec,
    BucketState,
    CircuitStore,
    DatabaseCircuitStore,
    Generation,
    LLMRouter,
    MemoryCircuitStore,
    utcnow,
)

if TYPE_CHECKING:  # pragma: no cover
    from .providers import (
        ChatClient,
        Completion,
        GeminiClient,
        GroqClient,
        NvidiaClient,
        build_clients,
        clients_from_settings,
    )

_PROVIDER_NAMES = {
    "ChatClient",
    "Completion",
    "GeminiClient",
    "GroqClient",
    "NvidiaClient",
    "build_clients",
    "clients_from_settings",
}

__all__ = [
    "CLOSED",
    "COOLDOWN_SECONDS",
    "FAILURE_THRESHOLD",
    "HALF_OPEN",
    "OPEN",
    "STATUS_ERROR",
    "STATUS_OK",
    "TEMPLATE",
    "Attempt",
    "BucketSpec",
    "BucketState",
    "Cache",
    "CachedResponse",
    "ChatClient",
    "CircuitStore",
    "Completion",
    "DatabaseCircuitStore",
    "GeminiClient",
    "Generation",
    "GroqClient",
    "LLMRouter",
    "MemoryCircuitStore",
    "NullCache",
    "NvidiaClient",
    "ResponseCache",
    "build_clients",
    "clients_from_settings",
    "prompt_hash",
    "utcnow",
]


def __getattr__(name: str):
    if name in _PROVIDER_NAMES:
        from . import providers

        return getattr(providers, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
