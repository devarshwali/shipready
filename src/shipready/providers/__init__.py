"""Pluggable grading model backends.

shipready's grading logic (prompt building, JSON parsing, verdict rules) is
provider-agnostic. A GraderBackend's only job is: given a system prompt and a
user prompt, return the model's raw text response. Everything about which
vendor, which model, which endpoint stays isolated in this package.
"""

from __future__ import annotations

from typing import Callable, Dict

from .base import GraderBackend
from .anthropic_provider import AnthropicBackend
from .openai_provider import OpenAIBackend

DEFAULT_PROVIDER = "anthropic"

_BACKENDS: Dict[str, Callable[..., GraderBackend]] = {
    "anthropic": AnthropicBackend,
    "openai": OpenAIBackend,
    # Any OpenAI-compatible local server (Ollama, vLLM, LM Studio, llama.cpp
    # server, etc.) is just the OpenAI backend pointed at a different
    # base_url, so it reuses the same class rather than needing its own.
    "local": OpenAIBackend,
}


def build_backend(
    provider: str,
    model: str,
    *,
    base_url: str | None = None,
    api_key: str | None = None,
) -> GraderBackend:
    """Construct a GraderBackend for the named provider.

    Raises ValueError for an unknown provider name, listing the known ones,
    so a typo in --provider fails fast with a helpful message.
    """
    try:
        cls = _BACKENDS[provider]
    except KeyError:
        known = ", ".join(sorted(_BACKENDS))
        raise ValueError(f"unknown provider {provider!r}; known providers: {known}")
    return cls(model=model, base_url=base_url, api_key=api_key)


__all__ = [
    "GraderBackend",
    "AnthropicBackend",
    "OpenAIBackend",
    "DEFAULT_PROVIDER",
    "build_backend",
]
