"""OpenAI-compatible grading backend.

Also serves any locally hosted, OpenAI-compatible chat completions endpoint
(Ollama's /v1, vLLM, LM Studio, llama.cpp server, etc.) -- point --base-url
at it and set --api-key-env to a var that holds any placeholder value the
local server expects (many accept a dummy key, or none at all).
"""

from __future__ import annotations

import os
from typing import Optional

from .base import BackendError

DEFAULT_MODEL = "gpt-4o"


class OpenAIBackend:
    """Wraps the OpenAI SDK (or an OpenAI-compatible server) behind
    GraderBackend."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.model = model
        self._base_url = base_url
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY") or "not-needed"
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI

            kwargs = {"api_key": self._api_key}
            if self._base_url:
                kwargs["base_url"] = self._base_url
            self._client = OpenAI(**kwargs)
        return self._client

    def complete(self, system: str, user: str, max_tokens: int) -> str:
        from openai import (
            APIConnectionError,
            AuthenticationError,
            RateLimitError,
        )

        client = self._get_client()
        try:
            response = client.chat.completions.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
        except AuthenticationError as exc:
            raise BackendError(
                "OpenAI API key is invalid (authentication failed).",
                kind="auth",
            ) from exc
        except RateLimitError as exc:
            raise BackendError(
                "OpenAI API rate limit reached. Wait a moment and retry.",
                kind="rate_limit",
            ) from exc
        except APIConnectionError as exc:
            raise BackendError(
                "Could not reach the OpenAI-compatible endpoint. Check the "
                "--base-url and your network connection.",
                kind="connection",
            ) from exc

        choice = response.choices[0]
        return choice.message.content or ""
