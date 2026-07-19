"""Anthropic (Claude) grading backend."""

from __future__ import annotations

import os
from typing import Optional

from .base import BackendError

DEFAULT_MODEL = "claude-opus-4-8"


class AnthropicBackend:
    """Wraps the Anthropic SDK behind the GraderBackend protocol."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.model = model
        self._base_url = base_url
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._client = None  # lazily constructed so import stays cheap

    def _get_client(self):
        if self._client is None:
            from anthropic import Anthropic

            kwargs = {}
            if self._api_key:
                kwargs["api_key"] = self._api_key
            if self._base_url:
                kwargs["base_url"] = self._base_url
            self._client = Anthropic(**kwargs)
        return self._client

    def complete(self, system: str, user: str, max_tokens: int) -> str:
        from anthropic import (
            APIConnectionError,
            AuthenticationError,
            RateLimitError,
        )

        client = self._get_client()
        try:
            response = client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
        except AuthenticationError as exc:
            raise BackendError(
                "ANTHROPIC_API_KEY is invalid (authentication failed).",
                kind="auth",
            ) from exc
        except RateLimitError as exc:
            raise BackendError(
                "Anthropic API rate limit reached. Wait a moment and retry.",
                kind="rate_limit",
            ) from exc
        except APIConnectionError as exc:
            raise BackendError(
                "Could not reach the Anthropic API. Check your network connection.",
                kind="connection",
            ) from exc

        text = "".join(
            block.text
            for block in response.content
            if getattr(block, "type", None) == "text"
        )
        return text
