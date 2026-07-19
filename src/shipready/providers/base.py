"""The GraderBackend protocol every provider implements."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


class BackendError(Exception):
    """Raised when a backend cannot produce a completion.

    Wraps provider SDK exceptions (auth, rate limit, connection) behind one
    exception type so grader.py and the CLI don't need to know about every
    vendor's exception hierarchy.
    """

    def __init__(self, message: str, *, kind: str = "unknown"):
        super().__init__(message)
        self.kind = kind  # "auth" | "rate_limit" | "connection" | "unknown"


@runtime_checkable
class GraderBackend(Protocol):
    """A model backend that turns (system, user) prompts into raw text.

    Implementations own their own SDK client construction and error mapping.
    They must not know anything about workbooks, criteria, or JSON report
    shapes -- that logic lives in grader.py and stays provider-agnostic.
    """

    model: str

    def complete(self, system: str, user: str, max_tokens: int) -> str:
        """Return the model's raw text response, or raise BackendError."""
        ...
